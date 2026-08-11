from __future__ import annotations

import csv
import ctypes
import hashlib
import heapq
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .domain import (
    Backend,
    DatasetProfile,
    DeviceSpec,
    HardwareSpec,
    MeasurementKind,
    MoETopology,
    ModelSpec,
    Provenance,
    ProvenanceKind,
    QuantizationLayout,
    QuantizationOverride,
    gibibytes,
)


@dataclass(frozen=True)
class RuntimeCapability:
    """A runtime fact measured in the interpreter that will execute the work."""

    installed: bool
    available: bool
    version: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ApplePlatformProfile:
    """Apple host facts that do not fit Aptus's legacy CUDA-shaped schema."""

    system: str
    architecture: str
    os_version: str
    os_build: str | None
    chip_name: str | None
    logical_cpu_count: int | None
    metal_gpu_core_count: int | None
    unified_memory_bytes: int
    available_memory_bytes: int | None
    memory_free_percent: int | None
    swap_total_bytes: int | None
    swap_used_bytes: int | None
    swap_free_bytes: int | None
    metal_recommended_working_set_bytes: int | None
    mlx: RuntimeCapability
    mlx_lm: RuntimeCapability
    pytorch_mps: RuntimeCapability
    observed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_examples(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Dataset row {line_number} must be an object.")
                yield value
        return
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict) and isinstance(value.get("train"), list):
            rows = value["train"]
        elif isinstance(value, dict):
            rows = [value]
        else:
            raise ValueError(
                "JSON dataset must contain an object or a list of objects."
            )
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"Dataset row {index} must be an object.")
            yield row
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as source:
            yield from csv.DictReader(source)
        return
    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                yield {"text": line.rstrip("\n")}
        return
    raise ValueError(f"Unsupported dataset format: {suffix or '<none>'}")


def _extract_text(example: dict[str, Any]) -> tuple[str, str]:
    text = example.get("text")
    if isinstance(text, str):
        if text.strip():
            return text, "text"
        raise ValueError("empty example")

    prompt, completion = example.get("prompt"), example.get("completion")
    if isinstance(prompt, str) or isinstance(completion, str):
        if (
            isinstance(prompt, str)
            and isinstance(completion, str)
            and completion.strip()
        ):
            return f"{prompt}\n{completion}", "prompt-completion"
        raise ValueError("empty example")

    instruction, output = example.get("instruction"), example.get("output")
    if isinstance(instruction, str) or isinstance(output, str):
        if isinstance(instruction, str) and isinstance(output, str) and output.strip():
            parts = [instruction]
            if isinstance(example.get("input"), str) and example["input"].strip():
                parts.append(example["input"])
            parts.append(output)
            return "\n".join(parts), "instruction-output"
        raise ValueError("empty example")

    messages = example.get("messages")
    if isinstance(messages, list):
        if not messages:
            raise ValueError("empty example")
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Every message must be an object.")
            role, content = message.get("role"), message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(
                    "Every message requires string role and content fields."
                )
            if content.strip():
                parts.append(f"<{role}>\n{content}")
        final = messages[-1]
        if final.get("role") != "assistant" or not final.get("content", "").strip():
            raise ValueError(
                "A messages example must end with a non-empty assistant message."
            )
        return "\n".join(parts), "messages"

    content = example.get("content")
    if isinstance(content, str):
        if content.strip():
            return content, "text"
        raise ValueError("empty example")
    raise ValueError(
        "Dataset example has no supported text, messages, prompt/completion, or instruction/output schema."
    )


def _estimated_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _token_count(text: str, tokenizer: Any | None) -> int:
    if tokenizer is None:
        return _estimated_tokens(text)
    if callable(tokenizer) and not hasattr(tokenizer, "encode"):
        value = tokenizer(text)
    else:
        value = tokenizer.encode(text, add_special_tokens=True)
    if isinstance(value, dict):
        value = value.get("input_ids", [])
    return max(1, len(value))


def _nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("Cannot compute a percentile from no values.")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def profile_dataset(
    source_path: Path,
    *,
    sample_limit: int | None = None,
    tokenizer: Any | None = None,
    sequence_length: int | None = None,
) -> DatasetProfile:
    source_path = source_path.resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"Dataset source is not a file: {source_path}")
    if sample_limit is not None and sample_limit <= 0:
        raise ValueError("sample_limit must be positive.")

    source_digest = _sha256(source_path)
    rng = random.Random(int(source_digest[:16], 16))
    limit = sample_limit
    reservoir: list[tuple[int, int]] = []
    schema_counts: Counter[str] = Counter()
    content_hashes: set[str] = set()
    duplicate_count = empty_count = truncation_count = valid_count = total_tokens = 0
    canonical_size_bytes = 0
    max_canonical_row_bytes = 0

    for row_index, example in enumerate(_iter_examples(source_path)):
        try:
            text, schema = _extract_text(example)
        except ValueError as error:
            if str(error) == "empty example":
                empty_count += 1
                continue
            raise ValueError(f"Dataset row {row_index + 1}: {error}") from error

        tokens = _token_count(text, tokenizer)
        canonical_row_bytes = len(
            (json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
        canonical_size_bytes += canonical_row_bytes
        max_canonical_row_bytes = max(max_canonical_row_bytes, canonical_row_bytes)
        total_tokens += tokens
        valid_count += 1
        schema_counts[schema] += 1
        if sequence_length is not None and tokens > sequence_length:
            truncation_count += 1

        normalized_hash = hashlib.sha256(
            " ".join(text.split()).encode("utf-8")
        ).hexdigest()
        if normalized_hash in content_hashes:
            duplicate_count += 1
        else:
            content_hashes.add(normalized_hash)

        item = (row_index, tokens)
        if limit is None:
            reservoir.append(item)
        elif len(reservoir) < limit:
            reservoir.append(item)
        else:
            replacement = rng.randint(0, valid_count - 1)
            if replacement < limit:
                reservoir[replacement] = item

    if valid_count == 0:
        raise ValueError("Dataset contains no non-empty supported examples.")

    reservoir.sort(key=lambda item: item[0])
    sample_counts = [tokens for _, tokens in reservoir]
    schema_name = next(iter(schema_counts)) if len(schema_counts) == 1 else "mixed"
    warnings: list[str] = []
    measurement = (
        MeasurementKind.TOKENIZER_MEASURED
        if tokenizer is not None
        else MeasurementKind.ESTIMATED
    )
    if measurement == MeasurementKind.ESTIMATED:
        warnings.append(
            "Token counts use a four-characters-per-token estimate; model-data validation must retokenize with the selected tokenizer."
        )
    if limit is not None and valid_count > len(reservoir):
        warnings.append(
            f"Length statistics use a deterministic representative sample of {len(reservoir)} from {valid_count} examples."
        )
    if duplicate_count:
        warnings.append(f"Detected {duplicate_count} normalized duplicate example(s).")
    if empty_count:
        warnings.append(f"Ignored {empty_count} empty example(s).")
    if truncation_count:
        warnings.append(
            f"{truncation_count} example(s) exceed sequence length {sequence_length} and would be truncated."
        )

    observed_at = datetime.fromtimestamp(
        source_path.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    return DatasetProfile(
        source_path=source_path,
        source_sha256=source_digest,
        source_format=source_path.suffix.lower().lstrip("."),
        schema_name=schema_name,
        example_count=valid_count,
        total_estimated_tokens=total_tokens,
        sequence_p50=_nearest_rank(sample_counts, 0.50),
        sequence_p95=_nearest_rank(sample_counts, 0.95),
        sequence_max=max(sample_counts),
        measurement=measurement,
        warnings=tuple(warnings),
        schema_counts=dict(sorted(schema_counts.items())),
        sampled_examples=len(reservoir),
        sample_indices=tuple(index for index, _ in reservoir),
        duplicate_count=duplicate_count,
        empty_count=empty_count,
        truncation_count=truncation_count,
        truncation_rate=truncation_count / valid_count,
        source_size_bytes=source_path.stat().st_size,
        canonical_size_bytes=canonical_size_bytes,
        max_canonical_row_bytes=max_canonical_row_bytes,
        provenance=Provenance(
            ProvenanceKind.MEASURED, str(source_path), observed_at, source_digest
        ),
    )


def pilot_sample_rows(
    profile: DatasetProfile, *, limit: int = 32
) -> tuple[dict[str, Any], ...]:
    """Recover a bounded, deterministic pressure sample from profiled row indices."""

    if limit <= 0:
        raise ValueError("pilot sample limit must be positive.")
    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    for row_index, example in enumerate(_iter_examples(profile.source_path)):
        try:
            text, _schema = _extract_text(example)
        except ValueError as error:
            if str(error) == "empty example":
                continue
            raise ValueError(f"Dataset row {row_index + 1}: {error}") from error
        item = (len(text), -row_index, row_index, example)
        if len(candidates) < limit:
            heapq.heappush(candidates, item)
        elif item[:2] > candidates[0][:2]:
            heapq.heapreplace(candidates, item)
    if not candidates:
        raise ValueError("The profiled dataset sample contains no usable pilot rows.")
    candidates.sort(key=lambda item: (-item[0], item[2]))
    return tuple(item[3] for item in candidates)


def canonical_training_rows(profile: DatasetProfile) -> Iterator[dict[str, Any]]:
    """Yield every non-empty, schema-valid row for canonical JSONL compilation."""

    for row_index, example in enumerate(_iter_examples(profile.source_path)):
        try:
            _extract_text(example)
        except ValueError as error:
            if str(error) == "empty example":
                continue
            raise ValueError(f"Dataset row {row_index + 1}: {error}") from error
        yield example


def build_hardware_spec(
    *,
    backend: Backend,
    gpu_count: int,
    vram_gib: float,
    supports_bf16: bool,
    supports_4bit: bool,
    supports_8bit: bool = False,
    host_ram_gib: float,
    host_ram_free_gib: float | None = None,
    reserve_gib: float,
    free_vram_gib: float | None = None,
    disk_free_gib: float | None = None,
) -> HardwareSpec:
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive.")
    provenance = Provenance(ProvenanceKind.USER_ATTESTED, "cli-or-api")
    devices = tuple(
        DeviceSpec(
            name=f"{backend.value.upper()} GPU {index}",
            backend=backend,
            total_vram_bytes=gibibytes(vram_gib),
            supports_bf16=supports_bf16,
            supports_4bit=supports_4bit,
            supports_8bit=supports_8bit,
            free_vram_bytes=gibibytes(free_vram_gib)
            if free_vram_gib is not None
            else None,
            provenance=provenance,
        )
        for index in range(gpu_count)
    )
    return HardwareSpec(
        devices=devices,
        host_ram_bytes=gibibytes(host_ram_gib),
        host_ram_free_bytes=gibibytes(host_ram_free_gib)
        if host_ram_free_gib is not None
        else None,
        reserve_per_device_bytes=gibibytes(reserve_gib),
        disk_free_bytes=gibibytes(disk_free_gib) if disk_free_gib is not None else None,
        provenance=provenance,
    )


def _linux_available_memory(
    meminfo_path: Path = Path("/proc/meminfo"),
) -> int | None:
    if platform.system() != "Linux":
        return None
    try:
        lines = meminfo_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        name, separator, raw_value = line.partition(":")
        if name != "MemAvailable" or not separator:
            continue
        fields = raw_value.split()
        if len(fields) != 2 or fields[1] != "kB":
            return None
        try:
            value = int(fields[0]) * 1024
        except ValueError:
            return None
        return value if value > 0 else None
    return None


def _host_memory() -> tuple[int, int | None]:
    if os.name == "nt":  # pragma: no cover - exercised on Windows hosts.
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ValueError("Windows host-memory inspection failed.")
        return int(status.total_physical), int(status.available_physical)
    if not hasattr(os, "sysconf"):
        raise ValueError("Host-memory inspection is unavailable on this platform.")
    host_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if platform.system() == "Darwin":
        measured_available = _darwin_available_memory(host_ram)
        if measured_available is not None:
            return host_ram, measured_available
    if platform.system() == "Linux":
        measured_available = _linux_available_memory()
        if measured_available is not None:
            return host_ram, min(measured_available, host_ram)
    try:
        host_free = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError):
        host_free = None
    return host_ram, host_free


def _nvidia_smi_devices() -> list[tuple[str, int, int, str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"nvidia-smi hardware probe failed: {error}") from error
    if completed.returncode:
        return []
    rows: list[tuple[str, int, int, str]] = []
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) != 4:
            raise ValueError("nvidia-smi returned an unexpected hardware row.")
        name, total_mib, free_mib, capability = (item.strip() for item in row)
        rows.append(
            (
                name,
                round(float(total_mib) * 1024**2),
                round(float(free_mib) * 1024**2),
                capability,
            )
        )
    return rows


def _bitsandbytes_capabilities(major: int, minor: int) -> tuple[bool, bool]:
    """Return current NF4/FP4 and LLM.int8 hardware eligibility."""

    capability = (major, minor)
    return capability >= (6, 0), capability >= (7, 5)


def _apple_silicon_chip_name() -> str | None:
    """Read only the Apple CPU brand key, never a serial or hardware UUID."""

    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not completed.returncode:
        chip_name = completed.stdout.strip()
        if re.fullmatch(r"Apple M[0-9]+(?: (?:Pro|Max|Ultra))?", chip_name):
            return chip_name

    # ``detailLevel mini`` omits the serial number and hardware UUID. Parse only
    # the chip field and discard the rest of the bounded local response.
    fallback = _fixed_command_text(
        [
            "/usr/sbin/system_profiler",
            "-detailLevel",
            "mini",
            "SPHardwareDataType",
            "-json",
        ],
        timeout=8,
    )
    if not fallback:
        return None
    try:
        payload = json.loads(fallback)
        rows = payload.get("SPHardwareDataType")
        chip_name = (
            rows[0].get("chip_type") if isinstance(rows, list) and rows else None
        )
    except (AttributeError, IndexError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(chip_name, str) or not re.fullmatch(
        r"Apple M[0-9]+(?: (?:Pro|Max|Ultra))?", chip_name
    ):
        return None
    return chip_name


def _fixed_command_text(arguments: list[str], *, timeout: float = 3) -> str | None:
    """Run one fixed local probe without a shell or inherited user arguments."""

    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode:
        return None
    return completed.stdout.strip()


def _apple_metal_gpu_core_count() -> int | None:
    """Read the built-in Apple GPU core count without collecting identifiers."""

    output = _fixed_command_text(
        [
            "/usr/sbin/system_profiler",
            "-detailLevel",
            "mini",
            "SPDisplaysDataType",
            "-json",
        ],
        timeout=8,
    )
    if not output:
        return None
    try:
        payload = json.loads(output)
        rows = payload.get("SPDisplaysDataType")
    except (AttributeError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(rows, list):
        return None
    values: set[int] = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("sppci_bus") != "spdisplays_builtin"
            or row.get("sppci_device_type") != "spdisplays_gpu"
        ):
            continue
        raw = row.get("sppci_cores")
        if isinstance(raw, str) and raw.isascii() and raw.isdigit():
            value = int(raw)
            if 0 < value <= 1024:
                values.add(value)
    return values.pop() if len(values) == 1 else None


def _darwin_available_memory(total_memory: int) -> int | None:
    """Return a conservative available-memory estimate from ``vm_stat``."""

    output = _fixed_command_text(["/usr/bin/vm_stat"])
    if not output:
        return None
    page_match = re.search(r"page size of\s+(\d+) bytes", output)
    if page_match is None:
        return None
    page_size = int(page_match.group(1))
    page_counts: dict[str, int] = {}
    for line in output.splitlines():
        match = re.fullmatch(r"([^:]+):\s*([0-9]+)\.", line.strip())
        if match is not None:
            page_counts[match.group(1)] = int(match.group(2))
    names = ("Pages free", "Pages inactive", "Pages speculative")
    if not any(name in page_counts for name in names):
        return None
    available = sum(page_counts.get(name, 0) for name in names) * page_size
    if available <= 0:
        return None
    return min(total_memory, available)


def _darwin_memory_free_percent() -> int | None:
    output = _fixed_command_text(["/usr/bin/memory_pressure", "-Q"])
    if not output:
        return None
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    if match is None:
        return None
    value = int(match.group(1))
    return value if 0 <= value <= 100 else None


def _size_token_bytes(value: str, unit: str) -> int:
    multipliers = {
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }
    return round(float(value) * multipliers[unit.upper()])


def _darwin_swap_usage() -> tuple[int | None, int | None, int | None]:
    output = _fixed_command_text(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    if not output:
        return None, None, None
    values: dict[str, int] = {}
    for name in ("total", "used", "free"):
        match = re.search(
            rf"\b{name}\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGT])\b",
            output,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None, None, None
        values[name] = _size_token_bytes(match.group(1), match.group(2))
    return values["total"], values["used"], values["free"]


def _metal_recommended_working_set_bytes() -> int | None:
    """Read ``MTLDevice.recommendedMaxWorkingSetSize`` without PyObjC."""

    if platform.system() != "Darwin":
        return None
    try:
        metal = ctypes.CDLL(
            "/System/Library/Frameworks/Metal.framework/Versions/Current/Metal"
        )
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        create_device = metal.MTLCreateSystemDefaultDevice
        create_device.restype = ctypes.c_void_p
        device = create_device()
        if not device:
            return None
        selector_for_name = objc.sel_registerName
        selector_for_name.argtypes = [ctypes.c_char_p]
        selector_for_name.restype = ctypes.c_void_p
        selector = selector_for_name(b"recommendedMaxWorkingSetSize")
        send = objc.objc_msgSend
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        send.restype = ctypes.c_uint64
        measured = int(send(device, selector))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return measured if measured > 0 else None


def _distribution_version(*distribution_names: str) -> str | None:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _module_is_installed(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:  # Native package discovery can execute a broken parent import.
        return False


def _probe_mlx_runtime() -> RuntimeCapability:
    installed = _module_is_installed("mlx.core")
    version = _distribution_version("mlx")
    if not installed:
        return RuntimeCapability(False, False, version, "mlx.core is not installed.")
    try:
        mlx_core = importlib.import_module("mlx.core")
        metal = getattr(mlx_core, "metal", None)
        available = bool(metal is not None and metal.is_available())
        if available:
            device_info = getattr(mlx_core, "device_info", None) or getattr(
                metal, "device_info", None
            )
            if device_info is None or not isinstance(device_info(), dict):
                available = False
    except Exception as error:  # A broken native runtime must remain unavailable.
        return RuntimeCapability(
            True,
            False,
            version,
            f"MLX import or Metal capability probe failed: {error}",
        )
    detail = (
        "MLX reports a usable Metal device."
        if available
        else "MLX is installed but reports no usable Metal device."
    )
    return RuntimeCapability(True, available, version, detail)


def _probe_mlx_lm_runtime(mlx: RuntimeCapability) -> RuntimeCapability:
    installed = _module_is_installed("mlx_lm")
    version = _distribution_version("mlx-lm", "mlx_lm")
    if not installed:
        return RuntimeCapability(False, False, version, "mlx_lm is not installed.")
    try:
        importlib.import_module("mlx_lm")
    except Exception as error:  # Native dependency failures are capability facts.
        return RuntimeCapability(True, False, version, f"MLX-LM import failed: {error}")
    if not mlx.available:
        return RuntimeCapability(
            True,
            False,
            version,
            "MLX-LM imports, but the current MLX runtime has no usable Metal device.",
        )
    return RuntimeCapability(
        True, True, version, "MLX-LM imports in the Metal-capable MLX runtime."
    )


def _probe_pytorch_mps_runtime() -> RuntimeCapability:
    installed = _module_is_installed("torch")
    version = _distribution_version("torch")
    if not installed:
        return RuntimeCapability(False, False, version, "torch is not installed.")
    try:
        torch = importlib.import_module("torch")
        mps_backend = torch.backends.mps
        built = bool(mps_backend.is_built())
        available = bool(mps_backend.is_available())
    except Exception as error:
        return RuntimeCapability(
            True, False, version, f"PyTorch MPS capability probe failed: {error}"
        )
    return RuntimeCapability(
        True,
        available,
        version,
        f"PyTorch MPS built={str(built).lower()}, available={str(available).lower()}.",
    )


def probe_apple_platform() -> ApplePlatformProfile:
    """Measure the current Apple host and interpreter without chip allowlists."""

    system = platform.system()
    architecture = platform.machine().lower()
    if system != "Darwin" or architecture != "arm64":
        raise ValueError("Apple platform probing requires a Darwin arm64 host.")
    host_ram, host_available = _host_memory()
    mlx = _probe_mlx_runtime()
    swap_total, swap_used, swap_free = _darwin_swap_usage()
    return ApplePlatformProfile(
        system=system,
        architecture=architecture,
        os_version=_fixed_command_text(["/usr/bin/sw_vers", "-productVersion"])
        or platform.mac_ver()[0],
        os_build=_fixed_command_text(["/usr/bin/sw_vers", "-buildVersion"]),
        chip_name=_apple_silicon_chip_name(),
        logical_cpu_count=os.cpu_count(),
        metal_gpu_core_count=_apple_metal_gpu_core_count(),
        unified_memory_bytes=host_ram,
        available_memory_bytes=host_available,
        memory_free_percent=_darwin_memory_free_percent(),
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swap_free_bytes=swap_free,
        metal_recommended_working_set_bytes=_metal_recommended_working_set_bytes(),
        mlx=mlx,
        mlx_lm=_probe_mlx_lm_runtime(mlx),
        pytorch_mps=_probe_pytorch_mps_runtime(),
        observed_at=datetime.now(timezone.utc).isoformat(),
    )


def probe_local_hardware(
    *, reserve_gib: float = 2.0, disk_path: Path | None = None
) -> HardwareSpec:
    """Measure only the machine running Aptus; never infer a remote client's hardware."""

    host_ram, host_free = _host_memory()
    disk_target = (disk_path or Path.cwd()).resolve()
    observed_at = datetime.now(timezone.utc).isoformat()
    nvidia_rows = _nvidia_smi_devices()
    disk_free = shutil.disk_usage(disk_target).free
    devices: list[DeviceSpec] = []
    torch_module: Any | None = None
    try:
        import torch

        torch_module = torch
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                free, total = torch.cuda.mem_get_info(index)
                major, minor = torch.cuda.get_device_capability(index)
                try:
                    with torch.cuda.device(index):
                        bf16 = bool(torch.cuda.is_bf16_supported())
                except RuntimeError as error:
                    raise ValueError(
                        f"CUDA device {index} capability probe failed: {error}"
                    ) from error
                supports_4bit, supports_8bit = _bitsandbytes_capabilities(major, minor)
                devices.append(
                    DeviceSpec(
                        name=properties.name,
                        backend=Backend.CUDA,
                        total_vram_bytes=total,
                        free_vram_bytes=free,
                        supports_bf16=bf16,
                        supports_4bit=supports_4bit,
                        supports_8bit=supports_8bit,
                        compute_capability=f"{major}.{minor}",
                        provenance=Provenance(
                            ProvenanceKind.MEASURED, "torch.cuda", observed_at
                        ),
                    )
                )
    except ImportError:
        torch_module = None
    except RuntimeError as error:
        raise ValueError(f"CUDA runtime hardware probe failed: {error}") from error
    if torch_module is not None and not devices and nvidia_rows:
        raise ValueError(
            "nvidia-smi sees physical GPUs, but the local torch runtime exposes no CUDA devices. Aptus will not plan against hardware the execution runtime cannot see."
        )
    if torch_module is None and nvidia_rows:
        visible_filter = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_filter is not None:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES is set, but torch is unavailable to reconcile its logical device order with nvidia-smi. Install the pinned runtime or supply manual facts."
            )
        for name, total, free, capability in nvidia_rows:
            try:
                capability_parts = capability.split(".", 1)
                compute_major = int(capability_parts[0])
                compute_minor = (
                    int(capability_parts[1]) if len(capability_parts) == 2 else 0
                )
            except ValueError:
                compute_major, compute_minor = 0, 0
            supports_4bit, supports_8bit = _bitsandbytes_capabilities(
                compute_major, compute_minor
            )
            devices.append(
                DeviceSpec(
                    name=name,
                    backend=Backend.CUDA,
                    total_vram_bytes=total,
                    free_vram_bytes=free,
                    supports_bf16=False,
                    supports_4bit=supports_4bit,
                    supports_8bit=supports_8bit,
                    compute_capability=capability,
                    provenance=Provenance(
                        ProvenanceKind.MEASURED,
                        "nvidia-smi without an installed torch runtime",
                        observed_at,
                        detail="CUDA_VISIBLE_DEVICES was unset. Name, VRAM, and compute capability came from nvidia-smi. BF16 remains false until a runtime probe. Four-bit requires compute capability 6.0 or newer; LLM.int8 requires 7.5 or newer. Both still require dependency and measured-preflight verification.",
                    ),
                )
            )
    if (
        not devices
        and platform.system() == "Darwin"
        and platform.machine().lower() == "arm64"
    ):
        chip_name = _apple_silicon_chip_name() or "Apple Silicon"
        metal_working_set = _metal_recommended_working_set_bytes()
        compatibility_capacity = metal_working_set or host_ram
        devices.append(
            DeviceSpec(
                name=f"{chip_name} (shared unified memory)",
                backend=Backend.MPS,
                total_vram_bytes=compatibility_capacity,
                free_vram_bytes=None,
                supports_bf16=False,
                supports_4bit=False,
                supports_8bit=False,
                provenance=Provenance(
                    ProvenanceKind.MEASURED,
                    "local Darwin arm64 host",
                    observed_at,
                    detail=(
                        "This compatibility device represents Apple shared unified "
                        "memory, not dedicated VRAM. total_vram_bytes is the Metal "
                        "recommended working-set ceiling when measurable, otherwise "
                        "the measured unified-memory capacity. free_vram_bytes is "
                        "intentionally omitted because host free RAM is not free VRAM. "
                        "CUDA and bitsandbytes flags do not establish MLX or MPS "
                        "support; use probe_apple_platform for runtime capability facts."
                    ),
                ),
            )
        )
    if not devices:
        raise ValueError(
            "CUDA hardware inspection is unavailable on this Aptus host. "
            "Supply explicit manual facts; Aptus will not infer a remote user's hardware."
        )
    provenance_detail = (
        "Host RAM and disk availability were measured on the server running Aptus."
    )
    if devices[0].backend == Backend.MPS:
        provenance_detail += (
            " The MPS compatibility device uses shared unified memory, not dedicated "
            "VRAM. Host available memory is recorded only as host RAM headroom."
        )
    return HardwareSpec(
        devices=tuple(devices),
        host_ram_bytes=host_ram,
        host_ram_free_bytes=host_free,
        reserve_per_device_bytes=gibibytes(reserve_gib),
        disk_free_bytes=disk_free,
        provenance=Provenance(
            ProvenanceKind.MEASURED,
            f"local-host:{disk_target}",
            observed_at,
            detail=provenance_detail,
        ),
    )


def build_model_spec(
    *,
    model_id: str,
    revision: str,
    family: str,
    parameters_b: float,
    hidden_size: int,
    layers: int,
    context_length: int,
    license_name: str,
    training_allowed: bool,
    intermediate_size: int | None = None,
    architecture: str = "causal-lm",
    model_type: str | None = None,
    quantization_bits: int | None = None,
    quantization_layout: QuantizationLayout | Mapping[str, Any] | None = None,
    moe: MoETopology | Mapping[str, Any] | None = None,
) -> ModelSpec:
    provenance = Provenance(ProvenanceKind.USER_ATTESTED, "cli-or-api")
    topology = (
        MoETopology(
            expert_count=int(moe["expert_count"]),
            experts_per_token=int(moe["experts_per_token"]),
            expert_intermediate_size=int(moe["expert_intermediate_size"]),
            decoder_sparse_step=int(moe["decoder_sparse_step"]),
            mlp_only_layers=tuple(moe.get("mlp_only_layers", ())),
            shared_expert_intermediate_size=(
                int(moe["shared_expert_intermediate_size"])
                if moe.get("shared_expert_intermediate_size") is not None
                else None
            ),
        )
        if isinstance(moe, Mapping)
        else moe
    )
    layout = (
        QuantizationLayout(
            default_bits=quantization_layout["default_bits"],
            default_group_size=quantization_layout["default_group_size"],
            module_overrides=tuple(
                sorted(
                    (
                        QuantizationOverride(
                            module_path=item["module_path"],
                            bits=item["bits"],
                            group_size=item["group_size"],
                        )
                        for item in quantization_layout.get("module_overrides", ())
                    ),
                    key=lambda item: item.module_path,
                )
            ),
        )
        if isinstance(quantization_layout, Mapping)
        else quantization_layout
    )
    return ModelSpec(
        model_id=model_id,
        revision=revision,
        family=family.lower(),
        parameters=round(parameters_b * 1_000_000_000),
        hidden_size=hidden_size,
        layers=layers,
        context_length=context_length,
        license_name=license_name,
        training_allowed=training_allowed,
        intermediate_size=intermediate_size,
        architecture=architecture,
        model_type=model_type,
        quantization_bits=quantization_bits,
        quantization_layout=layout,
        moe=topology,
        provenance={"all": provenance},
    )
