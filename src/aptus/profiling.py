from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .domain import (
    Backend,
    DatasetProfile,
    DeviceSpec,
    HardwareSpec,
    MeasurementKind,
    ModelSpec,
    Provenance,
    ProvenanceKind,
    gibibytes,
)


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
    if not devices:
        raise ValueError(
            "CUDA hardware inspection is unavailable on this Aptus host. "
            "Supply explicit manual facts; Aptus will not infer a remote user's hardware."
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
            detail="Host RAM and disk availability were measured on the server running Aptus.",
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
) -> ModelSpec:
    provenance = Provenance(ProvenanceKind.USER_ATTESTED, "cli-or-api")
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
        provenance={"all": provenance},
    )
