from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses the in-process lock.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX uses fcntl.
    msvcrt = None

from .domain import RunState, ValidationState
from .plan_contract import (
    sha256_file,
    validate_bundle_manifest,
    validate_plan_payload,
)


JobAction = Literal["dependency", "model-data", "preflight", "pilot", "train"]
JOB_ACTIONS = {"dependency", "model-data", "preflight", "pilot", "train"}
_GLOBAL_LEASE_THREAD_LOCK = threading.RLock()


class ActiveJobError(ValueError):
    """Raised when a host or state-root execution lease is already active."""


class JobPrerequisiteError(ValueError):
    """Raised when a job action skips its required validation-report stage."""

    code = "job_prerequisite_not_met"

    def __init__(
        self,
        *,
        action: JobAction,
        required_state: str,
        current_state: str | None,
        reason: str,
    ) -> None:
        self.action = action
        self.required_state = required_state
        self.current_state = current_state
        self.reason = reason
        observed = current_state if current_state is not None else "none"
        super().__init__(
            f"Cannot start {action}: validation-report.json has state {observed!r}; "
            f"the required prior state is {required_state!r} or later. "
            "Run each preceding validation stage in order, then retry."
        )


_VALIDATION_STATE_RANK = {
    ValidationState.CONTRACT_PASS.value: 1,
    ValidationState.STATIC_PASS.value: 2,
    ValidationState.DEPENDENCY_PASS.value: 3,
    ValidationState.MODEL_DATA_PASS.value: 4,
    ValidationState.MEASURED_PREFLIGHT_PASS.value: 5,
    ValidationState.PILOT_PASS.value: 6,
    ValidationState.EXECUTION_APPROVED.value: 7,
    ValidationState.MEASURED_RUN_PASS.value: 8,
}
_JOB_PREREQUISITE_STATES: dict[JobAction, str] = {
    "dependency": ValidationState.STATIC_PASS.value,
    "model-data": ValidationState.DEPENDENCY_PASS.value,
    "preflight": ValidationState.MODEL_DATA_PASS.value,
    "pilot": ValidationState.MEASURED_PREFLIGHT_PASS.value,
    "train": ValidationState.PILOT_PASS.value,
}


_PERMIT_LAUNCHER = r"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

spec_path = Path(sys.argv[1])
permit_path = Path(sys.argv[2])
deadline = time.monotonic() + 30
while not permit_path.is_file():
    if time.monotonic() >= deadline:
        raise SystemExit(125)
    time.sleep(0.05)
command = json.loads(spec_path.read_text(encoding="utf-8"))["command"]
if os.name == "nt":
    raise SystemExit(subprocess.run(command, check=False).returncode)
os.execvpe(command[0], command, os.environ)
"""

_CUDA_RUNTIME_PROBE = r"""
import json
import os
import shutil
import subprocess
import sys

try:
    import torch
except ImportError as error:
    raise SystemExit(f"torch import failed: {error}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA hardware is unavailable")
try:
    device_indices = json.loads(sys.argv[1])
except (IndexError, json.JSONDecodeError) as error:
    raise SystemExit(f"planned CUDA device indices are invalid: {error}")
if (
    not isinstance(device_indices, list)
    or any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in device_indices
    )
    or len(set(device_indices)) != len(device_indices)
    or any(index >= torch.cuda.device_count() for index in device_indices)
):
    raise SystemExit("the planned CUDA device selection is unavailable")
driver_getter = getattr(getattr(torch, "_C", None), "_cuda_getDriverVersion", None)
driver_version = None
if callable(driver_getter):
    try:
        driver_value = driver_getter()
    except (RuntimeError, TypeError):
        driver_value = None
    if isinstance(driver_value, int) and driver_value > 0:
        driver_version = str(driver_value)
if driver_version is None:
    executable = shutil.which("nvidia-smi")
    if executable is not None:
        try:
            completed = subprocess.run(
                [executable, "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
            if len(versions) == 1:
                driver_version = versions.pop()
if driver_version is None:
    raise SystemExit("CUDA driver identity is unavailable")
if os.name == "nt":
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
        raise SystemExit("Windows host-memory inspection failed")
    host_ram_free_bytes = int(status.available_physical)
elif hasattr(os, "sysconf"):
    try:
        host_ram_free_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"available host-memory inspection failed: {error}")
else:
    raise SystemExit("available host-memory inspection is unsupported")
devices = []
for logical_index, physical_index in enumerate(device_indices):
    properties = torch.cuda.get_device_properties(physical_index)
    major, minor = torch.cuda.get_device_capability(physical_index)
    device_uuid = str(getattr(properties, "uuid", "")).strip()
    if not device_uuid or device_uuid.lower() == "none":
        raise SystemExit(f"CUDA device {physical_index} has no stable UUID")
    devices.append({
        "index": logical_index,
        "name": properties.name,
        "uuid": device_uuid,
        "pci_bus_id": str(getattr(properties, "pci_bus_id", "")),
        "total_vram_bytes": properties.total_memory,
        "compute_capability": f"{major}.{minor}",
    })
print(json.dumps({
    "hardware": {
        "cuda_runtime": torch.version.cuda,
        "driver_version": driver_version,
        "devices": devices,
    },
    "free_cuda_bytes": [int(torch.cuda.mem_get_info(index)[0]) for index in device_indices],
    "host_ram_free_bytes": host_ram_free_bytes,
}, sort_keys=True))
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any] | dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _verify_run_metrics(metrics: dict[str, Any], candidate: dict[str, Any]) -> None:
    global_step = metrics.get("global_step")
    train_loss = metrics.get("train_loss")
    if (
        not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or global_step < 1
    ):
        raise ValueError(
            "Measured-run metrics do not record a completed training step."
        )
    if (
        not isinstance(train_loss, (int, float))
        or isinstance(train_loss, bool)
        or not math.isfinite(train_loss)
    ):
        raise ValueError("Measured-run metrics do not record a finite train_loss.")
    for name, value in metrics.items():
        if name.endswith("loss") and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"Measured-run metric {name} is not finite.")
    expected = {
        "candidate_id": candidate["candidate_id"],
        "pilot": False,
        "distribution": candidate["distribution"],
        "actual_world_size": candidate["world_size"],
    }
    for name, value in expected.items():
        if metrics.get(name) != value:
            raise ValueError(f"Measured-run metrics do not bind {name}.")
    world_size = int(candidate["world_size"])
    per_rank = metrics.get("per_rank_cuda_peaks")
    if not isinstance(per_rank, list) or len(per_rank) != world_size:
        raise ValueError("Measured-run metrics require one CUDA peak per rank.")
    allocated_values: list[int] = []
    reserved_values: list[int] = []
    for expected_rank, value in enumerate(per_rank):
        if not isinstance(value, dict) or value.get("rank") != expected_rank:
            raise ValueError("Measured-run CUDA peak ranks are invalid.")
        allocated = value.get("measured_peak_cuda_bytes")
        reserved = value.get("measured_reserved_cuda_bytes")
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in (allocated, reserved)
        ):
            raise ValueError("Measured-run CUDA peaks must be non-negative integers.")
        if reserved < allocated:
            raise ValueError(
                "Measured-run reserved CUDA memory is below allocated memory."
            )
        allocated_values.append(allocated)
        reserved_values.append(reserved)
    if not reserved_values or max(reserved_values) <= 0:
        raise ValueError("Measured-run metrics contain no positive CUDA memory peak.")
    if metrics.get("measured_peak_cuda_bytes") != max(allocated_values):
        raise ValueError("Measured-run aggregate allocated CUDA peak is inconsistent.")
    if metrics.get("measured_reserved_cuda_bytes") != max(reserved_values):
        raise ValueError("Measured-run aggregate reserved CUDA peak is inconsistent.")


def _verify_safetensors_structure(final_dir: Path, weight_files: list[Path]) -> None:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ValueError(
            "Parent verification requires the pinned safetensors runtime."
        ) from error
    tensor_shards: dict[str, str] = {}
    try:
        for weight_path in weight_files:
            with safe_open(str(weight_path), framework="pt", device="cpu") as tensors:
                tensor_keys = list(tensors.keys())
            if not tensor_keys:
                raise ValueError(
                    f"Final safetensors shard has no tensor keys: {weight_path.name}."
                )
            if any(not isinstance(key, str) or not key for key in tensor_keys):
                raise ValueError(
                    f"Final safetensors shard has an invalid tensor key: {weight_path.name}."
                )
            for key in tensor_keys:
                previous_shard = tensor_shards.get(key)
                if previous_shard is not None:
                    raise ValueError(
                        "Final safetensors shards contain duplicate tensor key "
                        f"{key!r}: {previous_shard} and {weight_path.name}."
                    )
                tensor_shards[key] = weight_path.name
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            "Final safetensors weights failed parent structural loading."
        ) from error

    index_files = sorted(final_dir.glob("*.safetensors.index.json"))
    if len(index_files) > 1:
        raise ValueError("Final export contains multiple safetensors indexes.")
    if len(weight_files) > 1 and not index_files:
        raise ValueError("A multi-shard final export requires one safetensors index.")
    if not index_files:
        return
    index = _read_json_object(index_files[0], "Final safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Final safetensors index has no weight map.")
    if any(
        not isinstance(key, str) or not key or not isinstance(shard, str) or not shard
        for key, shard in weight_map.items()
    ):
        raise ValueError("Final safetensors index has an invalid weight map.")
    if set(weight_map) != set(tensor_shards):
        raise ValueError("Final safetensors index keys do not match shard tensors.")
    mismatched_keys = [
        key for key, shard in weight_map.items() if tensor_shards[key] != shard
    ]
    if mismatched_keys:
        raise ValueError(
            "Final safetensors index maps tensor keys to the wrong shards: "
            + ", ".join(sorted(mismatched_keys)[:10])
        )


def _verify_train_artifacts(record: dict[str, Any]) -> dict[str, Any]:
    bundle = Path(record["bundle_dir"]).resolve(strict=True)
    run_value = record.get("run_output_dir")
    if not isinstance(run_value, str):
        raise ValueError("Training job has no bound output directory.")
    run_dir = Path(run_value).resolve(strict=True)
    expected_parent = (bundle / "runs").resolve()
    if run_dir.parent != expected_parent:
        raise ValueError(
            "Training output is outside the bundle's protected runs directory."
        )
    plan = _read_json_object(bundle / "plan.json", "Bundle plan")
    plan_errors = validate_plan_payload(plan, root=bundle, verify_dataset=True)
    if plan_errors:
        raise ValueError("Bundle plan is invalid: " + " | ".join(plan_errors))
    candidate = plan["recommended"]
    marker = _read_json_object(run_dir / ".aptus-run.json", "Run-output contract")
    marker_expected = {
        "schema_version": "aptus.run-output.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "pilot": False,
    }
    for name, value in marker_expected.items():
        if marker.get(name) != value:
            raise ValueError(f"Run-output contract does not bind {name}.")

    export_path = run_dir / "final-export.json"
    export = _read_json_object(export_path, "Final-export manifest")
    export_expected = {
        "schema_version": "aptus.final-export.v1",
        "verification_level": "structural-file-tree",
        "method": candidate["method"],
        "base_model": {
            "model_id": plan["model"]["model_id"],
            "revision": plan["model"]["revision"],
        },
        "distribution": candidate["distribution"],
        "world_size": candidate["world_size"],
        "device_indices": candidate.get(
            "device_indices", list(range(candidate["world_size"]))
        ),
    }
    for name, value in export_expected.items():
        if export.get(name) != value:
            raise ValueError(f"Final-export manifest does not bind {name}.")
    entries = export.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Final-export manifest has no file entries.")
    final_dir = (run_dir / "final").resolve(strict=True)
    observed_paths: set[str] = set()
    observed_total = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Final-export file entry is invalid.")
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Final-export manifest contains an unsafe path.")
        normalized = relative.as_posix()
        if normalized in observed_paths:
            raise ValueError("Final-export manifest contains a duplicate path.")
        path = final_dir.joinpath(*relative.parts)
        if not path.is_file() or final_dir not in path.resolve().parents:
            raise ValueError(f"Final-export file is missing or unsafe: {normalized}.")
        size = path.stat().st_size
        if entry.get("size_bytes") != size or entry.get("sha256") != sha256_file(path):
            raise ValueError(f"Final-export file changed: {normalized}.")
        observed_paths.add(normalized)
        observed_total += size
    actual_paths = {
        path.relative_to(final_dir).as_posix()
        for path in final_dir.rglob("*")
        if path.is_file()
    }
    if observed_paths != actual_paths or export.get("total_bytes") != observed_total:
        raise ValueError(
            "Final-export manifest does not match the exact artifact tree."
        )
    weight_files = export.get("weight_files")
    if not isinstance(weight_files, list) or not weight_files:
        raise ValueError("Final-export manifest has no method-specific weight files.")
    actual_weight_files = (
        sorted(path.name for path in final_dir.glob("model*.safetensors"))
        if candidate["method"] == "full"
        else sorted(path.name for path in final_dir.glob("adapter_model*.safetensors"))
    )
    if weight_files != actual_weight_files or not actual_weight_files:
        raise ValueError(
            "Final-export weight_files do not match method-specific weights."
        )
    _verify_safetensors_structure(
        final_dir, [final_dir / name for name in actual_weight_files]
    )
    required_config = (
        final_dir / "config.json"
        if candidate["method"] == "full"
        else final_dir / "adapter_config.json"
    )
    if not required_config.is_file():
        raise ValueError("Final export is missing its method-specific config.")
    if candidate["method"] != "full":
        adapter = _read_json_object(required_config, "Adapter config")
        if adapter.get("base_model_name_or_path") != plan["model"]["model_id"]:
            raise ValueError("Adapter config does not bind the planned base model.")
        if adapter.get("revision") != plan["model"]["revision"]:
            raise ValueError("Adapter config does not bind the immutable revision.")

    metrics_path = run_dir / "metrics.json"
    metrics = _read_json_object(metrics_path, "Measured-run metrics")
    _verify_run_metrics(metrics, candidate)
    if (
        metrics.get("plan_id") != plan["plan_id"]
        or metrics.get("final_export") != export
    ):
        raise ValueError("Measured-run metrics do not bind the plan and final export.")
    report = _read_json_object(bundle / "validation-report.json", "Validation report")
    final_report = report.get("pending_final_export")
    measured_report = report.get("pending_measured_run")
    if report.get("state") != "execution-approved":
        raise ValueError("Training exited without pending measured-run evidence.")
    if not isinstance(final_report, dict) or not isinstance(measured_report, dict):
        raise ValueError("Measured-run report is missing artifact bindings.")
    report_expected = {
        "path": str(final_dir),
        "manifest_sha256": sha256_file(export_path),
        "total_bytes": observed_total,
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "distribution": candidate["distribution"],
        "world_size": candidate["world_size"],
    }
    for name, value in report_expected.items():
        if final_report.get(name) != value:
            raise ValueError(f"Validation report does not bind final_export.{name}.")
    measured_expected = {
        "output_dir": str(run_dir),
        "metrics_sha256": sha256_file(metrics_path),
        "global_step": metrics["global_step"],
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "distribution": candidate["distribution"],
        "world_size": candidate["world_size"],
        "per_rank_cuda_peaks": metrics["per_rank_cuda_peaks"],
    }
    for name, value in measured_expected.items():
        if measured_report.get(name) != value:
            raise ValueError(f"Validation report does not bind measured_run.{name}.")
    active_run = report.get("active_run")
    expected_active = {
        "output_dir": str(run_dir),
        "run_id": run_dir.name,
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
    }
    if not isinstance(active_run, dict) or any(
        active_run.get(name) != value for name, value in expected_active.items()
    ):
        raise ValueError("Validation report does not bind the active run.")
    return {
        "final_export": final_report,
        "measured_run": measured_report,
        "active_run": active_run,
        "pending_at": report.get("measured_run_pending_at"),
    }


@contextmanager
def _bundle_report_lock(bundle: Path) -> Any:
    with (bundle / ".validation-report.lock").open("a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows only.
            lock_file.seek(0)
            if not lock_file.read(1):
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows only.
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _promote_train_attestation(
    record: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    bundle = Path(record["bundle_dir"]).resolve(strict=True)
    report_path = bundle / "validation-report.json"
    with _bundle_report_lock(bundle):
        report = _read_json_object(report_path, "Validation report")
        if (
            report.get("state") == "measured-run-pass"
            and report.get("final_export") == evidence["final_export"]
            and report.get("measured_run") == evidence["measured_run"]
        ):
            return {
                "state": report["state"],
                "measured_run_completed_at": report.get("measured_run_completed_at"),
                "final_export": report["final_export"],
                "measured_run": report["measured_run"],
            }
        if (
            report.get("state") != "execution-approved"
            or report.get("active_run") != evidence["active_run"]
            or report.get("pending_final_export") != evidence["final_export"]
            or report.get("pending_measured_run") != evidence["measured_run"]
            or report.get("measured_run_pending_at") != evidence["pending_at"]
        ):
            raise ValueError(
                "Pending measured-run evidence changed before parent promotion."
            )
        report["state"] = "measured-run-pass"
        report["measured_run_completed_at"] = _now()
        report["final_export"] = evidence["final_export"]
        report["measured_run"] = evidence["measured_run"]
        for name in (
            "active_run",
            "measured_run_pending_at",
            "pending_final_export",
            "pending_measured_run",
        ):
            report.pop(name, None)
        _atomic_write_json(report_path, report)
    return {
        "state": report["state"],
        "measured_run_completed_at": report["measured_run_completed_at"],
        "final_export": report["final_export"],
        "measured_run": report["measured_run"],
    }


def _verify_checkpoint_manifest(checkpoint: Path, contract: dict[str, Any]) -> None:
    if not checkpoint.is_dir():
        raise ValueError(f"Pilot checkpoint is missing: {checkpoint}.")
    expected_files = contract.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("Pilot checkpoint manifest is empty.")
    observed_files = [
        {
            "path": path.relative_to(checkpoint).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in checkpoint.rglob("*") if item.is_file())
    ]
    if (
        observed_files != expected_files
        or contract.get("manifest_sha256") != _json_hash(observed_files)
        or contract.get("total_bytes")
        != sum(item["size_bytes"] for item in observed_files)
    ):
        raise ValueError("Pilot checkpoint no longer matches its bound manifest.")


def _verify_pilot_artifacts(bundle: Path, metrics: dict[str, Any]) -> tuple[int, int]:
    if metrics.get("checkpoint_continuation_observed") is not True:
        raise ValueError("Pilot metrics do not attest checkpoint continuation.")
    pilot_run_dir = metrics.get("pilot_run_dir")
    if not isinstance(pilot_run_dir, str):
        raise ValueError("Pilot metrics do not bind an immutable pilot run.")
    relative = PurePosixPath(pilot_run_dir)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Pilot run path is unsafe.")
    pilot_root = bundle.joinpath(*relative.parts).resolve()
    if (
        len(relative.parts) != 2
        or relative.parts[0] != "runs"
        or not relative.parts[1].startswith("pilot_")
        or pilot_root.parent != (bundle / "runs").resolve()
        or metrics.get("pilot_run_id") != pilot_root.name
    ):
        raise ValueError("Pilot run path is not a bound Aptus pilot root.")
    marker = _read_json_object(
        pilot_root / ".aptus-pilot-run.json", "Pilot-run ownership contract"
    )
    plan = _read_json_object(bundle / "plan.json", "Bundle plan")
    marker_expected = {
        "schema_version": "aptus.pilot-run.v1",
        "pilot_run_id": pilot_root.name,
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
    }
    if any(marker.get(name) != value for name, value in marker_expected.items()):
        raise ValueError("Pilot-run ownership contract does not match the plan.")
    contracts: tuple[tuple[Path, Any], ...] = (
        (
            pilot_root / "phase-1" / "checkpoint-1",
            metrics.get("phase_one_checkpoint"),
        ),
        (
            pilot_root / "phase-2" / "checkpoint-2",
            metrics.get("phase_two_checkpoint"),
        ),
    )
    typed_contracts: list[dict[str, Any]] = []
    for checkpoint, contract in contracts:
        if not isinstance(contract, dict):
            raise ValueError("Pilot checkpoint contract is missing.")
        _verify_checkpoint_manifest(checkpoint, contract)
        typed_contracts.append(contract)
    checkpoint_bytes = metrics.get("measured_checkpoint_bytes")
    final_export_bytes = metrics.get("measured_final_export_bytes")
    expected_checkpoint_bytes = max(
        int(contract["total_bytes"]) for contract in typed_contracts
    )
    phases = (metrics.get("phase_one"), metrics.get("phase_two_resumed"))
    if not all(isinstance(phase, dict) for phase in phases):
        raise ValueError("Pilot phase metrics are missing.")
    try:
        expected_final_export_bytes = max(
            int(phase["final_export"]["total_bytes"])
            for phase in phases
            if isinstance(phase, dict)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Pilot final-export evidence is invalid.") from error
    if (
        checkpoint_bytes != expected_checkpoint_bytes
        or final_export_bytes != expected_final_export_bytes
    ):
        raise ValueError("Pilot capacity evidence is inconsistent.")
    if checkpoint_bytes <= 0 or final_export_bytes <= 0:
        raise ValueError("Pilot capacity evidence must be positive.")
    return checkpoint_bytes, final_export_bytes


def _environment_binding(bundle: Path) -> str:
    direct_constraints: dict[str, str] = {}
    for line in (bundle / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name = line.split("==", 1)[0]
        try:
            direct_constraints[name] = version(name)
        except PackageNotFoundError:
            direct_constraints[name] = "missing"
    runtime_distributions = _runtime_distribution_closure(direct_constraints)
    return _json_hash(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "direct_constraints": direct_constraints,
            "runtime_distributions": runtime_distributions,
        }
    )


def _runtime_distribution_closure(names: Mapping[str, str]) -> dict[str, str]:
    pending = list(names)
    observed: dict[str, str] = {}
    visited: set[str] = set()
    while pending:
        requested = pending.pop()
        normalized = requested.lower().replace("_", "-").replace(".", "-")
        if normalized in visited:
            continue
        visited.add(normalized)
        try:
            package = distribution(requested)
        except PackageNotFoundError:
            continue
        canonical = (package.metadata.get("Name") or requested).lower()
        canonical = canonical.replace("_", "-").replace(".", "-")
        observed[canonical] = package.version
        for requirement in package.requires or ():
            token = requirement.split(";", 1)[0].strip()
            boundary = min(
                (
                    token.find(character)
                    for character in "[ (<>=!~"
                    if character in token
                ),
                default=len(token),
            )
            dependency = token[:boundary].strip()
            if dependency:
                pending.append(dependency)
    return dict(sorted(observed.items()))


def _actual_runtime_snapshot(
    world_size: int = 0, device_indices: list[int] | None = None
) -> dict[str, Any]:
    selected_indices = (
        list(range(world_size)) if device_indices is None else list(device_indices)
    )
    if (
        len(selected_indices) != world_size
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in selected_indices
        )
        or len(set(selected_indices)) != len(selected_indices)
    ):
        raise ValueError("Selected CUDA device indices do not match the planned world.")
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _CUDA_RUNTIME_PROBE,
                json.dumps(selected_indices),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"CUDA runtime probe failed: {error}") from error
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"CUDA runtime probe failed: {detail}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("CUDA runtime probe returned invalid JSON.") from error
    if not isinstance(value, dict) or not isinstance(value.get("hardware"), dict):
        raise ValueError("CUDA runtime probe returned an invalid contract.")
    free = value.get("free_cuda_bytes")
    host_free = value.get("host_ram_free_bytes")
    if (
        not isinstance(free, list)
        or len(free) != world_size
        or any(not isinstance(item, int) or item < 0 for item in free)
        or not isinstance(host_free, int)
        or host_free <= 0
    ):
        raise ValueError("CUDA runtime probe returned invalid free-memory facts.")
    return value


def _actual_hardware_binding(device_indices: list[int]) -> str:
    selected_indices = device_indices
    return _json_hash(
        _actual_runtime_snapshot(len(selected_indices), selected_indices)["hardware"]
    )


def _actual_free_cuda_bytes(
    world_size: int, device_indices: list[int] | None = None
) -> tuple[int, ...]:
    return tuple(
        _actual_runtime_snapshot(world_size, device_indices)["free_cuda_bytes"]
    )


class JobService:
    """Local process manager with atomic records and cross-process submission locks.

    POSIX cancellation targets the launched process group. Windows uses a new process group
    and falls back to terminating the recorded process when tree termination is unavailable.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records_lock_state = threading.local()
        self._global_lease_lock_state = threading.local()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._threads: dict[str, threading.Thread] = {}
        identity = (
            str(os.getuid())
            if hasattr(os, "getuid")
            else os.environ.get("USERNAME", "default")
        )
        safe_identity = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        lease_parent = (
            Path("/tmp") if os.name == "posix" else Path(tempfile.gettempdir())
        )
        self._lease_root = lease_parent / f"aptus-gpu-lease-{safe_identity}"
        self._lease_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        lease_root_stat = self._lease_root.lstat()
        if self._lease_root.is_symlink() or not self._lease_root.is_dir():
            raise PermissionError(
                f"Aptus host-global lease root is not a secure directory: {self._lease_root}"
            )
        if hasattr(os, "getuid") and lease_root_stat.st_uid != os.getuid():
            raise PermissionError(
                f"Aptus host-global lease root is owned by another user: {self._lease_root}"
            )
        if os.name == "posix" and lease_root_stat.st_mode & 0o077:
            self._lease_root.chmod(0o700)
        self._lease_path = self._lease_root / "lease.json"
        self._recover_interrupted_jobs()

    @contextmanager
    def _records_lock(self) -> Any:
        with self._lock:
            depth = getattr(self._records_lock_state, "depth", 0)
            if depth:
                self._records_lock_state.depth = depth + 1
                try:
                    yield
                finally:
                    self._records_lock_state.depth -= 1
                return
            lock_path = self.root / ".jobs.lock"
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                elif msvcrt is not None:  # pragma: no cover - Windows only.
                    lock_file.seek(0)
                    if not lock_file.read(1):
                        lock_file.write("\0")
                        lock_file.flush()
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                self._records_lock_state.depth = 1
                try:
                    yield
                finally:
                    self._records_lock_state.depth = 0
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    elif msvcrt is not None:  # pragma: no cover - Windows only.
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    @contextmanager
    def _global_lease_lock(self) -> Any:
        depth = getattr(self._global_lease_lock_state, "depth", 0)
        if depth:
            self._global_lease_lock_state.depth = depth + 1
            try:
                yield
            finally:
                self._global_lease_lock_state.depth -= 1
            return
        lock_path = self._lease_root / ".lease.lock"
        with (
            _GLOBAL_LEASE_THREAD_LOCK,
            lock_path.open("a+", encoding="utf-8") as lock_file,
        ):
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows only.
                lock_file.seek(0)
                if not lock_file.read(1):
                    lock_file.write("\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            self._global_lease_lock_state.depth = 1
            try:
                yield
            finally:
                self._global_lease_lock_state.depth = 0
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows only.
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    def _record_path(self, job_id: str) -> Path:
        if not (
            job_id.startswith("job_")
            and len(job_id) == 36
            and all(character in "0123456789abcdef" for character in job_id[4:])
        ):
            raise KeyError(job_id)
        return self.root / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        self._record_path(job_id)
        return self.root / f"{job_id}.log"

    def _record_paths(self) -> list[Path]:
        with self._records_lock():
            result = []
            for path in self.root.glob("job_*.json"):
                try:
                    self._record_path(path.stem)
                except KeyError:
                    continue
                result.append(path)
            return sorted(result)

    def _write(self, record: dict[str, Any]) -> None:
        with self._records_lock():
            path = self._record_path(record["id"])
            _atomic_write_json(path, record)

    def _read(self, job_id: str) -> dict[str, Any]:
        with self._records_lock():
            path = self._record_path(job_id)
            if not path.is_file():
                raise KeyError(job_id)
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Unreadable Aptus job record {path}: {error}"
                ) from error
            if not isinstance(value, dict) or value.get("id") != job_id:
                raise ValueError(
                    f"Invalid Aptus job record {path}: the object ID must match its filename."
                )
            return value

    @staticmethod
    def _process_identity(value: Any) -> str | None:
        if not isinstance(value, int) or value <= 0:
            return None
        if sys.platform.startswith("linux"):
            try:
                fields = (
                    Path(f"/proc/{value}/stat")
                    .read_text(encoding="utf-8")
                    .rsplit(")", 1)[1]
                    .split()
                )
            except (OSError, IndexError):
                pass
            else:
                if len(fields) > 19:
                    return f"linux-start-ticks:{fields[19]}"
        if os.name == "posix":
            try:
                completed = subprocess.run(
                    ["ps", "-o", "lstart=", "-p", str(value)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            started = completed.stdout.strip()
            return f"{sys.platform}-started:{started}" if started else None
        if os.name == "nt":  # pragma: no cover - Windows only.
            try:
                import ctypes
                from ctypes import wintypes

                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, value)
                if not handle:
                    return None
                creation = wintypes.FILETIME()
                exit_time = wintypes.FILETIME()
                kernel = wintypes.FILETIME()
                user = wintypes.FILETIME()
                try:
                    succeeded = ctypes.windll.kernel32.GetProcessTimes(
                        handle,
                        ctypes.byref(creation),
                        ctypes.byref(exit_time),
                        ctypes.byref(kernel),
                        ctypes.byref(user),
                    )
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
                if succeeded:
                    ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                    return f"windows-created:{ticks}"
            except (AttributeError, OSError):
                return None
        return None

    @classmethod
    def _pid_alive(cls, value: Any, expected_identity: Any = None) -> bool:
        if not isinstance(value, int) or value <= 0:
            return False
        try:
            os.kill(value, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        if isinstance(expected_identity, str):
            actual_identity = cls._process_identity(value)
            if actual_identity is not None and actual_identity != expected_identity:
                return False
        return True

    def _read_global_lease(self) -> dict[str, Any] | None:
        if not self._lease_path.is_file():
            return None
        try:
            value = json.loads(self._lease_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"The host-global Aptus GPU lease is unreadable: {error}. "
                f"Inspect {self._lease_path} before launching another job."
            ) from error
        required = {"job_id", "state_root", "owner_pid", "created_at"}
        if not isinstance(value, dict) or not required.issubset(value):
            raise ValueError(
                f"The host-global Aptus GPU lease is invalid. Inspect {self._lease_path} before launching another job."
            )
        return value

    def _write_global_lease(self, value: dict[str, Any]) -> None:
        _atomic_write_json(self._lease_path, value)

    def _lease_record_state(self, lease: dict[str, Any]) -> str | None:
        state_root = lease.get("state_root")
        job_id = lease.get("job_id")
        if not isinstance(state_root, str) or not isinstance(job_id, str):
            raise ValueError(
                "The host-global Aptus GPU lease has invalid record fields."
            )
        record_path = Path(state_root) / f"{job_id}.json"
        if not record_path.is_file():
            return None
        try:
            value = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"The job record bound by the host-global GPU lease is unreadable: {record_path}: {error}"
            ) from error
        if not isinstance(value, dict) or value.get("id") != job_id:
            raise ValueError(
                f"The job record bound by the host-global GPU lease is invalid: {record_path}."
            )
        state = value.get("state")
        return state if isinstance(state, str) else None

    def _require_global_lease_available(self) -> None:
        lease = self._read_global_lease()
        if lease is None:
            return
        owner_live = self._pid_alive(
            lease.get("owner_pid"), lease.get("owner_process_identity")
        )
        child_live = self._pid_alive(
            lease.get("process_pid"), lease.get("process_identity")
        )
        process_group_id = lease.get("process_group_id")
        if os.name == "posix" and isinstance(process_group_id, int):
            child_live = child_live or self._process_group_alive(process_group_id)
        record_state = self._lease_record_state(lease)
        terminal_record = record_state in {
            RunState.COMPLETED.value,
            RunState.FAILED.value,
            RunState.CANCELLED.value,
        }
        if (
            (terminal_record and not child_live)
            or (record_state is None and not child_live)
            or not (owner_live or child_live)
        ):
            self._lease_path.unlink(missing_ok=True)
            return
        raise ActiveJobError(
            "Aptus already has active job "
            f"{lease.get('job_id')} from state root {lease.get('state_root')}. "
            "V0.2 runs one local GPU job at a time across all state roots for this user and host."
        )

    def _create_global_lease(self, record: dict[str, Any]) -> None:
        self._write_global_lease(
            {
                "schema_version": "aptus.gpu-lease.v1",
                "job_id": record["id"],
                "lease_token": record["id"],
                "state_root": str(self.root),
                "bundle_dir": record["bundle_dir"],
                "action": record["action"],
                "owner_pid": record["owner_pid"],
                "owner_process_identity": record["owner_process_identity"],
                "process_pid": None,
                "process_identity": None,
                "process_group_id": None,
                "created_at": record["created_at"],
            }
        )

    def _bind_global_lease_to_process(
        self, job_id: str, process_pid: int, process_identity: str | None
    ) -> None:
        lease = self._read_global_lease()
        if (
            lease is None
            or lease.get("job_id") != job_id
            or lease.get("state_root") != str(self.root)
        ):
            raise RuntimeError(
                "The host-global GPU lease changed before child registration."
            )
        lease.update(
            process_pid=process_pid,
            process_identity=process_identity,
            process_group_id=process_pid if os.name == "posix" else None,
            started_at=_now(),
        )
        self._write_global_lease(lease)

    def _clear_global_lease(self, job_id: str) -> None:
        lease = self._read_global_lease()
        if lease is None:
            return
        if lease.get("job_id") == job_id and lease.get("state_root") == str(self.root):
            self._lease_path.unlink(missing_ok=True)

    def _reconcile_external_record(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._records_lock():
            job_id = record.get("id")
            if not isinstance(job_id, str):
                raise ValueError("Aptus job record is missing its immutable ID.")
            record = self._read(job_id)
            active_states = {
                RunState.QUEUED.value,
                RunState.RUNNING.value,
                RunState.CANCELLING.value,
            }
            if record.get("state") not in active_states:
                return record
            if job_id in self._processes or job_id in self._threads:
                return record
            owner_pid = record.get("owner_pid")
            if self._pid_alive(owner_pid, record.get("owner_process_identity")):
                return record
            process_pid = record.get("process_pid")
            child_live = self._pid_alive(process_pid, record.get("process_identity"))
            process_group_id = record.get("process_group_id")
            if os.name == "posix" and isinstance(process_group_id, int):
                child_live = child_live or self._process_group_alive(process_group_id)
            if child_live:
                message = (
                    "The owning Aptus process is unavailable, but the persisted child PID is still live. "
                    "The job remains active and blocks new submissions; inspect the log and terminate it through the original service or operating system."
                )
                if record.get("error") != message:
                    record["error"] = message
                    self._write(record)
                return record
            verified_evidence = record.get("verified_pending_evidence")
            if (
                record.get("action") == "train"
                and record.get("return_code") == 0
                and isinstance(verified_evidence, dict)
            ):
                try:
                    attestation = _promote_train_attestation(record, verified_evidence)
                except (KeyError, OSError, TypeError, ValueError) as error:
                    record.update(
                        state=RunState.FAILED.value,
                        finished_at=_now(),
                        error=(
                            "Aptus recovered verified pending evidence, but terminal promotion failed: "
                            f"{error}"
                        ),
                    )
                else:
                    record.update(
                        state=RunState.COMPLETED.value,
                        finished_at=_now(),
                        error=None,
                        completion_attestation=attestation,
                        artifact_integrity_status="verified-at-completion",
                        artifact_verified_at=_now(),
                    )
                self._write(record)
                return record
            record.update(
                state=RunState.FAILED.value,
                finished_at=_now(),
                error="The owning Aptus process and persisted child PID are no longer live; the exit code is unavailable.",
            )
            self._write(record)
            return record

    def _recover_interrupted_jobs(self) -> None:
        with self._records_lock():
            for path in self._record_paths():
                self._reconcile_external_record(self._read(path.stem))

    def _require_no_active_job(self) -> None:
        with self._records_lock():
            for path in self._record_paths():
                record = self._reconcile_external_record(self._read(path.stem))
                if record.get("state") in {
                    RunState.QUEUED.value,
                    RunState.RUNNING.value,
                    RunState.CANCELLING.value,
                }:
                    raise ActiveJobError(
                        f"Aptus already has active job {record.get('id', path.stem)}. V0.2 runs one local GPU job at a time."
                    )

    @contextmanager
    def validation_guard(self) -> Any:
        """Serialize report mutation against submissions across local state roots."""

        with self._lock, self._global_lease_lock(), self._records_lock():
            self._require_global_lease_available()
            self._require_no_active_job()
            yield

    def _command(
        self,
        bundle: Path,
        action: JobAction,
        *,
        resume_from: str | None,
        run_id: str | None = None,
    ) -> list[str]:
        plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
        if action in {"dependency", "model-data", "preflight"}:
            level = {
                "dependency": "dependency",
                "model-data": "model-data",
                "preflight": "measured-preflight",
            }[action]
            return [
                sys.executable,
                str(bundle / "validate.py"),
                "--level",
                level,
            ]
        if action == "pilot":
            return [sys.executable, str(bundle / "validate.py"), "--level", "pilot"]
        if action != "train":
            raise ValueError(f"Unsupported job action: {action}")
        if resume_from is not None:
            raise ValueError("Full-training resume is unsupported in Aptus v0.2.")
        train_arguments = [str(bundle / "train.py"), "--confirm-full-train"]
        if run_id is None:
            raise ValueError("Full training requires an immutable Aptus run ID.")
        train_arguments.extend(("--output-dir", str(bundle / "runs" / run_id)))
        if plan["recommended"]["distribution"] == "single":
            return [sys.executable, *train_arguments]
        return [
            sys.executable,
            "-m",
            "accelerate.commands.accelerate_cli",
            "launch",
            "--config_file",
            str(bundle / "config" / "accelerate.yaml"),
            *train_arguments,
        ]

    def _require_action_prerequisite(self, bundle: Path, action: JobAction) -> None:
        required_state = _JOB_PREREQUISITE_STATES[action]
        report_path = bundle / "validation-report.json"
        if not report_path.is_file():
            raise JobPrerequisiteError(
                action=action,
                required_state=required_state,
                current_state=None,
                reason="missing_report",
            )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise JobPrerequisiteError(
                action=action,
                required_state=required_state,
                current_state=None,
                reason="unreadable_report",
            ) from error
        if not isinstance(report, dict):
            raise JobPrerequisiteError(
                action=action,
                required_state=required_state,
                current_state=None,
                reason="invalid_report",
            )
        current_state = report.get("state")
        if not isinstance(current_state, str) or current_state not in (
            _VALIDATION_STATE_RANK
        ):
            raise JobPrerequisiteError(
                action=action,
                required_state=required_state,
                current_state=current_state if isinstance(current_state, str) else None,
                reason="invalid_state",
            )
        if (
            _VALIDATION_STATE_RANK[current_state]
            < _VALIDATION_STATE_RANK[required_state]
        ):
            raise JobPrerequisiteError(
                action=action,
                required_state=required_state,
                current_state=current_state,
                reason="insufficient_state",
            )

    def _require_current_pilot(self, bundle: Path) -> dict[str, Any]:
        manifest_errors = validate_bundle_manifest(bundle)
        if manifest_errors:
            raise ValueError(
                "Bundle changed after compilation: " + " | ".join(manifest_errors)
            )
        report_path = bundle / "validation-report.json"
        if not report_path.is_file():
            raise ValueError("Full training requires a passing pilot for this bundle.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("Pilot validation report must be a JSON object.")
        if report.get("state") not in {
            "pilot-pass",
            "execution-approved",
            "measured-run-pass",
        }:
            raise ValueError(
                "Full training requires pilot-pass before execution approval."
            )
        plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ValueError("Bundle plan must be a JSON object.")
        plan_errors = validate_plan_payload(plan, root=bundle, verify_dataset=True)
        if plan_errors:
            raise ValueError("Bundle plan is invalid: " + " | ".join(plan_errors))
        bindings = report.get("bindings", {})
        if not isinstance(bindings, dict):
            raise ValueError("Pilot validation bindings must be a JSON object.")
        pilot_metrics = bundle / "pilot-output" / "metrics.json"
        if not pilot_metrics.is_file():
            raise ValueError("Pilot attestation is missing its bound metrics artifact.")
        world_size = int(plan["recommended"].get("world_size", 1))
        device_indices = plan["recommended"].get(
            "device_indices", list(range(world_size))
        )
        if not isinstance(device_indices, list):
            raise ValueError("Selected CUDA device indices must be a list.")
        runtime = _actual_runtime_snapshot(world_size, device_indices)
        expected = {
            "bundle": sha256_file(bundle / "bundle-manifest.json"),
            "dataset": plan["dataset"]["source_sha256"],
            "model_revision": plan["model"]["revision"],
            "plan_id": plan["plan_id"],
            "candidate_id": plan["recommended"]["candidate_id"],
            "environment": _environment_binding(bundle),
            "hardware": _json_hash(runtime["hardware"]),
            "pilot_metrics": sha256_file(pilot_metrics),
        }
        stale = [
            name for name, value in expected.items() if bindings.get(name) != value
        ]
        if stale:
            raise ValueError("Pilot attestation is stale for: " + ", ".join(stale))
        metrics = json.loads(pilot_metrics.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError("Pilot metrics must be a JSON object.")
        measured_checkpoint_bytes, measured_final_export_bytes = (
            _verify_pilot_artifacts(bundle, metrics)
        )
        phases = []
        for phase_name in ("phase_one", "phase_two_resumed"):
            phase = metrics.get(phase_name)
            if not isinstance(phase, dict):
                raise ValueError(f"Pilot metrics require object {phase_name}.")
            phases.append(phase)
        peaks: list[int] = []
        for phase_name, phase in zip(
            ("phase_one", "phase_two_resumed"), phases, strict=True
        ):
            per_rank = phase.get("per_rank_cuda_peaks")
            if world_size > 1 and (
                not isinstance(per_rank, list) or len(per_rank) != world_size
            ):
                raise ValueError(
                    f"Pilot metrics {phase_name} must bind one CUDA peak record per distributed rank."
                )
            values = per_rank if isinstance(per_rank, list) and per_rank else [phase]
            for rank_index, value in enumerate(values):
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Pilot metric {phase_name} rank {rank_index} must be an object."
                    )
                reserved = value.get("measured_reserved_cuda_bytes", 0)
                allocated = value.get("measured_peak_cuda_bytes", 0)
                for metric_name, metric_value in (
                    ("measured_reserved_cuda_bytes", reserved),
                    ("measured_peak_cuda_bytes", allocated),
                ):
                    if (
                        not isinstance(metric_value, int)
                        or isinstance(metric_value, bool)
                        or metric_value < 0
                    ):
                        raise ValueError(
                            f"Pilot metric {phase_name}.{metric_name} must be a non-negative integer."
                        )
                if reserved < allocated:
                    raise ValueError(
                        f"Pilot metric {phase_name} reserved CUDA memory is below allocated memory."
                    )
                peaks.append(max(reserved, allocated))
        measured_peak = max(peaks)
        if measured_peak <= 0:
            raise ValueError(
                "Pilot attestation does not contain a measured CUDA peak for pre-launch capacity checking."
            )
        reserve = int(plan.get("hardware", {}).get("reserve_per_device_bytes", 0))
        free_by_device = tuple(runtime["free_cuda_bytes"])
        required_free = measured_peak + reserve
        insufficient = [
            index for index, free in enumerate(free_by_device) if free < required_free
        ]
        if insufficient:
            raise ValueError(
                "Current free VRAM is below the pilot peak plus user reserve on CUDA device(s): "
                + ", ".join(str(item) for item in insufficient)
            )
        required_host_ram = int(plan["recommended"].get("required_host_ram_bytes", 0))
        host_ram_free = int(runtime["host_ram_free_bytes"])
        if required_host_ram <= 0 or host_ram_free < required_host_ram:
            raise ValueError(
                "Current free host RAM is below the plan-bound distributed loading requirement."
            )
        checkpoint_bytes = measured_checkpoint_bytes * 4
        final_export_bytes = measured_final_export_bytes
        required_output_disk = checkpoint_bytes + final_export_bytes
        disk_free = shutil.disk_usage(bundle).free
        if required_output_disk > 0 and disk_free < required_output_disk:
            raise ValueError(
                "Current free disk is below the measured four-checkpoint transient and final-export requirement."
            )
        return {
            "checked_at": _now(),
            "measured_peak_cuda_bytes": measured_peak,
            "required_free_cuda_bytes": required_free,
            "free_cuda_bytes": list(free_by_device),
            "required_host_ram_bytes": required_host_ram,
            "host_ram_free_bytes": host_ram_free,
            "required_checkpoint_disk_bytes": checkpoint_bytes,
            "required_final_export_disk_bytes": final_export_bytes,
            "required_training_output_disk_bytes": required_output_disk,
            "free_disk_bytes": disk_free,
            "checkpoint_basis": "4 * maximum measured pilot checkpoint bytes",
            "final_export_basis": "maximum measured pilot final export bytes",
        }

    def pilot_authorization(self, bundle_dir: Path) -> dict[str, Any]:
        try:
            with self._global_lease_lock():
                self._require_global_lease_available()
                capacity = self._require_current_pilot(bundle_dir.resolve(strict=True))
        except ActiveJobError as error:
            return {
                "current": False,
                "error": (
                    "Pilot authorization is not re-probed while another Aptus GPU job is active: "
                    + str(error)
                ),
                "capacity": None,
            }
        except (
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            return {"current": False, "error": str(error), "capacity": None}
        return {"current": True, "error": None, "capacity": capacity}

    def submit(
        self,
        bundle_dir: Path,
        *,
        action: JobAction = "preflight",
        confirm_full_train: bool = False,
        resume_from: str | None = None,
    ) -> dict[str, Any]:
        if action not in JOB_ACTIONS:
            raise ValueError(f"Unsupported job action: {action}")
        if action != "train" and confirm_full_train:
            raise ValueError("confirm_full_train is valid only for the train action.")
        if action != "train" and resume_from is not None:
            raise ValueError("resume_from is valid only for the train action.")
        if action == "train" and resume_from is not None:
            raise ValueError(
                "Full-training resume is fail-closed in Aptus v0.2 until a checkpoint manifest binds complete optimizer, scheduler, RNG, model, environment, and plan state."
            )
        bundle = bundle_dir.resolve(strict=True)
        if not bundle.is_dir() or not (bundle / "plan.json").is_file():
            raise ValueError(f"Not an Aptus bundle: {bundle}")
        manifest_errors = validate_bundle_manifest(bundle)
        if manifest_errors:
            raise ValueError(
                "Bundle integrity check failed: " + " | ".join(manifest_errors)
            )
        if action == "train":
            if not confirm_full_train:
                raise ValueError("Full training requires confirm_full_train=true.")
        job_id = "job_" + uuid.uuid4().hex
        run_id = f"run_{job_id[4:]}" if action == "train" else None
        command = self._command(bundle, action, resume_from=resume_from, run_id=run_id)
        record: dict[str, Any] = {
            "id": job_id,
            "job_id": job_id,
            "state": RunState.QUEUED.value,
            "action": action,
            "bundle_dir": str(bundle),
            "command": command,
            "log": str(self._log_path(job_id)),
            "return_code": None,
            "resume_from": resume_from,
            "run_id": run_id,
            "run_output_dir": str(bundle / "runs" / run_id) if run_id else None,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "owner_pid": os.getpid(),
            "owner_process_identity": self._process_identity(os.getpid()),
            "process_pid": None,
            "process_identity": None,
            "process_group_id": None,
            "prelaunch_capacity_check": None,
        }
        worker = threading.Thread(
            target=self._run, args=(job_id,), name=f"aptus-{job_id}", daemon=True
        )
        with self._lock:
            with self._global_lease_lock(), self._records_lock():
                self._require_global_lease_available()
                self._require_no_active_job()
                with _bundle_report_lock(bundle):
                    self._require_action_prerequisite(bundle, action)
                    if action == "train":
                        try:
                            record["prelaunch_capacity_check"] = (
                                self._require_current_pilot(bundle)
                            )
                        except RuntimeError as error:
                            raise ValueError(
                                f"Could not inspect the current CUDA runtime: {error}"
                            ) from error
                    self._write(record)
                    try:
                        self._create_global_lease(record)
                    except Exception:
                        record.update(
                            state=RunState.FAILED.value,
                            finished_at=_now(),
                            error="The host-global execution lease could not be persisted.",
                        )
                        self._write(record)
                        raise
            try:
                self._threads[job_id] = worker
                worker.start()
            except Exception as error:
                with self._global_lease_lock(), self._records_lock():
                    self._threads.pop(job_id, None)
                    current = self._read(job_id)
                    current.update(
                        state=RunState.FAILED.value,
                        finished_at=_now(),
                        error=f"The Aptus job worker could not start: {error}",
                    )
                    self._write(current)
                    self._clear_global_lease(job_id)
                raise
        return self.get(job_id)

    def _run(self, job_id: str) -> None:
        log_path = self._log_path(job_id)
        launch_spec = self.root / f".{job_id}.launch-spec"
        launch_permit = self.root / f".{job_id}.permit"
        process: subprocess.Popen[str] | None = None
        try:
            with self._lock, self._global_lease_lock(), self._records_lock():
                record = self._read(job_id)
                if record["state"] in {
                    RunState.CANCELLED.value,
                    RunState.CANCELLING.value,
                }:
                    record.update(
                        state=RunState.CANCELLED.value,
                        finished_at=_now(),
                    )
                    self._write(record)
                    self._clear_global_lease(job_id)
                    return
            _atomic_write_json(launch_spec, {"command": record["command"]})
            launch_permit.unlink(missing_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                process_options: dict[str, Any] = {}
                if os.name == "posix":
                    process_options["start_new_session"] = True
                elif os.name == "nt" and hasattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP"
                ):
                    process_options["creationflags"] = (
                        subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _PERMIT_LAUNCHER,
                        str(launch_spec),
                        str(launch_permit),
                    ],
                    cwd=record["bundle_dir"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "APTUS_GPU_LEASE_TOKEN": record["id"],
                    },
                    **process_options,
                )
                with self._lock, self._global_lease_lock(), self._records_lock():
                    self._processes[job_id] = process
                    current = self._read(job_id)
                    cancelled_before_registration = current["state"] in {
                        RunState.CANCELLED.value,
                        RunState.CANCELLING.value,
                    }
                    process_identity = self._process_identity(process.pid)
                    current.update(
                        process_pid=process.pid,
                        process_identity=process_identity,
                        process_group_id=process.pid if os.name == "posix" else None,
                        launch_protocol="permit-file-v1",
                    )
                    if not cancelled_before_registration:
                        current.update(state=RunState.RUNNING.value, started_at=_now())
                    self._write(current)
                    self._bind_global_lease_to_process(
                        job_id, process.pid, process_identity
                    )
                    if not cancelled_before_registration:
                        launch_permit.write_text("go\n", encoding="utf-8")
                if cancelled_before_registration and process.poll() is None:
                    self._terminate_process(process)
                return_code = process.wait()
                if self._process_tree_alive(process):
                    raise RuntimeError(
                        "The launcher exited while a descendant process remained in its execution tree."
                    )
            with self._lock, self._global_lease_lock(), self._records_lock():
                current = self._read(job_id)
                if current.get("state") not in {
                    RunState.CANCELLED.value,
                    RunState.CANCELLING.value,
                }:
                    current.update(
                        return_code=return_code,
                        completion_verification_started_at=_now(),
                    )
                    self._write(current)
            completion_error: str | None = None
            completion_attestation: dict[str, Any] | None = None
            if return_code == 0 and record.get("action") == "train":
                try:
                    pending_evidence = _verify_train_artifacts(record)
                    with self._lock, self._global_lease_lock(), self._records_lock():
                        verified_record = self._read(job_id)
                        verified_record["verified_pending_evidence"] = pending_evidence
                        verified_record["pending_evidence_verified_at"] = _now()
                        self._write(verified_record)
                    completion_attestation = _promote_train_attestation(
                        record, pending_evidence
                    )
                except (KeyError, OSError, TypeError, ValueError) as error:
                    completion_error = str(error)
            with self._lock, self._global_lease_lock(), self._records_lock():
                current = self._read(job_id)
                cancelled = current["state"] in {
                    RunState.CANCELLED.value,
                    RunState.CANCELLING.value,
                }
                if current["state"] not in {
                    RunState.COMPLETED.value,
                    RunState.FAILED.value,
                    RunState.CANCELLED.value,
                }:
                    current.update(
                        state=(
                            RunState.CANCELLED.value
                            if cancelled
                            else RunState.COMPLETED.value
                            if return_code == 0 and completion_error is None
                            else RunState.FAILED.value
                        ),
                        return_code=return_code,
                        finished_at=_now(),
                        error=(
                            None
                            if cancelled
                            or (return_code == 0 and completion_error is None)
                            else (
                                "Training process exited successfully, but completion verification failed: "
                                + completion_error
                                if return_code == 0 and completion_error is not None
                                else f"Process exited with code {return_code}."
                            )
                        ),
                    )
                    if completion_attestation is not None:
                        current["completion_attestation"] = completion_attestation
                        current["artifact_integrity_status"] = "verified-at-completion"
                        current["artifact_verified_at"] = _now()
                    self._write(current)
                self._clear_global_lease(job_id)
        except Exception as error:
            termination_error: Exception | None = None
            if process is not None and self._process_tree_alive(process):
                try:
                    self._terminate_process(process)
                except Exception as stop_error:  # pragma: no cover - OS failure path.
                    termination_error = stop_error
            with self._lock, self._global_lease_lock(), self._records_lock():
                try:
                    current = self._read(job_id)
                except (KeyError, OSError, ValueError):
                    current = None
                if current is not None and current.get("state") not in {
                    RunState.COMPLETED.value,
                    RunState.FAILED.value,
                    RunState.CANCELLED.value,
                }:
                    if termination_error is None:
                        current.update(
                            state=RunState.FAILED.value,
                            finished_at=_now(),
                            error=f"Job worker failed before completion: {error}",
                        )
                        self._clear_global_lease(job_id)
                    else:
                        current.update(
                            state=RunState.CANCELLING.value,
                            error=(
                                f"Job worker failed and the process tree could not be confirmed stopped: {error}; "
                                f"termination error: {termination_error}"
                            ),
                        )
                    self._write(current)
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
                self._threads.pop(job_id, None)
            launch_permit.unlink(missing_ok=True)
            launch_spec.unlink(missing_ok=True)

    def get(
        self, job_id: str, *, include_validation_report: bool = True
    ) -> dict[str, Any]:
        with self._lock, self._records_lock():
            record = self._reconcile_external_record(self._read(job_id))
            active = record.get("state") in {
                RunState.QUEUED.value,
                RunState.RUNNING.value,
                RunState.CANCELLING.value,
            }
            owned_here = job_id in self._processes or job_id in self._threads
            verifying = bool(
                active
                and record.get("completion_verification_started_at")
                and record.get("return_code") is not None
            )
            record["phase"] = "verifying" if verifying else record.get("state")
            record["cancellable"] = active and owned_here and not verifying
            if (
                not active
                and record.get("action") == "train"
                and record.get("state") == RunState.COMPLETED.value
            ):
                run_output = record.get("run_output_dir")
                required_paths = (
                    (
                        Path(run_output) / ".aptus-run.json",
                        Path(run_output) / "final-export.json",
                        Path(run_output) / "metrics.json",
                        Path(run_output) / "final",
                    )
                    if isinstance(run_output, str)
                    else ()
                )
                missing = [str(path) for path in required_paths if not path.exists()]
                record["artifact_integrity"] = {
                    "status": (
                        "missing-since-completion"
                        if missing
                        else "verified-at-completion-not-rehashed"
                    ),
                    "verified_at": record.get("artifact_verified_at"),
                    "missing_paths": missing,
                    "note": (
                        "Polling performs only a presence check. The completion-time file tree was deeply verified; submit an explicit verification workflow before treating later copies as current."
                    ),
                }
            if not active:
                record["owner_status"] = "terminal"
                record["cancellation_note"] = None
            elif verifying and owned_here:
                record["owner_status"] = "owning-service"
                record["cancellation_note"] = (
                    "The child process has exited. Aptus is verifying and promoting completion evidence; cancellation is no longer available."
                )
            elif owned_here:
                record["owner_status"] = "owning-service"
                record["cancellation_note"] = (
                    "This live Aptus service owns the worker and can cancel it."
                )
            elif self._pid_alive(
                record.get("owner_pid"), record.get("owner_process_identity")
            ):
                record["owner_status"] = "external-service"
                record["cancellation_note"] = (
                    "Another live Aptus process owns this job. Cancel it through that process."
                )
            elif self._pid_alive(
                record.get("process_pid"), record.get("process_identity")
            ):
                record["owner_status"] = "orphan-child"
                record["cancellation_note"] = (
                    "The Aptus owner exited while the child process remained live. Inspect the persisted PID and terminate it through the operating system."
                )
            else:
                record["owner_status"] = "unavailable"
                record["cancellation_note"] = (
                    "No live owner is attached to this active record. Refresh to reconcile its state."
                )
        log_path = self._log_path(job_id)
        if log_path.is_file():
            with log_path.open("rb") as log:
                log.seek(0, os.SEEK_END)
                length = log.tell()
                log.seek(max(0, length - 16_000))
                record["log_tail"] = log.read().decode("utf-8", errors="replace")
        else:
            record["log_tail"] = ""
        if not include_validation_report:
            return record
        bundle_dir = record.get("bundle_dir")
        report_path = (
            Path(bundle_dir) / "validation-report.json"
            if isinstance(bundle_dir, str)
            else None
        )
        if report_path is not None and report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                record["validation_report_error"] = (
                    f"Could not read the current validation report: {error}"
                )
            else:
                if isinstance(report, dict):
                    if report.get("state") in {
                        "pilot-pass",
                        "execution-approved",
                        "measured-run-pass",
                    }:
                        if active:
                            cached_capacity = record.get("prelaunch_capacity_check")
                            authorization = {
                                "current": bool(
                                    record.get("action") == "train"
                                    and isinstance(cached_capacity, dict)
                                ),
                                "error": (
                                    None
                                    if record.get("action") == "train"
                                    and isinstance(cached_capacity, dict)
                                    else "Pilot authorization is not re-probed while any Aptus GPU job is active."
                                ),
                                "capacity": (
                                    cached_capacity
                                    if isinstance(cached_capacity, dict)
                                    else None
                                ),
                            }
                        else:
                            authorization = {
                                "current": False,
                                "error": (
                                    "Deep pilot binding, checkpoint, environment, and current capacity authorization is performed atomically when full training is submitted. Polling does not rehash large pilot artifacts."
                                ),
                                "capacity": None,
                            }
                        report = {
                            **report,
                            "authorization_current": authorization["current"],
                            "authorization_error": authorization["error"],
                            "prelaunch_capacity_check": authorization["capacity"],
                        }
                    record["validation_report"] = report
                else:
                    record["validation_report_error"] = (
                        "The current validation report is not a JSON object."
                    )
        elif report_path is not None:
            record["validation_report_error"] = (
                "The current bundle validation report is missing. Revalidate the bundle before authorizing another action."
            )
        return record

    def list(self) -> list[dict[str, Any]]:
        records = [
            self.get(path.stem, include_validation_report=False)
            for path in self._record_paths()
        ]
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    @staticmethod
    def _process_group_alive(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @classmethod
    def _process_tree_alive(cls, process: subprocess.Popen[str]) -> bool:
        if os.name == "posix":
            return cls._process_group_alive(process.pid)
        return process.poll() is None

    @classmethod
    def _terminate_process(cls, process: subprocess.Popen[str]) -> None:
        if os.name == "posix":
            if not cls._process_group_alive(process.pid):
                return
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + 5
            while cls._process_group_alive(process.pid) and time.monotonic() < deadline:
                process.poll()
                time.sleep(0.05)
            if cls._process_group_alive(process.pid):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 2
                while (
                    cls._process_group_alive(process.pid)
                    and time.monotonic() < deadline
                ):
                    process.poll()
                    time.sleep(0.05)
            if cls._process_group_alive(process.pid):
                raise RuntimeError(
                    f"Process group {process.pid} remained live after SIGKILL."
                )
            if process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(
                        f"Process group leader {process.pid} did not become waitable."
                    ) from error
            return

        if process.poll() is not None:
            return
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        if process.poll() is None or (
            completed is not None and completed.returncode != 0
        ):
            raise RuntimeError(
                f"Windows process tree {process.pid} could not be confirmed stopped."
            )

    def cancel(self, job_id: str) -> dict[str, Any]:
        process: subprocess.Popen[str] | None = None
        worker: threading.Thread | None = None
        termination_error: Exception | None = None
        with self._lock, self._global_lease_lock(), self._records_lock():
            record = self._read(job_id)
            if record["state"] in {
                RunState.COMPLETED.value,
                RunState.FAILED.value,
                RunState.CANCELLED.value,
            }:
                pass
            else:
                process = self._processes.get(job_id)
                worker = self._threads.get(job_id)
                if process is None and worker is None:
                    raise ValueError(
                        "This JobService does not own the active process. Cancel it through the owning Aptus service; the record was not changed."
                    )
                process_tree_live = (
                    self._process_tree_alive(process) if process is not None else False
                )
                return_code = process.poll() if process is not None else None
                if process is not None and not process_tree_live:
                    if worker is None or not worker.is_alive():
                        record.update(
                            state=RunState.FAILED.value,
                            return_code=return_code,
                            finished_at=_now(),
                            error=(
                                "The process exited, but its owning verifier is unavailable. "
                                "Aptus will not infer successful completion."
                            ),
                        )
                        self._write(record)
                        self._clear_global_lease(job_id)
                else:
                    record.update(
                        state=RunState.CANCELLING.value,
                        cancel_requested_at=_now(),
                        error=None,
                    )
                    self._write(record)
                    if process is not None:
                        try:
                            self._terminate_process(process)
                        except (
                            Exception
                        ) as error:  # pragma: no cover - OS failure path.
                            termination_error = error
                            record = self._read(job_id)
                            record["error"] = (
                                "Cancellation was requested, but the process tree could not be confirmed stopped: "
                                f"{error}"
                            )
                            self._write(record)
                        else:
                            record = self._read(job_id)
                            record.update(
                                state=RunState.CANCELLED.value,
                                return_code=process.poll(),
                                finished_at=_now(),
                                error=None,
                            )
                            self._write(record)
                            self._clear_global_lease(job_id)
        if termination_error is not None:
            raise ValueError(str(termination_error)) from termination_error
        if (
            worker is not None
            and worker is not threading.current_thread()
            and worker.ident is not None
        ):
            worker.join(timeout=6)
        return self.get(job_id)
