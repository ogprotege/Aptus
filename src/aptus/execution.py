from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import stat
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
from typing import Any, Callable, Literal, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses the in-process lock.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX uses fcntl.
    msvcrt = None

from .attestation import require_trainable_parameter_census
from .domain import RunState, ValidationState
from .local_store import atomic_write_json, private_directory, quarantine_file
from .plan_contract import (
    MODEL_POLICY_SNAPSHOT_PATH,
    StaleModelPolicyError,
    expected_model_architecture_contract,
    mlx_quantized_storage_bytes_for_contract,
    require_current_model_policy,
    require_current_model_policy_snapshot,
    sha256_file,
    validate_bundle_manifest,
    validate_plan_payload,
)
from .profiling import probe_apple_platform
from .runtime_env import resolve_runtime_interpreter, runtime_environment_key


JobAction = Literal["dependency", "model-data", "preflight", "pilot", "train"]
ValidationAuthorizationStatus = Literal["current", "deferred", "blocked"]
JOB_ACTIONS = {"dependency", "model-data", "preflight", "pilot", "train"}
JOB_RECORD_SCHEMA_VERSION = "aptus.job-record.v1"
PARENT_PROMOTION_SCHEMA_VERSION = "aptus.parent-promotion.v1"
_GLOBAL_LEASE_THREAD_LOCK = threading.RLock()
_CAMPAIGN_EXPERIMENT_RUN_ID = re.compile(r"^xrun_[0-9a-f]{32}$")
_CAMPAIGN_EVENT_JOURNAL_MAX_BYTES = 1024 * 1024
_CAMPAIGN_ENVIRONMENT_NAMES = (
    "APTUS_CUDA_CAMPAIGN_EVENT_SINK",
    "APTUS_CUDA_CAMPAIGN_EVENT_SINK_IDENTITY",
    "APTUS_CUDA_CAMPAIGN_EXPERIMENT_RUN_ID",
    "APTUS_CUDA_CAMPAIGN_JOB_ID",
)
_PROCESS_MONOTONIC_BINDING = "process-monotonic:" + uuid.uuid4().hex
_TERMINAL_JOB_STATES = {
    RunState.COMPLETED.value,
    RunState.FAILED.value,
    RunState.CANCELLED.value,
}


def _current_monotonic_clock_binding() -> str:
    """Identify the clock epoch that makes persisted monotonic values comparable."""

    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    if os.name == "posix" and boot_id_path.is_file() and not boot_id_path.is_symlink():
        try:
            boot_id = boot_id_path.read_text(encoding="ascii").strip().lower()
        except (OSError, UnicodeError):
            boot_id = ""
        normalized = boot_id.replace("-", "")
        if re.fullmatch(r"[0-9a-f]{32}", normalized) is not None:
            digest = hashlib.sha256(normalized.encode("ascii")).hexdigest()
            return "linux-boot-sha256:" + digest
    return _PROCESS_MONOTONIC_BINDING


def decorate_validation_authorization(
    report: Mapping[str, Any],
    *,
    status: ValidationAuthorizationStatus,
    error: str | None,
    capacity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if status not in {"current", "deferred", "blocked"}:
        raise ValueError("Validation authorization status is invalid.")
    current = status == "current"
    if current and error is not None:
        raise ValueError("Current validation authorization cannot carry an error.")
    if not current and (
        not isinstance(error, str) or not error.strip() or error != error.strip()
    ):
        raise ValueError(
            "Deferred or blocked validation authorization requires a diagnostic."
        )
    return {
        **report,
        "authorization_status": status,
        "authorization_current": current,
        "authorization_error": error,
        "prelaunch_capacity_check": (
            dict(capacity) if isinstance(capacity, Mapping) else None
        ),
    }


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


class JobSubmissionFailure(RuntimeError):
    """A job record was persisted but submission could not safely complete."""

    def __init__(
        self, job_id: str, terminal_record: Mapping[str, Any], failure_code: str
    ) -> None:
        self.job_id = job_id
        self.terminal_record = dict(terminal_record)
        self.failure_code = failure_code
        super().__init__(
            f"Job submission failed after persistence ({failure_code}); "
            "inspect the terminal job record."
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
import hashlib
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
spec = json.loads(spec_path.read_text(encoding="utf-8"))
manifest_path = Path("bundle-manifest.json")
if manifest_path.is_symlink() or not manifest_path.is_file():
    raise SystemExit("Aptus launch blocked: bundle manifest is unavailable")
manifest_bytes = manifest_path.read_bytes()
expected_fingerprint = spec.get("expected_artifact_fingerprint")
if expected_fingerprint is not None and (
    hashlib.sha256(manifest_bytes).hexdigest() != expected_fingerprint
):
    raise SystemExit("Aptus launch blocked: project artifact fingerprint changed")
manifest = json.loads(manifest_bytes)
root = Path.cwd().resolve()
for entry in manifest.get("files", []):
    relative = entry.get("path")
    if not isinstance(relative, str):
        raise SystemExit("Aptus launch blocked: invalid manifest path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SystemExit("Aptus launch blocked: unsafe manifest path")
    path = root / candidate
    cursor = root
    for part in candidate.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SystemExit("Aptus launch blocked: manifested symlink")
    if not path.is_file():
        raise SystemExit("Aptus launch blocked: manifested file is missing")
    payload = path.read_bytes()
    if (
        len(payload) != entry.get("size_bytes")
        or hashlib.sha256(payload).hexdigest() != entry.get("sha256")
    ):
        raise SystemExit("Aptus launch blocked: manifested file changed")
plan = json.loads(Path("plan.json").read_text(encoding="utf-8"))
expected_policy_snapshot = spec.get("authorized_model_policy_snapshot_sha256")
if (
    not isinstance(expected_policy_snapshot, str)
    or plan.get("model_policy_snapshot_sha256") != expected_policy_snapshot
):
    raise SystemExit("Aptus launch blocked: host policy authorization changed")
command = spec["command"]
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

_RUNTIME_ENVIRONMENT_PROBE = r"""
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

bundle = Path(sys.argv[1])
direct_constraints = {}
for line in (bundle / "requirements.txt").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    name = line.split("==", 1)[0]
    try:
        direct_constraints[name] = version(name)
    except PackageNotFoundError:
        direct_constraints[name] = "missing"
pending = list(direct_constraints)
observed = {}
visited = set()
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
print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "direct_constraints": direct_constraints,
    "runtime_distributions": dict(sorted(observed.items())),
}, sort_keys=True, separators=(",", ":")))
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any] | dict[str, Any]) -> None:
    atomic_write_json(path, value)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, RecursionError, ValueError) as error:
        raise ValueError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _read_json_object_bytes(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, RecursionError, ValueError) as error:
        raise ValueError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value, payload


def _require_current_bundle_model_policy(
    bundle: Path,
    *,
    expected_artifact_fingerprint: str | None = None,
    enforce_current_policy: bool = True,
) -> tuple[dict[str, Any], str]:
    manifest_path = bundle / "bundle-manifest.json"
    plan_path = bundle / "plan.json"
    snapshot_path = bundle / MODEL_POLICY_SNAPSHOT_PATH
    manifest, manifest_bytes = _read_json_object_bytes(manifest_path, "Bundle manifest")
    plan, plan_bytes = _read_json_object_bytes(plan_path, "Bundle plan")
    snapshot, snapshot_bytes = _read_json_object_bytes(
        snapshot_path, "Bundle model policy snapshot"
    )
    manifest_fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    plan_digest = hashlib.sha256(plan_bytes).hexdigest()
    if manifest.get("plan_sha256") != plan_digest:
        raise ValueError("Bundle manifest plan digest does not match plan.json.")
    if manifest.get("policy_snapshot_path") != MODEL_POLICY_SNAPSHOT_PATH:
        raise ValueError(
            "Bundle manifest does not bind the required model policy snapshot path."
        )
    if manifest.get("policy_snapshot_sha256") != plan.get(
        "model_policy_snapshot_sha256"
    ):
        raise ValueError(
            "Bundle manifest and plan disagree on the model policy snapshot digest."
        )
    if expected_artifact_fingerprint is not None and (
        len(expected_artifact_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_artifact_fingerprint
        )
        or manifest_fingerprint != expected_artifact_fingerprint
    ):
        raise ValueError(
            "The bundle manifest does not match the expected project artifact "
            "fingerprint; the bundle changed from its project-bound artifact."
        )
    manifest_errors = validate_bundle_manifest(bundle)
    if manifest_errors:
        raise ValueError(
            "Bundle integrity check failed: " + " | ".join(manifest_errors)
        )
    try:
        stable = (
            manifest_path.read_bytes() == manifest_bytes
            and plan_path.read_bytes() == plan_bytes
            and snapshot_path.read_bytes() == snapshot_bytes
        )
    except OSError as error:
        raise ValueError(
            f"Bundle changed while policy was being validated: {error}"
        ) from error
    if not stable:
        raise ValueError("Bundle changed while its model policy was being validated.")
    try:
        require_current_model_policy(plan, policy_snapshot=snapshot)
    except StaleModelPolicyError as error:
        raise ValueError(
            "Bundle plan is inconsistent with its embedded model policy snapshot: "
            f"{error}"
        ) from error
    except ValueError as error:
        raise ValueError(
            "Bundle plan is inconsistent with its embedded model policy snapshot: "
            f"{error}"
        ) from error
    if enforce_current_policy:
        require_current_model_policy_snapshot(
            plan,
            historical_policy_snapshot=snapshot,
        )
    try:
        stable = (
            manifest_path.read_bytes() == manifest_bytes
            and plan_path.read_bytes() == plan_bytes
            and snapshot_path.read_bytes() == snapshot_bytes
        )
    except OSError as error:
        raise ValueError(
            f"Bundle changed while policy was being validated: {error}"
        ) from error
    if not stable:
        raise ValueError("Bundle changed while its model policy was being validated.")
    return plan, manifest_fingerprint


def _verify_run_metrics(
    metrics: dict[str, Any], candidate: dict[str, Any], target: dict[str, Any]
) -> None:
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
    optimizer_steps = target.get("optimizer_steps")
    if optimizer_steps is not None and (
        metrics.get("optimizer_step_target") != optimizer_steps
        or global_step != optimizer_steps
        or metrics.get("non_skipped_optimizer_steps") != optimizer_steps
    ):
        raise ValueError(
            "Measured-run metrics do not bind the exact optimizer-step target."
        )
    for name in ("split_seed", "training_seed", "data_order_seed"):
        if metrics.get(name) != target.get(name):
            raise ValueError(f"Measured-run metrics do not bind {name}.")
    counters = metrics.get("training_counters")
    progress = metrics.get("optimizer_progress")
    if (
        not isinstance(counters, dict)
        or counters.get("schema_version") != "aptus.training-counters.v1"
        or not isinstance(progress, dict)
        or progress.get("schema_version") != "aptus.optimizer-progress.v1"
    ):
        raise ValueError("Measured-run metrics do not bind Phase 3 counters.")
    training_counters = counters.get("training")
    evaluation_counters = counters.get("evaluation")
    counter_names = (
        "micro_iterations",
        "completed_non_skipped_optimizer_steps",
        "examples_consumed",
        "padded_input_elements",
        "non_padding_tokens",
        "supervised_tokens",
    )
    if any(
        not isinstance(values, dict)
        or any(
            not isinstance(values.get(name), int)
            or isinstance(values.get(name), bool)
            or values[name] < 0
            for name in counter_names
        )
        for values in (training_counters, evaluation_counters)
    ):
        raise ValueError("Measured-run metrics contain invalid Phase 3 counters.")
    assert isinstance(training_counters, dict)
    if (
        training_counters["completed_non_skipped_optimizer_steps"] != global_step
        or training_counters["micro_iterations"] < global_step
        or any(
            training_counters[name] < 1
            for name in (
                "examples_consumed",
                "padded_input_elements",
                "non_padding_tokens",
                "supervised_tokens",
            )
        )
    ):
        raise ValueError(
            "Measured-run training counters do not bind completed optimizer steps."
        )
    timestamps = progress.get("timestamps_monotonic_ns")
    if (
        progress.get("completed_non_skipped_optimizer_steps") != global_step
        or not isinstance(timestamps, list)
        or len(timestamps) != global_step
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in timestamps
        )
        or timestamps != sorted(timestamps)
    ):
        raise ValueError("Measured-run optimizer progress timing is invalid.")
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
            with safe_open(
                str(weight_path), framework="numpy", device="cpu"
            ) as tensors:
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


def _verify_cuda_train_artifacts(record: dict[str, Any]) -> dict[str, Any]:
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
    _verify_run_metrics(metrics, candidate, plan["target"])
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


def _plan_training_runtime(plan: Mapping[str, Any]) -> str:
    candidate = plan.get("recommended")
    runtime = candidate.get("runtime_contract") if isinstance(candidate, dict) else None
    runtime_id = runtime.get("training_runtime") if isinstance(runtime, dict) else None
    if not isinstance(runtime_id, str) or not runtime_id:
        return "transformers-peft-cuda"
    return runtime_id


def _sha256_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_mlx_admission(
    plan: dict[str, Any], admission: object, *, label: str
) -> dict[str, Any]:
    candidate = plan["recommended"]
    memory = candidate["memory"]
    point = int(memory["point_estimate_bytes"])
    upper = int(memory["upper_estimate_bytes"])
    planned_resident = int(memory["base_weights_bytes"]) + int(
        memory["quantization_metadata_bytes"]
    )
    reserve = max(
        int(plan["hardware"].get("reserve_per_device_bytes", 0)),
        8 * 1024**3,
    )
    if not isinstance(admission, dict):
        raise ValueError(f"{label} must be an object.")
    observed = admission.get("observed_safetensors_bytes")
    if not isinstance(observed, int) or isinstance(observed, bool) or observed <= 0:
        raise ValueError(f"{label} requires positive safetensors bytes.")
    adjustment = max(0, observed - planned_resident)
    adjusted_point = point + adjustment
    adjusted_upper = upper + adjustment
    required = max(adjusted_point, adjusted_upper) + reserve
    expected = {
        "schema_version": "aptus.mlx-unified-memory-admission.v2",
        "planned_resident_bytes": planned_resident,
        "observed_safetensors_bytes": observed,
        "resident_adjustment_bytes": adjustment,
        "adjusted_point_estimate_bytes": adjusted_point,
        "adjusted_upper_estimate_bytes": adjusted_upper,
        "reserve_bytes": reserve,
        "required_available_bytes": required,
    }
    if any(admission.get(name) != value for name, value in expected.items()):
        raise ValueError(f"{label} does not bind the plan memory contract.")
    available = admission.get("available_unified_memory_bytes")
    if (
        set(admission) != set(expected) | {"available_unified_memory_bytes"}
        or not isinstance(available, int)
        or isinstance(available, bool)
        or available < required
        or "free_vram_bytes" in admission
    ):
        raise ValueError(
            f"{label} does not contain a passing unified-memory admission."
        )
    return admission


def _require_mlx_model_load_binding(
    plan: Mapping[str, Any], binding: object
) -> dict[str, Any]:
    """Verify the portable MLX model identity, topology, and parameter census."""

    expected_keys = {
        "schema_version",
        "model_id",
        "model_revision",
        "resolved_local_snapshot",
        "trust_remote_code",
        "architecture_contract",
        "parameter_census",
        "packed_checkpoint_binding",
        "descriptor_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise ValueError("MLX model-load binding has an invalid shape.")
    model = plan.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("MLX model-load binding requires a plan model contract.")
    expected_static = {
        "schema_version": "aptus.mlx-model-load-binding.v3",
        "model_id": model.get("model_id"),
        "model_revision": model.get("revision"),
        "resolved_local_snapshot": True,
        "trust_remote_code": False,
        "architecture_contract": expected_model_architecture_contract(model),
    }
    if any(binding.get(key) != value for key, value in expected_static.items()):
        raise ValueError("MLX model-load binding does not match the plan architecture.")
    if binding.get("descriptor_sha256") != _json_hash(
        {key: value for key, value in binding.items() if key != "descriptor_sha256"}
    ):
        raise ValueError("MLX model-load binding digest is invalid.")

    census = binding.get("parameter_census")
    census_keys = {
        "schema_version",
        "census_method",
        "declared_total_parameters",
        "observed_total_parameters",
        "total_parameter_delta",
        "total_parameter_tolerance",
        "declared_active_parameters",
        "observed_active_parameters",
        "sparse_layer_count",
        "routed_expert_parameters",
        "active_routed_expert_parameters",
        "inactive_expert_parameters",
        "descriptor_sha256",
    }
    if not isinstance(census, dict) or set(census) != census_keys:
        raise ValueError("MLX model parameter census has an invalid shape.")
    if census.get("descriptor_sha256") != _json_hash(
        {key: value for key, value in census.items() if key != "descriptor_sha256"}
    ):
        raise ValueError("MLX model parameter census digest is invalid.")
    declared_total = model.get("parameters")
    declared_active = model.get("active_parameters", declared_total)
    if (
        not isinstance(declared_total, int)
        or isinstance(declared_total, bool)
        or declared_total <= 0
        or not isinstance(declared_active, int)
        or isinstance(declared_active, bool)
        or declared_active <= 0
    ):
        raise ValueError("MLX plan parameter counts are invalid.")
    tolerance = max(1_000_000, round(declared_total * 0.02))
    observed_total = census.get("observed_total_parameters")
    observed_active = census.get("observed_active_parameters")
    moe = model.get("moe")
    sparse_layer_count = model.get("sparse_layer_count", 0)
    if moe is None:
        routed = active_routed = inactive = 0
        census_method = "mlx-lm.get_total_parameters.v1"
    elif isinstance(moe, Mapping) and isinstance(sparse_layer_count, int):
        routed = (
            sparse_layer_count
            * int(moe["expert_count"])
            * 3
            * int(model["hidden_size"])
            * int(moe["expert_intermediate_size"])
        )
        active_routed = (
            routed * int(moe["experts_per_token"]) // int(moe["expert_count"])
        )
        inactive = routed - active_routed
        census_method = "mlx-lm.get_total_parameters-plus-exact-qwen3-moe-routing.v1"
    else:
        raise ValueError("MLX plan MoE topology is invalid.")
    if (
        not isinstance(observed_total, int)
        or isinstance(observed_total, bool)
        or observed_total <= 0
        or not isinstance(observed_active, int)
        or isinstance(observed_active, bool)
        or observed_active <= 0
        or abs(observed_total - declared_total) > tolerance
        or abs(observed_active - declared_active) > tolerance
        or observed_active != observed_total - inactive
        or census.get("schema_version") != "aptus.mlx-model-parameter-census.v1"
        or census.get("census_method") != census_method
        or census.get("declared_total_parameters") != declared_total
        or census.get("total_parameter_delta") != observed_total - declared_total
        or census.get("total_parameter_tolerance") != tolerance
        or census.get("declared_active_parameters") != declared_active
        or census.get("sparse_layer_count") != sparse_layer_count
        or census.get("routed_expert_parameters") != routed
        or census.get("active_routed_expert_parameters") != active_routed
        or census.get("inactive_expert_parameters") != inactive
    ):
        raise ValueError("MLX model parameter census does not match the plan.")
    packed = binding.get("packed_checkpoint_binding")
    packed_fields = {
        "schema_version",
        "observed_safetensors_bytes",
        "observed_logical_parameters",
        "expected_weight_bytes",
        "expected_quantization_metadata_bytes",
        "expected_packed_tensor_bytes",
        "container_overhead_bytes",
        "container_overhead_limit_bytes",
        "descriptor_sha256",
    }
    if not isinstance(packed, dict) or set(packed) != packed_fields:
        raise ValueError("MLX packed-checkpoint binding has an invalid shape.")
    if packed.get("descriptor_sha256") != _json_hash(
        {key: value for key, value in packed.items() if key != "descriptor_sha256"}
    ):
        raise ValueError("MLX packed-checkpoint binding digest is invalid.")
    observed_safetensors = packed.get("observed_safetensors_bytes")
    if (
        not isinstance(observed_safetensors, int)
        or isinstance(observed_safetensors, bool)
        or observed_safetensors <= 0
    ):
        raise ValueError("MLX packed-checkpoint bytes must be positive.")
    if plan["recommended"].get("method") == "qlora":
        expected_weight_bytes, expected_metadata_bytes = (
            mlx_quantized_storage_bytes_for_contract(
                model, logical_parameters=observed_total
            )
        )
    else:
        expected_weight_bytes = round(observed_total * 2.0)
        expected_metadata_bytes = 0
    expected_packed_bytes = expected_weight_bytes + expected_metadata_bytes
    overhead = observed_safetensors - expected_packed_bytes
    overhead_limit = max(1024**2, round(expected_packed_bytes * 0.0001))
    expected_packed = {
        "schema_version": "aptus.mlx-packed-checkpoint.v1",
        "observed_safetensors_bytes": observed_safetensors,
        "observed_logical_parameters": observed_total,
        "expected_weight_bytes": expected_weight_bytes,
        "expected_quantization_metadata_bytes": expected_metadata_bytes,
        "expected_packed_tensor_bytes": expected_packed_bytes,
        "container_overhead_bytes": overhead,
        "container_overhead_limit_bytes": overhead_limit,
    }
    expected_packed["descriptor_sha256"] = _json_hash(expected_packed)
    if overhead < 0 or overhead > overhead_limit or packed != expected_packed:
        raise ValueError(
            "MLX packed-checkpoint binding does not match the logical parameter census."
        )
    return binding


def _verify_mlx_runtime_metrics(
    bundle: Path,
    plan: dict[str, Any],
    metrics: object,
    *,
    action: Literal["pilot", "full"],
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ValueError("MLX runtime metrics must be a JSON object.")
    training_fields = {
        "schema_version",
        "plan_id",
        "candidate_id",
        "model_revision",
        "dataset_sha256",
        "method",
        "training_runtime",
        "compute_backend",
        "compiler_id",
        "scope",
        "action",
        "execution_semantics",
        "resume_supported",
        "micro_iterations",
        "global_step",
        "gradient_accumulation_steps",
        "optimizer_update_opportunities",
        "completed_optimizer_updates",
        "train_examples",
        "validation_examples",
        "source_train_examples",
        "source_validation_examples",
        "max_epochs",
        "distribution",
        "actual_world_size",
        "measured_peak_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
        "memory_metric_backend",
        "model_load_binding",
        "unified_memory_admission",
        "finite_train_loss",
        "train_loss_observations",
        "finite_validation_loss",
        "validation_loss_observations",
        "optimizer_update_observed",
        "trainable_target_binding",
        "adapter_delta_l1",
        "changed_adapter_tensor_count",
        "adapter_path",
        "adapter_manifest",
        "completed_at",
    }
    completion_fields = {
        "run_id",
        "output_dir",
        "run_marker_sha256",
        "artifact_manifest",
        "artifact_manifest_sha256",
        "reload_evidence",
        "reload_evidence_sha256",
        "final_export",
        "run_completed",
    }
    allowed_fields = (
        training_fields | completion_fields
        if metrics.get("run_completed") is True
        else training_fields
    )
    if set(metrics) != allowed_fields:
        raise ValueError("MLX runtime metrics contain an unexpected runtime field.")
    candidate = plan["recommended"]
    runtime = candidate.get("runtime_contract")
    scope = {
        "pilot": "uninterrupted-pilot",
        "full": "uninterrupted-full-train",
    }[action]
    expected = {
        "schema_version": "aptus.runtime-metrics.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "method": candidate["method"],
        "training_runtime": "mlx-lm",
        "compute_backend": "mps",
        "compiler_id": runtime.get("compiler_id")
        if isinstance(runtime, dict)
        else None,
        "scope": scope,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "distribution": "single",
        "actual_world_size": 1,
        "memory_metric_backend": "mlx",
        "finite_train_loss": True,
        "optimizer_update_observed": True,
    }
    if any(metrics.get(name) != value for name, value in expected.items()):
        raise ValueError("MLX runtime metrics do not bind the selected runtime action.")
    try:
        model_load_binding = _require_mlx_model_load_binding(
            plan, metrics.get("model_load_binding")
        )
    except ValueError as error:
        raise ValueError(
            "MLX runtime metrics do not prove a pinned safe model load."
        ) from error
    if candidate.get("method") not in {"lora", "qlora"}:
        raise ValueError("MLX runtime metrics require a LoRA or QLoRA candidate.")
    if (
        not isinstance(runtime, dict)
        or runtime.get("training_runtime") != "mlx-lm"
        or runtime.get("compute_backend") != "mps"
        or candidate.get("distribution") != "single"
        or candidate.get("world_size") != 1
    ):
        raise ValueError("MLX runtime metrics do not match the plan runtime contract.")

    updates = metrics.get("completed_optimizer_updates")
    opportunities = metrics.get("optimizer_update_opportunities")
    minimum_updates = 2 if action == "pilot" else 1
    if (
        not isinstance(updates, int)
        or isinstance(updates, bool)
        or not isinstance(opportunities, int)
        or isinstance(opportunities, bool)
        or updates != opportunities
        or updates < minimum_updates
    ):
        raise ValueError(
            "MLX runtime metrics do not prove completed optimizer updates."
        )
    for name in ("global_step", "micro_iterations", "gradient_accumulation_steps"):
        value = metrics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"MLX runtime metrics require positive {name}.")
    accumulation = int(candidate["gradient_accumulation_steps"])
    micro_iterations = int(metrics["micro_iterations"])
    if (
        metrics["global_step"] != micro_iterations
        or metrics["gradient_accumulation_steps"] != accumulation
        or metrics.get("max_epochs") != plan["target"]["max_epochs"]
        or opportunities != micro_iterations // accumulation
        or micro_iterations % accumulation
    ):
        raise ValueError("MLX runtime metrics do not bind the planned training extent.")

    split_contract = _read_json_object(
        bundle / "data" / "mlx" / "split-contract.json",
        "MLX split contract",
    )
    splits = split_contract.get("splits")
    train_split = splits.get("train") if isinstance(splits, dict) else None
    valid_split = splits.get("valid") if isinstance(splits, dict) else None
    micro_batch = candidate.get("micro_batch_size")
    if (
        split_contract.get("schema_version") != "aptus.mlx-split.v1"
        or not isinstance(micro_batch, int)
        or isinstance(micro_batch, bool)
        or micro_batch < 1
        or split_contract.get("micro_batch_size") != micro_batch
        or not isinstance(train_split, dict)
        or not isinstance(valid_split, dict)
    ):
        raise ValueError("MLX split contract does not bind the planned micro-batch.")
    train_examples = metrics.get("train_examples")
    validation_examples = metrics.get("validation_examples")
    source_train = metrics.get("source_train_examples")
    source_validation = metrics.get("source_validation_examples")
    if (
        set(split_contract)
        != {"schema_version", "micro_batch_size", "padding_policy", "splits"}
        or split_contract.get("padding_policy")
        != "repeat-within-disjoint-split-to-complete-final-batch"
        or not isinstance(splits, dict)
        or set(splits) != {"train", "valid"}
        or set(train_split) != {"source_row_count", "compiled_row_count"}
        or set(valid_split) != {"source_row_count", "compiled_row_count"}
        or not isinstance(train_examples, int)
        or isinstance(train_examples, bool)
        or train_examples < micro_batch
        or not isinstance(validation_examples, int)
        or isinstance(validation_examples, bool)
        or validation_examples < micro_batch
        or not isinstance(source_train, int)
        or isinstance(source_train, bool)
        or source_train < 1
        or not isinstance(source_validation, int)
        or isinstance(source_validation, bool)
        or source_validation < 1
        or train_split.get("source_row_count") != source_train
        or valid_split.get("source_row_count") != source_validation
        or train_split.get("compiled_row_count") != train_examples
        or valid_split.get("compiled_row_count") != validation_examples
        or train_examples % micro_batch
        or validation_examples % micro_batch
    ):
        raise ValueError("MLX runtime metrics do not match the compiled data split.")
    if action == "pilot" and micro_iterations != 2 * accumulation:
        raise ValueError(
            "MLX pilot metrics do not prove exactly two optimizer updates."
        )
    if action == "full":
        batches_per_epoch = train_examples // micro_batch
        epoch_iterations = batches_per_epoch * int(plan["target"]["max_epochs"])
        expected_iterations = math.ceil(epoch_iterations / accumulation) * accumulation
        if micro_iterations != expected_iterations:
            raise ValueError(
                "MLX full metrics do not match the dataset-derived epoch schedule."
            )

    train_losses = metrics.get("train_loss_observations")
    if (
        not isinstance(train_losses, list)
        or not train_losses
        or any(
            not isinstance(loss, (int, float))
            or isinstance(loss, bool)
            or not math.isfinite(loss)
            for loss in train_losses
        )
    ):
        raise ValueError("MLX runtime metrics require finite train-loss observations.")
    validation_examples = metrics.get("validation_examples")
    validation_losses = metrics.get("validation_loss_observations")
    if (
        not isinstance(validation_examples, int)
        or isinstance(validation_examples, bool)
        or validation_examples < 0
        or (
            validation_examples > 0
            and (
                metrics.get("finite_validation_loss") is not True
                or not isinstance(validation_losses, list)
                or not validation_losses
                or any(
                    not isinstance(loss, (int, float))
                    or isinstance(loss, bool)
                    or not math.isfinite(loss)
                    for loss in validation_losses
                )
            )
        )
    ):
        raise ValueError(
            "MLX runtime metrics require finite validation-loss observations."
        )

    measured_peak = metrics.get("measured_peak_bytes")
    if (
        not isinstance(measured_peak, int)
        or isinstance(measured_peak, bool)
        or measured_peak <= 0
        or "free_vram_bytes" in metrics
    ):
        raise ValueError("MLX runtime metrics require a positive MLX memory peak.")
    for name in ("active_memory_bytes", "cache_memory_bytes"):
        value = metrics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"MLX runtime metrics require non-negative {name}.")
    adapter_delta = metrics.get("adapter_delta_l1")
    changed_tensors = metrics.get("changed_adapter_tensor_count")
    if (
        not isinstance(adapter_delta, (int, float))
        or isinstance(adapter_delta, bool)
        or not math.isfinite(adapter_delta)
        or adapter_delta <= 0
        or not isinstance(changed_tensors, int)
        or isinstance(changed_tensors, bool)
        or changed_tensors <= 0
    ):
        raise ValueError("MLX runtime metrics require a positive adapter delta.")

    binding = metrics.get("trainable_target_binding")
    planned_targets = candidate.get("target_modules")
    if not isinstance(binding, dict) or not isinstance(planned_targets, list):
        raise ValueError("MLX runtime metrics require an exact target binding.")
    layer_count = int(plan["model"]["layers"])
    expected_instances = len(planned_targets) * layer_count
    binding_payload = {
        name: value for name, value in binding.items() if name != "descriptor_sha256"
    }
    resolved_keys = binding.get("resolved_layer_keys")
    binding_fields = {
        "schema_version",
        "planned_target_modules",
        "resolved_layer_keys",
        "transformer_layer_count",
        "expected_adapter_target_instance_count",
        "adapter_target_instance_count",
        "trainable_tensor_count",
        "target_instance_counts",
        "descriptor_sha256",
    }
    if (
        set(binding) != binding_fields
        or binding.get("schema_version") != "aptus.mlx-trainable-target-binding.v1"
        or binding.get("planned_target_modules") != planned_targets
        or binding.get("transformer_layer_count") != layer_count
        or binding.get("expected_adapter_target_instance_count") != expected_instances
        or binding.get("adapter_target_instance_count") != expected_instances
        or binding.get("trainable_tensor_count") != expected_instances * 2
        or binding.get("target_instance_counts")
        != {target: layer_count for target in planned_targets}
        or not isinstance(resolved_keys, list)
        or len(resolved_keys) != len(planned_targets)
        or any(not isinstance(key, str) or not key for key in resolved_keys)
        or len(set(resolved_keys)) != len(resolved_keys)
        or binding.get("descriptor_sha256") != _json_hash(binding_payload)
    ):
        raise ValueError("MLX runtime metrics target binding is stale or inexact.")

    admission = _verify_mlx_admission(
        plan,
        metrics.get("unified_memory_admission"),
        label=f"MLX {action} admission",
    )
    if (
        model_load_binding["packed_checkpoint_binding"]["observed_safetensors_bytes"]
        != admission["observed_safetensors_bytes"]
    ):
        raise ValueError(
            "MLX runtime metrics bind different checkpoint byte measurements."
        )
    adapter_manifest = metrics.get("adapter_manifest")
    if (
        not isinstance(adapter_manifest, list)
        or not adapter_manifest
        or not isinstance(metrics.get("adapter_path"), str)
        or not metrics["adapter_path"]
        or not isinstance(metrics.get("completed_at"), str)
        or not metrics["completed_at"]
    ):
        raise ValueError("MLX runtime metrics require an adapter manifest.")
    return metrics


def _verify_mlx_reload_evidence(
    plan: dict[str, Any], metrics: dict[str, Any], evidence: object
) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise ValueError("MLX reload evidence must be a JSON object.")
    candidate = plan["recommended"]
    expected = {
        "schema_version": "aptus.mlx-reload-evidence.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "method": candidate["method"],
        "training_runtime": "mlx-lm",
        "compute_backend": "mps",
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "fresh_process_observed": True,
        "generation_max_tokens": 4,
    }
    evidence_fields = set(expected) | {
        "parent_pid",
        "verifier_pid",
        "adapter_manifest_sha256",
        "generation_tokens",
        "generation_text_sha256",
        "measured_peak_bytes",
        "unified_memory_admission",
        "verified_at",
    }
    if (
        set(evidence) != evidence_fields
        or any(evidence.get(name) != value for name, value in expected.items())
        or not isinstance(evidence.get("verified_at"), str)
        or not evidence["verified_at"]
    ):
        raise ValueError("MLX reload evidence does not bind the selected run.")
    generated = evidence.get("generation_tokens")
    parent_pid = evidence.get("parent_pid")
    verifier_pid = evidence.get("verifier_pid")
    if (
        not isinstance(generated, int)
        or isinstance(generated, bool)
        or not 1 <= generated <= 4
        or not isinstance(parent_pid, int)
        or isinstance(parent_pid, bool)
        or parent_pid <= 0
        or not isinstance(verifier_pid, int)
        or isinstance(verifier_pid, bool)
        or verifier_pid <= 0
        or verifier_pid == parent_pid
        or not _sha256_string(evidence.get("generation_text_sha256"))
        or evidence.get("adapter_manifest_sha256")
        != _json_hash(metrics["adapter_manifest"])
    ):
        raise ValueError(
            "MLX reload evidence does not prove bounded fresh-process use."
        )
    reload_peak = evidence.get("measured_peak_bytes")
    if (
        not isinstance(reload_peak, int)
        or isinstance(reload_peak, bool)
        or reload_peak <= 0
    ):
        raise ValueError("MLX reload evidence requires a positive MLX memory peak.")
    admission = _verify_mlx_admission(
        plan,
        evidence.get("unified_memory_admission"),
        label="MLX reload admission",
    )
    model_load_binding = _require_mlx_model_load_binding(
        plan, metrics.get("model_load_binding")
    )
    if (
        model_load_binding["packed_checkpoint_binding"]["observed_safetensors_bytes"]
        != admission["observed_safetensors_bytes"]
    ):
        raise ValueError(
            "MLX reload evidence binds different checkpoint byte measurements."
        )
    return evidence


def _mlx_file_entry(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"MLX artifact is missing or unsafe: {path}.")
    resolved = path.resolve(strict=True)
    if root.resolve() not in resolved.parents:
        raise ValueError(f"MLX artifact escapes its owned run: {path}.")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _verify_mlx_completed_run(
    bundle: Path,
    plan: dict[str, Any],
    root: Path,
    *,
    action: Literal["pilot", "full"],
) -> dict[str, Any]:
    if root.is_symlink():
        raise ValueError("MLX run output cannot be a symlink.")
    resolved = root.resolve(strict=True)
    expected_parent = (
        (bundle / "pilot-output").resolve()
        if action == "pilot"
        else (bundle / "runs").resolve()
    )
    prefix = "pilot_" if action == "pilot" else "run_"
    if resolved.parent != expected_parent or not resolved.name.startswith(prefix):
        raise ValueError("MLX run output is outside its owned action directory.")

    metrics_path = resolved / "metrics.json"
    metrics = _read_json_object(metrics_path, "MLX completed-run metrics")
    _verify_mlx_runtime_metrics(bundle, plan, metrics, action=action)
    if (
        metrics.get("run_completed") is not True
        or metrics.get("run_id") != resolved.name
        or metrics.get("output_dir") != str(resolved)
        or metrics.get("final_export") is not None
        and action == "pilot"
    ):
        raise ValueError("MLX completed metrics do not bind the owned run.")

    marker_path = resolved / ".aptus-run.json"
    marker = _read_json_object(marker_path, "MLX run-output contract")
    marker_expected = {
        "schema_version": "aptus.mlx-run-output.v1",
        "run_id": resolved.name,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
    }
    if (
        any(marker.get(name) != value for name, value in marker_expected.items())
        or not isinstance(marker.get("created_at"), str)
        or not marker["created_at"]
        or set(marker) != set(marker_expected) | {"created_at"}
        or metrics.get("run_marker_sha256") != sha256_file(marker_path)
    ):
        raise ValueError("MLX run-output contract is stale or unbound.")

    training_metrics_path = resolved / "training-metrics.json"
    training_metrics = _read_json_object(training_metrics_path, "MLX training metrics")
    _verify_mlx_runtime_metrics(bundle, plan, training_metrics, action=action)
    completion_fields = {
        "run_id",
        "output_dir",
        "run_marker_sha256",
        "artifact_manifest",
        "artifact_manifest_sha256",
        "reload_evidence",
        "reload_evidence_sha256",
        "final_export",
        "run_completed",
    }
    if (
        any(metrics.get(name) != value for name, value in training_metrics.items())
        or set(metrics) != set(training_metrics) | completion_fields
    ):
        raise ValueError("MLX completed metrics changed their bound training metrics.")

    adapter_name = "adapters" if action == "pilot" else "final"
    adapter_dir = (resolved / adapter_name).resolve(strict=True)
    if adapter_dir.parent != resolved:
        raise ValueError("MLX adapter directory escapes its owned run.")
    adapter_paths = (
        adapter_dir / "adapter_config.json",
        adapter_dir / "adapters.safetensors",
    )
    actual_adapter_manifest = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in adapter_paths
        if path.is_file() and not path.is_symlink()
    ]
    if (
        len(actual_adapter_manifest) != 2
        or metrics.get("adapter_manifest") != actual_adapter_manifest
        or metrics.get("adapter_path")
        != adapter_dir.relative_to(bundle.resolve()).as_posix()
        or {path.name for path in adapter_dir.iterdir() if path.is_file()}
        != {"adapter_config.json", "adapters.safetensors"}
    ):
        raise ValueError("MLX adapter file tree does not match its immutable manifest.")
    adapter_config = _read_json_object(
        adapter_dir / "adapter_config.json", "MLX adapter configuration"
    )
    lora_parameters = adapter_config.get("lora_parameters")
    binding = metrics["trainable_target_binding"]
    candidate = plan["recommended"]
    scale = lora_parameters.get("scale") if isinstance(lora_parameters, dict) else None
    if (
        not isinstance(lora_parameters, dict)
        or lora_parameters.get("keys") != binding.get("resolved_layer_keys")
        or lora_parameters.get("rank") != candidate["rank"]
        or not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(scale)
        or scale != candidate["alpha"] / candidate["rank"]
    ):
        raise ValueError("MLX adapter configuration is not plan-bound.")

    reload_path = resolved / "reload-evidence.json"
    reload_evidence = _read_json_object(reload_path, "MLX reload evidence")
    _verify_mlx_reload_evidence(plan, metrics, reload_evidence)
    if metrics.get("reload_evidence") != reload_evidence or metrics.get(
        "reload_evidence_sha256"
    ) != sha256_file(reload_path):
        raise ValueError("MLX completed metrics do not bind reload evidence.")

    manifest_path = resolved / "artifact-manifest.json"
    manifest = _read_json_object(manifest_path, "MLX artifact manifest")
    manifest_expected = {
        "schema_version": "aptus.mlx-artifact-manifest.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
    }
    expected_manifest_paths = {
        ".aptus-run.json",
        "training-metrics.json",
        f"{adapter_name}/adapter_config.json",
        f"{adapter_name}/adapters.safetensors",
        "reload-evidence.json",
    }
    entries = manifest.get("files")
    if (
        set(manifest) != set(manifest_expected) | {"files", "total_bytes"}
        or any(manifest.get(name) != value for name, value in manifest_expected.items())
        or metrics.get("artifact_manifest") != manifest
        or metrics.get("artifact_manifest_sha256") != sha256_file(manifest_path)
        or not isinstance(entries, list)
        or {entry.get("path") for entry in entries if isinstance(entry, dict)}
        != expected_manifest_paths
    ):
        raise ValueError("MLX artifact manifest is missing, stale, or unbound.")
    observed_entries = sorted(
        (
            _mlx_file_entry(resolved.joinpath(*PurePosixPath(relative).parts), resolved)
            for relative in expected_manifest_paths
        ),
        key=lambda entry: entry["path"],
    )
    if entries != observed_entries or manifest.get("total_bytes") != sum(
        entry["size_bytes"] for entry in observed_entries
    ):
        raise ValueError("MLX artifact manifest no longer matches the run files.")

    final_export = None
    expected_files = expected_manifest_paths | {
        "artifact-manifest.json",
        "metrics.json",
    }
    if action == "full":
        final_export_path = resolved / "final-export.json"
        final_export = _read_json_object(final_export_path, "MLX final export")
        final_entries = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in adapter_paths
        ]
        final_expected = {
            "schema_version": "aptus.mlx-final-export.v1",
            "verification_level": "immutable-adapter-file-tree",
            "plan_id": plan["plan_id"],
            "candidate_id": candidate["candidate_id"],
            "model_revision": plan["model"]["revision"],
            "dataset_sha256": plan["dataset"]["source_sha256"],
            "method": candidate["method"],
            "training_runtime": "mlx-lm",
            "compute_backend": "mps",
            "distribution": "single",
            "world_size": 1,
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
            "files": final_entries,
            "total_bytes": sum(entry["size_bytes"] for entry in final_entries),
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "reload_evidence_sha256": sha256_file(reload_path),
        }
        if (
            final_export != final_expected
            or metrics.get("final_export") != final_export
        ):
            raise ValueError(
                "MLX final export is stale or does not bind its adapter tree."
            )
        expected_files.add("final-export.json")
    actual_files = {
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file()
    }
    actual_directories = {
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_dir()
    }
    if actual_files != expected_files or actual_directories != {adapter_name}:
        raise ValueError(
            "MLX completed run contains an unexpected or partial file tree."
        )

    return {
        "root": str(resolved),
        "metrics": metrics,
        "metrics_sha256": sha256_file(metrics_path),
        "measured_peak_bytes": metrics["measured_peak_bytes"],
        "artifact_manifest": manifest,
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "artifact_total_bytes": manifest["total_bytes"],
        "adapter_total_bytes": sum(
            entry["size_bytes"] for entry in actual_adapter_manifest
        ),
        "reload_evidence": reload_evidence,
        "reload_evidence_sha256": sha256_file(reload_path),
        "final_export": final_export,
        "final_export_sha256": (
            sha256_file(resolved / "final-export.json") if final_export else None
        ),
    }


def _verify_mlx_pilot_attestation(
    bundle: Path,
    plan: dict[str, Any],
    report: dict[str, Any],
    pilot_metrics_path: Path,
) -> dict[str, Any]:
    bindings = report.get("bindings")
    expected_bindings = {
        "bundle": sha256_file(bundle / "bundle-manifest.json"),
        "dataset": plan["dataset"]["source_sha256"],
        "model_revision": plan["model"]["revision"],
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "pilot_metrics": sha256_file(pilot_metrics_path),
    }
    if not isinstance(bindings, dict) or any(
        bindings.get(name) != value for name, value in expected_bindings.items()
    ):
        raise ValueError("MLX pilot attestation is stale for the current bundle.")
    metrics = _read_json_object(pilot_metrics_path, "MLX pilot metrics")
    if report.get("pilot_metrics") != metrics:
        raise ValueError("MLX validation report does not bind its pilot metrics.")
    output_dir = metrics.get("output_dir")
    if not isinstance(output_dir, str):
        raise ValueError("MLX pilot metrics do not bind an owned output directory.")
    evidence = _verify_mlx_completed_run(
        bundle,
        plan,
        Path(output_dir),
        action="pilot",
    )
    if sha256_file(Path(output_dir) / "metrics.json") != sha256_file(
        pilot_metrics_path
    ):
        raise ValueError("MLX pilot copy changed from its owned run metrics.")
    return evidence


def _current_available_unified_memory_bytes() -> int:
    try:
        profile = probe_apple_platform()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "Current Apple unified-memory admission probe failed."
        ) from error
    available = profile.available_memory_bytes
    if not isinstance(available, int) or isinstance(available, bool) or available <= 0:
        raise ValueError("Current available Apple unified memory is unknown.")
    return available


def _verify_mlx_train_artifacts(record: dict[str, Any]) -> dict[str, Any]:
    bundle = Path(record["bundle_dir"]).resolve(strict=True)
    manifest_errors = validate_bundle_manifest(bundle)
    if manifest_errors:
        raise ValueError(
            "Bundle changed before MLX completion verification: "
            + " | ".join(manifest_errors)
        )
    run_value = record.get("run_output_dir")
    if not isinstance(run_value, str):
        raise ValueError("MLX training job has no bound output directory.")
    plan = _read_json_object(bundle / "plan.json", "Bundle plan")
    plan_errors = validate_plan_payload(plan, root=bundle, verify_dataset=True)
    if plan_errors:
        raise ValueError("Bundle plan is invalid: " + " | ".join(plan_errors))
    if _plan_training_runtime(plan) != "mlx-lm":
        raise ValueError("MLX completion verifier received a non-MLX plan.")
    completed = _verify_mlx_completed_run(
        bundle,
        plan,
        Path(run_value),
        action="full",
    )
    report = _read_json_object(bundle / "validation-report.json", "Validation report")
    if report.get("state") not in {
        "pilot-pass",
        "execution-approved",
        "measured-run-pass",
    }:
        raise ValueError("MLX training exited without a current pilot attestation.")
    _verify_mlx_pilot_attestation(
        bundle,
        plan,
        report,
        bundle / "pilot-output" / "metrics.json",
    )
    command = record.get("command")
    managed_deferred = (
        isinstance(command, list) and "--defer-parent-promotion" in command
    )
    if managed_deferred and report.get("state") != "execution-approved":
        raise ValueError(
            "Managed MLX training exited without deferred parent promotion."
        )
    source_active_run = None
    if report.get("state") == "execution-approved":
        source_active_run = {
            "output_dir": str(Path(run_value).resolve()),
            "run_id": Path(run_value).resolve().name,
            "plan_id": plan["plan_id"],
            "candidate_id": plan["recommended"]["candidate_id"],
        }
        if report.get("active_run") != source_active_run:
            raise ValueError(
                "Managed MLX execution approval does not bind the completed run."
            )
    final_export = completed["final_export"]
    metrics = completed["metrics"]
    final_report = {
        "path": str(Path(run_value).resolve() / "final"),
        "manifest_sha256": completed["final_export_sha256"],
        "total_bytes": completed["adapter_total_bytes"],
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "distribution": "single",
        "world_size": 1,
        "training_runtime": "mlx-lm",
        "artifact_manifest_sha256": completed["artifact_manifest_sha256"],
        "reload_evidence_sha256": completed["reload_evidence_sha256"],
        "export_contract": final_export,
    }
    measured_report = {
        "output_dir": str(Path(run_value).resolve()),
        "metrics_sha256": completed["metrics_sha256"],
        "global_step": metrics["global_step"],
        "completed_optimizer_updates": metrics["completed_optimizer_updates"],
        "measured_peak_bytes": metrics["measured_peak_bytes"],
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "distribution": "single",
        "world_size": 1,
        "training_runtime": "mlx-lm",
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
    }
    if report.get("state") == "measured-run-pass" and (
        report.get("final_export") != final_report
        or report.get("measured_run") != measured_report
    ):
        raise ValueError(
            "Existing MLX measured-run attestation does not match the "
            "independently verified completed run."
        )
    return {
        "training_runtime": "mlx-lm",
        "final_export": final_report,
        "measured_run": measured_report,
        "source_report_state": report["state"],
        "source_bindings": report.get("bindings"),
        "source_active_run": source_active_run,
        "source_report_sha256": sha256_file(bundle / "validation-report.json"),
    }


def _verify_train_artifacts(record: dict[str, Any]) -> dict[str, Any]:
    bundle = Path(record["bundle_dir"]).resolve(strict=True)
    plan = _read_json_object(bundle / "plan.json", "Bundle plan")
    if _plan_training_runtime(plan) == "mlx-lm":
        return _verify_mlx_train_artifacts(record)
    return _verify_cuda_train_artifacts(record)


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


def _promote_cuda_train_attestation(
    record: dict[str, Any],
    evidence: dict[str, Any],
    *,
    _report: dict[str, Any] | None = None,
    _persist: bool = True,
) -> dict[str, Any]:
    bundle = Path(record["bundle_dir"]).resolve(strict=True)
    report_path = bundle / "validation-report.json"
    if _report is None:
        with _bundle_report_lock(bundle):
            return _promote_cuda_train_attestation(
                record,
                evidence,
                _report=_read_json_object(report_path, "Validation report"),
                _persist=True,
            )
    report = _report
    if (
        report.get("state") == "measured-run-pass"
        and report.get("final_export") == evidence["final_export"]
        and report.get("measured_run") == evidence["measured_run"]
        and sha256_file(report_path) == evidence.get("source_report_sha256")
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
    report["validation_level"] = "measured-run"
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
    if _persist:
        _atomic_write_json(report_path, report)
    return {
        "state": report["state"],
        "measured_run_completed_at": report["measured_run_completed_at"],
        "final_export": report["final_export"],
        "measured_run": report["measured_run"],
    }


def _promote_mlx_train_attestation(
    record: dict[str, Any],
    evidence: dict[str, Any],
    *,
    _report: dict[str, Any] | None = None,
    _persist: bool = True,
) -> dict[str, Any]:
    bundle = Path(record["bundle_dir"]).resolve(strict=True)
    report_path = bundle / "validation-report.json"
    if _report is None:
        with _bundle_report_lock(bundle):
            return _promote_mlx_train_attestation(
                record,
                evidence,
                _report=_read_json_object(report_path, "Validation report"),
                _persist=True,
            )
    report = _report
    if (
        report.get("state") == "measured-run-pass"
        and report.get("final_export") == evidence["final_export"]
        and report.get("measured_run") == evidence["measured_run"]
        and sha256_file(report_path) == evidence.get("source_report_sha256")
    ):
        return {
            "state": report["state"],
            "measured_run_completed_at": report.get("measured_run_completed_at"),
            "final_export": report["final_export"],
            "measured_run": report["measured_run"],
        }
    if (
        report.get("state") != evidence.get("source_report_state")
        or report.get("bindings") != evidence.get("source_bindings")
        or report.get("active_run") != evidence.get("source_active_run")
        or sha256_file(report_path) != evidence.get("source_report_sha256")
        or report.get("state") not in {"pilot-pass", "execution-approved"}
    ):
        raise ValueError(
            "MLX pilot evidence changed before parent measured-run promotion."
        )
    report["state"] = "measured-run-pass"
    report["validation_level"] = "measured-run"
    report["measured_run_completed_at"] = _now()
    report["final_export"] = evidence["final_export"]
    report["measured_run"] = evidence["measured_run"]
    runtime_evidence = report.get("runtime_evidence")
    if not isinstance(runtime_evidence, list):
        runtime_evidence = []
    report["runtime_evidence"] = [
        *runtime_evidence,
        "Parent verification accepted an uninterrupted MLX-LM measured run.",
    ]
    for name in (
        "active_run",
        "measured_run_pending_at",
        "pending_final_export",
        "pending_measured_run",
    ):
        report.pop(name, None)
    if _persist:
        _atomic_write_json(report_path, report)
    return {
        "state": report["state"],
        "measured_run_completed_at": report["measured_run_completed_at"],
        "final_export": report["final_export"],
        "measured_run": report["measured_run"],
    }


def _parent_promotion_receipt(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    promoted_at: Any,
) -> dict[str, Any]:
    job_id = record.get("id")
    run_id = record.get("run_id")
    fingerprint = record.get("artifact_fingerprint")
    if not isinstance(job_id, str) or not job_id.startswith("job_"):
        raise ValueError("Parent promotion requires the immutable Aptus job ID.")
    if not isinstance(run_id, str) or not run_id.startswith("run_"):
        raise ValueError("Parent promotion requires the immutable Aptus run ID.")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Parent promotion requires the bound artifact fingerprint.")
    if not isinstance(promoted_at, str) or not promoted_at:
        raise ValueError("Parent promotion requires its completion timestamp.")
    return {
        "schema_version": PARENT_PROMOTION_SCHEMA_VERSION,
        "job_id": job_id,
        "run_id": run_id,
        "artifact_fingerprint": fingerprint,
        "evidence_sha256": _json_hash(evidence),
        "promoted_at": promoted_at,
    }


def _already_promoted_train_attestation(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any] | None:
    if (
        report.get("state") != "measured-run-pass"
        or report.get("final_export") != evidence.get("final_export")
        or report.get("measured_run") != evidence.get("measured_run")
    ):
        return None
    pending_fields = {
        "active_run",
        "measured_run_pending_at",
        "pending_final_export",
        "pending_measured_run",
    }
    try:
        expected_receipt = _parent_promotion_receipt(
            record,
            evidence,
            promoted_at=report.get("measured_run_completed_at"),
        )
    except ValueError as error:
        raise ValueError(
            "Matching measured-run state lacks a valid parent-promotion receipt."
        ) from error
    if (
        report.get("validation_level") != "measured-run"
        or pending_fields.intersection(report)
        or report.get("parent_promotion") != expected_receipt
    ):
        raise ValueError(
            "Matching measured-run state lacks a valid parent-promotion receipt."
        )
    return {
        "state": report["state"],
        "measured_run_completed_at": report["measured_run_completed_at"],
        "final_export": report["final_export"],
        "measured_run": report["measured_run"],
    }


def _promote_train_attestation(
    record: dict[str, Any],
    evidence: dict[str, Any],
    *,
    _allow_legacy_mlx_child_completion: bool = False,
) -> dict[str, Any]:
    bundle = Path(record["bundle_dir"]).resolve(strict=True)
    expected_fingerprint = record.get("artifact_fingerprint")
    if not isinstance(expected_fingerprint, str):
        raise ValueError(
            "Pending measured-run promotion requires the submission-bound bundle fingerprint."
        )
    report_path = bundle / "validation-report.json"
    with _bundle_report_lock(bundle):
        _require_current_bundle_model_policy(
            bundle,
            expected_artifact_fingerprint=expected_fingerprint,
            enforce_current_policy=False,
        )
        report = _read_json_object(report_path, "Validation report")
        # Bundles compiled before managed MLX deferral let the verified child
        # write the terminal state. Only the live worker may bridge that exact
        # report; crash recovery still requires an existing parent receipt.
        command = record.get("command")
        legacy_mlx_command = (
            isinstance(command, list)
            and "run.py" in command
            and "--defer-parent-promotion" not in command
        )
        verified_mlx_child_completion = (
            _allow_legacy_mlx_child_completion
            and legacy_mlx_command
            and evidence.get("training_runtime") == "mlx-lm"
            and evidence.get("source_report_state") == "measured-run-pass"
            and report.get("state") == "measured-run-pass"
            and report.get("validation_level") == "measured-run"
            and report.get("bindings") == evidence.get("source_bindings")
            and report.get("final_export") == evidence.get("final_export")
            and report.get("measured_run") == evidence.get("measured_run")
            and "parent_promotion" not in report
            and sha256_file(report_path) == evidence.get("source_report_sha256")
        )
        promoted = (
            None
            if verified_mlx_child_completion
            else _already_promoted_train_attestation(record, evidence, report)
        )
        if promoted is not None:
            _require_current_bundle_model_policy(
                bundle,
                expected_artifact_fingerprint=expected_fingerprint,
                enforce_current_policy=False,
            )
            return promoted
        _require_current_bundle_model_policy(
            bundle,
            expected_artifact_fingerprint=expected_fingerprint,
        )
        if evidence.get("training_runtime") == "mlx-lm":
            attestation = _promote_mlx_train_attestation(
                record,
                evidence,
                _report=report,
                _persist=False,
            )
        else:
            attestation = _promote_cuda_train_attestation(
                record,
                evidence,
                _report=report,
                _persist=False,
            )
        _require_current_bundle_model_policy(
            bundle,
            expected_artifact_fingerprint=expected_fingerprint,
        )
        report["parent_promotion"] = _parent_promotion_receipt(
            record,
            evidence,
            promoted_at=report.get("measured_run_completed_at"),
        )
        _atomic_write_json(report_path, report)
        return attestation


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
    censuses = tuple(
        require_trainable_parameter_census(
            phase.get("trainable_parameter_census"),
            method=str(plan["recommended"]["method"]),
        )
        for phase in phases
        if isinstance(phase, dict)
    )
    if len(censuses) != 2 or censuses[0] != censuses[1]:
        raise ValueError("Pilot phases do not bind the same trainable parameter set.")
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
    world_size: int = 0,
    device_indices: list[int] | None = None,
    *,
    interpreter: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
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
    runtime_interpreter = (
        str(interpreter) if interpreter is not None else sys.executable
    )
    if not runtime_interpreter:
        raise ValueError("CUDA runtime interpreter is empty.")
    probe_environment = os.environ.copy()
    if environment is not None:
        probe_environment.update(environment)
    try:
        completed = subprocess.run(
            [
                runtime_interpreter,
                "-c",
                _CUDA_RUNTIME_PROBE,
                json.dumps(selected_indices),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=probe_environment,
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


def _actual_environment_binding(
    bundle: Path,
    *,
    interpreter: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> str:
    runtime_interpreter = (
        str(interpreter) if interpreter is not None else sys.executable
    )
    if not runtime_interpreter:
        raise ValueError("Runtime environment interpreter is empty.")
    probe_environment = os.environ.copy()
    if environment is not None:
        probe_environment.update(environment)
    try:
        completed = subprocess.run(
            [runtime_interpreter, "-c", _RUNTIME_ENVIRONMENT_PROBE, str(bundle)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=probe_environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"Runtime environment probe failed: {error}") from error
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"Runtime environment probe failed: {detail}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Runtime environment probe returned invalid JSON.") from error
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "python",
            "platform",
            "direct_constraints",
            "runtime_distributions",
        }
        or not isinstance(value["python"], str)
        or not isinstance(value["platform"], str)
        or not isinstance(value["direct_constraints"], dict)
        or not isinstance(value["runtime_distributions"], dict)
    ):
        raise ValueError("Runtime environment probe returned an invalid contract.")
    return _json_hash(value)


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

    def __init__(
        self,
        root: Path,
        *,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> None:
        self.root = private_directory(root)
        self._lock = threading.RLock()
        self._records_lock_state = threading.local()
        self._global_lease_lock_state = threading.local()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self.runtime_environment: dict[str, str] = dict(
            os.environ if runtime_environment is None else runtime_environment
        )
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
            record["schema_version"] = JOB_RECORD_SCHEMA_VERSION
            record.setdefault("job_id", record.get("id"))
            record.setdefault("created_at", _now())
            record.setdefault("action", "unknown")
            record.setdefault("bundle_dir", "")
            record.setdefault("monotonic_clock_binding", None)
            record.setdefault("queued_monotonic_ns", None)
            record.setdefault("child_process_started_monotonic_ns", None)
            record.setdefault("child_process_finished_monotonic_ns", None)
            record.setdefault("terminal_monotonic_ns", None)
            if (
                record.get("state") in _TERMINAL_JOB_STATES
                and record["queued_monotonic_ns"] is not None
                and record["terminal_monotonic_ns"] is None
            ):
                record["terminal_monotonic_ns"] = time.monotonic_ns()
            self._validate_monotonic_timing(record)
            path = self._record_path(record["id"])
            atomic_write_json(path, record, mode=0o600)

    @staticmethod
    def _validate_monotonic_timing(record: Mapping[str, Any]) -> None:
        """Validate queue, child-runtime, and terminal monotonic channels."""

        queued = record.get("queued_monotonic_ns")
        started = record.get("child_process_started_monotonic_ns")
        finished = record.get("child_process_finished_monotonic_ns")
        terminal = record.get("terminal_monotonic_ns")
        for field, value in (
            ("queued_monotonic_ns", queued),
            ("child_process_started_monotonic_ns", started),
            ("child_process_finished_monotonic_ns", finished),
            ("terminal_monotonic_ns", terminal),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"the persisted {field} field is invalid")
        if finished is not None and started is None:
            raise ValueError(
                "child_process_finished_monotonic_ns requires a child start boundary"
            )
        if started is not None and finished is not None and finished < started:
            raise ValueError(
                "the persisted child process monotonic boundaries are unordered"
            )
        if queued is not None and started is not None and started < queued:
            raise ValueError("the child process started before its queued boundary")
        if terminal is not None and queued is None:
            raise ValueError("terminal_monotonic_ns requires a queued boundary")
        if terminal is not None and terminal < queued:
            raise ValueError("the terminal boundary precedes the queued boundary")
        if terminal is not None and finished is not None and terminal < finished:
            raise ValueError("the terminal boundary precedes the child finish boundary")
        state = record.get("state")
        if state in _TERMINAL_JOB_STATES:
            if queued is not None and terminal is None:
                raise ValueError("a new terminal job lacks its monotonic boundary")
        elif terminal is not None:
            raise ValueError("an active job cannot contain a terminal boundary")
        binding = record.get("monotonic_clock_binding")
        if binding is not None and (
            not isinstance(binding, str)
            or re.fullmatch(
                r"(?:linux-boot-sha256:[0-9a-f]{64}|process-monotonic:[0-9a-f]{32})",
                binding,
            )
            is None
        ):
            raise ValueError("the persisted monotonic clock binding is invalid")
        if queued is not None and binding is None:
            raise ValueError("a queued monotonic boundary requires its clock binding")

    def _migrate_record(
        self, value: dict[str, Any], job_id: str
    ) -> tuple[dict[str, Any], bool]:
        schema_version = value.get("schema_version")
        migrated = dict(value)
        changed = False
        if schema_version is None:
            migrated["schema_version"] = JOB_RECORD_SCHEMA_VERSION
            migrated["persistence_migrated_from"] = "aptus.job-record.legacy"
            # Authorization is always established by a new atomic train submission.
            migrated.pop("authorization_status", None)
            migrated.pop("authorization_current", None)
            migrated.setdefault("created_at", _now())
            migrated.setdefault("action", "unknown")
            migrated.setdefault("bundle_dir", "")
            changed = True
        elif schema_version != JOB_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema version {schema_version!r}; expected {JOB_RECORD_SCHEMA_VERSION!r}"
            )
        if migrated.get("id") != job_id:
            raise ValueError("the object ID must match its filename")
        if migrated.get("job_id") is None:
            migrated["job_id"] = job_id
            changed = True
        if migrated.get("job_id") != job_id:
            raise ValueError("job_id must match the immutable record ID")
        valid_states = {item.value for item in RunState}
        if migrated.get("state") not in valid_states:
            raise ValueError("the persisted run state is invalid")
        for field in ("created_at", "action", "bundle_dir"):
            if not isinstance(migrated.get(field), str):
                raise ValueError(f"the persisted {field} field is invalid")
        for field in (
            "monotonic_clock_binding",
            "queued_monotonic_ns",
            "child_process_started_monotonic_ns",
            "child_process_finished_monotonic_ns",
            "terminal_monotonic_ns",
        ):
            if field not in migrated:
                migrated[field] = None
                changed = True
        self._validate_monotonic_timing(migrated)
        return migrated, changed

    def _quarantine_record(self, path: Path, reason: str) -> Path:
        return quarantine_file(path, self.root / "quarantine", reason=reason)

    def _read(self, job_id: str) -> dict[str, Any]:
        with self._records_lock():
            path = self._record_path(job_id)
            if path.is_symlink():
                destination = self._quarantine_record(
                    path, "Job record symlinks are not permitted."
                )
                raise ValueError(
                    f"Invalid Aptus job record {path}; preserved at {destination}: symlinks are not permitted."
                )
            if not path.is_file():
                raise KeyError(job_id)
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                destination = self._quarantine_record(path, str(error))
                raise ValueError(
                    f"Unreadable Aptus job record {path}; preserved at {destination}: {error}"
                ) from error
            if not isinstance(value, dict):
                destination = self._quarantine_record(
                    path, "The persisted value is not a JSON object."
                )
                raise ValueError(
                    f"Invalid Aptus job record {path}; preserved at {destination}: the value must be a JSON object."
                )
            try:
                migrated, changed = self._migrate_record(value, job_id)
            except ValueError as error:
                destination = self._quarantine_record(path, str(error))
                raise ValueError(
                    f"Invalid Aptus job record {path}; preserved at {destination}: {error}"
                ) from error
            if changed:
                atomic_write_json(path, migrated, mode=0o600)
            return migrated

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
        except FileNotFoundError:
            return None
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

    def _lease_snapshot_active(self, lease: dict[str, Any]) -> bool:
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
        stale = (
            (terminal_record and not child_live)
            or (record_state is None and not child_live)
            or not (owner_live or child_live)
        )
        return not stale

    def campaign_lease_active(self) -> bool:
        """Return a nonblocking, conservative host-global lease snapshot.

        Lease and job records are atomically replaced, so telemetry does not
        take the long-lived worker locks. A concurrent lease transition is
        projected as active, never as an optimistic inactive result. Malformed
        evidence still raises and remains fail-closed.
        """

        first = self._read_global_lease()
        if first is None:
            return self._read_global_lease() is not None
        active = self._lease_snapshot_active(first)
        if self._read_global_lease() != first:
            return True
        return active

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

    def _clear_global_lease(self, job_id: str) -> bool:
        """Reconcile this exact job's host-global lease.

        ``False`` means a different or otherwise inconsistent lease remained.
        Callers that make safety claims must not record reconciliation merely
        because cleanup was attempted.
        """

        lease = self._read_global_lease()
        if lease is None:
            return True
        if lease.get("job_id") == job_id and lease.get("state_root") == str(self.root):
            self._lease_path.unlink(missing_ok=True)
            return not self._lease_path.exists()
        return False

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
            child_timing_incomplete = (
                record.get("child_process_started_monotonic_ns") is not None
                and record.get("child_process_finished_monotonic_ns") is None
            )
            if (
                record.get("action") == "train"
                and record.get("return_code") == 0
                and isinstance(verified_evidence, dict)
                and not child_timing_incomplete
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
                error=(
                    "The owning Aptus process and persisted child PID are no longer live; "
                    + (
                        "the exact child-process finish boundary is unavailable."
                        if child_timing_incomplete
                        else "the exit code is unavailable."
                    )
                ),
            )
            self._write(record)
            return record

    def _recover_interrupted_jobs(self) -> None:
        with self._records_lock():
            for path in self._record_paths():
                try:
                    self._reconcile_external_record(self._read(path.stem))
                except (KeyError, ValueError):
                    # Invalid records are quarantined by _read. Other jobs remain usable.
                    continue

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

    def _runtime_interpreter(self, plan: Mapping[str, Any]) -> str:
        runtime_id = _plan_training_runtime(plan)
        runtime_key = runtime_environment_key(runtime_id)
        if (
            runtime_id in {"mlx-lm", "pytorch-mps"}
            or self.runtime_environment.get(runtime_key, "").strip()
        ):
            return resolve_runtime_interpreter(
                runtime_id, environment=self.runtime_environment
            ).path
        return sys.executable

    def _command(
        self,
        bundle: Path,
        action: JobAction,
        *,
        resume_from: str | None,
        run_id: str | None = None,
        plan: Mapping[str, Any] | None = None,
    ) -> list[str]:
        if plan is None:
            plan = _read_json_object(bundle / "plan.json", "Bundle plan")
        runtime_id = _plan_training_runtime(plan)
        interpreter = self._runtime_interpreter(plan)
        if action in {"dependency", "model-data", "preflight"}:
            level = {
                "dependency": "dependency",
                "model-data": "model-data",
                "preflight": "measured-preflight",
            }[action]
            return [
                interpreter,
                "validate.py",
                "--level",
                level,
            ]
        if action == "pilot":
            return [interpreter, "validate.py", "--level", "pilot"]
        if action != "train":
            raise ValueError(f"Unsupported job action: {action}")
        if resume_from is not None:
            raise ValueError("Full-training resume is unsupported in Aptus v0.2.")
        training_entrypoint = "run.py" if runtime_id == "mlx-lm" else "train.py"
        train_arguments = [
            training_entrypoint,
            "--confirm-full-train",
        ]
        if runtime_id == "mlx-lm" and "--defer-parent-promotion" in (
            bundle / "run.py"
        ).read_text(encoding="utf-8"):
            train_arguments.append("--defer-parent-promotion")
        if run_id is None:
            raise ValueError("Full training requires an immutable Aptus run ID.")
        train_arguments.extend(("--output-dir", str(Path("runs") / run_id)))
        if plan["recommended"]["distribution"] == "single":
            return [interpreter, *train_arguments]
        return [
            interpreter,
            "-m",
            "accelerate.commands.accelerate_cli",
            "launch",
            "--config_file",
            str(Path("config") / "accelerate.yaml"),
            *train_arguments,
        ]

    def _require_record_bundle_binding(self, record: Mapping[str, Any]) -> None:
        bundle = Path(str(record["bundle_dir"]))
        expected_fingerprint = record.get("artifact_fingerprint")
        if not isinstance(expected_fingerprint, str):
            raise ValueError(
                "Job record lacks the submission-bound bundle fingerprint required before worker launch."
            )
        plan, _ = _require_current_bundle_model_policy(
            bundle,
            expected_artifact_fingerprint=expected_fingerprint,
        )
        authorized_snapshot = record.get("authorized_model_policy_snapshot_sha256")
        if (
            not isinstance(authorized_snapshot, str)
            or plan.get("model_policy_snapshot_sha256") != authorized_snapshot
        ):
            raise ValueError(
                "Job record lacks the host-authorized model policy snapshot binding."
            )
        action = record.get("action")
        if action not in JOB_ACTIONS:
            raise ValueError("Job record has an unsupported action.")
        expected_command = self._command(
            bundle,
            action,
            resume_from=record.get("resume_from"),
            run_id=record.get("run_id"),
            plan=plan,
        )
        if record.get("command") != expected_command:
            raise ValueError(
                "Job command does not match its submission-bound bundle plan."
            )

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

    def _require_current_pilot(
        self,
        bundle: Path,
        *,
        expected_artifact_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        plan, _ = _require_current_bundle_model_policy(
            bundle,
            expected_artifact_fingerprint=expected_artifact_fingerprint,
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
        plan_errors = validate_plan_payload(plan, root=bundle, verify_dataset=True)
        if plan_errors:
            raise ValueError("Bundle plan is invalid: " + " | ".join(plan_errors))
        bindings = report.get("bindings", {})
        if not isinstance(bindings, dict):
            raise ValueError("Pilot validation bindings must be a JSON object.")
        pilot_metrics = bundle / "pilot-output" / "metrics.json"
        if not pilot_metrics.is_file():
            raise ValueError("Pilot attestation is missing its bound metrics artifact.")
        if _plan_training_runtime(plan) == "mlx-lm":
            pilot_evidence = _verify_mlx_pilot_attestation(
                bundle,
                plan,
                report,
                pilot_metrics,
            )
            measured_peak = int(pilot_evidence["measured_peak_bytes"])
            reserve = max(
                int(plan["hardware"].get("reserve_per_device_bytes", 0)),
                8 * 1024**3,
            )
            required_available = measured_peak + reserve
            available = _current_available_unified_memory_bytes()
            if available < required_available:
                raise ValueError(
                    "Current available Apple unified memory is below the measured "
                    "MLX pilot peak plus the required Aptus reserve."
                )
            candidate = plan["recommended"]
            planned_export = int(candidate.get("final_export_bytes", 0))
            measured_export = int(pilot_evidence["adapter_total_bytes"])
            if planned_export <= 0 or measured_export <= 0:
                raise ValueError("MLX pilot does not bind a positive export size.")
            required_export = max(planned_export, measured_export)
            pilot_artifacts = int(pilot_evidence["artifact_total_bytes"])
            planned_disk = int(candidate.get("required_disk_bytes", 0))
            if planned_disk <= 0 or pilot_artifacts <= 0:
                raise ValueError("MLX plan does not bind a positive output disk need.")
            required_output_disk = max(
                planned_disk,
                pilot_artifacts + required_export,
            )
            disk_free = shutil.disk_usage(bundle).free
            if disk_free < required_output_disk:
                raise ValueError(
                    "Current free disk is below the plan-bound MLX staging and "
                    "measured adapter-export requirement."
                )
            return {
                "checked_at": _now(),
                "training_runtime": "mlx-lm",
                "memory_metric_backend": "mlx",
                "execution_semantics": "uninterrupted",
                "resume_supported": False,
                "measured_pilot_peak_bytes": measured_peak,
                "reserve_bytes": reserve,
                "required_available_unified_memory_bytes": required_available,
                "available_unified_memory_bytes": available,
                "measured_pilot_artifact_bytes": pilot_artifacts,
                "measured_pilot_adapter_bytes": measured_export,
                "plan_final_export_bytes": planned_export,
                "required_final_export_disk_bytes": required_export,
                "plan_required_disk_bytes": planned_disk,
                "required_training_output_disk_bytes": required_output_disk,
                "free_disk_bytes": disk_free,
                "memory_basis": "measured pilot MLX peak plus max(plan reserve, 8 GiB)",
                "disk_basis": "max(plan required disk, pilot artifacts plus bound final export)",
            }
        world_size = int(plan["recommended"].get("world_size", 1))
        device_indices = plan["recommended"].get(
            "device_indices", list(range(world_size))
        )
        if not isinstance(device_indices, list):
            raise ValueError("Selected CUDA device indices must be a list.")
        runtime_interpreter = self._runtime_interpreter(plan)
        runtime = _actual_runtime_snapshot(
            world_size,
            device_indices,
            interpreter=runtime_interpreter,
            environment=self.runtime_environment,
        )
        expected = {
            "bundle": sha256_file(bundle / "bundle-manifest.json"),
            "dataset": plan["dataset"]["source_sha256"],
            "model_revision": plan["model"]["revision"],
            "plan_id": plan["plan_id"],
            "candidate_id": plan["recommended"]["candidate_id"],
            "environment": _actual_environment_binding(
                bundle,
                interpreter=runtime_interpreter,
                environment=self.runtime_environment,
            ),
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

    @staticmethod
    def _create_campaign_event_sink(path: Path) -> str:
        parent = private_directory(path.parent)
        if path.parent != parent or path.exists() or path.is_symlink():
            raise FileExistsError("Campaign event sink already exists.")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
            ):
                raise RuntimeError("Campaign event sink integrity check failed.")
            os.fsync(descriptor)
            identity = f"{metadata.st_dev}:{metadata.st_ino}"
        finally:
            os.close(descriptor)
        return identity

    @staticmethod
    def _append_campaign_verification_boundary(
        record: Mapping[str, Any],
        *,
        event_type: str,
        native_outcome: str | None = None,
        reason_code: str = "NONE",
    ) -> None:
        """Append one parent-owned verification boundary to the private sink."""

        if record.get("campaign_event_capture") is not True:
            return
        if event_type not in {"verification.started", "verification.finished"}:
            raise RuntimeError("Campaign verification event type is invalid.")
        started = event_type.endswith("started")
        if started and (native_outcome is not None or reason_code != "NONE"):
            raise RuntimeError("Campaign verification start cannot be terminal.")
        if not started and (
            native_outcome not in {"passed", "failed", "cancelled"}
            or (native_outcome == "passed") != (reason_code == "NONE")
        ):
            raise RuntimeError("Campaign verification finish is not terminal.")
        run_id = record.get("campaign_experiment_run_id")
        job_id = record.get("job_id", record.get("id"))
        sink_value = record.get("campaign_event_sink")
        expected_identity = record.get("campaign_event_sink_identity")
        if (
            record.get("action") != "train"
            or not isinstance(run_id, str)
            or _CAMPAIGN_EXPERIMENT_RUN_ID.fullmatch(run_id) is None
            or not isinstance(job_id, str)
            or not job_id.startswith("job_")
            or not isinstance(sink_value, str)
            or not os.path.isabs(sink_value)
            or not isinstance(expected_identity, str)
            or re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", expected_identity) is None
        ):
            raise RuntimeError("Campaign verification sink binding is invalid.")
        sink = Path(sink_value)
        if fcntl is None:
            raise RuntimeError("Campaign verification sink locking is unavailable.")

        def require_sink_metadata(metadata: os.stat_result, *, failure: str) -> None:
            observed_identity = f"{metadata.st_dev}:{metadata.st_ino}"
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
                or observed_identity != expected_identity
                or metadata.st_size > _CAMPAIGN_EVENT_JOURNAL_MAX_BYTES
            ):
                raise RuntimeError(failure)

        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(sink, flags)
        try:
            metadata = os.fstat(descriptor)
            require_sink_metadata(
                metadata,
                failure="Campaign verification sink integrity check failed.",
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                metadata = os.fstat(descriptor)
                require_sink_metadata(
                    metadata,
                    failure="Campaign verification sink changed before append.",
                )
                boundary = {
                    "schema_version": "aptus.cuda-campaign-runtime-boundary.v1",
                    "experiment_run_id": run_id,
                    "job_id": job_id,
                    "monotonic_ns": time.monotonic_ns(),
                    "wall_time_utc": _now(),
                    "event_type": event_type,
                    "phase": "parent-verification",
                    "action": "train",
                    "native_outcome": native_outcome,
                    "reason_code": reason_code,
                }
                payload = (
                    json.dumps(
                        boundary,
                        ensure_ascii=True,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                if metadata.st_size + len(payload) > (
                    _CAMPAIGN_EVENT_JOURNAL_MAX_BYTES
                ):
                    raise RuntimeError("Campaign verification journal is full.")
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise RuntimeError(
                            "Campaign verification append made no progress."
                        )
                    view = view[written:]
                os.fsync(descriptor)
                require_sink_metadata(
                    os.fstat(descriptor),
                    failure="Campaign verification sink changed during append.",
                )
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _fail_persisted_submission(
        self,
        job_id: str,
        record: Mapping[str, Any],
        *,
        failure_code: str,
        public_error: str,
    ) -> None:
        """Persist a fail-closed terminal handoff and raise its typed receipt."""

        terminal = dict(record)
        terminal.update(
            state=RunState.FAILED.value,
            finished_at=_now(),
            error=public_error,
            submission_failure_code=failure_code,
            terminal_record_persisted=True,
        )
        try:
            reconciled = self._clear_global_lease(job_id)
        except Exception:
            reconciled = False
        if reconciled:
            terminal.update(
                lease_reconciled_at=_now(),
                lease_reconciled_monotonic_ns=time.monotonic_ns(),
            )
        else:
            terminal["lease_reconciliation_error"] = (
                "The host-global lease could not be proven reconciled."
            )
        try:
            self._write(terminal)
        except Exception:
            # The typed exception still carries the complete terminal snapshot;
            # callers must treat a persistence failure as nonqualifying.
            terminal["terminal_record_persisted"] = False
        raise JobSubmissionFailure(job_id, terminal, failure_code) from None

    def submit(
        self,
        bundle_dir: Path,
        *,
        action: JobAction = "preflight",
        confirm_full_train: bool = False,
        resume_from: str | None = None,
        before_start: (
            Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None
        ) = None,
        admission_check: Callable[[], Any] | None = None,
        on_process_registered: Callable[[Mapping[str, Any]], None] | None = None,
        expected_artifact_fingerprint: str | None = None,
        campaign_event_capture: bool = False,
        campaign_experiment_run_id: str | None = None,
    ) -> dict[str, Any]:
        if action not in JOB_ACTIONS:
            raise ValueError(f"Unsupported job action: {action}")
        if on_process_registered is not None and not callable(on_process_registered):
            raise ValueError("on_process_registered must be callable when provided.")
        if action != "train" and confirm_full_train:
            raise ValueError("confirm_full_train is valid only for the train action.")
        if action != "train" and resume_from is not None:
            raise ValueError("resume_from is valid only for the train action.")
        if action == "train" and resume_from is not None:
            raise ValueError(
                "Full-training resume is fail-closed in Aptus v0.2 until a checkpoint manifest binds complete optimizer, scheduler, RNG, model, environment, and plan state."
            )
        if not isinstance(campaign_event_capture, bool):
            raise ValueError("campaign_event_capture must be boolean.")
        if campaign_event_capture:
            if (
                not isinstance(campaign_experiment_run_id, str)
                or _CAMPAIGN_EXPERIMENT_RUN_ID.fullmatch(campaign_experiment_run_id)
                is None
            ):
                raise ValueError(
                    "Campaign event capture requires an exact experiment run ID."
                )
        elif campaign_experiment_run_id is not None:
            raise ValueError(
                "campaign_experiment_run_id requires campaign_event_capture=true."
            )
        bundle = bundle_dir.resolve(strict=True)
        if not bundle.is_dir() or not (bundle / "plan.json").is_file():
            raise ValueError(f"Not an Aptus bundle: {bundle}")
        admitted_plan, observed_artifact_fingerprint = (
            _require_current_bundle_model_policy(
                bundle,
                expected_artifact_fingerprint=expected_artifact_fingerprint,
            )
        )
        if action == "train":
            if not confirm_full_train:
                raise ValueError("Full training requires confirm_full_train=true.")
        job_id = "job_" + uuid.uuid4().hex
        run_id = f"run_{job_id[4:]}" if action == "train" else None
        campaign_event_sink = (
            self.root / ".campaign-events" / f"{job_id}.jsonl"
            if campaign_event_capture
            else None
        )
        command = self._command(
            bundle,
            action,
            resume_from=resume_from,
            run_id=run_id,
            plan=admitted_plan,
        )
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
            "monotonic_clock_binding": None,
            "queued_monotonic_ns": None,
            "child_process_started_monotonic_ns": None,
            "child_process_finished_monotonic_ns": None,
            "terminal_monotonic_ns": None,
            "prelaunch_capacity_check": None,
            "artifact_fingerprint": observed_artifact_fingerprint,
            "plan_id": admitted_plan["plan_id"],
            "candidate_id": admitted_plan["recommended"]["candidate_id"],
            "bundle_manifest_sha256": sha256_file(bundle / "bundle-manifest.json"),
            "authorized_model_policy_snapshot_sha256": admitted_plan[
                "model_policy_snapshot_sha256"
            ],
            "campaign_event_capture": campaign_event_capture,
            "campaign_experiment_run_id": campaign_experiment_run_id,
            "campaign_event_sink": (
                str(campaign_event_sink) if campaign_event_sink is not None else None
            ),
            "campaign_event_sink_identity": None,
        }
        worker = threading.Thread(
            target=self._run,
            args=(job_id, on_process_registered),
            name=f"aptus-{job_id}",
            daemon=True,
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
                                self._require_current_pilot(
                                    bundle,
                                    expected_artifact_fingerprint=(
                                        observed_artifact_fingerprint
                                    ),
                                )
                            )
                        except RuntimeError as error:
                            raise ValueError(
                                "Could not inspect the current training-runtime "
                                f"resources: {error}"
                            ) from error
                    if admission_check is not None:
                        admission_check()
                    record["monotonic_clock_binding"] = (
                        _current_monotonic_clock_binding()
                    )
                    record["queued_monotonic_ns"] = time.monotonic_ns()
                    self._write(record)
                    try:
                        if campaign_event_sink is not None:
                            record["campaign_event_sink_identity"] = (
                                self._create_campaign_event_sink(campaign_event_sink)
                            )
                            self._write(record)
                        self._create_global_lease(record)
                    except Exception:
                        self._fail_persisted_submission(
                            job_id,
                            record,
                            failure_code="SUBMISSION_SETUP_FAILED",
                            public_error=(
                                "The persisted job submission could not establish "
                                "its private event sink and host-global lease."
                            ),
                        )
            if before_start is not None:
                try:
                    metadata = before_start(dict(record))
                    if metadata is not None:
                        unsupported = set(metadata) - {
                            "project_id",
                            "project_revision_id",
                        }
                        if unsupported:
                            raise ValueError(
                                "Job pre-start metadata contains unsupported fields: "
                                + ", ".join(sorted(unsupported))
                            )
                        with self._global_lease_lock(), self._records_lock():
                            current = self._read(job_id)
                            current.update(dict(metadata))
                            self._write(current)
                except Exception:
                    with self._global_lease_lock(), self._records_lock():
                        try:
                            current = self._read(job_id)
                        except Exception:
                            current = record
                        self._fail_persisted_submission(
                            job_id,
                            current,
                            failure_code="PRE_START_PERSISTENCE_FAILED",
                            public_error="Job pre-start persistence failed.",
                        )
            try:
                self._threads[job_id] = worker
                worker.start()
            except Exception:
                with self._global_lease_lock(), self._records_lock():
                    self._threads.pop(job_id, None)
                    try:
                        current = self._read(job_id)
                    except Exception:
                        current = record
                    self._fail_persisted_submission(
                        job_id,
                        current,
                        failure_code="WORKER_START_FAILED",
                        public_error="The Aptus job worker could not start.",
                    )
        try:
            return self.get(job_id)
        except Exception:
            detected_ns = time.monotonic_ns()
            try:
                terminal = self.cancel(
                    job_id,
                    reason_code="OWNERSHIP_UNCERTAIN",
                    trigger_detected_monotonic_ns=detected_ns,
                )
            except Exception:
                worker = self._threads.get(job_id)
                if worker is not None and worker is not threading.current_thread():
                    worker.join(timeout=6)
                with self._records_lock():
                    try:
                        terminal = self._read(job_id)
                    except Exception:
                        terminal = record
            if terminal.get("state") not in {
                RunState.COMPLETED.value,
                RunState.FAILED.value,
                RunState.CANCELLED.value,
            }:
                with self._global_lease_lock(), self._records_lock():
                    self._fail_persisted_submission(
                        job_id,
                        terminal,
                        failure_code="SUBMISSION_HANDOFF_FAILED",
                        public_error=(
                            "The persisted job submission could not complete its "
                            "caller handoff."
                        ),
                    )
            raise JobSubmissionFailure(
                job_id, terminal, "SUBMISSION_HANDOFF_FAILED"
            ) from None

    def _run(
        self,
        job_id: str,
        on_process_registered: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        log_path = self._log_path(job_id)
        launch_spec = self.root / f".{job_id}.launch-spec"
        launch_permit = self.root / f".{job_id}.permit"
        process: subprocess.Popen[str] | None = None
        child_process_started_monotonic_ns: int | None = None
        child_process_finished_monotonic_ns: int | None = None
        try:
            with self._lock, self._global_lease_lock(), self._records_lock():
                record = self._read(job_id)
                if record["state"] in {
                    RunState.CANCELLED.value,
                    RunState.CANCELLING.value,
                }:
                    terminated_at = _now()
                    record.update(
                        state=RunState.CANCELLED.value,
                        finished_at=terminated_at,
                        process_group_terminated_at=record.get(
                            "process_group_terminated_at", terminated_at
                        ),
                        process_group_terminated_monotonic_ns=record.get(
                            "process_group_terminated_monotonic_ns",
                            time.monotonic_ns(),
                        ),
                    )
                    self._write(record)
                    if self._clear_global_lease(job_id):
                        record.update(
                            lease_reconciled_at=_now(),
                            lease_reconciled_monotonic_ns=time.monotonic_ns(),
                        )
                    else:
                        record["lease_reconciliation_error"] = (
                            "The host-global lease did not match the cancelled job."
                        )
                    self._write(record)
                    return
            self._require_record_bundle_binding(record)
            _atomic_write_json(
                launch_spec,
                {
                    "command": record["command"],
                    "expected_artifact_fingerprint": record.get("artifact_fingerprint"),
                    "authorized_model_policy_snapshot_sha256": record.get(
                        "authorized_model_policy_snapshot_sha256"
                    ),
                },
            )
            launch_permit.unlink(missing_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                campaign_environment: dict[str, str] = {}
                if record.get("campaign_event_capture") is True:
                    campaign_run_id = record.get("campaign_experiment_run_id")
                    campaign_sink = record.get("campaign_event_sink")
                    campaign_sink_identity = record.get("campaign_event_sink_identity")
                    expected_sink = self.root / ".campaign-events" / f"{job_id}.jsonl"
                    if (
                        not isinstance(campaign_run_id, str)
                        or _CAMPAIGN_EXPERIMENT_RUN_ID.fullmatch(campaign_run_id)
                        is None
                        or campaign_sink != str(expected_sink)
                        or not isinstance(campaign_sink_identity, str)
                        or re.fullmatch(
                            r"[1-9][0-9]*:[1-9][0-9]*", campaign_sink_identity
                        )
                        is None
                        or expected_sink.is_symlink()
                        or not expected_sink.is_file()
                    ):
                        raise RuntimeError(
                            "The opt-in campaign event sink binding is invalid."
                        )
                    campaign_environment = {
                        "APTUS_CUDA_CAMPAIGN_EVENT_SINK": str(expected_sink),
                        "APTUS_CUDA_CAMPAIGN_EVENT_SINK_IDENTITY": (
                            campaign_sink_identity
                        ),
                        "APTUS_CUDA_CAMPAIGN_EXPERIMENT_RUN_ID": campaign_run_id,
                        "APTUS_CUDA_CAMPAIGN_JOB_ID": job_id,
                    }
                process_options: dict[str, Any] = {}
                if os.name == "posix":
                    process_options["start_new_session"] = True
                elif os.name == "nt" and hasattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP"
                ):
                    process_options["creationflags"] = (
                        subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                child_environment = os.environ.copy()
                for name in _CAMPAIGN_ENVIRONMENT_NAMES:
                    child_environment.pop(name, None)
                child_environment.update(
                    {
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "APTUS_GPU_LEASE_TOKEN": record["id"],
                        "APTUS_EXPECTED_ARTIFACT_FINGERPRINT": record[
                            "artifact_fingerprint"
                        ],
                        "APTUS_AUTHORIZED_MODEL_POLICY_SNAPSHOT_SHA256": record[
                            "authorized_model_policy_snapshot_sha256"
                        ],
                        **campaign_environment,
                    }
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
                    env=child_environment,
                    **process_options,
                )
                child_process_started_monotonic_ns = time.monotonic_ns()
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
                        child_process_started_monotonic_ns=(
                            child_process_started_monotonic_ns
                        ),
                    )
                    if not cancelled_before_registration:
                        current.update(state=RunState.RUNNING.value, started_at=_now())
                    self._write(current)
                    self._bind_global_lease_to_process(
                        job_id, process.pid, process_identity
                    )
                    if not cancelled_before_registration:
                        if on_process_registered is not None:
                            on_process_registered(dict(current))
                        self._require_record_bundle_binding(current)
                        launch_permit.write_text("go\n", encoding="utf-8")
                if cancelled_before_registration and process.poll() is None:
                    self._terminate_process(process)
                return_code = process.wait()
                child_process_finished_monotonic_ns = time.monotonic_ns()
                with self._lock, self._global_lease_lock(), self._records_lock():
                    current = self._read(job_id)
                    persisted_child_finish = current[
                        "child_process_finished_monotonic_ns"
                    ]
                    if (
                        persisted_child_finish is None
                        or child_process_finished_monotonic_ns < persisted_child_finish
                    ):
                        current["child_process_finished_monotonic_ns"] = (
                            child_process_finished_monotonic_ns
                        )
                    current["return_code"] = return_code
                    if current.get("state") in {
                        RunState.CANCELLED.value,
                        RunState.CANCELLING.value,
                    }:
                        current.setdefault("process_group_terminated_at", _now())
                        current.setdefault(
                            "process_group_terminated_monotonic_ns",
                            current["child_process_finished_monotonic_ns"],
                        )
                    self._write(current)
                if self._process_tree_alive(process):
                    raise RuntimeError(
                        "The launcher exited while a descendant process remained in its execution tree."
                    )
            completion_verification_authorized = False
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
                    completion_verification_authorized = True
            completion_error: str | None = None
            completion_attestation: dict[str, Any] | None = None
            if (
                completion_verification_authorized
                and return_code == 0
                and record.get("action") == "train"
            ):
                self._append_campaign_verification_boundary(
                    record, event_type="verification.started"
                )
                try:
                    pending_evidence = _verify_train_artifacts(record)
                    with self._lock, self._global_lease_lock(), self._records_lock():
                        verified_record = self._read(job_id)
                        verified_record["verified_pending_evidence"] = pending_evidence
                        verified_record["pending_evidence_verified_at"] = _now()
                        self._write(verified_record)
                    completion_attestation = _promote_train_attestation(
                        record,
                        pending_evidence,
                        _allow_legacy_mlx_child_completion=True,
                    )
                except (KeyError, OSError, TypeError, ValueError) as error:
                    completion_error = str(error)
                    self._append_campaign_verification_boundary(
                        record,
                        event_type="verification.finished",
                        native_outcome="failed",
                        reason_code="EXPORT_VERIFICATION_FAILURE",
                    )
                else:
                    self._append_campaign_verification_boundary(
                        record,
                        event_type="verification.finished",
                        native_outcome="passed",
                        reason_code="NONE",
                    )
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
                if self._clear_global_lease(job_id):
                    current.update(
                        lease_reconciled_at=_now(),
                        lease_reconciled_monotonic_ns=time.monotonic_ns(),
                    )
                else:
                    current["lease_reconciliation_error"] = (
                        "The host-global lease did not match the terminal job."
                    )
                self._write(current)
        except Exception as error:
            termination_error: Exception | None = None
            if process is not None:
                if self._process_tree_alive(process):
                    try:
                        self._terminate_process(process)
                    except (
                        Exception
                    ) as stop_error:  # pragma: no cover - OS failure path.
                        termination_error = stop_error
                    else:
                        if child_process_finished_monotonic_ns is None:
                            child_process_finished_monotonic_ns = time.monotonic_ns()
                elif (
                    process.poll() is not None
                    and child_process_finished_monotonic_ns is None
                ):
                    child_process_finished_monotonic_ns = time.monotonic_ns()
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
                    if (
                        current["child_process_started_monotonic_ns"] is None
                        and child_process_started_monotonic_ns is not None
                    ):
                        current["child_process_started_monotonic_ns"] = (
                            child_process_started_monotonic_ns
                        )
                    if child_process_finished_monotonic_ns is not None and (
                        current["child_process_finished_monotonic_ns"] is None
                        or child_process_finished_monotonic_ns
                        < current["child_process_finished_monotonic_ns"]
                    ):
                        current["child_process_finished_monotonic_ns"] = (
                            child_process_finished_monotonic_ns
                        )
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
                            authorization_is_current = bool(
                                record.get("action") == "train"
                                and isinstance(cached_capacity, dict)
                            )
                            authorization_status: ValidationAuthorizationStatus = (
                                "current" if authorization_is_current else "blocked"
                            )
                            authorization_error = (
                                None
                                if authorization_is_current
                                else "Pilot authorization is not re-probed while any Aptus GPU job is active."
                            )
                            authorization_capacity = (
                                cached_capacity
                                if isinstance(cached_capacity, dict)
                                else None
                            )
                        else:
                            authorization_status = "deferred"
                            authorization_error = "Deep pilot binding, checkpoint, environment, and current capacity authorization is performed atomically when full training is submitted. Polling does not rehash large pilot artifacts."
                            authorization_capacity = None
                        report = decorate_validation_authorization(
                            report,
                            status=authorization_status,
                            error=authorization_error,
                            capacity=authorization_capacity,
                        )
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
        records = []
        for path in self._record_paths():
            try:
                records.append(self.get(path.stem, include_validation_report=False))
            except (KeyError, ValueError):
                # _read() quarantines invalid records with a reason receipt. A
                # record corrupted after startup must not hide healthy jobs.
                continue
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

    def cancel(
        self,
        job_id: str,
        *,
        reason_code: str | None = None,
        trigger_detected_monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        """Cancel an exactly owned job and persist observable lifecycle milestones.

        The optional fields are used by the opt-in experiment harness.  They do
        not change cancellation authority: this exact ``JobService`` instance
        must still own the worker, and cancellation still targets only its
        verified process group.
        """

        if reason_code is not None and (
            not isinstance(reason_code, str) or not reason_code
        ):
            raise ValueError("Cancellation reason_code must be a non-empty string.")
        if trigger_detected_monotonic_ns is not None and (
            isinstance(trigger_detected_monotonic_ns, bool)
            or not isinstance(trigger_detected_monotonic_ns, int)
            or trigger_detected_monotonic_ns < 0
        ):
            raise ValueError(
                "Cancellation trigger_detected_monotonic_ns must be a non-negative integer."
            )
        if (
            trigger_detected_monotonic_ns is not None
            and trigger_detected_monotonic_ns > time.monotonic_ns()
        ):
            raise ValueError(
                "Cancellation trigger_detected_monotonic_ns cannot be in the future."
            )
        process: subprocess.Popen[str] | None = None
        worker: threading.Thread | None = None
        termination_error: Exception | None = None
        with self._lock, self._global_lease_lock(), self._records_lock():
            record = self._read(job_id)
            verification_in_progress = bool(
                record.get("state")
                in {
                    RunState.QUEUED.value,
                    RunState.RUNNING.value,
                    RunState.CANCELLING.value,
                }
                and record.get("completion_verification_started_at")
                and record.get("return_code") is not None
            )
            if verification_in_progress:
                raise ValueError(
                    "Completion verification is noncancellable after the child process exits."
                )
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
                    cancel_requested_at = _now()
                    cancel_requested_monotonic_ns = time.monotonic_ns()
                    terminated_at = _now()
                    terminated_monotonic_ns = time.monotonic_ns()
                    record.update(
                        state=RunState.CANCELLING.value,
                        cancel_requested_at=cancel_requested_at,
                        cancel_requested_monotonic_ns=cancel_requested_monotonic_ns,
                        cancel_reason_code=reason_code,
                        cancel_trigger_detected_monotonic_ns=(
                            trigger_detected_monotonic_ns
                        ),
                        process_group_terminated_at=terminated_at,
                        process_group_terminated_monotonic_ns=terminated_monotonic_ns,
                        error=None,
                    )
                    if (
                        record["child_process_started_monotonic_ns"] is not None
                        and record["child_process_finished_monotonic_ns"] is None
                    ):
                        record["child_process_finished_monotonic_ns"] = (
                            terminated_monotonic_ns
                        )
                    self._write(record)
                    if worker is None or not worker.is_alive():
                        terminated_at = _now()
                        terminated_monotonic_ns = time.monotonic_ns()
                        record.update(
                            state=RunState.FAILED.value,
                            return_code=return_code,
                            finished_at=terminated_at,
                            process_group_terminated_at=terminated_at,
                            process_group_terminated_monotonic_ns=terminated_monotonic_ns,
                            error=(
                                "The process exited, but its owning verifier is unavailable. "
                                "Aptus will not infer successful completion."
                            ),
                        )
                        self._write(record)
                        if self._clear_global_lease(job_id):
                            record.update(
                                lease_reconciled_at=_now(),
                                lease_reconciled_monotonic_ns=time.monotonic_ns(),
                            )
                        else:
                            record["lease_reconciliation_error"] = (
                                "The host-global lease did not match the cancelled job."
                            )
                        self._write(record)
                    # A live owning worker still has terminal verification and
                    # cleanup to perform.  It retains the lease until `_run`
                    # persists the terminal record; releasing it here would
                    # allow a second GPU job to overlap that ownership window.
                else:
                    cancel_requested_at = _now()
                    cancel_requested_monotonic_ns = time.monotonic_ns()
                    record.update(
                        state=RunState.CANCELLING.value,
                        cancel_requested_at=cancel_requested_at,
                        cancel_requested_monotonic_ns=cancel_requested_monotonic_ns,
                        cancel_reason_code=reason_code,
                        cancel_trigger_detected_monotonic_ns=(
                            trigger_detected_monotonic_ns
                        ),
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
                            terminated_at = _now()
                            terminated_monotonic_ns = time.monotonic_ns()
                            record.update(
                                state=RunState.CANCELLED.value,
                                return_code=process.poll(),
                                finished_at=terminated_at,
                                process_group_terminated_at=terminated_at,
                                process_group_terminated_monotonic_ns=terminated_monotonic_ns,
                                error=None,
                            )
                            if (
                                record["child_process_started_monotonic_ns"] is not None
                                and record["child_process_finished_monotonic_ns"]
                                is None
                            ):
                                record["child_process_finished_monotonic_ns"] = (
                                    terminated_monotonic_ns
                                )
                            self._write(record)
                            if self._clear_global_lease(job_id):
                                record.update(
                                    lease_reconciled_at=_now(),
                                    lease_reconciled_monotonic_ns=time.monotonic_ns(),
                                )
                            else:
                                record["lease_reconciliation_error"] = (
                                    "The host-global lease did not match the cancelled job."
                                )
                            self._write(record)
        if termination_error is not None:
            raise ValueError(str(termination_error)) from termination_error
        if (
            worker is not None
            and worker is not threading.current_thread()
            and worker.ident is not None
        ):
            worker.join(timeout=6)
        return self.get(job_id)
