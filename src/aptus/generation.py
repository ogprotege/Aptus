from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any

from .catalog import STACK_VERSIONS, bundle_requirements
from .domain import (
    Provenance,
    TrainingPlan,
    TrainingRuntime,
    ValidationReport,
    to_primitive,
)
from .methods import method_descriptor
from .profiling import canonical_training_rows, pilot_sample_rows


TRAIN_SCRIPT = r'''#!/usr/bin/env python3
"""Execute the selected Aptus candidate. All user facts are read from plan.json."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import heapq
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
from array import array
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent
if (_BOOTSTRAP_ROOT / "__pycache__").exists():
    raise RuntimeError(
        "Bundle contains an unmanifested __pycache__; remove it before execution."
    )

sys.dont_write_bytecode = True
from plan_contract import validate_bundle_manifest, validate_plan_payload
from runtime_lease import require_execution_lease


ROOT = _BOOTSTRAP_ROOT
TRAINER_CONFIG_PATH = ROOT / "config" / "trainer.json"


@contextmanager
def report_lock():
    with (ROOT / ".validation-report.lock").open("a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
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
            elif msvcrt is not None:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def load_plan() -> dict[str, Any]:
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    errors = validate_plan_payload(plan, root=ROOT, verify_dataset=True)
    if errors:
        raise ValueError("Invalid Aptus plan: " + " | ".join(errors))
    return plan


def bind_visible_cuda_devices(plan: dict[str, Any]) -> None:
    candidate = plan["recommended"]
    world_size = int(candidate["world_size"])
    device_indices = candidate.get("device_indices", list(range(world_size)))
    if (
        not isinstance(device_indices, list)
        or len(device_indices) != world_size
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in device_indices
        )
        or len(set(device_indices)) != len(device_indices)
    ):
        raise RuntimeError("Selected CUDA device indices do not match the planned world.")
    marker = os.environ.get("APTUS_BOUND_DEVICE_CANDIDATE")
    if marker is not None:
        if marker != candidate["candidate_id"]:
            raise RuntimeError("Inherited Aptus CUDA visibility belongs to another candidate.")
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        inherited_tokens = (
            [token.strip() for token in inherited.split(",") if token.strip()]
            if inherited is not None
            else []
        )
        if len(inherited_tokens) != world_size or any(
            token.lower() in {"-1", "nodevfiles", "none"}
            for token in inherited_tokens
        ):
            raise RuntimeError("Inherited Aptus CUDA visibility is missing or malformed.")
        return
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None:
        visible_tokens = [token.strip() for token in existing.split(",") if token.strip()]
        if not visible_tokens or any(
            token.lower() in {"-1", "nodevfiles", "none"}
            for token in visible_tokens
        ):
            raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no selectable CUDA devices.")
        if any(index >= len(visible_tokens) for index in device_indices):
            raise RuntimeError("Selected CUDA device index is outside CUDA_VISIBLE_DEVICES.")
        selected_tokens = [visible_tokens[index] for index in device_indices]
    else:
        selected_tokens = [str(index) for index in device_indices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_tokens)
    os.environ["APTUS_BOUND_DEVICE_CANDIDATE"] = candidate["candidate_id"]


def load_trainer_config() -> dict[str, Any]:
    return json.loads(TRAINER_CONFIG_PATH.read_text(encoding="utf-8"))


def require_compiler_contract(
    plan: dict[str, Any], trainer_config: dict[str, Any]
) -> None:
    expected = {
        "full": ("transformers.full.v2", "full-model-safetensors"),
        "lora": ("transformers.peft-lora.v2", "peft-adapter-safetensors"),
        "int8-lora": (
            "transformers.peft-int8-lora.v2",
            "peft-adapter-safetensors",
        ),
        "qlora": ("transformers.peft-qlora.v2", "peft-adapter-safetensors"),
    }
    method = plan["recommended"]["method"]
    if method not in expected:
        raise RuntimeError("The selected method has no generated compiler contract.")
    if (
        trainer_config.get("compiler_id"),
        trainer_config.get("export_kind"),
    ) != expected[method]:
        raise RuntimeError("Trainer configuration does not bind the selected compiler.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_checkpoint_contract(checkpoint: Path, contract: dict[str, Any]) -> None:
    if not checkpoint.is_dir() or not isinstance(contract, dict):
        raise RuntimeError("Pilot checkpoint evidence is missing.")
    expected_files = contract.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise RuntimeError("Pilot checkpoint manifest is empty.")
    observed_files = []
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        observed_files.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest_sha256 = _json_hash(observed_files)
    if (
        observed_files != expected_files
        or contract.get("manifest_sha256") != manifest_sha256
        or contract.get("total_bytes")
        != sum(item["size_bytes"] for item in observed_files)
    ):
        raise RuntimeError("Pilot checkpoint no longer matches its bound manifest.")


def require_census(value: Any, *, method: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Metrics do not contain a trainable-parameter census.")
    expected_scope = "all-parameters" if method == "full" else "lora-adapter-only"
    expected_identity = {
        "schema_version": "aptus.trainable-parameter-census.v1",
        "method": method,
        "parameter_scope": expected_scope,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected_identity.items()):
        raise RuntimeError("Trainable-parameter census violates the selected method scope.")
    if value.get("all_values_finite") is not True:
        raise RuntimeError("Trainable-parameter census does not attest finite values.")
    for name in ("trainable_parameter_count", "trainable_tensor_count"):
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise RuntimeError(f"Trainable-parameter census requires positive {name}.")
    counter_names = (
        "frozen_parameter_count",
        "frozen_tensor_count",
        "unexpected_trainable_tensor_count",
        "expected_adapter_target_match_count",
        "adapter_target_instance_count",
        "incomplete_adapter_target_instance_count",
    )
    for name in counter_names:
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RuntimeError(f"Trainable-parameter census requires non-negative integer {name}.")
    if method == "full":
        if any(value[name] != 0 for name in counter_names):
            raise RuntimeError("Full fine-tuning census contains frozen or adapter counters.")
    else:
        for name in ("frozen_parameter_count", "frozen_tensor_count"):
            if value[name] <= 0:
                raise RuntimeError(
                    f"Adapter census requires positive {name} for its frozen base."
                )
        if value["unexpected_trainable_tensor_count"] != 0:
            raise RuntimeError("Adapter census contains an unexpected trainable tensor.")
        if (
            value["expected_adapter_target_match_count"] <= 0
            or value["adapter_target_instance_count"] != value["expected_adapter_target_match_count"]
            or value["incomplete_adapter_target_instance_count"] != 0
        ):
            raise RuntimeError("Adapter census does not bind one complete LoRA A/B pair to every target instance.")
    digest = value.get("descriptor_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("Trainable-parameter census has an invalid descriptor digest.")
    return value


def verify_pilot_artifacts(metrics: dict[str, Any]) -> tuple[int, int]:
    if metrics.get("checkpoint_continuation_observed") is not True:
        raise RuntimeError("Pilot metrics do not attest checkpoint continuation.")
    pilot_run_dir = metrics.get("pilot_run_dir")
    if not isinstance(pilot_run_dir, str):
        raise RuntimeError("Pilot metrics do not bind an immutable pilot run.")
    relative = Path(pilot_run_dir)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError("Pilot run path is unsafe.")
    pilot_root = (ROOT / relative).resolve()
    if (
        len(relative.parts) != 2
        or relative.parts[0] != "runs"
        or not relative.parts[1].startswith("pilot_")
        or pilot_root.parent != (ROOT / "runs").resolve()
        or metrics.get("pilot_run_id") != pilot_root.name
    ):
        raise RuntimeError("Pilot run path is not a bound Aptus pilot root.")
    marker_path = pilot_root / ".aptus-pilot-run.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Pilot-run ownership contract is missing or unreadable.") from error
    plan = load_plan()
    marker_expected = {
        "schema_version": "aptus.pilot-run.v1",
        "pilot_run_id": pilot_root.name,
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
    }
    if not isinstance(marker, dict) or any(
        marker.get(name) != value for name, value in marker_expected.items()
    ):
        raise RuntimeError("Pilot-run ownership contract does not match the plan.")
    contracts = (
        (pilot_root / "phase-1" / "checkpoint-1", metrics.get("phase_one_checkpoint")),
        (pilot_root / "phase-2" / "checkpoint-2", metrics.get("phase_two_checkpoint")),
    )
    for checkpoint, contract in contracts:
        if not isinstance(contract, dict):
            raise RuntimeError("Pilot checkpoint contract is missing.")
        verify_checkpoint_contract(checkpoint, contract)
    checkpoint_bytes = metrics.get("measured_checkpoint_bytes")
    final_export_bytes = metrics.get("measured_final_export_bytes")
    expected_checkpoint_bytes = max(
        int(contract["total_bytes"]) for _path, contract in contracts
    )
    phases = (metrics.get("phase_one"), metrics.get("phase_two_resumed"))
    if not all(isinstance(phase, dict) for phase in phases):
        raise RuntimeError("Pilot phase metrics are missing.")
    censuses = tuple(
        require_census(
            phase.get("trainable_parameter_census"),
            method=plan["recommended"]["method"],
        )
        for phase in phases
    )
    if censuses[0] != censuses[1]:
        raise RuntimeError("Pilot phases do not bind the same trainable parameter set.")
    try:
        expected_final_export_bytes = max(
            int(phase["final_export"]["total_bytes"]) for phase in phases
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Pilot final-export evidence is invalid.") from error
    if checkpoint_bytes != expected_checkpoint_bytes or final_export_bytes != expected_final_export_bytes:
        raise RuntimeError("Pilot capacity evidence is inconsistent.")
    if checkpoint_bytes <= 0 or final_export_bytes <= 0:
        raise RuntimeError("Pilot capacity evidence must be positive.")
    return checkpoint_bytes, final_export_bytes


def environment_binding() -> str:
    direct_constraints = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name = line.split("==", 1)[0]
        try:
            direct_constraints[name] = version(name)
        except PackageNotFoundError:
            direct_constraints[name] = "missing"
    runtime_distributions = runtime_distribution_closure(direct_constraints)
    return _json_hash({
        "python": platform.python_version(),
        "platform": platform.platform(),
        "direct_constraints": direct_constraints,
        "runtime_distributions": runtime_distributions,
    })


def runtime_distribution_closure(names: dict[str, str]) -> dict[str, str]:
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
                (token.find(character) for character in "[ (<>=!~" if character in token),
                default=len(token),
            )
            dependency = token[:boundary].strip()
            if dependency:
                pending.append(dependency)
    return dict(sorted(observed.items()))


def available_host_memory_bytes() -> int:
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
            raise RuntimeError("Windows host-memory inspection failed.")
        return int(status.available_physical)
    if not hasattr(os, "sysconf"):
        raise RuntimeError("Available host-memory inspection is unsupported.")
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (OSError, ValueError) as error:
        raise RuntimeError("Available host-memory inspection failed.") from error


def local_runtime_snapshot(world_size: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Pilot-bound CUDA hardware is no longer available.")
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
        raise RuntimeError("CUDA driver identity is unavailable for pilot binding.")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        device_uuid = str(getattr(properties, "uuid", "")).strip()
        if not device_uuid or device_uuid.lower() == "none":
            raise RuntimeError(f"CUDA device {index} does not expose a stable UUID for pilot binding.")
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "uuid": device_uuid,
                "pci_bus_id": str(getattr(properties, "pci_bus_id", "")),
                "total_vram_bytes": properties.total_memory,
                "compute_capability": f"{major}.{minor}",
            }
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("The planned CUDA world is not currently available.")
    hardware = {
        "cuda_runtime": torch.version.cuda,
        "driver_version": driver_version,
        "devices": devices,
    }
    return {
        "hardware_binding": _json_hash(hardware),
        "free_cuda_bytes": [
            int(torch.cuda.mem_get_info(index)[0]) for index in range(world_size)
        ],
        "host_ram_free_bytes": available_host_memory_bytes(),
    }


def runtime_snapshot(world_size: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--runtime-probe",
            "--world-size",
            str(world_size),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError("CUDA runtime authorization probe failed: " + detail)
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("CUDA runtime authorization probe returned invalid JSON.") from error
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("hardware_binding"), str)
        or not isinstance(value.get("free_cuda_bytes"), list)
        or len(value["free_cuda_bytes"]) != world_size
        or not isinstance(value.get("host_ram_free_bytes"), int)
        or value["host_ram_free_bytes"] <= 0
    ):
        raise RuntimeError("CUDA runtime authorization probe returned an invalid contract.")
    return value


def require_pilot_and_record_approval(plan: dict[str, Any], output_dir: Path) -> None:
    with report_lock():
        _require_pilot_and_record_approval(plan, output_dir)


def _require_pilot_and_record_approval(plan: dict[str, Any], output_dir: Path) -> None:
    manifest_errors = validate_bundle_manifest(ROOT)
    if manifest_errors:
        raise RuntimeError("Bundle changed after compilation: " + " | ".join(manifest_errors))
    report_path = ROOT / "validation-report.json"
    if not report_path.is_file():
        raise RuntimeError("Full training requires `python validate.py --level pilot` first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError("Pilot validation report must be a JSON object.")
    if report.get("state") not in {"pilot-pass", "execution-approved", "measured-run-pass"}:
        raise RuntimeError("Full training requires a passing real-model/data pilot.")
    bindings = report.get("bindings", {})
    if not isinstance(bindings, dict):
        raise RuntimeError("Pilot validation bindings must be a JSON object.")
    metrics_path = ROOT / "pilot-output" / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("Pilot attestation is missing metrics.json.")
    world_size = int(plan["recommended"].get("world_size", 1))
    current_runtime = runtime_snapshot(world_size)
    expected = {
        "bundle": _sha256(ROOT / "bundle-manifest.json"),
        "dataset": plan["dataset"]["source_sha256"],
        "model_revision": plan["model"]["revision"],
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "environment": environment_binding(),
        "hardware": current_runtime["hardware_binding"],
        "pilot_metrics": _sha256(metrics_path),
    }
    mismatches = [name for name, value in expected.items() if bindings.get(name) != value]
    if mismatches:
        raise RuntimeError("Pilot attestation does not bind the current " + ", ".join(mismatches) + ".")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise RuntimeError("Pilot metrics must be a JSON object.")
    measured_checkpoint_bytes, measured_final_export_bytes = verify_pilot_artifacts(metrics)
    peaks = []
    for phase_name in ("phase_one", "phase_two_resumed"):
        phase = metrics.get(phase_name)
        if not isinstance(phase, dict):
            raise RuntimeError(f"Pilot metrics require object {phase_name}.")
        per_rank = phase.get("per_rank_cuda_peaks")
        if world_size > 1 and (
            not isinstance(per_rank, list) or len(per_rank) != world_size
        ):
            raise RuntimeError(
                f"Pilot metrics {phase_name} must bind one CUDA peak record per distributed rank."
            )
        values = per_rank if isinstance(per_rank, list) and per_rank else [phase]
        for rank_index, value in enumerate(values):
            if not isinstance(value, dict):
                raise RuntimeError(
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
                    raise RuntimeError(
                        f"Pilot metric {phase_name}.{metric_name} must be a non-negative integer."
                    )
            if reserved < allocated:
                raise RuntimeError(
                    f"Pilot metric {phase_name} reserved CUDA memory is below allocated memory."
                )
            peaks.append(max(reserved, allocated))
    measured_peak = max(peaks, default=0)
    if measured_peak <= 0:
        raise RuntimeError("Pilot metrics do not contain a measured CUDA peak.")
    reserve = int(plan.get("hardware", {}).get("reserve_per_device_bytes", 0))
    required_free = measured_peak + reserve
    free_by_device = current_runtime["free_cuda_bytes"]
    insufficient = [index for index, free in enumerate(free_by_device) if free < required_free]
    if insufficient:
        raise RuntimeError(
            "Current free VRAM is below the pilot peak plus reserve on CUDA device(s): "
            + ", ".join(str(value) for value in insufficient)
        )
    required_host_ram = int(plan["recommended"].get("required_host_ram_bytes", 0))
    host_ram_free = int(current_runtime["host_ram_free_bytes"])
    if required_host_ram <= 0 or host_ram_free < required_host_ram:
        raise RuntimeError(
            "Current free host RAM is below the plan-bound distributed loading requirement."
        )
    checkpoint_bytes = measured_checkpoint_bytes * 4
    final_export_bytes = measured_final_export_bytes
    required_output_disk = checkpoint_bytes + final_export_bytes
    disk_probe = output_dir.expanduser().resolve()
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    disk_free = shutil.disk_usage(disk_probe).free
    if required_output_disk > 0 and disk_free < required_output_disk:
        raise RuntimeError("Current free disk is below the measured four-checkpoint transient and final-export requirement.")
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        for stale_name in (
            "measured_run_completed_at",
            "measured_run_pending_at",
            "final_export",
            "measured_run",
            "pending_final_export",
            "pending_measured_run",
        ):
            report.pop(stale_name, None)
        report["state"] = "execution-approved"
        report["execution_approved_at"] = datetime.now(timezone.utc).isoformat()
        report["active_run"] = {
            "output_dir": str(output_dir.resolve()),
            "run_id": output_dir.name,
            "plan_id": plan["plan_id"],
            "candidate_id": plan["recommended"]["candidate_id"],
        }
        report["prelaunch_capacity_check"] = {
            "measured_peak_cuda_bytes": measured_peak,
            "required_free_cuda_bytes": required_free,
            "free_cuda_bytes": free_by_device,
            "required_host_ram_bytes": required_host_ram,
            "host_ram_free_bytes": host_ram_free,
            "required_checkpoint_disk_bytes": checkpoint_bytes,
            "required_final_export_disk_bytes": final_export_bytes,
            "required_training_output_disk_bytes": required_output_disk,
            "checkpoint_basis": "4 * maximum measured pilot checkpoint bytes",
            "final_export_basis": "maximum measured pilot final export bytes",
            "free_disk_bytes": disk_free,
            "output_filesystem_probe": str(disk_probe),
        }
        temporary = report_path.with_name(".validation-report.json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)


def collectively_require_full_train_approval(
    plan: dict[str, Any], output_dir: Path
) -> None:
    import torch

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    result: list[str | None] = [None]
    if rank == 0:
        try:
            require_pilot_and_record_approval(plan, output_dir)
        except Exception as error:
            result[0] = str(error)
    if distributed:
        torch.distributed.broadcast_object_list(result, src=0)
    if result[0] is not None:
        raise RuntimeError("Full-training authorization failed: " + result[0])
    if distributed:
        torch.distributed.barrier()


def record_measured_run(output_dir: Path, plan: dict[str, Any]) -> None:
    with report_lock():
        _record_measured_run(output_dir, plan)


def _record_measured_run(output_dir: Path, plan: dict[str, Any]) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    candidate = plan["recommended"]
    report_path = ROOT / "validation-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("state") != "execution-approved":
        raise RuntimeError("Measured run completion cannot follow an unapproved execution.")
    active_run = report.get("active_run")
    expected_active_run = {
        "output_dir": str(output_dir.resolve()),
        "run_id": output_dir.name,
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
    }
    if not isinstance(active_run, dict) or any(
        active_run.get(name) != value for name, value in expected_active_run.items()
    ):
        raise RuntimeError("Measured run does not match the active execution approval.")
    export_path = output_dir / "final-export.json"
    if not export_path.is_file():
        raise RuntimeError("Measured run completion requires final-export.json.")
    expected_export = json.loads(export_path.read_text(encoding="utf-8"))
    actual_export = verify_final_export(
        output_dir / "final", candidate, plan["model"]
    )
    if expected_export != actual_export:
        raise RuntimeError("Final export changed before measured-run attestation.")
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("Measured run completion requires metrics.json.")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Measured run metrics are unreadable.") from error
    if not isinstance(metrics, dict):
        raise RuntimeError("Measured run metrics must be a JSON object.")
    assert_measured_training_metrics(metrics, candidate=candidate, pilot=False)
    if metrics.get("final_export") != actual_export:
        raise RuntimeError("Measured run metrics do not bind the verified final export.")
    report["measured_run_pending_at"] = datetime.now(timezone.utc).isoformat()
    report["pending_final_export"] = {
        "path": str((output_dir / "final").resolve()),
        "manifest_sha256": _sha256(export_path),
        "total_bytes": actual_export["total_bytes"],
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "distribution": candidate["distribution"],
        "world_size": candidate["world_size"],
    }
    report["pending_measured_run"] = {
        "output_dir": str(output_dir.resolve()),
        "metrics_sha256": _sha256(metrics_path),
        "global_step": metrics["global_step"],
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "distribution": metrics["distribution"],
        "world_size": metrics["actual_world_size"],
        "per_rank_cuda_peaks": metrics["per_rank_cuda_peaks"],
    }
    temporary = report_path.with_name(".validation-report.json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)


def collectively_record_measured_run(
    output_dir: Path, plan: dict[str, Any]
) -> None:
    import torch

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    result: list[str | None] = [None]
    if rank == 0:
        try:
            record_measured_run(output_dir, plan)
        except Exception as error:
            result[0] = str(error)
    if distributed:
        torch.distributed.broadcast_object_list(result, src=0)
    if result[0] is not None:
        raise RuntimeError("Measured-run attestation failed: " + result[0])
    if distributed:
        torch.distributed.barrier()


def promote_pending_run(output_dir: Path, plan: dict[str, Any]) -> None:
    output_dir = output_dir.expanduser().resolve(strict=True)
    candidate = plan["recommended"]
    with report_lock():
        report_path = ROOT / "validation-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("state") != "execution-approved":
            raise RuntimeError("No execution-approved pending run can be promoted.")
        export_path = output_dir / "final-export.json"
        metrics_path = output_dir / "metrics.json"
        expected_export = json.loads(export_path.read_text(encoding="utf-8"))
        actual_export = verify_final_export(
            output_dir / "final", candidate, plan["model"]
        )
        if expected_export != actual_export:
            raise RuntimeError("Pending final export changed before promotion.")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise RuntimeError("Pending measured-run metrics must be an object.")
        assert_measured_training_metrics(metrics, candidate=candidate, pilot=False)
        if metrics.get("final_export") != actual_export:
            raise RuntimeError("Pending metrics do not bind the final export.")
        expected_active = {
            "output_dir": str(output_dir),
            "run_id": output_dir.name,
            "plan_id": plan["plan_id"],
            "candidate_id": candidate["candidate_id"],
        }
        expected_final = {
            "path": str((output_dir / "final").resolve()),
            "manifest_sha256": _sha256(export_path),
            "total_bytes": actual_export["total_bytes"],
            "plan_id": plan["plan_id"],
            "candidate_id": candidate["candidate_id"],
            "distribution": candidate["distribution"],
            "world_size": candidate["world_size"],
        }
        expected_measured = {
            "output_dir": str(output_dir),
            "metrics_sha256": _sha256(metrics_path),
            "global_step": metrics["global_step"],
            "plan_id": plan["plan_id"],
            "candidate_id": candidate["candidate_id"],
            "distribution": metrics["distribution"],
            "world_size": metrics["actual_world_size"],
            "per_rank_cuda_peaks": metrics["per_rank_cuda_peaks"],
        }
        if (
            report.get("active_run") != expected_active
            or report.get("pending_final_export") != expected_final
            or report.get("pending_measured_run") != expected_measured
        ):
            raise RuntimeError("Pending report evidence does not match the run artifacts.")
        report["state"] = "measured-run-pass"
        report["measured_run_completed_at"] = datetime.now(timezone.utc).isoformat()
        report["final_export"] = expected_final
        report["measured_run"] = expected_measured
        for name in (
            "active_run",
            "measured_run_pending_at",
            "pending_final_export",
            "pending_measured_run",
        ):
            report.pop(name, None)
        temporary = report_path.with_name(".validation-report.json.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, report_path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("train", [value])
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
    elif suffix == ".txt":
        rows = [{"text": line.rstrip("\n")} for line in path.read_text(encoding="utf-8").splitlines()]
    else:
        raise ValueError(f"Unsupported dataset format: {suffix or '<none>'}")
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Dataset must contain at least one object row.")
    filtered = [row for row in rows if not is_profiler_ignored_empty(row)]
    if not filtered:
        raise ValueError("Dataset contains no non-empty supported rows.")
    return filtered


def is_profiler_ignored_empty(record: dict[str, Any]) -> bool:
    text = record.get("text")
    if isinstance(text, str):
        return not text.strip()
    prompt, completion = record.get("prompt"), record.get("completion")
    if isinstance(prompt, str) or isinstance(completion, str):
        return not (
            isinstance(prompt, str)
            and isinstance(completion, str)
            and completion.strip()
        )
    instruction, output = record.get("instruction"), record.get("output")
    if isinstance(instruction, str) or isinstance(output, str):
        return not (
            isinstance(instruction, str)
            and isinstance(output, str)
            and output.strip()
        )
    messages = record.get("messages")
    if isinstance(messages, list):
        return not messages
    content = record.get("content")
    if isinstance(content, str):
        return not content.strip()
    return False


def record_to_parts(record: dict[str, Any], tokenizer: Any) -> tuple[str, str, bool]:
    """Return prompt, supervised completion, and whether the whole text is supervised."""
    if isinstance(record.get("text"), str) and record["text"].strip():
        return "", record["text"], True
    prompt, completion = record.get("prompt"), record.get("completion")
    if isinstance(prompt, str) or isinstance(completion, str):
        if not isinstance(prompt, str) or not isinstance(completion, str) or not completion.strip():
            raise ValueError("Prompt/completion rows require a non-empty completion.")
        return prompt, completion, False
    instruction, output = record.get("instruction"), record.get("output")
    if isinstance(instruction, str) or isinstance(output, str):
        if not isinstance(instruction, str) or not isinstance(output, str) or not output.strip():
            raise ValueError("Instruction/output rows require a non-empty output.")
        prompt = "### Instruction:\n" + instruction.strip() + "\n"
        if isinstance(record.get("input"), str) and record["input"].strip():
            prompt += "### Input:\n" + record["input"].strip() + "\n"
        return prompt + "### Response:\n", output, False
    messages = record.get("messages")
    if isinstance(messages, list):
        if not messages:
            raise ValueError("Messages rows require at least one message.")
        if not all(isinstance(item, dict) for item in messages):
            raise ValueError("Every message must be an object.")
        if (
            messages[-1].get("role") != "assistant"
            or not isinstance(messages[-1].get("content"), str)
            or not messages[-1]["content"].strip()
        ):
            raise ValueError("A messages row must end with a non-empty assistant message.")
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        return prompt, messages[-1]["content"], False
    if isinstance(record.get("content"), str) and record["content"].strip():
        return "", record["content"], True
    raise ValueError("Row does not match text, prompt/completion, instruction/output, messages, or content schema.")


def encode_record(record: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    messages = record.get("messages")
    prompt_value, completion_value = record.get("prompt"), record.get("completion")
    instruction_value, output_value = record.get("instruction"), record.get("output")
    uses_messages_schema = (
        not isinstance(record.get("text"), str)
        and not (isinstance(prompt_value, str) or isinstance(completion_value, str))
        and not (isinstance(instruction_value, str) or isinstance(output_value, str))
        and isinstance(messages, list)
    )
    if uses_messages_schema and messages:
        # Validate the final assistant turn, then preserve the tokenizer's complete chat template.
        record_to_parts(record, tokenizer)
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if full_ids[: len(prompt_ids)] == prompt_ids and len(full_ids) > len(prompt_ids):
            completion_ids = full_ids[len(prompt_ids) :][:max_length]
            prompt_budget = max(0, max_length - len(completion_ids))
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            if input_ids and any(label != -100 for label in labels):
                return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
        raise ValueError(
            "The pinned tokenizer chat template is not prefix-separable for assistant-only masking. Aptus refuses to alter its control-token format."
        )
    prompt, completion, supervise_all = record_to_parts(record, tokenizer)
    if supervise_all:
        input_ids = tokenizer.encode(completion, add_special_tokens=True)[:max_length]
        labels = list(input_ids)
    else:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        eos = tokenizer.eos_token_id
        if eos is not None and (not completion_ids or completion_ids[-1] != eos):
            completion_ids.append(eos)
        completion_ids = completion_ids[:max_length]
        prompt_budget = max(0, max_length - len(completion_ids))
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
    if not input_ids or not any(label != -100 for label in labels):
        raise ValueError("Tokenized row contains no supervised tokens.")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def require_hardware_parity(plan: dict[str, Any]) -> None:
    import torch

    candidate = plan["recommended"]
    planned_devices = plan["hardware"]["devices"]
    world_size = int(candidate["world_size"])
    device_indices = candidate.get("device_indices", list(range(world_size)))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= world_size:
        raise RuntimeError("LOCAL_RANK is outside the selected Aptus world.")
    physical_device_index = int(device_indices[local_rank])
    logical_device_index = local_rank
    if not torch.cuda.is_available():
        raise RuntimeError("The selected execution candidate requires CUDA, but CUDA is unavailable.")
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    actual_world_size = torch.distributed.get_world_size() if distributed else 1
    environment_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if candidate["distribution"] == "single":
        if actual_world_size != 1 or environment_world_size != 1:
            raise RuntimeError("A single-device candidate cannot run inside a multi-process world.")
    elif not distributed:
        raise RuntimeError(
            f"The {candidate['distribution']} candidate requires an initialized distributed process group."
        )
    elif actual_world_size != world_size or environment_world_size != world_size:
        raise RuntimeError("The actual distributed world size does not match the plan.")
    if torch.cuda.device_count() <= logical_device_index:
        raise RuntimeError("A selected CUDA device is not visible to this worker.")
    reserve = int(plan["hardware"].get("reserve_per_device_bytes", 0))
    required_free = int(candidate["memory"]["point_estimate_bytes"]) + reserve
    properties = torch.cuda.get_device_properties(logical_device_index)
    planned = planned_devices[physical_device_index]
    if int(properties.total_memory) < int(planned["total_vram_bytes"]):
        raise RuntimeError(
            f"CUDA device {physical_device_index} has less total VRAM than the plan-bound hardware fact."
        )
    free_bytes = int(torch.cuda.mem_get_info(logical_device_index)[0])
    if free_bytes < required_free:
        raise RuntimeError(
            f"CUDA device {physical_device_index} free VRAM is below the candidate point estimate plus user reserve."
        )
    capability = tuple(torch.cuda.get_device_capability(logical_device_index))
    if candidate["method"] == "qlora" and capability < (6, 0):
        raise RuntimeError(
            f"CUDA device {physical_device_index} lacks the 6.0-or-newer compute capability required by NF4/FP4 bitsandbytes kernels."
        )
    if candidate["method"] == "int8-lora" and capability < (7, 5):
        raise RuntimeError(
            f"CUDA device {physical_device_index} lacks the 7.5-or-newer compute capability required by the LLM.int8 bitsandbytes path."
        )
    if candidate["precision"] == "bf16":
        with torch.cuda.device(logical_device_index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    f"CUDA device {physical_device_index} does not support the plan-bound BF16 precision."
                )


def initialize_and_require_strategy(plan: dict[str, Any]) -> Any:
    from accelerate import DistributedType, PartialState

    state = PartialState()
    distribution = plan["recommended"]["distribution"]
    expected = {
        "single": DistributedType.NO,
        "ddp": DistributedType.MULTI_GPU,
        "fsdp": DistributedType.MULTI_GPU,
    }[distribution]
    if state.distributed_type != expected:
        raise RuntimeError(
            "The initialized Accelerate strategy does not match the selected Aptus distribution."
        )
    fsdp_enabled = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
    if fsdp_enabled != (distribution == "fsdp"):
        raise RuntimeError("The Accelerate FSDP launcher mode does not match the plan.")
    if distribution == "fsdp":
        expected_environment = {
            "FSDP_VERSION": "1",
            "FSDP_SHARDING_STRATEGY": "FULL_SHARD",
            "FSDP_OFFLOAD_PARAMS": "false",
            "FSDP_AUTO_WRAP_POLICY": "TRANSFORMER_BASED_WRAP",
            "FSDP_BACKWARD_PREFETCH": "BACKWARD_PRE",
            "FSDP_STATE_DICT_TYPE": "SHARDED_STATE_DICT",
            "FSDP_USE_ORIG_PARAMS": "true",
            "FSDP_CPU_RAM_EFFICIENT_LOADING": "true",
            "FSDP_SYNC_MODULE_STATES": "true",
        }
        mismatches = [
            name
            for name, value in expected_environment.items()
            if os.environ.get(name, "").lower() != value.lower()
        ]
        if mismatches:
            raise RuntimeError(
                "The Accelerate FSDP launcher policy does not match the compiled contract: "
                + ", ".join(mismatches)
            )
    return state


def require_trainer_strategy(trainer: Any, plan: dict[str, Any]) -> None:
    from accelerate import DistributedType

    expected = {
        "single": DistributedType.NO,
        "ddp": DistributedType.MULTI_GPU,
        "fsdp": DistributedType.FSDP,
    }[plan["recommended"]["distribution"]]
    if trainer.accelerator.distributed_type != expected:
        raise RuntimeError(
            "Trainer initialized a distribution strategy that differs from the Aptus plan."
        )


def verify_loaded_config_contract(config: Any, plan: dict[str, Any]) -> None:
    model_facts = plan["model"]
    aliases = {
        "hidden_size": ("hidden_size", "d_model", "n_embd"),
        "layers": ("num_hidden_layers", "num_layers", "n_layer"),
        "context_length": (
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "n_positions",
        ),
        "intermediate_size": ("intermediate_size", "ffn_dim", "n_inner"),
    }
    required = ["hidden_size", "layers", "context_length"]
    if model_facts.get("intermediate_size") is not None:
        required.append("intermediate_size")
    unavailable: list[str] = []
    mismatches: list[str] = []
    conflicts: list[str] = []
    for fact_name in required:
        observed = {
            alias: value
            for alias in aliases[fact_name]
            if isinstance((value := getattr(config, alias, None)), int)
            and not isinstance(value, bool)
        }
        if not observed:
            unavailable.append(fact_name)
            continue
        values = set(observed.values())
        if len(values) != 1:
            conflicts.append(fact_name)
            continue
        if values.pop() != int(model_facts[fact_name]):
            mismatches.append(fact_name)
    if unavailable or mismatches or conflicts:
        details = []
        if unavailable:
            details.append("unavailable: " + ", ".join(unavailable))
        if mismatches:
            details.append("mismatched: " + ", ".join(mismatches))
        if conflicts:
            details.append("conflicting aliases: " + ", ".join(conflicts))
        raise RuntimeError(
            "Pinned model config does not match the plan-bound structural facts ("
            + "; ".join(details)
            + ")."
        )


def _compiled_lora_binding(
    name: str, *, target_modules: tuple[str, ...]
) -> tuple[str, str] | None:
    parts = name.split(".")
    component_indices = [
        index for index, part in enumerate(parts) if part in {"lora_A", "lora_B"}
    ]
    if len(component_indices) != 1:
        return None
    index = component_indices[0]
    if (
        index == 0
        or parts[index - 1] not in target_modules
        or parts[index + 1 :] != ["default", "weight"]
    ):
        return None
    return ".".join(parts[:index]), parts[index]


def trainable_parameter_census(
    model: Any,
    *,
    method: str,
    target_modules: tuple[str, ...],
    expected_adapter_target_match_count: int,
) -> dict[str, Any]:
    """Describe trainable tensors without recording names or parameter values."""
    import torch

    if method not in {"full", "lora", "int8-lora", "qlora"}:
        raise RuntimeError("The selected method has no trainable-scope contract.")
    descriptors: list[tuple[str, tuple[int, ...], str]] = []
    trainable_parameter_count = 0
    frozen_parameter_count = 0
    frozen_tensor_count = 0
    unexpected_trainable_tensor_count = 0
    all_values_finite = True
    observed_names: set[str] = set()
    adapter_components: dict[str, set[str]] = {}
    if method == "full":
        if target_modules or expected_adapter_target_match_count != 0:
            raise RuntimeError("Full fine-tuning cannot declare adapter target matches.")
    elif not target_modules or expected_adapter_target_match_count <= 0:
        raise RuntimeError("Adapter census requires plan-bound target matches.")
    for name, parameter in model.named_parameters():
        if not isinstance(name, str) or not name:
            raise RuntimeError("A model parameter has no stable name.")
        if name in observed_names:
            raise RuntimeError("Model parameter names are not unique.")
        observed_names.add(name)
        try:
            shape = tuple(int(dimension) for dimension in parameter.shape)
            parameter_count = int(parameter.numel())
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "A trainable parameter has an invalid shape or element count."
            ) from error
        if parameter_count < 0 or any(dimension < 0 for dimension in shape):
            raise RuntimeError(
                "A trainable parameter has an invalid shape or element count."
            )
        if not parameter.requires_grad:
            frozen_parameter_count += parameter_count
            frozen_tensor_count += 1
            continue
        if method != "full":
            binding = _compiled_lora_binding(name, target_modules=target_modules)
            if binding is None:
                unexpected_trainable_tensor_count += 1
            else:
                parent, component = binding
                observed_components = adapter_components.setdefault(parent, set())
                if component in observed_components:
                    unexpected_trainable_tensor_count += 1
                observed_components.add(component)
        trainable_parameter_count += parameter_count
        descriptors.append((name, shape, str(parameter.dtype)))
        if parameter_count:
            try:
                finite = bool(torch.isfinite(parameter.detach()).all().item())
            except Exception as error:
                raise RuntimeError(
                    "A trainable parameter could not be checked for finite values."
                ) from error
            all_values_finite = all_values_finite and finite

    digest = hashlib.sha256()
    digest.update(b"aptus.trainable-parameter-descriptors.v1\n")
    for name, shape, dtype in sorted(descriptors, key=lambda item: item[0]):
        digest.update(
            (
                json.dumps(
                    [name, list(shape), dtype],
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
    scope = "all-parameters" if method == "full" else "lora-adapter-only"
    incomplete_adapter_target_instance_count = sum(
        components != {"lora_A", "lora_B"}
        for components in adapter_components.values()
    )
    return {
        "schema_version": "aptus.trainable-parameter-census.v1",
        "method": method,
        "parameter_scope": scope,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_tensor_count": len(descriptors),
        "frozen_parameter_count": frozen_parameter_count,
        "frozen_tensor_count": frozen_tensor_count,
        "unexpected_trainable_tensor_count": unexpected_trainable_tensor_count,
        "expected_adapter_target_match_count": expected_adapter_target_match_count,
        "adapter_target_instance_count": len(adapter_components),
        "incomplete_adapter_target_instance_count": incomplete_adapter_target_instance_count,
        "all_values_finite": all_values_finite,
        "descriptor_sha256": digest.hexdigest(),
    }


def require_trainable_parameter_census(
    model: Any,
    *,
    method: str,
    target_modules: tuple[str, ...],
    expected_adapter_target_match_count: int,
) -> dict[str, Any]:
    census = trainable_parameter_census(
        model,
        method=method,
        target_modules=target_modules,
        expected_adapter_target_match_count=expected_adapter_target_match_count,
    )
    if census["trainable_parameter_count"] <= 0 or census["trainable_tensor_count"] <= 0:
        raise RuntimeError(
            "Method preparation selected zero trainable parameters; training is refused."
        )
    if census["all_values_finite"] is not True:
        raise FloatingPointError(
            "Method preparation produced non-finite trainable parameter values."
        )
    if method == "full" and census["frozen_tensor_count"]:
        raise RuntimeError(
            "Full fine-tuning left one or more model parameters frozen; training is refused."
        )
    if method != "full" and census["unexpected_trainable_tensor_count"]:
        raise RuntimeError(
            "Adapter preparation left non-LoRA model parameters trainable; training is refused."
        )
    if method != "full" and (
        census["frozen_parameter_count"] <= 0 or census["frozen_tensor_count"] <= 0
    ):
        raise RuntimeError(
            "Adapter preparation did not retain a non-empty frozen base; training is refused."
        )
    if method != "full" and (
        census["adapter_target_instance_count"]
        != census["expected_adapter_target_match_count"]
        or census["incomplete_adapter_target_instance_count"] != 0
    ):
        raise RuntimeError(
            "Adapter preparation does not contain exactly one LoRA A/B pair for every plan-bound target instance."
        )
    return census


def prepare_model_for_training(model: Any, plan: dict[str, Any]) -> Any:
    candidate = plan["recommended"]
    model.config.use_cache = False
    use_reentrant_checkpointing = (
        candidate["method"] == "qlora" and candidate["distribution"] == "single"
    )
    checkpointing_kwargs = {"use_reentrant": use_reentrant_checkpointing}
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=checkpointing_kwargs
    )

    if candidate["method"] in {"int8-lora", "qlora"}:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs=checkpointing_kwargs,
        )
    if candidate["method"] != "full":
        from peft import LoraConfig, get_peft_model

        adapter = LoraConfig(
            r=candidate["rank"],
            lora_alpha=candidate["alpha"],
            lora_dropout=0.05,
            target_modules=candidate["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(
            model,
            adapter,
            revision=plan["model"]["revision"],
        )
    return model


def model_data_preflight(
    plan: dict[str, Any],
    trainer_config: dict[str, Any],
    *,
    local_files_only: bool,
) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    candidate = plan["recommended"]
    initialize_and_require_strategy(plan)
    require_hardware_parity(plan)
    if candidate["method"] in {"int8-lora", "qlora"}:
        import bitsandbytes  # noqa: F401
    model = plan["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model.get("tokenizer_id") or model["model_id"],
        revision=model["revision"],
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    config = AutoConfig.from_pretrained(
        model["model_id"],
        revision=model["revision"],
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    verify_loaded_config_contract(config, plan)
    loaded_model = None
    try:
        loaded_model = load_pinned_base_model(
            plan, local_files_only=local_files_only
        )
        actual_parameter_count, target_match_count = verify_loaded_model_contract(
            loaded_model, plan
        )
        loaded_model = prepare_model_for_training(loaded_model, plan)
        trainable_census = require_trainable_parameter_census(
            loaded_model,
            method=candidate["method"],
            target_modules=tuple(candidate.get("target_modules", ())),
            expected_adapter_target_match_count=target_match_count,
        )
    finally:
        loaded_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    row_count = 0
    with (ROOT / trainer_config["training_dataset_path"]).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Canonical training row {line_number} must be an object.")
            encode_record(row, tokenizer, trainer_config["sequence_length"])
            row_count += 1
    if row_count == 0:
        raise ValueError("Canonical training dataset contains no rows.")
    print(
        "Selected model revision, loaded weight structure, "
        f"{actual_parameter_count} parameters, {target_match_count} adapter target matches, "
        f"tokenizer, method capability, {trainable_census['trainable_parameter_count']} "
        f"finite trainable parameters ({trainable_census['descriptor_sha256']}), "
        f"and all {row_count} canonical training rows passed."
    )
    return trainable_census


def synthetic_preflight(plan: dict[str, Any]) -> None:
    """Exercise the selected method without claiming the real model or data ran."""
    import torch

    initialize_and_require_strategy(plan)
    require_hardware_parity(plan)
    from transformers import LlamaConfig, LlamaForCausalLM

    candidate = plan["recommended"]
    compute_dtype = (
        torch.bfloat16 if candidate["precision"] == "bf16" else torch.float16
    )
    torch.cuda.reset_peak_memory_stats()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_indices = candidate.get(
        "device_indices", list(range(int(candidate["world_size"])))
    )
    device = torch.device("cuda", local_rank)
    if candidate["method"] in {"int8-lora", "qlora"}:
        import bitsandbytes as bnb

        if candidate["method"] == "int8-lora":
            layer = bnb.nn.Linear8bitLt(32, 32, has_fp16_weights=False).to(device)
        else:
            layer = bnb.nn.Linear4bit(
                32,
                32,
                quant_type="nf4",
                compress_statistics=True,
                compute_dtype=compute_dtype,
            ).to(device)
        value = torch.randn(
            2, 32, device=device, dtype=compute_dtype, requires_grad=True
        )
        kernel_output = layer(value).float()
        if not torch.isfinite(kernel_output).all():
            raise RuntimeError("bitsandbytes kernel output is not finite.")
        kernel_output.sum().backward()

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).to(device=device, dtype=compute_dtype)
    synthetic_target_modules: tuple[str, ...] = ()
    synthetic_target_match_count = 0
    if candidate["method"] != "full":
        from peft import LoraConfig, get_peft_model

        synthetic_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
        synthetic_target_match_count = sum(
            1
            for name, _module in model.named_modules()
            if any(
                name == target or name.endswith("." + target)
                for target in synthetic_target_modules
            )
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=min(8, candidate["rank"]),
                lora_alpha=min(16, candidate["alpha"]),
                target_modules=list(synthetic_target_modules),
                task_type="CAUSAL_LM",
            ),
        )
    trainable_census = require_trainable_parameter_census(
        model,
        method=candidate["method"],
        target_modules=synthetic_target_modules,
        expected_adapter_target_match_count=synthetic_target_match_count,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    ids = torch.randint(0, 128, (2, 16), device=device)
    loss = model(input_ids=ids, labels=ids).loss
    if not torch.isfinite(loss):
        raise RuntimeError("Synthetic method preflight loss is not finite.")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    measured_peak = int(torch.cuda.max_memory_allocated(device))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        measured_peak_tensor = torch.tensor(
            measured_peak, device=device, dtype=torch.int64
        )
        torch.distributed.all_reduce(
            measured_peak_tensor, op=torch.distributed.ReduceOp.MAX
        )
        measured_peak = int(measured_peak_tensor.item())
    metrics = {
        "schema_version": "aptus.preflight-metrics.v1",
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        "precision": candidate["precision"],
        "quantization": candidate.get("quantization"),
        "distribution": candidate["distribution"],
        "world_size": candidate["world_size"],
        "measured_peak_cuda_bytes": measured_peak,
        "trainable_parameter_census": trainable_census,
        "scope": "synthetic-method-preflight-not-model-data-pilot",
    }
    if local_rank == 0:
        metrics_path = ROOT / "preflight-metrics.json"
        temporary = metrics_path.with_name(".preflight-metrics.json.tmp")
        temporary.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, metrics_path)
        print(json.dumps(metrics, indent=2, sort_keys=True))


def runtime_parameter_count(model: Any) -> int:
    count = 0
    for parameter in model.parameters():
        quant_state = getattr(parameter, "quant_state", None)
        original_shape = getattr(quant_state, "shape", None)
        if original_shape is not None:
            size = 1
            for dimension in original_shape:
                size *= int(dimension)
            count += size
        else:
            count += int(parameter.numel())
    return count


def base_model_load_kwargs(
    plan: dict[str, Any], *, local_files_only: bool
) -> dict[str, Any]:
    import torch
    from transformers import BitsAndBytesConfig

    candidate = plan["recommended"]
    dtype = torch.bfloat16 if candidate["precision"] == "bf16" else torch.float16
    quantization = None
    if candidate["method"] == "qlora":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif candidate["method"] == "int8-lora":
        quantization = BitsAndBytesConfig(load_in_8bit=True)

    model_spec = plan["model"]
    model_kwargs: dict[str, Any] = {
        "revision": model_spec["revision"],
        "trust_remote_code": False,
        "dtype": dtype,
        "local_files_only": local_files_only,
    }
    if quantization is not None:
        model_kwargs["quantization_config"] = quantization
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    return model_kwargs


def load_pinned_base_model(plan: dict[str, Any], *, local_files_only: bool) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        plan["model"]["model_id"],
        **base_model_load_kwargs(plan, local_files_only=local_files_only),
    )


def verify_loaded_model_contract(
    loaded_model: Any, plan: dict[str, Any]
) -> tuple[int, int]:
    candidate = plan["recommended"]
    model_spec = plan["model"]
    actual_parameter_count = runtime_parameter_count(loaded_model)
    expected_parameter_count = int(model_spec["parameters"])
    tolerance = max(1_000_000, round(expected_parameter_count * 0.02))
    if abs(actual_parameter_count - expected_parameter_count) > tolerance:
        raise RuntimeError(
            "Loaded model parameter count differs from the plan-bound declaration by more than the explicit 2% or one-million-parameter tolerance."
        )

    target_modules = tuple(candidate.get("target_modules", ()))
    if candidate["method"] == "full":
        if target_modules:
            raise RuntimeError("Full fine-tuning must not declare adapter target modules.")
        return actual_parameter_count, 0
    if not target_modules:
        raise RuntimeError("The selected adapter method has no target modules.")

    module_names = tuple(name for name, _module in loaded_model.named_modules())
    matches = {
        target: tuple(
            name
            for name in module_names
            if name == target or name.endswith("." + target)
        )
        for target in target_modules
    }
    missing = sorted(target for target, names in matches.items() if not names)
    if missing:
        raise RuntimeError(
            "Loaded model is missing the plan-bound adapter target module(s): "
            + ", ".join(missing)
        )
    return actual_parameter_count, sum(len(names) for names in matches.values())


def build_model(
    plan: dict[str, Any], *, local_files_only: bool
) -> tuple[Any, Any, int, dict[str, Any]]:
    from transformers import AutoTokenizer

    model_spec = plan["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.get("tokenizer_id") or model_spec["model_id"],
        revision=model_spec["revision"],
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS or padding token.")
        tokenizer.pad_token = tokenizer.eos_token

    model = load_pinned_base_model(plan, local_files_only=local_files_only)
    actual_parameter_count, target_match_count = verify_loaded_model_contract(
        model, plan
    )
    model = prepare_model_for_training(model, plan)
    trainable_census = require_trainable_parameter_census(
        model,
        method=plan["recommended"]["method"],
        target_modules=tuple(plan["recommended"].get("target_modules", ())),
        expected_adapter_target_match_count=target_match_count,
    )
    return model, tokenizer, actual_parameter_count, trainable_census


def resolve_max_steps(*, pilot: bool, max_steps: int | None) -> int:
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when supplied.")
    if max_steps is not None:
        return max_steps
    return 1 if pilot else -1


def default_output_dir(plan: dict[str, Any], *, pilot: bool) -> Path:
    if pilot:
        raise RuntimeError("Pilot output paths are assigned by preflight.py.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        shared_launch = "|".join(
            (
                os.environ.get("TORCHELASTIC_RUN_ID", ""),
                os.environ.get("MASTER_ADDR", ""),
                os.environ.get("MASTER_PORT", ""),
                str(os.getppid()),
            )
        )
        launch_id = hashlib.sha256(shared_launch.encode("utf-8")).hexdigest()[:16]
        return ROOT / "runs" / f"run_{plan['plan_id']}-{launch_id}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return ROOT / "runs" / f"run_{stamp}_{uuid.uuid4().hex[:8]}"


def require_real_runs_root() -> Path:
    runs_root = ROOT / "runs"
    if runs_root.exists() and (runs_root.is_symlink() or not runs_root.is_dir()):
        raise RuntimeError("The Aptus runs path must be a real directory.")
    runs_root.mkdir(mode=0o700, exist_ok=True)
    if runs_root.is_symlink() or runs_root.resolve() != ROOT / "runs":
        raise RuntimeError("The Aptus runs directory escapes the bundle root.")
    return runs_root.resolve()


def require_owned_output_path(
    output_dir: Path, plan: dict[str, Any], *, pilot: bool
) -> Path:
    unresolved = output_dir.expanduser()
    if unresolved.is_symlink():
        raise RuntimeError("Aptus output paths cannot be symlinks.")
    resolved = unresolved.resolve()
    runs_root = require_real_runs_root()
    if pilot:
        pilot_root = resolved.parent
        marker = pilot_root / ".aptus-pilot-run.json"
        if (
            resolved.name not in {"phase-1", "phase-2"}
            or pilot_root.parent != runs_root
            or not pilot_root.name.startswith("pilot_")
            or pilot_root.is_symlink()
            or not pilot_root.is_dir()
            or marker.is_symlink()
            or not marker.is_file()
        ):
            raise RuntimeError("Pilot output is not inside an owned Aptus pilot root.")
        ownership = json.loads(marker.read_text(encoding="utf-8"))
        expected = {
            "schema_version": "aptus.pilot-run.v1",
            "pilot_run_id": pilot_root.name,
            "plan_id": plan["plan_id"],
            "candidate_id": plan["recommended"]["candidate_id"],
            "model_revision": plan["model"]["revision"],
            "dataset_sha256": plan["dataset"]["source_sha256"],
        }
        if not isinstance(ownership, dict) or any(
            ownership.get(name) != value for name, value in expected.items()
        ):
            raise RuntimeError("Pilot output ownership does not match the plan.")
    elif resolved.parent != runs_root or not resolved.name.startswith("run_"):
        raise RuntimeError(
            "Full-training output must be a fresh ROOT/runs/run_* directory."
        )
    return resolved


def claim_output_dir(
    output_dir: Path, plan: dict[str, Any], *, pilot: bool
) -> Path:
    output_dir = require_owned_output_path(output_dir, plan, pilot=pilot)
    marker = output_dir / ".aptus-run.json"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    contract = {
        "schema_version": "aptus.run-output.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "pilot": pilot,
    }
    if local_rank == 0:
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RuntimeError(
                f"Aptus refuses to reuse the existing run output directory: {output_dir}"
            ) from error
        contract["created_at"] = datetime.now(timezone.utc).isoformat()
        temporary = marker.with_name(f".{marker.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, marker)
    else:
        deadline = time.monotonic() + 30
        while not marker.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not marker.is_file():
            raise RuntimeError("Rank zero did not publish the Aptus run-output contract.")
        observed = json.loads(marker.read_text(encoding="utf-8"))
        for key, value in contract.items():
            if observed.get(key) != value:
                raise RuntimeError("The distributed run-output contract does not match this worker.")
    return output_dir


def require_recorded_trainable_parameter_census(
    value: Any, *, expected_method: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Training metrics do not contain a trainable-parameter census.")
    if value.get("schema_version") != "aptus.trainable-parameter-census.v1":
        raise RuntimeError("Training metrics carry an unknown trainable-parameter census.")
    if value.get("method") != expected_method:
        raise RuntimeError("Training metrics bind the wrong trainable method scope.")
    expected_scope = (
        "all-parameters" if expected_method == "full" else "lora-adapter-only"
    )
    if value.get("parameter_scope") != expected_scope:
        raise RuntimeError("Training metrics bind the wrong trainable parameter scope.")
    for name in ("trainable_parameter_count", "trainable_tensor_count"):
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise RuntimeError(
                f"Training metrics require a positive {name.replace('_', ' ')}."
            )
    if value.get("all_values_finite") is not True:
        raise RuntimeError("Training metrics do not attest finite trainable parameters.")
    for name in (
        "frozen_parameter_count",
        "frozen_tensor_count",
        "unexpected_trainable_tensor_count",
        "expected_adapter_target_match_count",
        "adapter_target_instance_count",
        "incomplete_adapter_target_instance_count",
    ):
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RuntimeError(
                f"Training metrics require a non-negative {name.replace('_', ' ')}."
            )
    adapter_counter_names = (
        "expected_adapter_target_match_count",
        "adapter_target_instance_count",
        "incomplete_adapter_target_instance_count",
    )
    if expected_method == "full":
        if value["frozen_parameter_count"] or value["frozen_tensor_count"]:
            raise RuntimeError("Full-training metrics attest frozen model parameters.")
        if any(value[name] for name in adapter_counter_names):
            raise RuntimeError("Full-training metrics attest adapter target instances.")
    else:
        if value["unexpected_trainable_tensor_count"]:
            raise RuntimeError("Adapter metrics attest unexpected trainable parameters.")
        if value["frozen_parameter_count"] <= 0 or value["frozen_tensor_count"] <= 0:
            raise RuntimeError("Adapter metrics do not attest a non-empty frozen base.")
        if (
            value["expected_adapter_target_match_count"] <= 0
            or value["adapter_target_instance_count"]
            != value["expected_adapter_target_match_count"]
            or value["incomplete_adapter_target_instance_count"] != 0
        ):
            raise RuntimeError(
                "Adapter metrics do not bind one complete LoRA A/B pair to every target instance."
            )
    descriptor_digest = value.get("descriptor_sha256")
    if (
        not isinstance(descriptor_digest, str)
        or len(descriptor_digest) != 64
        or any(character not in "0123456789abcdef" for character in descriptor_digest)
    ):
        raise RuntimeError("Training metrics carry an invalid trainable descriptor digest.")
    return value


def require_recorded_dataset_split(
    value: Any, *, training_count: Any, evaluation_count: Any
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Full training metrics do not contain dataset-split evidence.")
    if value.get("schema_version") != "aptus.dataset-split.v1":
        raise RuntimeError("Full training metrics carry unknown dataset-split evidence.")
    integer_fields = (
        "total_row_count",
        "training_row_count",
        "evaluation_row_count",
        "declared_group_count",
        "training_declared_group_count",
        "evaluation_declared_group_count",
        "ungrouped_row_count",
        "split_unit_count",
        "training_split_unit_count",
        "evaluation_split_unit_count",
        "target_evaluation_row_count",
        "evaluation_row_error",
    )
    for name in integer_fields:
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RuntimeError(
                f"Dataset-split evidence requires a non-negative {name.replace('_', ' ')}."
            )
    if value["training_row_count"] <= 0:
        raise RuntimeError("Dataset-split evidence contains no training rows.")
    if value["total_row_count"] != (
        value["training_row_count"] + value["evaluation_row_count"]
    ):
        raise RuntimeError("Dataset-split row counts are inconsistent.")
    if value["declared_group_count"] != (
        value["training_declared_group_count"]
        + value["evaluation_declared_group_count"]
    ):
        raise RuntimeError("Dataset-split declared-group counts are inconsistent.")
    if value["split_unit_count"] != (
        value["training_split_unit_count"]
        + value["evaluation_split_unit_count"]
    ):
        raise RuntimeError("Dataset-split unit counts are inconsistent.")
    if (
        not isinstance(training_count, int)
        or isinstance(training_count, bool)
        or training_count != value["training_row_count"]
        or not isinstance(evaluation_count, int)
        or isinstance(evaluation_count, bool)
        or evaluation_count != value["evaluation_row_count"]
    ):
        raise RuntimeError("Dataset-split evidence differs from the trainer dataset sizes.")
    strategy = value.get("strategy")
    expected_strategy = (
        "deterministic-size-aware-group-sha256"
        if value["declared_group_count"]
        else "deterministic-exact-row-count-sha256"
    )
    if strategy != expected_strategy:
        raise RuntimeError("Dataset-split evidence carries the wrong strategy marker.")
    if value["target_evaluation_row_count"] >= value["total_row_count"]:
        raise RuntimeError("Dataset-split target leaves no training row.")
    if value["evaluation_row_error"] != abs(
        value["evaluation_row_count"] - value["target_evaluation_row_count"]
    ):
        raise RuntimeError("Dataset-split evaluation error is inconsistent.")
    realized_fraction = value.get("realized_evaluation_fraction")
    if (
        not isinstance(realized_fraction, (int, float))
        or isinstance(realized_fraction, bool)
        or not math.isfinite(realized_fraction)
        or realized_fraction != value["evaluation_row_count"] / value["total_row_count"]
    ):
        raise RuntimeError("Dataset-split realized fraction is inconsistent.")
    for name in ("canonical_jsonl_sha256", "assignment_sha256"):
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"Dataset-split evidence carries an invalid {name}.")
    return value


def assert_measured_training_metrics(
    metrics: dict[str, Any],
    *,
    candidate: dict[str, Any],
    pilot: bool,
) -> None:
    phase = "Pilot" if pilot else "Full training"
    global_step = metrics.get("global_step")
    if (
        not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or global_step < 1
    ):
        raise RuntimeError(f"{phase} did not complete a training step.")
    train_loss = metrics.get("train_loss")
    if (
        not isinstance(train_loss, (int, float))
        or isinstance(train_loss, bool)
        or not math.isfinite(train_loss)
    ):
        raise RuntimeError(f"{phase} did not report a finite train_loss.")
    for name, value in metrics.items():
        if name.endswith("loss") and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise RuntimeError(f"{phase} metric {name} is not finite.")
    for name in (
        "finite_raw_loss_checks",
        "finite_backward_loss_checks",
        "finite_gradient_norm_checks",
        "finite_trainable_parameter_scans",
        "optimizer_parameter_binding_checks",
        "non_skipped_optimizer_steps",
    ):
        value = metrics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise RuntimeError(
                f"{phase} did not attest at least one {name.replace('_', ' ')}."
            )
    require_recorded_trainable_parameter_census(
        metrics.get("trainable_parameter_census"),
        expected_method=str(candidate["method"]),
    )
    if pilot:
        if metrics.get("dataset_split") is not None:
            raise RuntimeError("Pilot metrics must not claim a full-dataset split.")
    else:
        require_recorded_dataset_split(
            metrics.get("dataset_split"),
            training_count=metrics.get("training_example_count"),
            evaluation_count=metrics.get("evaluation_example_count"),
        )
    if metrics.get("pilot") is not pilot:
        raise RuntimeError(f"{phase} metrics carry the wrong phase marker.")
    if metrics.get("candidate_id") != candidate["candidate_id"]:
        raise RuntimeError(f"{phase} metrics do not bind the selected candidate.")
    if metrics.get("distribution") != candidate["distribution"]:
        raise RuntimeError(f"{phase} metrics do not bind the selected distribution.")
    world_size = int(candidate["world_size"])
    if metrics.get("actual_world_size") != world_size:
        raise RuntimeError(f"{phase} metrics do not bind the selected world size.")
    per_rank = metrics.get("per_rank_cuda_peaks")
    if not isinstance(per_rank, list) or len(per_rank) != world_size:
        raise RuntimeError(f"{phase} metrics require one CUDA peak per rank.")
    expected_ranks = list(range(world_size))
    observed_ranks: list[int] = []
    allocated_values: list[int] = []
    reserved_values: list[int] = []
    for value in per_rank:
        if not isinstance(value, dict):
            raise RuntimeError(f"{phase} CUDA peak entries must be objects.")
        rank = value.get("rank")
        allocated = value.get("measured_peak_cuda_bytes")
        reserved = value.get("measured_reserved_cuda_bytes")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise RuntimeError(f"{phase} CUDA peak ranks must be integers.")
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in (allocated, reserved)
        ):
            raise RuntimeError(f"{phase} CUDA peaks must be non-negative integers.")
        if reserved < allocated:
            raise RuntimeError(f"{phase} reserved CUDA memory is below allocated memory.")
        observed_ranks.append(rank)
        allocated_values.append(allocated)
        reserved_values.append(reserved)
    if observed_ranks != expected_ranks:
        raise RuntimeError(f"{phase} CUDA peak ranks do not match the selected world.")
    if max(reserved_values) <= 0:
        raise RuntimeError(f"{phase} did not record a positive CUDA memory peak.")
    if metrics.get("measured_peak_cuda_bytes") != max(allocated_values):
        raise RuntimeError(f"{phase} aggregate allocated CUDA peak is inconsistent.")
    if metrics.get("measured_reserved_cuda_bytes") != max(reserved_values):
        raise RuntimeError(f"{phase} aggregate reserved CUDA peak is inconsistent.")
    if not isinstance(metrics.get("final_export"), dict):
        raise RuntimeError(f"{phase} metrics do not bind a final export.")


def verify_final_export(
    final_dir: Path,
    candidate: dict[str, Any],
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    method = candidate["method"]
    if method == "full":
        config_path = final_dir / "config.json"
        weight_files = sorted(final_dir.glob("model*.safetensors"))
        required_config = config_path
    else:
        required_config = final_dir / "adapter_config.json"
        weight_files = sorted(final_dir.glob("adapter_model*.safetensors"))
    if not required_config.is_file() or not weight_files:
        raise RuntimeError(
            "Final export is incomplete: expected the method-specific config and at least one weight artifact."
        )
    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoTokenizer.from_pretrained(
            str(final_dir), local_files_only=True, trust_remote_code=False
        )
        if method == "full":
            AutoConfig.from_pretrained(
                str(final_dir), local_files_only=True, trust_remote_code=False
            )
    except Exception as error:
        raise RuntimeError(
            "Final export tokenizer or model config cannot be loaded by pinned Transformers."
        ) from error
    if method != "full":
        try:
            adapter_config = json.loads(required_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Final adapter config is unreadable.") from error
        expected_base = model_spec["model_id"]
        expected_revision = model_spec["revision"]
        if adapter_config.get("base_model_name_or_path") != expected_base:
            raise RuntimeError(
                "Final adapter config does not bind the planned base model."
            )
        if adapter_config.get("revision") != expected_revision:
            raise RuntimeError(
                "Final adapter config does not bind the planned immutable model revision."
            )
        try:
            from peft import PeftConfig

            loaded_adapter = PeftConfig.from_pretrained(str(final_dir))
        except Exception as error:
            raise RuntimeError("Final adapter config cannot be loaded by pinned PEFT.") from error
        if (
            loaded_adapter.base_model_name_or_path != expected_base
            or loaded_adapter.revision != expected_revision
        ):
            raise RuntimeError("Pinned PEFT loaded different adapter provenance.")
    try:
        from safetensors import safe_open

        tensor_shards: dict[str, str] = {}
        for weight_path in weight_files:
            with safe_open(str(weight_path), framework="pt", device="cpu") as tensors:
                tensor_keys = list(tensors.keys())
                if not tensor_keys:
                    raise RuntimeError(
                        f"Final safetensors shard has no tensor keys: {weight_path.name}."
                    )
                if any(not isinstance(key, str) or not key for key in tensor_keys):
                    raise RuntimeError(
                        f"Final safetensors shard has an invalid tensor key: {weight_path.name}."
                    )
                for key in tensor_keys:
                    previous_shard = tensor_shards.get(key)
                    if previous_shard is not None:
                        raise RuntimeError(
                            "Final safetensors shards contain duplicate tensor key "
                            f"{key!r}: {previous_shard} and {weight_path.name}."
                        )
                    tensor_shards[key] = weight_path.name
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("Final safetensors weights failed structural loading.") from error
    index_files = sorted(final_dir.glob("*.safetensors.index.json"))
    if len(index_files) > 1:
        raise RuntimeError("Final export contains multiple safetensors indexes.")
    if len(weight_files) > 1 and not index_files:
        raise RuntimeError("A multi-shard final export requires one safetensors index.")
    if index_files:
        try:
            index = json.loads(index_files[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Final safetensors index is unreadable.") from error
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("Final safetensors index has no weight map.")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(shard, str)
            or not shard
            for key, shard in weight_map.items()
        ):
            raise RuntimeError("Final safetensors index has an invalid weight map.")
        if set(weight_map) != set(tensor_shards):
            raise RuntimeError("Final safetensors index keys do not match shard tensors.")
        mismatched_keys = [
            key for key, shard in weight_map.items() if tensor_shards[key] != shard
        ]
        if mismatched_keys:
            raise RuntimeError(
                "Final safetensors index maps tensor keys to the wrong shards: "
                + ", ".join(sorted(mismatched_keys)[:10])
            )
    files = sorted(path for path in final_dir.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("Final export contains no files.")
    return {
        "schema_version": "aptus.final-export.v1",
        "verification_level": "structural-file-tree",
        "method": method,
        "base_model": {
            "model_id": model_spec["model_id"],
            "revision": model_spec["revision"],
        },
        "distribution": candidate["distribution"],
        "world_size": candidate["world_size"],
        "device_indices": candidate.get("device_indices", list(range(candidate["world_size"]))),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": [
            {
                "path": path.relative_to(final_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "weight_files": [path.name for path in weight_files],
    }


def export_final_artifact(
    trainer: Any,
    tokenizer: Any,
    output_dir: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    import torch

    candidate = plan["recommended"]
    accelerator = trainer.accelerator
    final_dir = output_dir / "final"
    accelerator.wait_for_everyone()
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if candidate["distribution"] == "fsdp":
        if plugin is None:
            raise RuntimeError("FSDP execution did not initialize the Accelerate FSDP plugin.")
        plugin.set_state_dict_type("FULL_STATE_DICT")
    state_dict = accelerator.get_state_dict(trainer.model)
    unwrapped = accelerator.unwrap_model(trainer.model)
    if accelerator.is_main_process:
        unwrapped.save_pretrained(
            final_dir,
            state_dict=state_dict,
            safe_serialization=True,
        )
    accelerator.wait_for_everyone()
    result: list[dict[str, Any] | None] = [None]
    if accelerator.is_main_process:
        try:
            tokenizer.save_pretrained(final_dir)
            evidence = verify_final_export(final_dir, candidate, plan["model"])
            export_path = output_dir / "final-export.json"
            temporary = export_path.with_name(
                f".{export_path.name}.{uuid.uuid4().hex}.tmp"
            )
            temporary.write_text(
                json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, export_path)
            result[0] = evidence
        except Exception as error:
            result[0] = {"error": str(error)}
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(result, src=0)
    if result[0] is None:
        raise RuntimeError("Final export evidence was not published by rank zero.")
    if "error" in result[0]:
        raise RuntimeError("Final export verification failed: " + str(result[0]["error"]))
    accelerator.wait_for_everyone()
    return result[0]


def select_pilot_rows(
    rows: list[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("pilot row limit must be positive.")
    if len(rows) <= limit:
        return rows
    ranked = sorted(
        enumerate(rows),
        key=lambda item: (
            -len(json.dumps(item[1], ensure_ascii=False, sort_keys=True)),
            item[0],
        ),
    )[:limit]
    return [row for _, row in ranked]


def _declared_split_group(row: dict[str, Any], *, line_number: int) -> str | None:
    missing = object()
    top_level = row.get("split_group", missing)
    metadata = row.get("metadata")
    nested = metadata.get("split_group", missing) if isinstance(metadata, dict) else missing
    declared = [value for value in (top_level, nested) if value is not missing]
    if not declared:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in declared):
        raise ValueError(
            f"Canonical training row {line_number} split_group must be a non-empty string."
        )
    if len(declared) == 2 and declared[0] != declared[1]:
        raise ValueError(
            f"Canonical training row {line_number} declares conflicting split_group values."
        )
    return declared[0]


def _split_unit_digest(
    *, group: str | None, offset: int, line: bytes
) -> str:
    digest = hashlib.sha256()
    if group is None:
        digest.update(b"aptus.ungrouped-row.v1\0")
        digest.update(str(offset).encode("ascii"))
        digest.update(b"\0")
        digest.update(line)
    else:
        digest.update(b"aptus.declared-split-group.v1\0")
        digest.update(group.encode("utf-8"))
    return digest.hexdigest()


def _split_unit_priority(unit_digest: str, *, seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(
            f"aptus.split-priority.v2\0{seed}\0{unit_digest}".encode("ascii")
        ).digest(),
        "big",
    )


def _reachable_row_count_bits(counts: tuple[int, ...], *, limit: int) -> int:
    if limit < 0:
        return 0
    reachable = 1
    mask = (1 << (limit + 1)) - 1
    for count in counts:
        if count <= limit:
            reachable = (reachable | (reachable << count)) & mask
    return reachable


def _reconstruct_group_subset(
    groups: tuple[tuple[str, int, str], ...], *, target: int
) -> set[str]:
    if target == 0:
        return set()
    if not groups:
        raise RuntimeError("Dataset split could not reconstruct its group assignment.")
    if len(groups) == 1:
        group, count, _digest = groups[0]
        if count != target:
            raise RuntimeError("Dataset split could not reconstruct its group assignment.")
        return {group}

    midpoint = len(groups) // 2
    left, right = groups[:midpoint], groups[midpoint:]
    left_total = sum(item[1] for item in left)
    right_total = sum(item[1] for item in right)
    left_bits = _reachable_row_count_bits(
        tuple(item[1] for item in left), limit=min(target, left_total)
    )
    right_bits = _reachable_row_count_bits(
        tuple(item[1] for item in right), limit=min(target, right_total)
    )
    minimum_left = max(0, target - right_total)
    maximum_left = min(target, left_total)
    for left_target in range(maximum_left, minimum_left - 1, -1):
        right_target = target - left_target
        if (left_bits >> left_target) & 1 and (right_bits >> right_target) & 1:
            return _reconstruct_group_subset(
                left, target=left_target
            ) | _reconstruct_group_subset(right, target=right_target)
    raise RuntimeError("Dataset split could not reconstruct its group assignment.")


def split_jsonl_offsets_with_evidence(
    path: Path, *, evaluation_fraction: float, seed: int
) -> tuple[array, array, dict[str, Any]]:
    if not 0 <= evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be in [0, 1).")

    declared_group_counts: dict[str, int] = {}
    ungrouped_row_count = 0

    first_pass_digest = hashlib.sha256()
    with path.open("rb") as source:
        line_number = 0
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            first_pass_digest.update(line)
            line_number += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Canonical training row {line_number} is not valid UTF-8 JSON."
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Canonical training row {line_number} must be an object."
                )
            group = _declared_split_group(row, line_number=line_number)
            if group is not None:
                declared_group_counts[group] = declared_group_counts.get(group, 0) + 1
                continue
            ungrouped_row_count += 1

    if not declared_group_counts and ungrouped_row_count == 0:
        raise ValueError("Canonical training dataset contains no rows.")

    total_row_count = sum(declared_group_counts.values()) + ungrouped_row_count
    split_unit_count = len(declared_group_counts) + ungrouped_row_count
    target_evaluation_row_count = (
        min(total_row_count - 1, max(1, round(total_row_count * evaluation_fraction)))
        if evaluation_fraction > 0 and split_unit_count > 1
        else 0
    )
    minimum_group_rows = max(
        0, target_evaluation_row_count - ungrouped_row_count
    )
    maximum_group_rows = target_evaluation_row_count

    def group_distance(value: int) -> int:
        if value < minimum_group_rows:
            return minimum_group_rows - value
        if value > maximum_group_rows:
            return value - maximum_group_rows
        return 0

    ordered_groups = sorted(
        declared_group_counts.items(),
        key=lambda item: (
            _split_unit_priority(
                _split_unit_digest(group=item[0], offset=0, line=b""), seed=seed
            ),
            _split_unit_digest(group=item[0], offset=0, line=b""),
        ),
    )
    group_units = tuple(
        (group, count, _split_unit_digest(group=group, offset=0, line=b""))
        for group, count in ordered_groups
    )
    maximum_group_selection = min(
        sum(item[1] for item in group_units), total_row_count - 1
    )
    reachable_group_rows = _reachable_row_count_bits(
        tuple(item[1] for item in group_units), limit=maximum_group_selection
    )
    interval_width = maximum_group_rows - minimum_group_rows + 1
    interval = (
        reachable_group_rows >> minimum_group_rows
    ) & ((1 << interval_width) - 1)
    if interval:
        group_evaluation_rows = minimum_group_rows + interval.bit_length() - 1
    else:
        candidates: list[int] = []
        if minimum_group_rows > 0:
            lower = reachable_group_rows & ((1 << minimum_group_rows) - 1)
            if lower:
                candidates.append(lower.bit_length() - 1)
        upper = reachable_group_rows >> (maximum_group_rows + 1)
        if upper:
            candidates.append(
                maximum_group_rows
                + 1
                + ((upper & -upper).bit_length() - 1)
            )
        if not candidates:
            raise RuntimeError("Dataset split has no valid group assignment.")
        group_evaluation_rows = min(
            candidates, key=lambda value: (group_distance(value), value)
        )
    evaluation_groups = _reconstruct_group_subset(
        group_units, target=group_evaluation_rows
    )
    group_assignments = {
        group: (
            "evaluation" if group in evaluation_groups else "train",
            unit_digest,
        )
        for group, _count, unit_digest in group_units
    }

    ungrouped_evaluation_count = min(
        ungrouped_row_count,
        max(0, target_evaluation_row_count - group_evaluation_rows),
    )
    select_evaluation_rows = (
        ungrouped_evaluation_count <= ungrouped_row_count - ungrouped_evaluation_count
    )
    selected_side_count = (
        ungrouped_evaluation_count
        if select_evaluation_rows
        else ungrouped_row_count - ungrouped_evaluation_count
    )
    selected_ungrouped_digests: set[str] = set()
    selection_heap: list[tuple[int, str]] = []
    second_pass_digest = hashlib.sha256()
    with path.open("rb") as source:
        line_number = 0
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            second_pass_digest.update(line)
            line_number += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Canonical training row {line_number} is not valid UTF-8 JSON."
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Canonical training row {line_number} must be an object."
                )
            group = _declared_split_group(row, line_number=line_number)
            if group is not None or selected_side_count == 0:
                continue
            unit_digest = _split_unit_digest(group=None, offset=offset, line=line)
            priority = _split_unit_priority(unit_digest, seed=seed)
            candidate = (-priority, unit_digest)
            if len(selection_heap) < selected_side_count:
                heapq.heappush(selection_heap, candidate)
            elif priority < -selection_heap[0][0]:
                heapq.heapreplace(selection_heap, candidate)

    if first_pass_digest.digest() != second_pass_digest.digest():
        raise RuntimeError("Canonical training dataset changed while it was split.")
    selected_ungrouped_digests = {item[1] for item in selection_heap}
    if len(selected_ungrouped_digests) != selected_side_count:
        raise RuntimeError("Dataset split could not select unique ungrouped units.")

    declared_group_side_counts = {
        "train": sum(
            1 for side, _digest in group_assignments.values() if side == "train"
        ),
        "evaluation": sum(
            1
            for side, _digest in group_assignments.values()
            if side == "evaluation"
        )
    }
    row_counts = {
        "evaluation": group_evaluation_rows + ungrouped_evaluation_count,
        "train": total_row_count
        - group_evaluation_rows
        - ungrouped_evaluation_count,
    }
    unit_counts = {
        "evaluation": declared_group_side_counts["evaluation"]
        + ungrouped_evaluation_count,
        "train": declared_group_side_counts["train"]
        + ungrouped_row_count
        - ungrouped_evaluation_count,
    }
    if row_counts["train"] <= 0:
        raise RuntimeError("Dataset split could not retain a training unit.")

    train_offsets, eval_offsets = array("Q"), array("Q")
    assignment_digest = hashlib.sha256()
    assignment_digest.update(b"aptus.dataset-split-assignments.v2\n")
    third_pass_digest = hashlib.sha256()
    observed_rows = 0
    with path.open("rb") as source:
        line_number = 0
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            third_pass_digest.update(line)
            line_number += 1
            if not line.strip():
                continue
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError(
                    f"Canonical training row {line_number} must be an object."
                )
            group = _declared_split_group(row, line_number=line_number)
            if group is None:
                unit_digest = _split_unit_digest(group=None, offset=offset, line=line)
                selected = unit_digest in selected_ungrouped_digests
                side = (
                    "evaluation"
                    if selected == select_evaluation_rows
                    else "train"
                )
            else:
                side, unit_digest = group_assignments[group]
            target = eval_offsets if side == "evaluation" else train_offsets
            target.append(offset)
            assignment_digest.update(
                f"{offset}:{unit_digest}:{side}\n".encode("ascii")
            )
            observed_rows += 1

    if first_pass_digest.digest() != third_pass_digest.digest():
        raise RuntimeError("Canonical training dataset changed while it was split.")
    if observed_rows != row_counts["train"] + row_counts["evaluation"]:
        raise RuntimeError("Canonical training dataset changed while it was split.")
    if (
        len(train_offsets) != row_counts["train"]
        or len(eval_offsets) != row_counts["evaluation"]
    ):
        raise RuntimeError("Dataset split evidence does not match its row assignments.")
    evidence = {
        "schema_version": "aptus.dataset-split.v1",
        "strategy": (
            "deterministic-size-aware-group-sha256"
            if declared_group_counts
            else "deterministic-exact-row-count-sha256"
        ),
        "seed": seed,
        "evaluation_fraction": evaluation_fraction,
        "target_evaluation_row_count": target_evaluation_row_count,
        "evaluation_row_error": abs(
            len(eval_offsets) - target_evaluation_row_count
        ),
        "realized_evaluation_fraction": len(eval_offsets) / observed_rows,
        "total_row_count": observed_rows,
        "training_row_count": len(train_offsets),
        "evaluation_row_count": len(eval_offsets),
        "declared_group_count": len(declared_group_counts),
        "training_declared_group_count": declared_group_side_counts["train"],
        "evaluation_declared_group_count": declared_group_side_counts["evaluation"],
        "ungrouped_row_count": ungrouped_row_count,
        "split_unit_count": unit_counts["train"] + unit_counts["evaluation"],
        "training_split_unit_count": unit_counts["train"],
        "evaluation_split_unit_count": unit_counts["evaluation"],
        "canonical_jsonl_sha256": third_pass_digest.hexdigest(),
        "assignment_sha256": assignment_digest.hexdigest(),
    }
    return train_offsets, eval_offsets, evidence


def split_jsonl_offsets(
    path: Path, *, evaluation_fraction: float, seed: int
) -> tuple[array, array]:
    train_offsets, eval_offsets, _evidence = split_jsonl_offsets_with_evidence(
        path, evaluation_fraction=evaluation_fraction, seed=seed
    )
    return train_offsets, eval_offsets


def require_collective_dataset_split_binding(
    path: Path, evidence: dict[str, Any]
) -> None:
    import torch

    try:
        local_digest = _sha256(path)
        local_error = None
    except OSError as error:
        local_digest = None
        local_error = f"{type(error).__name__}: {error}"
    local = {
        "canonical_jsonl_sha256": local_digest,
        "assignment_sha256": evidence.get("assignment_sha256"),
        "training_row_count": evidence.get("training_row_count"),
        "evaluation_row_count": evidence.get("evaluation_row_count"),
        "error": local_error,
    }
    gathered = [local]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
    expected_digest = evidence.get("canonical_jsonl_sha256")
    if (
        local_error is not None
        or local_digest != expected_digest
        or any(item != local for item in gathered)
    ):
        raise RuntimeError(
            "Canonical training data or split assignments differ from their collective evidence."
        )


class LazyJsonlDataset:
    def __init__(
        self,
        path: Path,
        offsets: array,
        tokenizer: Any,
        sequence_length: int,
        expected_sha256: str,
    ) -> None:
        self.path = path
        self.offsets = offsets
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.expected_sha256 = expected_sha256
        self._source = None
        self._source_stat = None

    @staticmethod
    def _stat_identity(value: Any) -> tuple[int, int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    def _open_verified_source(self) -> None:
        source = self.path.open("rb")
        before = self._stat_identity(os.fstat(source.fileno()))
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        after = self._stat_identity(os.fstat(source.fileno()))
        if before != after:
            source.close()
            raise RuntimeError("Canonical training data changed while it was verified.")
        if digest.hexdigest() != self.expected_sha256:
            source.close()
            raise RuntimeError("Canonical training data no longer matches split evidence.")
        source.seek(0)
        self._source = source
        self._source_stat = after

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        if self._source is None:
            self._open_verified_source()
        if self._source_stat != self._stat_identity(os.fstat(self._source.fileno())):
            raise RuntimeError("Canonical training data changed during consumption.")
        self._source.seek(self.offsets[index])
        line = self._source.readline()
        if self._source_stat != self._stat_identity(os.fstat(self._source.fileno())):
            raise RuntimeError("Canonical training data changed during consumption.")
        row = json.loads(line.decode("utf-8"))
        if not isinstance(row, dict):
            raise ValueError("Canonical training row must be an object.")
        return encode_record(row, self.tokenizer, self.sequence_length)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_source"] = None
        state["_source_stat"] = None
        return state

    def __del__(self) -> None:
        if self._source is not None:
            self._source.close()


def run_training(
    plan: dict[str, Any],
    trainer_config: dict[str, Any],
    *,
    pilot: bool,
    max_steps: int | None,
    resume_from: str | None,
    output_dir: Path,
    local_files_only: bool,
    seed: int,
) -> None:
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    class FiniteGuardTrainer(Trainer):
        """Reject non-finite raw losses collectively before backward acceptance."""

        finite_raw_loss_checks = 0
        finite_backward_loss_checks = 0
        finite_gradient_norm_checks = 0
        finite_trainable_parameter_scans = 0
        optimizer_parameter_binding_checks = 0
        non_skipped_optimizer_steps = 0

        def create_optimizer(self) -> Any:
            optimizer = super().create_optimizer()
            if self.optimizer is None:
                raise RuntimeError("Trainer did not create an optimizer.")
            expected_parameters = [
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            ]
            observed_parameters = [
                parameter
                for group in self.optimizer.param_groups
                for parameter in group.get("params", ())
            ]
            expected_ids = [id(parameter) for parameter in expected_parameters]
            observed_ids = [id(parameter) for parameter in observed_parameters]
            if (
                not expected_ids
                or len(expected_ids) != len(set(expected_ids))
                or len(observed_ids) != len(set(observed_ids))
                or set(observed_ids) != set(expected_ids)
            ):
                raise RuntimeError(
                    "Optimizer parameters do not exactly match the validated trainable set."
                )
            self.optimizer_parameter_binding_checks += 1
            return optimizer

        @staticmethod
        def require_collective_condition(
            condition: bool, *, device: torch.device, message: str
        ) -> None:
            flag = torch.tensor(
                1 if condition else 0, dtype=torch.int32, device=device
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            if int(flag.item()) != 1:
                raise FloatingPointError(message)

        def require_collective_finite(self, value: Any, *, label: str) -> None:
            if not isinstance(value, torch.Tensor):
                raise FloatingPointError(f"{label} is not a tensor.")
            finite = bool(torch.isfinite(value.detach()).all().item())
            self.require_collective_condition(
                finite,
                device=value.device,
                message=f"{label} is non-finite on at least one rank.",
            )

        def compute_loss(self, model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
            result = super().compute_loss(model, inputs, *args, **kwargs)
            loss = result[0] if isinstance(result, tuple) else result
            self.require_collective_finite(loss, label="Raw training/evaluation loss")
            self.finite_raw_loss_checks += 1
            return result

        def training_step(
            self, model: Any, inputs: Any, *args: Any, **kwargs: Any
        ) -> Any:
            loss = super().training_step(model, inputs, *args, **kwargs)
            self.require_collective_finite(loss, label="Backpropagated training loss")
            self.finite_backward_loss_checks += 1
            return loss

        def _clip_grad_norm(self, model: Any) -> Any:
            gradient_norm = super()._clip_grad_norm(model)
            if isinstance(gradient_norm, torch.Tensor):
                finite = bool(torch.isfinite(gradient_norm.detach()).all().item())
            else:
                try:
                    finite = math.isfinite(float(gradient_norm))
                except (TypeError, ValueError):
                    finite = False
            self.require_collective_condition(
                finite,
                device=self.args.device,
                message="The pre-clip gradient norm is non-finite on at least one rank.",
            )
            self.finite_gradient_norm_checks += 1
            return gradient_norm

        def require_trainable_parameters_finite(self) -> None:
            local_finite = True
            local_count = 0
            for parameter in self.model.parameters():
                if not parameter.requires_grad:
                    continue
                local_count += parameter.numel()
                if parameter.numel() and not bool(
                    torch.isfinite(parameter.detach()).all().item()
                ):
                    local_finite = False
                    break
            count = torch.tensor(
                local_count, dtype=torch.int64, device=self.args.device
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            self.require_collective_condition(
                local_finite and int(count.item()) > 0,
                device=self.args.device,
                message=(
                    "Trainable parameters are absent or non-finite after the final "
                    "optimizer step on at least one rank."
                ),
            )
            self.finite_trainable_parameter_scans += 1

    class NonSkippedOptimizerCallback(TrainerCallback):
        trainer: FiniteGuardTrainer | None = None

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            del state, kwargs
            if self.trainer is None:
                raise RuntimeError("Finite-step callback is not bound to its trainer.")
            skipped = getattr(
                self.trainer.accelerator, "optimizer_step_was_skipped", None
            )
            self.trainer.require_collective_condition(
                skipped is False,
                device=args.device,
                message=(
                    "The optimizer step was skipped or its completion state was unavailable "
                    "on at least one rank."
                ),
            )
            self.trainer.non_skipped_optimizer_steps += 1
            return control

    set_seed(seed)
    random.seed(seed)
    candidate = plan["recommended"]
    initialize_and_require_strategy(plan)
    pilot_rows: list[dict[str, Any]] | None = None
    train_offsets: array | None = None
    eval_offsets: array | None = None
    split_evidence: dict[str, Any] | None = None
    if pilot:
        pilot_rows = load_rows(ROOT / trainer_config["pilot_dataset_path"])
        pilot_rows = select_pilot_rows(
            pilot_rows, limit=trainer_config["pilot_row_limit"]
        )
    else:
        training_path = ROOT / trainer_config["training_dataset_path"]
        train_offsets, eval_offsets, split_evidence = split_jsonl_offsets_with_evidence(
            training_path,
            evaluation_fraction=trainer_config["evaluation_fraction"],
            seed=seed,
        )
        require_collective_dataset_split_binding(training_path, split_evidence)

    effective_steps = resolve_max_steps(pilot=pilot, max_steps=max_steps)
    checkpoint_steps = trainer_config["checkpoint_steps"]
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1 if pilot else trainer_config["max_epochs"],
        max_steps=effective_steps,
        per_device_train_batch_size=trainer_config["per_device_train_batch_size"],
        per_device_eval_batch_size=trainer_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=trainer_config["gradient_accumulation_steps"],
        learning_rate=trainer_config["learning_rate"],
        optim=trainer_config["optimizer"],
        lr_scheduler_type=trainer_config["lr_scheduler_type"],
        weight_decay=trainer_config["weight_decay"],
        warmup_steps=trainer_config["warmup_steps"],
        max_grad_norm=trainer_config["max_grad_norm"],
        bf16=trainer_config["precision"] == "bf16",
        fp16=trainer_config["precision"] == "fp16",
        gradient_checkpointing=trainer_config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={
            "use_reentrant": trainer_config[
                "gradient_checkpointing_use_reentrant"
            ]
        },
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        logging_steps=1 if pilot else trainer_config["logging_steps"],
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=1 if pilot else checkpoint_steps,
        save_total_limit=trainer_config["save_total_limit"],
        eval_strategy="steps" if eval_offsets else "no",
        eval_steps=checkpoint_steps,
        report_to=trainer_config["report_to"],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=trainer_config["remove_unused_columns"],
    )
    if not pilot:
        collectively_require_full_train_approval(plan, output_dir)
    require_hardware_parity(plan)
    model, tokenizer, actual_parameter_count, trainable_census = build_model(
        plan, local_files_only=local_files_only
    )
    if pilot:
        assert pilot_rows is not None
        encoded = [
            encode_record(row, tokenizer, trainer_config["sequence_length"])
            for row in pilot_rows
        ]
        random.Random(seed).shuffle(encoded)

        class EncodedDataset:
            def __init__(self, values: list[dict[str, list[int]]]) -> None:
                self.values = values

            def __len__(self) -> int:
                return len(self.values)

            def __getitem__(self, index: int) -> dict[str, list[int]]:
                return self.values[index]

        train_dataset = EncodedDataset(encoded)
        eval_dataset = None
    else:
        assert train_offsets is not None and eval_offsets is not None
        train_dataset = LazyJsonlDataset(
            training_path,
            train_offsets,
            tokenizer,
            trainer_config["sequence_length"],
            split_evidence["canonical_jsonl_sha256"],
        )
        eval_dataset = (
            LazyJsonlDataset(
                training_path,
                eval_offsets,
                tokenizer,
                trainer_config["sequence_length"],
                split_evidence["canonical_jsonl_sha256"],
            )
            if eval_offsets
            else None
        )

    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = (
            trainer_config["sequence_length"]
            if pilot
            else max(len(item["input_ids"]) for item in features)
        )
        ids, masks, labels = [], [], []
        for item in features:
            padding = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    step_callback = NonSkippedOptimizerCallback()
    trainer = FiniteGuardTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
        processing_class=tokenizer,
        callbacks=[step_callback],
    )
    step_callback.trainer = trainer
    require_trainer_strategy(trainer, plan)
    if not pilot:
        assert split_evidence is not None
        require_collective_dataset_split_binding(training_path, split_evidence)
    result = trainer.train(resume_from_checkpoint=resume_from)
    if not pilot:
        require_collective_dataset_split_binding(training_path, split_evidence)
    trainer.require_trainable_parameters_finite()
    metrics = dict(result.metrics)
    if eval_dataset is not None:
        metrics.update(trainer.evaluate())
    if not pilot:
        require_collective_dataset_split_binding(training_path, split_evidence)
    export_evidence = export_final_artifact(trainer, tokenizer, output_dir, plan)
    if torch.cuda.is_available():
        local_peaks = torch.tensor(
            [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        gathered_peaks = [local_peaks]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered_peaks = [
                torch.zeros_like(local_peaks)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered_peaks, local_peaks)
        per_rank_peaks = [
            {
                "rank": rank,
                "measured_peak_cuda_bytes": int(values[0].item()),
                "measured_reserved_cuda_bytes": int(values[1].item()),
            }
            for rank, values in enumerate(gathered_peaks)
        ]
        metrics["per_rank_cuda_peaks"] = per_rank_peaks
        metrics["measured_peak_cuda_bytes"] = max(
            item["measured_peak_cuda_bytes"] for item in per_rank_peaks
        )
        metrics["measured_reserved_cuda_bytes"] = max(
            item["measured_reserved_cuda_bytes"] for item in per_rank_peaks
        )
    metrics.update({
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "pilot": pilot,
        "pilot_row_count": len(train_dataset) if pilot else None,
        "pilot_evaluation_enabled": False if pilot else eval_dataset is not None,
        "training_example_count": len(train_dataset),
        "evaluation_example_count": len(eval_dataset) if eval_dataset is not None else 0,
        "dataset_split": split_evidence,
        "global_step": trainer.state.global_step,
        "finite_raw_loss_checks": trainer.finite_raw_loss_checks,
        "finite_backward_loss_checks": trainer.finite_backward_loss_checks,
        "finite_gradient_norm_checks": trainer.finite_gradient_norm_checks,
        "finite_trainable_parameter_scans": trainer.finite_trainable_parameter_scans,
        "optimizer_parameter_binding_checks": trainer.optimizer_parameter_binding_checks,
        "non_skipped_optimizer_steps": trainer.non_skipped_optimizer_steps,
        "resume_from": resume_from,
        "distribution": candidate["distribution"],
        "actual_world_size": (
            torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        ),
        "actual_parameter_count": actual_parameter_count,
        "parameter_count_tolerance_fraction": 0.02,
        "trainable_parameter_census": trainable_census,
        "final_export": export_evidence,
    })
    assert_measured_training_metrics(metrics, candidate=candidate, pilot=pilot)
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        temporary = metrics_path.with_name(
            f".{metrics_path.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, metrics_path)
        print(json.dumps(metrics, indent=2, sort_keys=True))
    trainer.accelerator.wait_for_everyone()
    if not pilot:
        collectively_record_measured_run(output_dir, plan)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the selected Aptus training candidate.")
    parser.add_argument("--preflight-model-data", action="store_true")
    parser.add_argument("--synthetic-preflight", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--confirm-full-train", action="store_true")
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--runtime-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--promote-pending", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    require_execution_lease()
    if arguments.runtime_probe:
        if arguments.world_size is None or arguments.world_size <= 0:
            parser.error("--runtime-probe requires a positive --world-size.")
        print(json.dumps(local_runtime_snapshot(arguments.world_size), sort_keys=True))
        return 0
    plan = load_plan()
    bind_visible_cuda_devices(plan)
    trainer_config = load_trainer_config()
    require_compiler_contract(plan, trainer_config)
    if arguments.promote_pending is not None:
        promote_pending_run(arguments.promote_pending, plan)
        print(f"Promoted measured-run evidence for {arguments.promote_pending}.")
        return 0
    if arguments.preflight_model_data:
        model_data_preflight(plan, trainer_config, local_files_only=arguments.local_files_only)
        return 0
    if arguments.synthetic_preflight:
        synthetic_preflight(plan)
        return 0
    if not arguments.pilot and not arguments.confirm_full_train:
        parser.error("Full training requires --confirm-full-train.")
    if not arguments.pilot and arguments.resume_from is not None:
        parser.error("Full-training resume is fail-closed in Aptus v0.2 until checkpoint manifests bind complete optimizer, scheduler, RNG, model, and plan state.")
    if not arguments.pilot and (arguments.max_steps is not None or arguments.seed is not None):
        parser.error("--max-steps and --seed are pilot-only overrides; full training uses the compiled target contract.")
    output_dir = claim_output_dir(
        arguments.output_dir or default_output_dir(plan, pilot=arguments.pilot),
        plan,
        pilot=arguments.pilot,
    )
    run_training(
        plan,
        trainer_config,
        pilot=arguments.pilot,
        max_steps=arguments.max_steps,
        resume_from=arguments.resume_from,
        output_dir=output_dir,
        local_files_only=arguments.local_files_only,
        seed=arguments.seed if arguments.seed is not None else trainer_config["seed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


MLX_TRAIN_SCRIPT = r'''#!/usr/bin/env python3
"""Execute the MLX-LM compiler slice selected by an Aptus plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
from plan_contract import validate_bundle_manifest, validate_plan_payload


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    errors = validate_plan_payload(plan, root=ROOT, verify_dataset=True)
    errors += validate_bundle_manifest(ROOT)
    if errors:
        raise RuntimeError("Invalid Aptus bundle: " + " | ".join(errors))
    candidate = plan["recommended"]
    runtime = candidate.get("runtime_contract")
    if (
        not isinstance(runtime, dict)
        or runtime.get("training_runtime") != "mlx-lm"
        or runtime.get("compute_backend") != "mps"
        or candidate.get("distribution") != "single"
        or candidate.get("method") not in {"lora", "qlora"}
    ):
        raise RuntimeError("The selected candidate is not an executable MLX-LM contract.")
    return plan, candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_output(path: Path) -> Path:
    unresolved = path if path.is_absolute() else ROOT / path
    resolved = unresolved.resolve()
    allowed = ((ROOT / "runs").resolve(), (ROOT / "pilot-output").resolve())
    if not any(parent == resolved or parent in resolved.parents for parent in allowed):
        raise RuntimeError("MLX adapter output must remain under runs/ or pilot-output/.")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def require_data(path: Path) -> Path:
    unresolved = path if path.is_absolute() else ROOT / path
    resolved = unresolved.resolve(strict=True)
    expected = (ROOT / "data" / "mlx").resolve(strict=True)
    if resolved != expected:
        raise RuntimeError("The MLX data argument must match the compiler-bound data/mlx directory.")
    for name in ("train.jsonl", "valid.jsonl", "split-contract.json"):
        if not (resolved / name).is_file():
            raise RuntimeError(f"MLX dataset is missing {name}.")
    return resolved


def download_pinned_model(plan: dict[str, Any], requested_model: str) -> Path:
    model = plan["model"]
    if requested_model != model["model_id"]:
        raise RuntimeError("The model argument must equal the plan-bound provider model ID.")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=requested_model,
            revision=model["revision"],
        )
    ).resolve(strict=True)


def require_method_model(candidate: dict[str, Any], model_path: Path) -> None:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_file"):
        raise RuntimeError(
            "MLX-LM custom model_file code is unsupported; Aptus only executes pinned built-in MLX model implementations."
        )
    quantization = config.get("quantization") or config.get("quantization_config")
    text_config = config.get("text_config")
    if not quantization and isinstance(text_config, dict):
        quantization = text_config.get("quantization_config")
    bits = quantization.get("bits") if isinstance(quantization, dict) else None
    if candidate["method"] == "qlora" and bits != 4:
        raise RuntimeError(
            "MLX-LM QLoRA requires a pinned model revision with explicit four-bit MLX quantization metadata. Aptus will not substitute bitsandbytes or quantize an unbound model during training."
        )
    if candidate["method"] == "lora" and quantization:
        raise RuntimeError(
            "MLX-LM LoRA requires an unquantized pinned base model. A quantized base "
            "would execute QLoRA semantics under the wrong planned method."
        )


def current_available_unified_memory_bytes() -> int:
    try:
        completed = subprocess.run(
            ["/usr/bin/vm_stat"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Current Apple unified-memory admission probe failed.") from error
    if completed.returncode:
        raise RuntimeError("Current Apple unified-memory admission probe failed.")
    page_match = re.search(r"page size of\s+(\d+) bytes", completed.stdout)
    if page_match is None:
        raise RuntimeError("vm_stat did not report its page size.")
    counts: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        match = re.fullmatch(r"([^:]+):\s*([0-9]+)\.", line.strip())
        if match is not None:
            counts[match.group(1)] = int(match.group(2))
    names = ("Pages free", "Pages inactive", "Pages speculative")
    if not any(name in counts for name in names):
        raise RuntimeError("vm_stat did not report available-memory page classes.")
    available = sum(counts.get(name, 0) for name in names) * int(page_match.group(1))
    if available <= 0:
        raise RuntimeError("Current available Apple unified memory is zero or unknown.")
    return available


def require_unified_memory_admission(plan: dict[str, Any]) -> dict[str, Any]:
    candidate = plan["recommended"]
    memory = candidate["memory"]
    point = int(memory["point_estimate_bytes"])
    upper = int(memory["upper_estimate_bytes"])
    reserve = max(
        int(plan["hardware"].get("reserve_per_device_bytes", 0)),
        8 * 1024**3,
    )
    available = current_available_unified_memory_bytes()
    required = max(point, upper) + reserve
    if available < required:
        raise RuntimeError(
            "Current available Apple unified memory is below the candidate upper "
            "estimate plus the required 8 GiB Aptus reserve."
        )
    return {
        "schema_version": "aptus.mlx-unified-memory-admission.v1",
        "available_unified_memory_bytes": available,
        "point_estimate_bytes": point,
        "upper_estimate_bytes": upper,
        "reserve_bytes": reserve,
        "required_available_bytes": required,
    }


def resolve_lora_keys(model: Any, candidate: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    planned = candidate.get("target_modules")
    if (
        not isinstance(planned, list)
        or not planned
        or any(not isinstance(target, str) or not target for target in planned)
        or len(set(planned)) != len(planned)
    ):
        raise RuntimeError("The MLX-LM candidate requires unique planned target modules.")
    layers = tuple(getattr(model, "layers", ()))
    if not layers:
        raise RuntimeError("The loaded MLX-LM model exposes no transformer layers.")
    resolved: dict[str, str] = {}
    for target in planned:
        observed: list[str] = []
        for layer_index, layer in enumerate(layers):
            matches = sorted(
                name
                for name, _module in layer.named_modules()
                if name == target or name.endswith("." + target)
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"Planned MLX target {target!r} matched {len(matches)} modules "
                    f"in transformer layer {layer_index}; exactly one is required."
                )
            observed.append(matches[0])
        if len(set(observed)) != 1:
            raise RuntimeError(
                f"Planned MLX target {target!r} does not resolve to one stable layer-relative key."
            )
        resolved[target] = observed[0]
    resolved_keys = [resolved[target] for target in planned]
    if len(set(resolved_keys)) != len(resolved_keys):
        raise RuntimeError("Distinct planned MLX targets resolve to the same runtime key.")
    binding = {
        "schema_version": "aptus.mlx-trainable-target-binding.v1",
        "planned_target_modules": planned,
        "resolved_layer_keys": resolved_keys,
        "transformer_layer_count": len(layers),
        "expected_adapter_target_instance_count": len(layers) * len(planned),
    }
    return resolved_keys, binding


def require_trainable_binding(
    names: list[str], binding: dict[str, Any]
) -> dict[str, Any]:
    planned = binding["planned_target_modules"]
    pairs: dict[str, set[str]] = {}
    target_counts = {target: 0 for target in planned}
    for name in names:
        suffix = next(
            (suffix for suffix in (".lora_a", ".lora_b") if name.endswith(suffix)),
            None,
        )
        if suffix is None:
            raise RuntimeError(
                "MLX-LM left a non-LoRA parameter trainable in the bounded smoke."
            )
        base = name[: -len(suffix)]
        matches = [
            target
            for target in planned
            if base == target or base.endswith("." + target)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "A trainable MLX adapter parameter does not bind exactly one planned target."
            )
        pairs.setdefault(base, set()).add(suffix.removeprefix("."))
    if not pairs or any(kinds != {"lora_a", "lora_b"} for kinds in pairs.values()):
        raise RuntimeError("Every planned MLX adapter instance requires one LoRA A/B pair.")
    for base in pairs:
        target = next(
            target
            for target in planned
            if base == target or base.endswith("." + target)
        )
        target_counts[target] += 1
    layer_count = binding["transformer_layer_count"]
    if (
        len(pairs) != binding["expected_adapter_target_instance_count"]
        or any(count != layer_count for count in target_counts.values())
    ):
        raise RuntimeError(
            "The MLX trainable adapter set does not cover every planned target in every layer."
        )
    descriptor = {
        **binding,
        "adapter_target_instance_count": len(pairs),
        "trainable_tensor_count": len(names),
        "target_instance_counts": target_counts,
    }
    descriptor["descriptor_sha256"] = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return descriptor


def derive_iterations(
    *,
    action: str,
    requested_iterations: int,
    candidate: dict[str, Any],
    plan: dict[str, Any],
    train_examples: int,
) -> int:
    accumulation = int(candidate["gradient_accumulation_steps"])
    if accumulation <= 0 or requested_iterations <= 0 or train_examples <= 0:
        raise RuntimeError("MLX-LM iteration inputs must be positive.")
    if action == "bounded-smoke":
        iterations = max(requested_iterations, accumulation)
        if iterations > 8:
            raise RuntimeError(
                "The planned gradient accumulation exceeds the eight-iteration measured-preflight bound."
            )
        return iterations
    if action == "pilot":
        # Keep the pilot bounded and deterministic while proving two complete updates.
        return 2 * accumulation
    if action == "full":
        micro_batch = int(candidate["micro_batch_size"])
        max_epochs = int(plan["target"]["max_epochs"])
        if micro_batch <= 0 or max_epochs <= 0:
            raise RuntimeError("MLX-LM full-run batch and epoch values must be positive.")
        if train_examples < micro_batch:
            raise RuntimeError("MLX-LM full training has no complete micro-batch.")
        batches_per_epoch = train_examples // micro_batch
        epoch_iterations = batches_per_epoch * max_epochs
        return math.ceil(epoch_iterations / accumulation) * accumulation
    raise RuntimeError("Unknown MLX-LM training action.")


def load_pinned_local_model(
    loader: Any,
    requested_model: str,
    *args: Any,
    model_path: Path,
    plan: dict[str, Any],
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    try:
        requested_path = Path(requested_model).resolve(strict=True)
        expected_path = model_path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("MLX-LM attempted to load a missing model path.") from error
    tokenizer_config = kwargs.get("tokenizer_config")
    if (
        requested_path != expected_path
        or args
        or set(kwargs) != {"tokenizer_config"}
        or not isinstance(tokenizer_config, dict)
        or tokenizer_config != {"trust_remote_code": True}
    ):
        raise RuntimeError(
            "Pinned MLX-LM model loading changed shape; Aptus refuses unbound loader arguments."
        )
    binding = {
        "schema_version": "aptus.mlx-model-load-binding.v1",
        "model_id": plan["model"]["model_id"],
        "model_revision": plan["model"]["revision"],
        "resolved_local_snapshot": True,
        "trust_remote_code": False,
    }
    loaded = loader(str(expected_path), tokenizer_config={"trust_remote_code": False})
    return loaded, binding


def run_smoke(arguments: argparse.Namespace) -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX-LM execution requires Apple silicon macOS.")
    plan, candidate = load_contract()
    actions = {
        "bounded-smoke": arguments.bounded_smoke,
        "pilot": arguments.pilot,
        "full": arguments.confirm_full_train,
    }
    selected = [name for name, enabled in actions.items() if enabled]
    if len(selected) != 1:
        raise RuntimeError("Choose exactly one MLX-LM training action.")
    action = selected[0]
    if arguments.resume_from is not None:
        raise RuntimeError(
            "MLX-LM resume is unsupported. Aptus runs this path uninterrupted from scratch."
        )
    data_path = require_data(arguments.data)
    adapter_path = require_output(arguments.adapter_path)
    model_path = download_pinned_model(plan, arguments.model)
    require_method_model(candidate, model_path)

    import mlx.core as mx
    from mlx_lm import lora
    from mlx.utils import tree_flatten

    mx.reset_peak_memory()
    configured_keys = candidate.get("target_modules")
    expected_scale = float(candidate["alpha"]) / int(candidate["rank"])
    evidence: dict[str, Any] = {"train_losses": [], "validation_losses": []}
    original_load = lora.load
    original_linear_to_lora_layers = lora.linear_to_lora_layers
    original_train = lora.train
    original_get_reporting_callbacks = lora.get_reporting_callbacks

    class EvidenceCallback:
        def __init__(self) -> None:
            self.delegate = None

        def on_train_loss_report(self, info: dict[str, Any]) -> None:
            loss = float(info.get("train_loss", float("nan")))
            evidence["train_losses"].append(loss)
            if self.delegate is not None:
                self.delegate.on_train_loss_report(info)

        def on_val_loss_report(self, info: dict[str, Any]) -> None:
            loss = float(info.get("val_loss", float("nan")))
            evidence["validation_losses"].append(loss)
            if self.delegate is not None:
                self.delegate.on_val_loss_report(info)

    callback = EvidenceCallback()

    def pinned_local_load(requested_model: str, *args: Any, **kwargs: Any) -> Any:
        loaded, binding = load_pinned_local_model(
            original_load,
            requested_model,
            *args,
            model_path=model_path,
            plan=plan,
            **kwargs,
        )
        evidence["model_load_binding"] = binding
        return loaded

    def reporting_callbacks(*args: Any, **kwargs: Any) -> EvidenceCallback:
        callback.delegate = original_get_reporting_callbacks(*args, **kwargs)
        return callback

    def linear_to_lora_layers(
        model: Any,
        num_layers: int,
        config: dict[str, Any],
        use_dora: bool = False,
    ) -> None:
        if config.get("keys") != configured_keys:
            raise RuntimeError(
                "MLX-LM LoRA keys do not equal the plan-bound target modules."
            )
        if (
            config.get("rank") != candidate["rank"]
            or float(config.get("scale", float("nan"))) != expected_scale
        ):
            raise RuntimeError("MLX-LM LoRA rank or alpha/r scale violates the plan.")
        resolved_keys, binding = resolve_lora_keys(model, candidate)
        config["keys"] = resolved_keys
        evidence["resolved_binding"] = binding
        original_linear_to_lora_layers(
            model, num_layers, config, use_dora=use_dora
        )

    def instrumented_train(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        training_args = kwargs.get("args")
        if model is None or training_args is None:
            raise RuntimeError("Pinned MLX-LM train invocation changed shape.")
        update_opportunities = (
            int(training_args.iters) // int(training_args.grad_accumulation_steps)
        )
        if update_opportunities < 1:
            raise RuntimeError(
                "The bounded MLX smoke schedules no optimizer update after gradient accumulation."
            )
        before = dict(tree_flatten(model.trainable_parameters()))
        binding = require_trainable_binding(
            sorted(before), evidence.get("resolved_binding", {})
        )
        mx.eval(*before.values())
        optimizer = kwargs.get("optimizer")
        if optimizer is None or not hasattr(optimizer, "step"):
            raise RuntimeError("Pinned MLX-LM optimizer exposes no step counter.")
        mx.eval(optimizer.step)
        optimizer_step_before = int(optimizer.step.item())
        kwargs["training_callback"] = callback
        result = original_train(*args, **kwargs)
        after = dict(tree_flatten(model.trainable_parameters()))
        if set(after) != set(before):
            raise RuntimeError("The MLX trainable parameter set changed during training.")
        mx.eval(*after.values())
        mx.eval(optimizer.step)
        completed_optimizer_updates = int(optimizer.step.item()) - optimizer_step_before
        if completed_optimizer_updates != update_opportunities:
            raise RuntimeError(
                "MLX-LM optimizer step count does not equal the scheduled update count."
            )
        deltas = []
        for name in sorted(before):
            delta = float(mx.sum(mx.abs(after[name] - before[name])).item())
            if not math.isfinite(delta) or delta < 0:
                raise RuntimeError("MLX-LM produced a non-finite adapter delta.")
            deltas.append(delta)
        delta_l1 = sum(deltas)
        if not math.isfinite(delta_l1) or delta_l1 <= 0:
            raise RuntimeError(
                "MLX-LM produced no nonzero adapter delta; an optimizer update is unproven."
            )
        evidence.update(
            trainable_target_binding=binding,
            optimizer_update_opportunities=update_opportunities,
            completed_optimizer_updates=completed_optimizer_updates,
            optimizer_update_observed=(completed_optimizer_updates > 0),
            adapter_delta_l1=delta_l1,
            changed_adapter_tensor_count=sum(delta > 0 for delta in deltas),
            trainable_parameter_names=sorted(after),
        )
        return result

    lora.load = pinned_local_load
    lora.linear_to_lora_layers = linear_to_lora_layers
    lora.train = instrumented_train
    lora.get_reporting_callbacks = reporting_callbacks
    previous_argv = sys.argv
    accumulation = int(candidate["gradient_accumulation_steps"])
    train_examples = sum(
        1
        for line in (data_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    valid_examples = sum(
        1
        for line in (data_path / "valid.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    split_contract = json.loads(
        (data_path / "split-contract.json").read_text(encoding="utf-8")
    )
    split_values = split_contract.get("splits", {})
    if (
        split_contract.get("schema_version") != "aptus.mlx-split.v1"
        or split_contract.get("micro_batch_size") != candidate["micro_batch_size"]
        or split_values.get("train", {}).get("compiled_row_count") != train_examples
        or split_values.get("valid", {}).get("compiled_row_count") != valid_examples
        or train_examples < int(candidate["micro_batch_size"])
        or valid_examples < int(candidate["micro_batch_size"])
        or train_examples % int(candidate["micro_batch_size"])
        or valid_examples % int(candidate["micro_batch_size"])
    ):
        raise RuntimeError("Compiled MLX split counts do not match their bound contract.")
    source_train_examples = split_values["train"]["source_row_count"]
    source_validation_examples = split_values["valid"]["source_row_count"]
    required_iterations = derive_iterations(
        action=action,
        requested_iterations=int(arguments.iters),
        candidate=candidate,
        plan=plan,
        train_examples=train_examples,
    )
    if required_iterations <= 0:
        raise RuntimeError("MLX-LM derived no training iterations from the compiled data.")
    sys.argv = [
        "mlx_lm.lora",
        "--config",
        str(ROOT / "config" / "mlx-lm.yaml"),
        "--model",
        str(model_path),
        "--data",
        str(data_path),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(required_iterations),
        "--save-every",
        str(required_iterations + 1),
        "--train",
    ]
    memory_admission = require_unified_memory_admission(plan)
    try:
        lora.main()
    finally:
        sys.argv = previous_argv
        lora.load = original_load
        lora.linear_to_lora_layers = original_linear_to_lora_layers
        lora.train = original_train
        lora.get_reporting_callbacks = original_get_reporting_callbacks
    adapter_file = adapter_path / "adapters.safetensors"
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_file.is_file() or not adapter_config.is_file():
        raise RuntimeError("MLX-LM did not emit the required adapter artifact pair.")
    losses = evidence.get("train_losses")
    if (
        not isinstance(losses, list)
        or not losses
        or any(not math.isfinite(loss) for loss in losses)
    ):
        raise RuntimeError("MLX-LM did not report a finite measured training loss.")
    if evidence.get("optimizer_update_observed") is not True:
        raise RuntimeError("MLX-LM did not prove a non-skipped optimizer update.")
    if action == "pilot" and evidence.get("completed_optimizer_updates", 0) < 2:
        raise RuntimeError("MLX-LM pilot requires at least two completed optimizer updates.")
    validation_losses = evidence.get("validation_losses")
    if (
        valid_examples > 0
        and (
            not isinstance(validation_losses, list)
            or not validation_losses
            or any(not math.isfinite(loss) for loss in validation_losses)
        )
    ):
        raise RuntimeError("MLX-LM did not report finite validation loss evidence.")
    saved_parameters = mx.load(str(adapter_file))
    if sorted(saved_parameters) != evidence.get("trainable_parameter_names"):
        raise RuntimeError(
            "The saved MLX adapter does not exactly match the proven trainable set."
        )
    emitted_config = json.loads(adapter_config.read_text(encoding="utf-8"))
    emitted_lora = emitted_config.get("lora_parameters")
    expected_resolved = evidence["trainable_target_binding"]["resolved_layer_keys"]
    if (
        not isinstance(emitted_lora, dict)
        or emitted_lora.get("keys") != expected_resolved
        or emitted_lora.get("rank") != candidate["rank"]
        or float(emitted_lora.get("scale", float("nan"))) != expected_scale
    ):
        raise RuntimeError("The emitted MLX adapter config is not plan-bound.")
    manifest = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in (adapter_config, adapter_file)
    ]
    scope = {
        "bounded-smoke": "bounded-compiler-smoke-not-pilot-evidence",
        "pilot": "uninterrupted-pilot",
        "full": "uninterrupted-full-train",
    }[action]
    metrics = {
        "schema_version": "aptus.runtime-metrics.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "method": candidate["method"],
        "training_runtime": "mlx-lm",
        "compute_backend": "mps",
        "compiler_id": candidate["runtime_contract"]["compiler_id"],
        "scope": scope,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "micro_iterations": required_iterations,
        "global_step": required_iterations,
        "gradient_accumulation_steps": accumulation,
        "optimizer_update_opportunities": evidence["optimizer_update_opportunities"],
        "completed_optimizer_updates": evidence["completed_optimizer_updates"],
        "train_examples": train_examples,
        "validation_examples": valid_examples,
        "source_train_examples": source_train_examples,
        "source_validation_examples": source_validation_examples,
        "max_epochs": int(plan["target"]["max_epochs"]),
        "distribution": "single",
        "actual_world_size": 1,
        "measured_peak_bytes": int(mx.get_peak_memory()),
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "memory_metric_backend": "mlx",
        "model_load_binding": evidence["model_load_binding"],
        "unified_memory_admission": memory_admission,
        "finite_train_loss": True,
        "train_loss_observations": losses,
        "finite_validation_loss": bool(validation_losses) if valid_examples else True,
        "validation_loss_observations": validation_losses,
        "optimizer_update_observed": True,
        "trainable_target_binding": evidence["trainable_target_binding"],
        "adapter_delta_l1": evidence["adapter_delta_l1"],
        "changed_adapter_tensor_count": evidence["changed_adapter_tensor_count"],
        "adapter_path": str(adapter_path.relative_to(ROOT)),
        "adapter_manifest": manifest,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = adapter_path.parent / "training-metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"MLX-LM {action} training completed: {metrics_path}")
    return 0


def main() -> int:
    plan, _candidate = load_contract()
    parser = argparse.ArgumentParser(description="Run the Aptus MLX-LM compiler slice.")
    parser.add_argument("--model", default=plan["model"]["model_id"])
    parser.add_argument("--data", type=Path, default=Path("data/mlx"))
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--bounded-smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--confirm-full-train", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    arguments = parser.parse_args()
    if arguments.iters <= 0:
        parser.error("--iters must be positive.")
    if arguments.resume_from is not None:
        parser.error(
            "--resume-from is unsupported for MLX-LM; runs are uninterrupted from scratch."
        )
    return run_smoke(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
'''


MLX_RUN_SCRIPT = r'''#!/usr/bin/env python3
"""Lease, launch, and verify one uninterrupted Aptus MLX-LM action."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
from plan_contract import (
    bundle_fingerprint,
    validate_bundle_manifest,
    validate_plan_payload,
)
from runtime_lease import portable_execution_lease, run_with_lease


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_plan() -> dict:
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    errors = validate_plan_payload(plan, root=ROOT, verify_dataset=True)
    errors += validate_bundle_manifest(ROOT)
    if errors:
        raise RuntimeError("Invalid Aptus bundle: " + " | ".join(errors))
    runtime = plan["recommended"].get("runtime_contract")
    if (
        not isinstance(runtime, dict)
        or runtime.get("training_runtime") != "mlx-lm"
        or plan["recommended"].get("method") not in {"lora", "qlora"}
    ):
        raise RuntimeError("The selected plan is not an executable MLX-LM adapter contract.")
    return plan


def require_parent(parent: Path, expected: Path) -> Path:
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise RuntimeError("Aptus output parent must be a real directory.")
    parent.mkdir(mode=0o700, exist_ok=True)
    if parent.is_symlink() or parent.resolve() != expected.resolve():
        raise RuntimeError("Aptus output parent escapes the bundle root.")
    return parent.resolve()


def claim_output(plan: dict, action: str, requested: Path | None) -> Path:
    if action == "full":
        parent = require_parent(ROOT / "runs", ROOT / "runs")
        if requested is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            requested = parent / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
        unresolved = requested.expanduser()
        if unresolved.is_symlink():
            raise RuntimeError("Aptus run output cannot be a symlink.")
        output = unresolved.resolve()
        if output.parent != parent or not output.name.startswith("run_"):
            raise RuntimeError("Full MLX output must be a ROOT/runs/run_* child.")
    else:
        parent = require_parent(ROOT / "pilot-output", ROOT / "pilot-output")
        if requested is not None:
            raise RuntimeError("Only confirmed full training accepts --output-dir.")
        output = parent / f"{action}_{uuid.uuid4().hex}"
    if output.exists():
        raise RuntimeError(f"Aptus refuses to reuse output: {output}")
    output.mkdir(mode=0o700)
    marker = {
        "schema_version": "aptus.mlx-run-output.v1",
        "run_id": output.name,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / ".aptus-run.json", marker)
    return output


def file_entry(path: Path, root: Path) -> dict:
    resolved = path.resolve(strict=True)
    if root.resolve() not in resolved.parents:
        raise RuntimeError("MLX artifact escapes its owned output root.")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def require_training_metrics(plan: dict, metrics: dict, action: str) -> None:
    candidate = plan["recommended"]
    scope = {
        "bounded-smoke": "bounded-compiler-smoke-not-pilot-evidence",
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
        "compiler_id": candidate["runtime_contract"]["compiler_id"],
        "scope": scope,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
    }
    if not isinstance(metrics, dict) or any(
        metrics.get(name) != value for name, value in expected.items()
    ):
        raise RuntimeError("MLX training metrics do not bind the requested action.")
    if metrics.get("model_load_binding") != {
        "schema_version": "aptus.mlx-model-load-binding.v1",
        "model_id": plan["model"]["model_id"],
        "model_revision": plan["model"]["revision"],
        "resolved_local_snapshot": True,
        "trust_remote_code": False,
    }:
        raise RuntimeError("MLX training metrics do not prove a pinned local safe model load.")
    updates = metrics.get("completed_optimizer_updates")
    minimum_updates = 2 if action == "pilot" else 1
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < minimum_updates:
        raise RuntimeError("MLX training metrics do not prove enough optimizer updates.")
    if metrics.get("finite_train_loss") is not True or metrics.get("optimizer_update_observed") is not True:
        raise RuntimeError("MLX training metrics do not prove a finite updated run.")
    if metrics.get("validation_examples", 0) and metrics.get("finite_validation_loss") is not True:
        raise RuntimeError("MLX training metrics do not prove finite validation loss.")


def require_full_admission(plan: dict) -> dict:
    report_path = ROOT / "validation-report.json"
    pilot_path = ROOT / "pilot-output" / "metrics.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        pilot_metrics = json.loads(pilot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Confirmed full MLX training requires a current pilot-pass attestation."
        ) from error
    candidate = plan["recommended"]
    bindings = report.get("bindings") if isinstance(report, dict) else None
    expected_bindings = {
        "bundle": bundle_fingerprint(ROOT),
        "dataset": plan["dataset"]["source_sha256"],
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "pilot_metrics": sha256(pilot_path),
    }
    if (
        not isinstance(report, dict)
        or report.get("state")
        not in {"pilot-pass", "execution-approved", "measured-run-pass"}
        or report.get("artifact_fingerprint") != expected_bindings["bundle"]
        or not isinstance(bindings, dict)
        or any(bindings.get(name) != value for name, value in expected_bindings.items())
        or report.get("pilot_metrics") != pilot_metrics
    ):
        raise RuntimeError(
            "Confirmed full MLX training requires exact current pilot bindings."
        )
    output_value = pilot_metrics.get("output_dir")
    if not isinstance(output_value, str):
        raise RuntimeError("The MLX pilot attestation has no owned output directory.")
    from validate import require_completed_run

    verified_pilot = require_completed_run(
        plan, Path(output_value), action="pilot"
    )
    if verified_pilot != pilot_metrics:
        raise RuntimeError("The MLX pilot copy differs from its verified owned run.")
    from train import current_available_unified_memory_bytes

    reserve = max(
        int(plan["hardware"].get("reserve_per_device_bytes", 0)), 8 * 1024**3
    )
    measured_peak = pilot_metrics.get("measured_peak_bytes")
    if (
        not isinstance(measured_peak, int)
        or isinstance(measured_peak, bool)
        or measured_peak <= 0
    ):
        raise RuntimeError("The MLX pilot has no positive measured peak.")
    available = current_available_unified_memory_bytes()
    required_memory = measured_peak + reserve
    if available < required_memory:
        raise RuntimeError(
            "Current unified-memory headroom is below the measured MLX pilot peak plus reserve."
        )
    disk_free = shutil.disk_usage(ROOT).free
    artifact_manifest = pilot_metrics.get("artifact_manifest")
    adapter_manifest = pilot_metrics.get("adapter_manifest")
    if (
        not isinstance(artifact_manifest, dict)
        or not isinstance(artifact_manifest.get("total_bytes"), int)
        or not isinstance(adapter_manifest, list)
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] < 0
            for item in adapter_manifest
        )
    ):
        raise RuntimeError("The verified MLX pilot has no usable disk evidence.")
    pilot_artifact_bytes = artifact_manifest["total_bytes"]
    measured_adapter_bytes = sum(item["size_bytes"] for item in adapter_manifest)
    planned_export_bytes = int(candidate["final_export_bytes"])
    required_disk = max(
        int(candidate["required_disk_bytes"]),
        pilot_artifact_bytes + max(planned_export_bytes, measured_adapter_bytes),
    )
    if required_disk <= 0 or disk_free < required_disk:
        raise RuntimeError("Current disk headroom is below the plan-bound MLX requirement.")
    return {
        "pilot_metrics_sha256": sha256(pilot_path),
        "measured_pilot_peak_bytes": measured_peak,
        "available_unified_memory_bytes": available,
        "reserve_bytes": reserve,
        "required_available_bytes": required_memory,
        "disk_free_bytes": disk_free,
        "required_disk_bytes": required_disk,
        "pilot_artifact_bytes": pilot_artifact_bytes,
        "measured_adapter_bytes": measured_adapter_bytes,
        "planned_final_export_bytes": planned_export_bytes,
    }


def verify_adapter_manifest(root: Path, metrics: dict) -> list[dict]:
    adapter_path = (ROOT / str(metrics.get("adapter_path", ""))).resolve()
    if root.resolve() not in adapter_path.parents:
        raise RuntimeError("MLX adapter path is outside its owned output root.")
    expected = metrics.get("adapter_manifest")
    if not isinstance(expected, list) or not expected:
        raise RuntimeError("MLX training metrics contain no adapter manifest.")
    observed = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in adapter_path.iterdir() if item.is_file())
    ]
    if observed != expected:
        raise RuntimeError("MLX adapter artifacts changed after training.")
    return observed


def finalize(plan: dict, root: Path, action: str) -> dict:
    training_metrics_path = root / "training-metrics.json"
    try:
        metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("MLX training metrics are missing or unreadable.") from error
    require_training_metrics(plan, metrics, action)
    verify_adapter_manifest(root, metrics)
    adapter_path = (ROOT / metrics["adapter_path"]).resolve(strict=True)
    reload_evidence = None
    reload_path = root / "reload-evidence.json"
    if action in {"pilot", "full"}:
        completed = run_with_lease(
            [
                sys.executable,
                str(ROOT / "reload.py"),
                "--adapter-path",
                str(adapter_path),
                "--training-metrics",
                str(training_metrics_path),
                "--output",
                str(reload_path),
                "--expected-parent-pid",
                str(os.getpid()),
            ],
            cwd=ROOT,
        )
        if completed.returncode:
            raise RuntimeError("Fresh-process MLX adapter reload and generation failed.")
        reload_evidence = json.loads(reload_path.read_text(encoding="utf-8"))
        if (
            not isinstance(reload_evidence, dict)
            or reload_evidence.get("schema_version") != "aptus.mlx-reload-evidence.v1"
            or reload_evidence.get("candidate_id") != plan["recommended"]["candidate_id"]
            or reload_evidence.get("fresh_process_observed") is not True
            or reload_evidence.get("generation_tokens", 0) < 1
            or reload_evidence.get("generation_tokens", 0) > 4
        ):
            raise RuntimeError("MLX reload evidence does not prove bounded fresh-process generation.")
    artifact_paths = [
        root / ".aptus-run.json",
        training_metrics_path,
        adapter_path / "adapter_config.json",
        adapter_path / "adapters.safetensors",
    ]
    if reload_evidence is not None:
        artifact_paths.append(reload_path)
    files = sorted(
        (file_entry(path, root) for path in artifact_paths),
        key=lambda item: item["path"],
    )
    artifact_manifest = {
        "schema_version": "aptus.mlx-artifact-manifest.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "files": files,
        "total_bytes": sum(item["size_bytes"] for item in files),
    }
    artifact_manifest_path = root / "artifact-manifest.json"
    write_json(artifact_manifest_path, artifact_manifest)
    final_export = None
    if action == "full":
        final_files = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(item for item in adapter_path.iterdir() if item.is_file())
        ]
        final_export = {
            "schema_version": "aptus.mlx-final-export.v1",
            "verification_level": "immutable-adapter-file-tree",
            "plan_id": plan["plan_id"],
            "candidate_id": plan["recommended"]["candidate_id"],
            "model_revision": plan["model"]["revision"],
            "dataset_sha256": plan["dataset"]["source_sha256"],
            "method": plan["recommended"]["method"],
            "training_runtime": "mlx-lm",
            "compute_backend": "mps",
            "distribution": "single",
            "world_size": 1,
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
            "files": final_files,
            "total_bytes": sum(item["size_bytes"] for item in final_files),
            "artifact_manifest_sha256": sha256(artifact_manifest_path),
            "reload_evidence_sha256": sha256(reload_path),
        }
        write_json(root / "final-export.json", final_export)
    completed_metrics = {
        **metrics,
        "run_id": root.name,
        "output_dir": str(root.resolve()),
        "run_marker_sha256": sha256(root / ".aptus-run.json"),
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_sha256": sha256(artifact_manifest_path),
        "reload_evidence": reload_evidence,
        "reload_evidence_sha256": sha256(reload_path) if reload_evidence else None,
        "final_export": final_export,
        "run_completed": True,
    }
    write_json(root / "metrics.json", completed_metrics)
    return completed_metrics


def promote_full_completion(
    plan: dict, root: Path, metrics: dict, admission: dict
) -> None:
    report_path = ROOT / "validation-report.json"
    pilot_path = ROOT / "pilot-output" / "metrics.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("MLX validation report disappeared before promotion.") from error
    candidate = plan["recommended"]
    bindings = report.get("bindings") if isinstance(report, dict) else None
    expected_bindings = {
        "bundle": bundle_fingerprint(ROOT),
        "dataset": plan["dataset"]["source_sha256"],
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "pilot_metrics": admission["pilot_metrics_sha256"],
    }
    if (
        not isinstance(report, dict)
        or report.get("state")
        not in {"pilot-pass", "execution-approved", "measured-run-pass"}
        or report.get("artifact_fingerprint") != expected_bindings["bundle"]
        or not isinstance(bindings, dict)
        or any(bindings.get(name) != value for name, value in expected_bindings.items())
        or sha256(pilot_path) != admission["pilot_metrics_sha256"]
    ):
        raise RuntimeError("MLX pilot attestation changed during full training.")
    export_path = root / "final-export.json"
    metrics_path = root / "metrics.json"
    final_export = metrics["final_export"]
    final_report = {
        "path": str((root / "final").resolve()),
        "manifest_sha256": sha256(export_path),
        "total_bytes": final_export["total_bytes"],
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "distribution": "single",
        "world_size": 1,
        "training_runtime": "mlx-lm",
        "artifact_manifest_sha256": metrics["artifact_manifest_sha256"],
        "reload_evidence_sha256": metrics["reload_evidence_sha256"],
        "export_contract": final_export,
    }
    measured_report = {
        "output_dir": str(root.resolve()),
        "metrics_sha256": sha256(metrics_path),
        "global_step": metrics["global_step"],
        "completed_optimizer_updates": metrics["completed_optimizer_updates"],
        "measured_peak_bytes": metrics["measured_peak_bytes"],
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "distribution": "single",
        "world_size": 1,
        "training_runtime": "mlx-lm",
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
    }
    report.update(
        state="measured-run-pass",
        validation_level="measured-run",
        validated_at=datetime.now(timezone.utc).isoformat(),
        measured_run_completed_at=datetime.now(timezone.utc).isoformat(),
        final_export=final_report,
        measured_run=measured_report,
        latest_recheck=None,
    )
    for name in (
        "active_run",
        "measured_run_pending_at",
        "pending_final_export",
        "pending_measured_run",
    ):
        report.pop(name, None)
    write_json(report_path, report)


def launch(arguments: argparse.Namespace) -> int:
    plan = load_plan()
    selected = [
        name
        for name, enabled in (
            ("bounded-smoke", arguments.bounded_smoke),
            ("pilot", arguments.pilot),
            ("full", arguments.confirm_full_train),
        )
        if enabled
    ]
    if len(selected) != 1:
        raise RuntimeError("Choose exactly one MLX-LM action.")
    if arguments.resume_from is not None:
        raise RuntimeError(
            "MLX-LM resume is unsupported. Runs start from the pinned base model."
        )
    action = selected[0]
    full_admission = require_full_admission(plan) if action == "full" else None
    output = claim_output(plan, action, arguments.output_dir)
    adapter_path = output / ("final" if action == "full" else "adapters")
    train_action = "--confirm-full-train" if action == "full" else f"--{action}"
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        train_action,
        "--adapter-path",
        str(adapter_path),
    ]
    if action != "full":
        command.extend(("--iters", str(arguments.iters)))
    if arguments.model:
        command.extend(("--model", arguments.model))
    if arguments.data:
        command.extend(("--data", str(arguments.data)))
    completed = run_with_lease(command, cwd=ROOT)
    if completed.returncode:
        return completed.returncode
    metrics = finalize(plan, output, action)
    if action == "full":
        from validate import require_completed_run

        verified = require_completed_run(plan, output, action="full")
        if verified != metrics or full_admission is None:
            raise RuntimeError("MLX full completion changed before promotion.")
        promote_full_completion(plan, output, metrics, full_admission)
    print(f"Aptus MLX-LM {action} completed: {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch an Aptus MLX-LM action.")
    parser.add_argument("--bounded-smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--confirm-full-train", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--iters", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.iters <= 0:
        parser.error("--iters must be positive.")
    if arguments.resume_from is not None:
        parser.error("--resume-from is unsupported for MLX-LM.")
    with portable_execution_lease(ROOT, action="mlx-uninterrupted"):
        return launch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
'''


MLX_RELOAD_SCRIPT = r'''#!/usr/bin/env python3
"""Reload one MLX adapter in a fresh process and perform bounded generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
from plan_contract import validate_bundle_manifest, validate_plan_payload
from train import (
    download_pinned_model,
    require_method_model,
    require_unified_memory_admission,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.expected_parent_pid <= 0 or os.getppid() != arguments.expected_parent_pid:
        raise RuntimeError("Reload verifier is not the expected fresh child process.")
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    errors = validate_plan_payload(plan, root=ROOT, verify_dataset=True)
    errors += validate_bundle_manifest(ROOT)
    if errors:
        raise RuntimeError("Invalid Aptus bundle: " + " | ".join(errors))
    metrics_path = arguments.training_metrics.resolve(strict=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    candidate = plan["recommended"]
    if (
        not isinstance(metrics, dict)
        or metrics.get("plan_id") != plan["plan_id"]
        or metrics.get("candidate_id") != candidate["candidate_id"]
        or metrics.get("action") not in {"pilot", "full"}
        or metrics.get("execution_semantics") != "uninterrupted"
        or metrics.get("resume_supported") is not False
    ):
        raise RuntimeError("Reload verifier received unbound training metrics.")
    adapter_path = arguments.adapter_path.resolve(strict=True)
    output_root = metrics_path.parent.resolve()
    if output_root not in adapter_path.parents:
        raise RuntimeError("Reload adapter escapes the owned run root.")
    expected_manifest = metrics.get("adapter_manifest")
    observed_manifest = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in adapter_path.iterdir() if item.is_file())
    ]
    if not isinstance(expected_manifest, list) or observed_manifest != expected_manifest:
        raise RuntimeError("Reload adapter does not match its immutable training manifest.")
    output = arguments.output.resolve()
    if output.parent != output_root or output.exists():
        raise RuntimeError("Reload evidence path is not a fresh file in the owned run root.")

    import mlx.core as mx
    from mlx_lm import load, stream_generate

    admission = require_unified_memory_admission(plan)
    model_path = download_pinned_model(plan, plan["model"]["model_id"])
    require_method_model(candidate, model_path)
    mx.reset_peak_memory()
    model, tokenizer = load(
        str(model_path),
        adapter_path=str(adapter_path),
        tokenizer_config={"trust_remote_code": False},
    )
    responses = list(
        stream_generate(
            model,
            tokenizer,
            "Aptus adapter reload verification:",
            max_tokens=4,
        )
    )
    if not responses:
        raise RuntimeError("Fresh-process adapter generation returned no response evidence.")
    generation_tokens = int(responses[-1].generation_tokens)
    if generation_tokens < 1 or generation_tokens > 4:
        raise RuntimeError("Fresh-process adapter generation exceeded its token bound.")
    generated_text = "".join(str(response.text) for response in responses)
    peak = int(mx.get_peak_memory())
    if peak <= 0:
        raise RuntimeError("Fresh-process adapter reload reported no positive MLX peak.")
    evidence = {
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
        "parent_pid": os.getppid(),
        "verifier_pid": os.getpid(),
        "adapter_manifest_sha256": hashlib.sha256(
            json.dumps(
                observed_manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "generation_max_tokens": 4,
        "generation_tokens": generation_tokens,
        "generation_text_sha256": hashlib.sha256(
            generated_text.encode("utf-8")
        ).hexdigest(),
        "measured_peak_bytes": peak,
        "unified_memory_admission": admission,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"Fresh-process MLX adapter reload passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


MLX_PREFLIGHT_SCRIPT = r'''#!/usr/bin/env python3
"""Fail-closed MLX-LM dependency and uninterrupted-run preflight."""

from __future__ import annotations

import importlib.metadata
import platform


def main() -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX-LM requires Apple silicon macOS.")
    expected = {"mlx": "0.31.2", "mlx-lm": "0.31.3"}
    for package, version in expected.items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"Expected {package}=={version}.")
    print("MLX-LM dependencies and Apple silicon platform are present.")
    print("Pilot and full runs are uninterrupted from scratch; crash-resume remains unsupported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


MLX_VALIDATE_SCRIPT = r'''#!/usr/bin/env python3
"""Validate and monotonically attest an Aptus MLX-LM bundle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
from plan_contract import bundle_fingerprint, validate_bundle_manifest, validate_plan_payload

STATE_RANK = {
    "contract-pass": 1,
    "static-pass": 2,
    "dependency-pass": 3,
    "model-data-pass": 4,
    "measured-preflight-pass": 5,
    "pilot-pass": 6,
    "execution-approved": 7,
    "measured-run-pass": 8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def promote(
    plan: dict,
    state: str,
    *,
    preflight_metrics: dict | None = None,
    pilot_metrics: dict | None = None,
) -> None:
    report_path = ROOT / "validation-report.json"
    previous = None
    try:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    if (
        isinstance(previous, dict)
        and STATE_RANK.get(previous.get("state"), 0) > STATE_RANK[state]
        and stronger_attestation_is_current(previous, plan)
    ):
        previous["latest_recheck"] = {
            "state": state,
            "validation_level": state.removesuffix("-pass"),
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "artifact_fingerprint": bundle_fingerprint(ROOT),
        }
        temporary = report_path.with_name(".validation-report.json.tmp")
        temporary.write_text(
            json.dumps(previous, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, report_path)
        return
    candidate = plan["recommended"]
    bindings = {
        "bundle": bundle_fingerprint(ROOT),
        "dataset": plan["dataset"]["source_sha256"],
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
    }
    if preflight_metrics is not None:
        bindings["preflight_metrics"] = sha256(ROOT / "preflight-metrics.json")
    if pilot_metrics is not None:
        bindings["pilot_metrics"] = sha256(ROOT / "pilot-output" / "metrics.json")
    report = {
        "state": state,
        "findings": [],
        "checked_files": ["bundle-manifest.json", "plan.json", "requirements.txt"],
        "artifact_fingerprint": bindings["bundle"],
        "smoke_command": None,
        "runtime_evidence": [
            f"Observed MLX-LM validation state: {state}.",
            "No model-fit or quality guarantee is implied.",
        ],
        "validation_level": state.removesuffix("-pass"),
        "bindings": bindings,
        "validator_version": "aptus-validator-mlx-v1",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "preflight_metrics": preflight_metrics,
        "pilot_metrics": pilot_metrics,
        "final_export": None,
        "measured_run": None,
        "measured_run_completed_at": None,
        "latest_recheck": None,
    }
    temporary = report_path.with_name(".validation-report.json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)


def current_available_unified_memory_bytes() -> int:
    try:
        completed = subprocess.run(
            ["/usr/bin/vm_stat"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Current Apple unified-memory admission probe failed.") from error
    if completed.returncode:
        raise RuntimeError("Current Apple unified-memory admission probe failed.")
    page_match = re.search(r"page size of\s+(\d+) bytes", completed.stdout)
    if page_match is None:
        raise RuntimeError("vm_stat did not report its page size.")
    counts = {}
    for line in completed.stdout.splitlines():
        match = re.fullmatch(r"([^:]+):\s*([0-9]+)\.", line.strip())
        if match is not None:
            counts[match.group(1)] = int(match.group(2))
    names = ("Pages free", "Pages inactive", "Pages speculative")
    if not any(name in counts for name in names):
        raise RuntimeError("vm_stat did not report available-memory page classes.")
    available = sum(counts.get(name, 0) for name in names) * int(page_match.group(1))
    if available <= 0:
        raise RuntimeError("Current available Apple unified memory is zero or unknown.")
    return available


def require_unified_memory_admission(plan: dict) -> dict:
    memory = plan["recommended"]["memory"]
    point = int(memory["point_estimate_bytes"])
    upper = int(memory["upper_estimate_bytes"])
    reserve = max(
        int(plan["hardware"].get("reserve_per_device_bytes", 0)),
        8 * 1024**3,
    )
    available = current_available_unified_memory_bytes()
    required = max(point, upper) + reserve
    if available < required:
        raise RuntimeError(
            "Current available Apple unified memory is below the candidate upper "
            "estimate plus the required 8 GiB Aptus reserve."
        )
    return {
        "schema_version": "aptus.mlx-unified-memory-admission.v1",
        "available_unified_memory_bytes": available,
        "point_estimate_bytes": point,
        "upper_estimate_bytes": upper,
        "reserve_bytes": reserve,
        "required_available_bytes": required,
    }


def require_model_data(plan: dict) -> None:
    from huggingface_hub import snapshot_download
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load

    require_unified_memory_admission(plan)
    model_path = Path(
        snapshot_download(
            repo_id=plan["model"]["model_id"],
            revision=plan["model"]["revision"],
        )
    ).resolve(strict=True)
    pinned_config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8")
    )
    if pinned_config.get("model_file"):
        raise RuntimeError(
            "MLX-LM custom model_file code is unsupported; Aptus only executes pinned built-in MLX model implementations."
        )
    model, tokenizer, config = load(
        str(model_path),
        lazy=True,
        return_config=True,
        tokenizer_config={"trust_remote_code": False},
    )
    candidate = plan["recommended"]
    quantization = config.get("quantization") or config.get("quantization_config")
    text_config = config.get("text_config")
    if not quantization and isinstance(text_config, dict):
        quantization = text_config.get("quantization_config")
    if candidate["method"] == "qlora" and (
        not isinstance(quantization, dict) or quantization.get("bits") != 4
    ):
        raise RuntimeError(
            "MLX-LM QLoRA model-data validation requires explicit four-bit MLX quantization metadata."
        )
    if candidate["method"] == "lora" and quantization:
        raise RuntimeError(
            "MLX-LM LoRA model-data validation rejects quantized bases because "
            "they would execute QLoRA semantics under a LoRA plan."
        )
    args = types.SimpleNamespace(
        data=str(ROOT / "data" / "mlx"),
        train=True,
        test=False,
        mask_prompt=True,
        hf_dataset=False,
    )
    train, valid, _test = load_dataset(args, tokenizer)
    if not train or not valid:
        raise RuntimeError("MLX-LM train and validation datasets must both be non-empty.")
    max_seq_length = int(plan["target"]["sequence_length"])
    for dataset in (train, valid):
        for index in range(len(dataset)):
            raw = dataset[index]
            if not isinstance(raw, dict) or set(raw) - {"messages", "tools"}:
                raise RuntimeError(
                    "MLX-LM data must use the compiler-normalized messages schema."
                )
            try:
                tokens, prompt_offset = dataset.process(raw)
            except Exception as error:
                raise RuntimeError("MLX-LM dataset tokenization failed closed.") from error
            if not isinstance(prompt_offset, int) or isinstance(prompt_offset, bool):
                raise RuntimeError("MLX-LM prompt masking returned an invalid offset.")
            if prompt_offset <= 0 or prompt_offset >= len(tokens):
                raise RuntimeError(
                    "MLX-LM prompt masking did not preserve non-empty completion supervision."
                )
            if len(tokens) > max_seq_length:
                raise RuntimeError(
                    "Pinned MLX-LM 0.31.3 right-truncates overlength rows, which cannot "
                    "honor Aptus completion-first, left-truncate-prompt policy. Shorten "
                    "the row or increase sequence_length; Aptus refuses this dataset."
                )
    del model, tokenizer, train, valid
    gc.collect()


def require_runtime_metrics(
    plan: dict, metrics: dict, *, action: str = "bounded-smoke"
) -> dict:
    candidate = plan["recommended"]
    scope = {
        "bounded-smoke": "bounded-compiler-smoke-not-pilot-evidence",
        "pilot": "uninterrupted-pilot",
        "full": "uninterrupted-full-train",
    }.get(action)
    if scope is None:
        raise RuntimeError("Unknown MLX-LM runtime metrics action.")
    required = {
        "schema_version": "aptus.runtime-metrics.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "method": candidate["method"],
        "training_runtime": "mlx-lm",
        "compute_backend": "mps",
        "compiler_id": candidate["runtime_contract"]["compiler_id"],
        "memory_metric_backend": "mlx",
        "scope": scope,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "finite_train_loss": True,
        "optimizer_update_observed": True,
    }
    if any(metrics.get(key) != value for key, value in required.items()):
        raise RuntimeError("MLX-LM runtime metrics do not bind the selected candidate and proof scope.")
    if metrics.get("model_load_binding") != {
        "schema_version": "aptus.mlx-model-load-binding.v1",
        "model_id": plan["model"]["model_id"],
        "model_revision": plan["model"]["revision"],
        "resolved_local_snapshot": True,
        "trust_remote_code": False,
    }:
        raise RuntimeError("MLX-LM runtime metrics do not prove a pinned local safe model load.")
    if (
        not isinstance(metrics.get("measured_peak_bytes"), int)
        or isinstance(metrics.get("measured_peak_bytes"), bool)
        or metrics["measured_peak_bytes"] <= 0
        or not isinstance(metrics.get("active_memory_bytes"), int)
        or isinstance(metrics.get("active_memory_bytes"), bool)
        or metrics["active_memory_bytes"] < 0
        or not isinstance(metrics.get("cache_memory_bytes"), int)
        or isinstance(metrics.get("cache_memory_bytes"), bool)
        or metrics["cache_memory_bytes"] < 0
        or "free_vram_bytes" in metrics
    ):
        raise RuntimeError("MLX-LM runtime metrics require a positive measured_peak_bytes value.")
    losses = metrics.get("train_loss_observations")
    if (
        not isinstance(losses, list)
        or not losses
        or any(
            not isinstance(loss, (int, float))
            or isinstance(loss, bool)
            or not math.isfinite(loss)
            for loss in losses
        )
    ):
        raise RuntimeError("MLX-LM runtime metrics require finite measured train losses.")
    update_opportunities = metrics.get("optimizer_update_opportunities")
    completed_updates = metrics.get("completed_optimizer_updates")
    accumulation = int(candidate["gradient_accumulation_steps"])
    micro_iterations = metrics.get("micro_iterations")
    minimum_updates = 2 if action == "pilot" else 1
    if (
        not isinstance(micro_iterations, int)
        or isinstance(micro_iterations, bool)
        or micro_iterations <= 0
        or micro_iterations % accumulation
        or metrics.get("global_step") != micro_iterations
        or metrics.get("gradient_accumulation_steps") != accumulation
        or not isinstance(update_opportunities, int)
        or isinstance(update_opportunities, bool)
        or update_opportunities < 1
        or update_opportunities != micro_iterations // accumulation
        or not isinstance(completed_updates, int)
        or isinstance(completed_updates, bool)
        or completed_updates != update_opportunities
        or completed_updates < minimum_updates
    ):
        raise RuntimeError("MLX-LM runtime metrics do not prove completed optimizer updates.")
    split_contract = json.loads(
        (ROOT / "data" / "mlx" / "split-contract.json").read_text(encoding="utf-8")
    )
    splits = split_contract.get("splits", {})
    train_split = splits.get("train", {})
    valid_split = splits.get("valid", {})
    train_examples = metrics.get("train_examples")
    validation_examples = metrics.get("validation_examples")
    validation_losses = metrics.get("validation_loss_observations")
    if (
        split_contract.get("schema_version") != "aptus.mlx-split.v1"
        or split_contract.get("micro_batch_size") != candidate["micro_batch_size"]
        or not isinstance(train_examples, int)
        or isinstance(train_examples, bool)
        or train_examples <= 0
        or not isinstance(validation_examples, int)
        or isinstance(validation_examples, bool)
        or validation_examples <= 0
        or train_split.get("compiled_row_count") != train_examples
        or valid_split.get("compiled_row_count") != validation_examples
        or metrics.get("source_train_examples") != train_split.get("source_row_count")
        or metrics.get("source_validation_examples") != valid_split.get("source_row_count")
        or train_examples % int(candidate["micro_batch_size"])
        or validation_examples % int(candidate["micro_batch_size"])
        or metrics.get("max_epochs") != int(plan["target"]["max_epochs"])
        or (
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
    ):
        raise RuntimeError("MLX-LM runtime metrics require finite validation loss evidence.")
    if action == "pilot" and micro_iterations != 2 * accumulation:
        raise RuntimeError("MLX-LM pilot metrics are not the bounded two-update schedule.")
    if action == "bounded-smoke" and micro_iterations > 8:
        raise RuntimeError("MLX-LM measured-preflight metrics exceed the eight-iteration bound.")
    if action == "full":
        batches_per_epoch = train_examples // int(candidate["micro_batch_size"])
        epoch_iterations = batches_per_epoch * int(plan["target"]["max_epochs"])
        expected_iterations = math.ceil(epoch_iterations / accumulation) * accumulation
        if micro_iterations != expected_iterations:
            raise RuntimeError("MLX-LM full metrics do not match the dataset-derived epoch schedule.")
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
        raise RuntimeError("MLX-LM runtime metrics require a positive finite adapter delta.")
    binding = metrics.get("trainable_target_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("MLX-LM runtime metrics require an exact trainable-target binding.")
    planned = candidate["target_modules"]
    layer_count = int(plan["model"]["layers"])
    expected_instances = len(planned) * layer_count
    target_counts = binding.get("target_instance_counts")
    descriptor_sha256 = binding.get("descriptor_sha256")
    descriptor_payload = {
        key: value for key, value in binding.items() if key != "descriptor_sha256"
    }
    expected_descriptor_sha256 = hashlib.sha256(
        json.dumps(
            descriptor_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if (
        binding.get("schema_version") != "aptus.mlx-trainable-target-binding.v1"
        or binding.get("planned_target_modules") != planned
        or binding.get("transformer_layer_count") != layer_count
        or binding.get("expected_adapter_target_instance_count") != expected_instances
        or binding.get("adapter_target_instance_count") != expected_instances
        or binding.get("trainable_tensor_count") != expected_instances * 2
        or not isinstance(target_counts, dict)
        or target_counts != {target: layer_count for target in planned}
        or not isinstance(binding.get("resolved_layer_keys"), list)
        or len(binding["resolved_layer_keys"]) != len(planned)
        or len(set(binding["resolved_layer_keys"])) != len(planned)
        or not isinstance(descriptor_sha256, str)
        or descriptor_sha256 != expected_descriptor_sha256
    ):
        raise RuntimeError("MLX-LM trainable-target binding is not exact for the plan.")
    admission = metrics.get("unified_memory_admission")
    reserve = max(int(plan["hardware"].get("reserve_per_device_bytes", 0)), 8 * 1024**3)
    point = int(candidate["memory"]["point_estimate_bytes"])
    upper = int(candidate["memory"]["upper_estimate_bytes"])
    if (
        not isinstance(admission, dict)
        or admission.get("schema_version") != "aptus.mlx-unified-memory-admission.v1"
        or admission.get("point_estimate_bytes") != point
        or admission.get("upper_estimate_bytes") != upper
        or admission.get("reserve_bytes") != reserve
        or admission.get("required_available_bytes") != max(point, upper) + reserve
        or not isinstance(admission.get("available_unified_memory_bytes"), int)
        or admission["available_unified_memory_bytes"] < admission["required_available_bytes"]
        or "free_vram_bytes" in admission
    ):
        raise RuntimeError("MLX-LM runtime metrics do not bind a passing live unified-memory admission.")
    return metrics


def require_completed_run(plan: dict, root: Path, *, action: str) -> dict:
    expected_parent = (
        (ROOT / "pilot-output").resolve()
        if action in {"bounded-smoke", "pilot"}
        else (ROOT / "runs").resolve()
    )
    if root.is_symlink():
        raise RuntimeError("MLX completed-run root cannot be a symlink.")
    resolved = root.resolve(strict=True)
    expected_prefix = "run_" if action == "full" else action + "_"
    if resolved.parent != expected_parent or not resolved.name.startswith(expected_prefix):
        raise RuntimeError("MLX completed-run root is outside its owned action directory.")
    metrics_path = resolved / "metrics.json"
    if metrics_path.is_symlink():
        raise RuntimeError("MLX completed metrics cannot be a symlink.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    require_runtime_metrics(plan, metrics, action=action)
    if (
        metrics.get("run_completed") is not True
        or metrics.get("run_id") != resolved.name
        or metrics.get("output_dir") != str(resolved)
        or metrics.get("execution_semantics") != "uninterrupted"
        or metrics.get("resume_supported") is not False
    ):
        raise RuntimeError("MLX completed metrics do not bind the uninterrupted owned run.")
    marker_path = resolved / ".aptus-run.json"
    if marker_path.is_symlink():
        raise RuntimeError("MLX run marker cannot be a symlink.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
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
    if any(marker.get(name) != value for name, value in marker_expected.items()):
        raise RuntimeError("MLX run marker does not bind the plan and action.")
    if metrics.get("run_marker_sha256") != sha256(marker_path):
        raise RuntimeError("MLX completed metrics do not bind the immutable run marker.")
    training_metrics_path = resolved / "training-metrics.json"
    if training_metrics_path.is_symlink():
        raise RuntimeError("MLX training metrics cannot be a symlink.")
    training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
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
    if {
        name: value for name, value in metrics.items() if name not in completion_fields
    } != training_metrics:
        raise RuntimeError("MLX completed metrics do not preserve the exact training metrics.")
    adapter_path = resolved / ("final" if action == "full" else "adapters")
    if (
        adapter_path.is_symlink()
        or metrics.get("adapter_path") != adapter_path.relative_to(ROOT).as_posix()
    ):
        raise RuntimeError("MLX completed metrics do not bind the action adapter directory.")
    adapter_manifest = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in adapter_path.iterdir() if item.is_file())
    ]
    if (
        any(path.is_symlink() for path in adapter_path.iterdir())
        or
        metrics.get("adapter_manifest") != adapter_manifest
        or [item["path"] for item in adapter_manifest]
        != ["adapter_config.json", "adapters.safetensors"]
    ):
        raise RuntimeError("MLX completed metrics do not bind the exact adapter pair.")
    if action in {"pilot", "full"}:
        reload_path = resolved / "reload-evidence.json"
        if reload_path.is_symlink():
            raise RuntimeError("MLX reload evidence cannot be a symlink.")
        reload_evidence = json.loads(reload_path.read_text(encoding="utf-8"))
        reload_expected = {
            "schema_version": "aptus.mlx-reload-evidence.v1",
            "plan_id": plan["plan_id"],
            "candidate_id": plan["recommended"]["candidate_id"],
            "model_revision": plan["model"]["revision"],
            "dataset_sha256": plan["dataset"]["source_sha256"],
            "method": plan["recommended"]["method"],
            "training_runtime": "mlx-lm",
            "compute_backend": "mps",
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
            "fresh_process_observed": True,
            "generation_max_tokens": 4,
        }
        admission = reload_evidence.get("unified_memory_admission", {})
        memory = plan["recommended"]["memory"]
        reserve = max(
            int(plan["hardware"].get("reserve_per_device_bytes", 0)), 8 * 1024**3
        )
        required = max(
            int(memory["point_estimate_bytes"]), int(memory["upper_estimate_bytes"])
        ) + reserve
        expected_adapter_digest = hashlib.sha256(
            json.dumps(
                adapter_manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if (
            any(reload_evidence.get(name) != value for name, value in reload_expected.items())
            or metrics.get("reload_evidence") != reload_evidence
            or metrics.get("reload_evidence_sha256") != sha256(reload_path)
            or not isinstance(reload_evidence.get("generation_tokens"), int)
            or isinstance(reload_evidence.get("generation_tokens"), bool)
            or not 1 <= reload_evidence["generation_tokens"] <= 4
            or not isinstance(reload_evidence.get("measured_peak_bytes"), int)
            or isinstance(reload_evidence.get("measured_peak_bytes"), bool)
            or reload_evidence["measured_peak_bytes"] <= 0
            or not isinstance(reload_evidence.get("parent_pid"), int)
            or not isinstance(reload_evidence.get("verifier_pid"), int)
            or reload_evidence["parent_pid"] <= 0
            or reload_evidence["verifier_pid"] <= 0
            or reload_evidence["parent_pid"] == reload_evidence["verifier_pid"]
            or reload_evidence.get("adapter_manifest_sha256") != expected_adapter_digest
            or not isinstance(reload_evidence.get("generation_text_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", reload_evidence["generation_text_sha256"])
            is None
            or not isinstance(admission, dict)
            or admission.get("schema_version")
            != "aptus.mlx-unified-memory-admission.v1"
            or admission.get("point_estimate_bytes") != memory["point_estimate_bytes"]
            or admission.get("upper_estimate_bytes") != memory["upper_estimate_bytes"]
            or admission.get("reserve_bytes") != reserve
            or admission.get("required_available_bytes") != required
            or not isinstance(admission.get("available_unified_memory_bytes"), int)
            or isinstance(admission.get("available_unified_memory_bytes"), bool)
            or admission["available_unified_memory_bytes"] < required
            or "free_vram_bytes" in admission
        ):
            raise RuntimeError("MLX completed metrics do not prove fresh-process bounded generation.")
    manifest_path = resolved / "artifact-manifest.json"
    if manifest_path.is_symlink():
        raise RuntimeError("MLX artifact manifest cannot be a symlink.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        metrics.get("artifact_manifest") != manifest
        or metrics.get("artifact_manifest_sha256") != sha256(manifest_path)
        or manifest.get("schema_version") != "aptus.mlx-artifact-manifest.v1"
        or manifest.get("plan_id") != plan["plan_id"]
        or manifest.get("candidate_id") != plan["recommended"]["candidate_id"]
        or manifest.get("action") != action
        or manifest.get("execution_semantics") != "uninterrupted"
        or manifest.get("resume_supported") is not False
    ):
        raise RuntimeError("MLX immutable artifact manifest is missing or unbound.")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("MLX immutable artifact manifest is empty.")
    seen = set()
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError("MLX artifact manifest entry is invalid.")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise RuntimeError("MLX artifact manifest path is unsafe or duplicated.")
        artifact = resolved.joinpath(*relative.parts)
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or entry.get("size_bytes") != artifact.stat().st_size
            or entry.get("sha256") != sha256(artifact)
        ):
            raise RuntimeError("MLX artifact manifest no longer matches the run files.")
        seen.add(relative.as_posix())
        total += artifact.stat().st_size
    if manifest.get("total_bytes") != total:
        raise RuntimeError("MLX artifact manifest total is inconsistent.")
    expected_files = {
        ".aptus-run.json",
        "training-metrics.json",
        f"{adapter_path.name}/adapter_config.json",
        f"{adapter_path.name}/adapters.safetensors",
    }
    if action in {"pilot", "full"}:
        expected_files.add("reload-evidence.json")
    if seen != expected_files:
        raise RuntimeError("MLX artifact manifest does not cover the exact proof files.")
    expected_actual_files = expected_files | {"artifact-manifest.json", "metrics.json"}
    if action == "full":
        expected_actual_files.add("final-export.json")
    actual_files = {
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_actual_files:
        raise RuntimeError("MLX owned run contains an unexpected or missing file.")
    if action == "full":
        export_path = resolved / "final-export.json"
        if export_path.is_symlink():
            raise RuntimeError("MLX final export cannot be a symlink.")
        final_export = json.loads(export_path.read_text(encoding="utf-8"))
        export_expected = {
            "schema_version": "aptus.mlx-final-export.v1",
            "verification_level": "immutable-adapter-file-tree",
            "plan_id": plan["plan_id"],
            "candidate_id": plan["recommended"]["candidate_id"],
            "model_revision": plan["model"]["revision"],
            "dataset_sha256": plan["dataset"]["source_sha256"],
            "method": plan["recommended"]["method"],
            "training_runtime": "mlx-lm",
            "compute_backend": "mps",
            "distribution": "single",
            "world_size": 1,
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
            "files": adapter_manifest,
            "total_bytes": sum(item["size_bytes"] for item in adapter_manifest),
            "artifact_manifest_sha256": sha256(manifest_path),
            "reload_evidence_sha256": sha256(resolved / "reload-evidence.json"),
        }
        if final_export != export_expected or metrics.get("final_export") != final_export:
            raise RuntimeError("MLX final export is missing, mutable, or unbound.")
    elif metrics.get("final_export") is not None:
        raise RuntimeError("Only confirmed full MLX training may emit a final export.")
    return metrics


def stronger_attestation_is_current(previous: dict, plan: dict) -> bool:
    bindings = previous.get("bindings")
    candidate = plan["recommended"]
    expected = {
        "bundle": bundle_fingerprint(ROOT),
        "dataset": plan["dataset"]["source_sha256"],
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
    }
    if (
        not isinstance(bindings, dict)
        or previous.get("artifact_fingerprint") != expected["bundle"]
        or any(bindings.get(name) != value for name, value in expected.items())
    ):
        return False
    rank = STATE_RANK.get(previous.get("state"), 0)
    if rank >= STATE_RANK["measured-preflight-pass"]:
        preflight_path = ROOT / "preflight-metrics.json"
        try:
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            output = Path(preflight["output_dir"])
            verified = require_completed_run(plan, output, action="bounded-smoke")
        except (KeyError, OSError, RuntimeError, json.JSONDecodeError):
            return False
        if (
            verified != preflight
            or bindings.get("preflight_metrics") != sha256(preflight_path)
            or previous.get("preflight_metrics") != preflight
        ):
            return False
    if rank >= STATE_RANK["pilot-pass"]:
        pilot_path = ROOT / "pilot-output" / "metrics.json"
        try:
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            output = Path(pilot["output_dir"])
            verified = require_completed_run(plan, output, action="pilot")
        except (KeyError, OSError, RuntimeError, json.JSONDecodeError):
            return False
        if (
            verified != pilot
            or bindings.get("pilot_metrics") != sha256(pilot_path)
            or previous.get("pilot_metrics") != pilot
        ):
            return False
    if previous.get("state") == "measured-run-pass":
        measured_report = previous.get("measured_run")
        final_report = previous.get("final_export")
        if not isinstance(measured_report, dict) or not isinstance(final_report, dict):
            return False
        try:
            root = Path(measured_report["output_dir"])
            metrics = require_completed_run(plan, root, action="full")
            metrics_path = root / "metrics.json"
            export_path = root / "final-export.json"
        except (KeyError, OSError, RuntimeError):
            return False
        expected_final = {
            "path": str((root / "final").resolve()),
            "manifest_sha256": sha256(export_path),
            "total_bytes": metrics["final_export"]["total_bytes"],
            "plan_id": plan["plan_id"],
            "candidate_id": candidate["candidate_id"],
            "distribution": "single",
            "world_size": 1,
            "training_runtime": "mlx-lm",
            "artifact_manifest_sha256": metrics["artifact_manifest_sha256"],
            "reload_evidence_sha256": metrics["reload_evidence_sha256"],
            "export_contract": metrics["final_export"],
        }
        expected_measured = {
            "output_dir": str(root.resolve()),
            "metrics_sha256": sha256(metrics_path),
            "global_step": metrics["global_step"],
            "completed_optimizer_updates": metrics["completed_optimizer_updates"],
            "measured_peak_bytes": metrics["measured_peak_bytes"],
            "plan_id": plan["plan_id"],
            "candidate_id": candidate["candidate_id"],
            "distribution": "single",
            "world_size": 1,
            "training_runtime": "mlx-lm",
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
        }
        if final_report != expected_final or measured_report != expected_measured:
            return False
    return True


def run_measured_preflight() -> dict:
    before = set((ROOT / "pilot-output").glob("bounded-smoke_*")) if (ROOT / "pilot-output").exists() else set()
    completed = subprocess.run([sys.executable, str(ROOT / "run.py"), "--bounded-smoke"], cwd=ROOT)
    if completed.returncode:
        raise RuntimeError("The bounded MLX-LM compiler smoke failed.")
    after = set((ROOT / "pilot-output").glob("bounded-smoke_*"))
    created = sorted(after - before, key=lambda path: path.stat().st_mtime_ns)
    if len(created) != 1:
        raise RuntimeError("The bounded MLX-LM smoke did not create one owned evidence root.")
    metrics_path = created[0] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    require_completed_run(plan, created[0], action="bounded-smoke")
    destination = ROOT / "preflight-metrics.json"
    destination.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run_pilot(plan: dict) -> dict:
    pilot_root = ROOT / "pilot-output"
    before = set(pilot_root.glob("pilot_*")) if pilot_root.exists() else set()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--pilot"], cwd=ROOT
    )
    if completed.returncode:
        raise RuntimeError("The uninterrupted MLX-LM pilot failed.")
    after = set(pilot_root.glob("pilot_*"))
    created = sorted(after - before, key=lambda path: path.stat().st_mtime_ns)
    if len(created) != 1:
        raise RuntimeError("The MLX-LM pilot did not create one owned evidence root.")
    metrics = require_completed_run(plan, created[0], action="pilot")
    destination = pilot_root / "metrics.json"
    temporary = destination.with_name(".metrics.json.tmp")
    temporary.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        choices=("contract", "static", "dependency", "model-data", "measured-preflight", "pilot"),
        default="contract",
    )
    arguments = parser.parse_args()
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    errors = validate_plan_payload(plan, root=ROOT, verify_dataset=True)
    errors += validate_bundle_manifest(ROOT)
    if errors:
        raise RuntimeError("Invalid Aptus bundle: " + " | ".join(errors))
    states = {
        "contract": "contract-pass",
        "static": "static-pass",
        "dependency": "dependency-pass",
        "model-data": "model-data-pass",
        "measured-preflight": "measured-preflight-pass",
    }
    if arguments.level in {"dependency", "model-data", "measured-preflight", "pilot"}:
        completed = subprocess.run([sys.executable, str(ROOT / "preflight.py")], cwd=ROOT)
        if completed.returncode:
            return completed.returncode
    if arguments.level in {"model-data", "measured-preflight", "pilot"}:
        require_model_data(plan)
    metrics = None
    if arguments.level in {"measured-preflight", "pilot"}:
        metrics = run_measured_preflight()
    if arguments.level != "pilot":
        promote(plan, states[arguments.level], preflight_metrics=metrics)
    if arguments.level == "pilot":
        pilot_metrics = run_pilot(plan)
        promote(
            plan,
            "pilot-pass",
            preflight_metrics=metrics,
            pilot_metrics=pilot_metrics,
        )
    print(f"Aptus MLX-LM {arguments.level} validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


RUN_SCRIPT = r'''#!/usr/bin/env python3
"""Portable parent runner for an Aptus full-training bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / "__pycache__").exists():
    raise RuntimeError(
        "Bundle contains an unmanifested __pycache__; remove it before execution."
    )
sys.dont_write_bytecode = True
from runtime_lease import portable_execution_lease, run_with_lease


def bind_visible_cuda_devices(plan: dict) -> None:
    candidate = plan["recommended"]
    world_size = int(candidate["world_size"])
    device_indices = candidate.get("device_indices", list(range(world_size)))
    if (
        not isinstance(device_indices, list)
        or len(device_indices) != world_size
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in device_indices
        )
        or len(set(device_indices)) != len(device_indices)
    ):
        raise RuntimeError("Selected CUDA device indices do not match the planned world.")
    marker = os.environ.get("APTUS_BOUND_DEVICE_CANDIDATE")
    if marker is not None:
        if marker != candidate["candidate_id"]:
            raise RuntimeError("Inherited Aptus CUDA visibility belongs to another candidate.")
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        inherited_tokens = (
            [token.strip() for token in inherited.split(",") if token.strip()]
            if inherited is not None
            else []
        )
        if len(inherited_tokens) != world_size or any(
            token.lower() in {"-1", "nodevfiles", "none"}
            for token in inherited_tokens
        ):
            raise RuntimeError("Inherited Aptus CUDA visibility is missing or malformed.")
        return
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None:
        visible_tokens = [token.strip() for token in existing.split(",") if token.strip()]
        if not visible_tokens or any(
            token.lower() in {"-1", "nodevfiles", "none"}
            for token in visible_tokens
        ):
            raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no selectable CUDA devices.")
        if any(index >= len(visible_tokens) for index in device_indices):
            raise RuntimeError("Selected CUDA device index is outside CUDA_VISIBLE_DEVICES.")
        selected_tokens = [visible_tokens[index] for index in device_indices]
    else:
        selected_tokens = [str(index) for index in device_indices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_tokens)
    os.environ["APTUS_BOUND_DEVICE_CANDIDATE"] = candidate["candidate_id"]


def require_runs_root() -> Path:
    runs_root = ROOT / "runs"
    if runs_root.exists() and (runs_root.is_symlink() or not runs_root.is_dir()):
        raise RuntimeError("The Aptus runs path must be a real directory.")
    runs_root.mkdir(mode=0o700, exist_ok=True)
    if runs_root.is_symlink() or runs_root.resolve() != ROOT / "runs":
        raise RuntimeError("The Aptus runs directory escapes the bundle root.")
    return runs_root.resolve()


def normalize_run_output(output_dir: Path, *, fresh: bool) -> Path:
    runs_root = require_runs_root()
    unresolved = output_dir.expanduser()
    if unresolved.is_symlink():
        raise RuntimeError("Aptus run output cannot be a symlink.")
    resolved = unresolved.resolve()
    if resolved.parent != runs_root or not resolved.name.startswith("run_"):
        raise RuntimeError("Run output must be a ROOT/runs/run_* child.")
    if fresh and resolved.exists():
        raise RuntimeError(f"Aptus run output already exists: {resolved}")
    return resolved


def recover_pending_run() -> int | None:
    report_path = ROOT / "validation-report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pending_names = {
        "measured_run_pending_at",
        "pending_final_export",
        "pending_measured_run",
    }
    if not isinstance(report, dict) or not any(
        name in report for name in pending_names
    ):
        return None
    if report.get("state") != "execution-approved":
        raise RuntimeError("Pending run evidence exists outside execution-approved state.")
    active = report.get("active_run")
    if not isinstance(active, dict) or not isinstance(active.get("output_dir"), str):
        raise RuntimeError("Pending run evidence lacks a bound active output directory.")
    resolved = normalize_run_output(Path(active["output_dir"]), fresh=False)
    promoted = run_with_lease(
        [sys.executable, str(ROOT / "train.py"), "--promote-pending", str(resolved)],
        cwd=ROOT,
    )
    if promoted.returncode:
        return promoted.returncode
    print(f"Recovered and attested the completed pending Aptus run: {resolved}")
    return 0


def launch_full_training(arguments: argparse.Namespace) -> int:
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    bind_visible_cuda_devices(plan)
    recovered = recover_pending_run()
    if recovered is not None:
        return recovered
    output_dir = arguments.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = ROOT / "runs" / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    output_dir = normalize_run_output(output_dir, fresh=True)
    train_arguments = [
        str(ROOT / "train.py"),
        "--confirm-full-train",
        "--output-dir",
        str(output_dir),
    ]
    if arguments.local_files_only:
        train_arguments.append("--local-files-only")
    if plan["recommended"]["distribution"] == "single":
        command = [sys.executable, *train_arguments]
    else:
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.accelerate_cli",
            "launch",
            "--config_file",
            str(ROOT / "config" / "accelerate.yaml"),
            *train_arguments,
        ]
    completed = run_with_lease(command, cwd=ROOT)
    if completed.returncode:
        return completed.returncode
    promoted = run_with_lease(
        [sys.executable, str(ROOT / "train.py"), "--promote-pending", str(output_dir)],
        cwd=ROOT,
    )
    if promoted.returncode:
        return promoted.returncode
    print(f"Aptus measured run completed and was attested: {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch, wait for, verify, and attest one Aptus full-training run."
    )
    parser.add_argument("--confirm-full-train", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.confirm_full_train:
        parser.error("Full training requires --confirm-full-train.")
    with portable_execution_lease(ROOT, action="train"):
        return launch_full_training(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
'''


PREFLIGHT_SCRIPT = r'''#!/usr/bin/env python3
"""Run an honest, level-specific Aptus preflight."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent
if (_BOOTSTRAP_ROOT / "__pycache__").exists():
    raise RuntimeError(
        "Bundle contains an unmanifested __pycache__; remove it before validation."
    )

sys.dont_write_bytecode = True
from plan_contract import validate_bundle_manifest, validate_plan_payload
from runtime_lease import portable_execution_lease, run_with_lease


ROOT = _BOOTSTRAP_ROOT
LEVELS = ("contract", "static", "dependency", "model-data", "measured-preflight", "pilot")


def require_contract() -> dict:
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    errors = validate_plan_payload(plan, root=ROOT, verify_dataset=True)
    if errors:
        raise ValueError(" | ".join(errors))
    manifest_errors = validate_bundle_manifest(ROOT)
    if manifest_errors:
        raise ValueError(" | ".join(manifest_errors))
    return plan


def require_static() -> None:
    for relative in (
        "plan_contract.py",
        "preflight.py",
        "run.py",
        "runtime_lease.py",
        "train.py",
        "validate.py",
    ):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)


def require_dependencies() -> None:
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, expected = line.split("==", 1)
        try:
            actual = version(name)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Missing dependency: {name}=={expected}") from error
        if actual != expected:
            raise RuntimeError(f"Dependency mismatch for {name}: expected {expected}, found {actual}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_contract(
    checkpoint: Path,
    plan: dict,
    *,
    expected_step: int,
) -> dict:
    if not checkpoint.is_dir():
        raise RuntimeError(f"Pilot checkpoint is missing: {checkpoint}.")
    candidate = plan["recommended"]
    world_size = int(candidate["world_size"])
    trainer_state = checkpoint / "trainer_state.json"
    scheduler = checkpoint / "scheduler.pt"
    required_files = [trainer_state, scheduler]
    expected_rng = (
        {"rng_state.pth"}
        if world_size == 1
        else {f"rng_state_{rank}.pth" for rank in range(world_size)}
    )
    observed_rng = {path.name for path in checkpoint.glob("rng_state*.pth")}
    if observed_rng != expected_rng:
        raise RuntimeError("Pilot checkpoint RNG files do not match the selected world.")
    required_files.extend(checkpoint / name for name in sorted(expected_rng))
    if candidate["precision"] == "fp16":
        required_files.append(checkpoint / "scaler.pt")
    if candidate["distribution"] == "fsdp":
        model_state = checkpoint / "pytorch_model_fsdp_0"
        optimizer_state = checkpoint / "optimizer_0"
        for state_dir, label in (
            (model_state, "model"),
            (optimizer_state, "optimizer"),
        ):
            if not (state_dir / ".metadata").is_file() or not list(
                state_dir.rglob("*.distcp")
            ):
                raise RuntimeError(
                    f"Pilot FSDP checkpoint lacks complete {label} DCP state."
                )
    else:
        required_files.extend((checkpoint / "optimizer.pt", checkpoint / "training_args.bin"))
        if candidate["method"] == "full":
            weight_files = list(checkpoint.glob("model*.safetensors")) + list(
                checkpoint.glob("pytorch_model*.bin")
            )
        else:
            required_files.append(checkpoint / "adapter_config.json")
            weight_files = list(checkpoint.glob("adapter_model*.safetensors")) + list(
                checkpoint.glob("adapter_model*.bin")
            )
        if not weight_files:
            raise RuntimeError("Pilot checkpoint lacks method-specific model state.")
        required_files.extend(weight_files)
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required_files):
        raise RuntimeError("Pilot checkpoint contains a missing or empty required state file.")
    try:
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Pilot trainer_state.json is unreadable.") from error
    if not isinstance(state, dict) or state.get("global_step") != expected_step:
        raise RuntimeError("Pilot checkpoint global step does not match its phase.")
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    manifest = [
        {
            "path": path.relative_to(checkpoint).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    if not manifest:
        raise RuntimeError("Pilot checkpoint contains no files.")
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "global_step": expected_step,
        "total_bytes": sum(item["size_bytes"] for item in manifest),
        "manifest_sha256": manifest_sha256,
        "files": manifest,
    }


def pilot_root_contract(path: Path, plan: dict) -> dict | None:
    marker = path / ".aptus-pilot-run.json"
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "schema_version": "aptus.pilot-run.v1",
        "pilot_run_id": path.name,
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
    }
    if not isinstance(value, dict) or any(
        value.get(name) != expected_value
        for name, expected_value in expected.items()
    ):
        return None
    return value


def require_runs_root() -> Path:
    runs_root = ROOT / "runs"
    if runs_root.exists() and (runs_root.is_symlink() or not runs_root.is_dir()):
        raise RuntimeError("The Aptus runs path must be a real directory.")
    runs_root.mkdir(mode=0o700, exist_ok=True)
    if runs_root.is_symlink() or runs_root.resolve() != ROOT / "runs":
        raise RuntimeError("The Aptus runs directory escapes the bundle root.")
    return runs_root.resolve()


def claim_pilot_root(path: Path, plan: dict) -> None:
    runs_root = require_runs_root()
    if (
        path.is_symlink()
        or path.parent.resolve() != runs_root
        or not path.name.startswith("pilot_")
    ):
        raise RuntimeError("Pilot root must be a fresh ROOT/runs/pilot_* directory.")
    path.mkdir(exist_ok=False)
    marker = path / ".aptus-pilot-run.json"
    contract = {
        "schema_version": "aptus.pilot-run.v1",
        "pilot_run_id": path.name,
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = marker.with_name(f".{marker.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, marker)


def prune_pilot_runs(*, plan: dict, preserve: set[str]) -> None:
    runs_root = require_runs_root()
    for path in runs_root.glob("pilot_*"):
        if (
            path.name not in preserve
            and path.is_dir()
            and not path.is_symlink()
            and path.parent.resolve() == runs_root.resolve()
            and pilot_root_contract(path, plan) is not None
        ):
            shutil.rmtree(path)


def training_command(plan: dict, *arguments: str) -> list[str]:
    script = str(ROOT / "train.py")
    if plan["recommended"]["distribution"] == "single":
        return [sys.executable, script, *arguments]
    return [
        sys.executable, "-m", "accelerate.commands.accelerate_cli", "launch", "--config_file", str(ROOT / "config" / "accelerate.yaml"), script, *arguments
    ]


def require_census(value: Any, *, method: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Metrics do not contain a trainable-parameter census.")
    expected_scope = "all-parameters" if method == "full" else "lora-adapter-only"
    expected_identity = {
        "schema_version": "aptus.trainable-parameter-census.v1",
        "method": method,
        "parameter_scope": expected_scope,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected_identity.items()):
        raise RuntimeError("Trainable-parameter census violates the selected method scope.")
    if value.get("all_values_finite") is not True:
        raise RuntimeError("Trainable-parameter census does not attest finite values.")
    for name in ("trainable_parameter_count", "trainable_tensor_count"):
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise RuntimeError(f"Trainable-parameter census requires positive {name}.")
    counter_names = (
        "frozen_parameter_count",
        "frozen_tensor_count",
        "unexpected_trainable_tensor_count",
        "expected_adapter_target_match_count",
        "adapter_target_instance_count",
        "incomplete_adapter_target_instance_count",
    )
    for name in counter_names:
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RuntimeError(f"Trainable-parameter census requires non-negative integer {name}.")
    if method == "full":
        if any(value[name] != 0 for name in counter_names):
            raise RuntimeError("Full fine-tuning census contains frozen or adapter counters.")
    else:
        for name in ("frozen_parameter_count", "frozen_tensor_count"):
            if value[name] <= 0:
                raise RuntimeError(
                    f"Adapter census requires positive {name} for its frozen base."
                )
        if value["unexpected_trainable_tensor_count"] != 0:
            raise RuntimeError("Adapter census contains an unexpected trainable tensor.")
        if (
            value["expected_adapter_target_match_count"] <= 0
            or value["adapter_target_instance_count"] != value["expected_adapter_target_match_count"]
            or value["incomplete_adapter_target_instance_count"] != 0
        ):
            raise RuntimeError("Adapter census does not bind one complete LoRA A/B pair to every target instance.")
    digest = value.get("descriptor_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("Trainable-parameter census has an invalid descriptor digest.")
    return value


def run_validation(arguments: argparse.Namespace) -> int:
    target = LEVELS.index(arguments.level)
    plan = require_contract()
    if target >= LEVELS.index("static"):
        require_static()
    if target >= LEVELS.index("dependency"):
        require_dependencies()
    common = ["--local-files-only"] if arguments.local_files_only else []
    if target >= LEVELS.index("model-data"):
        completed = run_with_lease(
            training_command(plan, "--preflight-model-data", *common), cwd=ROOT
        )
        if completed.returncode:
            return completed.returncode
    if target >= LEVELS.index("measured-preflight"):
        completed = run_with_lease(
            training_command(plan, "--synthetic-preflight"), cwd=ROOT
        )
        if completed.returncode:
            return completed.returncode
    if target >= LEVELS.index("pilot"):
        preserved_run_ids: set[str] = set()
        current_metrics = ROOT / "pilot-output" / "metrics.json"
        if current_metrics.is_file():
            try:
                current_value = json.loads(current_metrics.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current_value = None
            if isinstance(current_value, dict) and isinstance(
                current_value.get("pilot_run_id"), str
            ):
                preserved_run_ids.add(current_value["pilot_run_id"])
        prune_pilot_runs(plan=plan, preserve=preserved_run_ids)
        pilot_run_id = "pilot_" + uuid.uuid4().hex
        pilot_root = ROOT / "runs" / pilot_run_id
        claim_pilot_root(pilot_root, plan)
        phase_one = pilot_root / "phase-1"
        completed = run_with_lease(
            training_command(plan, "--pilot", "--max-steps", "1", "--output-dir", str(phase_one), *common),
            cwd=ROOT,
        )
        if completed.returncode:
            return completed.returncode
        checkpoint = phase_one / "checkpoint-1"
        phase_one_checkpoint = checkpoint_contract(
            checkpoint, plan, expected_step=1
        )
        phase_two = pilot_root / "phase-2"
        completed = run_with_lease(
            training_command(
                plan,
                "--pilot",
                "--max-steps",
                "2",
                "--resume-from",
                str(checkpoint),
                "--output-dir",
                str(phase_two),
                *common,
            ),
            cwd=ROOT,
        )
        if completed.returncode:
            return completed.returncode
        phase_one_metrics = json.loads((phase_one / "metrics.json").read_text(encoding="utf-8"))
        phase_two_metrics = json.loads((phase_two / "metrics.json").read_text(encoding="utf-8"))
        phase_one_census = require_census(
            phase_one_metrics.get("trainable_parameter_census"),
            method=plan["recommended"]["method"],
        )
        phase_two_census = require_census(
            phase_two_metrics.get("trainable_parameter_census"),
            method=plan["recommended"]["method"],
        )
        if phase_one_census != phase_two_census:
            raise RuntimeError("Pilot phases do not bind the same trainable parameter set.")
        if phase_one_metrics.get("global_step") != 1 or phase_two_metrics.get("global_step", 0) < 2:
            raise RuntimeError("Pilot did not prove a step-one checkpoint and resumed step two.")
        for phase_name, metrics in (("phase one", phase_one_metrics), ("phase two", phase_two_metrics)):
            if "train_loss" not in metrics or not isinstance(metrics["train_loss"], (int, float)):
                raise RuntimeError(f"Pilot {phase_name} did not report train_loss.")
            if any(
                name.endswith("loss") and isinstance(value, (int, float)) and not math.isfinite(value)
                for name, value in metrics.items()
            ):
                raise RuntimeError(f"Pilot {phase_name} reported a nonfinite loss.")
        phase_two_checkpoint_path = phase_two / "checkpoint-2"
        phase_two_checkpoint = checkpoint_contract(
            phase_two_checkpoint_path, plan, expected_step=2
        )
        if checkpoint_contract(checkpoint, plan, expected_step=1) != phase_one_checkpoint:
            raise RuntimeError("Pilot phase-one checkpoint changed during continuation.")
        if Path(str(phase_two_metrics.get("resume_from", ""))).resolve() != checkpoint.resolve():
            raise RuntimeError("Pilot phase two does not bind the phase-one checkpoint path.")
        final_export_bytes = max(
            int(phase_one_metrics["final_export"]["total_bytes"]),
            int(phase_two_metrics["final_export"]["total_bytes"]),
        )
        aggregate = {
            "schema_version": "aptus.pilot-metrics.v2",
            "plan_id": plan["plan_id"],
            "candidate_id": plan["recommended"]["candidate_id"],
            "model_revision": plan["model"]["revision"],
            "dataset_sha256": plan["dataset"]["source_sha256"],
            "pilot_run_id": pilot_run_id,
            "pilot_run_dir": pilot_root.relative_to(ROOT).as_posix(),
            "phase_one": phase_one_metrics,
            "phase_two_resumed": phase_two_metrics,
            "phase_one_checkpoint": phase_one_checkpoint,
            "phase_two_checkpoint": phase_two_checkpoint,
            "checkpoint_continuation_observed": True,
            "measured_checkpoint_bytes": max(
                phase_one_checkpoint["total_bytes"],
                phase_two_checkpoint["total_bytes"],
            ),
            "measured_final_export_bytes": final_export_bytes,
        }
        current_root = ROOT / "pilot-output"
        current_root.mkdir(parents=True, exist_ok=True)
        metrics_path = current_root / "metrics.json"
        temporary = metrics_path.with_name(
            f".{metrics_path.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, metrics_path)
    print(f"Aptus {arguments.level} validation passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an Aptus bundle at one explicit level."
    )
    parser.add_argument("--level", choices=LEVELS, default="static")
    parser.add_argument("--local-files-only", action="store_true")
    arguments = parser.parse_args()
    with portable_execution_lease(ROOT, action=f"validate:{arguments.level}"):
        return run_validation(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
'''


VALIDATE_SCRIPT = r'''#!/usr/bin/env python3
"""Portable validation entrypoint for an Aptus bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None


ROOT = Path(__file__).resolve().parent
if (ROOT / "__pycache__").exists():
    raise RuntimeError(
        "Bundle contains an unmanifested __pycache__; remove it before validation."
    )
sys.dont_write_bytecode = True
from runtime_lease import portable_execution_lease, run_with_lease
STATE_BY_LEVEL = {
    "contract": "contract-pass",
    "static": "static-pass",
    "dependency": "dependency-pass",
    "model-data": "model-data-pass",
    "measured-preflight": "measured-preflight-pass",
    "pilot": "pilot-pass",
}
STATE_ORDER = (
    "invalid",
    "contract-pass",
    "static-pass",
    "dependency-pass",
    "model-data-pass",
    "measured-preflight-pass",
    "pilot-pass",
    "execution-approved",
    "measured-run-pass",
)


def bind_visible_cuda_devices(plan: dict) -> None:
    candidate = plan["recommended"]
    world_size = int(candidate["world_size"])
    device_indices = candidate.get("device_indices", list(range(world_size)))
    if (
        not isinstance(device_indices, list)
        or len(device_indices) != world_size
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in device_indices
        )
        or len(set(device_indices)) != len(device_indices)
    ):
        raise RuntimeError("Selected CUDA device indices do not match the planned world.")
    marker = os.environ.get("APTUS_BOUND_DEVICE_CANDIDATE")
    if marker is not None:
        if marker != candidate["candidate_id"]:
            raise RuntimeError("Inherited Aptus CUDA visibility belongs to another candidate.")
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        inherited_tokens = (
            [token.strip() for token in inherited.split(",") if token.strip()]
            if inherited is not None
            else []
        )
        if len(inherited_tokens) != world_size or any(
            token.lower() in {"-1", "nodevfiles", "none"}
            for token in inherited_tokens
        ):
            raise RuntimeError("Inherited Aptus CUDA visibility is missing or malformed.")
        return
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None:
        visible_tokens = [token.strip() for token in existing.split(",") if token.strip()]
        if not visible_tokens or any(
            token.lower() in {"-1", "nodevfiles", "none"}
            for token in visible_tokens
        ):
            raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no selectable CUDA devices.")
        if any(index >= len(visible_tokens) for index in device_indices):
            raise RuntimeError("Selected CUDA device index is outside CUDA_VISIBLE_DEVICES.")
        selected_tokens = [visible_tokens[index] for index in device_indices]
    else:
        selected_tokens = [str(index) for index in device_indices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_tokens)
    os.environ["APTUS_BOUND_DEVICE_CANDIDATE"] = candidate["candidate_id"]


@contextmanager
def report_lock():
    with (ROOT / ".validation-report.lock").open("a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
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
            elif msvcrt is not None:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_census(value: Any, *, method: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Metrics do not contain a trainable-parameter census.")
    expected_scope = "all-parameters" if method == "full" else "lora-adapter-only"
    expected_identity = {
        "schema_version": "aptus.trainable-parameter-census.v1",
        "method": method,
        "parameter_scope": expected_scope,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected_identity.items()):
        raise RuntimeError("Trainable-parameter census violates the selected method scope.")
    if value.get("all_values_finite") is not True:
        raise RuntimeError("Trainable-parameter census does not attest finite values.")
    for name in ("trainable_parameter_count", "trainable_tensor_count"):
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise RuntimeError(f"Trainable-parameter census requires positive {name}.")
    counter_names = (
        "frozen_parameter_count",
        "frozen_tensor_count",
        "unexpected_trainable_tensor_count",
        "expected_adapter_target_match_count",
        "adapter_target_instance_count",
        "incomplete_adapter_target_instance_count",
    )
    for name in counter_names:
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RuntimeError(f"Trainable-parameter census requires non-negative integer {name}.")
    if method == "full":
        if any(value[name] != 0 for name in counter_names):
            raise RuntimeError("Full fine-tuning census contains frozen or adapter counters.")
    else:
        for name in ("frozen_parameter_count", "frozen_tensor_count"):
            if value[name] <= 0:
                raise RuntimeError(
                    f"Adapter census requires positive {name} for its frozen base."
                )
        if value["unexpected_trainable_tensor_count"] != 0:
            raise RuntimeError("Adapter census contains an unexpected trainable tensor.")
        if (
            value["expected_adapter_target_match_count"] <= 0
            or value["adapter_target_instance_count"] != value["expected_adapter_target_match_count"]
            or value["incomplete_adapter_target_instance_count"] != 0
        ):
            raise RuntimeError("Adapter census does not bind one complete LoRA A/B pair to every target instance.")
    digest = value.get("descriptor_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("Trainable-parameter census has an invalid descriptor digest.")
    return value


def require_preflight_metrics(plan: dict) -> dict:
    metrics_path = ROOT / "preflight-metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Measured-preflight metrics are unreadable.") from error
    if not isinstance(metrics, dict):
        raise RuntimeError("Measured-preflight metrics must be a JSON object.")
    candidate = plan["recommended"]
    expected = {
        "schema_version": "aptus.preflight-metrics.v1",
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        "precision": candidate["precision"],
        "quantization": candidate.get("quantization"),
        "distribution": candidate["distribution"],
        "world_size": candidate["world_size"],
        "scope": "synthetic-method-preflight-not-model-data-pilot",
    }
    for name, value in expected.items():
        if metrics.get(name) != value:
            raise RuntimeError(f"Measured-preflight metrics do not bind {name}.")
    measured_peak = metrics.get("measured_peak_cuda_bytes")
    if (
        not isinstance(measured_peak, int)
        or isinstance(measured_peak, bool)
        or measured_peak <= 0
    ):
        raise RuntimeError(
            "Measured-preflight metrics require a positive measured_peak_cuda_bytes integer."
        )
    require_census(
        metrics.get("trainable_parameter_census"),
        method=candidate["method"],
    )
    return metrics


def completed_run_is_current(existing: dict, plan: dict) -> bool:
    final_report = existing.get("final_export")
    measured_report = existing.get("measured_run")
    candidate = plan["recommended"]
    if (
        not isinstance(final_report, dict)
        or not isinstance(measured_report, dict)
        or not existing.get("measured_run_completed_at")
    ):
        return False
    expected = {
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "distribution": candidate["distribution"],
        "world_size": candidate["world_size"],
    }
    if any(final_report.get(name) != value for name, value in expected.items()):
        return False
    if any(measured_report.get(name) != value for name, value in expected.items()):
        return False
    try:
        runs_root = (ROOT / "runs").resolve()
        run_dir = Path(measured_report["output_dir"]).resolve(strict=True)
        final_dir = Path(final_report["path"]).resolve(strict=True)
    except (KeyError, OSError, TypeError):
        return False
    if (
        run_dir.parent != runs_root
        or not run_dir.name.startswith("run_")
        or final_dir != (run_dir / "final").resolve()
    ):
        return False
    export_path = run_dir / "final-export.json"
    metrics_path = run_dir / "metrics.json"
    if not export_path.is_file() or not metrics_path.is_file():
        return False
    if (
        final_report.get("manifest_sha256") != sha256(export_path)
        or measured_report.get("metrics_sha256") != sha256(metrics_path)
    ):
        return False
    try:
        export = json.loads(export_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(export, dict) or not isinstance(metrics, dict):
        return False
    if (
        export.get("schema_version") != "aptus.final-export.v1"
        or export.get("method") != candidate["method"]
        or export.get("distribution") != candidate["distribution"]
        or export.get("world_size") != candidate["world_size"]
        or metrics.get("plan_id") != plan["plan_id"]
        or metrics.get("candidate_id") != candidate["candidate_id"]
        or metrics.get("distribution") != candidate["distribution"]
        or metrics.get("actual_world_size") != candidate["world_size"]
        or metrics.get("global_step") != measured_report.get("global_step")
        or metrics.get("per_rank_cuda_peaks")
        != measured_report.get("per_rank_cuda_peaks")
        or metrics.get("final_export") != export
    ):
        return False
    entries = export.get("files")
    if not isinstance(entries, list) or not entries:
        return False
    observed_paths = set()
    observed_total = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return False
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            return False
        normalized = relative.as_posix()
        if normalized in observed_paths:
            return False
        artifact = final_dir.joinpath(*relative.parts)
        try:
            resolved_artifact = artifact.resolve(strict=True)
        except OSError:
            return False
        if not artifact.is_file() or final_dir not in resolved_artifact.parents:
            return False
        size = artifact.stat().st_size
        if entry.get("size_bytes") != size or entry.get("sha256") != sha256(artifact):
            return False
        observed_paths.add(normalized)
        observed_total += size
    actual_paths = {
        path.relative_to(final_dir).as_posix()
        for path in final_dir.rglob("*")
        if path.is_file()
    }
    return bool(
        observed_paths == actual_paths
        and export.get("total_bytes") == observed_total
        and final_report.get("total_bytes") == observed_total
    )


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def environment_binding() -> str:
    direct_constraints = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name = line.split("==", 1)[0]
        try:
            direct_constraints[name] = version(name)
        except PackageNotFoundError:
            direct_constraints[name] = "missing"
    runtime_distributions = runtime_distribution_closure(direct_constraints)
    return json_hash({
        "python": platform.python_version(),
        "platform": platform.platform(),
        "direct_constraints": direct_constraints,
        "runtime_distributions": runtime_distributions,
    })


def runtime_distribution_closure(names: dict[str, str]) -> dict[str, str]:
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
                (token.find(character) for character in "[ (<>=!~" if character in token),
                default=len(token),
            )
            dependency = token[:boundary].strip()
            if dependency:
                pending.append(dependency)
    return dict(sorted(observed.items()))


def actual_hardware_binding() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Pilot completed without visible CUDA hardware.")
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
        raise RuntimeError("CUDA driver identity is unavailable for pilot binding.")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        device_uuid = str(getattr(properties, "uuid", "")).strip()
        if not device_uuid or device_uuid.lower() == "none":
            raise RuntimeError(f"CUDA device {index} does not expose a stable UUID for pilot binding.")
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "uuid": device_uuid,
                "pci_bus_id": str(getattr(properties, "pci_bus_id", "")),
                "total_vram_bytes": properties.total_memory,
                "compute_capability": f"{major}.{minor}",
            }
        )
    return json_hash(
        {"cuda_runtime": torch.version.cuda, "driver_version": driver_version, "devices": devices}
    )


def write_attestation(level: str) -> None:
    with report_lock():
        _write_attestation(level)


def _write_attestation(level: str) -> None:
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    manifest_digest = sha256(ROOT / "bundle-manifest.json")
    state = STATE_BY_LEVEL[level]
    existing_path = ROOT / "validation-report.json"
    if existing_path.is_file():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        existing_state = existing.get("state") if isinstance(existing, dict) else None
        historical_run = existing_state == "measured-run-pass"
        current_environment = environment_binding()
        environment_is_current = (
            historical_run
            or (
                isinstance(existing, dict)
                and existing.get("bindings", {}).get("environment")
                == current_environment
            )
        )
        pilot_is_current = True
        preflight_is_current = True
        hardware_is_current = True
        if (
            existing_state in STATE_ORDER
            and not historical_run
            and STATE_ORDER.index(existing_state)
            >= STATE_ORDER.index("model-data-pass")
        ):
            try:
                current_hardware = actual_hardware_binding()
            except (ImportError, RuntimeError):
                current_hardware = None
            hardware_is_current = bool(
                current_hardware is not None
                and existing.get("bindings", {}).get("hardware") == current_hardware
            )
        if existing_state in STATE_ORDER and STATE_ORDER.index(existing_state) >= STATE_ORDER.index("measured-preflight-pass"):
            try:
                current_preflight_metrics = require_preflight_metrics(plan)
            except RuntimeError:
                current_preflight_metrics = None
            preflight_metrics_path = ROOT / "preflight-metrics.json"
            preflight_is_current = bool(
                current_preflight_metrics is not None
                and existing.get("bindings", {}).get("preflight_metrics")
                == sha256(preflight_metrics_path)
                and existing.get("preflight_metrics") == current_preflight_metrics
            )
        if existing_state in STATE_ORDER and STATE_ORDER.index(existing_state) >= STATE_ORDER.index("pilot-pass"):
            pilot_metrics = ROOT / "pilot-output" / "metrics.json"
            pilot_is_current = bool(
                pilot_metrics.is_file()
                and existing.get("bindings", {}).get("pilot_metrics") == sha256(pilot_metrics)
            )
        if (
            isinstance(existing, dict)
            and existing_state in STATE_ORDER
            and STATE_ORDER.index(existing_state) > STATE_ORDER.index(state)
            and existing.get("artifact_fingerprint") == manifest_digest
            and existing.get("bindings", {}).get("bundle") == manifest_digest
            and existing.get("bindings", {}).get("plan_id") == plan["plan_id"]
            and existing.get("bindings", {}).get("candidate_id") == plan["recommended"]["candidate_id"]
            and existing.get("bindings", {}).get("model_revision") == plan["model"]["revision"]
            and existing.get("bindings", {}).get("dataset") == plan["dataset"]["source_sha256"]
            and environment_is_current
            and hardware_is_current
            and preflight_is_current
            and pilot_is_current
            and (
                existing_state != "measured-run-pass"
                or completed_run_is_current(existing, plan)
            )
        ):
            existing["latest_recheck"] = {
                "state": state,
                "validation_level": level,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "artifact_fingerprint": manifest_digest,
                "findings": [],
            }
            temporary = existing_path.with_name(".validation-report.json.tmp")
            temporary.write_text(
                json.dumps(existing, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, existing_path)
            print(
                f"Preserved stronger {existing['state']} attestation after {state} recheck."
            )
            return
    now = datetime.now(timezone.utc).isoformat()
    planned_hardware = json_hash(plan["hardware"])
    bindings = {
        "bundle": manifest_digest,
        "dataset": plan["dataset"]["source_sha256"],
        "environment": environment_binding(),
        "hardware": actual_hardware_binding()
        if level in {"model-data", "measured-preflight", "pilot"}
        else planned_hardware,
        "planned_hardware": planned_hardware,
        "validator": "aptus-portable-validator-v2",
        "validated_at": now,
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
    }
    if level in {"measured-preflight", "pilot"}:
        preflight_metrics_path = ROOT / "preflight-metrics.json"
        preflight_metrics = require_preflight_metrics(plan)
        bindings["preflight_metrics"] = sha256(preflight_metrics_path)
    else:
        preflight_metrics = None
    if level == "pilot":
        metrics_path = ROOT / "pilot-output" / "metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError("Pilot completed without metrics.json.")
        bindings["pilot_metrics"] = sha256(metrics_path)
        pilot_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        pilot_metrics = None
    report = {
        "state": state,
        "findings": [],
        "checked_files": [],
        "artifact_fingerprint": manifest_digest,
        "runtime_evidence": [f"portable validation level={level}"],
        "validation_level": level,
        "bindings": bindings,
        "validator_version": "aptus-portable-validator-v2",
        "validated_at": now,
        "preflight_metrics": preflight_metrics,
        "pilot_metrics": pilot_metrics,
    }
    temporary = existing_path.with_name(".validation-report.json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, existing_path)


def prune_attested_pilot_runs() -> None:
    """Remove only Aptus-owned stale pilots after the new report is durable."""

    report = json.loads((ROOT / "validation-report.json").read_text(encoding="utf-8"))
    metrics_path = ROOT / "pilot-output" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    current = metrics.get("pilot_run_id")
    if (
        report.get("state") != "pilot-pass"
        or report.get("bindings", {}).get("pilot_metrics") != sha256(metrics_path)
        or not isinstance(current, str)
    ):
        raise RuntimeError("Pilot retention cleanup requires a durable current attestation.")
    unresolved_runs_root = ROOT / "runs"
    if not unresolved_runs_root.exists():
        return
    if unresolved_runs_root.is_symlink() or not unresolved_runs_root.is_dir():
        raise RuntimeError("The Aptus runs path must be a real directory.")
    runs_root = unresolved_runs_root.resolve()
    if runs_root != ROOT / "runs":
        raise RuntimeError("The Aptus runs directory escapes the bundle root.")
    expected = {
        "schema_version": "aptus.pilot-run.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
    }
    for path in runs_root.glob("pilot_*"):
        if path.name == current or not path.is_dir() or path.is_symlink():
            continue
        if path.parent.resolve() != runs_root:
            continue
        marker = path / ".aptus-pilot-run.json"
        try:
            contract = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(contract, dict)
            or contract.get("pilot_run_id") != path.name
            or any(contract.get(name) != value for name, value in expected.items())
        ):
            continue
        shutil.rmtree(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate this Aptus bundle.")
    parser.add_argument("--level", choices=tuple(STATE_BY_LEVEL), default="static")
    parser.add_argument("--local-files-only", action="store_true")
    arguments = parser.parse_args()
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    bind_visible_cuda_devices(plan)
    with portable_execution_lease(ROOT, action=f"validate:{arguments.level}"):
        command = [
            sys.executable,
            str(ROOT / "preflight.py"),
            "--level",
            arguments.level,
        ]
        if arguments.local_files_only:
            command.append("--local-files-only")
        completed = run_with_lease(command, cwd=ROOT)
        if completed.returncode:
            return completed.returncode
        write_attestation(arguments.level)
        if arguments.level == "pilot":
            prune_attested_pilot_runs()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _portable_plan(plan: TrainingPlan, relative_dataset: str) -> TrainingPlan:
    provenance = plan.dataset.provenance
    portable_provenance = None
    if provenance is not None:
        portable_provenance = Provenance(
            kind=provenance.kind,
            source=f"bundle:{relative_dataset}",
            observed_at=provenance.observed_at,
            digest=provenance.digest,
            detail=provenance.detail,
        )
    dataset = replace(
        plan.dataset,
        source_path=Path(relative_dataset),
        bundle_path=relative_dataset,
        provenance=portable_provenance,
    )
    return replace(plan, dataset=dataset)


def _accelerate_config(plan: TrainingPlan) -> str:
    distribution = plan.recommended.distribution.value
    distributed_type = {"single": "NO", "ddp": "MULTI_GPU", "fsdp": "FSDP"}[
        distribution
    ]
    lines = [
        "compute_environment: LOCAL_MACHINE",
        f"distributed_type: {distributed_type}",
        f"mixed_precision: {plan.recommended.precision}",
        f"num_processes: {plan.recommended.world_size}",
        "num_machines: 1",
        "machine_rank: 0",
        "same_network: true",
        "use_cpu: false",
    ]
    if distribution == "fsdp":
        lines.extend(
            (
                "fsdp_config:",
                "  fsdp_version: 1",
                "  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP",
                "  fsdp_backward_prefetch: BACKWARD_PRE",
                "  fsdp_offload_params: false",
                "  fsdp_sharding_strategy: FULL_SHARD",
                "  fsdp_state_dict_type: SHARDED_STATE_DICT",
                "  fsdp_sync_module_states: true",
                "  fsdp_use_orig_params: true",
                "  fsdp_cpu_ram_efficient_loading: true",
            )
        )
    return "\n".join(lines) + "\n"


def _trainer_config(plan: TrainingPlan) -> dict[str, Any]:
    candidate = plan.recommended
    target = plan.target
    descriptor = method_descriptor(candidate.method)
    runtime = candidate.runtime_contract
    is_mlx = bool(runtime and runtime.training_runtime == TrainingRuntime.MLX_LM)
    return {
        "schema_version": "aptus.trainer-config.v2",
        "compiler_id": runtime.compiler_id if runtime else descriptor.compiler_id,
        "export_kind": runtime.export_kind if runtime else descriptor.export_kind,
        "training_runtime": (
            runtime.training_runtime.value
            if runtime
            else TrainingRuntime.TRANSFORMERS_PEFT_CUDA.value
        ),
        "compute_backend": runtime.compute_backend.value if runtime else "cuda",
        "task": target.task,
        "sequence_length": target.sequence_length,
        "packing": target.packing,
        "evaluation_fraction": target.evaluation_fraction,
        "max_epochs": target.max_epochs,
        "per_device_train_batch_size": candidate.micro_batch_size,
        "per_device_eval_batch_size": candidate.micro_batch_size,
        "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        "effective_global_batch_size": candidate.effective_batch_size,
        "world_size": candidate.world_size,
        "device_indices": list(candidate.device_indices),
        "learning_rate": candidate.learning_rate,
        "optimizer": "adamw" if is_mlx else "adamw_torch",
        "lr_scheduler_type": None if is_mlx else "linear",
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "max_grad_norm": None if is_mlx else 1.0,
        "precision": candidate.precision,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": (
            not is_mlx
            and candidate.method.value == "qlora"
            and candidate.distribution.value == "single"
        ),
        "checkpoint_steps": target.checkpoint_steps,
        "logging_steps": min(10, target.checkpoint_steps),
        "save_total_limit": 3,
        "checkpoint_retention_bytes": candidate.checkpoint_retention_bytes,
        "final_export_bytes": candidate.final_export_bytes,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": 17,
        "pilot_row_limit": max(32, target.effective_batch_size * 2),
        "pilot_dataset_path": "data/pilot-sample.jsonl",
        "training_dataset_path": "data/training.jsonl",
        "truncation_policy": "completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision",
    }


def _mlx_config(plan: TrainingPlan) -> str:
    candidate = plan.recommended
    values = {
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "seed": 17,
        "num_layers": -1,
        "batch_size": candidate.micro_batch_size,
        "iters": 2,
        "val_batches": 1,
        "learning_rate": candidate.learning_rate,
        "steps_per_report": 1,
        "steps_per_eval": 2,
        "grad_accumulation_steps": candidate.gradient_accumulation_steps,
        "save_every": 1,
        "max_seq_length": plan.target.sequence_length,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "lora_parameters": {
            "rank": candidate.rank,
            "dropout": 0.0,
            "scale": float(candidate.alpha) / candidate.rank,
            "keys": list(candidate.target_modules),
        },
    }
    lines: list[str] = []
    for key, value in values.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for nested_key, nested_value in value.items():
                lines.append(f"  {nested_key}: {json.dumps(nested_value)}")
        else:
            lines.append(f"{key}: {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def _mlx_training_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize an Aptus structured row to MLX-LM's masked chat contract."""

    text = row.get("text")
    if isinstance(text, str):
        raise ValueError(
            "MLX-LM compilation refuses text rows because pinned MLX-LM 0.31.3 "
            "cannot combine full-text supervision with the bundle's required prompt masking."
        )
    prompt, completion = row.get("prompt"), row.get("completion")
    if isinstance(prompt, str) or isinstance(completion, str):
        if (
            not isinstance(prompt, str)
            or not isinstance(completion, str)
            or not completion.strip()
        ):
            raise ValueError(
                "MLX-LM prompt/completion rows require a non-empty completion."
            )
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
        }
    instruction, output = row.get("instruction"), row.get("output")
    if isinstance(instruction, str) or isinstance(output, str):
        if (
            not isinstance(instruction, str)
            or not isinstance(output, str)
            or not output.strip()
        ):
            raise ValueError(
                "MLX-LM instruction/output rows require a non-empty output."
            )
        prompt_parts = ["### Instruction:\n" + instruction.strip()]
        input_value = row.get("input")
        if isinstance(input_value, str) and input_value.strip():
            prompt_parts.append("### Input:\n" + input_value.strip())
        prompt_parts.append("### Response:\n")
        return {
            "messages": [
                {"role": "user", "content": "\n".join(prompt_parts)},
                {"role": "assistant", "content": output},
            ]
        }
    messages = row.get("messages")
    if isinstance(messages, list):
        normalized = {"messages": messages}
        if "tools" in row:
            normalized["tools"] = row["tools"]
        return normalized
    content = row.get("content")
    if isinstance(content, str):
        raise ValueError(
            "MLX-LM compilation refuses content-only rows because they have no "
            "separable prompt and completion for masking."
        )
    raise ValueError("MLX-LM compilation encountered an unsupported dataset row.")


def _decision_report(plan: TrainingPlan) -> str:
    runtime = plan.recommended.runtime_contract
    optimizer_policy = (
        "MLX-LM AdamW; the pinned compiler config does not declare a separate learning-rate scheduler or CUDA gradient-scaler policy"
        if runtime and runtime.training_runtime == TrainingRuntime.MLX_LM
        else "adamw_torch, linear scheduler, weight decay 0.0, warmup steps 0, max grad norm 1.0"
    )
    rows = [
        "| Candidate | Method | Distribution | Status | Point GiB | Upper GiB | Batch | Frontier |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in plan.candidates:
        rows.append(
            "| {id} | {method} | {distribution} | {status} | {point:.2f} | {upper:.2f} | {batch} | {frontier} |".format(
                id=candidate.candidate_id,
                method=candidate.method.value,
                distribution=candidate.distribution.value,
                status=candidate.status.value,
                point=candidate.memory.point_estimate_bytes / 1024**3,
                upper=candidate.memory.upper_bytes / 1024**3,
                batch=candidate.effective_batch_size,
                frontier="yes" if candidate.pareto_frontier else "no",
            )
        )
    rationale = "\n".join(f"- {item}" for item in plan.recommendation_rationale)
    warnings = "\n".join(f"- {item}" for item in plan.warnings)
    assumptions = "\n".join(f"- {item}" for item in plan.recommended.assumptions)
    target_modules = ", ".join(plan.recommended.target_modules) or "full model"
    return f"""# Aptus decision report

Selected candidate: `{plan.recommended.candidate_id}`.

Formula: `{plan.formula_version}`. Point estimates sum named point components. Upper estimates sum `component_upper_bounds`, including a separate uncertainty term. The user reserve is not counted as usage. It reduces usable VRAM.

## Decision rationale

{rationale}

## Selected execution contract

- Candidate status: `{plan.recommended.status.value}`
- Training runtime: `{runtime.training_runtime.value if runtime else "transformers-peft-cuda"}`
- Compute backend: `{runtime.compute_backend.value if runtime else "cuda"}`
- Compiler and estimator: `{runtime.compiler_id if runtime else "legacy"}`, `{runtime.estimator_id if runtime else plan.formula_version}`
- Evidence requirement and export: `{runtime.evidence_requirement.value if runtime else "pilot-required"}`, `{runtime.export_kind if runtime else "legacy"}`
- Method: `{plan.recommended.method.value}`
- Distribution and world size: `{plan.recommended.distribution.value}`, `{plan.recommended.world_size}`
- Planned visible device indices: `{", ".join(str(item) for item in plan.recommended.device_indices)}`
- Precision and quantization: `{plan.recommended.precision}`, `{plan.recommended.quantization or "none"}`
- Adapter rank and alpha: `{plan.recommended.rank}`, `{plan.recommended.alpha}`
- Learning rate: `{plan.recommended.learning_rate:g}`
- Optimizer policy: `{optimizer_policy}`
- Truncation policy: completion first, then keep the prompt suffix that fits; refuse rows with no supervised tokens
- Target modules: `{target_modules}`
- Per-device micro-batch and accumulation: `{plan.recommended.micro_batch_size}`, `{plan.recommended.gradient_accumulation_steps}`
- Required host RAM: `{plan.recommended.required_host_ram_bytes}` bytes
- Required staging and output disk: `{plan.recommended.required_disk_bytes}` bytes
- Checkpoint-retention estimate: `{plan.recommended.checkpoint_retention_bytes}` bytes for up to three retained checkpoints
- Final-export estimate: `{plan.recommended.final_export_bytes}` bytes
- Confidence label: `{plan.recommended.confidence}`
- Pareto frontier: `{str(plan.recommended.pareto_frontier).lower()}`

## Assumptions

{assumptions}

## Candidate comparison

{chr(10).join(rows)}

This ranking does not claim measured throughput or model quality. Those require execution evidence.

## Warnings

{warnings}
"""


def _readme(plan: TrainingPlan) -> str:
    if (
        plan.recommended.runtime_contract
        and plan.recommended.runtime_contract.training_runtime == TrainingRuntime.MLX_LM
    ):
        return f"""# Aptus MLX-LM training bundle

This portable bundle contains candidate `{plan.recommended.candidate_id}` from
plan `{plan.plan_id}`. It is compiled for Apple silicon and MLX-LM.

The candidate is conditional and pilot-required. The generated wrapper runs a
bounded compiler smoke, an uninterrupted pilot, or a confirmed uninterrupted
full train from the pinned base revision. Each action rechecks live Apple
unified-memory headroom before loading the model.

```bash
python validate.py --level static
python validate.py --level dependency
python validate.py --level model-data
python validate.py --level measured-preflight
python validate.py --level pilot
python run.py --confirm-full-train --output-dir runs/run_<new-name>
```

The pilot proves at least two completed optimizer updates, finite train and
validation loss, exact target-module coverage, a positive adapter delta, and a
fresh-process adapter reload with bounded generation. Full training derives
its iteration count from the compiled train split, epoch count, micro-batch,
and gradient accumulation. Successful full runs publish an immutable adapter
manifest and `final-export.json`, then atomically promote `measured-run-pass`.
Failed or cancelled runs never promote.

MLX-LM crash resume is unsupported in this bundle. `--resume-from` always
fails closed. Pilot and full actions start from the pinned base revision and
must run uninterrupted.
"""
    return f"""# Aptus training bundle

This portable bundle contains candidate `{plan.recommended.candidate_id}` from
plan `{plan.plan_id}`.

## Before execution

1. Review `decision-report.md`, `plan.json`, and `evidence.jsonl`.
2. Confirm the model revision, data rights, target facts, and selected hardware.
3. Create the Python environment outside this directory. An in-bundle virtual
   environment is an unexpected path and invalidates the manifest.
4. Install the exact direct pins from `requirements.txt`.
5. Follow `runbook.md` in order.

The analytic point estimate and heuristic upper envelope are not calibration
results. The selected method must pass dependency, model-data, measured
preflight, and bounded real-model pilot gates before full training.

## Portable commands

```bash
python validate.py --level static
python validate.py --level dependency
python validate.py --level model-data
python validate.py --level measured-preflight
python validate.py --level pilot
python run.py --confirm-full-train
```

Do not launch `train.py` directly. `run.py` owns the full-run lease, aggregate
exit handling, artifact verification, and completion promotion.

## Evidence boundary

`pilot-pass` authorizes a later deep train-admission check. It does not guarantee
that current resources still suffice. `measured-run-pass` proves the bound run,
metrics, trainable census, dataset split, and structural export file tree passed
parent verification. It does not prove model quality, safety, or deployment
fitness.
"""


def _runbook(plan: TrainingPlan) -> str:
    if (
        plan.recommended.runtime_contract
        and plan.recommended.runtime_contract.training_runtime == TrainingRuntime.MLX_LM
    ):
        return """# MLX-LM runbook

## 1. Create an external environment

```bash
python -m venv ../aptus-mlx-env
source ../aptus-mlx-env/bin/activate
python -m pip install -r requirements.txt
```

## 2. Validate dependencies

```bash
python validate.py --level dependency
```

## 3. Exercise the compiler slice

```bash
python validate.py --level measured-preflight
```

The validator launches the owned bounded-smoke wrapper, downloads only the
plan-pinned model revision, disables remote model code, binds `data/mlx`, runs
at most eight iterations, and records the exact completed artifact tree plus
runtime-neutral MLX memory metrics in the validation report.

For QLoRA, the pinned model must contain explicit four-bit MLX quantization
metadata. Aptus never substitutes bitsandbytes and never quantizes an unbound
model during training.

## 4. Pilot gate

```bash
python validate.py --level pilot
```

The validator launches one owned, uninterrupted pilot from the pinned base.
It promotes `pilot-pass` only after two completed optimizer updates, finite
train and validation loss, exact adapter-target census, positive adapter
change, live headroom, and fresh-process adapter reload plus bounded generation.

## 5. Confirm full training

```bash
python run.py --confirm-full-train --output-dir runs/run_<new-name>
```

Before creating the run directory, the wrapper re-verifies the canonical pilot
report and artifact tree, live unified-memory headroom against the measured
pilot peak plus reserve, and evidence-derived disk headroom. The full action
derives its iteration count from the compiled train split and
`max_epochs`. It writes `metrics.json` last, after the adapter manifest, fresh
reload evidence, and `final-export.json` have passed verification. It then
re-verifies the owned tree and atomically promotes `measured-run-pass`. A failed
or cancelled process leaves an unpromoted owned directory.

## Resume boundary

MLX-LM optimizer, scheduler, and random-state crash continuation is not
supported. Do not pass `--resume-from`; the wrapper rejects it. Every pilot and
full run starts from the pinned base revision and runs uninterrupted.
"""
    return """# Runbook

## 1. Protect the bundle

Keep the bundle unchanged. Create the isolated environment beside it, never
inside it. For example, while your shell is in the bundle directory:

```bash
python -m venv ../aptus-runtime-env
source ../aptus-runtime-env/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` contains exact direct pins, not a transitive lock. Preserve
the installed-environment binding written by validation.

## 2. Validate dependencies

```bash
python validate.py --level dependency
```

This cumulatively checks the contract and static levels, then verifies the exact
direct requirements in the active environment.

## 3. Validate model and data

```bash
python validate.py --level model-data
```

This loads the pinned model and tokenizer, checks plan-driving architecture
facts and target modules, prepares the selected method, enables its compiled
checkpointing path, enforces the trainable-parameter census, and transforms
every canonical row. It constructs no optimizer and takes no training step.

## 4. Run measured preflight

```bash
python validate.py --level measured-preflight
```

This runs the selected broad method on a small synthetic CUDA model. It performs
a forward pass, backward pass, optimizer step, finite-loss check, census check,
and peak-memory measurement. It is method and kernel evidence, not planned-model
fit evidence.

## 5. Run the real-model pilot

```bash
python validate.py --level pilot
```

Each pilot phase reads only the compiler-produced pressure set. The compiler
supplies at least two effective batches and repeats real rows when needed. Phase
one writes a checkpoint. A fresh process continues from it and completes phase
two. Both phases must report the same plan-bound trainable census.

Review `pilot-output/metrics.json`, the checkpoint and export manifests, the
recorded CUDA peaks, and `validation-report.json`. A pilot failure blocks full
training.

## 6. Start a unique full run

```bash
python run.py --confirm-full-train
```

Do not launch `train.py` directly. `run.py` rechecks admission, holds the shared
lease, chooses the single or distributed command, waits for aggregate exit,
verifies pending metrics and the structural export, and promotes completion.

New outputs appear under a unique `runs/run_*` directory. The trainer binds the
full canonical dataset, deterministic train and evaluation assignment, rank
evidence, optimizer membership, trainable census, metrics, and export tree.

## 7. Interpret the result

Read `validation-report.json`, the selected run metrics, and the final export
manifest. `measured-run-pass` is operational and structural evidence. Use a
separate evaluation contract before making a quality claim.

## Recovery boundary

Full-training resume is fail-closed in Aptus v0.2. Checkpoints are emitted, but
Aptus will not resume one until a future contract binds complete model,
optimizer, scheduler, scaler, RNG, environment, plan, data progress, and
distributed state. Preserve a failed run for diagnosis and start a new run ID.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(root: Path, plan: TrainingPlan) -> None:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"bundle-manifest.json", "validation-report.json"}:
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    entrypoints = {
        "run": "run.py",
        "train": "train.py",
        "preflight": "preflight.py",
        "validate": "validate.py",
    }
    if (
        plan.recommended.runtime_contract
        and plan.recommended.runtime_contract.training_runtime == TrainingRuntime.MLX_LM
    ):
        entrypoints["reload"] = "reload.py"
    _write_json(
        root / "bundle-manifest.json",
        {
            "schema_version": "aptus.bundle.v2",
            "compiler": {"name": "aptus", "version": "0.2.0"},
            "stack_versions": STACK_VERSIONS,
            "plan_id": plan.plan_id,
            "plan_sha256": _sha256(root / "plan.json"),
            "candidate_id": plan.recommended.candidate_id,
            "formula_version": plan.formula_version,
            "entrypoints": entrypoints,
            "validation": {
                "levels": [
                    "contract",
                    "static",
                    "dependency",
                    "model-data",
                    "measured-preflight",
                    "pilot",
                ],
                "required_before_full_training": "pilot-pass",
            },
            "files": entries,
        },
    )


def _compile_into(plan: TrainingPlan, root: Path) -> TrainingPlan:
    suffix = plan.dataset.source_path.suffix.lower()
    if suffix not in {".jsonl", ".json", ".csv", ".txt"}:
        raise ValueError(f"Unsupported portable dataset suffix: {suffix or '<none>'}")
    relative_dataset = f"data/dataset{suffix}"
    portable = _portable_plan(plan, relative_dataset)
    dataset_destination = root / relative_dataset
    dataset_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plan.dataset.source_path, dataset_destination)
    if _sha256(dataset_destination) != plan.dataset.source_sha256:
        raise ValueError(
            "Dataset changed after profiling; re-profile before compiling."
        )
    pilot_row_count = max(32, plan.target.effective_batch_size * 2)
    pressure_rows = pilot_sample_rows(
        replace(plan.dataset, source_path=dataset_destination), limit=pilot_row_count
    )
    pilot_rows = tuple(
        pressure_rows[index % len(pressure_rows)] for index in range(pilot_row_count)
    )
    (root / "data" / "pilot-sample.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in pilot_rows
        ),
        encoding="utf-8",
    )
    training_rows = canonical_training_rows(
        replace(plan.dataset, source_path=dataset_destination)
    )
    is_mlx = bool(
        portable.recommended.runtime_contract
        and portable.recommended.runtime_contract.training_runtime
        == TrainingRuntime.MLX_LM
    )
    mlx_train = mlx_valid = None
    if is_mlx:
        if portable.dataset.example_count < 2:
            raise ValueError(
                "MLX-LM compilation requires at least two usable rows for disjoint train and validation files."
            )
        mlx_root = root / "data" / "mlx"
        mlx_root.mkdir(parents=True, exist_ok=True)
        mlx_train = (mlx_root / "train.jsonl").open("w", encoding="utf-8")
        mlx_valid = (mlx_root / "valid.jsonl").open("w", encoding="utf-8")
        valid_count = max(
            1,
            round(portable.dataset.example_count * portable.target.evaluation_fraction),
        )
        valid_start = portable.dataset.example_count - min(
            valid_count, portable.dataset.example_count - 1
        )
    try:
        with (root / "data" / "training.jsonl").open("w", encoding="utf-8") as output:
            for index, row in enumerate(training_rows):
                line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                output.write(line)
                if mlx_train is not None and mlx_valid is not None:
                    try:
                        mlx_row = _mlx_training_row(row)
                    except ValueError as error:
                        raise ValueError(
                            f"MLX-LM dataset row {index + 1}: {error}"
                        ) from error
                    mlx_line = (
                        json.dumps(mlx_row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    (mlx_valid if index >= valid_start else mlx_train).write(mlx_line)
    finally:
        if mlx_train is not None:
            mlx_train.close()
        if mlx_valid is not None:
            mlx_valid.close()
    if is_mlx:
        micro_batch = portable.recommended.micro_batch_size
        split_contract = {
            "schema_version": "aptus.mlx-split.v1",
            "micro_batch_size": micro_batch,
            "padding_policy": "repeat-within-disjoint-split-to-complete-final-batch",
            "splits": {},
        }
        for name in ("train", "valid"):
            path = root / "data" / "mlx" / f"{name}.jsonl"
            rows = [
                line for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
            if not rows:
                raise ValueError(f"MLX-LM {name} split must not be empty.")
            original_count = len(rows)
            padded_count = math.ceil(original_count / micro_batch) * micro_batch
            padded = [rows[index % original_count] for index in range(padded_count)]
            path.write_text("".join(line + "\n" for line in padded), encoding="utf-8")
            split_contract["splits"][name] = {
                "source_row_count": original_count,
                "compiled_row_count": padded_count,
            }
        _write_json(root / "data" / "mlx" / "split-contract.json", split_contract)

    payload = to_primitive(portable)
    _write_json(root / "plan.json", payload)
    _write_json(root / "profiles" / "model.json", payload["model"])
    _write_json(root / "profiles" / "dataset.json", payload["dataset"])
    _write_json(root / "profiles" / "hardware.json", payload["hardware"])
    _write_json(root / "candidates.json", payload["candidates"])
    evidence_lines = "".join(
        json.dumps(item, sort_keys=True) + "\n" for item in payload["evidence_records"]
    )
    (root / "evidence.jsonl").write_text(evidence_lines, encoding="utf-8")
    (root / "decision-report.md").write_text(
        _decision_report(portable), encoding="utf-8"
    )
    (root / "README.md").write_text(_readme(portable), encoding="utf-8")
    (root / "runbook.md").write_text(_runbook(portable), encoding="utf-8")
    training_runtime = (
        portable.recommended.runtime_contract.training_runtime
        if portable.recommended.runtime_contract
        else TrainingRuntime.TRANSFORMERS_PEFT_CUDA
    )
    requirements = (
        "\n".join(
            bundle_requirements(
                portable.recommended.method,
                training_runtime=training_runtime,
            )
        )
        + "\n"
    )
    (root / "requirements.txt").write_text(requirements, encoding="utf-8")
    (root / "config").mkdir(parents=True, exist_ok=True)
    accelerate = _accelerate_config(portable)
    (root / "config" / "accelerate.yaml").write_text(accelerate, encoding="utf-8")
    (root / "accelerate_config.yaml").write_text(accelerate, encoding="utf-8")
    _write_json(root / "config" / "trainer.json", _trainer_config(portable))
    if is_mlx:
        (root / "config" / "mlx-lm.yaml").write_text(
            _mlx_config(portable), encoding="utf-8"
        )
    contract_source = (
        resources.files("aptus")
        .joinpath("plan_contract.py")
        .read_text(encoding="utf-8")
    )
    (root / "plan_contract.py").write_text(contract_source, encoding="utf-8")
    runtime_lease_source = (
        resources.files("aptus")
        .joinpath("runtime_lease.py")
        .read_text(encoding="utf-8")
    )
    (root / "runtime_lease.py").write_text(runtime_lease_source, encoding="utf-8")
    (root / "train.py").write_text(
        MLX_TRAIN_SCRIPT if is_mlx else TRAIN_SCRIPT, encoding="utf-8"
    )
    (root / "run.py").write_text(
        MLX_RUN_SCRIPT if is_mlx else RUN_SCRIPT, encoding="utf-8"
    )
    if is_mlx:
        (root / "reload.py").write_text(MLX_RELOAD_SCRIPT, encoding="utf-8")
    (root / "preflight.py").write_text(
        MLX_PREFLIGHT_SCRIPT if is_mlx else PREFLIGHT_SCRIPT, encoding="utf-8"
    )
    (root / "validate.py").write_text(
        MLX_VALIDATE_SCRIPT if is_mlx else VALIDATE_SCRIPT, encoding="utf-8"
    )
    _write_manifest(root, portable)
    return portable


def generate_bundle(plan: TrainingPlan, output_dir: Path) -> ValidationReport:
    """Compile and statically validate a bundle, publishing it atomically."""

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Bundle output is not empty: {output_dir}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        _compile_into(plan, temporary)
        from .validation import validate_bundle

        report = validate_bundle(temporary, level="static", run=False)
        if report.state.value == "invalid":
            raise ValueError("Generated bundle failed static validation.")
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(temporary, output_dir)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def create_bundle_archive(bundle_dir: Path, archive_path: Path | None = None) -> Path:
    """Create a byte-deterministic ZIP from a compiled bundle."""

    bundle_dir = bundle_dir.resolve(strict=True)
    archive_path = (archive_path or bundle_dir.with_suffix(".zip")).resolve()
    if archive_path == bundle_dir or bundle_dir in archive_path.parents:
        raise ValueError(
            "Bundle archives must be written outside the bundle directory."
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(f"Archive output already exists: {archive_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.tmp-", dir=archive_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(
                item
                for item in bundle_dir.rglob("*")
                if item.is_file()
                and item.name
                not in {
                    ".validation-report.lock",
                    "preflight-metrics.json",
                    "validation-report.json",
                }
                and "pilot-output" not in item.relative_to(bundle_dir).parts
                and "runs" not in item.relative_to(bundle_dir).parts
            ):
                relative = path.relative_to(bundle_dir).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, path.read_bytes())
        try:
            os.link(temporary, archive_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Archive output already exists: {archive_path}"
            ) from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return archive_path


def bundle_files(bundle_dir: Path) -> list[str]:
    return sorted(
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file()
    )
