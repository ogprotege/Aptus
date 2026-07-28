#!/usr/bin/env python3
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
