#!/usr/bin/env python3
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
from plan_contract import load_json_object, validate_bundle_manifest, validate_plan_payload
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
        raise RuntimeError(
            "Selected CUDA device indices do not match the planned world."
        )
    marker = os.environ.get("APTUS_BOUND_DEVICE_CANDIDATE")
    if marker is not None:
        if marker != candidate["candidate_id"]:
            raise RuntimeError(
                "Inherited Aptus CUDA visibility belongs to another candidate."
            )
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        inherited_tokens = (
            [token.strip() for token in inherited.split(",") if token.strip()]
            if inherited is not None
            else []
        )
        if len(inherited_tokens) != world_size or any(
            token.lower() in {"-1", "nodevfiles", "none"} for token in inherited_tokens
        ):
            raise RuntimeError(
                "Inherited Aptus CUDA visibility is missing or malformed."
            )
        return
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None:
        visible_tokens = [
            token.strip() for token in existing.split(",") if token.strip()
        ]
        if not visible_tokens or any(
            token.lower() in {"-1", "nodevfiles", "none"} for token in visible_tokens
        ):
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES exposes no selectable CUDA devices."
            )
        if any(index >= len(visible_tokens) for index in device_indices):
            raise RuntimeError(
                "Selected CUDA device index is outside CUDA_VISIBLE_DEVICES."
            )
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
    if any(
        value.get(name) != expected_value
        for name, expected_value in expected_identity.items()
    ):
        raise RuntimeError(
            "Trainable-parameter census violates the selected method scope."
        )
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
            raise RuntimeError(
                f"Trainable-parameter census requires non-negative integer {name}."
            )
    if method == "full":
        if any(value[name] != 0 for name in counter_names):
            raise RuntimeError(
                "Full fine-tuning census contains frozen or adapter counters."
            )
    else:
        for name in ("frozen_parameter_count", "frozen_tensor_count"):
            if value[name] <= 0:
                raise RuntimeError(
                    f"Adapter census requires positive {name} for its frozen base."
                )
        if value["unexpected_trainable_tensor_count"] != 0:
            raise RuntimeError(
                "Adapter census contains an unexpected trainable tensor."
            )
        if (
            value["expected_adapter_target_match_count"] <= 0
            or value["adapter_target_instance_count"]
            != value["expected_adapter_target_match_count"]
            or value["incomplete_adapter_target_instance_count"] != 0
        ):
            raise RuntimeError(
                "Adapter census does not bind one complete LoRA A/B pair to every target instance."
            )
    digest = value.get("descriptor_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(
            "Trainable-parameter census has an invalid descriptor digest."
        )
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
    if final_report.get("manifest_sha256") != sha256(
        export_path
    ) or measured_report.get("metrics_sha256") != sha256(metrics_path):
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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
    return json_hash(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "direct_constraints": direct_constraints,
            "runtime_distributions": runtime_distributions,
        }
    )


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
                    [
                        executable,
                        "--query-gpu=driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                completed = None
            if completed is not None and completed.returncode == 0:
                versions = {
                    line.strip()
                    for line in completed.stdout.splitlines()
                    if line.strip()
                }
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
            raise RuntimeError(
                f"CUDA device {index} does not expose a stable UUID for pilot binding."
            )
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
        {
            "cuda_runtime": torch.version.cuda,
            "driver_version": driver_version,
            "devices": devices,
        }
    )


def write_attestation(level: str) -> None:
    with report_lock():
        _write_attestation(level)


def _write_attestation(level: str) -> None:
    plan = load_json_object(ROOT / "plan.json", "Bundle plan")
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
        environment_is_current = historical_run or (
            isinstance(existing, dict)
            and existing.get("bindings", {}).get("environment") == current_environment
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
        if existing_state in STATE_ORDER and STATE_ORDER.index(
            existing_state
        ) >= STATE_ORDER.index("measured-preflight-pass"):
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
        if existing_state in STATE_ORDER and STATE_ORDER.index(
            existing_state
        ) >= STATE_ORDER.index("pilot-pass"):
            pilot_metrics = ROOT / "pilot-output" / "metrics.json"
            pilot_is_current = bool(
                pilot_metrics.is_file()
                and existing.get("bindings", {}).get("pilot_metrics")
                == sha256(pilot_metrics)
            )
        if (
            isinstance(existing, dict)
            and existing_state in STATE_ORDER
            and STATE_ORDER.index(existing_state) > STATE_ORDER.index(state)
            and existing.get("artifact_fingerprint") == manifest_digest
            and existing.get("bindings", {}).get("bundle") == manifest_digest
            and existing.get("bindings", {}).get("plan_id") == plan["plan_id"]
            and existing.get("bindings", {}).get("candidate_id")
            == plan["recommended"]["candidate_id"]
            and existing.get("bindings", {}).get("model_revision")
            == plan["model"]["revision"]
            and existing.get("bindings", {}).get("dataset")
            == plan["dataset"]["source_sha256"]
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
                json.dumps(existing, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, existing_path)


def prune_attested_pilot_runs() -> None:
    """Remove only Aptus-owned stale pilots after the new report is durable."""

    report = json.loads((ROOT / "validation-report.json").read_text(encoding="utf-8"))
    metrics_path = ROOT / "pilot-output" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    plan = load_json_object(ROOT / "plan.json", "Bundle plan")
    current = metrics.get("pilot_run_id")
    if (
        report.get("state") != "pilot-pass"
        or report.get("bindings", {}).get("pilot_metrics") != sha256(metrics_path)
        or not isinstance(current, str)
    ):
        raise RuntimeError(
            "Pilot retention cleanup requires a durable current attestation."
        )
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
    manifest_errors = validate_bundle_manifest(ROOT)
    if manifest_errors:
        raise RuntimeError("Invalid Aptus bundle: " + " | ".join(manifest_errors))
    parser = argparse.ArgumentParser(description="Validate this Aptus bundle.")
    parser.add_argument("--level", choices=tuple(STATE_BY_LEVEL), default="static")
    parser.add_argument("--local-files-only", action="store_true")
    arguments = parser.parse_args()
    plan = load_json_object(ROOT / "plan.json", "Bundle plan")
    plan_errors = validate_plan_payload(plan, root=ROOT, verify_dataset=True)
    if plan_errors:
        raise RuntimeError("Invalid Aptus plan: " + " | ".join(plan_errors))
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
