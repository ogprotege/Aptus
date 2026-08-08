#!/usr/bin/env python3
"""Run an honest, level-specific Aptus preflight."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent
if (_BOOTSTRAP_ROOT / "__pycache__").exists():
    raise RuntimeError(
        "Bundle contains an unmanifested __pycache__; remove it before validation."
    )

sys.dont_write_bytecode = True
from campaign_events import emit_boundary  # noqa: E402
from plan_contract import (  # noqa: E402
    load_json_object,
    validate_bundle_manifest,
    validate_plan_payload,
)
from runtime_lease import portable_execution_lease, run_with_lease  # noqa: E402


ROOT = _BOOTSTRAP_ROOT
LEVELS = (
    "contract",
    "static",
    "dependency",
    "model-data",
    "measured-preflight",
    "pilot",
)


def require_contract() -> dict:
    plan = load_json_object(ROOT / "plan.json", "Bundle plan")
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
        "policy_snapshot.py",
        "campaign_events.py",
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
            raise RuntimeError(
                f"Dependency mismatch for {name}: expected {expected}, found {actual}"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign_pilot_failure_code(error: BaseException) -> str:
    """Project pilot validation failures onto the frozen stable reason set."""

    message = str(error).lower()
    if "cuda" in message and "out of memory" in message:
        return "CUDA_OOM"
    if "nonfinite" in message or "non-finite" in message:
        return "NONFINITE_VALUE"
    if "checkpoint" in message or "continuation" in message or "resume" in message:
        return "CHECKPOINT_CONTINUATION_FAILURE"
    if "artifact" in message or "changed" in message or "bind" in message:
        return "ARTIFACT_INTEGRITY_FAILURE"
    return "PROCESS_EXIT_NONZERO"


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
        raise RuntimeError(
            "Pilot checkpoint RNG files do not match the selected world."
        )
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
        required_files.extend(
            (checkpoint / "optimizer.pt", checkpoint / "training_args.bin")
        )
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
        raise RuntimeError(
            "Pilot checkpoint contains a missing or empty required state file."
        )
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
        value.get(name) != expected_value for name, expected_value in expected.items()
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
        sys.executable,
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--config_file",
        str(ROOT / "config" / "accelerate.yaml"),
        script,
        *arguments,
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
        emit_boundary("pilot.phase-started", phase="pilot-phase-1", action="pilot")
        phase_one_terminal_emitted = False
        try:
            completed = run_with_lease(
                training_command(
                    plan,
                    "--pilot",
                    "--max-steps",
                    "1",
                    "--campaign-pilot-phase",
                    "pilot-phase-1",
                    "--output-dir",
                    str(phase_one),
                    *common,
                ),
                cwd=ROOT,
            )
            if completed.returncode:
                phase_one_terminal_emitted = True
                return completed.returncode
            checkpoint = phase_one / "checkpoint-1"
            phase_one_checkpoint = checkpoint_contract(
                checkpoint, plan, expected_step=1
            )
        except BaseException as error:
            if not phase_one_terminal_emitted:
                emit_boundary(
                    "pilot.phase-finished",
                    phase="pilot-phase-1",
                    action="pilot",
                    native_outcome="failed",
                    reason_code=campaign_pilot_failure_code(error),
                )
            raise
        emit_boundary(
            "pilot.phase-finished",
            phase="pilot-phase-1",
            action="pilot",
            native_outcome="passed",
        )
        phase_two = pilot_root / "phase-2"
        emit_boundary("pilot.phase-started", phase="pilot-phase-2", action="pilot")
        phase_two_terminal_emitted = False
        try:
            completed = run_with_lease(
                training_command(
                    plan,
                    "--pilot",
                    "--max-steps",
                    "2",
                    "--resume-from",
                    str(checkpoint),
                    "--campaign-pilot-phase",
                    "pilot-phase-2",
                    "--output-dir",
                    str(phase_two),
                    *common,
                ),
                cwd=ROOT,
            )
            if completed.returncode:
                phase_two_terminal_emitted = True
                return completed.returncode
            phase_one_metrics = json.loads(
                (phase_one / "metrics.json").read_text(encoding="utf-8")
            )
            phase_two_metrics = json.loads(
                (phase_two / "metrics.json").read_text(encoding="utf-8")
            )
            phase_one_census = require_census(
                phase_one_metrics.get("trainable_parameter_census"),
                method=plan["recommended"]["method"],
            )
            phase_two_census = require_census(
                phase_two_metrics.get("trainable_parameter_census"),
                method=plan["recommended"]["method"],
            )
            if phase_one_census != phase_two_census:
                raise RuntimeError(
                    "Pilot phases do not bind the same trainable parameter set."
                )
            if (
                phase_one_metrics.get("global_step") != 1
                or phase_two_metrics.get("global_step", 0) < 2
            ):
                raise RuntimeError(
                    "Pilot did not prove a step-one checkpoint and resumed step two."
                )
            for phase_name, metrics in (
                ("phase one", phase_one_metrics),
                ("phase two", phase_two_metrics),
            ):
                if "train_loss" not in metrics or not isinstance(
                    metrics["train_loss"], (int, float)
                ):
                    raise RuntimeError(f"Pilot {phase_name} did not report train_loss.")
                if any(
                    name.endswith("loss")
                    and isinstance(value, (int, float))
                    and not math.isfinite(value)
                    for name, value in metrics.items()
                ):
                    raise RuntimeError(f"Pilot {phase_name} reported a nonfinite loss.")
            phase_two_checkpoint_path = phase_two / "checkpoint-2"
            phase_two_checkpoint = checkpoint_contract(
                phase_two_checkpoint_path, plan, expected_step=2
            )
            if (
                checkpoint_contract(checkpoint, plan, expected_step=1)
                != phase_one_checkpoint
            ):
                raise RuntimeError(
                    "Pilot phase-one checkpoint changed during continuation."
                )
            if (
                Path(str(phase_two_metrics.get("resume_from", ""))).resolve()
                != checkpoint.resolve()
            ):
                raise RuntimeError(
                    "Pilot phase two does not bind the phase-one checkpoint path."
                )
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
        except BaseException as error:
            if not phase_two_terminal_emitted:
                emit_boundary(
                    "pilot.phase-finished",
                    phase="pilot-phase-2",
                    action="pilot",
                    native_outcome="failed",
                    reason_code=campaign_pilot_failure_code(error),
                )
            raise
        emit_boundary(
            "pilot.phase-finished",
            phase="pilot-phase-2",
            action="pilot",
            native_outcome="passed",
        )
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
