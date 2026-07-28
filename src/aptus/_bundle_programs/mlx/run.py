#!/usr/bin/env python3
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
