from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Literal, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows only.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX only.
    msvcrt = None

from .attestation import require_trainable_parameter_census
from .catalog import bundle_requirements
from .domain import (
    ValidationFinding,
    ValidationReport,
    ValidationState,
    to_primitive,
    training_plan_from_primitive,
)
from .execution import (
    _actual_hardware_binding as _job_hardware_binding,
    _require_mlx_model_load_binding,
    _verify_mlx_admission,
)
from .model_compatibility import current_model_policy_snapshot_sha256
from .policy_snapshot import (
    model_policy_snapshot_bytes,
    model_policy_snapshot_sha256,
    validate_model_policy_snapshot,
)
from .plan_contract import (
    bundle_fingerprint,
    mlx_trainable_target_instance_total,
    sha256_file,
    validate_bundle_manifest,
    validate_plan_payload,
)
from .planning import plan_training
from .runtime_env import resolve_runtime_interpreter


ValidationLevel = Literal[
    "contract", "static", "dependency", "model-data", "measured-preflight", "pilot"
]
LEVELS: tuple[ValidationLevel, ...] = (
    "contract",
    "static",
    "dependency",
    "model-data",
    "measured-preflight",
    "pilot",
)
LEVEL_STATES = {
    "contract": ValidationState.CONTRACT_PASS,
    "static": ValidationState.STATIC_PASS,
    "dependency": ValidationState.DEPENDENCY_PASS,
    "model-data": ValidationState.MODEL_DATA_PASS,
    "measured-preflight": ValidationState.MEASURED_PREFLIGHT_PASS,
    "pilot": ValidationState.PILOT_PASS,
}
STATE_RANK = {
    ValidationState.CONTRACT_PASS: 1,
    ValidationState.STATIC_PASS: 2,
    ValidationState.DEPENDENCY_PASS: 3,
    ValidationState.MODEL_DATA_PASS: 4,
    ValidationState.MEASURED_PREFLIGHT_PASS: 5,
    ValidationState.PILOT_PASS: 6,
    ValidationState.EXECUTION_APPROVED: 7,
    ValidationState.MEASURED_RUN_PASS: 8,
}
REQUIRED_BUNDLE_FILES = (
    "README.md",
    "config/accelerate.yaml",
    "config/trainer.json",
    "bundle-manifest.json",
    "candidates.json",
    "decision-report.md",
    "evidence.jsonl",
    "plan.json",
    "plan_contract.py",
    "policy/model-policy-snapshot.v1.json",
    "policy_snapshot.py",
    "preflight.py",
    "profiles/dataset.json",
    "profiles/hardware.json",
    "profiles/model.json",
    "requirements.txt",
    "runbook.md",
    "run.py",
    "runtime_lease.py",
    "train.py",
    "validate.py",
)
_LOWERCASE_HEXADECIMAL = frozenset("0123456789abcdef")


def _finding(
    code: str, message: str, *, severity: str = "error", path: str | None = None
) -> ValidationFinding:
    return ValidationFinding(code=code, message=message, severity=severity, path=path)


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWERCASE_HEXADECIMAL for character in value)
    )


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _environment_binding(requirements: tuple[str, ...]) -> str:
    direct_constraints: dict[str, str] = {}
    for requirement in requirements:
        name = requirement.split("==", 1)[0]
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


def _runtime_distribution_closure(names: dict[str, str]) -> dict[str, str]:
    """Bind the installed dependency closure, excluding unrelated PYTHONPATH tools."""

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


def _actual_hardware_binding(device_indices: list[int]) -> str:
    return _job_hardware_binding(device_indices)


def _load_json(
    path: Path,
    findings: list[ValidationFinding],
    code: str,
    *,
    require_object: bool = False,
) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as error:
        findings.append(_finding(code, str(error), path=path.name))
        return None
    if require_object and not isinstance(value, dict):
        findings.append(
            _finding(
                code,
                f"{path.name} must contain a JSON object.",
                path=path.name,
            )
        )
        return None
    return value


def _mlx_finite(value: Any, label: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(f"{label} must be {'positive ' if positive else ''}finite.")
    return float(value)


def _require_mlx_admission(value: Any, plan: Mapping[str, Any], label: str) -> None:
    if not isinstance(plan, dict):
        raise ValueError(f"{label} cannot bind an incomplete plan.")
    _verify_mlx_admission(plan, value, label=label)


def _require_mlx_target_binding(value: Any, plan: Mapping[str, Any]) -> None:
    candidate = plan.get("recommended")
    model = plan.get("model")
    if not isinstance(candidate, Mapping) or not isinstance(model, Mapping):
        raise ValueError("MLX target evidence cannot bind an incomplete plan.")
    targets = candidate.get("target_modules")
    layers = model.get("layers")
    if (
        not isinstance(value, Mapping)
        or not isinstance(targets, list)
        or not targets
        or not isinstance(layers, int)
        or isinstance(layers, bool)
        or layers <= 0
    ):
        raise ValueError("MLX trainable-target binding is missing.")
    counts = value.get("target_instance_counts")
    try:
        count = mlx_trainable_target_instance_total(
            targets, layers, counts, family=model.get("family")
        )
    except ValueError as error:
        raise ValueError(
            "MLX trainable-target binding is not exact for the plan."
        ) from error
    expected = {
        "schema_version": "aptus.mlx-trainable-target-binding.v1",
        "planned_target_modules": targets,
        "transformer_layer_count": layers,
        "expected_adapter_target_instance_count": count,
        "adapter_target_instance_count": count,
        "trainable_tensor_count": count * 2,
        "target_instance_counts": counts,
    }
    resolved = value.get("resolved_layer_keys")
    descriptor = value.get("descriptor_sha256")
    descriptor_value = {
        str(key): item for key, item in value.items() if key != "descriptor_sha256"
    }
    if (
        any(value.get(name) != item for name, item in expected.items())
        or not isinstance(resolved, list)
        or len(resolved) != len(targets)
        or len(set(resolved)) != len(targets)
        or descriptor != _json_hash(descriptor_value)
    ):
        raise ValueError("MLX trainable-target binding is not exact for the plan.")


def _mlx_bound_file(root: Path, relative_value: Any) -> Path:
    if not isinstance(relative_value, str):
        raise ValueError("MLX artifact path must be a string.")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("MLX artifact path is unsafe.")
    unresolved = root.joinpath(*relative.parts)
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise ValueError("MLX artifact is missing.") from error
    if (
        unresolved.is_symlink()
        or root.resolve() not in resolved.parents
        or not resolved.is_file()
    ):
        raise ValueError("MLX artifact escapes its owned run root.")
    return resolved


def _require_mlx_artifact_manifest(value: Any, *, run_root: Path, action: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("MLX immutable artifact manifest is missing.")
    entries = value.get("files")
    expected = {
        "schema_version": "aptus.mlx-artifact-manifest.v1",
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
    }
    if (
        any(value.get(name) != item for name, item in expected.items())
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("MLX immutable artifact manifest has the wrong contract.")
    seen: set[str] = set()
    total = 0
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError("MLX artifact manifest entry is invalid.")
        normalized = Path(entry["path"]).as_posix()
        if normalized in seen:
            raise ValueError("MLX artifact manifest contains a duplicate path.")
        artifact = _mlx_bound_file(run_root, normalized)
        size = artifact.stat().st_size
        if entry.get("size_bytes") != size or entry.get("sha256") != sha256_file(
            artifact
        ):
            raise ValueError("MLX artifact manifest no longer matches the run files.")
        seen.add(normalized)
        total += size
    if value.get("total_bytes") != total:
        raise ValueError("MLX artifact manifest total is inconsistent.")


def _read_mlx_runtime_metrics(
    path: Path, plan: dict[str, Any], *, action: str
) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("MLX runtime metrics cannot be a symlink.")
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("MLX runtime metrics are unreadable.") from error
    if not isinstance(metrics, dict):
        raise ValueError("MLX runtime metrics must be a JSON object.")
    candidate = plan.get("recommended")
    model = plan.get("model")
    dataset = plan.get("dataset")
    if not all(isinstance(item, dict) for item in (candidate, model, dataset)):
        raise ValueError("Plan has no selected MLX candidate.")
    assert (
        isinstance(candidate, dict)
        and isinstance(model, dict)
        and isinstance(dataset, dict)
    )
    runtime = candidate.get("runtime_contract")
    scope = {
        "bounded-smoke": "bounded-compiler-smoke-not-pilot-evidence",
        "pilot": "uninterrupted-pilot",
        "full": "uninterrupted-full-train",
    }.get(action)
    if not isinstance(runtime, dict) or scope is None:
        raise ValueError("MLX runtime contract or action is invalid.")
    path_parent = path.parent.resolve()
    if path_parent.name == "pilot-output":
        bundle_root = path_parent.parent
    elif path_parent.parent.name in {"pilot-output", "runs"}:
        bundle_root = path_parent.parent.parent
    else:
        bundle_root = path_parent
    expected = {
        "schema_version": "aptus.runtime-metrics.v1",
        "plan_id": plan.get("plan_id"),
        "candidate_id": candidate.get("candidate_id"),
        "model_revision": model.get("revision"),
        "dataset_sha256": dataset.get("source_sha256"),
        "method": candidate.get("method"),
        "training_runtime": "mlx-lm",
        "compute_backend": "mps",
        "compiler_id": runtime.get("compiler_id"),
        "scope": scope,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "memory_metric_backend": "mlx",
        "finite_train_loss": True,
        "optimizer_update_observed": True,
        "distribution": "single",
        "actual_world_size": 1,
    }
    if any(metrics.get(name) != item for name, item in expected.items()):
        raise ValueError(
            "MLX runtime metrics do not bind the plan and uninterrupted action."
        )
    try:
        _require_mlx_model_load_binding(plan, metrics.get("model_load_binding"))
    except ValueError as error:
        raise ValueError(
            "MLX runtime metrics do not prove a pinned local safe model load."
        ) from error
    _mlx_finite(metrics.get("measured_peak_bytes"), "MLX peak", positive=True)
    for name in ("active_memory_bytes", "cache_memory_bytes"):
        value = metrics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"MLX runtime metrics contain an invalid {name} value.")
    if "free_vram_bytes" in metrics:
        raise ValueError("MLX runtime metrics cannot report discrete-VRAM fields.")
    losses = metrics.get("train_loss_observations")
    if not isinstance(losses, list) or not losses:
        raise ValueError("MLX runtime metrics contain no train losses.")
    for loss in losses:
        _mlx_finite(loss, "MLX train loss")
    opportunities = metrics.get("optimizer_update_opportunities")
    updates = metrics.get("completed_optimizer_updates")
    accumulation = candidate.get("gradient_accumulation_steps")
    micro_batch = candidate.get("micro_batch_size")
    micro_iterations = metrics.get("micro_iterations")
    minimum_updates = 2 if action == "pilot" else 1
    if (
        not isinstance(accumulation, int)
        or isinstance(accumulation, bool)
        or accumulation <= 0
        or not isinstance(micro_batch, int)
        or isinstance(micro_batch, bool)
        or micro_batch <= 0
        or not isinstance(micro_iterations, int)
        or isinstance(micro_iterations, bool)
        or micro_iterations <= 0
        or micro_iterations % accumulation
        or metrics.get("global_step") != micro_iterations
        or metrics.get("gradient_accumulation_steps") != accumulation
        or not isinstance(opportunities, int)
        or isinstance(opportunities, bool)
        or opportunities != micro_iterations // accumulation
        or not isinstance(updates, int)
        or isinstance(updates, bool)
        or opportunities != updates
        or updates < minimum_updates
    ):
        raise ValueError(
            "MLX runtime metrics do not prove completed optimizer updates."
        )
    split_path = bundle_root / "data" / "mlx" / "split-contract.json"
    try:
        split_contract = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("MLX split contract is unreadable.") from error
    splits = (
        split_contract.get("splits", {}) if isinstance(split_contract, dict) else {}
    )
    train_split = splits.get("train", {}) if isinstance(splits, dict) else {}
    valid_split = splits.get("valid", {}) if isinstance(splits, dict) else {}
    train_count = metrics.get("train_examples")
    _mlx_finite(metrics.get("adapter_delta_l1"), "MLX adapter delta", positive=True)
    changed = metrics.get("changed_adapter_tensor_count")
    if not isinstance(changed, int) or isinstance(changed, bool) or changed <= 0:
        raise ValueError("MLX runtime metrics contain no changed adapter tensor.")
    validation_count = metrics.get("validation_examples")
    validation_losses = metrics.get("validation_loss_observations")
    if (
        not isinstance(split_contract, dict)
        or split_contract.get("schema_version") != "aptus.mlx-split.v1"
        or split_contract.get("micro_batch_size") != micro_batch
        or not isinstance(train_count, int)
        or isinstance(train_count, bool)
        or train_count <= 0
        or not isinstance(validation_count, int)
        or isinstance(validation_count, bool)
        or validation_count <= 0
        or train_count % micro_batch
        or validation_count % micro_batch
        or train_split.get("compiled_row_count") != train_count
        or valid_split.get("compiled_row_count") != validation_count
        or metrics.get("source_train_examples") != train_split.get("source_row_count")
        or metrics.get("source_validation_examples")
        != valid_split.get("source_row_count")
        or metrics.get("max_epochs") != plan.get("target", {}).get("max_epochs")
    ):
        raise ValueError("MLX validation example count is invalid.")
    if (
        metrics.get("finite_validation_loss") is not True
        or not isinstance(validation_losses, list)
        or not validation_losses
    ):
        raise ValueError("MLX runtime metrics contain no validation loss evidence.")
    for loss in validation_losses:
        _mlx_finite(loss, "MLX validation loss")
    if action == "pilot" and micro_iterations != 2 * accumulation:
        raise ValueError(
            "MLX pilot metrics do not use the bounded two-update schedule."
        )
    if action == "bounded-smoke" and micro_iterations > 8:
        raise ValueError("MLX preflight metrics exceed the eight-iteration bound.")
    if action == "full":
        max_epochs = int(plan.get("target", {}).get("max_epochs", 0))
        batches_per_epoch = train_count // micro_batch
        epoch_iterations = batches_per_epoch * max_epochs
        expected_iterations = math.ceil(epoch_iterations / accumulation) * accumulation
        if micro_iterations != expected_iterations:
            raise ValueError(
                "MLX full metrics do not match the dataset-derived epoch schedule."
            )
    _require_mlx_target_binding(metrics.get("trainable_target_binding"), plan)
    _require_mlx_admission(
        metrics.get("unified_memory_admission"), plan, "MLX training"
    )
    if metrics.get("run_completed") is not True:
        raise ValueError("MLX runtime metrics do not attest completed orchestration.")
    try:
        unresolved_run_root = Path(str(metrics["output_dir"]))
        if unresolved_run_root.is_symlink():
            raise ValueError("MLX runtime output cannot be a symlink.")
        run_root = unresolved_run_root.resolve(strict=True)
    except (KeyError, OSError) as error:
        raise ValueError("MLX runtime output is missing.") from error
    parent = (
        (bundle_root / "pilot-output").resolve()
        if action in {"bounded-smoke", "pilot"}
        else (bundle_root / "runs").resolve()
    )
    prefix = "run_" if action == "full" else action + "_"
    if (
        run_root.parent != parent
        or not run_root.name.startswith(prefix)
        or metrics.get("run_id") != run_root.name
    ):
        raise ValueError("MLX runtime output is outside its owned action root.")
    persisted_path = run_root / "metrics.json"
    if persisted_path.is_symlink():
        raise ValueError("MLX owned run metrics cannot be a symlink.")
    try:
        persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("MLX owned run metrics are unreadable.") from error
    if persisted != metrics:
        raise ValueError("MLX copied metrics do not equal the owned run metrics.")
    marker_path = run_root / ".aptus-run.json"
    if marker_path.is_symlink():
        raise ValueError("MLX run marker cannot be a symlink.")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("MLX run marker is unreadable.") from error
    marker_expected = {
        "schema_version": "aptus.mlx-run-output.v1",
        "run_id": run_root.name,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "plan_id": plan.get("plan_id"),
        "candidate_id": candidate.get("candidate_id"),
        "model_revision": model.get("revision"),
        "dataset_sha256": dataset.get("source_sha256"),
    }
    if not isinstance(marker, dict) or any(
        marker.get(name) != item for name, item in marker_expected.items()
    ):
        raise ValueError("MLX run marker does not bind the plan and action.")
    if metrics.get("run_marker_sha256") != sha256_file(marker_path):
        raise ValueError("MLX metrics do not bind the immutable run marker.")
    training_metrics_path = run_root / "training-metrics.json"
    if training_metrics_path.is_symlink():
        raise ValueError("MLX training metrics cannot be a symlink.")
    try:
        training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("MLX training metrics are unreadable.") from error
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
        raise ValueError(
            "MLX completed metrics do not preserve the exact training metrics."
        )
    manifest_path = run_root / "artifact-manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("MLX artifact manifest cannot be a symlink.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("MLX artifact manifest is unreadable.") from error
    if metrics.get("artifact_manifest") != manifest or metrics.get(
        "artifact_manifest_sha256"
    ) != sha256_file(manifest_path):
        raise ValueError("MLX metrics do not bind the immutable artifact manifest.")
    if manifest.get("plan_id") != plan.get("plan_id") or manifest.get(
        "candidate_id"
    ) != candidate.get("candidate_id"):
        raise ValueError("MLX artifact manifest does not bind the plan.")
    _require_mlx_artifact_manifest(manifest, run_root=run_root, action=action)
    adapter_value = metrics.get("adapter_path")
    if not isinstance(adapter_value, str):
        raise ValueError("MLX metrics have no adapter path.")
    unresolved_adapter_dir = bundle_root / adapter_value
    if unresolved_adapter_dir.is_symlink():
        raise ValueError("MLX adapter directory cannot be a symlink.")
    adapter_dir = unresolved_adapter_dir.resolve()
    expected_adapter_dir = run_root / ("final" if action == "full" else "adapters")
    if adapter_dir != expected_adapter_dir or not adapter_dir.is_dir():
        raise ValueError("MLX adapter directory escapes its owned run.")
    expected_adapter = metrics.get("adapter_manifest")
    if any(item.is_symlink() for item in adapter_dir.iterdir()):
        raise ValueError("MLX adapter files cannot be symlinks.")
    observed_adapter = [
        {
            "path": item.name,
            "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(item for item in adapter_dir.iterdir() if item.is_file())
    ]
    if (
        not isinstance(expected_adapter, list)
        or observed_adapter != expected_adapter
        or [item["path"] for item in observed_adapter]
        != ["adapter_config.json", "adapters.safetensors"]
    ):
        raise ValueError("MLX adapter manifest does not match the saved adapter.")
    if action in {"pilot", "full"}:
        reload_path = run_root / "reload-evidence.json"
        if reload_path.is_symlink():
            raise ValueError("MLX reload evidence cannot be a symlink.")
        try:
            reload_value = json.loads(reload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("MLX reload evidence is unreadable.") from error
        reload_expected = {
            "schema_version": "aptus.mlx-reload-evidence.v1",
            "plan_id": plan.get("plan_id"),
            "candidate_id": candidate.get("candidate_id"),
            "model_revision": model.get("revision"),
            "dataset_sha256": dataset.get("source_sha256"),
            "method": candidate.get("method"),
            "training_runtime": "mlx-lm",
            "compute_backend": "mps",
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
            "fresh_process_observed": True,
            "generation_max_tokens": 4,
        }
        tokens = (
            reload_value.get("generation_tokens")
            if isinstance(reload_value, dict)
            else None
        )
        parent_pid = (
            reload_value.get("parent_pid") if isinstance(reload_value, dict) else None
        )
        verifier_pid = (
            reload_value.get("verifier_pid") if isinstance(reload_value, dict) else None
        )
        adapter_digest = _json_hash(observed_adapter)
        if (
            not isinstance(reload_value, dict)
            or any(
                reload_value.get(name) != item for name, item in reload_expected.items()
            )
            or metrics.get("reload_evidence") != reload_value
            or metrics.get("reload_evidence_sha256") != sha256_file(reload_path)
            or not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or not 1 <= tokens <= 4
            or not isinstance(parent_pid, int)
            or isinstance(parent_pid, bool)
            or parent_pid <= 0
            or not isinstance(verifier_pid, int)
            or isinstance(verifier_pid, bool)
            or verifier_pid <= 0
            or parent_pid == verifier_pid
            or reload_value.get("adapter_manifest_sha256") != adapter_digest
            or not isinstance(reload_value.get("generation_text_sha256"), str)
            or len(reload_value["generation_text_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in reload_value["generation_text_sha256"]
            )
        ):
            raise ValueError(
                "MLX metrics do not prove fresh-process bounded generation."
            )
        _mlx_finite(
            reload_value.get("measured_peak_bytes"), "MLX reload peak", positive=True
        )
        _require_mlx_admission(
            reload_value.get("unified_memory_admission"), plan, "MLX reload"
        )
    elif (
        metrics.get("reload_evidence") is not None
        or metrics.get("reload_evidence_sha256") is not None
    ):
        raise ValueError(
            "Bounded MLX preflight cannot claim fresh-process reload evidence."
        )
    expected_proof_files = {
        ".aptus-run.json",
        "training-metrics.json",
        f"{adapter_dir.name}/adapter_config.json",
        f"{adapter_dir.name}/adapters.safetensors",
    }
    if action in {"pilot", "full"}:
        expected_proof_files.add("reload-evidence.json")
    manifest_entries = manifest.get("files")
    manifested_paths = (
        {
            str(item.get("path"))
            for item in manifest_entries
            if isinstance(item, Mapping)
        }
        if isinstance(manifest_entries, list)
        else set()
    )
    if manifested_paths != expected_proof_files:
        raise ValueError("MLX artifact manifest does not cover the exact proof files.")
    expected_actual_files = expected_proof_files | {
        "artifact-manifest.json",
        "metrics.json",
    }
    if action == "full":
        expected_actual_files.add("final-export.json")
    actual_files = {
        item.relative_to(run_root).as_posix()
        for item in run_root.rglob("*")
        if item.is_file() or item.is_symlink()
    }
    if actual_files != expected_actual_files:
        raise ValueError("MLX owned run contains an unexpected or missing file.")
    if action == "full":
        export_path = run_root / "final-export.json"
        if export_path.is_symlink():
            raise ValueError("MLX final export cannot be a symlink.")
        try:
            final_export = json.loads(export_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("MLX final export is unreadable.") from error
        expected_export = {
            "schema_version": "aptus.mlx-final-export.v1",
            "verification_level": "immutable-adapter-file-tree",
            "plan_id": plan.get("plan_id"),
            "candidate_id": candidate.get("candidate_id"),
            "model_revision": model.get("revision"),
            "dataset_sha256": dataset.get("source_sha256"),
            "method": candidate.get("method"),
            "training_runtime": "mlx-lm",
            "compute_backend": "mps",
            "distribution": "single",
            "world_size": 1,
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
            "files": observed_adapter,
            "total_bytes": sum(item["size_bytes"] for item in observed_adapter),
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "reload_evidence_sha256": sha256_file(run_root / "reload-evidence.json"),
        }
        if (
            final_export != expected_export
            or metrics.get("final_export") != final_export
        ):
            raise ValueError("MLX final export is mutable or unbound.")
    elif metrics.get("final_export") is not None:
        raise ValueError("Only a confirmed full MLX run may publish a final export.")
    return metrics


def _read_preflight_metrics(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Measured-preflight metrics are unreadable.") from error
    if not isinstance(metrics, dict):
        raise ValueError("Measured-preflight metrics must be a JSON object.")
    candidate = plan.get("recommended")
    if not isinstance(candidate, dict):
        raise ValueError("Plan has no selected candidate for measured preflight.")
    runtime_contract = candidate.get("runtime_contract")
    runtime_id = (
        runtime_contract.get("training_runtime")
        if isinstance(runtime_contract, dict)
        else "transformers-peft-cuda"
    )
    if runtime_id == "mlx-lm":
        return _read_mlx_runtime_metrics(path, plan, action="bounded-smoke")
    expected = {
        "schema_version": (
            "aptus.runtime-metrics.v1"
            if runtime_id == "mlx-lm"
            else "aptus.preflight-metrics.v1"
        ),
        "candidate_id": candidate.get("candidate_id"),
        "method": candidate.get("method"),
    }
    expected.update(
        precision=candidate.get("precision"),
        quantization=candidate.get("quantization"),
        distribution=candidate.get("distribution"),
        world_size=candidate.get("world_size"),
        scope="synthetic-method-preflight-not-model-data-pilot",
    )
    for name, value in expected.items():
        if metrics.get(name) != value:
            raise ValueError(f"Measured-preflight metrics do not bind {name}.")
    peak_key = "measured_peak_cuda_bytes"
    measured_peak = metrics.get(peak_key)
    if (
        not isinstance(measured_peak, int)
        or isinstance(measured_peak, bool)
        or measured_peak <= 0
    ):
        raise ValueError(
            f"Measured-preflight metrics require a positive {peak_key} integer."
        )
    require_trainable_parameter_census(
        metrics.get("trainable_parameter_census"),
        method=str(candidate.get("method")),
    )
    return metrics


def _completed_run_evidence_is_current(
    previous: ValidationReport, bundle_dir: Path, plan: dict[str, Any]
) -> bool:
    final_export = previous.final_export
    measured_run = previous.measured_run
    candidate = plan.get("recommended")
    if (
        not isinstance(final_export, Mapping)
        or not isinstance(measured_run, Mapping)
        or not isinstance(candidate, dict)
        or not previous.measured_run_completed_at
    ):
        return False
    runtime_contract = candidate.get("runtime_contract")
    runtime_id = (
        runtime_contract.get("training_runtime")
        if isinstance(runtime_contract, Mapping)
        else "transformers-peft-cuda"
    )
    if runtime_id == "mlx-lm":
        try:
            runs_root = (bundle_dir / "runs").resolve()
            run_dir = Path(str(measured_run["output_dir"])).resolve(strict=True)
            if run_dir.parent != runs_root or not run_dir.name.startswith("run_"):
                return False
            metrics_path = run_dir / "metrics.json"
            metrics = _read_mlx_runtime_metrics(metrics_path, plan, action="full")
            export_path = run_dir / "final-export.json"
            export = json.loads(export_path.read_text(encoding="utf-8"))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(export, dict) or metrics.get("final_export") != export:
            return False
        expected_final_report = {
            "path": str((run_dir / "final").resolve()),
            "manifest_sha256": sha256_file(export_path),
            "total_bytes": export.get("total_bytes"),
            "plan_id": plan.get("plan_id"),
            "candidate_id": candidate.get("candidate_id"),
            "distribution": "single",
            "world_size": 1,
            "training_runtime": "mlx-lm",
            "artifact_manifest_sha256": metrics.get("artifact_manifest_sha256"),
            "reload_evidence_sha256": metrics.get("reload_evidence_sha256"),
            "export_contract": export,
        }
        expected_measured_report = {
            "output_dir": str(run_dir),
            "metrics_sha256": sha256_file(metrics_path),
            "global_step": metrics.get("global_step"),
            "completed_optimizer_updates": metrics.get("completed_optimizer_updates"),
            "measured_peak_bytes": metrics.get("measured_peak_bytes"),
            "plan_id": plan.get("plan_id"),
            "candidate_id": candidate.get("candidate_id"),
            "distribution": "single",
            "world_size": 1,
            "training_runtime": "mlx-lm",
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
        }
        return (
            dict(final_export) == expected_final_report
            and dict(measured_run) == expected_measured_report
        )
    expected_binding = {
        "plan_id": plan.get("plan_id"),
        "candidate_id": candidate.get("candidate_id"),
        "distribution": candidate.get("distribution"),
        "world_size": candidate.get("world_size"),
    }
    if any(final_export.get(name) != value for name, value in expected_binding.items()):
        return False
    if any(measured_run.get(name) != value for name, value in expected_binding.items()):
        return False
    try:
        runs_root = (bundle_dir / "runs").resolve()
        run_dir = Path(str(measured_run["output_dir"])).resolve(strict=True)
        final_dir = Path(str(final_export["path"])).resolve(strict=True)
    except (KeyError, OSError):
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
    if final_export.get("manifest_sha256") != sha256_file(
        export_path
    ) or measured_run.get("metrics_sha256") != sha256_file(metrics_path):
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
        or export.get("method") != candidate.get("method")
        or export.get("distribution") != candidate.get("distribution")
        or export.get("world_size") != candidate.get("world_size")
        or metrics.get("plan_id") != plan.get("plan_id")
        or metrics.get("candidate_id") != candidate.get("candidate_id")
        or metrics.get("distribution") != candidate.get("distribution")
        or metrics.get("actual_world_size") != candidate.get("world_size")
        or metrics.get("global_step") != measured_run.get("global_step")
        or metrics.get("per_rank_cuda_peaks") != measured_run.get("per_rank_cuda_peaks")
        or metrics.get("final_export") != export
    ):
        return False
    entries = export.get("files")
    if not isinstance(entries, list) or not entries:
        return False
    observed_paths: set[str] = set()
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
        if entry.get("size_bytes") != size or entry.get("sha256") != sha256_file(
            artifact
        ):
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
        and final_export.get("total_bytes") == observed_total
    )


def _write_report(path: Path, report: ValidationReport) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(to_primitive(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def _report_lock(bundle_dir: Path) -> Any:
    path = bundle_dir / ".validation-report.lock"
    with path.open("a+b") as lock_file:
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


def _read_report(path: Path) -> ValidationReport | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        state = ValidationState(value["state"])
        findings = tuple(
            ValidationFinding(
                code=str(item["code"]),
                message=str(item["message"]),
                severity=str(item["severity"]),
                path=str(item["path"]) if item.get("path") is not None else None,
            )
            for item in value.get("findings", [])
            if isinstance(item, dict)
        )
        command = value.get("smoke_command")
        return ValidationReport(
            state=state,
            findings=findings,
            checked_files=tuple(str(item) for item in value.get("checked_files", [])),
            artifact_fingerprint=str(value.get("artifact_fingerprint", "")),
            smoke_command=(
                tuple(str(item) for item in command)
                if isinstance(command, list)
                else None
            ),
            runtime_evidence=tuple(
                str(item) for item in value.get("runtime_evidence", [])
            ),
            validation_level=str(value.get("validation_level", "contract")),
            bindings={
                str(key): str(item) for key, item in value.get("bindings", {}).items()
            }
            if isinstance(value.get("bindings"), dict)
            else {},
            validator_version=str(value.get("validator_version", "aptus-validator-v2")),
            validated_at=value.get("validated_at"),
            preflight_metrics=(
                value.get("preflight_metrics")
                if isinstance(value.get("preflight_metrics"), dict)
                else None
            ),
            pilot_metrics=(
                value.get("pilot_metrics")
                if isinstance(value.get("pilot_metrics"), dict)
                else None
            ),
            final_export=(
                value.get("final_export")
                if isinstance(value.get("final_export"), dict)
                else None
            ),
            measured_run=(
                value.get("measured_run")
                if isinstance(value.get("measured_run"), dict)
                else None
            ),
            measured_run_completed_at=(
                str(value["measured_run_completed_at"])
                if value.get("measured_run_completed_at") is not None
                else None
            ),
            latest_recheck=(
                value.get("latest_recheck")
                if isinstance(value.get("latest_recheck"), dict)
                else None
            ),
            parent_promotion=(
                value.get("parent_promotion")
                if isinstance(value.get("parent_promotion"), dict)
                else None
            ),
        )
    except (
        KeyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


def _preserves_stronger_attestation(
    previous: ValidationReport,
    current: ValidationReport,
    bundle_dir: Path,
) -> bool:
    plan: dict[str, Any] | None = None
    previous_rank = STATE_RANK.get(previous.state, 0)
    current_rank = STATE_RANK.get(current.state, 0)
    if previous_rank <= current_rank or current.state == ValidationState.INVALID:
        return False
    if (
        not current.artifact_fingerprint
        or previous.artifact_fingerprint != current.artifact_fingerprint
        or previous.bindings.get("bundle") != current.artifact_fingerprint
    ):
        return False
    for key in ("dataset", "plan_id", "candidate_id", "model_revision"):
        if previous.bindings.get(key) != current.bindings.get(key):
            return False
    historical_run = previous.state == ValidationState.MEASURED_RUN_PASS
    if not historical_run and previous.bindings.get(
        "environment"
    ) != current.bindings.get("environment"):
        return False
    if (
        not historical_run
        and previous_rank >= STATE_RANK[ValidationState.MODEL_DATA_PASS]
    ):
        if plan is None:
            try:
                loaded_plan = json.loads(
                    (bundle_dir / "plan.json").read_text(encoding="utf-8")
                )
            except (OSError, RecursionError, ValueError):
                return False
            if not isinstance(loaded_plan, dict):
                return False
            plan = loaded_plan
        candidate = plan.get("recommended")
        if not isinstance(candidate, dict):
            return False
        world_size = candidate.get("world_size")
        device_indices = candidate.get("device_indices")
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or not isinstance(device_indices, list)
            or len(device_indices) != world_size
        ):
            return False
        runtime_contract = candidate.get("runtime_contract")
        runtime_id = (
            runtime_contract.get("training_runtime")
            if isinstance(runtime_contract, dict)
            else "transformers-peft-cuda"
        )
        if runtime_id == "transformers-peft-cuda":
            try:
                current_hardware = _actual_hardware_binding(device_indices)
            except (RuntimeError, ValueError):
                return False
            if previous.bindings.get("hardware") != current_hardware:
                return False
    if previous_rank >= STATE_RANK[ValidationState.MEASURED_PREFLIGHT_PASS]:
        metrics_path = bundle_dir / "preflight-metrics.json"
        try:
            loaded_plan = json.loads(
                (bundle_dir / "plan.json").read_text(encoding="utf-8")
            )
            if not isinstance(loaded_plan, dict):
                return False
            plan = loaded_plan
            metrics = _read_preflight_metrics(metrics_path, plan)
        except (OSError, RecursionError, TypeError, ValueError):
            return False
        if (
            previous.bindings.get("preflight_metrics") != sha256_file(metrics_path)
            or previous.preflight_metrics != metrics
        ):
            return False
    if previous_rank >= STATE_RANK[ValidationState.PILOT_PASS]:
        metrics_path = bundle_dir / "pilot-output" / "metrics.json"
        if not metrics_path.is_file() or previous.bindings.get(
            "pilot_metrics"
        ) != sha256_file(metrics_path):
            return False
        if plan is None:
            try:
                loaded_plan = json.loads(
                    (bundle_dir / "plan.json").read_text(encoding="utf-8")
                )
            except (OSError, RecursionError, ValueError):
                return False
            if not isinstance(loaded_plan, dict):
                return False
            plan = loaded_plan
        runtime_contract = plan.get("recommended", {}).get("runtime_contract")
        if (
            isinstance(runtime_contract, dict)
            and runtime_contract.get("training_runtime") == "mlx-lm"
        ):
            try:
                pilot_metrics = _read_mlx_runtime_metrics(
                    metrics_path, plan, action="pilot"
                )
            except ValueError:
                return False
            if previous.pilot_metrics != pilot_metrics:
                return False
    if previous.state == ValidationState.MEASURED_RUN_PASS:
        if plan is None:
            try:
                loaded_plan = json.loads(
                    (bundle_dir / "plan.json").read_text(encoding="utf-8")
                )
            except (OSError, RecursionError, ValueError):
                return False
            if not isinstance(loaded_plan, dict):
                return False
            plan = loaded_plan
        if not _completed_run_evidence_is_current(previous, bundle_dir, plan):
            return False
    return True


def validate_bundle(
    bundle_dir: Path,
    *,
    level: ValidationLevel = "static",
    run: bool = False,
) -> ValidationReport:
    """Validate only the requested evidence level and never synthesize runtime passes."""

    if level not in LEVELS:
        raise ValueError(f"Unknown validation level: {level}")
    bundle_dir = bundle_dir.resolve()
    if not bundle_dir.is_dir():
        raise ValueError(f"Bundle directory does not exist: {bundle_dir}")
    report_path = bundle_dir / "validation-report.json"
    findings: list[ValidationFinding] = []
    checked: set[str] = set()
    runtime_evidence: list[str] = []
    portable_bindings: dict[str, str] = {}
    portable_preflight_metrics: dict[str, Any] | None = None
    portable_pilot_metrics: dict[str, Any] | None = None

    for relative in REQUIRED_BUNDLE_FILES:
        path = bundle_dir / relative
        if path.is_file():
            checked.add(relative)
        else:
            findings.append(
                _finding(
                    "MISSING_FILE",
                    f"Required bundle file is missing: {relative}",
                    path=relative,
                )
            )

    plan = (
        _load_json(
            bundle_dir / "plan.json",
            findings,
            "PLAN_JSON_ERROR",
            require_object=True,
        )
        if (bundle_dir / "plan.json").is_file()
        else None
    )
    manifest = (
        _load_json(
            bundle_dir / "bundle-manifest.json",
            findings,
            "MANIFEST_JSON_ERROR",
            require_object=True,
        )
        if (bundle_dir / "bundle-manifest.json").is_file()
        else None
    )
    snapshot_path = bundle_dir / "policy/model-policy-snapshot.v1.json"
    snapshot: Any = None
    snapshot_digest: str | None = None
    snapshot_loaded = False
    if not snapshot_path.is_file():
        findings.append(
            _finding(
                "POLICY_SNAPSHOT_MISSING",
                "Portable model policy snapshot is missing.",
                path="policy/model-policy-snapshot.v1.json",
            )
        )
    else:
        checked.add("policy/model-policy-snapshot.v1.json")
        try:
            snapshot_bytes = snapshot_path.read_bytes()
            snapshot = json.loads(snapshot_bytes)
        except (OSError, ValueError, RecursionError):
            findings.append(
                _finding(
                    "POLICY_SNAPSHOT_JSON_ERROR",
                    "Portable model policy snapshot is not valid UTF-8 JSON.",
                    path="policy/model-policy-snapshot.v1.json",
                )
            )
        else:
            snapshot_loaded = True
        if snapshot_loaded:
            try:
                validate_model_policy_snapshot(snapshot)
                canonical_bytes = model_policy_snapshot_bytes(snapshot)
                snapshot_digest = model_policy_snapshot_sha256(snapshot)
            except (RecursionError, TypeError, ValueError) as error:
                findings.append(
                    _finding(
                        "POLICY_SNAPSHOT_CONTRACT",
                        f"Portable model policy snapshot is invalid: {error}",
                        path="policy/model-policy-snapshot.v1.json",
                    )
                )
            else:
                if snapshot_bytes != canonical_bytes:
                    findings.append(
                        _finding(
                            "POLICY_SNAPSHOT_NONCANONICAL",
                            "Portable model policy snapshot is not canonically encoded.",
                            path="policy/model-policy-snapshot.v1.json",
                        )
                    )
                bindings: dict[str, Any] = {
                    "snapshot": snapshot_digest,
                    "plan": plan.get("model_policy_snapshot_sha256")
                    if isinstance(plan, dict)
                    else None,
                    "manifest": manifest.get("policy_snapshot_sha256")
                    if isinstance(manifest, dict)
                    else None,
                    "host": current_model_policy_snapshot_sha256(),
                }
                invalid_bindings = [
                    name
                    for name, value in bindings.items()
                    if not _is_sha256_digest(value)
                ]
                differing_bindings = (
                    [
                        name
                        for name in ("plan", "manifest", "host")
                        if name not in invalid_bindings
                        and bindings[name] != bindings["snapshot"]
                    ]
                    if "snapshot" not in invalid_bindings
                    else []
                )
                if invalid_bindings or differing_bindings:
                    message_parts = []
                    if invalid_bindings:
                        message_parts.append(
                            "digest bindings must be lowercase 64-character "
                            "hexadecimal text; invalid bindings: "
                            + ", ".join(invalid_bindings)
                        )
                    if differing_bindings:
                        message_parts.append(
                            "valid bindings differing from snapshot: "
                            + ", ".join(differing_bindings)
                        )
                    findings.append(
                        _finding(
                            "POLICY_SNAPSHOT_DIGEST",
                            "Policy snapshot " + "; ".join(message_parts) + ".",
                            path="policy/model-policy-snapshot.v1.json",
                        )
                    )
    is_mlx_bundle = False
    validated_plan: dict[str, Any] | None = None
    if plan is not None:
        plan_contract_errors = validate_plan_payload(
            plan, root=bundle_dir, verify_dataset=True
        )
        for error in plan_contract_errors:
            findings.append(_finding("PLAN_CONTRACT_ERROR", error, path="plan.json"))
        if not plan_contract_errors:
            validated_plan = plan
            runtime_contract = validated_plan["recommended"]["runtime_contract"]
            is_mlx_bundle = runtime_contract.get("training_runtime") == "mlx-lm"
            if is_mlx_bundle:
                for relative in ("reload.py", "eval.py"):
                    path = bundle_dir / relative
                    if path.is_file():
                        checked.add(relative)
                    else:
                        findings.append(
                            _finding(
                                "MISSING_FILE",
                                f"Required MLX bundle file is missing: {relative}",
                                path=relative,
                            )
                        )
            else:
                campaign_events_path = bundle_dir / "campaign_events.py"
                if campaign_events_path.is_file():
                    checked.add("campaign_events.py")
                else:
                    findings.append(
                        _finding(
                            "MISSING_FILE",
                            "Required CUDA bundle file is missing: campaign_events.py",
                            path="campaign_events.py",
                        )
                    )
            try:
                restored = training_plan_from_primitive(plan)
                replanned = plan_training(
                    model=restored.model,
                    dataset=restored.dataset,
                    hardware=restored.hardware,
                    target=restored.target,
                    inspection_receipt=restored.inspection_receipt,
                )
            except (
                AttributeError,
                KeyError,
                RecursionError,
                TypeError,
                ValueError,
            ) as error:
                findings.append(
                    _finding(
                        "PLANNER_PARITY_ERROR",
                        f"Could not reproduce the plan from its bound facts: {error}",
                        path="plan.json",
                    )
                )
            else:
                reproduced = to_primitive(replanned)
                if (
                    reproduced["candidates"] != plan.get("candidates")
                    or reproduced["recommended"] != plan.get("recommended")
                    or reproduced["plan_id"] != plan.get("plan_id")
                ):
                    findings.append(
                        _finding(
                            "PLANNER_PARITY_MISMATCH",
                            "Candidates or recommendation do not match deterministic Aptus v0.2 replanning.",
                            path="plan.json",
                        )
                    )

    if isinstance(manifest, dict):
        if manifest.get("schema_version") != "aptus.bundle.v3":
            findings.append(
                _finding(
                    "MANIFEST_SCHEMA",
                    "Manifest schema must be aptus.bundle.v3.",
                    path="bundle-manifest.json",
                )
            )
        if (
            manifest.get("policy_snapshot_path")
            != "policy/model-policy-snapshot.v1.json"
        ):
            findings.append(
                _finding(
                    "POLICY_SNAPSHOT_PATH",
                    "Manifest policy snapshot path is invalid.",
                    path="bundle-manifest.json",
                )
            )
        if (bundle_dir / "plan.json").is_file() and manifest.get(
            "plan_sha256"
        ) != sha256_file(bundle_dir / "plan.json"):
            findings.append(
                _finding(
                    "MANIFEST_PLAN_DIGEST",
                    "Manifest plan digest does not match plan.json.",
                    path="bundle-manifest.json",
                )
            )
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            findings.append(
                _finding(
                    "MANIFEST_EMPTY",
                    "Manifest files must be a non-empty list.",
                    path="bundle-manifest.json",
                )
            )
        else:
            seen: set[str] = set()
            for item in entries:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    findings.append(
                        _finding(
                            "MANIFEST_ENTRY_INVALID",
                            "Every manifest entry requires a path.",
                            path="bundle-manifest.json",
                        )
                    )
                    continue
                relative = item["path"]
                if (
                    relative in seen
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                ):
                    findings.append(
                        _finding(
                            "MANIFEST_PATH_INVALID",
                            f"Unsafe or duplicate manifest path: {relative}",
                            path="bundle-manifest.json",
                        )
                    )
                    continue
                seen.add(relative)
                path = bundle_dir / relative
                if not path.is_file():
                    findings.append(
                        _finding(
                            "MANIFEST_FILE_MISSING",
                            f"Manifest file is absent: {relative}",
                            path=relative,
                        )
                    )
                    continue
                checked.add(relative)
                if (
                    item.get("sha256") != sha256_file(path)
                    or item.get("size_bytes") != path.stat().st_size
                ):
                    findings.append(
                        _finding(
                            "MANIFEST_MISMATCH",
                            f"Checksum or size mismatch: {relative}",
                            path=relative,
                        )
                    )
        for error in validate_bundle_manifest(bundle_dir):
            findings.append(
                _finding("MANIFEST_INTEGRITY", error, path="bundle-manifest.json")
            )

    if LEVELS.index(level) >= LEVELS.index("static"):
        python_sources = [
            *([] if is_mlx_bundle else ["campaign_events.py"]),
            "plan_contract.py",
            "policy_snapshot.py",
            "preflight.py",
            "run.py",
            "runtime_lease.py",
            "train.py",
            "validate.py",
        ]
        if is_mlx_bundle:
            python_sources.extend(["reload.py", "eval.py"])
        for relative in python_sources:
            path = bundle_dir / relative
            if not path.is_file():
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except SyntaxError as error:
                findings.append(
                    _finding(
                        "PYTHON_PARSE_ERROR",
                        f"{error.msg} at line {error.lineno}.",
                        path=relative,
                    )
                )
        template_sources = [
            *([] if is_mlx_bundle else ["campaign_events.py"]),
            "README.md",
            "decision-report.md",
            "runbook.md",
            "run.py",
            "runtime_lease.py",
            "train.py",
            "preflight.py",
            "validate.py",
        ]
        if is_mlx_bundle:
            template_sources.extend(["reload.py", "eval.py"])
        for relative in template_sources:
            path = bundle_dir / relative
            if path.is_file() and any(
                marker in path.read_text(encoding="utf-8")
                for marker in ("{{", "}}", "TODO")
            ):
                findings.append(
                    _finding(
                        "UNRESOLVED_TEMPLATE",
                        "Generated file contains an unresolved marker.",
                        path=relative,
                    )
                )

    expected_requirements: tuple[str, ...] = ()
    if validated_plan is not None:
        method = validated_plan["recommended"].get("method")
        try:
            runtime_contract = validated_plan["recommended"].get("runtime_contract")
            runtime_id = (
                runtime_contract.get("training_runtime")
                if isinstance(runtime_contract, dict)
                else "transformers-peft-cuda"
            )
            expected_requirements = bundle_requirements(
                method, training_runtime=runtime_id
            )
        except ValueError:
            expected_requirements = ()
        requirements_path = bundle_dir / "requirements.txt"
        if requirements_path.is_file():
            actual = tuple(
                line.strip()
                for line in requirements_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if actual != expected_requirements:
                findings.append(
                    _finding(
                        "DEPENDENCY_SET_MISMATCH",
                        "requirements.txt does not equal the method-specific direct pinned set.",
                        path="requirements.txt",
                    )
                )
        train_path = bundle_dir / "train.py"
        if train_path.is_file():
            source = train_path.read_text(encoding="utf-8")
            values = (
                validated_plan["model"].get("model_id"),
                validated_plan["dataset"].get("source_path"),
            )
            if any(
                isinstance(value, str) and value and value in source for value in values
            ):
                findings.append(
                    _finding(
                        "USER_VALUE_EMBEDDED_IN_SOURCE",
                        "Executable source contains a user model or dataset value.",
                        path="train.py",
                    )
                )
        trainer_path = bundle_dir / "config" / "trainer.json"
        if trainer_path.is_file():
            trainer = _load_json(
                trainer_path,
                findings,
                "TRAINER_CONFIG_JSON_ERROR",
                require_object=True,
            )
            candidate = validated_plan["recommended"]
            target = validated_plan["target"]
            expected = {
                "schema_version": "aptus.trainer-config.v3",
                "task": target.get("task"),
                "sequence_length": target.get("sequence_length"),
                "packing": target.get("packing"),
                "per_device_train_batch_size": candidate.get("micro_batch_size"),
                "gradient_accumulation_steps": candidate.get(
                    "gradient_accumulation_steps"
                ),
                "effective_global_batch_size": candidate.get("effective_batch_size"),
                "world_size": candidate.get("world_size"),
                "precision": candidate.get("precision"),
                "optimizer_steps": target.get("optimizer_steps"),
                "split_seed": target.get("split_seed"),
                "training_seed": target.get("training_seed"),
                "data_order_seed": target.get("data_order_seed"),
            }
            if isinstance(trainer, dict):
                for key, value in expected.items():
                    if trainer.get(key) != value:
                        findings.append(
                            _finding(
                                "TRAINER_CONFIG_MISMATCH",
                                f"config/trainer.json {key} does not match plan.json.",
                                path="config/trainer.json",
                            )
                        )
                counter_contract = trainer.get("counter_contract")
                if (
                    not isinstance(counter_contract, dict)
                    or counter_contract.get("schema_version")
                    != "aptus.training-counters.v1"
                ):
                    findings.append(
                        _finding(
                            "TRAINER_CONFIG_MISMATCH",
                            "config/trainer.json lacks the Phase 3 counter contract.",
                            path="config/trainer.json",
                        )
                    )

    structural_errors = any(item.severity == "error" for item in findings)
    achieved_level: ValidationLevel = "contract"
    if LEVELS.index(level) >= LEVELS.index("static"):
        achieved_level = "static"
    runtime_level = LEVELS.index(level) >= LEVELS.index("dependency")
    if runtime_level and not run:
        findings.append(
            _finding(
                "RUNTIME_NOT_EXECUTED",
                f"{level} was requested without run=true; report remains at static-pass.",
                severity="warning",
            )
        )
    elif runtime_level and not structural_errors:
        runtime_contract = (
            validated_plan["recommended"]["runtime_contract"]
            if validated_plan is not None
            else None
        )
        runtime_id = (
            runtime_contract.get("training_runtime")
            if isinstance(runtime_contract, dict)
            else "transformers-peft-cuda"
        )
        interpreter = sys.executable
        if runtime_id in {"mlx-lm", "pytorch-mps"}:
            interpreter = resolve_runtime_interpreter(runtime_id).path
        command = [interpreter, str(bundle_dir / "validate.py"), "--level", level]
        with tempfile.TemporaryFile() as runtime_log:
            completed = subprocess.run(
                command,
                cwd=bundle_dir,
                stdout=runtime_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            runtime_log.seek(0, os.SEEK_END)
            length = runtime_log.tell()
            runtime_log.seek(max(0, length - 16_000))
            output_tail = runtime_log.read().decode("utf-8", errors="replace")
        runtime_evidence.extend(
            (
                "command=" + json.dumps(command),
                f"return_code={completed.returncode}",
                "output_tail=" + output_tail,
            )
        )
        if completed.returncode:
            findings.append(
                _finding(
                    "RUNTIME_VALIDATION_FAILED",
                    f"{level} validation exited {completed.returncode}.",
                )
            )
        else:
            runtime_attestation_valid = True
            portable_report_path = bundle_dir / "validation-report.json"
            try:
                portable_report = json.loads(
                    portable_report_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                portable_report = None
            if not isinstance(portable_report, dict) or not isinstance(
                portable_report.get("bindings"), dict
            ):
                findings.append(
                    _finding(
                        "RUNTIME_ATTESTATION_INVALID",
                        "Runtime validation did not publish a readable bound validation report.",
                        path="validation-report.json",
                    )
                )
                runtime_attestation_valid = False
            else:
                portable_bindings = {
                    str(key): str(value)
                    for key, value in portable_report["bindings"].items()
                }
            if LEVELS.index(level) >= LEVELS.index("measured-preflight") and isinstance(
                validated_plan, dict
            ):
                metrics_path = bundle_dir / "preflight-metrics.json"
                try:
                    measured_metrics = _read_preflight_metrics(
                        metrics_path, validated_plan
                    )
                except ValueError as error:
                    findings.append(
                        _finding(
                            "PREFLIGHT_METRICS_INVALID",
                            str(error),
                            path="preflight-metrics.json",
                        )
                    )
                    runtime_attestation_valid = False
                else:
                    expected_digest = sha256_file(metrics_path)
                    if (
                        portable_bindings.get("preflight_metrics") != expected_digest
                        or not isinstance(portable_report, dict)
                        or portable_report.get("preflight_metrics") != measured_metrics
                    ):
                        findings.append(
                            _finding(
                                "PREFLIGHT_METRICS_UNBOUND",
                                "Runtime validation report does not bind the exact measured-preflight metrics.",
                                path="validation-report.json",
                            )
                        )
                        runtime_attestation_valid = False
                    else:
                        portable_preflight_metrics = measured_metrics
            if (
                level == "pilot"
                and runtime_id == "mlx-lm"
                and validated_plan is not None
            ):
                pilot_path = bundle_dir / "pilot-output" / "metrics.json"
                try:
                    measured_pilot = _read_mlx_runtime_metrics(
                        pilot_path, validated_plan, action="pilot"
                    )
                except ValueError as error:
                    findings.append(
                        _finding(
                            "PILOT_METRICS_INVALID",
                            str(error),
                            path="pilot-output/metrics.json",
                        )
                    )
                    runtime_attestation_valid = False
                else:
                    expected_digest = sha256_file(pilot_path)
                    if (
                        portable_bindings.get("pilot_metrics") != expected_digest
                        or not isinstance(portable_report, dict)
                        or portable_report.get("pilot_metrics") != measured_pilot
                    ):
                        findings.append(
                            _finding(
                                "PILOT_METRICS_UNBOUND",
                                "Runtime validation report does not bind the exact MLX pilot metrics.",
                                path="validation-report.json",
                            )
                        )
                        runtime_attestation_valid = False
                    else:
                        portable_pilot_metrics = measured_pilot
            if runtime_attestation_valid:
                achieved_level = level

    has_errors = any(item.severity == "error" for item in findings)
    state = ValidationState.INVALID if has_errors else LEVEL_STATES[achieved_level]
    try:
        fingerprint = bundle_fingerprint(bundle_dir)
    except FileNotFoundError:
        fingerprint = ""
    validated_at = datetime.now(timezone.utc).isoformat()
    data_digest = (
        validated_plan["dataset"].get("source_sha256", "")
        if validated_plan is not None
        else ""
    )
    hardware_value = validated_plan["hardware"] if validated_plan is not None else {}
    planned_hardware = _json_hash(hardware_value)
    bindings = {
        "bundle": fingerprint,
        "dataset": str(data_digest),
        "environment": portable_bindings.get(
            "environment", _environment_binding(expected_requirements)
        ),
        "hardware": portable_bindings.get("hardware", planned_hardware),
        "planned_hardware": planned_hardware,
        "validator": "aptus-validator-v2",
        "validated_at": validated_at,
    }
    if validated_plan is not None:
        bindings["plan_id"] = str(validated_plan.get("plan_id", ""))
        bindings["candidate_id"] = str(
            validated_plan["recommended"].get("candidate_id", "")
        )
        bindings["model_revision"] = str(validated_plan["model"].get("revision", ""))
    preflight_metrics_path = bundle_dir / "preflight-metrics.json"
    if (
        LEVELS.index(achieved_level) >= LEVELS.index("measured-preflight")
        and preflight_metrics_path.is_file()
    ):
        bindings["preflight_metrics"] = portable_bindings.get(
            "preflight_metrics", sha256_file(preflight_metrics_path)
        )
    pilot_metrics = bundle_dir / "pilot-output" / "metrics.json"
    if achieved_level == "pilot" and pilot_metrics.is_file():
        bindings["pilot_metrics"] = portable_bindings.get(
            "pilot_metrics", sha256_file(pilot_metrics)
        )
    pilot_metrics_payload: dict[str, Any] | None = portable_pilot_metrics
    if (
        achieved_level == "pilot"
        and pilot_metrics.is_file()
        and pilot_metrics_payload is None
    ):
        try:
            loaded_pilot_metrics = json.loads(pilot_metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_pilot_metrics = None
        if isinstance(loaded_pilot_metrics, dict):
            pilot_metrics_payload = loaded_pilot_metrics
    report = ValidationReport(
        state=state,
        findings=tuple(findings),
        checked_files=tuple(sorted(checked)),
        artifact_fingerprint=fingerprint,
        smoke_command=(sys.executable, "validate.py", "--level", "measured-preflight"),
        runtime_evidence=tuple(runtime_evidence),
        validation_level=achieved_level,
        bindings=bindings,
        validated_at=validated_at,
        preflight_metrics=portable_preflight_metrics,
        pilot_metrics=pilot_metrics_payload,
    )
    with _report_lock(bundle_dir):
        latest_report = _read_report(report_path) if report_path.is_file() else None
        if latest_report is not None and _preserves_stronger_attestation(
            latest_report, report, bundle_dir
        ):
            if latest_report.state == ValidationState.MEASURED_RUN_PASS:
                latest_report = replace(
                    latest_report,
                    latest_recheck={
                        "state": report.state.value,
                        "validation_level": report.validation_level,
                        "validated_at": report.validated_at,
                        "artifact_fingerprint": report.artifact_fingerprint,
                        "findings": to_primitive(report.findings),
                    },
                )
                _write_report(report_path, latest_report)
            return latest_report
        _write_report(report_path, report)
        return report
