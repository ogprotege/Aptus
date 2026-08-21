#!/usr/bin/env python3
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
from plan_contract import (
    bundle_fingerprint,
    load_json_object,
    mlx_trainable_target_instance_total,
    validate_bundle_manifest,
    validate_plan_payload,
)
from train import (
    build_mlx_model_load_binding,
    require_method_model,
    require_mlx_model_load_binding,
    require_unified_memory_admission,
    require_unified_memory_admission_binding,
)

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
    model_data_evidence: dict | None = None,
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
    if model_data_evidence is not None:
        bindings["model_data_evidence"] = sha256(ROOT / "model-data-evidence.json")
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


def require_model_data_evidence(plan: dict, evidence: object) -> dict:
    candidate = plan["recommended"]
    expected = {
        "schema_version": "aptus.mlx-model-data-evidence.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
    }
    expected_fields = set(expected) | {
        "model_load_binding",
        "unified_memory_admission",
        "validated_at",
    }
    if (
        not isinstance(evidence, dict)
        or set(evidence) != expected_fields
        or any(evidence.get(name) != value for name, value in expected.items())
        or not isinstance(evidence.get("validated_at"), str)
        or not evidence["validated_at"]
    ):
        raise RuntimeError("MLX-LM model-data evidence does not bind the plan.")
    model_load_binding = require_mlx_model_load_binding(
        plan, evidence.get("model_load_binding")
    )
    admission = require_unified_memory_admission_binding(
        plan, evidence.get("unified_memory_admission")
    )
    if (
        model_load_binding["packed_checkpoint_binding"][
            "observed_safetensors_bytes"
        ]
        != admission["observed_safetensors_bytes"]
    ):
        raise RuntimeError(
            "MLX-LM model-data evidence binds different checkpoint byte measurements."
        )
    return evidence


def require_model_data(plan: dict) -> dict:
    from huggingface_hub import snapshot_download
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load

    model_path = Path(
        snapshot_download(
            repo_id=plan["model"]["model_id"],
            revision=plan["model"]["revision"],
        )
    ).resolve(strict=True)
    candidate = plan["recommended"]
    architecture_contract = require_method_model(plan, candidate, model_path)
    admission = require_unified_memory_admission(plan, model_path)
    model, tokenizer, config = load(
        str(model_path),
        lazy=True,
        return_config=True,
        tokenizer_config={"trust_remote_code": False},
    )
    try:
        from plan_contract import validate_model_config_against_plan

        validate_model_config_against_plan(plan["model"], config)
    except ValueError as error:
        raise RuntimeError(
            "Loaded MLX-LM config does not match the pinned model architecture contract."
        ) from error
    model_load_binding = build_mlx_model_load_binding(
        model,
        plan,
        observed_safetensors_bytes=admission["observed_safetensors_bytes"],
        architecture_contract=architecture_contract,
    )
    require_mlx_model_load_binding(plan, model_load_binding)
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
    evidence = {
        "schema_version": "aptus.mlx-model-data-evidence.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "model_load_binding": model_load_binding,
        "unified_memory_admission": admission,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    require_model_data_evidence(plan, evidence)
    evidence_path = ROOT / "model-data-evidence.json"
    temporary = evidence_path.with_name(".model-data-evidence.json.tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, evidence_path)
    return evidence


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
    try:
        model_load_binding = require_mlx_model_load_binding(
            plan, metrics.get("model_load_binding")
        )
    except RuntimeError as error:
        raise RuntimeError(
            "MLX-LM runtime metrics do not prove a pinned local safe model load."
        ) from error
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
    target_counts = binding.get("target_instance_counts")
    try:
        expected_instances = mlx_trainable_target_instance_total(
            planned,
            layer_count,
            target_counts,
            family=plan["model"].get("family"),
        )
    except ValueError as error:
        raise RuntimeError(
            "MLX-LM trainable-target binding is not exact for the plan."
        ) from error
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
        or not isinstance(binding.get("resolved_layer_keys"), list)
        or len(binding["resolved_layer_keys"]) != len(planned)
        or len(set(binding["resolved_layer_keys"])) != len(planned)
        or not isinstance(descriptor_sha256, str)
        or descriptor_sha256 != expected_descriptor_sha256
    ):
        raise RuntimeError("MLX-LM trainable-target binding is not exact for the plan.")
    try:
        admission = require_unified_memory_admission_binding(
            plan, metrics.get("unified_memory_admission")
        )
    except RuntimeError as error:
        raise RuntimeError(
            "MLX-LM runtime metrics do not bind a passing live unified-memory admission."
        ) from error
    if (
        model_load_binding["packed_checkpoint_binding"][
            "observed_safetensors_bytes"
        ]
        != admission["observed_safetensors_bytes"]
    ):
        raise RuntimeError(
            "MLX-LM runtime metrics bind different checkpoint byte measurements."
        )
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
        try:
            admission = require_unified_memory_admission_binding(
                plan, reload_evidence.get("unified_memory_admission")
            )
        except RuntimeError as error:
            raise RuntimeError(
                "MLX reload evidence does not bind packed-checkpoint admission."
            ) from error
        model_load_binding = require_mlx_model_load_binding(
            plan, metrics.get("model_load_binding")
        )
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
            or model_load_binding["packed_checkpoint_binding"][
                "observed_safetensors_bytes"
            ]
            != admission["observed_safetensors_bytes"]
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
    if rank >= STATE_RANK["model-data-pass"]:
        evidence_path = ROOT / "model-data-evidence.json"
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            require_model_data_evidence(plan, evidence)
        except (OSError, RuntimeError, json.JSONDecodeError):
            return False
        if bindings.get("model_data_evidence") != sha256(evidence_path):
            return False
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
    plan = load_json_object(ROOT / "plan.json", "Bundle plan")
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
    plan = load_json_object(ROOT / "plan.json", "Bundle plan")
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
    model_data_evidence = None
    if arguments.level in {"model-data", "measured-preflight", "pilot"}:
        model_data_evidence = require_model_data(plan)
    metrics = None
    if arguments.level in {"measured-preflight", "pilot"}:
        metrics = run_measured_preflight()
    if arguments.level != "pilot":
        promote(
            plan,
            states[arguments.level],
            model_data_evidence=model_data_evidence,
            preflight_metrics=metrics,
        )
    if arguments.level == "pilot":
        pilot_metrics = run_pilot(plan)
        promote(
            plan,
            "pilot-pass",
            model_data_evidence=model_data_evidence,
            preflight_metrics=metrics,
            pilot_metrics=pilot_metrics,
        )
    print(f"Aptus MLX-LM {arguments.level} validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
