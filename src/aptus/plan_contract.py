from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "aptus.training-plan.v2"
FORMULA_VERSION = "aptus-memory-v2"
MLX_FORMULA_VERSION = "aptus-memory-mlx-v1"
RUNTIME_CONTRACT_VERSION = "aptus.runtime-contract.v1"
CANDIDATE_STATUSES = {"feasible", "conditional", "infeasible", "unsupported"}
METHODS = {"full", "lora", "int8-lora", "qlora"}
DISTRIBUTIONS = {"single", "ddp", "fsdp"}
TRAINING_RUNTIMES = {"transformers-peft-cuda", "mlx-lm", "pytorch-mps"}
EVIDENCE_REQUIREMENTS = {"pilot-required", "implementation-required"}

# This table is intentionally self-contained because plan_contract.py is copied
# into every generated bundle. It mirrors the executable RuntimeBinding entries
# in aptus.methods and lets a bundle reject invented compiler identities without
# importing the Aptus package at validation time.
RUNTIME_BINDING_IDENTITIES = {
    ("full", "transformers-peft-cuda", "cuda"): (
        "transformers.full.v2",
        "aptus-memory-v2",
        "full-model-safetensors",
        "pilot-required",
    ),
    ("lora", "transformers-peft-cuda", "cuda"): (
        "transformers.peft-lora.v2",
        "aptus-memory-v2",
        "peft-adapter-safetensors",
        "pilot-required",
    ),
    ("lora", "mlx-lm", "mps"): (
        "mlx-lm.lora.v1",
        "aptus-memory-mlx-v1",
        "mlx-lm-adapter",
        "pilot-required",
    ),
    ("int8-lora", "transformers-peft-cuda", "cuda"): (
        "transformers.peft-int8-lora.v2",
        "aptus-memory-v2",
        "peft-adapter-safetensors",
        "pilot-required",
    ),
    ("qlora", "transformers-peft-cuda", "cuda"): (
        "transformers.peft-qlora.v2",
        "aptus-memory-v2",
        "peft-adapter-safetensors",
        "pilot-required",
    ),
    ("qlora", "mlx-lm", "mps"): (
        "mlx-lm.qlora.v1",
        "aptus-memory-mlx-v1",
        "mlx-lm-adapter",
        "pilot-required",
    ),
}
UNAVAILABLE_RUNTIME_IDENTITY = (
    None,
    "unavailable",
    None,
    "implementation-required",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_fingerprint(root: Path) -> str:
    manifest = root / "bundle-manifest.json"
    if manifest.is_file():
        return sha256_file(manifest)
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != "validation-report.json"
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\n")
    return digest.hexdigest()


def validate_bundle_manifest(root: Path) -> tuple[str, ...]:
    """Verify the immutable file set bound by bundle-manifest.json."""

    errors: list[str] = []
    if root.is_symlink():
        errors.append("Bundle root cannot be a symlink.")
    try:
        symlinks = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_symlink()
        )
    except OSError as error:
        return (f"Bundle tree could not be inspected safely: {error}",)
    if symlinks:
        errors.append("Bundle tree contains symlink(s): " + ", ".join(symlinks) + ".")
    manifest_path = root / "bundle-manifest.json"
    if not manifest_path.is_file():
        return ("Bundle manifest is missing.",)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return (f"Bundle manifest is invalid JSON: {error}",)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "aptus.bundle.v2"
    ):
        errors.append("Bundle manifest schema must be aptus.bundle.v2.")
    plan_path = root / "plan.json"
    if not plan_path.is_file() or manifest.get("plan_sha256") != sha256_file(plan_path):
        errors.append("Bundle manifest plan digest does not match plan.json.")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        errors.append("Bundle manifest files must be a non-empty list.")
        return tuple(errors)
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            errors.append("Every bundle manifest entry requires a path.")
            continue
        relative = item["path"]
        relative_path = Path(relative)
        if (
            relative in seen
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            errors.append(f"Unsafe or duplicate bundle manifest path: {relative}.")
            continue
        seen.add(relative)
        path = root / relative_path
        if path.is_symlink():
            errors.append(f"Manifested file cannot be a symlink: {relative}.")
            continue
        if not path.is_file():
            errors.append(f"Manifested file is missing: {relative}.")
            continue
        if (
            item.get("sha256") != sha256_file(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            errors.append(f"Manifested file changed: {relative}.")
    mutable_files = {
        ".validation-report.lock",
        "validation-report.json",
        "preflight-metrics.json",
    }
    mutable_prefixes = ("pilot-output/", "runs/")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "bundle-manifest.json"
    }
    unexpected = sorted(
        relative
        for relative in actual - seen
        if relative not in mutable_files and not relative.startswith(mutable_prefixes)
    )
    if unexpected:
        errors.append(
            "Bundle contains unmanifested input file(s): " + ", ".join(unexpected) + "."
        )
    return tuple(errors)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _content_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:20]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _select(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    source = _mapping(value)
    return {key: source.get(key) for key in keys}


def _normalized_model(value: Any) -> dict[str, Any]:
    return _select(
        value,
        (
            "model_id",
            "revision",
            "family",
            "parameters",
            "hidden_size",
            "intermediate_size",
            "layers",
            "context_length",
            "architecture",
            "tokenizer_id",
            "license_name",
            "training_allowed",
        ),
    )


def _normalized_dataset(value: Any) -> dict[str, Any]:
    dataset = _select(
        value,
        (
            "source_sha256",
            "source_format",
            "schema_name",
            "example_count",
            "total_estimated_tokens",
            "sequence_p50",
            "sequence_p95",
            "sequence_max",
            "measurement",
            "sampled_examples",
            "sample_indices",
            "duplicate_count",
            "empty_count",
            "truncation_count",
            "truncation_rate",
            "source_size_bytes",
            "canonical_size_bytes",
            "max_canonical_row_bytes",
        ),
    )
    dataset["schema_counts"] = dict(
        sorted(_mapping(_mapping(value).get("schema_counts")).items())
    )
    return dataset


def _normalized_hardware(value: Any) -> dict[str, Any]:
    hardware = _mapping(value)
    devices = hardware.get("devices")
    normalized_devices = (
        [
            _select(
                item,
                (
                    "name",
                    "backend",
                    "total_vram_bytes",
                    "free_vram_bytes",
                    "supports_bf16",
                    "supports_4bit",
                    "supports_8bit",
                    "compute_capability",
                    "driver_version",
                ),
            )
            for item in devices
        ]
        if isinstance(devices, (list, tuple))
        else []
    )
    return {
        "devices": normalized_devices,
        **_select(
            hardware,
            (
                "host_ram_bytes",
                "host_ram_free_bytes",
                "reserve_per_device_bytes",
                "disk_free_bytes",
                "cuda_version",
                "interconnect",
            ),
        ),
    }


def _normalized_target(value: Any) -> dict[str, Any]:
    return _select(
        value,
        (
            "objective",
            "sequence_length",
            "effective_batch_size",
            "max_epochs",
            "method_preference",
            "task",
            "evaluation_fraction",
            "packing",
            "checkpoint_steps",
            "max_wall_time_minutes",
            "training_runtime",
        ),
    )


def _normalized_memory(value: Any) -> dict[str, Any]:
    memory = _select(
        value,
        (
            "base_weights_bytes",
            "quantization_metadata_bytes",
            "adapter_weights_bytes",
            "adapter_gradients_bytes",
            "optimizer_states_bytes",
            "activations_bytes",
            "temporary_overhead_bytes",
            "safety_margin_bytes",
            "communication_bytes",
            "workspace_bytes",
            "allocator_bytes",
            "load_transient_bytes",
            "point_estimate_bytes",
            "estimated_peak_bytes",
            "upper_estimate_bytes",
            "uncertainty_bytes",
            "formula_version",
        ),
    )
    memory["component_upper_bounds"] = dict(
        sorted(_mapping(_mapping(value).get("component_upper_bounds")).items())
    )
    return memory


def candidate_id_for_payload(
    candidate: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    hardware: Mapping[str, Any],
    target: Mapping[str, Any],
) -> str:
    """Derive the portable content ID for an executable candidate contract."""

    target_modules = candidate.get("target_modules")
    runtime_contract = candidate.get("runtime_contract")
    identity = {
        "strategy": {
            **_select(
                candidate,
                (
                    "method",
                    "distribution",
                    "precision",
                    "quantization",
                    "micro_batch_size",
                    "gradient_accumulation_steps",
                    "effective_batch_size",
                    "world_size",
                    "device_indices",
                    "rank",
                    "alpha",
                    "learning_rate",
                    "user_reserve_bytes",
                    "required_host_ram_bytes",
                    "required_disk_bytes",
                    "checkpoint_retention_bytes",
                    "final_export_bytes",
                    "status",
                    "feasible",
                ),
            ),
            "target_modules": sorted(target_modules)
            if isinstance(target_modules, (list, tuple))
            else [],
            "memory": _normalized_memory(candidate.get("memory")),
            **(
                {
                    "runtime_contract": _select(
                        runtime_contract,
                        (
                            "schema_version",
                            "compute_backend",
                            "training_runtime",
                            "compiler_id",
                            "estimator_id",
                            "evidence_requirement",
                            "export_kind",
                        ),
                    )
                }
                if isinstance(runtime_contract, Mapping)
                else {}
            ),
        },
        "facts": {
            "model": _normalized_model(model),
            "dataset": _normalized_dataset(dataset),
            "hardware": _normalized_hardware(hardware),
            "target": _normalized_target(target),
        },
    }
    return _content_id("cand_", identity)


def plan_id_for_payload(plan: Mapping[str, Any]) -> str:
    """Derive the plan ID from normalized facts, candidates, and recommendation."""

    candidates = plan.get("candidates")
    candidate_ids = (
        [
            item.get("candidate_id") if isinstance(item, Mapping) else None
            for item in candidates
        ]
        if isinstance(candidates, (list, tuple))
        else []
    )
    recommended = _mapping(plan.get("recommended"))
    identity = {
        "schema_version": plan.get("schema_version"),
        "formula_version": plan.get("formula_version"),
        "facts": {
            "model": _normalized_model(plan.get("model")),
            "dataset": _normalized_dataset(plan.get("dataset")),
            "hardware": _normalized_hardware(plan.get("hardware")),
            "target": _normalized_target(plan.get("target")),
        },
        "candidate_ids": candidate_ids,
        "recommended_candidate_id": recommended.get("candidate_id"),
    }
    return _content_id("plan_", identity)


def validate_plan_payload(
    plan_value: Any,
    *,
    root: Path | None = None,
    verify_dataset: bool = True,
) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(plan_value, dict) or not plan_value:
        return ("Plan must be a non-empty object.",)
    plan = plan_value
    if _contains_nonfinite(plan):
        errors.append("Plan numbers must be finite JSON values.")
    if plan.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Plan schema_version must be {SCHEMA_VERSION}.")
    if plan.get("formula_version") != FORMULA_VERSION:
        errors.append(f"Plan formula_version must be {FORMULA_VERSION}.")
    for key in (
        "model",
        "dataset",
        "hardware",
        "target",
        "recommended",
        "candidates",
        "evidence_records",
    ):
        if key not in plan:
            errors.append(f"Plan requires {key}.")

    model = plan.get("model") if isinstance(plan.get("model"), dict) else {}
    dataset = plan.get("dataset") if isinstance(plan.get("dataset"), dict) else {}
    hardware = plan.get("hardware") if isinstance(plan.get("hardware"), dict) else {}
    target = plan.get("target") if isinstance(plan.get("target"), dict) else {}
    for key in ("model_id", "revision", "family", "license_name"):
        if not isinstance(model.get(key), str) or not model.get(key, "").strip():
            errors.append(f"Model {key} is required.")
    revision = model.get("revision")
    if (
        not isinstance(revision, str)
        or not (40 <= len(revision) <= 64)
        or any(c not in "0123456789abcdefABCDEF" for c in revision)
    ):
        errors.append("Model revision must be an immutable hexadecimal commit ID.")
    if model.get("training_allowed") is not True:
        errors.append("Model training permission must be explicitly true.")
    for key in ("parameters", "hidden_size", "layers", "context_length"):
        if not _positive_int(model.get(key)):
            errors.append(f"Model {key} must be positive.")
    if model.get("intermediate_size") is not None and not _positive_int(
        model.get("intermediate_size")
    ):
        errors.append("Model intermediate_size must be positive when supplied.")

    if dataset.get("schema_name") not in {
        "text",
        "prompt-completion",
        "instruction-output",
        "messages",
        "mixed",
    }:
        errors.append("Dataset schema is unsupported.")
    if (
        not isinstance(dataset.get("source_path"), str)
        or not dataset.get("source_path", "").strip()
    ):
        errors.append("Dataset source_path is required.")
    if dataset.get("source_format") not in {"jsonl", "json", "csv", "txt"}:
        errors.append("Dataset source_format is unsupported.")
    for key in (
        "example_count",
        "total_estimated_tokens",
        "sequence_p50",
        "sequence_p95",
        "sequence_max",
        "sampled_examples",
    ):
        if not _positive_int(dataset.get(key)):
            errors.append(f"Dataset {key} must be positive.")
    if all(
        _positive_int(dataset.get(key))
        for key in ("sequence_p50", "sequence_p95", "sequence_max")
    ) and not (
        dataset["sequence_p50"] <= dataset["sequence_p95"] <= dataset["sequence_max"]
    ):
        errors.append("Dataset sequence percentiles must be ordered.")
    digest = dataset.get("source_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in digest)
    ):
        errors.append("Dataset source_sha256 is invalid.")
    for key in (
        "source_size_bytes",
        "canonical_size_bytes",
        "max_canonical_row_bytes",
    ):
        if not _positive_int(dataset.get(key)):
            errors.append(f"Dataset {key} must be positive.")
    if verify_dataset and isinstance(dataset.get("source_path"), str):
        source = Path(dataset["source_path"])
        if root is not None:
            root = root.resolve()
            if source.is_absolute():
                errors.append("Bundled dataset source_path must be relative.")
            else:
                unresolved = root / source
                if unresolved.is_symlink():
                    errors.append("Bundled dataset source_path cannot be a symlink.")
                try:
                    resolved = unresolved.resolve(strict=True)
                except FileNotFoundError:
                    resolved = unresolved.resolve()
                if resolved != root and root not in resolved.parents:
                    errors.append(
                        "Bundled dataset source_path escapes the bundle root."
                    )
                source = resolved
        if not source.is_file():
            errors.append("Dataset source file is unavailable.")
        elif isinstance(digest, str) and sha256_file(source) != digest:
            errors.append("Dataset source hash does not match the plan.")

    devices = hardware.get("devices")
    if not isinstance(devices, list) or not devices:
        errors.append("Hardware requires at least one device.")
    elif any(
        not isinstance(item, dict) or item.get("backend") not in {"cuda", "mps"}
        for item in devices
    ):
        errors.append("Aptus execution plans support CUDA or MPS compute devices.")
    if isinstance(devices, list):
        for index, device in enumerate(devices):
            if not isinstance(device, dict):
                continue
            for capability in ("supports_bf16", "supports_8bit", "supports_4bit"):
                if not isinstance(device.get(capability), bool):
                    errors.append(
                        f"Hardware device {index} {capability} must be boolean."
                    )
            total_vram = device.get("total_vram_bytes")
            free_vram = device.get("free_vram_bytes")
            if not _positive_int(total_vram):
                errors.append(
                    f"Hardware device {index} total_vram_bytes must be positive."
                )
            if free_vram is not None and (
                not _positive_int(free_vram)
                or (_positive_int(total_vram) and free_vram > total_vram)
            ):
                errors.append(
                    f"Hardware device {index} free_vram_bytes must be positive and no greater than total VRAM."
                )
    reserve = hardware.get("reserve_per_device_bytes")
    if not isinstance(reserve, int) or isinstance(reserve, bool) or reserve < 0:
        errors.append(
            "Hardware reserve_per_device_bytes must be a non-negative integer."
        )
    if not _positive_int(hardware.get("host_ram_bytes")):
        errors.append("Hardware host_ram_bytes must be positive.")
    host_free = hardware.get("host_ram_free_bytes")
    if host_free is not None and (
        not _positive_int(host_free)
        or (
            _positive_int(hardware.get("host_ram_bytes"))
            and host_free > hardware["host_ram_bytes"]
        )
    ):
        errors.append(
            "Hardware host_ram_free_bytes must be positive and no greater than host RAM."
        )
    if hardware.get("disk_free_bytes") is not None and not _positive_int(
        hardware.get("disk_free_bytes")
    ):
        errors.append("Hardware disk_free_bytes must be positive when supplied.")
    if (
        isinstance(devices, list)
        and isinstance(reserve, int)
        and not isinstance(reserve, bool)
        and reserve >= 0
    ):
        for index, device in enumerate(devices):
            if (
                isinstance(device, dict)
                and _positive_int(device.get("total_vram_bytes"))
                and reserve >= device["total_vram_bytes"]
            ):
                errors.append(
                    f"Hardware reserve must be smaller than device {index} total VRAM."
                )
    if not _positive_int(target.get("sequence_length")) or not _positive_int(
        target.get("effective_batch_size")
    ):
        errors.append("Target sequence length and effective batch must be positive.")
    if not _positive_int(target.get("max_epochs")):
        errors.append("Target max_epochs must be positive.")
    if not _positive_int(target.get("checkpoint_steps")):
        errors.append("Target checkpoint_steps must be positive.")
    if target.get("max_wall_time_minutes") is not None and not _positive_int(
        target.get("max_wall_time_minutes")
    ):
        errors.append("Target max_wall_time_minutes must be positive when supplied.")
    elif target.get("max_wall_time_minutes") is not None:
        errors.append(
            "Target max_wall_time_minutes must be null in Aptus v0.2 because execution does not enforce it."
        )
    if target.get("task") != "sft":
        errors.append("Aptus v0.2 target task must be sft.")
    if target.get("objective") not in {"quality", "memory", "speed"}:
        errors.append("Target objective is invalid.")
    if target.get("method_preference") not in METHODS | {None}:
        errors.append("Target method_preference is invalid.")
    if target.get("training_runtime") not in TRAINING_RUNTIMES | {None}:
        errors.append("Target training_runtime is invalid.")
    if target.get("packing") is not False:
        errors.append("Aptus v0.2 target packing must be false.")
    evaluation_fraction = target.get("evaluation_fraction")
    if (
        not isinstance(evaluation_fraction, (int, float))
        or isinstance(evaluation_fraction, bool)
        or not math.isfinite(evaluation_fraction)
        or not 0 <= evaluation_fraction < 1
    ):
        errors.append("Target evaluation_fraction must be in [0, 1).")
    if _positive_int(model.get("context_length")) and _positive_int(
        target.get("sequence_length")
    ):
        if target["sequence_length"] > model["context_length"]:
            errors.append("Target sequence length exceeds model context length.")

    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("Plan candidates must be a non-empty list.")
        candidates = []
    candidate_ids: set[str] = set()
    candidate_by_id: dict[str, dict[str, Any]] = {}
    strategy_pairs: set[tuple[str, str]] = set()
    evidence_records = plan.get("evidence_records")
    if not isinstance(evidence_records, list):
        errors.append("Plan evidence_records must be a list.")
        evidence_records = []
    evidence_ids: set[str] = set()
    required_evidence_fields = (
        "evidence_id",
        "claim",
        "source",
        "source_kind",
        "scope",
        "confidence",
    )
    for index, record in enumerate(evidence_records):
        name = f"Evidence record {index}"
        if not isinstance(record, dict):
            errors.append(f"{name} must be an object.")
            continue
        for field in required_evidence_fields:
            if not isinstance(record.get(field), str) or not record[field].strip():
                errors.append(f"{name} requires non-empty string {field}.")
        evidence_id = record.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id:
            if evidence_id in evidence_ids:
                errors.append(f"Duplicate evidence ID: {evidence_id}.")
            else:
                evidence_ids.add(evidence_id)
        revision = record.get("revision")
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            errors.append(f"{name} revision must be null or a non-empty string.")
    for index, candidate in enumerate(candidates):
        name = f"Candidate {index}"
        if not isinstance(candidate, dict):
            errors.append(f"{name} must be an object.")
            continue
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            errors.append(f"{name} requires candidate_id.")
        elif candidate_id in candidate_ids:
            errors.append(f"Duplicate candidate ID: {candidate_id}.")
        else:
            candidate_ids.add(candidate_id)
            candidate_by_id[candidate_id] = candidate
        candidate_method = candidate.get("method")
        if candidate_method not in METHODS:
            errors.append(f"{name} method is invalid.")
        runtime_contract = candidate.get("runtime_contract")
        runtime_id = "transformers-peft-cuda"
        runtime_backend = "cuda"
        runtime_estimator = FORMULA_VERSION
        if not isinstance(runtime_contract, dict):
            errors.append(f"{name} runtime_contract must be an object.")
        elif isinstance(runtime_contract, dict):
            runtime_id = runtime_contract.get("training_runtime")
            runtime_backend = runtime_contract.get("compute_backend")
            runtime_estimator = runtime_contract.get("estimator_id")
            if runtime_contract.get("schema_version") != RUNTIME_CONTRACT_VERSION:
                errors.append(
                    f"{name} runtime contract schema must be {RUNTIME_CONTRACT_VERSION}."
                )
            if runtime_id not in TRAINING_RUNTIMES:
                errors.append(f"{name} training runtime is invalid.")
            if runtime_backend not in {"cuda", "mps"}:
                errors.append(f"{name} runtime compute backend is invalid.")
            expected_runtime_backend = {
                "transformers-peft-cuda": "cuda",
                "mlx-lm": "mps",
                "pytorch-mps": "mps",
            }.get(runtime_id)
            if expected_runtime_backend and runtime_backend != expected_runtime_backend:
                errors.append(f"{name} runtime and compute backend do not match.")
            if (
                runtime_contract.get("evidence_requirement")
                not in EVIDENCE_REQUIREMENTS
            ):
                errors.append(f"{name} runtime evidence requirement is invalid.")
            expected_runtime_identity = RUNTIME_BINDING_IDENTITIES.get(
                (candidate_method, runtime_id, runtime_backend)
            )
            actual_runtime_identity = (
                runtime_contract.get("compiler_id"),
                runtime_contract.get("estimator_id"),
                runtime_contract.get("export_kind"),
                runtime_contract.get("evidence_requirement"),
            )
            if expected_runtime_identity is None:
                if actual_runtime_identity != UNAVAILABLE_RUNTIME_IDENTITY:
                    errors.append(
                        f"{name} unregistered method/runtime/backend contract must use the exact unavailable identity."
                    )
            elif actual_runtime_identity != expected_runtime_identity:
                errors.append(
                    f"{name} runtime contract does not match its registered compiler, estimator, export, and evidence identity."
                )
            viable_runtime = candidate.get("status") in {
                "feasible",
                "conditional",
            }
            if viable_runtime:
                if expected_runtime_identity is None:
                    errors.append(
                        f"{name} viable runtime requires a registered method/runtime/backend compiler binding."
                    )
                elif runtime_contract.get("evidence_requirement") != "pilot-required":
                    errors.append(f"{name} viable runtime must remain pilot-required.")
        if candidate.get("precision") not in {"bf16", "fp16"}:
            errors.append(f"{name} precision is invalid.")
        learning_rate = candidate.get("learning_rate")
        if not _finite_number(learning_rate) or learning_rate <= 0:
            errors.append(f"{name} learning_rate must be positive and finite.")
        if (
            candidate.get("method") == "full"
            and candidate.get("precision") == "fp16"
            and candidate.get("status") in {"feasible", "conditional"}
        ):
            errors.append(
                f"{name} full-parameter FP16 execution is unsupported in Aptus v0.2."
            )
        if candidate.get("distribution") not in DISTRIBUTIONS:
            errors.append(f"{name} distribution is invalid.")
        elif candidate.get("method") in METHODS:
            strategy_pairs.add((candidate["method"], candidate["distribution"]))
        if candidate.get("status") not in CANDIDATE_STATUSES:
            errors.append(f"{name} status is invalid.")
        elif candidate.get("feasible") is not (
            candidate["status"] in {"feasible", "conditional"}
        ):
            errors.append(f"{name} feasible flag does not match status.")
        for key in (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
            "world_size",
        ):
            if not _positive_int(candidate.get(key)):
                errors.append(f"{name} {key} must be positive.")
        device_indices = candidate.get("device_indices")
        selected_devices: list[dict[str, Any]] = []
        if (
            not isinstance(device_indices, list)
            or len(device_indices) != candidate.get("world_size")
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in device_indices
            )
            or len(set(device_indices)) != len(device_indices)
        ):
            errors.append(
                f"{name} device_indices must contain one unique non-negative integer per world rank."
            )
        elif not isinstance(devices, list) or any(
            item >= len(devices) for item in device_indices
        ):
            errors.append(f"{name} device_indices reference unavailable hardware.")
        elif not all(isinstance(devices[item], dict) for item in device_indices):
            errors.append(f"{name} device_indices reference invalid hardware facts.")
        else:
            selected_devices = [devices[item] for item in device_indices]
            selected_backends = {item.get("backend") for item in selected_devices}
            if len(selected_backends) != 1:
                errors.append(f"{name} cannot mix compute backends.")
            elif runtime_backend not in selected_backends:
                errors.append(
                    f"{name} runtime compute backend does not match selected hardware."
                )
        for key in (
            "required_host_ram_bytes",
            "required_disk_bytes",
            "checkpoint_retention_bytes",
            "final_export_bytes",
        ):
            if not _positive_int(candidate.get(key)):
                errors.append(f"{name} {key} must be positive.")
        if all(
            _positive_int(candidate.get(key))
            for key in ("micro_batch_size", "gradient_accumulation_steps", "world_size")
        ):
            calculated = (
                candidate["micro_batch_size"]
                * candidate["gradient_accumulation_steps"]
                * candidate["world_size"]
            )
            if calculated != candidate.get(
                "effective_batch_size"
            ) or calculated != target.get("effective_batch_size"):
                errors.append(f"{name} global batch arithmetic is invalid.")
            expected_world = (
                1
                if candidate.get("distribution") == "single"
                else len(devices)
                if isinstance(devices, list)
                else 0
            )
            if candidate["world_size"] != expected_world:
                errors.append(
                    f"{name} world_size does not match its distribution and hardware."
                )
        memory = candidate.get("memory")
        if not isinstance(memory, dict):
            errors.append(f"{name} memory must be an object.")
        else:
            component_names = (
                "base_weights_bytes",
                "quantization_metadata_bytes",
                "adapter_weights_bytes",
                "adapter_gradients_bytes",
                "optimizer_states_bytes",
                "activations_bytes",
                "temporary_overhead_bytes",
                "communication_bytes",
                "workspace_bytes",
                "allocator_bytes",
                "load_transient_bytes",
            )
            if all(
                isinstance(memory.get(key), int) and memory[key] >= 0
                for key in component_names
            ):
                point = sum(memory[key] for key in component_names)
                if (
                    memory.get("point_estimate_bytes") != point
                    or memory.get("estimated_peak_bytes") != point
                ):
                    errors.append(f"{name} point memory does not equal its components.")
                bounds = memory.get("component_upper_bounds")
                if not isinstance(bounds, dict) or not bounds:
                    errors.append(
                        f"{name} requires transparent component_upper_bounds."
                    )
                elif not all(
                    isinstance(value, int) and value >= 0 for value in bounds.values()
                ):
                    errors.append(
                        f"{name} upper memory components must be non-negative integers."
                    )
                else:
                    upper = sum(bounds.values())
                    if memory.get("upper_estimate_bytes") != upper or upper < point:
                        errors.append(
                            f"{name} upper memory must equal its component upper bounds and cover the point estimate."
                        )
                    if bounds.get("uncertainty_bytes") != memory.get(
                        "safety_margin_bytes"
                    ):
                        errors.append(
                            f"{name} uncertainty_bytes must equal the named safety margin."
                        )
                if memory.get("uncertainty_bytes") != memory.get("safety_margin_bytes"):
                    errors.append(
                        f"{name} uncertainty alias must equal the named safety margin."
                    )
                expected_memory_formula = (
                    MLX_FORMULA_VERSION
                    if runtime_id == "mlx-lm"
                    and candidate.get("method") in {"lora", "qlora"}
                    else FORMULA_VERSION
                )
                if memory.get("formula_version") != expected_memory_formula:
                    errors.append(
                        f"{name} memory formula must be {expected_memory_formula}."
                    )
                if (
                    candidate.get("status") in {"feasible", "conditional"}
                    and runtime_estimator != expected_memory_formula
                ):
                    errors.append(
                        f"{name} runtime estimator does not match its memory formula."
                    )
                if (
                    candidate.get("status") in {"feasible", "conditional"}
                    and selected_devices
                    and isinstance(reserve, int)
                ):
                    capacities = [
                        item.get("free_vram_bytes") or item.get("total_vram_bytes", 0)
                        for item in selected_devices
                    ]
                    if runtime_backend == "mps" and _positive_int(host_free):
                        capacities = [
                            min(capacity, host_free) for capacity in capacities
                        ]
                    usable = min(capacity - reserve for capacity in capacities)
                    if point > usable:
                        errors.append(
                            f"{name} viable status exceeds usable per-device memory at its point estimate."
                        )
                    if (
                        candidate.get("status") == "feasible"
                        and memory.get("upper_estimate_bytes", 0) > usable
                    ):
                        errors.append(
                            f"{name} feasible status exceeds usable per-device memory at its heuristic upper envelope."
                        )
            else:
                errors.append(
                    f"{name} memory components must be non-negative integers."
                )
        method = candidate.get("method")
        quantization = candidate.get("quantization")
        expected_quantization = (
            "mlx-4bit-groupwise"
            if runtime_id == "mlx-lm" and method == "qlora"
            else {
                "full": None,
                "lora": None,
                "int8-lora": "int8-bitsandbytes",
                "qlora": "nf4-double-quant",
            }.get(method)
        )
        if method in METHODS and quantization != expected_quantization:
            errors.append(f"{name} quantization does not match method.")
        if method == "full" and (
            candidate.get("rank") != 0
            or candidate.get("alpha") != 0
            or candidate.get("target_modules") not in ([], ())
        ):
            errors.append(f"{name} full fine-tuning cannot carry adapter fields.")
        if (
            method != "full"
            and method in METHODS
            and (
                not _positive_int(candidate.get("rank"))
                or not _positive_int(candidate.get("alpha"))
                or not candidate.get("target_modules")
            )
        ):
            errors.append(
                f"{name} adapter method requires rank, alpha, and target modules."
            )
        if candidate.get("status") in {"feasible", "conditional"} and selected_devices:
            if candidate.get("precision") == "bf16" and any(
                not item.get("supports_bf16") for item in selected_devices
            ):
                errors.append(f"{name} uses bf16 without device support.")
            if (
                runtime_id != "mlx-lm"
                and method == "qlora"
                and any(not item.get("supports_4bit") for item in selected_devices)
            ):
                errors.append(
                    f"{name} uses four-bit quantization without device support."
                )
            if method == "int8-lora" and any(
                not item.get("supports_8bit") for item in selected_devices
            ):
                errors.append(
                    f"{name} uses eight-bit quantization without device support."
                )
            if candidate.get("distribution") == "fsdp" and method in {
                "int8-lora",
                "qlora",
            }:
                errors.append(f"{name} uses an unsupported quantized FSDP combination.")
            if runtime_id == "mlx-lm":
                if method not in {"lora", "qlora"}:
                    errors.append(f"{name} MLX-LM method is unsupported.")
                if candidate.get("distribution") != "single":
                    errors.append(f"{name} MLX-LM distribution must be single.")
                if candidate.get("status") != "conditional":
                    errors.append(
                        f"{name} MLX-LM status must remain conditional until pilot evidence."
                    )
                if runtime_contract and "bitsandbytes" in json.dumps(
                    runtime_contract, sort_keys=True
                ):
                    errors.append(
                        f"{name} MLX-LM contract cannot use bitsandbytes identity."
                    )
        if isinstance(candidate_id, str) and candidate_id != candidate_id_for_payload(
            candidate,
            model=model,
            dataset=dataset,
            hardware=hardware,
            target=target,
        ):
            errors.append(
                f"{name} immutable candidate ID does not match its normalized execution contract."
            )
        candidate_evidence = candidate.get("evidence")
        if not isinstance(candidate_evidence, list) or any(
            not isinstance(item, str) or not item for item in candidate_evidence or ()
        ):
            errors.append(f"{name} evidence must be a list of non-empty IDs.")
        else:
            if len(candidate_evidence) != len(set(candidate_evidence)):
                errors.append(f"{name} evidence IDs must be unique.")
            for evidence_id in candidate_evidence:
                if evidence_id not in evidence_ids:
                    errors.append(
                        f"{name} references missing evidence ID {evidence_id}."
                    )

    expected_pairs = {
        (method, distribution) for method in METHODS for distribution in DISTRIBUTIONS
    }
    if strategy_pairs != expected_pairs or len(candidates) != len(expected_pairs):
        errors.append(
            "Plan must contain exactly one candidate for every method and distribution pair."
        )

    recommended = plan.get("recommended")
    if not isinstance(recommended, dict):
        errors.append("Recommended candidate must be an object.")
    else:
        recommended_id = recommended.get("candidate_id")
        listed = candidate_by_id.get(recommended_id)
        if listed is None:
            errors.append("Recommended candidate must appear in candidates.")
        elif listed != recommended:
            errors.append(
                "Recommended candidate must exactly match its listed candidate."
            )
        if recommended.get("status") not in {"feasible", "conditional"}:
            errors.append("Recommended candidate must be feasible or conditional.")
    if plan.get("plan_id") != plan_id_for_payload(plan):
        errors.append(
            "Plan immutable ID does not match its normalized facts, candidates, and recommendation."
        )
    return tuple(errors)
