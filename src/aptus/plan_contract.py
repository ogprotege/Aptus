from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")
SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
FAMILY_TARGET_MODULES = {
    "llama": {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    },
    "mistral": {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    },
    "gemma": {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    },
    "qwen": {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    },
}
BUNDLE_FINGERPRINT_FILES = (
    "README.md",
    "plan.json",
    "plan_contract.py",
    "requirements.txt",
    "train.py",
    "validate.py",
)


def bundle_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for filename in BUNDLE_FINGERPRINT_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Bundle fingerprint input is missing: {filename}")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _mapping(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        errors.append(f"{name} must be a non-empty object.")
        return {}
    return value


def _validate_candidate(
    candidate_value: Any,
    *,
    name: str,
    target: dict[str, Any],
    errors: list[str],
    require_feasible: bool,
) -> None:
    candidate = _mapping(candidate_value, name, errors)
    if not candidate:
        return

    method = candidate.get("method")
    if method not in {"lora", "qlora"}:
        errors.append(f"{name} method must be lora or qlora.")
    if require_feasible and candidate.get("feasible") is not True:
        errors.append("Recommended candidate must be feasible.")
    if candidate.get("precision") not in {"bf16", "fp16"}:
        errors.append(f"{name} precision must be bf16 or fp16.")
    if method == "qlora" and candidate.get("quantization") != "nf4-double-quant":
        errors.append(f"{name} QLoRA quantization contract is invalid.")
    if method == "lora" and candidate.get("quantization") is not None:
        errors.append(f"{name} LoRA candidate must not declare quantization.")

    for key, label in (
        ("micro_batch_size", "micro batch"),
        ("gradient_accumulation_steps", "gradient accumulation"),
        ("effective_batch_size", "effective batch"),
        ("rank", "rank"),
        ("alpha", "alpha"),
    ):
        if not _is_positive_int(candidate.get(key)):
            errors.append(f"{name} {label} must be a positive integer.")
    learning_rate = candidate.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        errors.append(f"{name} learning rate must be positive.")
    modules = candidate.get("target_modules")
    if not isinstance(modules, list) or not modules or not all(
        isinstance(module, str) and module.strip() for module in modules
    ):
        errors.append(f"{name} target modules must be non-empty strings.")

    micro = candidate.get("micro_batch_size")
    accumulation = candidate.get("gradient_accumulation_steps")
    effective = candidate.get("effective_batch_size")
    requested_effective = target.get("effective_batch_size")
    if all(_is_positive_int(value) for value in (micro, accumulation, effective)):
        if micro * accumulation != effective:
            errors.append(
                f"{name} effective batch must equal micro batch times "
                "gradient accumulation."
            )
    if (
        _is_positive_int(effective)
        and _is_positive_int(requested_effective)
        and effective != requested_effective
    ):
        errors.append(
            f"{name} effective batch must preserve the requested effective batch."
        )

    memory = _mapping(candidate.get("memory"), f"{name} memory", errors)
    if memory:
        component_keys = (
            "base_weights_bytes",
            "quantization_metadata_bytes",
            "adapter_weights_bytes",
            "adapter_gradients_bytes",
            "optimizer_states_bytes",
            "activations_bytes",
            "temporary_overhead_bytes",
            "safety_margin_bytes",
        )
        for key in component_keys:
            if not _is_non_negative_int(memory.get(key)):
                errors.append(f"{name} memory {key} must be non-negative.")
        if all(_is_non_negative_int(memory.get(key)) for key in component_keys):
            expected_peak = sum(memory[key] for key in component_keys)
            if memory.get("estimated_peak_bytes") != expected_peak:
                errors.append(
                    f"{name} memory estimated peak does not equal its components."
                )


def _validate_candidate_feasibility(
    candidate_value: Any,
    *,
    name: str,
    model: dict[str, Any],
    hardware: dict[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(candidate_value, dict) or candidate_value.get("feasible") is not True:
        return

    devices = hardware.get("devices")
    reserve = hardware.get("reserve_per_device_bytes")
    if (
        isinstance(devices, list)
        and devices
        and _is_non_negative_int(reserve)
        and all(
            isinstance(device, dict)
            and _is_positive_int(device.get("total_vram_bytes"))
            and reserve < device["total_vram_bytes"]
            for device in devices
        )
    ):
        usable_vram = min(
            device["total_vram_bytes"] - reserve for device in devices
        )
        peak = candidate_value.get("memory", {}).get("estimated_peak_bytes")
        if _is_non_negative_int(peak) and peak > usable_vram:
            errors.append(
                f"{name} estimated peak exceeds usable per-device VRAM."
            )

    if candidate_value.get("method") == "qlora" and isinstance(devices, list):
        if not devices or any(
            not isinstance(device, dict)
            or device.get("supports_4bit") is not True
            for device in devices
        ):
            errors.append(f"{name} QLoRA requires 4-bit support on every device.")
    if candidate_value.get("precision") == "bf16" and isinstance(devices, list):
        if not devices or any(
            not isinstance(device, dict)
            or device.get("supports_bf16") is not True
            for device in devices
        ):
            errors.append(f"{name} BF16 requires BF16 support on every device.")

    family = model.get("family")
    modules = candidate_value.get("target_modules")
    allowed_modules = FAMILY_TARGET_MODULES.get(family)
    if allowed_modules is None:
        errors.append(f"{name} model family has no target-module contract.")
    elif isinstance(modules, list):
        unsupported = sorted(set(modules) - allowed_modules)
        if unsupported:
            errors.append(
                f"{name} target module(s) are incompatible with {family}: "
                + ", ".join(unsupported)
            )


def validate_plan_payload(
    plan_value: Any,
    *,
    verify_dataset: bool = True,
) -> tuple[str, ...]:
    errors: list[str] = []
    plan = _mapping(plan_value, "Plan", errors)
    if not plan:
        return tuple(errors)
    if plan.get("schema_version") != "aptus.training-plan.v1":
        errors.append("Plan schema version must be aptus.training-plan.v1.")

    model = _mapping(plan.get("model"), "Model", errors)
    dataset = _mapping(plan.get("dataset"), "Dataset", errors)
    hardware = _mapping(plan.get("hardware"), "Hardware", errors)
    target = _mapping(plan.get("target"), "Target", errors)

    if model:
        for key in ("model_id", "family"):
            if not isinstance(model.get(key), str) or not model[key].strip():
                errors.append(f"Model {key} is required.")
        revision = model.get("revision")
        if not isinstance(revision, str) or not IMMUTABLE_REVISION.fullmatch(
            revision
        ):
            errors.append(
                "Model immutable revision must be a 40-64 character hexadecimal "
                "commit identifier."
            )
        if not isinstance(model.get("license_name"), str) or not model[
            "license_name"
        ].strip():
            errors.append("Model license is required.")
        if model.get("training_allowed") is not True:
            errors.append("Model training permission must be explicitly true.")
        for key in (
            "parameters",
            "hidden_size",
            "layers",
            "context_length",
        ):
            if not _is_positive_int(model.get(key)):
                errors.append(f"Model {key} must be a positive integer.")

    if dataset:
        if dataset.get("schema_name") != "text":
            errors.append("Dataset schema must be text.")
        if not isinstance(dataset.get("source_path"), str) or not dataset[
            "source_path"
        ]:
            errors.append("Dataset source path is required.")
        digest = dataset.get("source_sha256")
        if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
            errors.append("Dataset source SHA-256 is invalid.")
        for key in (
            "example_count",
            "total_estimated_tokens",
            "sequence_p50",
            "sequence_p95",
            "sequence_max",
        ):
            if not _is_positive_int(dataset.get(key)):
                errors.append(f"Dataset {key} must be a positive integer.")
        sequence_values = (
            dataset.get("sequence_p50"),
            dataset.get("sequence_p95"),
            dataset.get("sequence_max"),
        )
        if all(_is_positive_int(value) for value in sequence_values) and not (
            sequence_values[0] <= sequence_values[1] <= sequence_values[2]
        ):
            errors.append("Dataset sequence percentiles must be ordered.")
        if verify_dataset and isinstance(dataset.get("source_path"), str):
            source = Path(dataset["source_path"])
            if not source.is_file():
                errors.append("Dataset source file is unavailable.")
            elif isinstance(digest, str) and hashlib.sha256(
                source.read_bytes()
            ).hexdigest() != digest:
                errors.append("Dataset source hash does not match the plan.")

    if hardware:
        devices = hardware.get("devices")
        if not isinstance(devices, list) or not devices:
            errors.append("Hardware devices must contain at least one device.")
            devices = []
        reserve = hardware.get("reserve_per_device_bytes")
        if not _is_non_negative_int(reserve):
            errors.append("Hardware reserve must be non-negative.")
        if not _is_positive_int(hardware.get("host_ram_bytes")):
            errors.append("Hardware host RAM must be positive.")
        for index, device_value in enumerate(devices):
            device = _mapping(device_value, f"Hardware device {index}", errors)
            if not device:
                continue
            if device.get("backend") != "cuda":
                errors.append(f"Hardware device {index} backend must be cuda.")
            total = device.get("total_vram_bytes")
            if not _is_positive_int(total):
                errors.append(
                    f"Hardware device {index} total VRAM must be positive."
                )
            if (
                _is_positive_int(total)
                and _is_non_negative_int(reserve)
                and reserve >= total
            ):
                errors.append(
                    f"Hardware device {index} reserve must be below total VRAM."
                )

    if target:
        if target.get("objective") not in {"quality", "memory", "speed"}:
            errors.append("Target objective is invalid.")
        for key, label in (
            ("sequence_length", "sequence length"),
            ("effective_batch_size", "effective batch"),
            ("max_epochs", "epochs"),
        ):
            if not _is_positive_int(target.get(key)):
                errors.append(f"Target {label} must be a positive integer.")
        if (
            model
            and _is_positive_int(model.get("context_length"))
            and _is_positive_int(target.get("sequence_length"))
            and target["sequence_length"] > model["context_length"]
        ):
            errors.append("Target sequence length exceeds model context length.")

    recommended = plan.get("recommended")
    _validate_candidate(
        recommended,
        name="Recommended",
        target=target,
        errors=errors,
        require_feasible=True,
    )
    _validate_candidate_feasibility(
        recommended,
        name="Recommended",
        model=model,
        hardware=hardware,
        errors=errors,
    )
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("Plan candidates must be a non-empty list.")
    else:
        for index, candidate in enumerate(candidates):
            _validate_candidate(
                candidate,
                name=f"Candidate {index}",
                target=target,
                errors=errors,
                require_feasible=False,
            )
            _validate_candidate_feasibility(
                candidate,
                name=f"Candidate {index}",
                model=model,
                hardware=hardware,
                errors=errors,
            )

    rationale = plan.get("recommendation_rationale")
    if not isinstance(rationale, list) or not rationale or not all(
        isinstance(item, str) and item.strip() for item in rationale
    ):
        errors.append("Recommendation rationale must be non-empty.")

    return tuple(errors)
