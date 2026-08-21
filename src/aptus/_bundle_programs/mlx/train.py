#!/usr/bin/env python3
"""Execute the MLX-LM compiler slice selected by an Aptus plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
from plan_contract import (
    expected_model_architecture_contract,
    load_json_object,
    mlx_quantized_storage_bytes_for_contract,
    mlx_packed_checkpoint_overhead_limit,
    mlx_trainable_target_instance_total,
    validate_bundle_manifest,
    validate_model_config_against_plan,
    validate_plan_payload,
)


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_json_object(ROOT / "plan.json", "Bundle plan")
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
        raise RuntimeError(
            "The selected candidate is not an executable MLX-LM contract."
        )
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
        raise RuntimeError(
            "MLX adapter output must remain under runs/ or pilot-output/."
        )
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def require_data(path: Path) -> Path:
    unresolved = path if path.is_absolute() else ROOT / path
    resolved = unresolved.resolve(strict=True)
    expected = (ROOT / "data" / "mlx").resolve(strict=True)
    if resolved != expected:
        raise RuntimeError(
            "The MLX data argument must match the compiler-bound data/mlx directory."
        )
    for name in ("train.jsonl", "valid.jsonl", "split-contract.json"):
        if not (resolved / name).is_file():
            raise RuntimeError(f"MLX dataset is missing {name}.")
    return resolved


def download_pinned_model(plan: dict[str, Any], requested_model: str) -> Path:
    model = plan["model"]
    if requested_model != model["model_id"]:
        raise RuntimeError(
            "The model argument must equal the plan-bound provider model ID."
        )
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=requested_model,
            revision=model["revision"],
        )
    ).resolve(strict=True)


def _descriptor_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def require_method_model(
    plan: dict[str, Any], candidate: dict[str, Any], model_path: Path
) -> dict[str, Any]:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_file"):
        raise RuntimeError(
            "MLX-LM custom model_file code is unsupported; Aptus only executes pinned built-in MLX model implementations."
        )
    try:
        architecture_contract = validate_model_config_against_plan(
            plan["model"], config
        )
    except ValueError as error:
        raise RuntimeError(
            "Pinned MLX-LM model architecture does not match the Aptus plan."
        ) from error
    quantization = config.get("quantization") or config.get("quantization_config")
    text_config = config.get("text_config")
    if not quantization and isinstance(text_config, dict):
        quantization = text_config.get("quantization_config")
    bits = quantization.get("bits") if isinstance(quantization, dict) else None
    planned_bits = plan["model"].get("quantization_bits")
    if candidate["method"] == "qlora" and (
        not isinstance(bits, int)
        or isinstance(bits, bool)
        or bits != planned_bits
        or not 1 <= bits <= 16
    ):
        raise RuntimeError(
            "MLX-LM QLoRA requires a pinned model revision whose declared "
            "quantization bits match the plan (1 through 16). Aptus will not "
            "substitute bitsandbytes or quantize an unbound model during training."
        )
    if candidate["method"] == "lora" and quantization:
        raise RuntimeError(
            "MLX-LM LoRA requires an unquantized pinned base model. A quantized base "
            "would execute QLoRA semantics under the wrong planned method."
        )
    return architecture_contract


_UNUSED_MULTIMODAL_TENSOR_PREFIXES = (
    "vision_tower",
    "multi_modal_projector",
    "audio_tower",
    "embed_audio",
    "embed_vision",
)


def _is_unused_multimodal_tensor(name: str) -> bool:
    stripped = name.removeprefix("model.")
    return stripped.startswith(_UNUSED_MULTIMODAL_TENSOR_PREFIXES) or name.startswith(
        _UNUSED_MULTIMODAL_TENSOR_PREFIXES
    )


def _safetensors_payload_accounting(path: Path) -> tuple[int, int]:
    """Return (total tensor payload bytes, unused multimodal payload bytes)."""

    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        if header_size <= 0 or header_size > 64 * 1024 * 1024:
            raise RuntimeError("Pinned MLX-LM safetensors header is invalid.")
        header = json.loads(handle.read(header_size).decode("utf-8"))
    if not isinstance(header, dict):
        raise RuntimeError("Pinned MLX-LM safetensors header is not an object.")
    payload = 0
    unused = 0
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        offsets = meta.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not isinstance(offsets[0], int)
            or not isinstance(offsets[1], int)
            or offsets[1] < offsets[0]
        ):
            raise RuntimeError("Pinned MLX-LM safetensors tensor offsets are invalid.")
        size = offsets[1] - offsets[0]
        payload += size
        if _is_unused_multimodal_tensor(name):
            unused += size
    return payload, unused


def snapshot_safetensors_bytes(model_path: Path) -> int:
    """Return safetensors bytes bound to tensors mlx-lm actually loads.

    Gemma 4 Hub snapshots also contain vision/audio shards that mlx-lm drops at
    sanitize. Those leftover payloads are not container overhead and are
    excluded from the packed-checkpoint comparison.
    """

    files = sorted(model_path.rglob("*.safetensors"))
    if not files:
        raise RuntimeError("Pinned MLX-LM snapshot contains no safetensors weights.")
    total = 0
    payload = 0
    unused = 0
    for path in files:
        if not path.is_file():
            raise RuntimeError("Pinned MLX-LM safetensors path is not a regular file.")
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError("Pinned MLX-LM safetensors shard is empty.")
        shard_payload, shard_unused = _safetensors_payload_accounting(path)
        total += size
        payload += shard_payload
        unused += shard_unused
    if payload <= 0:
        raise RuntimeError("Pinned MLX-LM snapshot has no positive tensor bytes.")
    if unused >= payload:
        raise RuntimeError(
            "Pinned MLX-LM snapshot has no language-tower safetensors payload."
        )
    bound = total - unused
    if bound <= 0:
        raise RuntimeError("Pinned MLX-LM snapshot has no positive tensor bytes.")
    return bound


def build_mlx_packed_checkpoint_binding(
    plan: dict[str, Any],
    parameter_census: dict[str, Any],
    *,
    observed_safetensors_bytes: int,
) -> dict[str, Any]:
    """Bind file bytes to the loaded model's logical parameter census."""

    observed_total = parameter_census.get("observed_total_parameters")
    if not _positive_int(observed_total) or not _positive_int(
        observed_safetensors_bytes
    ):
        raise RuntimeError("MLX packed-checkpoint evidence requires positive counts.")
    if plan["recommended"].get("method") == "qlora":
        expected_weights, expected_metadata = mlx_quantized_storage_bytes_for_contract(
            plan["model"], logical_parameters=int(observed_total)
        )
    else:
        expected_weights = round(int(observed_total) * 2.0)
        expected_metadata = 0
    expected_packed = expected_weights + expected_metadata
    overhead = int(observed_safetensors_bytes) - expected_packed
    overhead_limit = mlx_packed_checkpoint_overhead_limit(expected_packed)
    if overhead < 0:
        raise RuntimeError(
            "Pinned MLX-LM safetensors bytes are smaller than the bound packed tensor arithmetic."
        )
    if overhead > overhead_limit:
        raise RuntimeError(
            "Pinned MLX-LM safetensors container overhead exceeds the fail-closed Aptus bound."
        )
    binding = {
        "schema_version": "aptus.mlx-packed-checkpoint.v1",
        "observed_safetensors_bytes": int(observed_safetensors_bytes),
        "observed_logical_parameters": int(observed_total),
        "expected_weight_bytes": expected_weights,
        "expected_quantization_metadata_bytes": expected_metadata,
        "expected_packed_tensor_bytes": expected_packed,
        "container_overhead_bytes": overhead,
        "container_overhead_limit_bytes": overhead_limit,
    }
    binding["descriptor_sha256"] = _descriptor_sha256(binding)
    return binding


def require_mlx_packed_checkpoint_binding(
    plan: dict[str, Any], parameter_census: dict[str, Any], binding: Any
) -> dict[str, Any]:
    expected_keys = {
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
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise RuntimeError("MLX packed-checkpoint binding has an invalid shape.")
    if binding.get("descriptor_sha256") != _descriptor_sha256(
        {key: value for key, value in binding.items() if key != "descriptor_sha256"}
    ):
        raise RuntimeError("MLX packed-checkpoint binding digest is invalid.")
    rebuilt = build_mlx_packed_checkpoint_binding(
        plan,
        parameter_census,
        observed_safetensors_bytes=binding.get("observed_safetensors_bytes"),
    )
    if rebuilt != binding:
        raise RuntimeError(
            "MLX packed-checkpoint binding does not match the logical parameter census."
        )
    return binding


def build_mlx_model_parameter_census(
    model: Any,
    plan: dict[str, Any],
    *,
    parameter_counter: Any | None = None,
) -> dict[str, Any]:
    """Count logical MLX parameters and prove the planned routed-expert graph."""

    if parameter_counter is None:
        from mlx_lm.utils import get_total_parameters

        parameter_counter = get_total_parameters
    model_spec = plan["model"]
    declared_total = model_spec.get("parameters")
    declared_active = model_spec.get("active_parameters", declared_total)
    if not _positive_int(declared_total) or not _positive_int(declared_active):
        raise RuntimeError("The Aptus model contract has invalid parameter counts.")
    observed_total = parameter_counter(model)
    if not _positive_int(observed_total):
        raise RuntimeError("MLX-LM reported no positive logical model parameter count.")
    observed_total = int(observed_total)
    tolerance = max(1_000_000, round(int(declared_total) * 0.02))
    if abs(observed_total - int(declared_total)) > tolerance:
        raise RuntimeError(
            "MLX-LM logical model parameters differ from the declared total beyond the two-percent Aptus tolerance."
        )

    layers = tuple(getattr(model, "layers", ()))
    expected_layer_count = model_spec.get("layers")
    if len(layers) != expected_layer_count:
        raise RuntimeError(
            "Loaded MLX-LM transformer layer count does not match the Aptus plan."
        )

    moe = model_spec.get("moe")
    sparse_layer_count = 0
    routed_expert_parameters = 0
    active_routed_expert_parameters = 0
    inactive_expert_parameters = 0
    census_method = "mlx-lm.get_total_parameters.v1"
    if moe is not None:
        if not isinstance(moe, dict):
            raise RuntimeError("The Aptus MoE topology contract must be an object.")
        expert_count = moe.get("expert_count")
        experts_per_token = moe.get("experts_per_token")
        expert_intermediate_size = moe.get("expert_intermediate_size")
        decoder_sparse_step = moe.get("decoder_sparse_step")
        mlp_only_layers = moe.get("mlp_only_layers")
        if (
            not all(
                _positive_int(value)
                for value in (
                    expert_count,
                    experts_per_token,
                    expert_intermediate_size,
                    decoder_sparse_step,
                )
            )
            or not isinstance(mlp_only_layers, list)
            or any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in mlp_only_layers
            )
        ):
            raise RuntimeError("The Aptus MoE topology contract is invalid.")
        assert isinstance(expert_count, int)
        assert isinstance(experts_per_token, int)
        assert isinstance(expert_intermediate_size, int)
        assert isinstance(decoder_sparse_step, int)
        expected_sparse_indices = {
            index
            for index in range(int(expected_layer_count))
            if index not in set(mlp_only_layers)
            and (index + 1) % decoder_sparse_step == 0
        }
        expected_per_layer = (
            expert_count * 3 * int(model_spec["hidden_size"]) * expert_intermediate_size
        )
        for index, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if index in expected_sparse_indices:
                switch_mlp = getattr(mlp, "switch_mlp", None)
                if (
                    getattr(mlp, "num_experts", None) != expert_count
                    or getattr(mlp, "top_k", None) != experts_per_token
                    or switch_mlp is None
                ):
                    raise RuntimeError(
                        f"Loaded MLX-LM sparse layer {index} does not match the planned expert topology."
                    )
                observed_layer_parameters = parameter_counter(switch_mlp)
                if observed_layer_parameters != expected_per_layer:
                    raise RuntimeError(
                        f"Loaded MLX-LM sparse layer {index} has an unexpected logical expert parameter count."
                    )
                routed_expert_parameters += int(observed_layer_parameters)
            elif mlp is not None and (
                hasattr(mlp, "switch_mlp") or hasattr(mlp, "num_experts")
            ):
                raise RuntimeError(
                    f"Loaded MLX-LM layer {index} is sparse where the Aptus plan requires a dense MLP."
                )
        sparse_layer_count = len(expected_sparse_indices)
        if sparse_layer_count != model_spec.get("sparse_layer_count"):
            raise RuntimeError(
                "Loaded MLX-LM sparse layer count does not match the Aptus plan."
            )
        if routed_expert_parameters % expert_count:
            raise RuntimeError(
                "Loaded MLX-LM routed expert parameters cannot be divided by the planned expert count."
            )
        active_routed_expert_parameters = (
            routed_expert_parameters * experts_per_token // expert_count
        )
        inactive_expert_parameters = (
            routed_expert_parameters - active_routed_expert_parameters
        )
        census_method = "mlx-lm.get_total_parameters-plus-exact-qwen3-moe-routing.v1"
    else:
        for index, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and any(
                getattr(mlp, name, None) is not None
                for name in ("switch_mlp", "num_experts", "top_k")
            ):
                raise RuntimeError(
                    f"Loaded MLX-LM layer {index} is sparse where the Aptus plan requires dense topology."
                )

    observed_active = observed_total - inactive_expert_parameters
    if (
        observed_active <= 0
        or observed_active > observed_total
        or abs(observed_active - int(declared_active)) > tolerance
    ):
        raise RuntimeError(
            "MLX-LM active logical parameters differ from the Aptus model contract beyond tolerance."
        )
    census = {
        "schema_version": "aptus.mlx-model-parameter-census.v1",
        "census_method": census_method,
        "declared_total_parameters": int(declared_total),
        "observed_total_parameters": observed_total,
        "total_parameter_delta": observed_total - int(declared_total),
        "total_parameter_tolerance": tolerance,
        "declared_active_parameters": int(declared_active),
        "observed_active_parameters": observed_active,
        "sparse_layer_count": sparse_layer_count,
        "routed_expert_parameters": routed_expert_parameters,
        "active_routed_expert_parameters": active_routed_expert_parameters,
        "inactive_expert_parameters": inactive_expert_parameters,
    }
    census["descriptor_sha256"] = _descriptor_sha256(census)
    return census


def build_mlx_model_load_binding(
    model: Any,
    plan: dict[str, Any],
    *,
    observed_safetensors_bytes: int,
    architecture_contract: dict[str, Any] | None = None,
    parameter_counter: Any | None = None,
) -> dict[str, Any]:
    if architecture_contract is None:
        architecture_contract = expected_model_architecture_contract(plan["model"])
    parameter_census = build_mlx_model_parameter_census(
        model, plan, parameter_counter=parameter_counter
    )
    binding = {
        "schema_version": "aptus.mlx-model-load-binding.v3",
        "model_id": plan["model"]["model_id"],
        "model_revision": plan["model"]["revision"],
        "resolved_local_snapshot": True,
        "trust_remote_code": False,
        "architecture_contract": architecture_contract,
        "parameter_census": parameter_census,
        "packed_checkpoint_binding": build_mlx_packed_checkpoint_binding(
            plan,
            parameter_census,
            observed_safetensors_bytes=observed_safetensors_bytes,
        ),
    }
    binding["descriptor_sha256"] = _descriptor_sha256(binding)
    return binding


def require_mlx_model_load_binding(
    plan: dict[str, Any], binding: Any
) -> dict[str, Any]:
    if not isinstance(binding, dict) or set(binding) != {
        "schema_version",
        "model_id",
        "model_revision",
        "resolved_local_snapshot",
        "trust_remote_code",
        "architecture_contract",
        "parameter_census",
        "packed_checkpoint_binding",
        "descriptor_sha256",
    }:
        raise RuntimeError("MLX-LM model-load binding has an invalid shape.")
    expected_static = {
        "schema_version": "aptus.mlx-model-load-binding.v3",
        "model_id": plan["model"]["model_id"],
        "model_revision": plan["model"]["revision"],
        "resolved_local_snapshot": True,
        "trust_remote_code": False,
        "architecture_contract": expected_model_architecture_contract(plan["model"]),
    }
    if any(binding.get(key) != value for key, value in expected_static.items()):
        raise RuntimeError(
            "MLX-LM model-load binding does not match the pinned model architecture contract."
        )
    if binding.get("descriptor_sha256") != _descriptor_sha256(
        {key: value for key, value in binding.items() if key != "descriptor_sha256"}
    ):
        raise RuntimeError("MLX-LM model-load binding digest is invalid.")

    census = binding.get("parameter_census")
    expected_census_keys = {
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
    if not isinstance(census, dict) or set(census) != expected_census_keys:
        raise RuntimeError("MLX-LM model parameter census has an invalid shape.")
    if census.get("descriptor_sha256") != _descriptor_sha256(
        {key: value for key, value in census.items() if key != "descriptor_sha256"}
    ):
        raise RuntimeError("MLX-LM model parameter census digest is invalid.")
    model_spec = plan["model"]
    declared_total = int(model_spec["parameters"])
    declared_active = int(model_spec.get("active_parameters", declared_total))
    tolerance = max(1_000_000, round(declared_total * 0.02))
    observed_total = census.get("observed_total_parameters")
    observed_active = census.get("observed_active_parameters")
    moe = model_spec.get("moe")
    sparse_layer_count = int(model_spec.get("sparse_layer_count", 0))
    if moe is None:
        routed = active_routed = inactive = 0
        method = "mlx-lm.get_total_parameters.v1"
    else:
        routed = (
            sparse_layer_count
            * int(moe["expert_count"])
            * 3
            * int(model_spec["hidden_size"])
            * int(moe["expert_intermediate_size"])
        )
        active_routed = (
            routed * int(moe["experts_per_token"]) // int(moe["expert_count"])
        )
        inactive = routed - active_routed
        method = "mlx-lm.get_total_parameters-plus-exact-qwen3-moe-routing.v1"
    if (
        not _positive_int(observed_total)
        or not _positive_int(observed_active)
        or abs(int(observed_total) - declared_total) > tolerance
        or abs(int(observed_active) - declared_active) > tolerance
        or int(observed_active) != int(observed_total) - inactive
        or census.get("schema_version") != "aptus.mlx-model-parameter-census.v1"
        or census.get("census_method") != method
        or census.get("declared_total_parameters") != declared_total
        or census.get("total_parameter_delta") != int(observed_total) - declared_total
        or census.get("total_parameter_tolerance") != tolerance
        or census.get("declared_active_parameters") != declared_active
        or census.get("sparse_layer_count") != sparse_layer_count
        or census.get("routed_expert_parameters") != routed
        or census.get("active_routed_expert_parameters") != active_routed
        or census.get("inactive_expert_parameters") != inactive
    ):
        raise RuntimeError(
            "MLX-LM model parameter census does not match the Aptus model contract."
        )
    require_mlx_packed_checkpoint_binding(
        plan, census, binding.get("packed_checkpoint_binding")
    )
    return binding


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
        raise RuntimeError(
            "Current Apple unified-memory admission probe failed."
        ) from error
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


def require_unified_memory_admission_binding(
    plan: dict[str, Any], admission: Any
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
        raise RuntimeError("MLX unified-memory admission must be an object.")
    observed = admission.get("observed_safetensors_bytes")
    if not _positive_int(observed):
        raise RuntimeError(
            "MLX unified-memory admission requires positive safetensors bytes."
        )
    adjustment = max(0, int(observed) - planned_resident)
    adjusted_point = point + adjustment
    adjusted_upper = upper + adjustment
    required = max(adjusted_point, adjusted_upper) + reserve
    expected = {
        "schema_version": "aptus.mlx-unified-memory-admission.v2",
        "planned_resident_bytes": planned_resident,
        "observed_safetensors_bytes": int(observed),
        "resident_adjustment_bytes": adjustment,
        "adjusted_point_estimate_bytes": adjusted_point,
        "adjusted_upper_estimate_bytes": adjusted_upper,
        "reserve_bytes": reserve,
        "required_available_bytes": required,
    }
    available = admission.get("available_unified_memory_bytes")
    if (
        set(admission) != set(expected) | {"available_unified_memory_bytes"}
        or any(admission.get(key) != value for key, value in expected.items())
        or not _positive_int(available)
        or int(available) < required
        or "free_vram_bytes" in admission
    ):
        raise RuntimeError(
            "MLX unified-memory admission does not bind a passing packed-checkpoint measurement."
        )
    return admission


def require_unified_memory_admission(
    plan: dict[str, Any], model_path: Path
) -> dict[str, Any]:
    candidate = plan["recommended"]
    memory = candidate["memory"]
    planned_resident = int(memory["base_weights_bytes"]) + int(
        memory["quantization_metadata_bytes"]
    )
    observed = snapshot_safetensors_bytes(model_path)
    adjustment = max(0, observed - planned_resident)
    point = int(memory["point_estimate_bytes"]) + adjustment
    upper = int(memory["upper_estimate_bytes"]) + adjustment
    reserve = max(
        int(plan["hardware"].get("reserve_per_device_bytes", 0)),
        8 * 1024**3,
    )
    available = current_available_unified_memory_bytes()
    required = max(point, upper) + reserve
    if available < required:
        raise RuntimeError(
            "Current available Apple unified memory is below the packed-checkpoint-adjusted "
            "candidate upper estimate plus the required 8 GiB Aptus reserve. "
            f"required={required} bytes; available={available} bytes; "
            f"shortfall={required - available} bytes."
        )
    admission = {
        "schema_version": "aptus.mlx-unified-memory-admission.v2",
        "available_unified_memory_bytes": available,
        "planned_resident_bytes": planned_resident,
        "observed_safetensors_bytes": observed,
        "resident_adjustment_bytes": adjustment,
        "adjusted_point_estimate_bytes": point,
        "adjusted_upper_estimate_bytes": upper,
        "reserve_bytes": reserve,
        "required_available_bytes": required,
    }
    return require_unified_memory_admission_binding(plan, admission)


def resolve_lora_keys(
    model: Any, candidate: dict[str, Any], *, family: str
) -> tuple[list[str], dict[str, Any]]:
    planned = candidate.get("target_modules")
    if (
        not isinstance(planned, list)
        or not planned
        or any(not isinstance(target, str) or not target for target in planned)
        or len(set(planned)) != len(planned)
    ):
        raise RuntimeError(
            "The MLX-LM candidate requires unique planned target modules."
        )
    layers = tuple(getattr(model, "layers", ()))
    if not layers:
        raise RuntimeError("The loaded MLX-LM model exposes no transformer layers.")
    resolved: dict[str, str] = {}
    target_layer_counts: dict[str, int] = {}
    for target in planned:
        observed: list[str] = []
        for layer_index, layer in enumerate(layers):
            matches = sorted(
                name
                for name, _module in layer.named_modules()
                if name == target or name.endswith("." + target)
            )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Planned MLX target {target!r} matched {len(matches)} modules "
                    f"in transformer layer {layer_index}; at most one is required."
                )
            if matches:
                observed.append(matches[0])
        if not observed:
            raise RuntimeError(
                f"Planned MLX target {target!r} matched 0 modules in the loaded transformer."
            )
        if len(set(observed)) != 1:
            raise RuntimeError(
                f"Planned MLX target {target!r} does not resolve to one stable layer-relative key."
            )
        resolved[target] = observed[0]
        target_layer_counts[target] = len(observed)
    try:
        expected_instances = mlx_trainable_target_instance_total(
            planned, len(layers), target_layer_counts, family=family
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    resolved_keys = [resolved[target] for target in planned]
    if len(set(resolved_keys)) != len(resolved_keys):
        raise RuntimeError(
            "Distinct planned MLX targets resolve to the same runtime key."
        )
    binding = {
        "schema_version": "aptus.mlx-trainable-target-binding.v1",
        "planned_target_modules": planned,
        "resolved_layer_keys": resolved_keys,
        "transformer_layer_count": len(layers),
        "expected_adapter_target_instance_count": expected_instances,
        "target_instance_counts": target_layer_counts,
    }
    return resolved_keys, binding


def require_trainable_binding(
    names: list[str], binding: dict[str, Any], *, family: str
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
        raise RuntimeError(
            "Every planned MLX adapter instance requires one LoRA A/B pair."
        )
    for base in pairs:
        target = next(
            target
            for target in planned
            if base == target or base.endswith("." + target)
        )
        target_counts[target] += 1
    layer_count = binding["transformer_layer_count"]
    inspect_counts = binding.get("target_instance_counts")
    try:
        expected_instances = mlx_trainable_target_instance_total(
            planned, layer_count, target_counts, family=family
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if (
        inspect_counts != target_counts
        or len(pairs) != expected_instances
        or binding.get("expected_adapter_target_instance_count") != expected_instances
    ):
        raise RuntimeError(
            "The MLX trainable adapter set does not match the loaded planned-target instances."
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
            raise RuntimeError(
                "MLX-LM full-run batch and epoch values must be positive."
            )
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
    observed_safetensors_bytes: int,
    parameter_counter: Any | None = None,
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
    architecture_contract = require_method_model(
        plan, plan["recommended"], expected_path
    )
    loaded = loader(str(expected_path), tokenizer_config={"trust_remote_code": False})
    if not isinstance(loaded, tuple) or len(loaded) < 2:
        raise RuntimeError(
            "Pinned MLX-LM loader returned an unsupported model payload."
        )
    binding = build_mlx_model_load_binding(
        loaded[0],
        plan,
        observed_safetensors_bytes=observed_safetensors_bytes,
        architecture_contract=architecture_contract,
        parameter_counter=parameter_counter,
    )
    require_mlx_model_load_binding(plan, binding)
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
    require_method_model(plan, candidate, model_path)
    memory_admission = require_unified_memory_admission(plan, model_path)

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
            observed_safetensors_bytes=memory_admission["observed_safetensors_bytes"],
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
        resolved_keys, binding = resolve_lora_keys(
            model, candidate, family=str(plan["model"]["family"])
        )
        config["keys"] = resolved_keys
        evidence["resolved_binding"] = binding
        original_linear_to_lora_layers(model, num_layers, config, use_dora=use_dora)

    def instrumented_train(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        training_args = kwargs.get("args")
        if model is None or training_args is None:
            raise RuntimeError("Pinned MLX-LM train invocation changed shape.")
        update_opportunities = int(training_args.iters) // int(
            training_args.grad_accumulation_steps
        )
        if update_opportunities < 1:
            raise RuntimeError(
                "The bounded MLX smoke schedules no optimizer update after gradient accumulation."
            )
        before = dict(tree_flatten(model.trainable_parameters()))
        binding = require_trainable_binding(
            sorted(before),
            evidence.get("resolved_binding", {}),
            family=str(plan["model"]["family"]),
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
            raise RuntimeError(
                "The MLX trainable parameter set changed during training."
            )
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
        raise RuntimeError(
            "Compiled MLX split counts do not match their bound contract."
        )
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
        raise RuntimeError(
            "MLX-LM derived no training iterations from the compiled data."
        )
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
        raise RuntimeError(
            "MLX-LM pilot requires at least two completed optimizer updates."
        )
    validation_losses = evidence.get("validation_losses")
    if valid_examples > 0 and (
        not isinstance(validation_losses, list)
        or not validation_losses
        or any(not math.isfinite(loss) for loss in validation_losses)
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
