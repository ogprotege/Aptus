from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "aptus.training-plan.v3"
FORMULA_VERSION = "aptus-memory-v2"
MLX_FORMULA_VERSION = "aptus-memory-mlx-v2"
RUNTIME_CONTRACT_VERSION = "aptus.runtime-contract.v1"
CANDIDATE_STATUSES = {"feasible", "conditional", "infeasible", "unsupported"}
METHODS = {"full", "lora", "int8-lora", "qlora"}
DISTRIBUTIONS = {"single", "ddp", "fsdp"}
TRAINING_RUNTIMES = {"transformers-peft-cuda", "mlx-lm", "pytorch-mps"}
EVIDENCE_REQUIREMENTS = {"pilot-required", "implementation-required"}
QWEN3_MOE_FAMILY = "qwen3_moe"
QWEN3_MOE_MODEL_TYPE = "qwen3_moe"
QWEN3_MOE_ARCHITECTURE = "Qwen3MoeForCausalLM"
DENSE_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
MODEL_TARGET_MODULES = {
    family: DENSE_TARGET_MODULES for family in ("gemma", "llama", "mistral", "qwen")
}
MODEL_TARGET_MODULES[QWEN3_MOE_FAMILY] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]
MOE_TOPOLOGY_FIELDS = (
    "expert_count",
    "experts_per_token",
    "expert_intermediate_size",
    "decoder_sparse_step",
    "mlp_only_layers",
    "shared_expert_intermediate_size",
)
QUANTIZATION_LAYOUT_FIELDS = (
    "default_bits",
    "default_group_size",
    "module_overrides",
)
QUANTIZATION_OVERRIDE_FIELDS = ("module_path", "bits", "group_size")

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
        "aptus-memory-mlx-v2",
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
        "aptus-memory-mlx-v2",
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
        "model-data-evidence.json",
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


def _model_config_source(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, Mapping) else config


def _config_first(config: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = config.get(name)
        if value is not None:
            return value
    return None


def _config_quantization_bits(config: Mapping[str, Any]) -> Any:
    source = _model_config_source(config)
    quantization = (
        source.get("quantization")
        or source.get("quantization_config")
        or config.get("quantization")
        or config.get("quantization_config")
    )
    return quantization.get("bits") if isinstance(quantization, Mapping) else None


def _normalized_quantization_layout(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(QUANTIZATION_LAYOUT_FIELDS):
        raise ValueError(
            "Model quantization_layout must contain the exact v3 layout fields."
        )
    default_bits = value.get("default_bits")
    default_group_size = value.get("default_group_size")
    if (
        not _positive_int(default_bits)
        or default_bits > 16
        or not _positive_int(default_group_size)
    ):
        raise ValueError(
            "Model quantization layout defaults require bits from 1 through 16 "
            "and a positive group_size."
        )
    raw_overrides = value.get("module_overrides")
    if not isinstance(raw_overrides, (list, tuple)):
        raise ValueError("Model quantization module_overrides must be a list.")
    overrides: list[dict[str, Any]] = []
    for item in raw_overrides:
        if not isinstance(item, Mapping) or set(item) != set(
            QUANTIZATION_OVERRIDE_FIELDS
        ):
            raise ValueError(
                "Each model quantization override must contain only module_path, "
                "bits, and group_size."
            )
        module_path = item.get("module_path")
        bits = item.get("bits")
        group_size = item.get("group_size")
        if (
            not isinstance(module_path, str)
            or not module_path
            or len(module_path) > 256
            or any(
                not part
                or any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                    for character in part
                )
                for part in module_path.split(".")
            )
        ):
            raise ValueError(
                "Model quantization override module_path must be a dotted module identifier."
            )
        if not _positive_int(bits) or bits > 16 or not _positive_int(group_size):
            raise ValueError(
                "Model quantization override bits or group_size is invalid."
            )
        overrides.append(
            {
                "module_path": module_path,
                "bits": bits,
                "group_size": group_size,
            }
        )
    paths = [item["module_path"] for item in overrides]
    if paths != sorted(set(paths)):
        raise ValueError(
            "Model quantization module_overrides must be sorted and unique."
        )
    return {
        "default_bits": default_bits,
        "default_group_size": default_group_size,
        "module_overrides": overrides,
    }


def _reviewed_qwen3_moe_quantization_layout(layers: int) -> dict[str, Any]:
    if not _positive_int(layers):
        raise ValueError("Reviewed Qwen3 MoE quantization requires positive layers.")
    return {
        "default_bits": 4,
        "default_group_size": 64,
        "module_overrides": [
            {
                "module_path": f"model.layers.{index}.mlp.gate",
                "bits": 8,
                "group_size": 64,
            }
            for index in sorted(range(layers), key=lambda value: str(value))
        ],
    }


def _canonical_config_quantization_layout(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    layout = {
        "default_bits": value.get("bits"),
        "default_group_size": value.get("group_size"),
        "module_overrides": [
            {
                "module_path": key,
                "bits": item.get("bits") if isinstance(item, Mapping) else None,
                "group_size": (
                    item.get("group_size") if isinstance(item, Mapping) else None
                ),
            }
            for key, item in value.items()
            if key not in {"bits", "group_size"}
        ],
    }
    layout["module_overrides"].sort(key=lambda item: item["module_path"])
    try:
        normalized = _normalized_quantization_layout(layout)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Pinned model quantization layout contains unsupported fields or values."
        ) from error
    assert normalized is not None
    for key, item in value.items():
        if key in {"bits", "group_size"}:
            continue
        if not isinstance(item, Mapping) or set(item) != {"bits", "group_size"}:
            raise ValueError(
                "Pinned model quantization layout contains unsupported override fields."
            )
    return normalized


def _config_quantization_layout(config: Mapping[str, Any]) -> dict[str, Any] | None:
    source = _model_config_source(config)
    candidates: list[Mapping[str, Any]] = []
    for container in (source, config):
        for name in ("quantization", "quantization_config"):
            value = container.get(name)
            if isinstance(value, Mapping) and all(
                value is not item for item in candidates
            ):
                candidates.append(value)
    if not candidates:
        return None
    layouts = [_canonical_config_quantization_layout(item) for item in candidates]
    if any(item != layouts[0] for item in layouts[1:]):
        raise ValueError(
            "Pinned model quantization and quantization_config layouts disagree."
        )
    return layouts[0]


def _normalized_moe(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(MOE_TOPOLOGY_FIELDS):
        raise ValueError("Model moe must contain the exact v3 topology fields.")
    positive_names = (
        "expert_count",
        "experts_per_token",
        "expert_intermediate_size",
        "decoder_sparse_step",
    )
    if any(not _positive_int(value.get(name)) for name in positive_names):
        raise ValueError("Model MoE topology integer facts must be positive.")
    if value["experts_per_token"] > value["expert_count"]:
        raise ValueError("Model experts_per_token cannot exceed expert_count.")
    shared = value.get("shared_expert_intermediate_size")
    if shared is not None and not _positive_int(shared):
        raise ValueError(
            "Model shared_expert_intermediate_size must be positive when supplied."
        )
    mlp_only = value.get("mlp_only_layers")
    if (
        not isinstance(mlp_only, (list, tuple))
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in mlp_only
        )
        or list(mlp_only) != sorted(set(mlp_only))
    ):
        raise ValueError("Model mlp_only_layers must be a sorted unique integer list.")
    return {
        "expert_count": value["expert_count"],
        "experts_per_token": value["experts_per_token"],
        "expert_intermediate_size": value["expert_intermediate_size"],
        "decoder_sparse_step": value["decoder_sparse_step"],
        "mlp_only_layers": list(mlp_only),
        "shared_expert_intermediate_size": shared,
    }


def _derived_model_topology(
    model: Mapping[str, Any], moe: Mapping[str, Any] | None
) -> tuple[int, int]:
    parameters = model.get("parameters")
    layers = model.get("layers")
    hidden_size = model.get("hidden_size")
    if not all(_positive_int(value) for value in (parameters, layers, hidden_size)):
        raise ValueError("Model structural facts are incomplete.")
    assert isinstance(parameters, int)
    assert isinstance(layers, int)
    assert isinstance(hidden_size, int)
    if moe is None:
        return 0, parameters
    mlp_only = set(moe["mlp_only_layers"])
    if any(index >= layers for index in mlp_only):
        raise ValueError("Model mlp_only_layers references a missing layer.")
    sparse_layers = sum(
        1
        for index in range(layers)
        if (index + 1) % moe["decoder_sparse_step"] == 0 and index not in mlp_only
    )
    if sparse_layers <= 0:
        raise ValueError("Model MoE topology must contain at least one sparse layer.")
    inactive = (
        sparse_layers
        * (moe["expert_count"] - moe["experts_per_token"])
        * 3
        * hidden_size
        * moe["expert_intermediate_size"]
    )
    active = parameters - inactive
    if active <= 0 or active > parameters:
        raise ValueError(
            "Model derived active_parameters must be positive and no greater than total parameters."
        )
    return sparse_layers, active


def expected_model_architecture_contract(
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic model architecture contract carried by runtime proof."""

    if not isinstance(model, Mapping):
        raise ValueError("Model architecture contract requires a model object.")
    architecture = model.get("architecture")
    model_type = model.get("model_type")
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("Model architecture is required.")
    if model_type is not None and (
        not isinstance(model_type, str) or not model_type.strip()
    ):
        raise ValueError("Model model_type must be non-empty when supplied.")
    for name in ("parameters", "hidden_size", "layers", "context_length"):
        if not _positive_int(model.get(name)):
            raise ValueError(f"Model {name} must be positive.")
    intermediate_size = model.get("intermediate_size")
    if intermediate_size is not None and not _positive_int(intermediate_size):
        raise ValueError("Model intermediate_size must be positive when supplied.")
    quantization_bits = model.get("quantization_bits")
    if quantization_bits is not None and (
        not _positive_int(quantization_bits) or quantization_bits > 16
    ):
        raise ValueError("Model quantization_bits must be between 1 and 16.")
    quantization_layout = _normalized_quantization_layout(
        model.get("quantization_layout")
    )
    if quantization_layout is not None and (
        quantization_bits != quantization_layout["default_bits"]
    ):
        raise ValueError(
            "Model quantization_bits must equal quantization_layout default_bits."
        )
    moe = _normalized_moe(model.get("moe"))
    sparse_layers, active_parameters = _derived_model_topology(model, moe)
    if model.get("sparse_layer_count") != sparse_layers:
        raise ValueError("Model sparse_layer_count does not match its topology.")
    if model.get("active_parameters") != active_parameters:
        raise ValueError("Model active_parameters does not match its topology.")
    if moe is not None:
        if (
            model.get("family") != QWEN3_MOE_FAMILY
            or model_type != QWEN3_MOE_MODEL_TYPE
            or architecture != QWEN3_MOE_ARCHITECTURE
        ):
            raise ValueError(
                "Model MoE topology requires the exact reviewed Qwen3 MoE identity."
            )
        if moe["shared_expert_intermediate_size"] is not None:
            raise ValueError(
                "The reviewed Qwen3 MoE runtime does not support a shared expert."
            )
        if quantization_bits != 4:
            raise ValueError(
                "The reviewed Qwen3 MoE runtime requires explicit four-bit metadata."
            )
        if quantization_layout != _reviewed_qwen3_moe_quantization_layout(
            model["layers"]
        ):
            raise ValueError(
                "The reviewed Qwen3 MoE runtime requires the exact four-bit "
                "group-64 layout with one eight-bit group-64 router-gate override "
                "for every layer."
            )
    quantization_layout_sha256 = (
        hashlib.sha256(
            json.dumps(
                quantization_layout, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if quantization_layout is not None
        else None
    )
    payload = {
        "schema_version": "aptus.model-architecture-contract.v1",
        "model_type": model_type,
        "architecture": architecture,
        "parameters": model["parameters"],
        "hidden_size": model["hidden_size"],
        "intermediate_size": intermediate_size,
        "layers": model["layers"],
        "context_length": model["context_length"],
        "quantization_bits": quantization_bits,
        "quantization_layout": quantization_layout,
        "quantization_layout_sha256": quantization_layout_sha256,
        "moe": moe,
        "sparse_layer_count": sparse_layers,
        "active_parameters": active_parameters,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def validate_model_config_against_plan(
    model: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Match a pinned provider config to the plan and return its proof payload."""

    expected = expected_model_architecture_contract(model)
    if not isinstance(config, Mapping):
        raise ValueError("Pinned model config must be an object.")
    source = _model_config_source(config)
    expected_model_type = expected["model_type"]
    observed_model_type = source.get("model_type") or config.get("model_type")
    if expected_model_type is not None and observed_model_type != expected_model_type:
        raise ValueError("Pinned model_type does not match the plan.")
    raw_architectures = config.get("architectures") or source.get("architectures")
    observed_architecture = (
        raw_architectures[0]
        if isinstance(raw_architectures, list)
        and raw_architectures
        and isinstance(raw_architectures[0], str)
        else observed_model_type
    )
    if (
        expected["architecture"] != "causal-lm"
        and observed_architecture != expected["architecture"]
    ):
        raise ValueError("Pinned model architecture does not match the plan.")
    structural_names = {
        "hidden_size": ("hidden_size", "d_model", "n_embd"),
        "intermediate_size": ("intermediate_size", "ffn_dim", "n_inner"),
        "layers": ("num_hidden_layers", "n_layer", "num_layers"),
        "context_length": (
            "max_position_embeddings",
            "n_positions",
            "seq_length",
        ),
    }
    for planned_name, config_names in structural_names.items():
        planned = expected[planned_name]
        observed = _config_first(source, *config_names)
        if planned is not None and (
            (expected["moe"] is not None and observed != planned)
            or (observed is not None and observed != planned)
        ):
            raise ValueError(f"Pinned model {planned_name} does not match the plan.")
    observed_bits = _config_quantization_bits(config)
    if (
        expected["quantization_bits"] is not None
        and observed_bits != expected["quantization_bits"]
    ):
        raise ValueError("Pinned model quantization bits do not match the plan.")
    expected_layout = expected["quantization_layout"]
    if expected_layout is not None:
        observed_layout = _config_quantization_layout(config)
        if observed_layout != expected_layout:
            raise ValueError(
                "Pinned model quantization layout does not match the plan."
            )
    expected_moe = expected["moe"]
    if expected_moe is not None:
        observed_moe = {
            "expert_count": source.get("num_experts"),
            "experts_per_token": source.get("num_experts_per_tok"),
            "expert_intermediate_size": source.get("moe_intermediate_size"),
            "decoder_sparse_step": source.get("decoder_sparse_step"),
            "mlp_only_layers": source.get("mlp_only_layers"),
            "shared_expert_intermediate_size": source.get(
                "shared_expert_intermediate_size"
            ),
        }
        if observed_moe != expected_moe:
            raise ValueError("Pinned model MoE topology does not match the plan.")
    return expected


def _select(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    source = _mapping(value)
    return {key: source.get(key) for key in keys}


def _normalized_model(value: Any) -> dict[str, Any]:
    model = _select(
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
            "model_type",
            "quantization_bits",
            "quantization_layout",
            "sparse_layer_count",
            "active_parameters",
            "tokenizer_id",
            "license_name",
            "training_allowed",
        ),
    )
    moe = _mapping(value).get("moe")
    quantization_layout = _mapping(value).get("quantization_layout")
    model["quantization_layout"] = (
        _normalized_quantization_layout(quantization_layout)
        if quantization_layout is not None
        else None
    )
    model["moe"] = (
        _select(moe, MOE_TOPOLOGY_FIELDS) if isinstance(moe, Mapping) else None
    )
    return model


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


def _mlx_adapter_parameter_count(
    model: Mapping[str, Any], *, rank: int, target_modules: list[str] | tuple[str, ...]
) -> int:
    """Return the deterministic adapter cardinality used by the MLX estimator."""

    hidden_size = model.get("hidden_size")
    layers = model.get("layers")
    if not _positive_int(hidden_size) or not _positive_int(layers):
        raise ValueError("MLX memory recomputation requires positive model dimensions.")
    if not _positive_int(rank):
        raise ValueError("MLX memory recomputation requires a positive adapter rank.")
    intermediate_size = model.get("intermediate_size")
    if intermediate_size is None:
        intermediate_size = hidden_size * 4
    if not _positive_int(intermediate_size):
        raise ValueError(
            "MLX memory recomputation requires a positive intermediate dimension."
        )
    moe = model.get("moe")
    if moe is not None and any(
        module in {"gate_proj", "up_proj", "down_proj"} for module in target_modules
    ):
        raise ValueError(
            "MLX MoE memory recomputation refuses topology-free expert adapters."
        )
    per_layer = 0
    for module in target_modules:
        if module in {"gate_proj", "up_proj", "down_proj"}:
            per_layer += hidden_size + intermediate_size
        elif module in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            per_layer += hidden_size * 2
        else:
            raise ValueError(
                f"MLX memory recomputation does not recognize target module {module!r}."
            )
    return layers * rank * per_layer


def mlx_quantized_storage_bytes_for_contract(
    model: Mapping[str, Any], *, logical_parameters: int | None = None
) -> tuple[int, int]:
    """Price the bound MLX affine layout, including reviewed router overrides."""

    parameters = (
        logical_parameters
        if logical_parameters is not None
        else model.get("parameters")
    )
    if not _positive_int(parameters):
        raise ValueError(
            "MLX memory recomputation requires a positive parameter count."
        )
    layout = model.get("quantization_layout")
    if layout is None:
        # Dense v3 plans predate an exact layout binding. Preserve their named
        # analytical prior while the runtime still requires real four-bit metadata.
        return round(parameters * 0.5), round(parameters * 0.0625)
    if not isinstance(layout, Mapping):
        raise ValueError("MLX quantization layout must be an object.")
    default_bits = layout.get("default_bits")
    default_group_size = layout.get("default_group_size")
    overrides = layout.get("module_overrides")
    if (
        not _positive_int(default_bits)
        or not _positive_int(default_group_size)
        or not isinstance(overrides, list)
    ):
        raise ValueError("MLX quantization layout is incomplete.")

    moe = model.get("moe")
    hidden_size = model.get("hidden_size")
    if not isinstance(moe, Mapping) or not _positive_int(hidden_size):
        raise ValueError(
            "MLX quantization overrides require a bound MoE topology and hidden size."
        )
    expert_count = moe.get("expert_count")
    if not _positive_int(expert_count):
        raise ValueError("MLX quantization overrides require a positive expert count.")

    overridden_parameters = 0
    weighted_storage_bits = 0
    weighted_metadata_bytes = 0.0
    seen_paths: set[str] = set()
    for override in overrides:
        if not isinstance(override, Mapping):
            raise ValueError("MLX quantization overrides must be objects.")
        module_path = override.get("module_path")
        bits = override.get("bits")
        group_size = override.get("group_size")
        if (
            not isinstance(module_path, str)
            or not module_path
            or module_path in seen_paths
            or not module_path.startswith("model.layers.")
            or not module_path.endswith(".mlp.gate")
            or not _positive_int(bits)
            or not _positive_int(group_size)
        ):
            raise ValueError("MLX quantization override is not a unique router gate.")
        seen_paths.add(module_path)
        parameter_count = hidden_size * expert_count
        overridden_parameters += parameter_count
        weighted_storage_bits += parameter_count * bits
        # MLX affine quantization stores one half-precision scale and bias per group.
        weighted_metadata_bytes += parameter_count * 4 / group_size
    if overridden_parameters > parameters:
        raise ValueError("MLX quantization overrides exceed the model parameter count.")
    default_parameters = parameters - overridden_parameters
    storage_bytes = round(
        (default_parameters * default_bits + weighted_storage_bits) / 8
    )
    metadata_bytes = round(
        default_parameters * 4 / default_group_size + weighted_metadata_bytes
    )
    return storage_bytes, metadata_bytes


def mlx_memory_breakdown_for_contract(
    *,
    model: Mapping[str, Any],
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the portable MLX memory contract from normalized plan facts."""

    method = candidate.get("method")
    if method not in {"lora", "qlora"}:
        raise ValueError("The MLX estimator supports LoRA and QLoRA only.")
    parameters = model.get("parameters")
    hidden_size = model.get("hidden_size")
    layers = model.get("layers")
    sequence_length = target.get("sequence_length")
    micro_batch_size = candidate.get("micro_batch_size")
    rank = candidate.get("rank")
    target_modules = candidate.get("target_modules")
    if not all(
        _positive_int(value)
        for value in (
            parameters,
            hidden_size,
            layers,
            sequence_length,
            micro_batch_size,
            rank,
        )
    ) or not isinstance(target_modules, list):
        raise ValueError("MLX memory recomputation requires complete positive facts.")

    if method == "lora":
        base_weights = round(parameters * 2.0)
        quantization_metadata = 0
    else:
        base_weights, quantization_metadata = mlx_quantized_storage_bytes_for_contract(
            model
        )
    trainable = _mlx_adapter_parameter_count(
        model, rank=rank, target_modules=target_modules
    )
    adapter_weights = round(trainable * 4)
    gradients = round(trainable * 4)
    optimizer = round(trainable * 8)
    dense_activations = round(
        micro_batch_size * sequence_length * hidden_size * layers * 2 * 3.0
    )
    routed_expert_activations = 0
    moe = model.get("moe")
    if isinstance(moe, Mapping):
        sparse_layer_count = model.get("sparse_layer_count")
        experts_per_token = moe.get("experts_per_token")
        expert_intermediate_size = moe.get("expert_intermediate_size")
        if not all(
            _positive_int(value)
            for value in (
                sparse_layer_count,
                experts_per_token,
                expert_intermediate_size,
            )
        ):
            raise ValueError(
                "MLX MoE memory recomputation requires complete derived topology."
            )
        routed_expert_activations = round(
            micro_batch_size
            * sequence_length
            * sparse_layer_count
            * experts_per_token
            * expert_intermediate_size
            * 2
            * 3.0
        )
    activations = dense_activations + routed_expert_activations
    resident_bytes = base_weights + quantization_metadata
    workspace = max(round(0.75 * 1024**3), round(resident_bytes * 0.04))
    temporary = max(round(0.75 * 1024**3), round(resident_bytes * 0.08))
    load_transient = round(resident_bytes * 0.30)
    point_components = {
        "base_weights_bytes": base_weights,
        "quantization_metadata_bytes": quantization_metadata,
        "adapter_weights_bytes": adapter_weights,
        "adapter_gradients_bytes": gradients,
        "optimizer_states_bytes": optimizer,
        "activations_bytes": activations,
        "communication_bytes": 0,
        "workspace_bytes": workspace,
        "temporary_overhead_bytes": temporary,
        "load_transient_bytes": load_transient,
    }
    allocator = round(sum(point_components.values()) * 0.15)
    point_components["allocator_bytes"] = allocator
    point = sum(point_components.values())
    safety = round(point * 0.25)
    component_upper_bounds = {
        **point_components,
        "activations_bytes": round(activations * 1.60),
        "workspace_bytes": round(workspace * 1.75),
        "temporary_overhead_bytes": round(temporary * 1.75),
        "allocator_bytes": round(allocator * 1.75),
        "load_transient_bytes": round(load_transient * 1.50),
        "uncertainty_bytes": safety,
    }
    return {
        **point_components,
        "safety_margin_bytes": safety,
        "point_estimate_bytes": point,
        "estimated_peak_bytes": point,
        "upper_estimate_bytes": sum(component_upper_bounds.values()),
        "uncertainty_bytes": safety,
        "formula_version": MLX_FORMULA_VERSION,
        "component_upper_bounds": component_upper_bounds,
    }


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
    if plan.get("schema_version") == "aptus.training-plan.v2" and any(
        key in _mapping(plan.get("model"))
        for key in (
            "model_type",
            "quantization_bits",
            "quantization_layout",
            "moe",
            "sparse_layer_count",
            "active_parameters",
        )
    ):
        errors.append("A v2 plan cannot contain v3 model architecture or MoE fields.")
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
    architecture = model.get("architecture")
    if not isinstance(architecture, str) or not architecture.strip():
        errors.append("Model architecture is required.")
    model_type = model.get("model_type")
    if model_type is not None and (
        not isinstance(model_type, str) or not model_type.strip()
    ):
        errors.append("Model model_type must be non-empty when supplied.")
    moe_identity = bool(
        model.get("family") == QWEN3_MOE_FAMILY
        or model_type == QWEN3_MOE_MODEL_TYPE
        or architecture == QWEN3_MOE_ARCHITECTURE
        or model.get("moe") is not None
    )
    if moe_identity and model.get("moe") is None:
        errors.append("Qwen3 MoE plans require complete expert topology facts.")
    try:
        expected_model_architecture_contract(model)
    except ValueError as error:
        errors.append(str(error))

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
                isinstance(memory.get(key), int)
                and not isinstance(memory[key], bool)
                and memory[key] >= 0
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
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in bounds.values()
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
                if runtime_id == "mlx-lm" and candidate.get("method") in {
                    "lora",
                    "qlora",
                }:
                    try:
                        recomputed_memory = mlx_memory_breakdown_for_contract(
                            model=model,
                            target=target,
                            candidate=candidate,
                        )
                    except (TypeError, ValueError) as error:
                        errors.append(
                            f"{name} MLX memory could not be recomputed from bound facts: {error}"
                        )
                    else:
                        observed_memory = {
                            key: memory.get(key) for key in recomputed_memory
                        }
                        if observed_memory != recomputed_memory:
                            errors.append(
                                f"{name} MLX memory does not match deterministic recomputation from bound facts."
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
        expected_targets = MODEL_TARGET_MODULES.get(model.get("family"))
        if (
            method != "full"
            and method in METHODS
            and expected_targets is not None
            and candidate.get("target_modules") != expected_targets
        ):
            errors.append(
                f"{name} target modules do not match the exact model-family policy."
            )
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
        if moe_identity:
            reviewed_moe_runtime = (
                method == "qlora"
                and candidate.get("distribution") == "single"
                and runtime_id == "mlx-lm"
                and runtime_backend == "mps"
                and quantization == "mlx-4bit-groupwise"
                and model.get("quantization_bits") == 4
                and model.get("quantization_layout")
                == _reviewed_qwen3_moe_quantization_layout(model.get("layers"))
                and model.get("family") == QWEN3_MOE_FAMILY
                and model_type == QWEN3_MOE_MODEL_TYPE
                and architecture == QWEN3_MOE_ARCHITECTURE
                and isinstance(model.get("moe"), dict)
                and model["moe"].get("shared_expert_intermediate_size") is None
            )
            if not reviewed_moe_runtime and candidate.get("status") != "unsupported":
                errors.append(
                    f"{name} violates the exact single-device MLX-LM QLoRA MoE policy."
                )
            if reviewed_moe_runtime and candidate.get("status") == "feasible":
                errors.append(
                    f"{name} Qwen3 MoE execution must remain conditional pending its measured pilot."
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
