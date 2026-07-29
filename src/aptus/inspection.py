from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from . import __version__
from .catalog import (
    DENSE_CAUSAL_LM_TARGET_MODULES,
    QWEN3_MOE_FAMILY,
    QWEN3_MOE_MODEL_TYPE,
    TARGET_MODULES,
)
from .domain import (
    ModelCompatibilitySubject,
    MoETopology,
    QuantizationLayout,
    QuantizationOverride,
)
from .model_compatibility import (
    compatibility_response_v1,
    evaluate_model_compatibility,
)


Transport = Callable[..., Any]
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# Provider model types are intentionally matched exactly. Prefix matching would
# incorrectly admit MoE and multimodal variants whose module graphs need their
# own catalog entries. The expected target tuple makes aliases fail closed if a
# canonical catalog policy changes without a matching compatibility review.
_PROVIDER_MODEL_TYPE_ALIASES = {
    "qwen2": "qwen",
    "qwen3": "qwen",
    "gemma2": "gemma",
    "gemma3": "gemma",
    "gemma3_text": "gemma",
}
_TEXT_ONLY_GEMMA3_ARCHITECTURES = {
    "Gemma3ForCausalLM",
    "Gemma3TextModel",
}


def _catalog_family(model_type: Any, architecture: Any) -> tuple[Any, str | None]:
    """Map an exact provider type only when its catalog module policy still matches."""

    if not isinstance(model_type, str) or not model_type.strip():
        return model_type, None
    raw_model_type = model_type.strip()
    normalized = raw_model_type.lower()
    if normalized == QWEN3_MOE_MODEL_TYPE:
        return QWEN3_MOE_FAMILY, (
            "Provider model_type 'qwen3_moe' was routed to exact Aptus model-policy "
            "evaluation. Raw provider identifiers remain in the model_type and "
            "architecture evidence fields."
        )
    if normalized in TARGET_MODULES:
        return normalized, None

    catalog_family = _PROVIDER_MODEL_TYPE_ALIASES.get(normalized)
    if catalog_family is None:
        return raw_model_type, None
    if normalized == "gemma3" and architecture not in _TEXT_ONLY_GEMMA3_ARCHITECTURES:
        return raw_model_type, (
            "Provider model_type 'gemma3' was not mapped to catalog family "
            f"'gemma' because architecture {architecture!r} is not an explicitly "
            "supported text-only Gemma 3 architecture."
        )
    if TARGET_MODULES.get(catalog_family) != DENSE_CAUSAL_LM_TARGET_MODULES:
        return raw_model_type, (
            f"Provider model_type '{raw_model_type}' was not normalized because "
            f"the '{catalog_family}' target-module catalog policy has changed."
        )
    return catalog_family, (
        f"Provider model_type '{raw_model_type}' was normalized to Aptus catalog "
        f"family '{catalog_family}'. Raw provider identifiers remain in the "
        "model_type and architecture evidence fields."
    )


def _fetch_json(
    url: str, *, timeout: float, transport: Transport | None
) -> tuple[dict[str, Any], str | None]:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": f"aptus/{__version__}"},
    )
    response = (transport or urlopen)(request, timeout=timeout)
    with response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_RESPONSE_BYTES:
            raise ValueError("Provider response exceeds the Aptus inspection bound.")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ValueError("Provider response exceeds the Aptus inspection bound.")
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Provider returned a non-object JSON document.")
        return value, response.headers.get("X-Repo-Commit") or response.headers.get(
            "x-repo-commit"
        )


def _first(config: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = config.get(name)
        if value is not None:
            return value
    return None


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("text_config")
    return value if isinstance(value, dict) else config


def _quantization_bits(config: dict[str, Any]) -> Any:
    text = _text_config(config)
    quantization = (
        text.get("quantization")
        or text.get("quantization_config")
        or config.get("quantization")
        or config.get("quantization_config")
    )
    return quantization.get("bits") if isinstance(quantization, dict) else None


def _canonical_quantization_mapping(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    bits = value.get("bits")
    group_size = value.get("group_size")
    if (
        not isinstance(bits, int)
        or isinstance(bits, bool)
        or not 1 <= bits <= 16
        or not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size <= 0
    ):
        return None, (
            "Provider quantization metadata requires integer bits from 1 through "
            "16 and a positive integer group_size."
        )
    overrides: list[dict[str, Any]] = []
    for module_path, override in value.items():
        if module_path in {"bits", "group_size"}:
            continue
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
            return None, (
                "Provider quantization metadata contains an invalid module path."
            )
        if not isinstance(override, Mapping) or set(override) != {
            "bits",
            "group_size",
        }:
            return None, (
                "Provider quantization metadata contains an unsupported field or "
                f"module override at {module_path!r}."
            )
        override_bits = override.get("bits")
        override_group_size = override.get("group_size")
        if (
            not isinstance(override_bits, int)
            or isinstance(override_bits, bool)
            or not 1 <= override_bits <= 16
            or not isinstance(override_group_size, int)
            or isinstance(override_group_size, bool)
            or override_group_size <= 0
        ):
            return None, (
                f"Provider quantization override {module_path!r} has invalid bits "
                "or group_size."
            )
        overrides.append(
            {
                "module_path": module_path,
                "bits": override_bits,
                "group_size": override_group_size,
            }
        )
    return {
        "default_bits": bits,
        "default_group_size": group_size,
        "module_overrides": sorted(overrides, key=lambda item: item["module_path"]),
    }, None


def _quantization_layout(
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    text = _text_config(config)
    candidates: list[Mapping[str, Any]] = []
    for container in (text, config):
        for name in ("quantization", "quantization_config"):
            value = container.get(name)
            if isinstance(value, Mapping) and all(
                value is not item for item in candidates
            ):
                candidates.append(value)
    if not candidates:
        return None, None
    layouts: list[dict[str, Any]] = []
    for candidate in candidates:
        layout, error = _canonical_quantization_mapping(candidate)
        if error is not None:
            return None, error
        assert layout is not None
        layouts.append(layout)
    if any(layout != layouts[0] for layout in layouts[1:]):
        return None, (
            "Provider quantization and quantization_config metadata disagree."
        )
    return layouts[0], None


def _moe_facts(config: dict[str, Any]) -> dict[str, Any] | None:
    text = _text_config(config)
    names = (
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "decoder_sparse_step",
        "mlp_only_layers",
        "shared_expert_intermediate_size",
    )
    if not any(name in text for name in names):
        return None
    return {
        "expert_count": text.get("num_experts"),
        "experts_per_token": text.get("num_experts_per_tok"),
        "expert_intermediate_size": text.get("moe_intermediate_size"),
        "decoder_sparse_step": text.get("decoder_sparse_step"),
        "mlp_only_layers": text.get("mlp_only_layers"),
        "shared_expert_intermediate_size": text.get("shared_expert_intermediate_size"),
    }


def _moe_topology_error(moe: dict[str, Any] | None, layers: Any) -> str | None:
    if not isinstance(moe, dict):
        return "Provider MoE topology metadata is missing."
    positive_names = (
        "expert_count",
        "experts_per_token",
        "expert_intermediate_size",
        "decoder_sparse_step",
    )
    if any(
        not isinstance(moe.get(name), int)
        or isinstance(moe.get(name), bool)
        or moe[name] <= 0
        for name in positive_names
    ):
        return "Provider MoE topology integer facts must be positive integers."
    if moe["experts_per_token"] > moe["expert_count"]:
        return "Provider experts_per_token cannot exceed expert_count."
    if not isinstance(layers, int) or isinstance(layers, bool) or layers <= 0:
        return "Provider MoE topology requires a positive model layer count."
    mlp_only = moe.get("mlp_only_layers")
    if (
        not isinstance(mlp_only, list)
        or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= layers
            for index in mlp_only
        )
        or mlp_only != sorted(set(mlp_only))
    ):
        return (
            "Provider mlp_only_layers must be sorted, unique, non-negative, and "
            "within the model layer count."
        )
    sparse_layers = sum(
        1
        for index in range(layers)
        if (index + 1) % moe["decoder_sparse_step"] == 0 and index not in set(mlp_only)
    )
    if sparse_layers <= 0:
        return "Provider MoE topology declares no executable sparse layer."
    shared = moe.get("shared_expert_intermediate_size")
    if shared is not None and (
        not isinstance(shared, int) or isinstance(shared, bool) or shared <= 0
    ):
        return "Provider shared expert width must be a positive integer when present."
    return None


def _compatibility_subject(
    *,
    family: Any,
    model_type: Any,
    architecture: Any,
    layers: Any,
    quantization_bits: Any,
    quantization_layout: dict[str, Any] | None,
    quantization_error: str | None,
    moe: dict[str, Any] | None,
    moe_error: str | None,
) -> ModelCompatibilitySubject:
    fact_errors: list[str] = []
    layout_value: QuantizationLayout | None = None
    if quantization_error is not None:
        fact_errors.append(f"quantization: {quantization_error}")
    elif quantization_layout is not None:
        try:
            layout_value = QuantizationLayout(
                default_bits=quantization_layout["default_bits"],
                default_group_size=quantization_layout["default_group_size"],
                module_overrides=tuple(
                    QuantizationOverride(
                        module_path=item["module_path"],
                        bits=item["bits"],
                        group_size=item["group_size"],
                    )
                    for item in quantization_layout["module_overrides"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            fact_errors.append(f"quantization: {error}")

    moe_value: MoETopology | None = None
    if moe_error is not None:
        fact_errors.append(f"moe: {moe_error}")
    elif moe is not None:
        try:
            moe_value = MoETopology(
                expert_count=moe["expert_count"],
                experts_per_token=moe["experts_per_token"],
                expert_intermediate_size=moe["expert_intermediate_size"],
                decoder_sparse_step=moe["decoder_sparse_step"],
                mlp_only_layers=tuple(moe["mlp_only_layers"]),
                shared_expert_intermediate_size=moe.get(
                    "shared_expert_intermediate_size"
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            fact_errors.append(f"moe: {error}")

    def exact_text(value: Any) -> str | None:
        return (
            value
            if isinstance(value, str) and value.strip() and value == value.strip()
            else None
        )

    return ModelCompatibilitySubject(
        family=exact_text(family),
        model_type=exact_text(model_type),
        architecture=exact_text(architecture),
        layers=(
            layers
            if isinstance(layers, int) and not isinstance(layers, bool) and layers > 0
            else None
        ),
        quantization_bits=(
            quantization_bits
            if isinstance(quantization_bits, int)
            and not isinstance(quantization_bits, bool)
            and 1 <= quantization_bits <= 16
            else None
        ),
        quantization_layout=layout_value,
        moe=moe_value,
        fact_errors=tuple(fact_errors),
    )


def inspect_huggingface_model(
    model_id: str,
    revision: str,
    *,
    timeout: float = 10.0,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """Resolve bounded provider-declared facts without guessing permission or size."""

    if not model_id.strip() or not revision.strip():
        raise ValueError("model_id and revision are required.")
    if not 0 < timeout <= 30:
        raise ValueError("timeout must be in (0, 30].")
    model_path = quote(model_id.strip(), safe="/")
    revision_path = quote(revision.strip(), safe="")
    config_url = (
        f"https://huggingface.co/{model_path}/resolve/{revision_path}/config.json"
    )
    metadata_url = (
        f"https://huggingface.co/api/models/{model_path}/revision/{revision_path}"
    )
    observed_at = datetime.now(timezone.utc).isoformat()
    try:
        config, commit_header = _fetch_json(
            config_url, timeout=timeout, transport=transport
        )
    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {
            "status": "unavailable",
            "model_id": model_id,
            "requested_revision": revision,
            "error": str(error),
            "source": config_url,
        }

    resolved = commit_header or config.get("_commit_hash")
    if (
        resolved is None
        and 40 <= len(revision) <= 64
        and all(character in "0123456789abcdefABCDEF" for character in revision)
    ):
        resolved = revision.lower()
    if (
        not isinstance(resolved, str)
        or not (40 <= len(resolved) <= 64)
        or any(character not in "0123456789abcdefABCDEF" for character in resolved)
    ):
        return {
            "status": "unsupported",
            "model_id": model_id,
            "requested_revision": revision,
            "error": "Provider did not bind config.json to an immutable commit.",
            "source": config_url,
        }

    metadata: dict[str, Any] = {}
    warnings: list[str] = []
    try:
        metadata, metadata_commit = _fetch_json(
            metadata_url, timeout=timeout, transport=transport
        )
        if metadata_commit and metadata_commit.lower() != resolved.lower():
            warnings.append(
                "Metadata response resolved to a different commit; its license field was ignored."
            )
            metadata = {}
    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        warnings.append(f"Provider metadata was unavailable: {error}")

    text_config = _text_config(config)
    raw_architectures = config.get("architectures") or text_config.get("architectures")
    architectures = (
        [item for item in raw_architectures if isinstance(item, str)]
        if isinstance(raw_architectures, list)
        else []
    )
    raw_model_type = text_config.get("model_type") or config.get("model_type")
    architecture = architectures[0] if architectures else raw_model_type
    family, family_warning = _catalog_family(raw_model_type, architecture)
    if family_warning is not None:
        warnings.append(family_warning)
    card_data = (
        metadata.get("cardData") if isinstance(metadata.get("cardData"), dict) else {}
    )
    license_name = (
        card_data.get("license") or metadata.get("license") or config.get("license")
    )
    moe = _moe_facts(config)
    layers = _first(text_config, "num_hidden_layers", "n_layer", "num_layers")
    moe_error = _moe_topology_error(moe, layers) if moe is not None else None
    if moe_error is not None:
        warnings.append(moe_error)
    quantization_bits = _quantization_bits(config)
    quantization_layout, quantization_error = _quantization_layout(config)
    if quantization_error is not None:
        warnings.append(quantization_error)
    facts = {
        "architecture": architecture,
        "architectures": architectures or None,
        "model_type": raw_model_type,
        "family": family,
        "hidden_size": _first(text_config, "hidden_size", "d_model", "n_embd"),
        "intermediate_size": _first(
            text_config, "intermediate_size", "ffn_dim", "n_inner"
        ),
        "layers": layers,
        "context_length": _first(
            text_config, "max_position_embeddings", "n_positions", "seq_length"
        ),
        "attention_heads": _first(text_config, "num_attention_heads", "n_head"),
        "key_value_heads": text_config.get("num_key_value_heads"),
        "vocab_size": text_config.get("vocab_size"),
        "quantization_bits": quantization_bits,
        "quantization_layout": quantization_layout,
        "moe": moe,
        "license_name": license_name,
        "parameters": None,
        "training_allowed": None,
    }
    provenance = {
        key: {
            "kind": "provider-declared",
            "source": config_url if key != "license_name" else metadata_url,
            "observed_at": observed_at,
            "resolved_revision": resolved.lower(),
        }
        for key, value in facts.items()
        if value is not None
    }
    if family is not None and family != raw_model_type:
        provenance["family"] = {
            "kind": "inferred",
            "source": f"Aptus exact model-type compatibility mapping of {config_url}",
            "observed_at": observed_at,
            "resolved_revision": resolved.lower(),
        }
    compatibility_subject = _compatibility_subject(
        family=family,
        model_type=raw_model_type,
        architecture=architecture,
        layers=facts["layers"],
        quantization_bits=quantization_bits,
        quantization_layout=quantization_layout,
        quantization_error=quantization_error,
        moe=moe,
        moe_error=moe_error,
    )
    return {
        "status": "ok",
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved.lower(),
        "facts": facts,
        "provenance": provenance,
        "warnings": warnings,
        "compatibility": compatibility_response_v1(
            evaluate_model_compatibility(compatibility_subject)
        ),
        "explicit_user_facts_required": ["parameters", "training_allowed"],
    }
