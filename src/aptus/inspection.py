from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .catalog import TARGET_MODULES


Transport = Callable[..., Any]
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# Provider model types are intentionally matched exactly. Prefix matching would
# incorrectly admit MoE and multimodal variants whose module graphs need their
# own catalog entries. The expected target tuple makes aliases fail closed if a
# canonical catalog policy changes without a matching compatibility review.
_DENSE_CAUSAL_LM_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
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
    if TARGET_MODULES.get(catalog_family) != _DENSE_CAUSAL_LM_TARGET_MODULES:
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
        url, headers={"Accept": "application/json", "User-Agent": "aptus/0.2.0"}
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

    raw_architectures = config.get("architectures")
    architectures = (
        [item for item in raw_architectures if isinstance(item, str)]
        if isinstance(raw_architectures, list)
        else []
    )
    architecture = architectures[0] if architectures else config.get("model_type")
    raw_model_type = config.get("model_type")
    family, family_warning = _catalog_family(raw_model_type, architecture)
    if family_warning is not None:
        warnings.append(family_warning)
    card_data = (
        metadata.get("cardData") if isinstance(metadata.get("cardData"), dict) else {}
    )
    license_name = (
        card_data.get("license") or metadata.get("license") or config.get("license")
    )
    facts = {
        "architecture": architecture,
        "architectures": architectures or None,
        "model_type": raw_model_type,
        "family": family,
        "hidden_size": _first(config, "hidden_size", "d_model", "n_embd"),
        "intermediate_size": _first(config, "intermediate_size", "ffn_dim", "n_inner"),
        "layers": _first(config, "num_hidden_layers", "n_layer", "num_layers"),
        "context_length": _first(
            config, "max_position_embeddings", "n_positions", "seq_length"
        ),
        "attention_heads": _first(config, "num_attention_heads", "n_head"),
        "key_value_heads": config.get("num_key_value_heads"),
        "vocab_size": config.get("vocab_size"),
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
    return {
        "status": "ok",
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved.lower(),
        "facts": facts,
        "provenance": provenance,
        "warnings": warnings,
        "explicit_user_facts_required": ["parameters", "training_allowed"],
    }
