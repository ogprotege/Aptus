from __future__ import annotations

from typing import Any


def require_trainable_parameter_census(value: Any, *, method: str) -> dict[str, Any]:
    """Validate trainable-set evidence before a host trusts an attestation."""
    if not isinstance(value, dict):
        raise ValueError("Metrics do not contain a trainable-parameter census.")
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
        raise ValueError(
            "Trainable-parameter census violates the selected method scope."
        )
    if value.get("all_values_finite") is not True:
        raise ValueError("Trainable-parameter census does not attest finite values.")
    for name in ("trainable_parameter_count", "trainable_tensor_count"):
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"Trainable-parameter census requires positive {name}.")
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
            raise ValueError(
                f"Trainable-parameter census requires non-negative integer {name}."
            )
    if method == "full":
        if any(value[name] != 0 for name in counter_names):
            raise ValueError(
                "Full fine-tuning census cannot contain frozen or adapter counters."
            )
    else:
        for name in ("frozen_parameter_count", "frozen_tensor_count"):
            if value[name] <= 0:
                raise ValueError(
                    f"Adapter census requires positive {name} for its frozen base."
                )
        if value["unexpected_trainable_tensor_count"] != 0:
            raise ValueError("Adapter census contains an unexpected trainable tensor.")
        if (
            value["expected_adapter_target_match_count"] <= 0
            or value["adapter_target_instance_count"]
            != value["expected_adapter_target_match_count"]
            or value["incomplete_adapter_target_instance_count"] != 0
        ):
            raise ValueError(
                "Adapter census does not bind one complete LoRA A/B pair to every target instance."
            )
    digest = value.get("descriptor_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Trainable-parameter census has an invalid descriptor digest.")
    return value
