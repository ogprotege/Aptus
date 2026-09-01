from __future__ import annotations

from .domain import (
    Method,
    QuantizationLayout,
    QuantizationOverride,
    TrainingRuntime,
)


QWEN3_MOE_FAMILY = "qwen3_moe"
QWEN3_MOE_MODEL_TYPE = "qwen3_moe"
QWEN3_MOE_ARCHITECTURE = "Qwen3MoeForCausalLM"
QWEN3_MOE_QUANTIZATION_PROFILE = "qwen3-moe-4bit-group64-router-gates-8bit"
GEMMA4_MOE_FAMILY = "gemma4_moe"
GEMMA4_MOE_MODEL_TYPE = "gemma4_text"
GEMMA4_MOE_ARCHITECTURE = "Gemma4ForConditionalGeneration"
DENSE_CAUSAL_LM_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
QWEN3_MOE_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
GEMMA4_MOE_TARGET_MODULES = QWEN3_MOE_TARGET_MODULES


def reviewed_qwen3_moe_quantization_layout(layers: int) -> QuantizationLayout:
    """Return the exact quantization map reviewed for the pinned MLX artifact."""

    if not isinstance(layers, int) or isinstance(layers, bool) or layers <= 0:
        raise ValueError("Reviewed Qwen3 MoE quantization requires positive layers.")
    return QuantizationLayout(
        default_bits=4,
        default_group_size=64,
        module_overrides=tuple(
            QuantizationOverride(
                module_path=f"model.layers.{index}.mlp.gate",
                bits=8,
                group_size=64,
            )
            for index in sorted(range(layers), key=lambda value: str(value))
        ),
    )


def reviewed_gemma4_moe_quantization_layout(layers: int) -> QuantizationLayout:
    """Return the exact 4-bit plus 8-bit router.proj map for Gemma 4 MoE."""

    if not isinstance(layers, int) or isinstance(layers, bool) or layers <= 0:
        raise ValueError("Reviewed Gemma 4 MoE quantization requires positive layers.")
    return QuantizationLayout(
        default_bits=4,
        default_group_size=64,
        module_overrides=tuple(
            QuantizationOverride(
                module_path=f"model.layers.{index}.router.proj",
                bits=8,
                group_size=64,
            )
            for index in sorted(range(layers), key=lambda value: str(value))
        ),
    )


TARGET_MODULES: dict[str, tuple[str, ...]] = {
    family: DENSE_CAUSAL_LM_TARGET_MODULES
    for family in ("llama", "mistral", "gemma", "gemma4", "qwen")
}
TARGET_MODULES[QWEN3_MOE_FAMILY] = QWEN3_MOE_TARGET_MODULES
TARGET_MODULES[GEMMA4_MOE_FAMILY] = GEMMA4_MOE_TARGET_MODULES

MODULE_DIMENSION_FACTORS = {
    "q_proj": 2.0,
    "k_proj": 2.0,
    "v_proj": 2.0,
    "o_proj": 2.0,
    "gate_proj": 5.0,
    "up_proj": 5.0,
    "down_proj": 5.0,
}

STACK_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "5.14.1",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.49.2",
    "safetensors": "0.8.0",
    "mlx": "0.31.2",
    "mlx-lm": "0.31.3",
}


def bundle_requirements(
    method: str | Method,
    training_runtime: str | TrainingRuntime = TrainingRuntime.TRANSFORMERS_PEFT_CUDA,
) -> tuple[str, ...]:
    normalized = Method(method)
    runtime = TrainingRuntime(training_runtime)
    if runtime == TrainingRuntime.MLX_LM:
        if normalized not in {Method.LORA, Method.QLORA}:
            raise ValueError("MLX-LM bundles currently compile LoRA and QLoRA only.")
        return tuple(f"{name}=={STACK_VERSIONS[name]}" for name in ("mlx", "mlx-lm"))
    packages = ["torch", "transformers", "accelerate", "safetensors"]
    if normalized != Method.FULL:
        packages.append("peft")
    if runtime == TrainingRuntime.TRANSFORMERS_PEFT_CUDA and normalized in {
        Method.INT8_LORA,
        Method.QLORA,
    }:
        packages.append("bitsandbytes")
    return tuple(f"{name}=={STACK_VERSIONS[name]}" for name in packages)


def target_modules_for(family: str) -> tuple[str, ...]:
    normalized = family.lower()
    if normalized not in TARGET_MODULES:
        raise ValueError(
            f"Unsupported model family '{family}'. Supported families: {', '.join(sorted(TARGET_MODULES))}."
        )
    return TARGET_MODULES[normalized]


def is_exact_qwen3_moe(
    *, family: str, model_type: str | None, architecture: str
) -> bool:
    """Return true only for the reviewed Qwen3 MoE provider identity."""

    return (
        family.lower() == QWEN3_MOE_FAMILY
        and model_type == QWEN3_MOE_MODEL_TYPE
        and architecture == QWEN3_MOE_ARCHITECTURE
    )


def has_reviewed_qwen3_moe_quantization_layout(
    quantization_layout: QuantizationLayout | None,
    *,
    layers: int,
) -> bool:
    """Return true only for the reviewed MLX mixed-precision layout."""

    return quantization_layout == reviewed_qwen3_moe_quantization_layout(layers)


def is_exact_gemma4_moe(
    *, family: str, model_type: str | None, architecture: str
) -> bool:
    """Return true only for the reviewed Gemma 4 MoE provider identity."""

    return (
        family.lower() == GEMMA4_MOE_FAMILY
        and model_type == GEMMA4_MOE_MODEL_TYPE
        and architecture == GEMMA4_MOE_ARCHITECTURE
    )


def has_reviewed_gemma4_moe_quantization_layout(
    quantization_layout: QuantizationLayout | None,
    *,
    layers: int,
) -> bool:
    """Return true only for the reviewed Gemma 4 MoE mixed-precision layout."""

    return quantization_layout == reviewed_gemma4_moe_quantization_layout(layers)
