from __future__ import annotations

from .domain import Method, TrainingRuntime


TARGET_MODULES: dict[str, tuple[str, ...]] = {
    family: (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    for family in ("llama", "mistral", "gemma", "qwen")
}

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
