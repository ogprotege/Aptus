from __future__ import annotations

from .domain import Method


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
}


def bundle_requirements(method: str | Method) -> tuple[str, ...]:
    normalized = Method(method)
    packages = ["torch", "transformers", "accelerate", "safetensors"]
    if normalized != Method.FULL:
        packages.append("peft")
    if normalized in {Method.INT8_LORA, Method.QLORA}:
        packages.append("bitsandbytes")
    return tuple(f"{name}=={STACK_VERSIONS[name]}" for name in packages)


def target_modules_for(family: str) -> tuple[str, ...]:
    normalized = family.lower()
    if normalized not in TARGET_MODULES:
        raise ValueError(
            f"Unsupported model family '{family}'. Supported families: {', '.join(sorted(TARGET_MODULES))}."
        )
    return TARGET_MODULES[normalized]
