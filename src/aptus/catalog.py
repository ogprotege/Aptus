from __future__ import annotations


TARGET_MODULES: dict[str, tuple[str, ...]] = {
    "llama": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    "mistral": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    "gemma": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    "qwen": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
}

MODULE_DIMENSION_FACTORS: dict[str, int] = {
    "q_proj": 2,
    "k_proj": 2,
    "v_proj": 2,
    "o_proj": 2,
    "gate_proj": 5,
    "up_proj": 5,
    "down_proj": 5,
}

METHOD_EVIDENCE: dict[str, tuple[str, ...]] = {
    "lora": (
        "LoRA research prior: Hu et al., arXiv:2106.09685.",
        "Aptus heuristic-v1 memory model; not empirically calibrated.",
    ),
    "qlora": (
        "QLoRA method prior: Dettmers et al., arXiv:2305.14314.",
        "NF4 plus double-quantization metadata modeled from the paper.",
        "Aptus heuristic-v1 memory model; not empirically calibrated.",
    ),
}

BUNDLE_DEPENDENCY_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "5.14.1",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.49.2",
}


def bundle_requirements(method: str) -> tuple[str, ...]:
    packages = ["torch", "transformers", "peft", "accelerate"]
    if method == "qlora":
        packages.append("bitsandbytes")
    elif method != "lora":
        raise ValueError(f"Unsupported bundle method: {method}")
    return tuple(
        f"{package}=={BUNDLE_DEPENDENCY_VERSIONS[package]}"
        for package in packages
    )


def target_modules_for(family: str) -> tuple[str, ...]:
    normalized = family.lower()
    if normalized not in TARGET_MODULES:
        raise ValueError(
            f"Unsupported model family '{family}'. Supported families: "
            f"{', '.join(sorted(TARGET_MODULES))}."
        )
    return TARGET_MODULES[normalized]
