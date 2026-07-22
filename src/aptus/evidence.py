from __future__ import annotations

from .domain import EvidenceRecord


EVIDENCE_REGISTRY: dict[str, EvidenceRecord] = {
    "method.full.transformers": EvidenceRecord(
        "method.full.transformers",
        "Full fine-tuning updates all model parameters.",
        "https://huggingface.co/docs/transformers/training",
        "official-documentation",
        "Transformers causal language-model training",
        "documented",
    ),
    "method.lora.paper": EvidenceRecord(
        "method.lora.paper",
        "LoRA injects trainable low-rank matrices while freezing base weights.",
        "https://arxiv.org/abs/2106.09685",
        "research-paper",
        "Method definition; not an Aptus quality guarantee",
        "paper-reported",
        "arXiv:2106.09685",
    ),
    "method.qlora.paper": EvidenceRecord(
        "method.qlora.paper",
        "QLoRA uses a frozen four-bit base with trainable LoRA adapters.",
        "https://arxiv.org/abs/2305.14314",
        "research-paper",
        "NF4 and double-quantized QLoRA method; hardware support still required",
        "paper-reported",
        "arXiv:2305.14314",
    ),
    "method.bitsandbytes.int8": EvidenceRecord(
        "method.bitsandbytes.int8",
        "Bitsandbytes provides eight-bit model loading for supported CUDA environments.",
        "https://huggingface.co/docs/transformers/quantization/bitsandbytes",
        "official-documentation",
        "Transformers bitsandbytes integration",
        "documented",
    ),
    "estimate.memory.v2": EvidenceRecord(
        "estimate.memory.v2",
        "Aptus v2 reports an analytic point estimate and an explicit heuristic upper envelope.",
        "aptus://methodology/memory-envelope-v2",
        "aptus-methodology",
        "Uncalibrated component accounting; exact real-model pilot required",
        "uncalibrated",
    ),
}


def evidence_for(*evidence_ids: str) -> tuple[EvidenceRecord, ...]:
    missing = [item for item in evidence_ids if item not in EVIDENCE_REGISTRY]
    if missing:
        raise KeyError(f"Unknown evidence ID(s): {', '.join(missing)}")
    return tuple(EVIDENCE_REGISTRY[item] for item in evidence_ids)
