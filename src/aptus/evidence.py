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
    "method.dora.paper": EvidenceRecord(
        "method.dora.paper",
        "DoRA decomposes pretrained weights into magnitude and direction and applies a low-rank update to the directional component.",
        "https://arxiv.org/abs/2402.09353",
        "research-paper",
        "Method definition; not evidence that an Aptus runtime path is implemented or calibrated",
        "paper-reported",
        "arXiv:2402.09353",
    ),
    "method.bitfit.paper": EvidenceRecord(
        "method.bitfit.paper",
        "BitFit freezes a pretrained transformer and updates only its existing bias terms.",
        "https://arxiv.org/abs/2106.10199",
        "research-paper",
        "Method definition and masked-language-model experiments; applicability depends on the exact architecture exposing biases",
        "paper-reported",
        "arXiv:2106.10199",
    ),
    "method.loreft.paper": EvidenceRecord(
        "method.loreft.paper",
        "LoReFT learns low-rank interventions on hidden representations rather than weight adapters.",
        "https://arxiv.org/abs/2404.03592",
        "research-paper",
        "Method definition; requires an intervention-aware runtime, collator, checkpoint, and export contract",
        "paper-reported",
        "arXiv:2404.03592",
    ),
    "method.aflora.paper": EvidenceRecord(
        "method.aflora.paper",
        "AFLoRA dynamically evaluates and freezes low-rank parameter groups during training.",
        "https://arxiv.org/abs/2403.13269",
        "research-paper",
        "Method definition; dynamic optimizer and checkpoint state are not implemented by Aptus",
        "paper-reported",
        "arXiv:2403.13269",
    ),
    "method.bilora.paper": EvidenceRecord(
        "method.bilora.paper",
        "BiLoRA uses bilevel optimization over disjoint data partitions with a pseudo-SVD low-rank parameterization.",
        "https://arxiv.org/abs/2403.13037",
        "research-paper",
        "Method definition; requires a dedicated bilevel trainer and partition contract",
        "paper-reported",
        "arXiv:2403.13037",
    ),
    "method.adalora.paper": EvidenceRecord(
        "method.adalora.paper",
        "AdaLoRA allocates a changing rank budget across pseudo-SVD parameter groups using importance scores and a pruning schedule.",
        "https://arxiv.org/abs/2303.10512",
        "research-paper",
        "Method definition and reported experiments; Aptus still requires a pinned adaptive-budget compiler and restart contract",
        "paper-reported",
        "arXiv:2303.10512",
    ),
    "method.sharelora.paper": EvidenceRecord(
        "method.sharelora.paper",
        "ShareLoRA shares one or both low-rank factors across compatible layers.",
        "https://arxiv.org/abs/2406.10785",
        "research-paper",
        "Method definition; serialization and distributed synchronization require independent verification",
        "paper-reported",
        "arXiv:2406.10785",
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
    "policy.qwen3-moe.mlx-qlora.v1": EvidenceRecord(
        "policy.qwen3-moe.mlx-qlora.v1",
        "Aptus defines one exact Qwen3 MoE MLX-LM QLoRA path as eligible for gated validation.",
        "aptus://operations/evidence/2026-07-28-qwen3-moe-admission",
        "aptus-compatibility-policy",
        "Exact qwen3_moe and Qwen3MoeForCausalLM identity, reviewed mixed-bit layout, routed topology, and attention-only adapter scope",
        "implementation-reviewed",
        "1.0.0",
    ),
    "admission.qwen3-30b-a3b.memory-blocked.2026-07-28": EvidenceRecord(
        "admission.qwen3-30b-a3b.memory-blocked.2026-07-28",
        "The exact 30B target-host attempt passed dependency validation and then stopped before model loading at live unified-memory admission.",
        "aptus://operations/evidence/2026-07-28-qwen3-moe-admission",
        "measured-admission-record",
        "Exact pinned Qwen3 30B-A3B revision on the recorded Apple Silicon host; not a passing model-data or pilot result",
        "measured-blocked",
        "e9675aa3ca5f900ccef55267914466d55ab325fa",
    ),
}


def evidence_for(*evidence_ids: str) -> tuple[EvidenceRecord, ...]:
    missing = [item for item in evidence_ids if item not in EVIDENCE_REGISTRY]
    if missing:
        raise KeyError(f"Unknown evidence ID(s): {', '.join(missing)}")
    return tuple(EVIDENCE_REGISTRY[item] for item in evidence_ids)
