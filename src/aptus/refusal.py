"""Operator-facing refusal guidance for plan candidates (presentation only).

Maps free-text ``rejection_reasons`` to stable codes, titles, and changeable
facts without altering plan identity or candidate payloads used for digests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RefusalGuidance:
    """Structured answer to: what failed, why, and what can change."""

    reason_code: str
    title: str
    explanation: str
    changeable_facts: tuple[str, ...]
    operator_actionable: bool
    source_reason: str
    none_in_catalog: bool = False

    def to_primitive(self) -> dict[str, object]:
        return {
            "reason_code": self.reason_code,
            "title": self.title,
            "explanation": self.explanation,
            "changeable_facts": list(self.changeable_facts),
            "operator_actionable": self.operator_actionable,
            "none_in_catalog": self.none_in_catalog,
            "source_reason": self.source_reason,
        }


# (substring match, lowered), code, title, explanation, facts, actionable, none_in_catalog
_RULES: tuple[tuple[str, str, str, str, tuple[str, ...], bool, bool], ...] = (
    (
        "full-parameter fp16",
        "full_fp16",
        "Full FP16 training is closed",
        "Full-parameter training requires BF16 on every participating device. "
        "The FP16 full path is fail-closed because it lacks a verified FP32 master-weight contract.",
        ("hardware.devices[].supports_bf16", "method"),
        True,
        False,
    ),
    (
        "full-parameter fsdp",
        "full_fsdp",
        "Full FSDP is closed",
        "Full-parameter FSDP is outside the verified v0.2 compiler and export matrix.",
        ("distribution", "method"),
        True,
        False,
    ),
    (
        "with fsdp is outside the verified",
        "quantized_fsdp",
        "Quantized FSDP is closed",
        "int8-LoRA and QLoRA with FSDP are outside the verified v0.2 compiler matrix.",
        ("distribution", "method"),
        True,
        False,
    ),
    (
        "requires at least two gpus",
        "multi_gpu_on_single",
        "Multi-GPU placement needs at least two devices",
        "DDP and FSDP rows stay visible when only one GPU is declared, but they are "
        "unsupported until the inventory has two or more devices. Planner support is "
        "not multi-GPU runtime proof.",
        ("hardware.devices", "distribution"),
        True,
        False,
    ),
    (
        "has no registered full compiler",
        "mlx_full",
        "Full fine-tuning is not compiled for this runtime",
        "MLX-LM and PyTorch MPS do not register a full-parameter compiler. Use LoRA or "
        "QLoRA on MLX, or a CUDA full path on CUDA hardware.",
        ("method", "training_runtime", "hardware.backend"),
        True,
        False,
    ),
    (
        "has no registered",
        "runtime_compiler_missing",
        "No compiler for this method and runtime",
        "The selected training runtime has no registered compiler for this method and "
        "compute backend combination.",
        ("method", "training_runtime", "hardware.backend"),
        True,
        False,
    ),
    (
        "usable per-device memory is unknown",
        "unknown_device_free_memory",
        "Free device memory is unknown",
        "Aptus will not treat total VRAM or total unified memory as free. Measure or declare current free memory.",
        ("hardware.devices[].free_vram_bytes", "hardware.host_ram_free_bytes"),
        True,
        False,
    ),
    (
        "host ram free is unknown",
        "unknown_host_ram_free",
        "Free host RAM is unknown",
        "Aptus will not treat total host RAM as free. Measure or declare current host RAM free.",
        ("hardware.host_ram_free_bytes",),
        True,
        False,
    ),
    (
        "free disk is unknown",
        "unknown_disk_free",
        "Free disk is unknown",
        "Aptus will not assume enough staging space when free disk is omitted.",
        ("hardware.disk_free_bytes",),
        True,
        False,
    ),
    (
        "will not invent 4",
        "missing_intermediate_size",
        "MLP adapter width is unknown",
        "intermediate_size is required for MLP adapter targets. Aptus will not invent 4 × hidden_size.",
        ("model.intermediate_size",),
        True,
        False,
    ),
    (
        "even the point estimate exceeds",
        "infeasible_memory",
        "Estimated memory exceeds usable device capacity",
        "The analytic point estimate already exceeds usable per-device memory after reserve.",
        (
            "target.sequence_length",
            "target.effective_batch_size",
            "target.micro_batch_size",
            "method",
            "hardware.vram / free memory",
            "hardware.reserve",
        ),
        True,
        False,
    ),
    (
        "upper envelope exceeds usable",
        "conditional_upper_envelope",
        "Upper memory envelope exceeds capacity",
        "The point estimate fits, but the uncalibrated upper envelope does not. The "
        "candidate remains conditional and pilot-required.",
        (
            "target.sequence_length",
            "target.effective_batch_size",
            "method",
            "hardware free memory",
        ),
        True,
        False,
    ),
    (
        "mlx-lm support is pilot-required",
        "conditional_pilot_required",
        "MLX-LM path is pilot-required",
        "Unified-memory estimates are provisional. A real-model pilot must pass before "
        "confirmed full-duration training.",
        (),
        False,
        True,
    ),
    (
        "fsdp uses a simplified uncalibrated",
        "conditional_fsdp_pilot",
        "LoRA FSDP requires a real-model pilot",
        "FSDP sharding priors are uncalibrated. Analytic fit is not multi-rank proof.",
        (),
        False,
        True,
    ),
    (
        "sequence length exceeds the model context",
        "sequence_length_exceeds_context",
        "Sequence length exceeds model context",
        "The requested sequence length is longer than the model's context length.",
        ("target.sequence_length", "model.context_length"),
        True,
        False,
    ),
    (
        "host ram is below",
        "host_ram_infeasible",
        "Host RAM is below the staging heuristic",
        "Estimated host staging for model load exceeds available host RAM.",
        ("hardware.host_ram", "method", "distribution"),
        True,
        False,
    ),
    (
        "free disk is below",
        "disk_infeasible",
        "Free disk is below the staging estimate",
        "Estimated staging, pilot, checkpoint retention, and export need more free disk.",
        ("hardware.disk_free", "model size", "dataset size"),
        True,
        False,
    ),
    (
        "packing is not implemented",
        "packing_unsupported",
        "Sequence packing is closed",
        "v0.2 does not implement sequence packing. Set packing=false.",
        ("target.packing",),
        True,
        False,
    ),
    (
        "task='sft'",
        "task_not_sft",
        "Only supervised fine-tuning is compiled",
        "Non-SFT tasks are fail-closed in v0.2.",
        ("target.task",),
        True,
        False,
    ),
    (
        "max_wall_time_minutes is fail-closed",
        "max_wall_time_closed",
        "Wall-time limits are closed",
        "Graceful deadline enforcement is not implemented. Omit max_wall_time_minutes.",
        ("target.max_wall_time_minutes",),
        True,
        False,
    ),
    (
        "qwen3 moe is executable only",
        "qwen3_moe_path_only",
        "Qwen3 MoE path is fixed",
        "Only single-device MLX-LM QLoRA with attention-only adapters is executable for "
        "the reviewed Qwen3 MoE contract.",
        ("method", "distribution", "training_runtime", "adapter targets"),
        True,
        False,
    ),
    (
        "qwen3 moe identity was recognized",
        "moe_near_match",
        "Qwen3 MoE near-match is blocked",
        "The MoE identity was recognized but quantization layout or topology does not "
        "match the reviewed contract.",
        ("model quantization layout", "moe topology", "shared-expert flags"),
        True,
        False,
    ),
    (
        "does not match the reviewed",
        "policy_blocked_near_match",
        "Model policy near-match is blocked",
        "Provider identity is recognized but configuration facts fail the reviewed policy.",
        ("model architecture", "layers", "quantization", "topology"),
        True,
        False,
    ),
    (
        "four-bit support on every participating device",
        "qlora_four_bit_device",
        "QLoRA needs four-bit device support",
        "CUDA QLoRA requires explicit four-bit support on every participating device.",
        ("hardware.devices[].supports_4bit", "method"),
        True,
        False,
    ),
    (
        "eight-bit support on every participating",
        "int8_lora_device",
        "int8-LoRA needs eight-bit device support",
        "Every participating GPU must declare eight-bit support.",
        ("hardware.devices[].supports_8bit", "method"),
        True,
        False,
    ),
    (
        "not divisible by world size",
        "batch_not_divisible_world",
        "Global batch is not divisible by world size",
        "Effective batch must divide evenly across the distributed world size.",
        ("target.effective_batch_size", "hardware.gpu_count", "distribution"),
        True,
        False,
    ),
    (
        "registry contract does not support",
        "registry_distribution_unsupported",
        "Method does not support this distribution",
        "The method registry does not list this distribution for the selected method.",
        ("distribution", "method"),
        True,
        False,
    ),
    (
        "compiler does not support",
        "runtime_distribution_unsupported",
        "Runtime compiler does not support this distribution",
        "The bound runtime compiler does not support this placement.",
        ("distribution", "training_runtime"),
        True,
        False,
    ),
    (
        "unsupported model family",
        "unsupported_model_family",
        "Model family has no adapter target catalog",
        "Non-full methods need a known family for target-module inspection.",
        ("model.family", "method"),
        True,
        False,
    ),
    (
        "at least one supported compute device",
        "no_compute_device",
        "No compute device in inventory",
        "Planning requires at least one supported compute device fact.",
        ("hardware.devices",),
        True,
        False,
    ),
    (
        "cannot mix compute backends",
        "mixed_compute_backends",
        "Mixed compute backends",
        "A single candidate cannot span multiple compute backends.",
        ("hardware.devices",),
        True,
        False,
    ),
)


_NONE_CATALOG_MESSAGE = (
    "No supported correction exists in the current Aptus catalog for these facts."
)


def guide_rejection_reason(reason: str) -> RefusalGuidance:
    """Map one free-text rejection reason to structured operator guidance."""

    text = reason.strip()
    if not text:
        return RefusalGuidance(
            reason_code="empty_reason",
            title="Unspecified refusal",
            explanation="The planner returned an empty rejection reason.",
            changeable_facts=(),
            operator_actionable=False,
            source_reason=reason,
            none_in_catalog=True,
        )
    lowered = text.lower()
    for (
        needle,
        code,
        title,
        explanation,
        facts,
        actionable,
        none_in_catalog,
    ) in _RULES:
        if needle in lowered:
            return RefusalGuidance(
                reason_code=code,
                title=title,
                explanation=explanation,
                changeable_facts=() if none_in_catalog else facts,
                operator_actionable=actionable,
                source_reason=text,
                none_in_catalog=none_in_catalog,
            )
    return RefusalGuidance(
        reason_code="unclassified_reason",
        title="Planner refused this path",
        explanation=text,
        changeable_facts=(),
        operator_actionable=False,
        source_reason=text,
        none_in_catalog=True,
    )


def guide_rejection_reasons(reasons: Sequence[str]) -> tuple[RefusalGuidance, ...]:
    return tuple(guide_rejection_reason(reason) for reason in reasons)


def format_guidance_lines(guidance: RefusalGuidance) -> list[str]:
    """CLI/UI multi-line presentation answering what / why / what can change."""

    lines = [
        f"What: {guidance.title}",
        f"Why: {guidance.explanation}",
    ]
    if guidance.none_in_catalog or not guidance.changeable_facts:
        lines.append(f"What can change: {_NONE_CATALOG_MESSAGE}")
    else:
        lines.append(
            "What can change: " + ", ".join(guidance.changeable_facts),
        )
    lines.append(f"Detail: {guidance.source_reason}")
    return lines


def format_candidate_refusal_block(
    *,
    status: str,
    reasons: Iterable[str],
) -> str:
    """Render a multi-reason refusal block for CLI output."""

    items = list(reasons)
    if not items:
        if status == "conditional":
            return (
                "Status: conditional (pilot-required). No hard rejection strings; "
                "runtime evidence is still required before a measured run claim."
            )
        return ""
    chunks: list[str] = [f"Status: {status}"]
    for index, reason in enumerate(items, start=1):
        guided = guide_rejection_reason(reason)
        chunks.append(f"[{index}] " + " | ".join(format_guidance_lines(guided)))
    return "\n".join(chunks)


def enrich_candidate_presentation(candidate: dict[str, object]) -> dict[str, object]:
    """Attach presentation-only refusal_guidance without mutating identity fields."""

    reasons = candidate.get("rejection_reasons")
    if not isinstance(reasons, list):
        return candidate
    guided = [item.to_primitive() for item in guide_rejection_reasons(reasons)]
    enriched = dict(candidate)
    enriched["refusal_guidance"] = guided
    return enriched
