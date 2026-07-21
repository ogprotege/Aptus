from __future__ import annotations

import math

from .catalog import METHOD_EVIDENCE, MODULE_DIMENSION_FACTORS, target_modules_for
from .domain import (
    Backend,
    CandidatePlan,
    DatasetProfile,
    HardwareSpec,
    MemoryBreakdown,
    Method,
    ModelSpec,
    Objective,
    TrainingPlan,
    TrainingTarget,
    gibibytes,
)


class NoFeasiblePlanError(ValueError):
    def __init__(self, candidates: tuple[CandidatePlan, ...]) -> None:
        self.candidates = candidates
        reasons = sorted(
            {
                reason
                for candidate in candidates
                for reason in candidate.rejection_reasons
            }
        )
        super().__init__(
            "No feasible LoRA or QLoRA plan: " + "; ".join(reasons)
        )


def _adapter_parameter_count(
    model: ModelSpec,
    rank: int,
    target_modules: tuple[str, ...],
) -> int:
    dimension_factor = sum(
        MODULE_DIMENSION_FACTORS[module] for module in target_modules
    )
    return model.layers * model.hidden_size * rank * dimension_factor


def _rank_prior(dataset: DatasetProfile, objective: Objective) -> int:
    if objective == Objective.MEMORY:
        return 8
    if dataset.total_estimated_tokens >= 1_000_000:
        return 32
    return 16


def _memory_breakdown(
    *,
    method: Method,
    model: ModelSpec,
    target: TrainingTarget,
    target_modules: tuple[str, ...],
    rank: int,
    micro_batch_size: int,
) -> MemoryBreakdown:
    if method == Method.LORA:
        base_weights = model.parameters * 2
        quantization_metadata = 0
        temporary_rate = 0.08
    else:
        base_weights = round(model.parameters * 0.5)
        quantization_metadata = round(model.parameters * (0.127 / 8))
        temporary_rate = 0.15

    adapter_parameters = _adapter_parameter_count(
        model,
        rank,
        target_modules,
    )
    adapter_weights = adapter_parameters * 4
    adapter_gradients = adapter_parameters * 4
    optimizer_states = adapter_parameters * 8

    activation_factor = 3.0
    activations = round(
        micro_batch_size
        * target.sequence_length
        * model.hidden_size
        * model.layers
        * 2
        * activation_factor
    )
    temporary_overhead = max(
        gibibytes(1),
        round((base_weights + quantization_metadata) * temporary_rate),
    )
    subtotal = (
        base_weights
        + quantization_metadata
        + adapter_weights
        + adapter_gradients
        + optimizer_states
        + activations
        + temporary_overhead
    )
    safety_margin = round(subtotal * 0.15)
    return MemoryBreakdown(
        base_weights_bytes=base_weights,
        quantization_metadata_bytes=quantization_metadata,
        adapter_weights_bytes=adapter_weights,
        adapter_gradients_bytes=adapter_gradients,
        optimizer_states_bytes=optimizer_states,
        activations_bytes=activations,
        temporary_overhead_bytes=temporary_overhead,
        safety_margin_bytes=safety_margin,
    )


def _candidate_score(
    *,
    method: Method,
    objective: Objective,
    peak_bytes: int,
    accumulation_steps: int,
    preferred: Method | None,
) -> float:
    if objective == Objective.QUALITY:
        score = 100 if method == Method.LORA else 90
    elif objective == Objective.MEMORY:
        score = -peak_bytes / gibibytes(1)
    else:
        score = (100 if method == Method.LORA else 86) - accumulation_steps
    if preferred == method and objective != Objective.MEMORY:
        score += 5
    return float(score)


def estimate_candidate(
    *,
    method: Method,
    model: ModelSpec,
    dataset: DatasetProfile,
    hardware: HardwareSpec,
    target: TrainingTarget,
) -> CandidatePlan:
    rejection_reasons: list[str] = []
    assumptions = [
        "Memory model is Aptus heuristic-v1 and is not empirically calibrated.",
        "Gradient checkpointing is enabled.",
        "Peak feasibility uses the smallest per-device usable VRAM.",
        "No tensor, pipeline, or optimizer-state sharding is assumed.",
    ]

    try:
        target_modules = target_modules_for(model.family)
    except ValueError as error:
        target_modules = ()
        rejection_reasons.append(str(error))

    if not hardware.devices:
        rejection_reasons.append("At least one GPU device is required.")
    elif any(device.backend != Backend.CUDA for device in hardware.devices):
        rejection_reasons.append(
            "The first Aptus training bundle supports CUDA GPUs only."
        )
    if target.sequence_length > model.context_length:
        rejection_reasons.append(
            f"Requested sequence length {target.sequence_length} exceeds model "
            f"context length {model.context_length}."
        )
    if dataset.schema_name != "text":
        rejection_reasons.append(
            f"Generated training supports plain text datasets, not "
            f"'{dataset.schema_name}'."
        )
    if method == Method.QLORA and (
        not hardware.devices
        or any(not device.supports_4bit for device in hardware.devices)
    ):
        rejection_reasons.append(
            "QLoRA requires explicit 4-bit support on every selected device."
        )

    precision = (
        "bf16"
        if hardware.devices
        and all(device.supports_bf16 for device in hardware.devices)
        else "fp16"
    )
    quantization = "nf4-double-quant" if method == Method.QLORA else None
    rank = _rank_prior(dataset, target.objective)
    alpha = rank * 2
    learning_rate = 2e-4

    micro_batch_candidates = sorted(
        {
            value
            for value in (8, 4, 2, 1)
            if value <= target.effective_batch_size
            and target.effective_batch_size % value == 0
        },
        reverse=True,
    )
    if not micro_batch_candidates:
        micro_batch_candidates = [1]

    selected_micro_batch = 1
    selected_memory = _memory_breakdown(
        method=method,
        model=model,
        target=target,
        target_modules=target_modules,
        rank=rank,
        micro_batch_size=1,
    )
    memory_fit = False
    for micro_batch_size in micro_batch_candidates:
        memory = _memory_breakdown(
            method=method,
            model=model,
            target=target,
            target_modules=target_modules,
            rank=rank,
            micro_batch_size=micro_batch_size,
        )
        if memory.estimated_peak_bytes <= hardware.limiting_vram_bytes:
            selected_micro_batch = micro_batch_size
            selected_memory = memory
            memory_fit = True
            break

    if not memory_fit:
        rejection_reasons.append(
            "Estimated peak memory does not fit per-device usable VRAM, even "
            "with micro-batch size 1."
        )

    accumulation_steps = math.ceil(
        target.effective_batch_size / selected_micro_batch
    )
    effective_batch_size = selected_micro_batch * accumulation_steps
    feasible = not rejection_reasons
    score = _candidate_score(
        method=method,
        objective=target.objective,
        peak_bytes=selected_memory.estimated_peak_bytes,
        accumulation_steps=accumulation_steps,
        preferred=target.method_preference,
    )

    return CandidatePlan(
        method=method,
        feasible=feasible,
        rejection_reasons=tuple(rejection_reasons),
        precision=precision,
        quantization=quantization,
        micro_batch_size=selected_micro_batch,
        gradient_accumulation_steps=accumulation_steps,
        effective_batch_size=effective_batch_size,
        rank=rank,
        alpha=alpha,
        learning_rate=learning_rate,
        target_modules=target_modules,
        memory=selected_memory,
        preference_score=score,
        confidence="low-until-calibrated",
        assumptions=tuple(assumptions),
        evidence=METHOD_EVIDENCE[method.value],
    )


def plan_training(
    *,
    model: ModelSpec,
    dataset: DatasetProfile,
    hardware: HardwareSpec,
    target: TrainingTarget,
) -> TrainingPlan:
    candidates = tuple(
        estimate_candidate(
            method=method,
            model=model,
            dataset=dataset,
            hardware=hardware,
            target=target,
        )
        for method in (Method.LORA, Method.QLORA)
    )
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        raise NoFeasiblePlanError(candidates)
    recommended = max(feasible, key=lambda candidate: candidate.preference_score)

    warnings = [
        *dataset.warnings,
        (
            "Aptus heuristic-v1 estimates are transparent planning priors, "
            "not calibrated peak-VRAM guarantees."
        ),
    ]
    if hardware.gpu_count > 1:
        warnings.append(
            "This vertical slice plans against one limiting GPU and does not "
            "claim multi-GPU sharding."
        )

    rationale = [
        f"Selected {recommended.method.value} from "
        f"{len(feasible)} feasible candidate(s) for the "
        f"{target.objective.value} objective.",
    ]
    if target.objective == Objective.QUALITY:
        rationale.append(
            "Quality ranking prefers non-quantized LoRA when it fits; QLoRA "
            "remains the lower-memory alternative."
        )
    elif target.objective == Objective.MEMORY:
        rationale.append(
            f"{recommended.method.value} has the lowest estimated peak memory "
            f"at {recommended.memory.estimated_peak_bytes / gibibytes(1):.2f} GiB."
        )
    else:
        rationale.append(
            "Speed ranking prefers lower quantization and accumulation overhead."
        )
    rejected = [
        candidate
        for candidate in candidates
        if not candidate.feasible
    ]
    for candidate in rejected:
        rationale.append(
            f"Rejected {candidate.method.value}: "
            + "; ".join(candidate.rejection_reasons)
        )
    if (
        target.method_preference == recommended.method
        and target.objective != Objective.MEMORY
    ):
        rationale.append(
            f"Applied the explicit {recommended.method.value} method preference "
            "after feasibility checks."
        )

    return TrainingPlan(
        schema_version="aptus.training-plan.v1",
        model=model,
        dataset=dataset,
        hardware=hardware,
        target=target,
        recommended=recommended,
        candidates=candidates,
        warnings=tuple(warnings),
        recommendation_rationale=tuple(rationale),
    )
