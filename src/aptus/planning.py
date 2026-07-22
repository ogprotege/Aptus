from __future__ import annotations

import math
from dataclasses import replace

from .catalog import METHOD_EVIDENCE, target_modules_for
from .domain import (
    Backend,
    CandidatePlan,
    CandidateStatus,
    DatasetProfile,
    DeviceSpec,
    Distribution,
    HardwareSpec,
    MemoryBreakdown,
    Method,
    ModelSpec,
    Objective,
    SCHEMA_VERSION,
    TrainingPlan,
    TrainingTarget,
    gibibytes,
    to_primitive,
)
from .evidence import evidence_for
from .plan_contract import candidate_id_for_payload, plan_id_for_payload


FORMULA_VERSION = "aptus-memory-v2"


class NoFeasiblePlanError(ValueError):
    def __init__(self, candidates: tuple[CandidatePlan, ...]) -> None:
        self.candidates = candidates
        reasons = sorted(
            {reason for item in candidates for reason in item.rejection_reasons}
        )
        super().__init__(
            "No feasible or conditional training plan: " + "; ".join(reasons)
        )


def _adapter_parameter_count(
    model: ModelSpec, rank: int, modules: tuple[str, ...]
) -> int:
    intermediate = model.intermediate_size or model.hidden_size * 4
    per_layer = 0
    for module in modules:
        if module in {"gate_proj", "up_proj", "down_proj"}:
            per_layer += model.hidden_size + intermediate
        else:
            per_layer += model.hidden_size * 2
    return model.layers * rank * per_layer


def _rank_prior(dataset: DatasetProfile, objective: Objective) -> int:
    if objective == Objective.MEMORY:
        return 8
    if dataset.total_estimated_tokens >= 1_000_000:
        return 32
    return 16


def _memory_breakdown(
    *,
    method: Method,
    distribution: Distribution,
    world_size: int,
    model: ModelSpec,
    target: TrainingTarget,
    target_modules: tuple[str, ...],
    rank: int,
    micro_batch_size: int,
) -> MemoryBreakdown:
    shard = world_size if distribution == Distribution.FSDP else 1
    storage_bytes = {
        Method.FULL: 2.0,
        Method.LORA: 2.0,
        Method.INT8_LORA: 1.0,
        Method.QLORA: 0.5,
    }[method]
    base_weights = round(model.parameters * storage_bytes / shard)
    quantization_metadata = 0
    if method == Method.INT8_LORA:
        quantization_metadata = round(model.parameters * 0.05 / shard)
    elif method == Method.QLORA:
        quantization_metadata = round(model.parameters * (0.127 / 8) / shard)

    trainable = (
        model.parameters
        if method == Method.FULL
        else _adapter_parameter_count(model, rank, target_modules)
    )
    adapter_weights = 0 if method == Method.FULL else round(trainable * 4 / shard)
    gradients = round(trainable * (2 if method == Method.FULL else 4) / shard)
    optimizer = round(trainable * 8 / shard)

    activation_factor = 2.5
    activations = round(
        micro_batch_size
        * target.sequence_length
        * model.hidden_size
        * model.layers
        * 2
        * activation_factor
    )
    communication = 0
    if distribution == Distribution.DDP:
        communication = min(round(trainable * 2), gibibytes(2))
    elif distribution == Distribution.FSDP:
        communication = min(round(trainable * 2 / world_size), gibibytes(3))
    workspace = max(
        gibibytes(0.5), round((base_weights + quantization_metadata) * 0.02)
    )
    temporary = max(
        gibibytes(0.5), round((base_weights + quantization_metadata) * 0.04)
    )
    load_transient = round((base_weights + quantization_metadata) * 0.20)
    point_components = {
        "base_weights_bytes": base_weights,
        "quantization_metadata_bytes": quantization_metadata,
        "adapter_weights_bytes": adapter_weights,
        "adapter_gradients_bytes": gradients,
        "optimizer_states_bytes": optimizer,
        "activations_bytes": activations,
        "communication_bytes": communication,
        "workspace_bytes": workspace,
        "temporary_overhead_bytes": temporary,
        "load_transient_bytes": load_transient,
    }
    before_allocator = sum(point_components.values())
    allocator = round(before_allocator * 0.08)
    point_components["allocator_bytes"] = allocator
    point = sum(point_components.values())
    safety = round(point * 0.10)
    component_upper_bounds = {
        **point_components,
        "activations_bytes": round(activations * 1.35),
        "communication_bytes": round(communication * 1.25),
        "workspace_bytes": round(workspace * 1.50),
        "temporary_overhead_bytes": round(temporary * 1.50),
        "allocator_bytes": round(allocator * 1.50),
        "load_transient_bytes": round(load_transient * 1.25),
        "uncertainty_bytes": safety,
    }
    upper = sum(component_upper_bounds.values())
    return MemoryBreakdown(
        base_weights_bytes=base_weights,
        quantization_metadata_bytes=quantization_metadata,
        adapter_weights_bytes=adapter_weights,
        adapter_gradients_bytes=gradients,
        optimizer_states_bytes=optimizer,
        activations_bytes=activations,
        temporary_overhead_bytes=temporary,
        safety_margin_bytes=safety,
        communication_bytes=communication,
        workspace_bytes=workspace,
        allocator_bytes=allocator,
        load_transient_bytes=load_transient,
        component_upper_bounds=component_upper_bounds,
        upper_estimate_bytes=upper,
        formula_version=FORMULA_VERSION,
        assumptions=(
            "Analytic envelope is not empirically calibrated.",
            "Gradient checkpointing is enabled.",
            f"{distribution.value} placement rules are applied per device.",
            "Point estimate is the sum of named point components and excludes uncertainty.",
            "Upper component factors: activations 1.35x; communication and load transient 1.25x; workspace, temporary, and allocator 1.50x.",
            "The upper envelope is the sum of component_upper_bounds, including the named uncertainty_bytes term.",
        ),
    )


def _distributions() -> tuple[Distribution, ...]:
    return Distribution.SINGLE, Distribution.DDP, Distribution.FSDP


def _usable_vram_bytes(hardware: HardwareSpec, device_index: int) -> int:
    device = hardware.devices[device_index]
    return (
        device.free_vram_bytes or device.total_vram_bytes
    ) - hardware.reserve_per_device_bytes


def _single_device_is_compatible(*, method: Method, device: DeviceSpec) -> bool:
    if device.backend != Backend.CUDA:
        return False
    if method == Method.FULL:
        return bool(device.supports_bf16)
    if method == Method.INT8_LORA:
        return bool(device.supports_8bit)
    if method == Method.QLORA:
        return bool(device.supports_4bit)
    return True


def _participating_device_indices(
    *,
    method: Method,
    distribution: Distribution,
    hardware: HardwareSpec,
) -> tuple[int, ...]:
    if distribution != Distribution.SINGLE:
        return tuple(range(len(hardware.devices)))
    compatible = [
        index
        for index, device in enumerate(hardware.devices)
        if _single_device_is_compatible(method=method, device=device)
    ]
    # Unsupported candidates still need a structurally valid device binding so their
    # rejection can be represented in the comparison matrix.
    selection_pool = compatible or list(range(len(hardware.devices)))
    if not selection_pool:
        return (0,)
    selected_index = max(
        selection_pool,
        key=lambda index: (_usable_vram_bytes(hardware, index), -index),
    )
    return (selected_index,)


def estimate_candidate(
    *,
    method: Method,
    model: ModelSpec,
    dataset: DatasetProfile,
    hardware: HardwareSpec,
    target: TrainingTarget,
    distribution: Distribution = Distribution.SINGLE,
) -> CandidatePlan:
    unsupported: list[str] = []
    infeasible: list[str] = []
    conditional: list[str] = []
    try:
        target_modules = (
            () if method == Method.FULL else target_modules_for(model.family)
        )
    except ValueError as error:
        target_modules = ()
        unsupported.append(str(error))

    devices = hardware.devices
    device_indices = _participating_device_indices(
        method=method,
        distribution=distribution,
        hardware=hardware,
    )
    participating_devices = tuple(
        devices[index] for index in device_indices if index < len(devices)
    )
    if not devices:
        unsupported.append("At least one CUDA GPU is required.")
    elif any(device.backend != Backend.CUDA for device in participating_devices):
        unsupported.append("Aptus v0.2 execution supports CUDA GPUs only.")
    if target.sequence_length > model.context_length:
        infeasible.append("Requested sequence length exceeds the model context length.")
    if dataset.schema_name not in {
        "text",
        "prompt-completion",
        "instruction-output",
        "messages",
        "mixed",
    }:
        unsupported.append(f"Unsupported dataset schema: {dataset.schema_name}.")
    if target.task != "sft":
        unsupported.append(
            "Aptus v0.2 compiles supervised fine-tuning (task='sft') only."
        )
    if target.packing:
        unsupported.append(
            "Sequence packing is not implemented in the v0.2 masking compiler; set packing=false."
        )
    if target.max_wall_time_minutes is not None:
        unsupported.append(
            "max_wall_time_minutes is fail-closed in Aptus v0.2 because the local process manager does not yet enforce a graceful checkpointing deadline."
        )

    if distribution in {Distribution.DDP, Distribution.FSDP} and len(devices) < 2:
        unsupported.append(f"{distribution.value} requires at least two GPUs.")
    if distribution == Distribution.FSDP and method in {Method.INT8_LORA, Method.QLORA}:
        unsupported.append(
            f"{method.value} with FSDP is outside the verified v0.2 compiler matrix."
        )
    if distribution == Distribution.FSDP and method == Method.FULL:
        unsupported.append(
            "Full-parameter FSDP is fail-closed in Aptus v0.2 because the pinned runtime upcasts trainable shards and full-state export to FP32, while that transient path is not yet calibrated."
        )
    if method == Method.QLORA and (
        not participating_devices
        or any(not item.supports_4bit for item in participating_devices)
    ):
        unsupported.append(
            "QLoRA requires explicit four-bit support on every participating GPU."
        )
    if method == Method.INT8_LORA and (
        not participating_devices
        or any(not item.supports_8bit for item in participating_devices)
    ):
        unsupported.append(
            "Eight-bit LoRA requires explicit eight-bit support on every participating GPU."
        )

    world_size = (
        1 if distribution == Distribution.SINGLE else max(1, len(participating_devices))
    )
    if target.effective_batch_size % world_size:
        infeasible.append(
            f"Global batch {target.effective_batch_size} is not divisible by world size {world_size}."
        )
    micro_batches = [
        value
        for value in range(min(32, target.effective_batch_size), 0, -1)
        if target.effective_batch_size % (value * world_size) == 0
    ]
    if not micro_batches:
        micro_batches = [1]

    precision = (
        "bf16"
        if participating_devices
        and all(item.supports_bf16 for item in participating_devices)
        else "fp16"
    )
    if method == Method.FULL and precision == "fp16":
        unsupported.append(
            "Full-parameter FP16 training is fail-closed in Aptus v0.2 because the generated mixed-precision path does not retain verified FP32 trainable master weights."
        )
    quantization = {
        Method.FULL: None,
        Method.LORA: None,
        Method.INT8_LORA: "int8-bitsandbytes",
        Method.QLORA: "nf4-double-quant",
    }[method]
    rank = 0 if method == Method.FULL else _rank_prior(dataset, target.objective)
    alpha = 0 if method == Method.FULL else rank * 2
    learning_rate = 2e-5 if method == Method.FULL else 2e-4
    policy_assumptions = [
        f"Learning rate {learning_rate:g} is an Aptus v0.2 method-class prior, not a tuned optimum.",
        f"Precision {precision} follows the shared capability rule across the candidate's bound device indices.",
        "The compiler uses AdamW (torch), a linear scheduler, zero weight decay, zero warmup steps, and max_grad_norm=1.0 as explicit Aptus v0.2 defaults.",
        "The compiler preserves supervised completion tokens first, left-truncates the prompt suffix to fit sequence_length, and refuses empty supervision.",
        "Base-model host staging and disk use 2.2 bytes per declared parameter as an explicit uncalibrated heuristic; provider artifact bytes and cache transients are verified only at runtime.",
    ]
    if distribution == Distribution.SINGLE and participating_devices:
        selected_index = device_indices[0]
        selected_device = participating_devices[0]
        if _single_device_is_compatible(method=method, device=selected_device):
            policy_assumptions.append(
                f"Single-device placement binds CUDA device index {selected_index} ({selected_device.name}), the method-compatible device with the greatest usable VRAM ({_usable_vram_bytes(hardware, selected_index)} bytes after reserve)."
            )
    elif distribution != Distribution.SINGLE and participating_devices:
        policy_assumptions.append(
            "Distributed placement binds every scanned CUDA device; precision and quantization require shared capability support, and memory fit uses the minimum usable per-device VRAM."
        )
    if method == Method.QLORA and distribution == Distribution.SINGLE:
        policy_assumptions.append(
            "Single-device QLoRA uses reentrant gradient checkpointing, following the pinned PEFT runtime guidance."
        )
    else:
        policy_assumptions.append(
            "This strategy uses non-reentrant gradient checkpointing; distributed QLoRA requires that mode in the pinned PEFT runtime."
        )
    if distribution == Distribution.DDP:
        policy_assumptions.append(
            f"DDP host staging budgets one independent model load per rank ({world_size} rank(s))."
        )
    elif distribution == Distribution.FSDP:
        policy_assumptions.append(
            f"FSDP host staging budgets CPU parameter materialization on every rank ({world_size} rank(s)), even when only rank zero reads checkpoint bytes."
        )
        if method == Method.LORA:
            policy_assumptions.append(
                "LoRA FSDP compiles with use_orig_params=true so interspersed frozen and trainable parameters remain supported. Its gradient and transient memory behavior is uncalibrated, so the exact real-model pilot remains mandatory."
            )
    if method != Method.FULL:
        policy_assumptions.extend(
            (
                f"Adapter rank {rank} is the Aptus v0.2 objective and dataset-volume prior.",
                f"Adapter alpha {alpha} follows the Aptus v0.2 alpha=2*rank policy.",
                "Target modules come from the versioned model-family catalog and require real-model inspection before execution approval.",
            )
        )
        if model.intermediate_size is None and any(
            item in {"gate_proj", "up_proj", "down_proj"} for item in target_modules
        ):
            policy_assumptions.append(
                "Model intermediate_size was not supplied; adapter parameter estimates use the explicit 4*hidden_size fallback."
            )

    selected_micro = micro_batches[-1]
    selected_memory = _memory_breakdown(
        method=method,
        distribution=distribution,
        world_size=world_size,
        model=model,
        target=target,
        target_modules=target_modules,
        rank=rank,
        micro_batch_size=selected_micro,
    )
    usable_vram = (
        min(
            (device.free_vram_bytes or device.total_vram_bytes)
            - hardware.reserve_per_device_bytes
            for device in participating_devices
        )
        if participating_devices
        else 0
    )
    point_fit = upper_fit = False
    if not unsupported:
        for micro in micro_batches:
            memory = _memory_breakdown(
                method=method,
                distribution=distribution,
                world_size=world_size,
                model=model,
                target=target,
                target_modules=target_modules,
                rank=rank,
                micro_batch_size=micro,
            )
            if memory.upper_bytes <= usable_vram:
                selected_micro, selected_memory, point_fit, upper_fit = (
                    micro,
                    memory,
                    True,
                    True,
                )
                break
            if not point_fit and memory.point_estimate_bytes <= usable_vram:
                selected_micro, selected_memory, point_fit = micro, memory, True
        if point_fit and not upper_fit:
            conditional.append(
                "Point estimate fits, but the uncalibrated heuristic upper envelope exceeds usable VRAM."
            )
        elif not point_fit:
            infeasible.append("Even the point estimate exceeds usable per-device VRAM.")
    if distribution == Distribution.FSDP and not unsupported:
        conditional.append(
            "FSDP uses a simplified uncalibrated per-device sharding prior; the exact wrapping and transient path requires a real-model pilot."
        )

    accumulation = math.ceil(
        target.effective_batch_size / (selected_micro * world_size)
    )
    exact_batch = selected_micro * accumulation * world_size
    if exact_batch != target.effective_batch_size:
        infeasible.append(
            "Exact global batch arithmetic could not preserve the requested batch."
        )

    host_loader_copies = 1 if distribution == Distribution.SINGLE else world_size
    required_host = round(model.parameters * 2.2 * host_loader_copies)
    trainable_parameters = (
        model.parameters
        if method == Method.FULL
        else _adapter_parameter_count(model, rank, target_modules)
    )
    checkpoint_unit = round(
        trainable_parameters * (10 if method == Method.FULL else 12)
    )
    checkpoint_retention = checkpoint_unit * 3
    pilot_checkpoint_workspace = checkpoint_unit * 4
    pilot_row_count = max(32, target.effective_batch_size * 2)
    pilot_sample_bytes = dataset.max_canonical_row_bytes * pilot_row_count
    final_export = max(
        gibibytes(0.0625),
        round(trainable_parameters * (2 if method == Method.FULL else 4)),
    )
    required_disk = round(
        model.parameters * 2.2
        + dataset.source_size_bytes
        + dataset.canonical_size_bytes
        + pilot_sample_bytes
        + checkpoint_retention
        + final_export
        + pilot_checkpoint_workspace
    )
    available_host = hardware.host_ram_free_bytes or hardware.host_ram_bytes
    if available_host < required_host:
        infeasible.append("Host RAM is below the minimum model-loading heuristic.")
    if (
        hardware.disk_free_bytes is not None
        and hardware.disk_free_bytes < required_disk
    ):
        infeasible.append(
            "Free disk is below the compiled staging, bounded-pilot, and three-checkpoint retention estimate."
        )

    if unsupported:
        status = CandidateStatus.UNSUPPORTED
    elif infeasible:
        status = CandidateStatus.INFEASIBLE
    elif conditional:
        status = CandidateStatus.CONDITIONAL
    else:
        status = CandidateStatus.FEASIBLE
    reasons = tuple(unsupported + infeasible + conditional)
    candidate = CandidatePlan(
        method=method,
        feasible=status in {CandidateStatus.FEASIBLE, CandidateStatus.CONDITIONAL},
        rejection_reasons=reasons,
        precision=precision,
        quantization=quantization,
        micro_batch_size=selected_micro,
        gradient_accumulation_steps=accumulation,
        effective_batch_size=exact_batch,
        rank=rank,
        alpha=alpha,
        learning_rate=learning_rate,
        target_modules=target_modules,
        memory=selected_memory,
        preference_score=0.0,
        confidence="uncalibrated-pilot-required",
        assumptions=selected_memory.assumptions + tuple(policy_assumptions),
        evidence=METHOD_EVIDENCE[method],
        candidate_id="",
        status=status,
        distribution=distribution,
        world_size=world_size,
        device_indices=device_indices,
        user_reserve_bytes=hardware.reserve_per_device_bytes,
        ranking_basis=(),
        required_host_ram_bytes=required_host,
        required_disk_bytes=required_disk,
        checkpoint_retention_bytes=checkpoint_retention,
        final_export_bytes=final_export,
    )
    return replace(
        candidate,
        candidate_id=candidate_id_for_payload(
            to_primitive(candidate),
            model=to_primitive(model),
            dataset=to_primitive(dataset),
            hardware=to_primitive(hardware),
            target=to_primitive(target),
        ),
    )


def _fidelity_order(method: Method) -> int:
    return {Method.FULL: 0, Method.LORA: 1, Method.INT8_LORA: 2, Method.QLORA: 3}[
        method
    ]


def _rank_key(candidate: CandidatePlan, target: TrainingTarget) -> tuple[object, ...]:
    status = 0 if candidate.status == CandidateStatus.FEASIBLE else 1
    preferred = 0 if target.method_preference == candidate.method else 1
    if target.objective == Objective.MEMORY:
        return (
            status,
            candidate.memory.upper_bytes,
            preferred,
            candidate.gradient_accumulation_steps,
        )
    if target.objective == Objective.SPEED:
        return (
            status,
            candidate.gradient_accumulation_steps,
            preferred,
            _fidelity_order(candidate.method),
        )
    return (
        status,
        _fidelity_order(candidate.method),
        preferred,
        candidate.memory.upper_bytes,
    )


def _mark_frontier(candidates: tuple[CandidatePlan, ...]) -> tuple[CandidatePlan, ...]:
    viable = [item for item in candidates if item.feasible]
    result: list[CandidatePlan] = []
    for item in candidates:
        fidelity = _fidelity_order(item.method)
        dominated = any(
            other.candidate_id != item.candidate_id
            and other.memory.upper_bytes <= item.memory.upper_bytes
            and _fidelity_order(other.method) <= fidelity
            and other.gradient_accumulation_steps <= item.gradient_accumulation_steps
            and (
                other.memory.upper_bytes < item.memory.upper_bytes
                or _fidelity_order(other.method) < fidelity
                or other.gradient_accumulation_steps < item.gradient_accumulation_steps
            )
            for other in viable
        )
        result.append(replace(item, pareto_frontier=item.feasible and not dominated))
    return tuple(result)


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
            distribution=distribution,
            model=model,
            dataset=dataset,
            hardware=hardware,
            target=target,
        )
        for method in Method
        for distribution in _distributions()
    )
    candidates = _mark_frontier(candidates)
    viable = [item for item in candidates if item.feasible]
    if not viable:
        raise NoFeasiblePlanError(candidates)
    ordered = sorted(viable, key=lambda item: _rank_key(item, target))
    ranked_ids = {item.candidate_id: index for index, item in enumerate(ordered)}
    candidates = tuple(
        replace(
            item,
            preference_score=float(-ranked_ids[item.candidate_id])
            if item.candidate_id in ranked_ids
            else -1_000_000_000.0,
            ranking_basis=(
                f"Objective policy: {target.objective.value}.",
                "No model-quality or throughput value was fabricated.",
                "A measured synthetic method preflight and exact real-model/data pilot remain required before execution approval.",
            ),
        )
        for item in candidates
    )
    recommended = next(
        item for item in candidates if item.candidate_id == ordered[0].candidate_id
    )
    rationale = (
        f"Selected {recommended.candidate_id}: {recommended.method.value} with {recommended.distribution.value}.",
        f"Selection used the explicit {target.objective.value} policy over {len(viable)} viable candidates.",
        "The ranking is a configuration policy, not a prediction of model quality or measured speed.",
    )
    warnings = tuple(
        dict.fromkeys(
            dataset.warnings
            + (
                "Memory envelopes are analytic and uncalibrated.",
                "Disk requirement is a conservative prior covering staging, two bounded pilot checkpoint workspaces, and three retained training checkpoints.",
                "A candidate-specific synthetic method preflight and exact real-model/data pilot are required before full execution.",
            )
            + (
                (
                    "Model intermediate_size was not supplied. Adapter estimates for MLP targets use the explicit 4*hidden_size fallback.",
                )
                if model.intermediate_size is None
                else ()
            )
        )
    )
    evidence_ids = sorted(
        {item for candidate in candidates for item in candidate.evidence}
    )
    evidence_records = evidence_for(*evidence_ids)
    plan = TrainingPlan(
        schema_version=SCHEMA_VERSION,
        model=model,
        dataset=dataset,
        hardware=hardware,
        target=target,
        recommended=recommended,
        candidates=candidates,
        warnings=warnings,
        recommendation_rationale=rationale,
        evidence_records=evidence_records,
        formula_version=FORMULA_VERSION,
        plan_id="",
    )
    return replace(plan, plan_id=plan_id_for_payload(to_primitive(plan)))
