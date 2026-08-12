from __future__ import annotations

import math
from dataclasses import replace

from .catalog import target_modules_for
from .domain import (
    Backend,
    CandidatePlan,
    CandidateStatus,
    DatasetProfile,
    DeviceSpec,
    Distribution,
    EvidenceRequirement,
    HardwareSpec,
    MemoryBreakdown,
    Method,
    ModelInspectionReceipt,
    ModelPolicyBindingSource,
    ModelPolicyDecision,
    ModelSpec,
    Objective,
    RuntimeContract,
    SCHEMA_VERSION,
    TrainingPlan,
    TrainingRuntime,
    TrainingTarget,
    gibibytes,
    to_primitive,
)
from .evidence import evidence_for
from .methods import (
    method_descriptor,
    runtime_binding,
    runtime_contract_for as registered_runtime_contract_for,
    selectable_method_descriptors,
)
from .model_compatibility import (
    current_model_policy_snapshot_sha256,
    evaluate_model_compatibility,
    matching_model_policy_path,
    model_with_inspection_provenance,
    model_with_user_attested_provenance,
    model_policy_binding_for_path,
    model_policy_rejection_reasons,
    subject_from_model,
    validate_model_inspection_receipt,
)
from .plan_contract import (
    candidate_id_for_payload,
    mlx_memory_breakdown_for_contract,
    plan_id_for_payload,
    validate_plan_payload,
)


FORMULA_VERSION = "aptus-memory-v2"
MLX_FORMULA_VERSION = "aptus-memory-mlx-v2"


class NoFeasiblePlanError(ValueError):
    def __init__(
        self,
        candidates: tuple[CandidatePlan, ...],
        *,
        model: ModelSpec,
        model_policy_decision: ModelPolicyDecision,
        model_policy_decision_source: ModelPolicyBindingSource,
        inspection_receipt: ModelInspectionReceipt | None,
        ranking_objective: str | None = None,
    ) -> None:
        if not isinstance(model, ModelSpec):
            raise TypeError("No-feasible-plan errors require a model subject.")
        if not isinstance(model_policy_decision, ModelPolicyDecision):
            raise TypeError("No-feasible-plan errors require a policy decision.")
        if not isinstance(model_policy_decision_source, ModelPolicyBindingSource):
            raise TypeError("No-feasible-plan errors require a policy source.")
        if not candidates or any(
            candidate.model_policy_decision_id != model_policy_decision.decision_id
            for candidate in candidates
        ):
            raise ValueError(
                "No-feasible-plan candidates must bind the policy decision."
            )
        if (
            model_policy_decision_source == ModelPolicyBindingSource.PROVIDER_INSPECTION
            and inspection_receipt is None
        ):
            raise ValueError(
                "Provider-inspection no-feasible-plan errors require a receipt."
            )
        if (
            model_policy_decision_source == ModelPolicyBindingSource.USER_ATTESTED
            and inspection_receipt is not None
        ):
            raise ValueError(
                "User-attested no-feasible-plan errors cannot carry a receipt."
            )
        if inspection_receipt is not None and (
            inspection_receipt.decision.decision_id != model_policy_decision.decision_id
        ):
            raise ValueError(
                "The inspection receipt must bind the no-feasible-plan decision."
            )
        if (
            inspection_receipt is not None
            and inspection_receipt.model_id != model.model_id
        ):
            raise ValueError(
                "The no-feasible-plan inspection receipt model ID must match the model."
            )
        if inspection_receipt is not None and (
            inspection_receipt.resolved_revision.lower() != model.revision.lower()
        ):
            raise ValueError(
                "The no-feasible-plan inspection receipt revision must match the model."
            )
        expected_receipt_id = (
            inspection_receipt.receipt_id if inspection_receipt is not None else None
        )
        for candidate in candidates:
            binding = candidate.policy_binding
            if binding is None:
                continue
            if binding.source != model_policy_decision_source:
                raise ValueError(
                    "Candidate policy binding source must match the error source."
                )
            if binding.inspection_receipt_id != expected_receipt_id:
                raise ValueError(
                    "Candidate policy binding receipt must match the error receipt."
                )
        self.candidates = candidates
        self.model = model
        self.model_policy_decision = model_policy_decision
        self.model_policy_decision_source = model_policy_decision_source
        self.inspection_receipt = inspection_receipt
        self.ranking_objective = ranking_objective
        reasons = sorted(
            {reason for item in candidates for reason in item.rejection_reasons}
        )
        super().__init__(
            "No feasible or conditional training plan: " + "; ".join(reasons)
        )


def _adapter_parameter_count(
    model: ModelSpec, rank: int, modules: tuple[str, ...]
) -> int:
    if model.moe is not None and any(
        module in {"gate_proj", "up_proj", "down_proj"} for module in modules
    ):
        raise ValueError(
            "Expert adapter targets require a topology-aware cardinality contract."
        )
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


def _mlx_memory_breakdown(
    *,
    method: Method,
    model: ModelSpec,
    target: TrainingTarget,
    target_modules: tuple[str, ...],
    rank: int,
    micro_batch_size: int,
) -> MemoryBreakdown:
    """Conservative unified-memory envelope for the MLX-LM pilot path."""

    calculated = mlx_memory_breakdown_for_contract(
        model=to_primitive(model),
        target=to_primitive(target),
        candidate={
            "method": method.value,
            "rank": rank,
            "micro_batch_size": micro_batch_size,
            "target_modules": list(target_modules),
        },
    )
    return MemoryBreakdown(
        base_weights_bytes=calculated["base_weights_bytes"],
        quantization_metadata_bytes=calculated["quantization_metadata_bytes"],
        adapter_weights_bytes=calculated["adapter_weights_bytes"],
        adapter_gradients_bytes=calculated["adapter_gradients_bytes"],
        optimizer_states_bytes=calculated["optimizer_states_bytes"],
        activations_bytes=calculated["activations_bytes"],
        temporary_overhead_bytes=calculated["temporary_overhead_bytes"],
        safety_margin_bytes=calculated["safety_margin_bytes"],
        communication_bytes=calculated["communication_bytes"],
        workspace_bytes=calculated["workspace_bytes"],
        allocator_bytes=calculated["allocator_bytes"],
        load_transient_bytes=calculated["load_transient_bytes"],
        component_upper_bounds=calculated["component_upper_bounds"],
        upper_estimate_bytes=calculated["upper_estimate_bytes"],
        formula_version=calculated["formula_version"],
        assumptions=(
            "MLX-LM uses Apple unified memory; this estimate does not add a second CUDA-style host staging pool.",
            *(
                (
                    "The reviewed Qwen3 MoE layout prices its four-bit group-64 default, eight-bit group-64 router gates, and affine scale and bias metadata separately.",
                )
                if model.quantization_layout is not None and model.moe is not None
                else (
                    "The bound dense MLX affine layout prices weights and scale and bias metadata from its declared default bit width and group size.",
                )
                if model.quantization_layout is not None
                else (
                    "MLX four-bit storage is modeled as groupwise quantized weights plus explicit metadata, not bitsandbytes NF4.",
                )
            ),
            "The formula is a conservative uncalibrated prior and cannot establish feasibility.",
            "Gradient checkpointing is enabled for the generated MLX-LM pilot.",
            "Point estimate is the sum of named components and excludes uncertainty.",
            "Upper envelope uses wider MLX allocator, activation, workspace, and load-transient factors plus a 25 percent uncertainty term.",
            *(
                (
                    "MoE residency, quantization metadata, staging, and disk use total parameters; active parameters never substitute for resident weights.",
                    "MoE activation memory adds a routed SwiGLU term for every selected expert in each sparse layer, with a conservative 3x checkpointed-intermediate factor.",
                )
                if model.moe is not None
                else ()
            ),
        ),
    )


def _memory_for_runtime(
    *,
    training_runtime: TrainingRuntime,
    method: Method,
    distribution: Distribution,
    world_size: int,
    model: ModelSpec,
    target: TrainingTarget,
    target_modules: tuple[str, ...],
    rank: int,
    micro_batch_size: int,
) -> MemoryBreakdown:
    if training_runtime == TrainingRuntime.MLX_LM and method in {
        Method.LORA,
        Method.QLORA,
    }:
        return _mlx_memory_breakdown(
            method=method,
            model=model,
            target=target,
            target_modules=target_modules,
            rank=rank,
            micro_batch_size=micro_batch_size,
        )
    return _memory_breakdown(
        method=method,
        distribution=distribution,
        world_size=world_size,
        model=model,
        target=target,
        target_modules=target_modules,
        rank=rank,
        micro_batch_size=micro_batch_size,
    )


def _distributions() -> tuple[Distribution, ...]:
    return Distribution.SINGLE, Distribution.DDP, Distribution.FSDP


def _usable_vram_bytes(hardware: HardwareSpec, device_index: int) -> int:
    device = hardware.devices[device_index]
    capacity = device.free_vram_bytes or device.total_vram_bytes
    if device.backend == Backend.MPS and hardware.host_ram_free_bytes is not None:
        # Apple unified memory is one live pool. Keep the host measurement
        # explicit instead of copying it into a fictional free-VRAM field.
        capacity = min(capacity, hardware.host_ram_free_bytes)
    return capacity - hardware.reserve_per_device_bytes


def _single_device_is_compatible(*, method: Method, device: DeviceSpec) -> bool:
    if device.backend == Backend.MPS:
        # MLX QLoRA eligibility belongs to the pinned model revision, not to a
        # CUDA-style device capability bit. The generated model-data gate
        # requires explicit four-bit MLX quantization metadata before work can
        # advance.
        return method in {Method.LORA, Method.QLORA}
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


def _runtime_contract_for(
    *,
    method: Method,
    target: TrainingTarget,
    devices: tuple[DeviceSpec, ...],
) -> RuntimeContract:
    compute_backend = devices[0].backend if devices else Backend.CUDA
    if target.training_runtime is not None:
        training_runtime = TrainingRuntime(target.training_runtime)
    elif compute_backend == Backend.MPS:
        training_runtime = (
            TrainingRuntime.MLX_LM
            if method in {Method.LORA, Method.QLORA}
            else TrainingRuntime.PYTORCH_MPS
        )
    else:
        training_runtime = TrainingRuntime.TRANSFORMERS_PEFT_CUDA
    contract = registered_runtime_contract_for(
        method,
        training_runtime=training_runtime,
        compute_backend=compute_backend,
    )
    if contract is None:
        return RuntimeContract(
            compute_backend=(
                Backend.MPS
                if training_runtime
                in {TrainingRuntime.MLX_LM, TrainingRuntime.PYTORCH_MPS}
                else Backend.CUDA
            ),
            training_runtime=training_runtime,
            compiler_id=None,
            estimator_id="unavailable",
            evidence_requirement=EvidenceRequirement.IMPLEMENTATION_REQUIRED,
            export_kind=None,
        )
    return contract


def _estimate_candidate_with_policy(
    *,
    method: Method,
    model: ModelSpec,
    dataset: DatasetProfile,
    hardware: HardwareSpec,
    target: TrainingTarget,
    distribution: Distribution = Distribution.SINGLE,
    policy_decision: ModelPolicyDecision,
    inspection_receipt: ModelInspectionReceipt | None,
) -> CandidatePlan:
    unsupported: list[str] = []
    infeasible: list[str] = []
    conditional: list[str] = []
    descriptor = method_descriptor(method)
    if distribution.value not in descriptor.supported_distributions:
        unsupported.append(
            f"The {descriptor.method_id} registry contract does not support "
            f"{distribution.value} distribution."
        )
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
    runtime_contract = _runtime_contract_for(
        method=method,
        target=target,
        devices=participating_devices,
    )
    policy_path = matching_model_policy_path(
        policy_decision,
        method=method,
        distribution=distribution,
        target_modules=target_modules,
        runtime_contract=runtime_contract,
    )
    policy_binding = (
        model_policy_binding_for_path(
            decision=policy_decision,
            path=policy_path,
            receipt=inspection_receipt,
        )
        if policy_path is not None
        else None
    )
    unsupported.extend(
        model_policy_rejection_reasons(
            policy_decision,
            method=method,
            distribution=distribution,
            target_modules=target_modules,
            runtime_contract=runtime_contract,
        )
    )
    binding = runtime_binding(
        method,
        training_runtime=runtime_contract.training_runtime,
        compute_backend=runtime_contract.compute_backend,
    )
    if binding is None:
        unsupported.append(
            f"{runtime_contract.training_runtime.value} has no registered {method.value} "
            f"compiler on {runtime_contract.compute_backend.value}."
        )
    elif distribution.value not in binding.supported_distributions:
        unsupported.append(
            f"The {runtime_contract.training_runtime.value} {descriptor.method_id} "
            f"compiler does not support {distribution.value} distribution."
        )
    if not devices:
        unsupported.append("At least one supported compute device is required.")
    elif len({device.backend for device in participating_devices}) != 1:
        unsupported.append("A candidate cannot mix compute backends.")
    elif any(
        device.backend != runtime_contract.compute_backend
        for device in participating_devices
    ):
        unsupported.append(
            f"{runtime_contract.training_runtime.value} requires "
            f"{runtime_contract.compute_backend.value} compute."
        )
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
    if (
        method == Method.QLORA
        and runtime_contract.training_runtime != TrainingRuntime.MLX_LM
        and (
            not participating_devices
            or any(not item.supports_4bit for item in participating_devices)
        )
    ):
        unsupported.append(
            "QLoRA requires explicit runtime-native four-bit support on every participating device."
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
    if target.micro_batch_size is not None:
        explicit_batch = (
            target.micro_batch_size * target.gradient_accumulation_steps * world_size
        )
        if explicit_batch != target.effective_batch_size:
            infeasible.append(
                "Explicit micro-batch, accumulation, and world-size arithmetic "
                "does not equal the requested effective batch."
            )
        micro_batches = [target.micro_batch_size]
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
    quantization = (
        "mlx-4bit-groupwise"
        if runtime_contract.training_runtime == TrainingRuntime.MLX_LM
        and method == Method.QLORA
        else {
            Method.FULL: None,
            Method.LORA: None,
            Method.INT8_LORA: "int8-bitsandbytes",
            Method.QLORA: "nf4-double-quant",
        }[method]
    )
    rank = 0 if method == Method.FULL else _rank_prior(dataset, target.objective)
    alpha = 0 if method == Method.FULL else rank * 2
    learning_rate = 2e-5 if method == Method.FULL else 2e-4
    policy_assumptions = [
        f"Learning rate {learning_rate:g} is an Aptus v0.2 method-class prior, not a tuned optimum.",
        f"Precision {precision} follows the shared capability rule across the candidate's bound device indices.",
        "The compiler preserves supervised completion tokens first, left-truncates the prompt suffix to fit sequence_length, and refuses empty supervision.",
        "Base-model host staging and disk use 2.2 bytes per declared parameter as an explicit uncalibrated heuristic; provider artifact bytes and cache transients are verified only at runtime.",
    ]
    if runtime_contract.training_runtime == TrainingRuntime.MLX_LM:
        policy_assumptions.extend(
            (
                "The generated compiler uses MLX-LM with AdamW, gradient checkpointing, and an MLX-native adapter export.",
                "MLX-LM QLoRA requires an already four-bit MLX model revision; it does not invoke bitsandbytes or assume NF4 kernels.",
                "MLX-LM QLoRA eligibility is verified from the pinned model's four-bit quantization metadata during model-data validation, not inferred from a CUDA-style device flag.",
                "The Apple unified-memory envelope must pass model-data validation and a bounded measured preflight; neither guarantees full-run fit.",
            )
        )
        if model.moe is not None:
            policy_assumptions.extend(
                (
                    f"The model has {model.parameters} total resident parameters and {model.active_parameters} derived active parameters per token.",
                    f"The reviewed topology routes {model.moe.experts_per_token} of {model.moe.expert_count} experts across {model.sparse_layer_count} sparse layers.",
                    "The first MoE compiler scope trains attention projections only; routed experts and router weights remain frozen.",
                )
            )
    else:
        policy_assumptions.append(
            "The compiler uses AdamW (torch), a linear scheduler, zero weight decay, zero warmup steps, and max_grad_norm=1.0 as explicit Aptus v0.2 defaults."
        )
    if distribution == Distribution.SINGLE and participating_devices:
        selected_index = device_indices[0]
        selected_device = participating_devices[0]
        if _single_device_is_compatible(method=method, device=selected_device):
            capacity_label = (
                "usable VRAM"
                if selected_device.backend == Backend.CUDA
                else "usable unified memory"
            )
            policy_assumptions.append(
                f"Single-device placement binds {selected_device.backend.value} device index {selected_index} ({selected_device.name}), the method-compatible device with the greatest {capacity_label} ({_usable_vram_bytes(hardware, selected_index)} bytes after reserve)."
            )
    elif distribution != Distribution.SINGLE and participating_devices:
        policy_assumptions.append(
            "Distributed placement binds every scanned device; precision and quantization require shared capability support, and memory fit uses the minimum usable per-device memory."
        )
    if (
        runtime_contract.training_runtime == TrainingRuntime.TRANSFORMERS_PEFT_CUDA
        and method == Method.QLORA
        and distribution == Distribution.SINGLE
    ):
        policy_assumptions.append(
            "Single-device QLoRA uses reentrant gradient checkpointing, following the pinned PEFT runtime guidance."
        )
    elif runtime_contract.training_runtime == TrainingRuntime.TRANSFORMERS_PEFT_CUDA:
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
    selected_memory = _memory_for_runtime(
        training_runtime=runtime_contract.training_runtime,
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
            _usable_vram_bytes(hardware, device_index)
            for device_index in device_indices
            if device_index < len(hardware.devices)
        )
        if participating_devices
        else 0
    )
    point_fit = upper_fit = False
    if not unsupported:
        for micro in micro_batches:
            memory = _memory_for_runtime(
                training_runtime=runtime_contract.training_runtime,
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
                "Point estimate fits, but the uncalibrated heuristic upper envelope exceeds usable per-device memory."
            )
        elif not point_fit:
            infeasible.append(
                "Even the point estimate exceeds usable per-device memory."
            )
    if distribution == Distribution.FSDP and not unsupported:
        conditional.append(
            "FSDP uses a simplified uncalibrated per-device sharding prior; the exact wrapping and transient path requires a real-model pilot."
        )
    if runtime_contract.training_runtime == TrainingRuntime.MLX_LM and not unsupported:
        conditional.append(
            "MLX-LM support is pilot-required: the unified-memory estimate is provisional and cannot guarantee that the exact model and data fit."
        )

    accumulation = (
        target.gradient_accumulation_steps
        if target.gradient_accumulation_steps is not None
        else math.ceil(target.effective_batch_size / (selected_micro * world_size))
    )
    exact_batch = selected_micro * accumulation * world_size
    if exact_batch != target.effective_batch_size:
        infeasible.append(
            "Exact global batch arithmetic could not preserve the requested batch."
        )

    host_loader_copies = 1 if distribution == Distribution.SINGLE else world_size
    required_host = (
        selected_memory.point_estimate_bytes
        if runtime_contract.training_runtime == TrainingRuntime.MLX_LM
        else round(model.parameters * 2.2 * host_loader_copies)
    )
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
    if (
        runtime_contract.training_runtime != TrainingRuntime.MLX_LM
        and available_host < required_host
    ):
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
        evidence=tuple(
            dict.fromkeys(
                descriptor.evidence_ids
                + (policy_binding.evidence_ids if policy_binding is not None else ())
            )
        ),
        model_policy_decision_id=policy_decision.decision_id,
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
        runtime_contract=runtime_contract,
        policy_binding=policy_binding,
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


def estimate_candidate(
    *,
    method: Method,
    model: ModelSpec,
    dataset: DatasetProfile,
    hardware: HardwareSpec,
    target: TrainingTarget,
    distribution: Distribution = Distribution.SINGLE,
) -> CandidatePlan:
    """Estimate one candidate after evaluating its model policy."""

    return _estimate_candidate_with_policy(
        method=method,
        model=model,
        dataset=dataset,
        hardware=hardware,
        target=target,
        distribution=distribution,
        policy_decision=evaluate_model_compatibility(subject_from_model(model)),
        inspection_receipt=None,
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


def select_candidate(plan: TrainingPlan, candidate_id: str) -> TrainingPlan:
    """Select one complete viable candidate and derive a new immutable plan."""

    payload = to_primitive(plan)
    errors = validate_plan_payload(payload, verify_dataset=False)
    if errors:
        raise ValueError(
            "Candidate selection requires a current, unmodified plan: "
            + " ".join(errors)
        )
    matches = [item for item in plan.candidates if item.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValueError(
            "Candidate is stale, unknown, or does not belong to this plan."
        )
    selected = matches[0]
    if (
        candidate_id_for_payload(
            to_primitive(selected),
            model=payload["model"],
            dataset=payload["dataset"],
            hardware=payload["hardware"],
            target=payload["target"],
        )
        != candidate_id
    ):
        raise ValueError(
            "Candidate identity is mutated or inconsistent with plan facts."
        )
    if not selected.feasible or selected.status not in {
        CandidateStatus.FEASIBLE,
        CandidateStatus.CONDITIONAL,
    }:
        raise ValueError("Candidate is rejected or nonselectable.")
    if selected.candidate_id == plan.recommended.candidate_id:
        raise ValueError("Candidate is already selected; choose a different candidate.")
    revised = replace(
        plan,
        recommended=selected,
        recommendation_rationale=(
            f"Explicitly selected complete candidate {selected.candidate_id}.",
            "Selection preserved the candidate's planning facts, policy binding, and evidence chain.",
            "The selected configuration remains subject to its recorded validation and pilot gates.",
        ),
        plan_id="",
    )
    return replace(revised, plan_id=plan_id_for_payload(to_primitive(revised)))


def plan_training(
    *,
    model: ModelSpec,
    dataset: DatasetProfile,
    hardware: HardwareSpec,
    target: TrainingTarget,
    inspection_receipt: ModelInspectionReceipt | None = None,
) -> TrainingPlan:
    policy_decision = evaluate_model_compatibility(subject_from_model(model))
    if inspection_receipt is not None:
        if not isinstance(inspection_receipt, ModelInspectionReceipt):
            raise ValueError(
                "inspection_receipt must be a typed model inspection receipt."
            )
        validate_model_inspection_receipt(
            receipt=inspection_receipt,
            model=model,
            decision=policy_decision,
        )
        model = model_with_inspection_provenance(model, inspection_receipt)
    else:
        model = model_with_user_attested_provenance(model)
    policy_source = (
        ModelPolicyBindingSource.PROVIDER_INSPECTION
        if inspection_receipt is not None
        else ModelPolicyBindingSource.USER_ATTESTED
    )
    candidates = tuple(
        _estimate_candidate_with_policy(
            method=method,
            distribution=distribution,
            model=model,
            dataset=dataset,
            hardware=hardware,
            target=target,
            policy_decision=policy_decision,
            inspection_receipt=inspection_receipt,
        )
        for descriptor in selectable_method_descriptors()
        for method in (Method(descriptor.method_id),)
        for distribution in _distributions()
    )
    candidates = _mark_frontier(candidates)
    viable = [item for item in candidates if item.feasible]
    if not viable:
        raise NoFeasiblePlanError(
            candidates,
            model=model,
            model_policy_decision=policy_decision,
            model_policy_decision_source=policy_source,
            inspection_receipt=inspection_receipt,
            ranking_objective=target.objective.value,
        )
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
        model_policy_decision=policy_decision,
        model_policy_decision_source=policy_source,
        inspection_receipt=inspection_receipt,
        model_policy_snapshot_sha256=current_model_policy_snapshot_sha256(),
        evidence_records=evidence_records,
        formula_version=FORMULA_VERSION,
        plan_id="",
    )
    return replace(plan, plan_id=plan_id_for_payload(to_primitive(plan)))
