from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .catalog import (
    DENSE_CAUSAL_LM_TARGET_MODULES,
    QWEN3_MOE_ARCHITECTURE,
    QWEN3_MOE_FAMILY,
    QWEN3_MOE_MODEL_TYPE,
    QWEN3_MOE_TARGET_MODULES,
    TARGET_MODULES,
    has_reviewed_qwen3_moe_quantization_layout,
    is_exact_qwen3_moe,
)
from .domain import (
    AdapterProfile,
    Backend,
    Distribution,
    EvidenceRequirement,
    Method,
    ModelCompatibilitySubject,
    ModelPolicyDecision,
    ModelPolicyDecisionKind,
    ModelPolicyPath,
    ModelSpec,
    RuntimeContract,
    TrainingRuntime,
)
from .methods import (
    method_descriptor,
    runtime_binding,
    runtime_contract_for,
)


QWEN3_MOE_IDENTITY_REASON = (
    "MoE execution requires the exact reviewed qwen3_moe and "
    "Qwen3MoeForCausalLM provider identity."
)
QWEN3_MOE_LAYOUT_REASON = (
    "Qwen3 MoE execution requires the exact four-bit plus eight-bit "
    "router-gate MLX quantization layout."
)
QWEN3_MOE_TOPOLOGY_REASON = (
    "Qwen3 MoE execution requires the complete provider-declared expert topology."
)
QWEN3_MOE_SHARED_EXPERT_REASON = (
    "The first Qwen3 MoE MLX-LM contract does not support a shared expert."
)
QWEN3_MOE_FOUR_BIT_REASON = (
    "The first Qwen3 MoE MLX-LM contract requires explicit four-bit model metadata."
)
QWEN3_MOE_PATH_REASON = (
    "Qwen3 MoE is executable only as single-device MLX-LM QLoRA with "
    "attention-only adapters."
)
QWEN3_MOE_MATCHED_REASON = (
    "The model identity, mixed-precision layout, routed-expert topology, and "
    "attention-only q/k/v/o target policy match the reviewed Qwen3 MoE slice. "
    "Measured preflight and a real-model pilot remain mandatory."
)
QWEN3_MOE_BLOCKED_INSPECTION_REASON = (
    "The exact Qwen3 MoE identity was recognized, but this revision does not "
    "match the reviewed four-bit default, eight-bit router-gate overrides, and "
    "no-shared-expert topology."
)
FAMILY_RECOGNIZED_REASON = (
    "The provider identity maps to an existing dense Aptus family; the planner "
    "still decides the executable runtime and method."
)
UNKNOWN_POLICY_REASON = (
    "No exact Aptus model-family compatibility policy matches this provider "
    "model type and architecture."
)
UNREVIEWED_SPARSE_MODEL_REASON = (
    "Sparse model execution requires an exact reviewed model compatibility policy."
)
INVALID_COMPATIBILITY_FACTS_REASON = (
    "Model compatibility facts are malformed or contradictory."
)
_QWEN3_EXACT_IDENTITY_BLOCKING_REASONS = {
    QWEN3_MOE_LAYOUT_REASON,
    QWEN3_MOE_TOPOLOGY_REASON,
    QWEN3_MOE_SHARED_EXPERT_REASON,
    QWEN3_MOE_FOUR_BIT_REASON,
    INVALID_COMPATIBILITY_FACTS_REASON,
}


ADAPTER_PROFILE_TARGET_MODULES: dict[AdapterProfile, tuple[str, ...]] = {
    AdapterProfile.ATTENTION_QKVO_V1: QWEN3_MOE_TARGET_MODULES,
}


@dataclass(frozen=True)
class _ModelCompatibilityPolicy:
    family: str
    claims: Callable[[ModelCompatibilitySubject], bool]
    first_blocking_reason: Callable[[ModelCompatibilitySubject], str | None]
    paths: tuple[ModelPolicyPath, ...]
    matched_reason: str


def adapter_target_modules(
    profile: AdapterProfile,
) -> tuple[str, ...]:
    try:
        return ADAPTER_PROFILE_TARGET_MODULES[profile]
    except KeyError as error:
        raise ValueError(f"Unknown adapter profile: {profile.value!r}") from error


def validate_execution_path_selection(
    *,
    method: Method,
    training_runtime: TrainingRuntime,
    compute_backend: Backend,
    distribution: Distribution,
    adapter_profile_id: AdapterProfile | None,
) -> RuntimeContract:
    """Validate one path through the method registry and return its contract."""

    descriptor = method_descriptor(method)
    if not descriptor.selectable:
        raise ValueError("Compatibility paths require a selectable method.")
    if descriptor.parameterization == "lora":
        if adapter_profile_id is None:
            raise ValueError("Adapter compatibility paths require an adapter profile.")
        adapter_target_modules(adapter_profile_id)
    elif adapter_profile_id is not None:
        raise ValueError(
            "Conditional compatibility adapter profiles require adapter methods."
        )

    binding = runtime_binding(
        method,
        training_runtime=training_runtime,
        compute_backend=compute_backend,
    )
    if binding is None:
        raise ValueError(
            "Conditional compatibility requires a registered method, runtime, "
            "and compute-backend binding."
        )
    if distribution.value not in binding.supported_distributions:
        raise ValueError(
            "Conditional compatibility distribution is not supported by the "
            "registered runtime binding."
        )
    contract = runtime_contract_for(
        method,
        training_runtime=training_runtime,
        compute_backend=compute_backend,
    )
    if contract is None:  # pragma: no cover - guarded by the binding above
        raise RuntimeError("A registered runtime binding did not produce a contract.")
    return contract


def validate_registered_compatibility_path(
    *,
    family: str,
    method: Method,
    training_runtime: TrainingRuntime,
    compute_backend: Backend,
    distribution: Distribution,
    adapter_profile_id: AdapterProfile | None,
    evidence_requirement: str | EvidenceRequirement,
) -> RuntimeContract:
    """Validate that a conditional API claim is an actual model-policy path."""

    contract = validate_execution_path_selection(
        method=method,
        training_runtime=training_runtime,
        compute_backend=compute_backend,
        distribution=distribution,
        adapter_profile_id=adapter_profile_id,
    )
    required_evidence = EvidenceRequirement(evidence_requirement)
    if any(
        policy.family == family
        and any(
            path.method == method
            and path.distribution == distribution
            and path.adapter_profile_id == adapter_profile_id
            and path.runtime_contract == contract
            and path.runtime_contract.evidence_requirement == required_evidence
            for path in policy.paths
        )
        for policy in MODEL_COMPATIBILITY_POLICIES
    ):
        return contract
    raise ValueError(
        "Conditional compatibility requires a path registered for the model family."
    )


def _policy_path(
    *,
    family: str,
    method: Method,
    training_runtime: TrainingRuntime,
    compute_backend: Backend,
    distribution: Distribution,
    adapter_profile_id: AdapterProfile,
) -> ModelPolicyPath:
    target_modules = adapter_target_modules(adapter_profile_id)
    if TARGET_MODULES.get(family) != target_modules:
        raise RuntimeError(
            "Model policy adapter targets differ from the family catalog."
        )
    return ModelPolicyPath(
        method=method,
        distribution=distribution,
        adapter_profile_id=adapter_profile_id,
        target_modules=target_modules,
        runtime_contract=validate_execution_path_selection(
            method=method,
            training_runtime=training_runtime,
            compute_backend=compute_backend,
            distribution=distribution,
            adapter_profile_id=adapter_profile_id,
        ),
    )


def _qwen3_moe_claims(subject: ModelCompatibilitySubject) -> bool:
    return bool(
        (isinstance(subject.family, str) and subject.family.lower() == QWEN3_MOE_FAMILY)
        or subject.model_type == QWEN3_MOE_MODEL_TYPE
        or subject.architecture == QWEN3_MOE_ARCHITECTURE
    )


def _has_sparse_identity_marker(subject: ModelCompatibilitySubject) -> bool:
    markers = ("moe", "mixtral")
    return any(
        marker in value.lower()
        for value in (subject.family, subject.model_type, subject.architecture)
        if value is not None
        for marker in markers
    )


def _has_executable_sparse_layer(subject: ModelCompatibilitySubject) -> bool:
    if subject.moe is None or subject.layers is None:
        return False
    dense_layers = set(subject.moe.mlp_only_layers)
    return any(
        (index + 1) % subject.moe.decoder_sparse_step == 0 and index not in dense_layers
        for index in range(subject.layers)
    )


def _is_exact_qwen3_moe_subject(subject: ModelCompatibilitySubject) -> bool:
    return bool(
        isinstance(subject.family, str)
        and is_exact_qwen3_moe(
            family=subject.family,
            model_type=subject.model_type,
            architecture=subject.architecture or "",
        )
    )


def _qwen3_moe_first_blocking_reason(
    subject: ModelCompatibilitySubject,
) -> str | None:
    if not _is_exact_qwen3_moe_subject(subject):
        return QWEN3_MOE_IDENTITY_REASON
    if subject.layers is None or not has_reviewed_qwen3_moe_quantization_layout(
        subject.quantization_layout,
        layers=subject.layers,
    ):
        return QWEN3_MOE_LAYOUT_REASON
    if subject.moe is None or not _has_executable_sparse_layer(subject):
        return QWEN3_MOE_TOPOLOGY_REASON
    if any(index >= subject.layers for index in subject.moe.mlp_only_layers):
        return QWEN3_MOE_TOPOLOGY_REASON
    if subject.moe.shared_expert_intermediate_size is not None:
        return QWEN3_MOE_SHARED_EXPERT_REASON
    if subject.quantization_bits != 4:
        return QWEN3_MOE_FOUR_BIT_REASON
    return None


_QWEN3_MOE_POLICY = _ModelCompatibilityPolicy(
    family=QWEN3_MOE_FAMILY,
    claims=_qwen3_moe_claims,
    first_blocking_reason=_qwen3_moe_first_blocking_reason,
    paths=(
        _policy_path(
            family=QWEN3_MOE_FAMILY,
            method=Method.QLORA,
            training_runtime=TrainingRuntime.MLX_LM,
            compute_backend=Backend.MPS,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.ATTENTION_QKVO_V1,
        ),
    ),
    matched_reason=QWEN3_MOE_MATCHED_REASON,
)

MODEL_COMPATIBILITY_POLICIES: tuple[_ModelCompatibilityPolicy, ...] = (
    _QWEN3_MOE_POLICY,
)


def validate_model_policy_path(
    *,
    family: str,
    path: ModelPolicyPath,
) -> RuntimeContract:
    """Seal a domain policy path against adapter, method, and family registries."""

    if path.adapter_profile_id is None:
        raise ValueError("Registered model policy paths require an adapter profile.")
    if path.target_modules != adapter_target_modules(path.adapter_profile_id):
        raise ValueError("Model policy path targets do not match its adapter profile.")
    contract = validate_registered_compatibility_path(
        family=family,
        method=path.method,
        training_runtime=path.runtime_contract.training_runtime,
        compute_backend=path.runtime_contract.compute_backend,
        distribution=path.distribution,
        adapter_profile_id=path.adapter_profile_id,
        evidence_requirement=path.runtime_contract.evidence_requirement,
    )
    if path.runtime_contract != contract:
        raise ValueError(
            "Model policy path runtime contract differs from the method registry."
        )
    return contract


def subject_from_model(model: ModelSpec) -> ModelCompatibilitySubject:
    return ModelCompatibilitySubject(
        family=model.family,
        model_type=model.model_type,
        architecture=model.architecture,
        layers=model.layers,
        quantization_bits=model.quantization_bits,
        quantization_layout=model.quantization_layout,
        moe=model.moe,
    )


def evaluate_model_compatibility(
    subject: ModelCompatibilitySubject,
) -> ModelPolicyDecision:
    """Return artifact-policy status without deciding hardware feasibility."""

    if subject.fact_errors:
        claimed_qwen = _qwen3_moe_claims(subject)
        exact_qwen = _is_exact_qwen3_moe_subject(subject)
        return ModelPolicyDecision(
            kind=ModelPolicyDecisionKind.BLOCKED,
            family=QWEN3_MOE_FAMILY if claimed_qwen else subject.family,
            paths=(),
            reason=(
                INVALID_COMPATIBILITY_FACTS_REASON
                if not claimed_qwen or exact_qwen
                else QWEN3_MOE_IDENTITY_REASON
            ),
        )

    for policy in MODEL_COMPATIBILITY_POLICIES:
        if not policy.claims(subject):
            continue
        blocking_reason = policy.first_blocking_reason(subject)
        if blocking_reason is not None:
            return ModelPolicyDecision(
                kind=ModelPolicyDecisionKind.BLOCKED,
                family=policy.family,
                paths=(),
                reason=blocking_reason,
            )
        return ModelPolicyDecision(
            kind=ModelPolicyDecisionKind.PATH_MATCHED,
            family=policy.family,
            paths=policy.paths,
            reason=policy.matched_reason,
        )

    if (
        subject.moe is not None
        or _has_sparse_identity_marker(subject)
        or any(item.startswith("moe:") for item in subject.fact_errors)
    ):
        return ModelPolicyDecision(
            kind=ModelPolicyDecisionKind.BLOCKED,
            family=subject.family,
            paths=(),
            reason=UNREVIEWED_SPARSE_MODEL_REASON,
        )

    normalized_family = subject.family.lower() if subject.family is not None else None
    if (
        normalized_family is not None
        and TARGET_MODULES.get(normalized_family) == DENSE_CAUSAL_LM_TARGET_MODULES
    ):
        return ModelPolicyDecision(
            kind=ModelPolicyDecisionKind.FAMILY_RECOGNIZED,
            family=normalized_family,
            paths=(),
            reason=FAMILY_RECOGNIZED_REASON,
        )
    return ModelPolicyDecision(
        kind=ModelPolicyDecisionKind.UNKNOWN,
        family=subject.family,
        paths=(),
        reason=UNKNOWN_POLICY_REASON,
    )


def candidate_matches_policy_path(
    decision: ModelPolicyDecision,
    *,
    method: Method,
    distribution: Distribution,
    target_modules: tuple[str, ...],
    runtime_contract: RuntimeContract,
) -> bool:
    if decision.kind != ModelPolicyDecisionKind.PATH_MATCHED:
        return False
    return any(
        path.method == method
        and path.distribution == distribution
        and path.target_modules == target_modules
        and path.runtime_contract == runtime_contract
        for path in decision.paths
    )


def model_policy_rejection_reasons(
    decision: ModelPolicyDecision,
    *,
    method: Method,
    distribution: Distribution,
    target_modules: tuple[str, ...],
    runtime_contract: RuntimeContract,
) -> tuple[str, ...]:
    """Return only model-policy reasons, leaving feasibility to the planner."""

    if decision.kind in {
        ModelPolicyDecisionKind.FAMILY_RECOGNIZED,
        ModelPolicyDecisionKind.UNKNOWN,
    }:
        return ()
    if candidate_matches_policy_path(
        decision,
        method=method,
        distribution=distribution,
        target_modules=target_modules,
        runtime_contract=runtime_contract,
    ):
        return ()
    if decision.kind == ModelPolicyDecisionKind.BLOCKED:
        if decision.family == QWEN3_MOE_FAMILY:
            return tuple(dict.fromkeys((decision.reason, QWEN3_MOE_PATH_REASON)))
        return (decision.reason,)
    return (QWEN3_MOE_PATH_REASON,)


def compatibility_response_v1(
    decision: ModelPolicyDecision,
) -> dict[str, Any]:
    """Project one sealed domain decision into the unchanged API v1 shape."""

    if decision.kind == ModelPolicyDecisionKind.PATH_MATCHED:
        assert decision.family is not None  # enforced by the domain invariant
        for path in decision.paths:
            validate_model_policy_path(family=decision.family, path=path)
        first = decision.paths[0]
        assert first.adapter_profile_id is not None
        shared_claim = (
            first.runtime_contract.training_runtime,
            first.runtime_contract.compute_backend,
            first.distribution,
            first.adapter_profile_id,
            first.runtime_contract.evidence_requirement,
        )
        if any(
            (
                path.runtime_contract.training_runtime,
                path.runtime_contract.compute_backend,
                path.distribution,
                path.adapter_profile_id,
                path.runtime_contract.evidence_requirement,
            )
            != shared_claim
            for path in decision.paths[1:]
        ):
            raise ValueError(
                "The v1 compatibility response cannot flatten heterogeneous paths."
            )
        methods = tuple(dict.fromkeys(path.method for path in decision.paths))
        return {
            "status": "conditional",
            "family": decision.family,
            "supported_runtime": first.runtime_contract.training_runtime.value,
            "supported_methods": [method.value for method in methods],
            "compute_backend": first.runtime_contract.compute_backend.value,
            "distribution": first.distribution.value,
            "evidence_requirement": first.runtime_contract.evidence_requirement.value,
            "adapter_profile_id": first.adapter_profile_id.value,
            "reason": decision.reason,
        }
    if decision.kind == ModelPolicyDecisionKind.FAMILY_RECOGNIZED:
        return {
            "status": "recognized",
            "family": decision.family,
            "supported_runtime": None,
            "supported_methods": [],
            "compute_backend": None,
            "distribution": None,
            "evidence_requirement": "pilot-required",
            "adapter_profile_id": None,
            "reason": decision.reason,
        }
    exact_identity_blocked = (
        decision.kind == ModelPolicyDecisionKind.BLOCKED
        and decision.family == QWEN3_MOE_FAMILY
        and decision.reason in _QWEN3_EXACT_IDENTITY_BLOCKING_REASONS
    )
    return {
        "status": "unsupported",
        "family": decision.family,
        "supported_runtime": None,
        "supported_methods": [],
        "compute_backend": None,
        "distribution": None,
        "evidence_requirement": "implementation-required",
        "adapter_profile_id": None,
        "reason": (
            QWEN3_MOE_BLOCKED_INSPECTION_REASON
            if exact_identity_blocked
            else UNKNOWN_POLICY_REASON
        ),
    }
