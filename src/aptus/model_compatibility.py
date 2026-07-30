from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Callable, Mapping

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
    ModelInspectionProvenance,
    ModelInspectionReceipt,
    ModelPolicyBinding,
    ModelPolicyBindingSource,
    ModelPolicyDecision,
    ModelPolicyDecisionKind,
    ModelPolicyPath,
    ModelPolicyReasonCode,
    ModelSpec,
    MODEL_COMPATIBILITY_SCHEMA_VERSION,
    MODEL_INSPECTION_RECEIPT_SCHEMA_VERSION,
    MODEL_POLICY_BINDING_SCHEMA_VERSION,
    ProvenanceKind,
    Provenance,
    RuntimeContract,
    TrainingRuntime,
    to_primitive,
)
from .evidence import evidence_for
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
QWEN3_MOE_POLICY_ID = "model.qwen3-moe.mlx-qlora"
QWEN3_MOE_POLICY_VERSION = "1.0.0"
QWEN3_MOE_PATH_ID = "mlx-lm.qlora.single.attention-qkvo.v1"
QWEN3_MOE_POLICY_EVIDENCE_IDS = (
    "policy.qwen3-moe.mlx-qlora.v1",
    "admission.qwen3-30b-a3b.memory-blocked.2026-07-28",
)
QWEN3_MOE_REQUIRED_PROVENANCE_FIELDS = (
    "architecture",
    "layers",
    "model_type",
    "moe",
    "quantization_bits",
    "quantization_layout",
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
    policy_id: str
    policy_version: str
    family: str
    claims: Callable[[ModelCompatibilitySubject], bool]
    first_blocking_reason: Callable[[ModelCompatibilitySubject], str | None]
    paths: tuple[ModelPolicyPath, ...]
    matched_reason: str
    matched_reason_codes: tuple[ModelPolicyReasonCode, ...]
    evidence_ids: tuple[str, ...]
    required_provenance_fields: tuple[str, ...]


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
    path_id: str,
    family: str,
    method: Method,
    training_runtime: TrainingRuntime,
    compute_backend: Backend,
    distribution: Distribution,
    adapter_profile_id: AdapterProfile,
    evidence_ids: tuple[str, ...],
) -> ModelPolicyPath:
    target_modules = adapter_target_modules(adapter_profile_id)
    if TARGET_MODULES.get(family) != target_modules:
        raise RuntimeError(
            "Model policy adapter targets differ from the family catalog."
        )
    return ModelPolicyPath(
        path_id=path_id,
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
        required_validation_levels=("model-data", "measured-preflight", "pilot"),
        evidence_ids=evidence_ids,
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
    policy_id=QWEN3_MOE_POLICY_ID,
    policy_version=QWEN3_MOE_POLICY_VERSION,
    family=QWEN3_MOE_FAMILY,
    claims=_qwen3_moe_claims,
    first_blocking_reason=_qwen3_moe_first_blocking_reason,
    paths=(
        _policy_path(
            path_id=QWEN3_MOE_PATH_ID,
            family=QWEN3_MOE_FAMILY,
            method=Method.QLORA,
            training_runtime=TrainingRuntime.MLX_LM,
            compute_backend=Backend.MPS,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.ATTENTION_QKVO_V1,
            evidence_ids=QWEN3_MOE_POLICY_EVIDENCE_IDS,
        ),
    ),
    matched_reason=QWEN3_MOE_MATCHED_REASON,
    matched_reason_codes=(
        ModelPolicyReasonCode.EXACT_REVIEWED_ARTIFACT,
        ModelPolicyReasonCode.PILOT_NOT_YET_PROVEN,
    ),
    evidence_ids=QWEN3_MOE_POLICY_EVIDENCE_IDS,
    required_provenance_fields=QWEN3_MOE_REQUIRED_PROVENANCE_FIELDS,
)

MODEL_COMPATIBILITY_POLICIES: tuple[_ModelCompatibilityPolicy, ...] = (
    _QWEN3_MOE_POLICY,
)


def _validate_model_compatibility_policies() -> None:
    policy_ids = tuple(item.policy_id for item in MODEL_COMPATIBILITY_POLICIES)
    if len(set(policy_ids)) != len(policy_ids):
        raise RuntimeError("Model compatibility policy IDs must be unique.")
    path_ids = tuple(
        path.path_id for policy in MODEL_COMPATIBILITY_POLICIES for path in policy.paths
    )
    if len(set(path_ids)) != len(path_ids):
        raise RuntimeError("Model compatibility path IDs must be unique.")
    for policy in MODEL_COMPATIBILITY_POLICIES:
        if policy.required_provenance_fields != tuple(
            sorted(set(policy.required_provenance_fields))
        ):
            raise RuntimeError(
                "Model compatibility provenance fields must be sorted and unique."
            )
        evidence_for(*policy.evidence_ids)
        for path in policy.paths:
            evidence_for(*path.evidence_ids)


_validate_model_compatibility_policies()


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
    registered_path = next(
        (
            registered
            for policy in MODEL_COMPATIBILITY_POLICIES
            if policy.family == family
            for registered in policy.paths
            if registered.path_id == path.path_id
        ),
        None,
    )
    if registered_path is None or path != registered_path:
        raise ValueError(
            "Model policy path identity differs from the registered policy path."
        )
    evidence_for(*path.evidence_ids)
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


_RECEIPT_FACT_FIELDS = (
    "architecture",
    "context_length",
    "family",
    "hidden_size",
    "intermediate_size",
    "layers",
    "license_name",
    "model_type",
    "moe",
    "quantization_bits",
    "quantization_layout",
)

_COMPATIBILITY_SUBJECT_FACT_FIELDS = (
    "architecture",
    "family",
    "layers",
    "model_type",
    "moe",
    "quantization_bits",
    "quantization_layout",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _content_id(prefix: str, value: Any) -> str:
    return prefix + _sha256(value)[:20]


def compatibility_subject_payload(subject: ModelCompatibilitySubject) -> dict[str, Any]:
    """Return the canonical compatibility-only facts used by decision identity."""

    return {
        "family": subject.family,
        "model_type": subject.model_type,
        "architecture": subject.architecture,
        "layers": subject.layers,
        "quantization_bits": subject.quantization_bits,
        "quantization_layout": to_primitive(subject.quantization_layout),
        "moe": to_primitive(subject.moe),
        "fact_errors": sorted(subject.fact_errors),
    }


def compatibility_subject_sha256(subject: ModelCompatibilitySubject) -> str:
    return _sha256(compatibility_subject_payload(subject))


def _required_subject_provenance_fields(
    subject: ModelCompatibilitySubject,
) -> set[str]:
    payload = compatibility_subject_payload(subject)
    return {
        field
        for field in _COMPATIBILITY_SUBJECT_FACT_FIELDS
        if payload[field] is not None
    }


def _decision_identity_payload(
    *,
    subject_facts_sha256: str,
    kind: ModelPolicyDecisionKind,
    family: str | None,
    policy_id: str | None,
    policy_version: str | None,
    paths: tuple[ModelPolicyPath, ...],
    reason_codes: tuple[ModelPolicyReasonCode, ...],
    evidence_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_COMPATIBILITY_SCHEMA_VERSION,
        "subject_facts_sha256": subject_facts_sha256,
        "kind": kind.value,
        "family": family,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "paths": to_primitive(paths),
        "reason_codes": [item.value for item in reason_codes],
        "evidence_ids": list(evidence_ids),
    }


def _decision(
    subject: ModelCompatibilitySubject,
    *,
    kind: ModelPolicyDecisionKind,
    family: str | None,
    policy: _ModelCompatibilityPolicy | None,
    paths: tuple[ModelPolicyPath, ...],
    reason_codes: tuple[ModelPolicyReasonCode, ...],
    evidence_ids: tuple[str, ...],
    reason: str,
) -> ModelPolicyDecision:
    subject_digest = compatibility_subject_sha256(subject)
    policy_id = policy.policy_id if policy is not None else None
    policy_version = policy.policy_version if policy is not None else None
    identity = _decision_identity_payload(
        subject_facts_sha256=subject_digest,
        kind=kind,
        family=family,
        policy_id=policy_id,
        policy_version=policy_version,
        paths=paths,
        reason_codes=reason_codes,
        evidence_ids=evidence_ids,
    )
    return ModelPolicyDecision(
        schema_version=MODEL_COMPATIBILITY_SCHEMA_VERSION,
        decision_id=_content_id("compat_", identity),
        subject_facts_sha256=subject_digest,
        kind=kind,
        family=family,
        policy_id=policy_id,
        policy_version=policy_version,
        paths=paths,
        reason_codes=reason_codes,
        evidence_ids=evidence_ids,
        reason=reason,
    )


_BLOCKING_REASON_CODES = {
    QWEN3_MOE_IDENTITY_REASON: ModelPolicyReasonCode.IDENTITY_MISMATCH,
    QWEN3_MOE_LAYOUT_REASON: ModelPolicyReasonCode.QUANTIZATION_LAYOUT_MISMATCH,
    QWEN3_MOE_TOPOLOGY_REASON: ModelPolicyReasonCode.TOPOLOGY_INCOMPLETE,
    QWEN3_MOE_SHARED_EXPERT_REASON: ModelPolicyReasonCode.SHARED_EXPERT_UNSUPPORTED,
    QWEN3_MOE_FOUR_BIT_REASON: ModelPolicyReasonCode.FOUR_BIT_REQUIRED,
    INVALID_COMPATIBILITY_FACTS_REASON: ModelPolicyReasonCode.INVALID_FACTS,
}


def _normalized_observed_fact(field: str, value: Any) -> Any:
    if field in {"quantization_layout", "moe"}:
        return to_primitive(value)
    return value


def _observed_facts_payload(
    facts: Mapping[str, Any],
    provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        field: _normalized_observed_fact(field, facts.get(field))
        for field in _RECEIPT_FACT_FIELDS
        if field in provenance
    }


def _observed_model_facts_payload(
    model: ModelSpec,
    provenance_summary: tuple[ModelInspectionProvenance, ...],
) -> dict[str, Any]:
    return {
        item.field: _normalized_observed_fact(item.field, getattr(model, item.field))
        for item in provenance_summary
    }


def _receipt_identity_payload(
    *,
    model_id: str,
    resolved_revision: str,
    observed_facts_sha256: str,
    decision: ModelPolicyDecision,
    provenance_summary: tuple[ModelInspectionProvenance, ...],
    provenance_requirement: ProvenanceKind | None,
    provenance_requirement_met: bool,
    evaluated_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_INSPECTION_RECEIPT_SCHEMA_VERSION,
        "model_id": model_id,
        "resolved_revision": resolved_revision.lower(),
        "observed_facts_sha256": observed_facts_sha256,
        "decision_id": decision.decision_id,
        "provenance_summary": to_primitive(provenance_summary),
        "provenance_requirement": (
            provenance_requirement.value if provenance_requirement is not None else None
        ),
        "provenance_requirement_met": provenance_requirement_met,
        "evaluated_at": evaluated_at,
    }


def _policy_for_decision(
    decision: ModelPolicyDecision,
) -> _ModelCompatibilityPolicy | None:
    if decision.policy_id is None:
        return None
    return next(
        (
            policy
            for policy in MODEL_COMPATIBILITY_POLICIES
            if policy.policy_id == decision.policy_id
            and policy.policy_version == decision.policy_version
        ),
        None,
    )


def create_model_inspection_receipt(
    *,
    model_id: str,
    resolved_revision: str,
    facts: Mapping[str, Any],
    provenance: Mapping[str, Mapping[str, Any]],
    subject: ModelCompatibilitySubject,
    evaluated_at: str,
) -> ModelInspectionReceipt:
    """Create a content-bound receipt for one revision-resolved observation."""

    decision = evaluate_model_compatibility(subject)
    observed_facts = _observed_facts_payload(facts, provenance)
    summary: list[ModelInspectionProvenance] = []
    for field in sorted(observed_facts):
        item = provenance.get(field)
        if not isinstance(item, Mapping):
            raise ValueError(f"Inspection provenance is missing for {field}.")
        summary.append(
            ModelInspectionProvenance(
                field=field,
                kind=ProvenanceKind(item["kind"]),
                source=str(item["source"]),
                observed_at=str(item["observed_at"]),
                resolved_revision=str(item["resolved_revision"]),
            )
        )
    provenance_summary = tuple(summary)
    required_subject_fields = _required_subject_provenance_fields(subject)
    missing_subject_fields = required_subject_fields.difference(
        item.field for item in provenance_summary
    )
    if missing_subject_fields:
        raise ValueError(
            "Inspection receipt provenance does not cover compatibility subject "
            "facts: " + ", ".join(sorted(missing_subject_fields)) + "."
        )
    if not any(
        item.field in required_subject_fields
        and item.kind == ProvenanceKind.PROVIDER_DECLARED
        for item in provenance_summary
    ):
        raise ValueError(
            "Inspection receipt compatibility facts require at least one "
            "provider-declared observation."
        )
    policy = _policy_for_decision(decision)
    requirement = ProvenanceKind.PROVIDER_DECLARED if policy is not None else None
    summary_by_field = {item.field: item for item in provenance_summary}
    requirement_met = bool(
        policy is not None
        and all(
            field in summary_by_field
            and summary_by_field[field].kind == ProvenanceKind.PROVIDER_DECLARED
            and summary_by_field[field].resolved_revision.lower()
            == resolved_revision.lower()
            for field in policy.required_provenance_fields
        )
    )
    observed_digest = _sha256(observed_facts)
    identity = _receipt_identity_payload(
        model_id=model_id,
        resolved_revision=resolved_revision,
        observed_facts_sha256=observed_digest,
        decision=decision,
        provenance_summary=provenance_summary,
        provenance_requirement=requirement,
        provenance_requirement_met=requirement_met,
        evaluated_at=evaluated_at,
    )
    return ModelInspectionReceipt(
        schema_version=MODEL_INSPECTION_RECEIPT_SCHEMA_VERSION,
        receipt_id=_content_id("receipt_", identity),
        model_id=model_id,
        resolved_revision=resolved_revision.lower(),
        observed_facts_sha256=observed_digest,
        decision=decision,
        provenance_summary=provenance_summary,
        provenance_requirement=requirement,
        provenance_requirement_met=requirement_met,
        evaluated_at=evaluated_at,
    )


def validate_model_inspection_receipt(
    *,
    receipt: ModelInspectionReceipt,
    model: ModelSpec,
    decision: ModelPolicyDecision,
) -> None:
    """Recompute every receipt binding before it can label a plan."""

    if receipt.model_id != model.model_id:
        raise ValueError("Inspection receipt model ID does not match the plan facts.")
    if receipt.resolved_revision.lower() != model.revision.lower():
        raise ValueError("Inspection receipt revision does not match the plan facts.")
    observed_fields = tuple(item.field for item in receipt.provenance_summary)
    if any(field not in _RECEIPT_FACT_FIELDS for field in observed_fields):
        raise ValueError("Inspection receipt contains an unknown planning fact.")
    required_subject_fields = _required_subject_provenance_fields(
        subject_from_model(model)
    )
    missing_subject_fields = required_subject_fields.difference(observed_fields)
    if missing_subject_fields:
        raise ValueError(
            "Inspection receipt provenance does not cover compatibility subject "
            "facts: " + ", ".join(sorted(missing_subject_fields)) + "."
        )
    if not any(
        item.field in required_subject_fields
        and item.kind == ProvenanceKind.PROVIDER_DECLARED
        for item in receipt.provenance_summary
    ):
        raise ValueError(
            "Inspection receipt compatibility facts require at least one "
            "provider-declared observation."
        )
    current_observed_digest = _sha256(
        _observed_model_facts_payload(model, receipt.provenance_summary)
    )
    if receipt.observed_facts_sha256 != current_observed_digest:
        raise ValueError(
            "Inspection receipt observed facts do not match the plan facts."
        )
    if not model_policy_decisions_match(receipt.decision, decision):
        raise ValueError(
            "Inspection receipt decision is stale or does not match the facts."
        )
    if any(
        item.resolved_revision.lower() != receipt.resolved_revision.lower()
        for item in receipt.provenance_summary
    ):
        raise ValueError("Inspection receipt provenance revision is inconsistent.")
    policy = _policy_for_decision(decision)
    expected_requirement = (
        ProvenanceKind.PROVIDER_DECLARED if policy is not None else None
    )
    summary_by_field = {item.field: item for item in receipt.provenance_summary}
    expected_met = bool(
        policy is not None
        and all(
            field in summary_by_field
            and summary_by_field[field].kind == ProvenanceKind.PROVIDER_DECLARED
            for field in policy.required_provenance_fields
        )
    )
    if receipt.provenance_requirement != expected_requirement:
        raise ValueError("Inspection receipt provenance requirement is stale.")
    if receipt.provenance_requirement_met != expected_met:
        raise ValueError("Inspection receipt provenance summary is inconsistent.")
    if decision.kind == ModelPolicyDecisionKind.PATH_MATCHED and not expected_met:
        raise ValueError(
            "A matched provider policy receipt requires provider-declared path facts."
        )
    identity = _receipt_identity_payload(
        model_id=receipt.model_id,
        resolved_revision=receipt.resolved_revision,
        observed_facts_sha256=receipt.observed_facts_sha256,
        decision=receipt.decision,
        provenance_summary=receipt.provenance_summary,
        provenance_requirement=receipt.provenance_requirement,
        provenance_requirement_met=receipt.provenance_requirement_met,
        evaluated_at=receipt.evaluated_at,
    )
    if receipt.receipt_id != _content_id("receipt_", identity):
        raise ValueError("Inspection receipt immutable ID does not match its content.")


def _decision_semantic_payload(decision: ModelPolicyDecision) -> dict[str, Any]:
    return _decision_identity_payload(
        subject_facts_sha256=decision.subject_facts_sha256,
        kind=decision.kind,
        family=decision.family,
        policy_id=decision.policy_id,
        policy_version=decision.policy_version,
        paths=decision.paths,
        reason_codes=decision.reason_codes,
        evidence_ids=decision.evidence_ids,
    )


def model_policy_decisions_match(
    left: ModelPolicyDecision,
    right: ModelPolicyDecision,
) -> bool:
    """Compare decision semantics while excluding explanatory prose."""

    return bool(
        left.schema_version == right.schema_version
        and left.decision_id == right.decision_id
        and _decision_semantic_payload(left) == _decision_semantic_payload(right)
        and left.decision_id == _content_id("compat_", _decision_semantic_payload(left))
    )


def validate_current_model_policy_decision(
    *,
    decision: ModelPolicyDecision,
    model: ModelSpec,
) -> ModelPolicyDecision:
    """Fail closed when persisted policy output differs from today's registry."""

    current = evaluate_model_compatibility(subject_from_model(model))
    if not model_policy_decisions_match(decision, current):
        raise ValueError("Model policy decision is stale or does not match the facts.")
    if current.kind == ModelPolicyDecisionKind.PATH_MATCHED:
        assert current.family is not None
        for path in current.paths:
            validate_model_policy_path(family=current.family, path=path)
    return current


def model_with_inspection_provenance(
    model: ModelSpec,
    receipt: ModelInspectionReceipt,
) -> ModelSpec:
    """Carry receipt-backed fact sources into the persisted model ledger."""

    default_user_attested = Provenance(
        kind=ProvenanceKind.USER_ATTESTED,
        source="cli-or-api",
    )
    shared_user_attested = model.provenance.get("all")
    if (
        shared_user_attested is None
        or shared_user_attested.kind != ProvenanceKind.USER_ATTESTED
    ):
        shared_user_attested = default_user_attested

    def user_attested_for(field: str) -> Provenance:
        current = model.provenance.get(field)
        if current is not None and current.kind == ProvenanceKind.USER_ATTESTED:
            return current
        return shared_user_attested

    provenance: dict[str, Provenance] = {
        field: user_attested_for(field) for field in _RECEIPT_FACT_FIELDS
    }
    provenance["parameters"] = user_attested_for("parameters")
    provenance["training_allowed"] = user_attested_for("training_allowed")
    for item in receipt.provenance_summary:
        if item.field not in _RECEIPT_FACT_FIELDS:
            raise ValueError("Inspection receipt contains an unknown planning fact.")
        provenance[item.field] = Provenance(
            kind=item.kind,
            source=item.source,
            observed_at=item.observed_at,
            digest=receipt.receipt_id,
            detail=f"Provider observation at immutable revision {item.resolved_revision}.",
        )
    return replace(model, provenance=provenance)


def model_with_user_attested_provenance(model: ModelSpec) -> ModelSpec:
    """Close the receipt-free path as an explicit user attestation."""

    provenance = model.provenance.get("all")
    if provenance is None or provenance.kind != ProvenanceKind.USER_ATTESTED:
        provenance = Provenance(
            kind=ProvenanceKind.USER_ATTESTED,
            source="cli-or-api",
        )
    return replace(model, provenance={"all": provenance})


def matching_model_policy_path(
    decision: ModelPolicyDecision,
    *,
    method: Method,
    distribution: Distribution,
    target_modules: tuple[str, ...],
    runtime_contract: RuntimeContract,
) -> ModelPolicyPath | None:
    if decision.kind != ModelPolicyDecisionKind.PATH_MATCHED:
        return None
    return next(
        (
            path
            for path in decision.paths
            if path.method == method
            and path.distribution == distribution
            and path.target_modules == target_modules
            and path.runtime_contract == runtime_contract
        ),
        None,
    )


def model_policy_binding_for_path(
    *,
    decision: ModelPolicyDecision,
    path: ModelPolicyPath,
    receipt: ModelInspectionReceipt | None,
) -> ModelPolicyBinding:
    if decision.policy_id is None or decision.policy_version is None:
        raise ValueError("A selected model policy path requires a policy identity.")
    if path not in decision.paths:
        raise ValueError("Selected model policy path is not part of the decision.")
    source = (
        ModelPolicyBindingSource.PROVIDER_INSPECTION
        if receipt is not None
        else ModelPolicyBindingSource.USER_ATTESTED
    )
    return ModelPolicyBinding(
        schema_version=MODEL_POLICY_BINDING_SCHEMA_VERSION,
        decision_id=decision.decision_id,
        subject_facts_sha256=decision.subject_facts_sha256,
        policy_id=decision.policy_id,
        policy_version=decision.policy_version,
        path_id=path.path_id,
        source=source,
        inspection_receipt_id=receipt.receipt_id if receipt is not None else None,
        reason_codes=decision.reason_codes,
        evidence_ids=tuple(dict.fromkeys(decision.evidence_ids + path.evidence_ids)),
    )


def evaluate_model_compatibility(
    subject: ModelCompatibilitySubject,
) -> ModelPolicyDecision:
    """Return artifact-policy status without deciding hardware feasibility."""

    if subject.fact_errors:
        claimed_qwen = _qwen3_moe_claims(subject)
        exact_qwen = _is_exact_qwen3_moe_subject(subject)
        reason = (
            INVALID_COMPATIBILITY_FACTS_REASON
            if not claimed_qwen or exact_qwen
            else QWEN3_MOE_IDENTITY_REASON
        )
        policy = _QWEN3_MOE_POLICY if claimed_qwen else None
        return _decision(
            subject,
            kind=ModelPolicyDecisionKind.BLOCKED,
            family=QWEN3_MOE_FAMILY if claimed_qwen else subject.family,
            policy=policy,
            paths=(),
            reason_codes=(_BLOCKING_REASON_CODES[reason],),
            evidence_ids=policy.evidence_ids if policy is not None else (),
            reason=reason,
        )

    for policy in MODEL_COMPATIBILITY_POLICIES:
        if not policy.claims(subject):
            continue
        blocking_reason = policy.first_blocking_reason(subject)
        if blocking_reason is not None:
            return _decision(
                subject,
                kind=ModelPolicyDecisionKind.BLOCKED,
                family=policy.family,
                policy=policy,
                paths=(),
                reason_codes=(_BLOCKING_REASON_CODES[blocking_reason],),
                evidence_ids=policy.evidence_ids,
                reason=blocking_reason,
            )
        return _decision(
            subject,
            kind=ModelPolicyDecisionKind.PATH_MATCHED,
            family=policy.family,
            policy=policy,
            paths=policy.paths,
            reason_codes=policy.matched_reason_codes,
            evidence_ids=policy.evidence_ids,
            reason=policy.matched_reason,
        )

    if (
        subject.moe is not None
        or _has_sparse_identity_marker(subject)
        or any(item.startswith("moe:") for item in subject.fact_errors)
    ):
        return _decision(
            subject,
            kind=ModelPolicyDecisionKind.BLOCKED,
            family=subject.family,
            policy=None,
            paths=(),
            reason_codes=(ModelPolicyReasonCode.UNREVIEWED_SPARSE_MODEL,),
            evidence_ids=(),
            reason=UNREVIEWED_SPARSE_MODEL_REASON,
        )

    normalized_family = subject.family.lower() if subject.family is not None else None
    if (
        normalized_family is not None
        and TARGET_MODULES.get(normalized_family) == DENSE_CAUSAL_LM_TARGET_MODULES
    ):
        return _decision(
            subject,
            kind=ModelPolicyDecisionKind.FAMILY_RECOGNIZED,
            family=normalized_family,
            policy=None,
            paths=(),
            reason_codes=(ModelPolicyReasonCode.FAMILY_RECOGNIZED,),
            evidence_ids=(),
            reason=FAMILY_RECOGNIZED_REASON,
        )
    return _decision(
        subject,
        kind=ModelPolicyDecisionKind.UNKNOWN,
        family=subject.family,
        policy=None,
        paths=(),
        reason_codes=(ModelPolicyReasonCode.NO_POLICY_MATCH,),
        evidence_ids=(),
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
