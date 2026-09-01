from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping

from .catalog import (
    DENSE_CAUSAL_LM_TARGET_MODULES,
    QWEN3_MOE_ARCHITECTURE,
    QWEN3_MOE_FAMILY,
    QWEN3_MOE_MODEL_TYPE,
    QWEN3_MOE_TARGET_MODULES,
    TARGET_MODULES,
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
    model_policy_decision_from_primitive,
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
from .policy_snapshot import (
    apply_operator_unreviewed_runtime_confirm,
    evaluate_model_policy_snapshot,
    model_policy_snapshot_bytes,
    model_policy_snapshot_payload,
    model_policy_snapshot_sha256,
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

QWEN2_FAMILY = "qwen"
QWEN2_MODEL_TYPE = "qwen2"
QWEN2_ARCHITECTURE = "Qwen2ForCausalLM"
QWEN2_LAYERS = 24
QWEN2_IDENTITY_REASON = (
    "Dense Qwen2 execution requires the reviewed qwen, qwen2, and "
    "Qwen2ForCausalLM provider identity."
)
QWEN2_LAYER_REASON = (
    "The reviewed Qwen2 MLX-LM runtime footprint requires exactly 24 "
    "transformer layers."
)
QWEN2_LAYOUT_REASON = (
    "The reviewed Qwen2 MLX-LM runtime footprint requires a uniform four-bit, "
    "group-size-64 quantization layout with no module overrides."
)
QWEN2_DENSE_REASON = (
    "The reviewed Qwen2 MLX-LM runtime footprint requires dense topology with "
    "no MoE configuration."
)
QWEN2_FOUR_BIT_REASON = (
    "The reviewed Qwen2 MLX-LM runtime footprint requires explicit four-bit "
    "model metadata."
)
QWEN2_PATH_REASON = (
    "The reviewed Qwen2 runtime footprint is executable only as single-device "
    "MLX-LM QLoRA with dense q/k/v/o/gate/up/down adapters."
)
QWEN2_MATCHED_REASON = (
    "The provider identity and 24-layer dense four-bit, group-size-64 "
    "configuration match the reviewed Qwen2 MLX-LM runtime footprint. Runtime "
    "evidence remains scoped to the pinned artifact, and every execution still "
    "requires model-data, measured-preflight, and pilot validation."
)
QWEN2_BLOCKED_INSPECTION_REASON = (
    "The Qwen2 identity was recognized, but this configuration does not match "
    "the reviewed 24-layer dense four-bit, group-size-64 runtime footprint."
)
QWEN2_POLICY_ID = "model.qwen2-24l.mlx-qlora"
QWEN2_POLICY_VERSION = "1.0.0"
QWEN2_PATH_ID = "mlx-lm.qlora.single.dense-causal-lm.v1"
QWEN2_POLICY_EVIDENCE_IDS = (
    "policy.qwen2-24l.mlx-qlora.v1",
    "runtime.qwen2-0.5b.mlx-qlora.2026-07-27",
)
QWEN2_REQUIRED_PROVENANCE_FIELDS = (
    "architecture",
    "layers",
    "model_type",
    "quantization_bits",
    "quantization_layout",
)

GEMMA4_FAMILY = "gemma4"
GEMMA4_MODEL_TYPE = "gemma4_text"
GEMMA4_ARCHITECTURE = "Gemma4ForConditionalGeneration"
GEMMA4_IDENTITY_REASON = (
    "Dense Gemma 4 execution requires the gemma4 family, gemma4_text provider "
    "type, and Gemma4ForConditionalGeneration architecture."
)
GEMMA4_DENSE_REASON = (
    "The Gemma 4 MLX-LM family path requires dense topology with no MoE "
    "configuration. Sparse Gemma 4 remains a later named slice."
)
GEMMA4_PATH_REASON = (
    "The Gemma 4 runtime is executable only as single-device MLX-LM LoRA or "
    "QLoRA with dense q/k/v/o/gate/up/down adapters. QLoRA requires declared "
    "quantization bits from 1 through 16; LoRA requires an unquantized base."
)
GEMMA4_MATCHED_REASON = (
    "The provider identity matches the Gemma 4 dense MLX-LM family. Size and "
    "bitwidth come from the pinned revision. Runtime evidence remains "
    "artifact-scoped, and every execution still requires model-data, "
    "measured-preflight, and pilot validation."
)
GEMMA4_BLOCKED_INSPECTION_REASON = (
    "The Gemma 4 identity was recognized, but this configuration is not the "
    "dense language-tower path."
)
GEMMA4_POLICY_ID = "model.gemma4.mlx.v1"
GEMMA4_POLICY_VERSION = "1.1.0"
GEMMA4_QLORA_PATH_ID = "mlx-lm.qlora.single.gemma4-dense.v1"
GEMMA4_LORA_PATH_ID = "mlx-lm.lora.single.gemma4-dense.v1"
GEMMA4_POLICY_EVIDENCE_IDS = ("policy.gemma4.mlx.v1",)
GEMMA4_REQUIRED_PROVENANCE_FIELDS = (
    "architecture",
    "layers",
    "model_type",
)
# Quantization bits/layout stay off this list: the LoRA path is unquantized and
# has no provider-declared bits. QLoRA still re-checks declared bits at plan
# and train. Requiring those fields here would refuse unquantized LoRA receipts.

GEMMA4_UNIFIED_MODEL_TYPE = "gemma4_unified_text"
GEMMA4_UNIFIED_ARCHITECTURE = "Gemma4UnifiedForConditionalGeneration"
GEMMA4_UNIFIED_IDENTITY_REASON = (
    "Gemma 4 unified execution requires the gemma4 family, gemma4_unified_text "
    "provider type, and Gemma4UnifiedForConditionalGeneration architecture."
)
GEMMA4_UNIFIED_COMPILER_REASON = (
    "The Gemma 4 unified identity is recognized, but the bound MLX-LM compiler "
    "does not load Gemma4UnifiedForConditionalGeneration. This is unsupported "
    "by the current compiler contract, not an unknown family."
)
GEMMA4_UNIFIED_MATCHED_REASON = (
    "The provider identity matches the Gemma 4 unified MLX-LM family. Size and "
    "bitwidth come from the pinned revision. Runtime evidence remains "
    "artifact-scoped, and every execution still requires model-data, "
    "measured-preflight, and pilot validation."
)
GEMMA4_UNIFIED_BLOCKED_INSPECTION_REASON = GEMMA4_UNIFIED_COMPILER_REASON
GEMMA4_UNIFIED_POLICY_ID = "model.gemma4-unified.mlx.v1"
GEMMA4_UNIFIED_POLICY_VERSION = "1.0.0"
GEMMA4_UNIFIED_QLORA_PATH_ID = "mlx-lm.qlora.single.gemma4-unified.v1"
GEMMA4_UNIFIED_LORA_PATH_ID = "mlx-lm.lora.single.gemma4-unified.v1"
GEMMA4_UNIFIED_POLICY_EVIDENCE_IDS = ("policy.gemma4-unified.mlx.v1",)
GEMMA4_UNIFIED_REQUIRED_PROVENANCE_FIELDS = (
    "architecture",
    "layers",
    "model_type",
)


ADAPTER_PROFILE_TARGET_MODULES: dict[AdapterProfile, tuple[str, ...]] = {
    AdapterProfile.ATTENTION_QKVO_V1: QWEN3_MOE_TARGET_MODULES,
    AdapterProfile.DENSE_CAUSAL_LM_V1: DENSE_CAUSAL_LM_TARGET_MODULES,
}


@dataclass(frozen=True)
class _ModelCompatibilityPolicy:
    policy_id: str
    policy_version: str
    family: str
    claims: Mapping[str, tuple[str, ...]]
    constraints: tuple[Mapping[str, Any], ...]
    paths: tuple[ModelPolicyPath, ...]
    matched_reason_key: str
    matched_reason: str
    matched_reason_codes: tuple[ModelPolicyReasonCode, ...]
    evidence_ids: tuple[str, ...]
    required_provenance_fields: tuple[str, ...]
    path_rejection_reason: str
    blocked_inspection_reason: str
    inspection_blocking_reason_codes: tuple[ModelPolicyReasonCode, ...]


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


_QWEN3_MOE_POLICY = _ModelCompatibilityPolicy(
    policy_id=QWEN3_MOE_POLICY_ID,
    policy_version=QWEN3_MOE_POLICY_VERSION,
    family=QWEN3_MOE_FAMILY,
    claims={
        "architecture": (QWEN3_MOE_ARCHITECTURE,),
        "family": (QWEN3_MOE_FAMILY,),
        "model_type": (QWEN3_MOE_MODEL_TYPE,),
    },
    constraints=(
        {
            "kind": "exact_identity",
            "values": {
                "architecture": QWEN3_MOE_ARCHITECTURE,
                "family": QWEN3_MOE_FAMILY,
                "model_type": QWEN3_MOE_MODEL_TYPE,
            },
            "reason": "identity",
            "reason_code": ModelPolicyReasonCode.IDENTITY_MISMATCH.value,
        },
        {
            "kind": "quantization_layout",
            "default_bits": 4,
            "default_group_size": 64,
            "override_module_template": "model.layers.{layer}.mlp.gate",
            "override_bits": 8,
            "override_group_size": 64,
            "reason": "layout",
            "reason_code": ModelPolicyReasonCode.QUANTIZATION_LAYOUT_MISMATCH.value,
        },
        {
            "kind": "sparse_topology",
            "reason": "topology",
            "reason_code": ModelPolicyReasonCode.TOPOLOGY_INCOMPLETE.value,
        },
        {
            "kind": "no_shared_expert",
            "reason": "shared",
            "reason_code": ModelPolicyReasonCode.SHARED_EXPERT_UNSUPPORTED.value,
        },
        {
            "kind": "field_equals",
            "field": "quantization_bits",
            "value": 4,
            "reason": "four_bit",
            "reason_code": ModelPolicyReasonCode.FOUR_BIT_REQUIRED.value,
        },
    ),
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
    matched_reason_key="matched",
    matched_reason=QWEN3_MOE_MATCHED_REASON,
    matched_reason_codes=(
        ModelPolicyReasonCode.EXACT_REVIEWED_ARTIFACT,
        ModelPolicyReasonCode.PILOT_NOT_YET_PROVEN,
    ),
    evidence_ids=QWEN3_MOE_POLICY_EVIDENCE_IDS,
    required_provenance_fields=QWEN3_MOE_REQUIRED_PROVENANCE_FIELDS,
    path_rejection_reason=QWEN3_MOE_PATH_REASON,
    blocked_inspection_reason=QWEN3_MOE_BLOCKED_INSPECTION_REASON,
    inspection_blocking_reason_codes=(
        ModelPolicyReasonCode.INVALID_FACTS,
        ModelPolicyReasonCode.QUANTIZATION_LAYOUT_MISMATCH,
        ModelPolicyReasonCode.TOPOLOGY_INCOMPLETE,
        ModelPolicyReasonCode.SHARED_EXPERT_UNSUPPORTED,
        ModelPolicyReasonCode.FOUR_BIT_REQUIRED,
    ),
)

_QWEN2_POLICY = _ModelCompatibilityPolicy(
    policy_id=QWEN2_POLICY_ID,
    policy_version=QWEN2_POLICY_VERSION,
    family=QWEN2_FAMILY,
    claims={
        "architecture": (QWEN2_ARCHITECTURE,),
        "model_type": (QWEN2_MODEL_TYPE,),
    },
    constraints=(
        {
            "kind": "exact_identity",
            "values": {
                "architecture": QWEN2_ARCHITECTURE,
                "family": QWEN2_FAMILY,
                "model_type": QWEN2_MODEL_TYPE,
            },
            "reason": "qwen2_identity",
            "reason_code": ModelPolicyReasonCode.IDENTITY_MISMATCH.value,
        },
        {
            "kind": "field_equals",
            "field": "layers",
            "value": QWEN2_LAYERS,
            "reason": "qwen2_layers",
            "reason_code": ModelPolicyReasonCode.LAYER_COUNT_MISMATCH.value,
        },
        {
            "kind": "field_equals",
            "field": "quantization_bits",
            "value": 4,
            "reason": "qwen2_four_bit",
            "reason_code": ModelPolicyReasonCode.FOUR_BIT_REQUIRED.value,
        },
        {
            "kind": "field_equals",
            "field": "quantization_layout",
            "value": {
                "default_bits": 4,
                "default_group_size": 64,
                "module_overrides": [],
            },
            "reason": "qwen2_layout",
            "reason_code": ModelPolicyReasonCode.QUANTIZATION_LAYOUT_MISMATCH.value,
        },
        {
            "kind": "field_equals",
            "field": "moe",
            "value": None,
            "reason": "qwen2_dense",
            "reason_code": ModelPolicyReasonCode.DENSE_TOPOLOGY_REQUIRED.value,
        },
    ),
    paths=(
        _policy_path(
            path_id=QWEN2_PATH_ID,
            family=QWEN2_FAMILY,
            method=Method.QLORA,
            training_runtime=TrainingRuntime.MLX_LM,
            compute_backend=Backend.MPS,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.DENSE_CAUSAL_LM_V1,
            evidence_ids=QWEN2_POLICY_EVIDENCE_IDS,
        ),
    ),
    matched_reason_key="qwen2_matched",
    matched_reason=QWEN2_MATCHED_REASON,
    matched_reason_codes=(
        ModelPolicyReasonCode.REVIEWED_RUNTIME_PATH,
        ModelPolicyReasonCode.PILOT_NOT_YET_PROVEN,
    ),
    evidence_ids=QWEN2_POLICY_EVIDENCE_IDS,
    required_provenance_fields=QWEN2_REQUIRED_PROVENANCE_FIELDS,
    path_rejection_reason=QWEN2_PATH_REASON,
    blocked_inspection_reason=QWEN2_BLOCKED_INSPECTION_REASON,
    inspection_blocking_reason_codes=(
        ModelPolicyReasonCode.INVALID_FACTS,
        ModelPolicyReasonCode.LAYER_COUNT_MISMATCH,
        ModelPolicyReasonCode.QUANTIZATION_LAYOUT_MISMATCH,
        ModelPolicyReasonCode.DENSE_TOPOLOGY_REQUIRED,
        ModelPolicyReasonCode.FOUR_BIT_REQUIRED,
    ),
)

_GEMMA4_POLICY = _ModelCompatibilityPolicy(
    policy_id=GEMMA4_POLICY_ID,
    policy_version=GEMMA4_POLICY_VERSION,
    family=GEMMA4_FAMILY,
    claims={
        "architecture": (GEMMA4_ARCHITECTURE,),
        "model_type": (GEMMA4_MODEL_TYPE, "gemma4"),
    },
    constraints=(
        {
            "kind": "exact_identity",
            "values": {
                "architecture": GEMMA4_ARCHITECTURE,
                "family": GEMMA4_FAMILY,
                "model_type": GEMMA4_MODEL_TYPE,
            },
            "reason": "gemma4_identity",
            "reason_code": ModelPolicyReasonCode.IDENTITY_MISMATCH.value,
        },
        {
            "kind": "field_equals",
            "field": "moe",
            "value": None,
            "reason": "gemma4_dense",
            "reason_code": ModelPolicyReasonCode.DENSE_TOPOLOGY_REQUIRED.value,
        },
    ),
    paths=(
        _policy_path(
            path_id=GEMMA4_QLORA_PATH_ID,
            family=GEMMA4_FAMILY,
            method=Method.QLORA,
            training_runtime=TrainingRuntime.MLX_LM,
            compute_backend=Backend.MPS,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.DENSE_CAUSAL_LM_V1,
            evidence_ids=GEMMA4_POLICY_EVIDENCE_IDS,
        ),
        _policy_path(
            path_id=GEMMA4_LORA_PATH_ID,
            family=GEMMA4_FAMILY,
            method=Method.LORA,
            training_runtime=TrainingRuntime.MLX_LM,
            compute_backend=Backend.MPS,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.DENSE_CAUSAL_LM_V1,
            evidence_ids=GEMMA4_POLICY_EVIDENCE_IDS,
        ),
    ),
    matched_reason_key="gemma4_matched",
    matched_reason=GEMMA4_MATCHED_REASON,
    matched_reason_codes=(
        ModelPolicyReasonCode.REVIEWED_RUNTIME_PATH,
        ModelPolicyReasonCode.PILOT_NOT_YET_PROVEN,
    ),
    evidence_ids=GEMMA4_POLICY_EVIDENCE_IDS,
    required_provenance_fields=GEMMA4_REQUIRED_PROVENANCE_FIELDS,
    path_rejection_reason=GEMMA4_PATH_REASON,
    blocked_inspection_reason=GEMMA4_BLOCKED_INSPECTION_REASON,
    inspection_blocking_reason_codes=(
        ModelPolicyReasonCode.INVALID_FACTS,
        ModelPolicyReasonCode.DENSE_TOPOLOGY_REQUIRED,
    ),
)

_GEMMA4_UNIFIED_POLICY = _ModelCompatibilityPolicy(
    policy_id=GEMMA4_UNIFIED_POLICY_ID,
    policy_version=GEMMA4_UNIFIED_POLICY_VERSION,
    family=GEMMA4_FAMILY,
    claims={
        "architecture": (GEMMA4_UNIFIED_ARCHITECTURE,),
        "model_type": (GEMMA4_UNIFIED_MODEL_TYPE,),
    },
    constraints=(
        {
            "kind": "exact_identity",
            "values": {
                "architecture": GEMMA4_UNIFIED_ARCHITECTURE,
                "family": GEMMA4_FAMILY,
                "model_type": GEMMA4_UNIFIED_MODEL_TYPE,
            },
            "reason": "gemma4_unified_identity",
            "reason_code": ModelPolicyReasonCode.IDENTITY_MISMATCH.value,
        },
        {
            "kind": "field_equals",
            "field": "moe",
            "value": None,
            "reason": "gemma4_dense",
            "reason_code": ModelPolicyReasonCode.DENSE_TOPOLOGY_REQUIRED.value,
        },
        {
            "kind": "compiler_contract",
            "reason": "gemma4_unified_compiler",
            "reason_code": ModelPolicyReasonCode.COMPILER_CONTRACT_UNSUPPORTED.value,
        },
    ),
    paths=(
        _policy_path(
            path_id=GEMMA4_UNIFIED_QLORA_PATH_ID,
            family=GEMMA4_FAMILY,
            method=Method.QLORA,
            training_runtime=TrainingRuntime.MLX_LM,
            compute_backend=Backend.MPS,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.DENSE_CAUSAL_LM_V1,
            evidence_ids=GEMMA4_UNIFIED_POLICY_EVIDENCE_IDS,
        ),
        _policy_path(
            path_id=GEMMA4_UNIFIED_LORA_PATH_ID,
            family=GEMMA4_FAMILY,
            method=Method.LORA,
            training_runtime=TrainingRuntime.MLX_LM,
            compute_backend=Backend.MPS,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.DENSE_CAUSAL_LM_V1,
            evidence_ids=GEMMA4_UNIFIED_POLICY_EVIDENCE_IDS,
        ),
    ),
    matched_reason_key="gemma4_unified_matched",
    matched_reason=GEMMA4_UNIFIED_MATCHED_REASON,
    matched_reason_codes=(
        ModelPolicyReasonCode.REVIEWED_RUNTIME_PATH,
        ModelPolicyReasonCode.PILOT_NOT_YET_PROVEN,
    ),
    evidence_ids=GEMMA4_UNIFIED_POLICY_EVIDENCE_IDS,
    required_provenance_fields=GEMMA4_UNIFIED_REQUIRED_PROVENANCE_FIELDS,
    path_rejection_reason=GEMMA4_UNIFIED_COMPILER_REASON,
    blocked_inspection_reason=GEMMA4_UNIFIED_BLOCKED_INSPECTION_REASON,
    inspection_blocking_reason_codes=(
        ModelPolicyReasonCode.COMPILER_CONTRACT_UNSUPPORTED,
    ),
)

MODEL_COMPATIBILITY_POLICIES: tuple[_ModelCompatibilityPolicy, ...] = (
    _QWEN3_MOE_POLICY,
    _QWEN2_POLICY,
    _GEMMA4_POLICY,
    _GEMMA4_UNIFIED_POLICY,
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


def current_model_policy_snapshot() -> dict[str, Any]:
    """Return the canonical portable snapshot generated from the host registry."""

    policies = [
        {
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "family": policy.family,
            "claims": {
                "any_identity": {
                    field: list(values) for field, values in policy.claims.items()
                }
            },
            "constraints": [
                to_primitive(constraint) for constraint in policy.constraints
            ],
            "paths": [to_primitive(path) for path in policy.paths],
            "matched_reason": policy.matched_reason_key,
            "matched_reason_codes": [
                item.value for item in policy.matched_reason_codes
            ],
            "evidence_ids": list(policy.evidence_ids),
            "required_provenance_fields": list(policy.required_provenance_fields),
        }
        for policy in MODEL_COMPATIBILITY_POLICIES
    ]
    return model_policy_snapshot_payload(
        {
            "compatibility_schema_version": MODEL_COMPATIBILITY_SCHEMA_VERSION,
            "dense_families": sorted(
                family
                for family, modules in TARGET_MODULES.items()
                if modules == DENSE_CAUSAL_LM_TARGET_MODULES
            ),
            "sparse_identity_markers": ["mixtral", "moe"],
            "reasons": {
                "identity": QWEN3_MOE_IDENTITY_REASON,
                "layout": QWEN3_MOE_LAYOUT_REASON,
                "topology": QWEN3_MOE_TOPOLOGY_REASON,
                "shared": QWEN3_MOE_SHARED_EXPERT_REASON,
                "four_bit": QWEN3_MOE_FOUR_BIT_REASON,
                "invalid": INVALID_COMPATIBILITY_FACTS_REASON,
                "matched": QWEN3_MOE_MATCHED_REASON,
                "qwen2_identity": QWEN2_IDENTITY_REASON,
                "qwen2_layers": QWEN2_LAYER_REASON,
                "qwen2_layout": QWEN2_LAYOUT_REASON,
                "qwen2_dense": QWEN2_DENSE_REASON,
                "qwen2_four_bit": QWEN2_FOUR_BIT_REASON,
                "qwen2_matched": QWEN2_MATCHED_REASON,
                "gemma4_identity": GEMMA4_IDENTITY_REASON,
                "gemma4_dense": GEMMA4_DENSE_REASON,
                "gemma4_matched": GEMMA4_MATCHED_REASON,
                "gemma4_unified_identity": GEMMA4_UNIFIED_IDENTITY_REASON,
                "gemma4_unified_compiler": GEMMA4_UNIFIED_COMPILER_REASON,
                "gemma4_unified_matched": GEMMA4_UNIFIED_MATCHED_REASON,
                "dense": FAMILY_RECOGNIZED_REASON,
                "sparse": UNREVIEWED_SPARSE_MODEL_REASON,
                "unknown": UNKNOWN_POLICY_REASON,
            },
            "policies": policies,
        }
    )


def current_model_policy_snapshot_bytes() -> bytes:
    return model_policy_snapshot_bytes(current_model_policy_snapshot())


def current_model_policy_snapshot_sha256() -> str:
    return model_policy_snapshot_sha256(current_model_policy_snapshot())


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
    *,
    confirm_unreviewed_runtime: bool = False,
) -> ModelPolicyDecision:
    """Return model-policy status without deciding hardware feasibility."""

    snapshot = current_model_policy_snapshot()
    primitive = evaluate_model_policy_snapshot(
        snapshot,
        to_primitive(subject),
    )
    primitive = apply_operator_unreviewed_runtime_confirm(
        snapshot,
        to_primitive(subject),
        primitive,
        confirmed=confirm_unreviewed_runtime,
    )
    return model_policy_decision_from_primitive(primitive)


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
    policy = _policy_for_decision(decision)
    if decision.kind == ModelPolicyDecisionKind.BLOCKED:
        if policy is not None:
            return tuple(dict.fromkeys((decision.reason, policy.path_rejection_reason)))
        return (decision.reason,)
    if policy is None:  # pragma: no cover - guarded by domain decision invariants
        return (decision.reason,)
    return (policy.path_rejection_reason,)


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
    policy = _policy_for_decision(decision)
    inspected_configuration_blocked = bool(
        decision.kind == ModelPolicyDecisionKind.BLOCKED
        and policy is not None
        and any(
            code in policy.inspection_blocking_reason_codes
            for code in decision.reason_codes
        )
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
            policy.blocked_inspection_reason
            if inspected_configuration_blocked and policy is not None
            else UNKNOWN_POLICY_REASON
        ),
    }
