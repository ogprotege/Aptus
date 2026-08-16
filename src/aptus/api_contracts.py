from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from .domain import (
    AdapterProfile,
    Backend,
    CandidateStatus,
    Distribution,
    EvidenceRequirement,
    Method,
    ModelPolicyDecisionKind,
    ModelPolicyReasonCode,
    TrainingRuntime,
)
from .model_compatibility import validate_registered_compatibility_path


API_CONTRACT_VERSION = "aptus.api.v1"

_NonEmptyCompatibilityText = Annotated[
    str,
    Field(min_length=1, pattern=r"^\S(?:[\s\S]*\S)?$"),
]


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ClosedResponseModel(ResponseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ResponseModel):
    status: Literal["ok"]
    version: str
    api_contract_version: Literal["aptus.api.v1"]


class ErrorResponse(ResponseModel):
    error: str
    details: Any | None = None
    message: str | None = None


class ServiceIdentity(ResponseModel):
    name: str
    version: str
    scope: str


class CapabilitiesResponse(ResponseModel):
    backends: list[str]
    known_backends: list[str]
    training_runtimes: list[str]
    known_training_runtimes: list[str]
    inference_services: list[str]
    methods: list[str]
    method_catalog: list[dict[str, Any]]
    objectives: list[str]
    supported_execution_backend: str
    supported_execution_backends: list[str]
    local_execution_enabled: bool
    model_families: list[str]
    validation_levels: list[str]


class ProjectRevisionSummary(ResponseModel):
    revision_id: str
    ordinal: int = Field(gt=0)
    created_at: str
    reason: str
    plan_id: str | None = None
    selected_candidate_id: str | None = None
    bundle_dir: str | None = None
    validation_state: str | None = None
    job_count: int = Field(ge=0)


class ProjectRevisionResponse(ResponseModel):
    schema_version: Literal["aptus.project-revision.v1"]
    revision_id: str
    project_id: str
    parent_revision_id: str | None = None
    ordinal: int = Field(gt=0)
    created_at: str
    reason: str
    facts: dict[str, Any] | None = None
    plan_id: str | None = None
    plan_snapshot: dict[str, Any] | None = None
    selected_candidate_id: str | None = None
    bundle: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    job_ids: list[str]
    training_authorization: dict[str, Any]
    content_sha256: str


class ProjectSummaryResponse(ResponseModel):
    schema_version: Literal["aptus.project.v1"]
    project_id: str
    name: str
    created_at: str
    updated_at: str
    latest_revision_id: str | None = None
    revision_count: int = Field(ge=0)
    latest: ProjectRevisionSummary | None = None


class ProjectResponse(ResponseModel):
    schema_version: Literal["aptus.project.v1"]
    project_id: str
    name: str
    created_at: str
    updated_at: str
    latest_revision_id: str | None = None
    revision_ids: list[str]
    revision_count: int = Field(ge=0)
    latest_revision: ProjectRevisionResponse | None = None


class ReplanRequiredResponse(ClosedResponseModel):
    status: Literal["replan_required"]
    plan_id: str | None = None
    found_schema: str | None = None
    required_schema: Literal["aptus.training-plan.v6"]
    source: Literal["project-revision", "compiled-bundle"]
    project_id: str | None = None
    project_revision_id: str | None = None
    message: str


class BootstrapResponse(ResponseModel):
    api_contract_version: Literal["aptus.api.v1"]
    version: str
    service: ServiceIdentity
    capabilities: CapabilitiesResponse
    defaults: dict[str, Any]
    stack_versions: dict[str, Any]
    evidence: list[dict[str, Any]]
    calibrated: bool
    projects: list[ProjectSummaryResponse]
    project: ProjectResponse | None = None
    project_history: list[ProjectRevisionSummary]
    plan: TrainingPlanResponse | None = None
    bundle: dict[str, Any] | None = None
    job: dict[str, Any] | None = None
    replan_required: ReplanRequiredResponse | None = None


class HardwareProbeResponse(ResponseModel):
    status: Literal["ok", "unavailable"]
    scope: str
    hardware: dict[str, Any] | None = None
    error: str | None = None
    manual_facts_supported: bool | None = None


class RuntimeInventoryResponse(ResponseModel):
    schema_version: Literal["aptus.runtime-inventory.v1"]
    interpreters: list[dict[str, Any]]
    available: dict[str, list[str]]
    compatible: dict[str, list[str]]
    configuration: dict[str, str]
    selected: dict[str, str]


class RuntimeConfiguredResponse(ResponseModel):
    status: Literal["ok"]
    runtime_id: str
    interpreter_path: str
    interpreter: dict[str, Any]
    persisted: bool


class PlatformResponse(ResponseModel):
    status: Literal["ok", "unsupported"]
    platform: dict[str, Any] | None
    error: str | None = None


class InferenceServicesResponse(ResponseModel):
    status: str
    scope: str
    training_capability: bool
    services: list[dict[str, Any]] | dict[str, Any]


class ProfileProvenanceResponse(ResponseModel):
    kind: str
    source: str
    observed_at: str | None = None
    digest: str | None = None
    detail: str | None = None


class ProfileResponse(ClosedResponseModel):
    source_path: str
    source_sha256: str
    source_format: str
    schema_name: str
    example_count: int = Field(gt=0)
    total_estimated_tokens: int = Field(gt=0)
    sequence_p50: int = Field(gt=0)
    sequence_p95: int = Field(gt=0)
    sequence_max: int = Field(gt=0)
    measurement: Literal["estimated", "tokenizer-measured"]
    warnings: list[str]
    schema_counts: dict[str, int]
    sampled_examples: int = Field(ge=0)
    sample_indices: list[int]
    duplicate_count: int = Field(ge=0)
    empty_count: int = Field(ge=0)
    truncation_count: int = Field(ge=0)
    truncation_rate: float = Field(ge=0, le=1)
    source_size_bytes: int = Field(gt=0)
    canonical_size_bytes: int = Field(gt=0)
    max_canonical_row_bytes: int = Field(gt=0)
    bundle_path: str | None = None
    provenance: ProfileProvenanceResponse | None = None


class InspectedMoETopologyResponse(ClosedResponseModel):
    expert_count: int | None = None
    experts_per_token: int | None = None
    expert_intermediate_size: int | None = None
    decoder_sparse_step: int | None = None
    mlp_only_layers: list[int] | None = None
    shared_expert_intermediate_size: int | None = None


class InspectedQuantizationOverrideResponse(ClosedResponseModel):
    module_path: str
    bits: int = Field(ge=1, le=16)
    group_size: int = Field(gt=0)


class InspectedQuantizationLayoutResponse(ClosedResponseModel):
    default_bits: int = Field(ge=1, le=16)
    default_group_size: int = Field(gt=0)
    module_overrides: list[InspectedQuantizationOverrideResponse]


class ModelInspectionFactsResponse(ClosedResponseModel):
    architecture: str | None = None
    architectures: list[str] | None = None
    model_type: str | None = None
    family: str | None = None
    hidden_size: int | None = Field(default=None, gt=0)
    intermediate_size: int | None = Field(default=None, gt=0)
    layers: int | None = Field(default=None, gt=0)
    context_length: int | None = Field(default=None, gt=0)
    attention_heads: int | None = Field(default=None, gt=0)
    key_value_heads: int | None = Field(default=None, gt=0)
    vocab_size: int | None = Field(default=None, gt=0)
    quantization_bits: int | None = Field(default=None, ge=1, le=16)
    quantization_layout: InspectedQuantizationLayoutResponse | None = None
    moe: InspectedMoETopologyResponse | None = None
    license_name: str | None = None
    parameters: None = None
    training_allowed: None = None


class ConditionalModelCompatibilityResponse(ClosedResponseModel):
    status: Literal["conditional"]
    family: _NonEmptyCompatibilityText
    supported_runtime: TrainingRuntime
    supported_methods: list[Method] = Field(min_length=1)
    compute_backend: Backend
    distribution: Distribution
    evidence_requirement: Literal["pilot-required"]
    adapter_profile_id: AdapterProfile
    reason: _NonEmptyCompatibilityText

    @model_validator(mode="after")
    def require_registered_execution_bindings(self) -> Self:
        if len(set(self.supported_methods)) != len(self.supported_methods):
            raise ValueError("Conditional compatibility methods must be unique.")
        for method in self.supported_methods:
            validate_registered_compatibility_path(
                family=self.family,
                method=method,
                training_runtime=self.supported_runtime,
                compute_backend=self.compute_backend,
                distribution=self.distribution,
                adapter_profile_id=self.adapter_profile_id,
                evidence_requirement=self.evidence_requirement,
            )
        return self


class RecognizedModelCompatibilityResponse(ClosedResponseModel):
    status: Literal["recognized"]
    family: _NonEmptyCompatibilityText
    supported_runtime: None
    supported_methods: list[Method] = Field(max_length=0)
    compute_backend: None
    distribution: None
    evidence_requirement: Literal["pilot-required"]
    adapter_profile_id: None
    reason: _NonEmptyCompatibilityText


class UnsupportedModelCompatibilityResponse(ClosedResponseModel):
    status: Literal["unsupported"]
    family: _NonEmptyCompatibilityText | None
    supported_runtime: None
    supported_methods: list[Method] = Field(max_length=0)
    compute_backend: None
    distribution: None
    evidence_requirement: Literal["implementation-required"]
    adapter_profile_id: None
    reason: _NonEmptyCompatibilityText


ModelCompatibilityVariant = Annotated[
    ConditionalModelCompatibilityResponse
    | RecognizedModelCompatibilityResponse
    | UnsupportedModelCompatibilityResponse,
    Field(discriminator="status"),
]


class ModelCompatibilityResponse(RootModel[ModelCompatibilityVariant]):
    pass


class InspectedRuntimeContractResponse(ClosedResponseModel):
    schema_version: Literal["aptus.runtime-contract.v1"]
    compute_backend: Backend
    training_runtime: TrainingRuntime
    compiler_id: str | None
    estimator_id: str
    evidence_requirement: EvidenceRequirement
    export_kind: str | None


class InspectedModelPolicyPathResponse(ClosedResponseModel):
    path_id: str
    method: Method
    distribution: Distribution
    adapter_profile_id: AdapterProfile | None
    target_modules: list[str]
    runtime_contract: InspectedRuntimeContractResponse
    required_validation_levels: list[
        Literal["model-data", "measured-preflight", "pilot"]
    ]
    evidence_ids: list[str]


class InspectedModelPolicyDecisionResponse(ClosedResponseModel):
    schema_version: Literal["aptus.model-compatibility.v2"]
    decision_id: str = Field(pattern=r"^compat_[0-9a-f]{20}$")
    subject_facts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: ModelPolicyDecisionKind
    family: str | None
    policy_id: str | None
    policy_version: str | None
    paths: list[InspectedModelPolicyPathResponse]
    reason_codes: list[ModelPolicyReasonCode]
    evidence_ids: list[str]
    reason: _NonEmptyCompatibilityText


class InspectedModelProvenanceResponse(ClosedResponseModel):
    field: str
    kind: Literal["provider-declared", "inferred"]
    source: str
    observed_at: str
    resolved_revision: str = Field(pattern=r"^[0-9A-Fa-f]{40,64}$")


class ModelInspectionReceiptResponse(ClosedResponseModel):
    schema_version: Literal["aptus.model-inspection-receipt.v1"]
    receipt_id: str = Field(pattern=r"^receipt_[0-9a-f]{20}$")
    model_id: str
    resolved_revision: str = Field(pattern=r"^[0-9A-Fa-f]{40,64}$")
    observed_facts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: InspectedModelPolicyDecisionResponse
    provenance_summary: list[InspectedModelProvenanceResponse] = Field(min_length=1)
    provenance_requirement: Literal["provider-declared"] | None
    provenance_requirement_met: Annotated[bool, Field(strict=True)]
    evaluated_at: str

    @model_validator(mode="after")
    def require_coherent_provenance_summary(self) -> Self:
        fields = [item.field for item in self.provenance_summary]
        if fields != sorted(set(fields)):
            raise ValueError(
                "Inspection receipt provenance fields must be sorted and unique."
            )
        if any(
            item.resolved_revision.lower() != self.resolved_revision.lower()
            for item in self.provenance_summary
        ):
            raise ValueError(
                "Inspection receipt provenance revisions must match the receipt."
            )
        has_provider_declared = any(
            item.kind == "provider-declared" for item in self.provenance_summary
        )
        if self.provenance_requirement_met and (
            self.provenance_requirement != "provider-declared"
            or not has_provider_declared
        ):
            raise ValueError(
                "A met inspection provenance requirement requires "
                "provider-declared evidence."
            )
        if not has_provider_declared:
            raise ValueError("Inspection receipts require provider-declared evidence.")
        return self

    @model_validator(mode="after")
    def require_matched_path_provenance(self) -> Self:
        if self.decision.kind == ModelPolicyDecisionKind.PATH_MATCHED and (
            self.provenance_requirement != "provider-declared"
            or not self.provenance_requirement_met
        ):
            raise ValueError(
                "Path-matched provider decisions require provider-declared provenance."
            )
        return self


class ModelInspectionResponse(ClosedResponseModel):
    status: Literal["ok", "unavailable", "unsupported"]
    model_id: str
    requested_revision: str
    resolved_revision: str | None = None
    facts: ModelInspectionFactsResponse | None = None
    compatibility: ModelCompatibilityResponse | None = None
    provenance: dict[str, dict[str, Any]] | None = None
    inspection_receipt: ModelInspectionReceiptResponse | None = None
    warnings: list[str] | None = None
    explicit_user_facts_required: list[str] | None = None
    error: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def require_complete_success_receipt(self) -> Self:
        if self.status == "ok":
            missing = [
                field
                for field in (
                    "resolved_revision",
                    "facts",
                    "compatibility",
                    "provenance",
                    "inspection_receipt",
                )
                if getattr(self, field) is None
            ]
            if missing:
                raise ValueError(
                    "Successful model inspection requires " + ", ".join(missing) + "."
                )
            assert self.inspection_receipt is not None
            assert self.resolved_revision is not None
            if self.inspection_receipt.model_id != self.model_id:
                raise ValueError("Inspection receipt model_id must match the response.")
            if (
                self.inspection_receipt.resolved_revision.lower()
                != self.resolved_revision.lower()
            ):
                raise ValueError(
                    "Inspection receipt revision must match the resolved response revision."
                )
        elif self.inspection_receipt is not None:
            raise ValueError("Unsuccessful model inspection cannot claim a receipt.")
        return self


class InferenceModelsResponse(ClosedResponseModel):
    status: Literal["ok"]
    service: Literal["lm-studio", "omlx"]
    endpoint: str
    models: list[dict[str, Any]]


class InferenceGenerateResponse(ClosedResponseModel):
    status: Literal["ok"]
    service: Literal["lm-studio", "omlx"]
    endpoint: str
    model: str
    content: str
    usage: dict[str, Any] | None = None
    response_id: Any | None = None
    payload: dict[str, Any]


class PlanModelPolicyBindingResponse(ClosedResponseModel):
    schema_version: Literal["aptus.model-policy-binding.v1"]
    decision_id: str = Field(pattern=r"^compat_[0-9a-f]{20}$")
    subject_facts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_id: str
    policy_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    path_id: str
    source: Literal["provider-inspection", "user-attested"]
    inspection_receipt_id: Annotated[
        str | None, Field(pattern=r"^receipt_[0-9a-f]{20}$")
    ]
    reason_codes: list[ModelPolicyReasonCode]
    evidence_ids: list[str]

    @model_validator(mode="after")
    def require_source_receipt_coherence(self) -> Self:
        if self.source == "provider-inspection" and self.inspection_receipt_id is None:
            raise ValueError("Provider policy bindings require a receipt ID.")
        if self.source == "user-attested" and self.inspection_receipt_id is not None:
            raise ValueError("User-attested policy bindings cannot claim a receipt.")
        return self


class PlanModelSubjectResponse(ResponseModel):
    model_id: _NonEmptyCompatibilityText
    revision: str = Field(pattern=r"^[0-9A-Fa-f]{40,64}$")


class PlanCandidateResponse(ResponseModel):
    candidate_id: str = Field(pattern=r"^cand_[0-9a-f]{20}$")
    model_policy_decision_id: str = Field(pattern=r"^compat_[0-9a-f]{20}$")
    policy_binding: PlanModelPolicyBindingResponse | None
    method: Method
    distribution: Distribution
    status: CandidateStatus
    feasible: Annotated[bool, Field(strict=True)]
    rejection_reasons: list[str]
    target_modules: list[str]
    runtime_contract: InspectedRuntimeContractResponse

    @model_validator(mode="after")
    def require_status_feasibility_coherence(self) -> Self:
        expected_feasible = self.status in {
            CandidateStatus.FEASIBLE,
            CandidateStatus.CONDITIONAL,
        }
        if self.feasible != expected_feasible:
            raise ValueError("Candidate status and feasibility must agree.")
        return self


def _matching_candidate_policy_path(
    decision: InspectedModelPolicyDecisionResponse,
    candidate: PlanCandidateResponse,
) -> InspectedModelPolicyPathResponse | None:
    if decision.kind != ModelPolicyDecisionKind.PATH_MATCHED:
        return None
    return next(
        (
            path
            for path in decision.paths
            if path.method == candidate.method
            and path.distribution == candidate.distribution
            and path.target_modules == candidate.target_modules
            and path.runtime_contract == candidate.runtime_contract
        ),
        None,
    )


def _require_exact_candidate_policy_bindings(
    *,
    candidates: list[PlanCandidateResponse],
    decision: InspectedModelPolicyDecisionResponse,
    source: Literal["provider-inspection", "user-attested"],
    receipt_id: str | None,
    context: str,
) -> None:
    for candidate in candidates:
        matching_path = _matching_candidate_policy_path(decision, candidate)
        binding = candidate.policy_binding
        if matching_path is None:
            if binding is not None:
                raise ValueError(
                    f"Candidate bindings must be null when no {context} policy path "
                    "matches."
                )
            continue
        if binding is None:
            raise ValueError(
                f"Candidates matching a registered {context} policy path require "
                "a binding."
            )
        expected_binding = {
            "schema_version": "aptus.model-policy-binding.v1",
            "decision_id": decision.decision_id,
            "subject_facts_sha256": decision.subject_facts_sha256,
            "policy_id": decision.policy_id,
            "policy_version": decision.policy_version,
            "path_id": matching_path.path_id,
            "source": source,
            "inspection_receipt_id": receipt_id,
            "reason_codes": [item.value for item in decision.reason_codes],
            "evidence_ids": list(
                dict.fromkeys(decision.evidence_ids + matching_path.evidence_ids)
            ),
        }
        if binding.model_dump(mode="json") != expected_binding:
            raise ValueError(
                f"Candidate bindings must exactly match the registered {context} "
                "policy path."
            )


def _model_policy_decisions_share_semantics(
    left: InspectedModelPolicyDecisionResponse,
    right: InspectedModelPolicyDecisionResponse,
) -> bool:
    return left.model_dump(mode="json", exclude={"reason"}) == right.model_dump(
        mode="json", exclude={"reason"}
    )


def _require_receipt_model_subject(
    *,
    receipt: ModelInspectionReceiptResponse,
    model: PlanModelSubjectResponse,
    context: str,
) -> None:
    if receipt.model_id != model.model_id:
        raise ValueError(
            f"The {context} inspection receipt model ID must match the model subject."
        )
    if receipt.resolved_revision.lower() != model.revision.lower():
        raise ValueError(
            f"The {context} inspection receipt revision must match the model subject."
        )


class PlanCorrectionFactHintResponse(ClosedResponseModel):
    fact: str = Field(min_length=1)
    direction: Literal["decrease", "increase", "set", "review"]
    why: str = Field(min_length=1)
    source_reason_codes: list[str]


class PlanCorrectionDisallowedSuggestionResponse(ClosedResponseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class PlanCorrectionNextStepResponse(ClosedResponseModel):
    action: Literal[
        "compile-recommended",
        "confirm-pilot-then-train",
        "change-facts",
    ]
    label: str = Field(min_length=1)


class PlanCorrectionResponse(ClosedResponseModel):
    schema_version: Literal["aptus.plan-correction.v1"]
    kind: Literal["select-candidate", "no-path"]
    summary: str = Field(min_length=1, max_length=240)
    primary_reason_codes: list[str]
    recommended_candidate_id: str | None
    recommended_status: Literal["feasible", "conditional"] | None
    pilot_required: bool
    ranking_objective: Literal["quality", "memory", "speed"] | None
    fact_hints: list[PlanCorrectionFactHintResponse]
    disallowed_suggestions: list[PlanCorrectionDisallowedSuggestionResponse]
    operator_next_step: PlanCorrectionNextStepResponse


class TrainingKnobResponse(ClosedResponseModel):
    name: Literal[
        "rank",
        "alpha",
        "learning_rate",
        "completions_mask",
        "epochs",
        "dataset_size",
    ]
    value: str = Field(min_length=1)
    prior_kind: Literal[
        "method-class-prior",
        "objective-and-token-volume-prior",
        "compiler-contract",
    ]
    rationale: str = Field(min_length=1)


class TrainingPolicyResponse(ClosedResponseModel):
    schema_version: Literal["aptus.training-policy.v1"]
    policy_version: Literal["aptus-training-policy-v1"]
    knobs: list[TrainingKnobResponse]
    non_claims: list[str]


class RunCorrectionFactHintResponse(ClosedResponseModel):
    fact: str = Field(min_length=1)
    direction: Literal["decrease", "increase", "set", "review"]
    why: str = Field(min_length=1)


class RunCorrectionDisallowedSuggestionResponse(ClosedResponseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class RunCorrectionNextStepResponse(ClosedResponseModel):
    action: Literal["replan-with-fact-hints", "none"]
    label: str = Field(min_length=1)


class RunCorrectionResponse(ClosedResponseModel):
    schema_version: Literal["aptus.run-correction.v1"]
    kind: Literal["loss-collapsed", "loss-flat", "eval-rose", "none"]
    summary: str = Field(min_length=1)
    source: Literal["train_loss_observations+validation_loss_observations"]
    next_plan_hints: list[RunCorrectionFactHintResponse]
    disallowed_suggestions: list[RunCorrectionDisallowedSuggestionResponse]
    operator_next_step: RunCorrectionNextStepResponse
    non_claims: list[str]


class EvaluationNormalizationResponse(ClosedResponseModel):
    strip: bool
    collapse_whitespace: bool
    casefold: bool


class EvaluationDatasetBindingResponse(ClosedResponseModel):
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    format: Literal["jsonl"]
    gold_field: Literal["completion", "output", "gold"]
    row_count: Annotated[int, Field(strict=True, gt=0)]
    id_field: str | None
    path: str | None = None


class EvaluationMetricResponse(ClosedResponseModel):
    name: Literal["exact_match"]
    direction: Literal["higher_is_better"]
    implementation_version: Literal["aptus.exact-match.v1"]
    normalization: EvaluationNormalizationResponse


class EvaluationThresholdResponse(ClosedResponseModel):
    minimum: float = Field(ge=0, le=1)
    comparison: Literal["gte"]


class EvaluationArtifactBindingResponse(ClosedResponseModel):
    plan_id: str | None
    candidate_id: str | None
    job_id: str | None
    export_digest: str | None
    export_kind: Literal["adapter", "final-export"] | None


class EvaluationContractResponse(ClosedResponseModel):
    schema_version: Literal["aptus.evaluation-contract.v1"]
    claim: str = Field(min_length=1)
    dataset: EvaluationDatasetBindingResponse
    metric: EvaluationMetricResponse
    threshold: EvaluationThresholdResponse
    artifact_binding: EvaluationArtifactBindingResponse
    non_claims: list[str]


class EvaluationResultResponse(ClosedResponseModel):
    schema_version: Literal["aptus.evaluation-result.v1"]
    contract_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    gold_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    predictions_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    artifact_binding: EvaluationArtifactBindingResponse
    metric: Literal["exact_match"]
    score: float | None
    threshold: float = Field(ge=0, le=1)
    n_gold: Annotated[int, Field(strict=True, ge=0)]
    n_predictions: Annotated[int, Field(strict=True, ge=0)]
    n_scored: Annotated[int, Field(strict=True, ge=0)]
    n_missing: Annotated[int, Field(strict=True, ge=0)]
    n_extra: Annotated[int, Field(strict=True, ge=0)]
    decision: Literal["pass", "fail", "abstain"]
    decision_reasons: list[str]
    non_claims: list[str]
    evaluated_at: str = Field(min_length=1)


class TrainingPlanResponse(ResponseModel):
    schema_version: Literal["aptus.training-plan.v6"]
    plan_id: str = Field(pattern=r"^plan_[0-9a-f]{20}$")
    model_policy_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: PlanModelSubjectResponse
    recommended: PlanCandidateResponse
    candidates: list[PlanCandidateResponse] = Field(min_length=1)
    warnings: list[str]
    recommendation_rationale: list[str]
    model_policy_decision: InspectedModelPolicyDecisionResponse
    model_policy_decision_source: Literal["provider-inspection", "user-attested"]
    inspection_receipt: ModelInspectionReceiptResponse | None
    project_id: str | None = None
    project_revision_id: str | None = None
    correction: PlanCorrectionResponse | None = None
    training_policy: TrainingPolicyResponse | None = None

    @model_validator(mode="after")
    def require_complete_policy_chain(self) -> Self:
        decision_id = self.model_policy_decision.decision_id
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Plan candidate IDs must be unique.")
        if any(
            candidate.model_policy_decision_id != decision_id
            for candidate in [self.recommended, *self.candidates]
        ):
            raise ValueError("Every candidate must link to the plan policy decision.")
        if self.recommended.candidate_id not in {
            candidate.candidate_id for candidate in self.candidates
        }:
            raise ValueError("The recommended candidate must be listed in candidates.")
        listed_recommendation = next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.recommended.candidate_id
        )
        if self.recommended.model_dump(mode="json") != listed_recommendation.model_dump(
            mode="json"
        ):
            raise ValueError(
                "The recommended candidate must equal its listed candidate record."
            )
        if not self.recommended.feasible:
            raise ValueError("The recommended candidate must be viable.")
        if self.model_policy_decision_source == "provider-inspection":
            if self.inspection_receipt is None:
                raise ValueError("Provider-inspection plans require a receipt.")
            if not _model_policy_decisions_share_semantics(
                self.inspection_receipt.decision, self.model_policy_decision
            ):
                raise ValueError(
                    "The inspection receipt decision must semantically match the plan."
                )
            _require_receipt_model_subject(
                receipt=self.inspection_receipt,
                model=self.model,
                context="plan",
            )
        elif self.inspection_receipt is not None:
            raise ValueError("User-attested plans cannot carry a receipt.")
        expected_receipt_id = (
            self.inspection_receipt.receipt_id
            if self.inspection_receipt is not None
            else None
        )
        _require_exact_candidate_policy_bindings(
            candidates=[self.recommended, *self.candidates],
            decision=self.model_policy_decision,
            source=self.model_policy_decision_source,
            receipt_id=expected_receipt_id,
            context="plan",
        )
        return self


class NoFeasiblePlanResponse(ClosedResponseModel):
    error: Literal["no_feasible_plan"]
    message: str = Field(min_length=1)
    model: PlanModelSubjectResponse
    candidates: list[PlanCandidateResponse] = Field(min_length=1)
    model_policy_decision: InspectedModelPolicyDecisionResponse
    model_policy_decision_source: Literal["provider-inspection", "user-attested"]
    inspection_receipt: ModelInspectionReceiptResponse | None
    correction: PlanCorrectionResponse | None = None

    @model_validator(mode="after")
    def require_complete_policy_chain(self) -> Self:
        decision_id = self.model_policy_decision.decision_id
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("No-feasible-plan candidate IDs must be unique.")
        if any(
            candidate.feasible
            or candidate.status
            not in {CandidateStatus.INFEASIBLE, CandidateStatus.UNSUPPORTED}
            or not candidate.rejection_reasons
            for candidate in self.candidates
        ):
            raise ValueError(
                "No-feasible-plan candidates must be infeasible or unsupported "
                "with rejection reasons."
            )
        if any(
            candidate.model_policy_decision_id != decision_id
            for candidate in self.candidates
        ):
            raise ValueError("Every candidate must link to the policy decision.")
        if self.model_policy_decision_source == "provider-inspection":
            if self.inspection_receipt is None:
                raise ValueError("Provider-inspection failures require a receipt.")
            if not _model_policy_decisions_share_semantics(
                self.inspection_receipt.decision, self.model_policy_decision
            ):
                raise ValueError(
                    "The inspection receipt decision must semantically match the "
                    "failure."
                )
            _require_receipt_model_subject(
                receipt=self.inspection_receipt,
                model=self.model,
                context="failure",
            )
        elif self.inspection_receipt is not None:
            raise ValueError("User-attested failures cannot carry a receipt.")
        expected_receipt_id = (
            self.inspection_receipt.receipt_id
            if self.inspection_receipt is not None
            else None
        )
        _require_exact_candidate_policy_bindings(
            candidates=self.candidates,
            decision=self.model_policy_decision,
            source=self.model_policy_decision_source,
            receipt_id=expected_receipt_id,
            context="failure",
        )
        return self


class CompileResponse(ResponseModel):
    bundle_dir: str
    archive_path: str
    files: list[str]
    runtime_contract: dict[str, Any] | None
    report: dict[str, Any]
    project_id: str
    project_revision_id: str


class ValidationResponse(ResponseModel):
    state: str
    project_id: str
    project_revision_id: str
    authorization_status: Literal["current", "deferred", "blocked"] | None = None
    authorization_current: Annotated[bool, Field(strict=True)] | None = None
    authorization_error: str | None = None
    prelaunch_capacity_check: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_authorization_status_coherence(self) -> Self:
        if self.authorization_status is None:
            if (
                self.authorization_current is not None
                or self.authorization_error is not None
                or self.prelaunch_capacity_check is not None
            ):
                raise ValueError(
                    "Validation authorization fields require authorization_status."
                )
            return self
        if self.authorization_current is None:
            raise ValueError(
                "Validation authorization status requires authorization_current."
            )
        if self.authorization_status == "current":
            if self.state not in {
                "pilot-pass",
                "execution-approved",
                "measured-run-pass",
            }:
                raise ValueError(
                    "Current validation authorization requires an authorizable state."
                )
            if not self.authorization_current or self.authorization_error is not None:
                raise ValueError(
                    "Current validation authorization requires true with no error."
                )
            return self
        if self.authorization_current or not isinstance(self.authorization_error, str):
            raise ValueError(
                "Deferred or blocked validation authorization requires false and a "
                "diagnostic."
            )
        if (
            not self.authorization_error.strip()
            or self.authorization_error != self.authorization_error.strip()
        ):
            raise ValueError(
                "Deferred or blocked validation authorization requires false and a "
                "diagnostic."
            )
        return self


class JobResponse(ResponseModel):
    schema_version: Literal["aptus.job-record.v1"]
    id: str
    job_id: str
    state: str
    action: str
    bundle_dir: str
    created_at: str
    project_id: str | None = None
    project_revision_id: str | None = None
    run_correction: RunCorrectionResponse | None = None


class ProjectRecoveryResponse(ResponseModel):
    status: Literal["recovered"]
    project_id: str
    revision: ProjectRevisionResponse
    training_authorization_current: Literal[False]
