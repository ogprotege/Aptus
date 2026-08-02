from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from .domain import (
    AdapterProfile,
    Backend,
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
    required_schema: Literal["aptus.training-plan.v5"]
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
    provenance_summary: list[InspectedModelProvenanceResponse]
    provenance_requirement: Literal["provider-declared"] | None
    provenance_requirement_met: Annotated[bool, Field(strict=True)]
    evaluated_at: str


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


class PlanCandidateResponse(ResponseModel):
    candidate_id: str = Field(pattern=r"^cand_[0-9a-f]{20}$")
    model_policy_decision_id: str = Field(pattern=r"^compat_[0-9a-f]{20}$")
    policy_binding: PlanModelPolicyBindingResponse | None


class TrainingPlanResponse(ResponseModel):
    schema_version: Literal["aptus.training-plan.v5"]
    plan_id: str = Field(pattern=r"^plan_[0-9a-f]{20}$")
    model_policy_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recommended: PlanCandidateResponse
    candidates: list[PlanCandidateResponse] = Field(min_length=1)
    warnings: list[str]
    recommendation_rationale: list[str]
    model_policy_decision: InspectedModelPolicyDecisionResponse
    model_policy_decision_source: Literal["provider-inspection", "user-attested"]
    inspection_receipt: ModelInspectionReceiptResponse | None
    project_id: str | None = None
    project_revision_id: str | None = None

    @model_validator(mode="after")
    def require_complete_policy_chain(self) -> Self:
        decision_id = self.model_policy_decision.decision_id
        if any(
            candidate.model_policy_decision_id != decision_id
            for candidate in self.candidates
        ):
            raise ValueError("Every candidate must link to the plan policy decision.")
        if self.recommended.candidate_id not in {
            candidate.candidate_id for candidate in self.candidates
        }:
            raise ValueError("The recommended candidate must be listed in candidates.")
        if self.model_policy_decision_source == "provider-inspection":
            if self.inspection_receipt is None:
                raise ValueError("Provider-inspection plans require a receipt.")
            if self.inspection_receipt.decision.decision_id != decision_id:
                raise ValueError("The inspection receipt must bind the plan decision.")
        elif self.inspection_receipt is not None:
            raise ValueError("User-attested plans cannot carry a receipt.")
        for candidate in self.candidates:
            binding = candidate.policy_binding
            if binding is None:
                continue
            if binding.decision_id != decision_id:
                raise ValueError("Candidate bindings must link to the plan decision.")
            if binding.source != self.model_policy_decision_source:
                raise ValueError("Candidate binding sources must match the plan.")
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


class ProjectRecoveryResponse(ResponseModel):
    status: Literal["recovered"]
    project_id: str
    revision: ProjectRevisionResponse
    training_authorization_current: Literal[False]
