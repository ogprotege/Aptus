from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


API_CONTRACT_VERSION = "aptus.api.v1"


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
    required_schema: Literal["aptus.training-plan.v3"]
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
    plan: dict[str, Any] | None = None
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


class ModelCompatibilityResponse(ClosedResponseModel):
    status: Literal["conditional", "recognized", "unsupported"]
    family: str | None = None
    supported_runtime: str | None = None
    supported_methods: list[str]
    distribution: str | None = None
    evidence_requirement: Literal["pilot-required", "implementation-required"]
    adapter_scope: str | None = None
    reason: str


class ModelInspectionResponse(ClosedResponseModel):
    status: Literal["ok", "unavailable", "unsupported"]
    model_id: str
    requested_revision: str
    resolved_revision: str | None = None
    facts: ModelInspectionFactsResponse | None = None
    compatibility: ModelCompatibilityResponse | None = None
    provenance: dict[str, dict[str, Any]] | None = None
    warnings: list[str] | None = None
    explicit_user_facts_required: list[str] | None = None
    error: str | None = None
    source: str | None = None


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


class TrainingPlanResponse(ResponseModel):
    schema_version: str
    plan_id: str
    recommended: dict[str, Any]
    candidates: list[dict[str, Any]]
    warnings: list[str]
    project_id: str | None = None
    project_revision_id: str | None = None


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
