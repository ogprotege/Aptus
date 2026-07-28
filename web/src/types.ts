export type ProvenanceKind =
  | "measured"
  | "provider-declared"
  | "user-attested"
  | "declared"
  | "inferred"
  | "user-supplied"
  | "unknown"
  | "example";

export interface SourcedFact<T> {
  value: T | null;
  unit?: string;
  provenance: ProvenanceKind;
  source: string;
  observed_at?: string;
  confidence?: "high" | "medium" | "low" | string;
}

export interface MoETopology {
  expert_count: number;
  experts_per_token: number;
  expert_intermediate_size: number;
  decoder_sparse_step: number;
  mlp_only_layers: number[];
  shared_expert_intermediate_size?: number | null;
}

export interface InspectedMoETopology {
  expert_count?: number | null;
  experts_per_token?: number | null;
  expert_intermediate_size?: number | null;
  decoder_sparse_step?: number | null;
  mlp_only_layers?: number[] | null;
  shared_expert_intermediate_size?: number | null;
}

export interface QuantizationOverride {
  module_path: string;
  bits: number;
  group_size: number;
}

export interface QuantizationLayout {
  default_bits: number;
  default_group_size: number;
  module_overrides: QuantizationOverride[];
}

export interface ModelFacts {
  model_id: string;
  revision: string;
  family: string;
  parameters_b: number | null;
  hidden_size: number | null;
  layers: number | null;
  context_length: number | null;
  intermediate_size: number | null;
  license_name: string;
  training_allowed: boolean;
  model_type?: string | null;
  architecture?: string | null;
  quantization_bits?: number | null;
  quantization_layout?: QuantizationLayout | null;
  moe?: MoETopology | null;
  active_parameters_b?: number | null;
  sparse_layer_count?: number | null;
}

export interface ModelInspectionFactProvenance {
  kind: "provider-declared" | string;
  source: string;
  observed_at?: string;
  resolved_revision?: string;
}

export interface ModelCompatibility {
  status: "conditional" | "recognized" | "unsupported";
  family?: string | null;
  supported_runtime?: string | null;
  supported_methods?: string[];
  distribution?: string | null;
  evidence_requirement?: string | null;
  adapter_scope?: string | null;
  reason?: string | null;
  [key: string]: unknown;
}

export interface ModelInspectionResponse {
  status: "ok" | "unavailable" | "unsupported" | string;
  model_id: string;
  requested_revision: string;
  resolved_revision?: string;
  facts?: {
    architecture?: string | null;
    architectures?: string[] | null;
    model_type?: string | null;
    family?: string | null;
    hidden_size?: number | null;
    intermediate_size?: number | null;
    layers?: number | null;
    context_length?: number | null;
    license_name?: string | null;
    quantization_bits?: number | null;
    quantization_layout?: QuantizationLayout | null;
    moe?: InspectedMoETopology | null;
    parameters?: null;
    training_allowed?: null;
    [key: string]: unknown;
  };
  compatibility?: ModelCompatibility | null;
  provenance?: Record<string, ModelInspectionFactProvenance>;
  warnings?: string[];
  explicit_user_facts_required?: string[];
  error?: string;
  source?: string;
}

export interface DatasetFacts {
  source_path: string;
  format: string;
  schema_name: string;
  tokenizer_id: string;
  sample_limit: number | null;
}

export interface DeviceFacts {
  name: string;
  backend: string;
  total_vram_gib: number | null;
  free_vram_gib: number | null;
  supports_bf16: boolean;
  supports_8bit: boolean;
  supports_4bit: boolean;
}

export interface HardwareFacts {
  discovery: "local-scan" | "manual";
  gpu_count: number;
  devices: DeviceFacts[];
  host_ram_gib: number | null;
  host_ram_free_gib: number | null;
  reserve_per_device_gib: number | null;
  disk_free_gib: number | null;
}

export interface TargetFacts {
  task: string;
  objective: "quality" | "memory" | "speed";
  sequence_length: number | null;
  effective_batch_size: number | null;
  max_epochs: number | null;
  method_preference: string;
  runtime: string;
  evaluation_fraction: number;
  packing: boolean;
  checkpoint_steps: number;
}

export interface FactDraft {
  project_name: string;
  model: ModelFacts;
  dataset: DatasetFacts;
  hardware: HardwareFacts;
  target: TargetFacts;
}

export interface InputProfile {
  model?: Record<string, unknown>;
  dataset?: Record<string, unknown>;
  hardware?: Record<string, unknown>;
  target?: Record<string, unknown>;
  facts?: Array<SourcedFact<unknown> & { key?: string; label?: string }>;
  warnings?: string[];
  [key: string]: unknown;
}

export interface ProfileRequest {
  dataset_path: string;
  sample_limit?: number;
  sequence_length?: number;
}

export type ProfileResponse = InputProfile;

export interface MemoryComponent {
  key?: string;
  label: string;
  expected_bytes: number;
  upper_bytes?: number;
  evidence?: string;
}

export interface MemoryComponentValue {
  expected_bytes?: number;
  upper_bytes?: number;
  bytes?: number;
  label?: string;
}

export interface MemoryEstimate {
  expected_bytes?: number;
  upper_bytes?: number;
  estimated_peak_bytes?: number;
  point_estimate_bytes?: number;
  upper_estimate_bytes?: number;
  limit_bytes?: number;
  device_total_bytes?: number;
  components?: MemoryComponent[] | Record<string, number | MemoryComponentValue>;
  component_upper_bounds?: Record<string, number>;
  [key: string]: unknown;
}

export interface BatchStrategy {
  micro_batch_size?: number;
  gradient_accumulation_steps?: number;
  effective_batch_size?: number;
  [key: string]: unknown;
}

export interface CandidatePlan {
  id?: string;
  candidate_id?: string;
  method: string;
  distribution?: string;
  status?: "feasible" | "infeasible" | "unknown" | string;
  feasible?: boolean;
  precision?: string;
  quantization?: string | null;
  batches?: BatchStrategy;
  micro_batch_size?: number;
  gradient_accumulation_steps?: number;
  effective_batch_size?: number;
  memory?: MemoryEstimate;
  assumptions?: string[];
  evidence?: string[];
  rationale?: string[];
  rejection_reasons?: string[];
  confidence?: string;
  rank?: number;
  alpha?: number;
  learning_rate?: number;
  target_modules?: string[];
  preference_score?: number;
  user_reserve_bytes?: number;
  world_size?: number;
  device_indices?: number[];
  pareto_frontier?: boolean;
  ranking_basis?: string[];
  required_host_ram_bytes?: number;
  required_disk_bytes?: number;
  checkpoint_retention_bytes?: number;
  final_export_bytes?: number;
  runtime_contract?: {
    schema_version: string;
    compute_backend: string;
    training_runtime: string;
    compiler_id: string | null;
    estimator_id: string;
    evidence_requirement: string;
    export_kind: string | null;
  };
  [key: string]: unknown;
}

export interface MethodDescriptor {
  schema_version: string;
  method_id: string;
  display_name: string;
  summary: string;
  lifecycle: "gated-executable" | "experimental" | "research-only";
  selectable: boolean;
  parameter_scope: string;
  parameterization: string;
  base_storage: string;
  compiler_id: string | null;
  export_kind: string | null;
  supported_backends: string[];
  supported_runtimes?: string[];
  supported_distributions: string[];
  evidence_ids: string[];
  pilot_requirement: string;
  blocker?: string | null;
  aliases?: string[];
}

export interface EvidenceRecord {
  evidence_id: string;
  claim: string;
  source: string;
  source_kind: string;
  scope: string;
  confidence: string;
  revision?: string | null;
}

export interface TrainingPlan {
  schema_version?: string;
  plan_id?: string;
  project_id?: string;
  project_revision_id?: string;
  recommended: CandidatePlan | null;
  candidates: CandidatePlan[];
  warnings: string[];
  rationale: string[];
  recommendation_rationale?: string[];
  assumptions?: string[];
  evidence?: string[];
  evidence_records?: EvidenceRecord[];
  model?: Record<string, unknown>;
  dataset?: Record<string, unknown>;
  hardware?: Record<string, unknown>;
  target?: Record<string, unknown>;
  example?: boolean;
  [key: string]: unknown;
}

export interface PlanRequest {
  project_id?: string;
  project_name: string;
  model: {
    model_id: string;
    revision: string;
    family: string;
    parameters_b: number;
    hidden_size: number;
    layers: number;
    context_length: number;
    license_name: string;
    training_allowed: boolean;
    intermediate_size?: number;
    model_type?: string;
    architecture?: string;
    quantization_bits?: number;
    quantization_layout?: QuantizationLayout;
    moe?: MoETopology;
  };
  hardware: {
    discovery: "manual" | "local-scan";
    backend: string;
    gpu_count: number;
    vram_gib: number;
    free_vram_gib?: number;
    supports_bf16: boolean;
    supports_8bit: boolean;
    supports_4bit: boolean;
    host_ram_gib: number;
    host_ram_free_gib?: number;
    reserve_gib: number;
    disk_free_gib?: number;
  };
  target: {
    objective: string;
    sequence_length: number;
    effective_batch_size: number;
    max_epochs: number;
    method_preference?: string;
    training_runtime?: string;
    task: string;
    evaluation_fraction: number;
    packing: boolean;
    checkpoint_steps: number;
  };
  dataset_path: string;
  sample_limit?: number;
}

export interface ProjectRevisionSummary {
  revision_id: string;
  ordinal: number;
  created_at: string;
  reason: string;
  plan_id?: string | null;
  selected_candidate_id?: string | null;
  bundle_dir?: string | null;
  validation_state?: string | null;
  job_count: number;
}

export interface ProjectRevision {
  schema_version: string;
  revision_id: string;
  project_id: string;
  parent_revision_id?: string | null;
  ordinal: number;
  created_at: string;
  reason: string;
  plan_id?: string | null;
  selected_candidate_id?: string | null;
  facts?: Record<string, unknown> | null;
  plan_snapshot?: TrainingPlan | null;
  bundle?: CompileResponse | Record<string, unknown> | null;
  validation?: Record<string, unknown> | null;
  job_ids: string[];
  training_authorization: {
    current: false;
    reason: string;
  };
  content_sha256: string;
}

export interface ProjectSummary {
  schema_version: string;
  project_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  latest_revision_id?: string | null;
  revision_count: number;
  latest?: ProjectRevisionSummary | null;
}

export interface ProjectDetail extends ProjectSummary {
  latest_revision?: ProjectRevision | null;
}

export interface ProjectRecoveryResponse {
  status: "recovered";
  project_id: string;
  revision: ProjectRevision;
  training_authorization_current: false;
}

export interface ValidationFinding {
  code?: string;
  message: string;
  severity: "info" | "warning" | "error" | string;
  path?: string | null;
  gate?: string;
}

export interface ValidationGate {
  id?: string;
  label: string;
  state: "pending" | "running" | "passed" | "failed" | "blocked" | string;
  detail?: string;
}

export interface UnifiedMemoryAdmission {
  schema_version?: string;
  available_unified_memory_bytes?: number;
  point_estimate_bytes?: number;
  upper_estimate_bytes?: number;
  reserve_bytes?: number;
  required_available_bytes?: number;
  [key: string]: unknown;
}

export interface AdapterTargetBinding {
  schema_version?: string;
  planned_target_modules?: string[];
  resolved_layer_keys?: string[];
  transformer_layer_count?: number;
  expected_adapter_target_instance_count?: number;
  adapter_target_instance_count?: number;
  trainable_tensor_count?: number;
  target_instance_counts?: Record<string, number>;
  descriptor_sha256?: string;
  [key: string]: unknown;
}

export interface ArtifactManifestEntry {
  path?: string;
  size_bytes?: number;
  sha256?: string;
  [key: string]: unknown;
}

export interface MlxReloadEvidence {
  schema_version?: string;
  execution_semantics?: string;
  resume_supported?: boolean;
  fresh_process_observed?: boolean;
  generation_max_tokens?: number;
  generation_tokens?: number;
  measured_peak_bytes?: number;
  unified_memory_admission?: UnifiedMemoryAdmission;
  [key: string]: unknown;
}

export interface ValidationReport {
  state?: string;
  findings?: ValidationFinding[];
  gates?: ValidationGate[];
  checked_files?: string[];
  artifact_fingerprint?: string;
  runtime_evidence?: string[];
  smoke_command?: string[];
  validation_level?: string;
  validator_version?: string;
  validated_at?: string | null;
  bindings?: Record<string, string>;
  preflight_metrics?: {
    schema_version?: string;
    candidate_id?: string;
    method?: string;
    precision?: string;
    quantization?: string | null;
    distribution?: string;
    world_size?: number;
    measured_peak_cuda_bytes?: number;
    measured_peak_bytes?: number;
    memory_metric_backend?: string;
    training_runtime?: string;
    compute_backend?: string;
    execution_semantics?: string;
    resume_supported?: boolean;
    unified_memory_admission?: UnifiedMemoryAdmission;
    scope?: string;
    [key: string]: unknown;
  } | null;
  pilot_metrics?: {
    training_runtime?: string;
    compute_backend?: string;
    scope?: string;
    action?: string;
    execution_semantics?: string;
    resume_supported?: boolean;
    global_step?: number;
    completed_optimizer_updates?: number;
    finite_train_loss?: boolean;
    finite_validation_loss?: boolean;
    validation_examples?: number;
    measured_peak_bytes?: number;
    memory_metric_backend?: string;
    unified_memory_admission?: UnifiedMemoryAdmission;
    trainable_target_binding?: AdapterTargetBinding;
    adapter_manifest?: ArtifactManifestEntry[];
    artifact_manifest?: {
      schema_version?: string;
      total_bytes?: number;
      files?: ArtifactManifestEntry[];
      [key: string]: unknown;
    };
    reload_evidence?: MlxReloadEvidence;
    checkpoint_continuation_observed?: boolean;
    measured_checkpoint_bytes?: number;
    measured_final_export_bytes?: number;
    pilot_run_id?: string;
    pilot_run_dir?: string;
    run_id?: string;
    output_dir?: string;
    phase_one_checkpoint?: {
      total_bytes?: number;
      manifest_sha256?: string;
      files?: Array<Record<string, unknown>>;
      [key: string]: unknown;
    };
    phase_two_checkpoint?: {
      total_bytes?: number;
      manifest_sha256?: string;
      files?: Array<Record<string, unknown>>;
      [key: string]: unknown;
    };
    phase_one?: Record<string, unknown>;
    phase_two_resumed?: Record<string, unknown>;
    [key: string]: unknown;
  } | null;
  authorization_current?: boolean;
  authorization_error?: string | null;
  project_id?: string;
  project_revision_id?: string;
  prelaunch_capacity_check?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface BundleFile {
  path: string;
  kind?: string;
  size_bytes?: number;
  sha256?: string;
}

export interface CompileRequest {
  plan_id: string;
  output_dir: string;
  project_id: string;
  expected_project_revision_id: string;
}

export interface CompileResponse {
  bundle_dir: string;
  archive_path?: string | null;
  files: Array<string | BundleFile>;
  runtime_contract?: CandidatePlan["runtime_contract"];
  report?: ValidationReport | null;
  project_id?: string;
  project_revision_id?: string;
  [key: string]: unknown;
}

export interface ValidateRequest {
  bundle_dir: string;
  project_id: string;
  expected_project_revision_id: string;
  level: "contract" | "static" | "dependency" | "model-data" | "measured-preflight" | "pilot";
  run: boolean;
}

export interface JobRequest {
  bundle_dir: string;
  project_id: string;
  expected_project_revision_id: string;
  action: "dependency" | "model-data" | "preflight" | "pilot" | "train";
  confirm_full_train: boolean;
}

export interface Job {
  id: string;
  job_id?: string;
  state: "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled" | string;
  phase?: string;
  mode: string;
  log: string | string[];
  return_code: number | null;
  validation_report?: ValidationReport;
  validation_report_error?: string;
  project_id?: string;
  project_revision_id?: string;
  error?: string | null;
  log_path?: string;
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  completed_at?: string | null;
  bundle_dir?: string;
  cancellable?: boolean;
  owner_status?: string;
  cancellation_note?: string | null;
  run_id?: string | null;
  run_output_dir?: string | null;
  prelaunch_capacity_check?: Record<string, unknown> | null;
  completion_attestation?: {
    state?: string;
    measured_run_completed_at?: string;
    final_export?: Record<string, unknown>;
    measured_run?: Record<string, unknown>;
    [key: string]: unknown;
  } | null;
  artifact_integrity?: {
    status?: string;
    verified_at?: string | null;
    missing_paths?: string[];
    note?: string;
    [key: string]: unknown;
  } | null;
  [key: string]: unknown;
}

export interface BootstrapResponse {
  api_contract_version: "aptus.api.v1";
  service?: { name?: string; version?: string; status?: string };
  version?: string;
  defaults?: Partial<FactDraft> & {
    sample_limit?: number;
    backend?: string;
    training_runtime?: string;
    reserve_gib?: number;
    task?: string;
    packing?: boolean;
  };
  capabilities?: {
    methods?: string[];
    method_catalog?: MethodDescriptor[];
    [key: string]: unknown;
  };
  plan?: TrainingPlan | null;
  bundle?: CompileResponse | null;
  job?: Job | null;
  projects: ProjectSummary[];
  project?: ProjectDetail | null;
  project_history: ProjectRevisionSummary[];
  replan_required?: ReplanRequired | null;
  [key: string]: unknown;
}

export interface ReplanRequired {
  status: "replan_required";
  plan_id?: string | null;
  found_schema?: string | null;
  required_schema: "aptus.training-plan.v3";
  source: "project-revision" | "compiled-bundle";
  project_id?: string | null;
  project_revision_id?: string | null;
  message: string;
}

export interface HardwareProbeResponse {
  status?: string;
  scope?: string;
  error?: string;
  manual_facts_supported?: boolean;
  hardware?: HardwareProbeResponse;
  devices?: Array<{
    name?: string;
    backend?: string;
    total_vram_bytes?: number | null;
    total_vram_gib?: number | null;
    free_vram_bytes?: number | null;
    free_vram_gib?: number | null;
    supports_bf16?: boolean;
    supports_8bit?: boolean;
    supports_4bit?: boolean;
  }>;
  gpu_count?: number;
  backend?: string;
  vram_gib?: number;
  host_ram_bytes?: number | null;
  host_ram_gib?: number | null;
  host_ram_free_bytes?: number | null;
  host_ram_free_gib?: number | null;
  reserve_per_device_bytes?: number | null;
  reserve_gib?: number | null;
  disk_free_bytes?: number | null;
  disk_free_gib?: number | null;
  provenance?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ApplePlatformResponse {
  status: "ok" | "unsupported" | string;
  error?: string;
  platform?: {
    os_version?: string;
    os_build?: string | null;
    chip_name?: string | null;
    logical_cpu_count?: number | null;
    metal_gpu_core_count?: number | null;
    unified_memory_bytes?: number;
    available_memory_bytes?: number | null;
    memory_free_percent?: number | null;
    metal_recommended_working_set_bytes?: number | null;
    mlx?: Record<string, unknown>;
    mlx_lm?: Record<string, unknown>;
    pytorch_mps?: Record<string, unknown>;
    [key: string]: unknown;
  } | null;
}

export interface RuntimeInventory {
  schema_version: "aptus.runtime-inventory.v1";
  interpreters: Array<Record<string, unknown>>;
  available: Record<string, string[]>;
  compatible: Record<string, string[]>;
  configuration: Record<string, string>;
  selected: Record<string, string>;
}

export interface InferenceServiceRequest {
  service: "lm-studio" | "omlx";
  endpoint?: string;
  timeout_seconds?: number;
}

export interface InferenceGenerateRequest extends InferenceServiceRequest {
  model: string;
  messages: Array<{
    role: "system" | "user" | "assistant" | "tool";
    content: string;
  }>;
  max_tokens?: number;
  temperature?: number;
}

export type WorkflowStage = "facts" | "compare" | "compile" | "validate" | "run";
