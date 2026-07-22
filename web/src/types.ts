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
}

export interface ModelInspectionFactProvenance {
  kind: "provider-declared" | string;
  source: string;
  observed_at?: string;
  resolved_revision?: string;
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
    parameters?: null;
    training_allowed?: null;
    [key: string]: unknown;
  };
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
  [key: string]: unknown;
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
    task: string;
    evaluation_fraction: number;
    packing: boolean;
    checkpoint_steps: number;
  };
  dataset_path: string;
  sample_limit?: number;
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
    scope?: string;
    [key: string]: unknown;
  } | null;
  pilot_metrics?: {
    checkpoint_continuation_observed?: boolean;
    measured_checkpoint_bytes?: number;
    measured_final_export_bytes?: number;
    pilot_run_id?: string;
    pilot_run_dir?: string;
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
}

export interface CompileResponse {
  bundle_dir: string;
  archive_path?: string | null;
  files: Array<string | BundleFile>;
  report?: ValidationReport | null;
  [key: string]: unknown;
}

export interface ValidateRequest {
  bundle_dir: string;
  level: "contract" | "static" | "dependency" | "model-data" | "measured-preflight" | "pilot";
  run: boolean;
}

export interface JobRequest {
  bundle_dir: string;
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
  service?: { name?: string; version?: string; status?: string };
  version?: string;
  defaults?: Partial<FactDraft> & {
    sample_limit?: number;
    reserve_gib?: number;
    task?: string;
    packing?: boolean;
  };
  capabilities?: Record<string, unknown>;
  plan?: TrainingPlan | null;
  bundle?: CompileResponse | null;
  job?: Job | null;
  [key: string]: unknown;
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
    total_vram_bytes?: number;
    total_vram_gib?: number;
    free_vram_bytes?: number;
    free_vram_gib?: number;
    supports_bf16?: boolean;
    supports_8bit?: boolean;
    supports_4bit?: boolean;
  }>;
  gpu_count?: number;
  backend?: string;
  vram_gib?: number;
  host_ram_bytes?: number;
  host_ram_gib?: number;
  host_ram_free_bytes?: number;
  host_ram_free_gib?: number;
  reserve_per_device_bytes?: number;
  reserve_gib?: number;
  disk_free_bytes?: number;
  disk_free_gib?: number;
  provenance?: Record<string, unknown>;
  [key: string]: unknown;
}

export type WorkflowStage = "facts" | "compare" | "compile" | "validate" | "run";
