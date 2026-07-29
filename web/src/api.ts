import type {
  BootstrapResponse,
  CandidatePlan,
  CompileRequest,
  CompileResponse,
  FactDraft,
  HardwareProbeResponse,
  ApplePlatformResponse,
  RuntimeInventory,
  InferenceGenerateRequest,
  InferenceServiceRequest,
  Job,
  JobRequest,
  MethodDescriptor,
  ModelInspectionResponse,
  PlanRequest,
  ProjectDetail,
  ProjectRecoveryResponse,
  ProjectRevision,
  ProjectRevisionSummary,
  ProjectSummary,
  ProfileRequest,
  ProfileResponse,
  TrainingPlan,
  ValidateRequest,
  ValidationReport,
} from "./types";
import type { components, paths } from "./generated/openapi";
import { normalizeModelCompatibility } from "./lib/modelCompatibility";

export type OpenApiBootstrapResponse = components["schemas"]["BootstrapResponse"];
export type OpenApiCompileResponse = components["schemas"]["CompileResponse"];
export type OpenApiHardwareProbeResponse = components["schemas"]["HardwareProbeResponse"];
export type OpenApiInferenceGenerateResponse = components["schemas"]["InferenceGenerateResponse"];
export type OpenApiInferenceModelsResponse = components["schemas"]["InferenceModelsResponse"];
export type OpenApiInferenceServicesResponse = components["schemas"]["InferenceServicesResponse"];
export type OpenApiJobResponse = components["schemas"]["JobResponse"];
export type OpenApiModelInspectionResponse = components["schemas"]["ModelInspectionResponse"];
export type OpenApiPlatformResponse = components["schemas"]["PlatformResponse"];
export type OpenApiProfileResponse = components["schemas"]["ProfileResponse"];
export type OpenApiProjectResponse = components["schemas"]["ProjectResponse"];
export type OpenApiProjectRecoveryResponse = components["schemas"]["ProjectRecoveryResponse"];
export type OpenApiProjectRevisionResponse = components["schemas"]["ProjectRevisionResponse"];
export type OpenApiProjectRevisionSummary = components["schemas"]["ProjectRevisionSummary"];
export type OpenApiProjectSummaryResponse = components["schemas"]["ProjectSummaryResponse"];
export type OpenApiRuntimeInventoryResponse = components["schemas"]["RuntimeInventoryResponse"];
export type OpenApiTrainingPlanResponse = components["schemas"]["TrainingPlanResponse"];
export type OpenApiValidationResponse = components["schemas"]["ValidationResponse"];

type OpenApiPath = keyof paths;

export const API_PATHS = {
  bootstrap: "/api/v1/bootstrap",
  compile: "/api/v1/compile",
  hardware: "/api/v1/hardware",
  inferenceGenerate: "/api/v1/inference/generate",
  inferenceModels: "/api/v1/inference/models",
  inferenceServices: "/api/v1/inference/services",
  job: "/api/v1/jobs/{job_id}",
  jobCancel: "/api/v1/jobs/{job_id}/cancel",
  jobs: "/api/v1/jobs",
  modelInspect: "/api/v1/models/inspect",
  plan: "/api/v1/plan",
  platform: "/api/v1/platform",
  profile: "/api/v1/profile",
  project: "/api/v1/projects/{project_id}",
  projectRecover: "/api/v1/projects/{project_id}/recover",
  projectRevision: "/api/v1/projects/{project_id}/revisions/{revision_id}",
  projectRevisions: "/api/v1/projects/{project_id}/revisions",
  projects: "/api/v1/projects",
  runtimes: "/api/v1/runtimes",
  validate: "/api/v1/validate",
} as const satisfies Record<string, OpenApiPath>;

function bindApiPath(
  template: OpenApiPath,
  parameters: Record<string, string>,
): string {
  let result: string = template;
  for (const [name, value] of Object.entries(parameters)) {
    result = result.replace(`{${name}}`, encodeURIComponent(value));
  }
  if (result.includes("{")) {
    throw new Error(`Aptus API path ${template} is missing a path parameter.`);
  }
  return result;
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const METHOD_LIFECYCLES = new Set([
  "gated-executable",
  "experimental",
  "research-only",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function normalizeMethodCatalog(value: unknown): MethodDescriptor[] {
  if (!Array.isArray(value)) {
    throw new Error("Aptus returned an invalid method catalog.");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) {
      throw new Error(`Aptus method descriptor ${index} is not an object.`);
    }
    const requiredStrings = [
      "schema_version",
      "method_id",
      "display_name",
      "summary",
      "parameter_scope",
      "parameterization",
      "base_storage",
      "pilot_requirement",
    ];
    const requiredArrays = [
      "supported_backends",
      "supported_distributions",
      "evidence_ids",
    ];
    if (
      requiredStrings.some(
        (key) => typeof item[key] !== "string" || !item[key],
      )
      || requiredArrays.some((key) => !isStringArray(item[key]))
      || !METHOD_LIFECYCLES.has(String(item.lifecycle))
      || typeof item.selectable !== "boolean"
      || !(typeof item.compiler_id === "string" || item.compiler_id === null)
      || !(typeof item.export_kind === "string" || item.export_kind === null)
      || !(
        item.blocker === undefined
        || item.blocker === null
        || typeof item.blocker === "string"
      )
      || !(item.aliases === undefined || isStringArray(item.aliases))
    ) {
      throw new Error(`Aptus method descriptor ${index} violates its API contract.`);
    }
    if (
      item.selectable
      && (!item.compiler_id || !item.export_kind)
    ) {
      throw new Error(`Selectable Aptus method descriptor ${index} has no compiler contract.`);
    }
    return item as unknown as MethodDescriptor;
  });
}

function requiredNumber(value: number | null, label: string): number {
  if (value === null || !Number.isFinite(value)) {
    throw new Error(`${label} is required before planning.`);
  }
  return value;
}

function planRequest(facts: FactDraft, projectId?: string | null): PlanRequest {
  const device = facts.hardware.devices[0];
  if (!device) throw new Error("At least one hardware device is required before planning.");
  return {
    ...(projectId ? { project_id: projectId } : {}),
    project_name: facts.project_name.trim() || "Untitled project",
    model: {
      model_id: facts.model.model_id,
      revision: facts.model.revision,
      family: facts.model.family,
      parameters_b: requiredNumber(facts.model.parameters_b, "Model parameters"),
      hidden_size: requiredNumber(facts.model.hidden_size, "Model hidden size"),
      layers: requiredNumber(facts.model.layers, "Model layer count"),
      context_length: requiredNumber(facts.model.context_length, "Model context length"),
      license_name: facts.model.license_name,
      training_allowed: facts.model.training_allowed,
      ...(facts.model.intermediate_size
        ? { intermediate_size: facts.model.intermediate_size }
        : {}),
      ...(facts.model.model_type
        ? { model_type: facts.model.model_type }
        : {}),
      ...(facts.model.architecture
        ? { architecture: facts.model.architecture }
        : {}),
      ...(facts.model.quantization_bits !== null
        && facts.model.quantization_bits !== undefined
        ? { quantization_bits: facts.model.quantization_bits }
        : {}),
      ...(facts.model.quantization_layout
        ? {
            quantization_layout: {
              ...facts.model.quantization_layout,
              module_overrides: facts.model.quantization_layout.module_overrides.map(
                (override) => ({ ...override }),
              ),
            },
          }
        : {}),
      ...(facts.model.moe
        ? {
            moe: {
              ...facts.model.moe,
              mlp_only_layers: [...facts.model.moe.mlp_only_layers],
            },
          }
        : {}),
    },
    hardware: {
      discovery: facts.hardware.discovery,
      backend: device.backend,
      gpu_count: facts.hardware.gpu_count,
      vram_gib: requiredNumber(device.total_vram_gib, "Per-device VRAM"),
      ...(device.free_vram_gib !== null
        ? { free_vram_gib: device.free_vram_gib }
        : {}),
      supports_bf16: device.supports_bf16,
      supports_8bit: device.supports_8bit,
      supports_4bit: device.supports_4bit,
      host_ram_gib: requiredNumber(facts.hardware.host_ram_gib, "Host RAM"),
      ...(facts.hardware.host_ram_free_gib !== null
        ? { host_ram_free_gib: facts.hardware.host_ram_free_gib }
        : {}),
      reserve_gib: requiredNumber(
        facts.hardware.reserve_per_device_gib,
        "Per-device reserve",
      ),
      ...(facts.hardware.disk_free_gib
        ? { disk_free_gib: facts.hardware.disk_free_gib }
        : {}),
    },
    target: {
      objective: facts.target.objective,
      sequence_length: requiredNumber(facts.target.sequence_length, "Sequence length"),
      effective_batch_size: requiredNumber(
        facts.target.effective_batch_size,
        "Effective batch size",
      ),
      max_epochs: requiredNumber(facts.target.max_epochs, "Maximum epochs"),
      ...(facts.target.method_preference
        ? { method_preference: facts.target.method_preference }
        : {}),
      training_runtime: facts.target.runtime,
      task: facts.target.task,
      evaluation_fraction: facts.target.evaluation_fraction,
      packing: facts.target.packing,
      checkpoint_steps: facts.target.checkpoint_steps,
    },
    dataset_path: facts.dataset.source_path,
    ...(facts.dataset.sample_limit
      ? { sample_limit: facts.dataset.sample_limit }
      : {}),
  };
}

function normalizeJob(payload: Record<string, unknown>): Job {
  const id = String(payload.id ?? payload.job_id ?? "");
  if (!id) throw new Error("The Aptus API returned a job without an id.");
  const rawLog = payload.log;
  const hasLogTail = typeof payload.log_tail === "string" || Array.isArray(payload.log_tail);
  const logPath = typeof payload.log_path === "string"
    ? payload.log_path
    : hasLogTail && typeof rawLog === "string"
      ? rawLog
      : undefined;
  return {
    ...payload,
    id,
    job_id: typeof payload.job_id === "string" ? payload.job_id : undefined,
    state: String(payload.state ?? "queued"),
    mode: String(payload.mode ?? payload.action ?? "preflight"),
    log: (payload.log_tail ?? payload.logs ?? (Array.isArray(rawLog) ? rawLog : "")) as string | string[],
    log_path: logPath,
    created_at: typeof payload.created_at === "string" ? payload.created_at : null,
    started_at: typeof payload.started_at === "string" ? payload.started_at : null,
    finished_at:
      typeof payload.finished_at === "string"
        ? payload.finished_at
        : typeof payload.completed_at === "string"
          ? payload.completed_at
          : null,
    return_code:
      typeof payload.return_code === "number" ? payload.return_code : null,
    validation_report:
      typeof payload.validation_report === "object" && payload.validation_report !== null
        ? payload.validation_report as ValidationReport
        : undefined,
    validation_report_error:
      typeof payload.validation_report_error === "string"
        ? payload.validation_report_error
        : undefined,
  } as Job;
}

function normalizeProfile(payload: Record<string, unknown>): ProfileResponse {
  const provenance = (payload.provenance as { kind?: string; source?: string } | undefined);
  const fileKind = provenance?.kind === "measured" ? "measured" : "inferred";
  const tokenKind = payload.measurement === "tokenizer-measured" ? "measured" : "inferred";
  return {
    ...payload,
    facts: [
      {
        key: "dataset_hash",
        label: "Dataset fingerprint",
        value: typeof payload.source_sha256 === "string" ? `${payload.source_sha256.slice(0, 8)}…${payload.source_sha256.slice(-4)}` : null,
        provenance: fileKind,
        source: provenance?.source ?? "Aptus dataset profiler",
      },
      {
        key: "example_count",
        label: "Examples",
        value: payload.example_count ?? null,
        provenance: fileKind,
        source: provenance?.source ?? "Aptus dataset profiler",
      },
      {
        key: "sequence_p95",
        label: "Sequence p95",
        value: payload.sequence_p95 ?? null,
        unit: "tokens",
        provenance: tokenKind,
        source: provenance?.source ?? "Aptus dataset profiler",
      },
      {
        key: "truncation_rate",
        label: "Truncation rate",
        value: payload.truncation_rate ?? null,
        unit: "fraction",
        provenance: tokenKind,
        source: provenance?.source ?? "Aptus dataset profiler",
      },
    ],
  } as ProfileResponse;
}

function normalizePlan(payload: Record<string, unknown>): TrainingPlan {
  const hardware = payload.hardware as Record<string, unknown> | undefined;
  const devices = hardware?.devices as Array<Record<string, unknown>> | undefined;
  const reserveValue = hardware?.reserve_per_device_bytes;
  const reserveBytes = typeof reserveValue === "number" ? reserveValue : 0;
  const capacities = (devices ?? []).flatMap((device, index) => {
      const total = typeof device.total_vram_bytes === "number"
        ? device.total_vram_bytes
        : undefined;
      const free = typeof device.free_vram_bytes === "number"
        ? device.free_vram_bytes
        : total;
      return free === undefined
        ? []
        : [{ index, total, limit: free - reserveBytes }];
    });
  const limitingDevice = capacities.reduce<{ index: number; total?: number; limit: number } | undefined>(
      (current, device) =>
        current === undefined || device.limit < current.limit
          ? device
          : current,
      undefined,
    );
  const totalBytes = limitingDevice?.total;
  const limitBytes = limitingDevice?.limit;

  const normalizeCandidate = (value: unknown): CandidatePlan => {
    const candidate = value as CandidatePlan;
    const memory = candidate.memory ?? {};
    const selectedIndices = Array.isArray(candidate.device_indices)
      ? new Set(candidate.device_indices)
      : null;
    const selectedCapacities = selectedIndices
      ? capacities.filter((device) => selectedIndices.has(device.index))
      : [];
    const candidateLimit = selectedCapacities.reduce<typeof limitingDevice>(
      (current, device) => current === undefined || device.limit < current.limit ? device : current,
      undefined,
    ) ?? limitingDevice;
    return {
      ...candidate,
      id: candidate.id ?? candidate.candidate_id,
      batches: candidate.batches ?? {
        micro_batch_size: candidate.micro_batch_size,
        gradient_accumulation_steps: candidate.gradient_accumulation_steps,
        effective_batch_size: candidate.effective_batch_size,
      },
      memory: {
        ...memory,
        expected_bytes: memory.expected_bytes ?? memory.point_estimate_bytes ?? memory.estimated_peak_bytes,
        upper_bytes: memory.upper_bytes ?? memory.upper_estimate_bytes,
        limit_bytes: memory.limit_bytes ?? candidateLimit?.limit ?? limitBytes,
        device_total_bytes: memory.device_total_bytes ?? candidateLimit?.total ?? totalBytes,
      },
    };
  };

  const candidates = Array.isArray(payload.candidates)
    ? payload.candidates.map(normalizeCandidate)
    : [];
  const recommended = payload.recommended
    ? normalizeCandidate(payload.recommended)
    : null;
  const rationale = Array.isArray(payload.rationale)
    ? payload.rationale as string[]
    : Array.isArray(payload.recommendation_rationale)
      ? payload.recommendation_rationale as string[]
      : [];
  return {
    ...payload,
    recommended,
    candidates,
    warnings: Array.isArray(payload.warnings) ? payload.warnings as string[] : [],
    rationale,
    recommendation_rationale: rationale,
  } as TrainingPlan;
}

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export const SUPPORTED_API_CONTRACT_VERSION = "aptus.api.v1" as const;

function assertSupportedApiContract(payload: Record<string, unknown>): void {
  const contractVersion = payload.api_contract_version;
  if (contractVersion !== SUPPORTED_API_CONTRACT_VERSION) {
    throw new Error(
      `Missing or unsupported Aptus API contract ${JSON.stringify(contractVersion)}. ` +
      `This workbench requires ${SUPPORTED_API_CONTRACT_VERSION}.`,
    );
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload: unknown = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "object" && payload !== null
      ? "candidates" in payload
        ? payload
        : "detail" in payload
          ? (payload as { detail: unknown }).detail
          : "details" in payload
            ? (payload as { details: unknown }).details
            : payload
      : payload;
    const message =
      typeof payload === "object" && payload !== null && "message" in payload && typeof payload.message === "string"
        ? payload.message
        : typeof detail === "string"
          ? detail
          : `Aptus API returned ${response.status}.`;
    throw new ApiError(message, response.status, detail);
  }

  return payload as T;
}

export const api = {
  async bootstrap(signal?: AbortSignal) {
    const response = await request<OpenApiBootstrapResponse>(API_PATHS.bootstrap, { signal });
    const payload = response as unknown as Record<string, unknown>;
    assertSupportedApiContract(payload);
    const capabilities = isRecord(payload.capabilities)
      ? { ...payload.capabilities }
      : undefined;
    if (capabilities && "method_catalog" in capabilities) {
      capabilities.method_catalog = normalizeMethodCatalog(
        capabilities.method_catalog,
      );
    }
    const plan = typeof payload.plan === "object" && payload.plan !== null
      ? normalizePlan(payload.plan as Record<string, unknown>)
      : null;
    const job = typeof payload.job === "object" && payload.job !== null
      ? normalizeJob(payload.job as Record<string, unknown>)
      : null;
    return {
      ...payload,
      capabilities,
      plan,
      bundle:
        typeof payload.bundle === "object" && payload.bundle !== null
          ? payload.bundle as CompileResponse
          : null,
      job,
      projects: Array.isArray(payload.projects)
        ? payload.projects as ProjectSummary[]
        : [],
      project:
        typeof payload.project === "object" && payload.project !== null
          ? payload.project as ProjectDetail
          : null,
      project_history: Array.isArray(payload.project_history)
        ? payload.project_history as ProjectRevisionSummary[]
        : [],
    } as BootstrapResponse;
  },

  async profile(facts: FactDraft) {
    const body: ProfileRequest = {
      dataset_path: facts.dataset.source_path,
      ...(facts.dataset.sample_limit ? { sample_limit: facts.dataset.sample_limit } : {}),
      ...(facts.target.sequence_length ? { sequence_length: facts.target.sequence_length } : {}),
    };
    const response = await request<OpenApiProfileResponse>(API_PATHS.profile, {
      method: "POST",
      body: JSON.stringify(body),
    });
    return normalizeProfile(response as unknown as Record<string, unknown>);
  },

  async inspectModel(modelId: string, revision: string) {
    const response = await request<OpenApiModelInspectionResponse>(API_PATHS.modelInspect, {
      method: "POST",
      body: JSON.stringify({ model_id: modelId, revision }),
    });
    const payload = response as unknown as Record<string, unknown>;
    return {
      ...payload,
      compatibility: normalizeModelCompatibility(payload.compatibility),
    } as unknown as ModelInspectionResponse;
  },

  async plan(facts: FactDraft, projectId?: string | null) {
    const body = planRequest(facts, projectId);
    try {
      const response = await request<OpenApiTrainingPlanResponse>(API_PATHS.plan, {
        method: "POST",
        body: JSON.stringify(body),
      });
      return normalizePlan(response as unknown as Record<string, unknown>);
    } catch (error) {
      const detail = error instanceof ApiError && typeof error.detail === "object" && error.detail !== null
        ? error.detail as Record<string, unknown>
        : null;
      if (
        error instanceof ApiError &&
        error.status === 422 &&
        detail?.error === "no_feasible_plan" &&
        Array.isArray(detail.candidates)
      ) {
        return normalizePlan({
          recommended: null,
          candidates: detail.candidates,
          warnings: [typeof detail.message === "string" ? detail.message : error.message],
          recommendation_rationale: [
            "No candidate passed every hard gate. Review the rejection reasons before changing facts.",
          ],
          no_feasible_plan: true,
        });
      }
      throw error;
    }
  },

  async compileBundle(plan: TrainingPlan, outputDir: string) {
    if (!plan.plan_id) throw new Error("The plan id is required before compilation.");
    if (!plan.project_id || !plan.project_revision_id) {
      throw new Error(
        "Compilation requires the plan's exact project and project revision identity.",
      );
    }
    const body: CompileRequest = {
      plan_id: plan.plan_id,
      output_dir: outputDir,
      project_id: plan.project_id,
      expected_project_revision_id: plan.project_revision_id,
    };
    const payload = await request<OpenApiCompileResponse>(API_PATHS.compile, {
      method: "POST",
      body: JSON.stringify(body),
    });
    return payload as unknown as CompileResponse & {
      project_id: string;
      project_revision_id: string;
    };
  },

  validate(
    bundleDir: string,
    level: ValidateRequest["level"],
    run: boolean,
    projectId: string,
    expectedProjectRevisionId: string,
  ) {
    const body: ValidateRequest = {
      bundle_dir: bundleDir,
      project_id: projectId,
      expected_project_revision_id: expectedProjectRevisionId,
      level,
      run,
    };
    return request<OpenApiValidationResponse>(API_PATHS.validate, {
      method: "POST",
      body: JSON.stringify(body),
    }) as Promise<ValidationReport & {
      project_id: string;
      project_revision_id: string;
    }>;
  },

  async createJob(body: JobRequest) {
    const payload = await request<OpenApiJobResponse>(API_PATHS.jobs, {
      method: "POST",
      body: JSON.stringify(body),
    });
    return normalizeJob(payload as unknown as Record<string, unknown>);
  },

  async getJob(id: string, signal?: AbortSignal) {
    const payload = await request<OpenApiJobResponse>(
      bindApiPath(API_PATHS.job, { job_id: id }),
      { signal },
    );
    return normalizeJob(payload as unknown as Record<string, unknown>);
  },

  async cancelJob(id: string) {
    const payload = await request<OpenApiJobResponse>(
      bindApiPath(API_PATHS.jobCancel, { job_id: id }),
      { method: "POST" },
    );
    return normalizeJob(payload as unknown as Record<string, unknown>);
  },

  listProjects() {
    return request<OpenApiProjectSummaryResponse[]>(API_PATHS.projects) as Promise<ProjectSummary[]>;
  },

  getProject(projectId: string) {
    return request<OpenApiProjectResponse>(
      bindApiPath(API_PATHS.project, { project_id: projectId }),
    ) as Promise<ProjectDetail>;
  },

  projectHistory(projectId: string) {
    return request<OpenApiProjectRevisionSummary[]>(
      bindApiPath(API_PATHS.projectRevisions, { project_id: projectId }),
    ) as Promise<ProjectRevisionSummary[]>;
  },

  projectRevision(projectId: string, revisionId: string) {
    return request<OpenApiProjectRevisionResponse>(
      bindApiPath(API_PATHS.projectRevision, {
        project_id: projectId,
        revision_id: revisionId,
      }),
    ) as unknown as Promise<ProjectRevision>;
  },

  recoverProjectRevision(projectId: string, revisionId: string) {
    return request<OpenApiProjectRecoveryResponse>(
      bindApiPath(API_PATHS.projectRecover, { project_id: projectId }),
      {
        method: "POST",
        body: JSON.stringify({ revision_id: revisionId }),
      },
    ) as unknown as Promise<ProjectRecoveryResponse>;
  },

  async hardware() {
    const envelope = await request<OpenApiHardwareProbeResponse>(API_PATHS.hardware);
    if (envelope.status === "unavailable" || !envelope.hardware) {
      throw new Error(envelope.error ?? "Hardware inspection is unavailable on this Aptus host.");
    }
    return envelope.hardware;
  },

  platform() {
    return request<OpenApiPlatformResponse>(API_PATHS.platform) as Promise<ApplePlatformResponse>;
  },

  runtimes() {
    return request<OpenApiRuntimeInventoryResponse>(API_PATHS.runtimes) as Promise<RuntimeInventory>;
  },

  inferenceServices() {
    return request<OpenApiInferenceServicesResponse>(API_PATHS.inferenceServices);
  },

  inferenceModels(body: InferenceServiceRequest) {
    return request<OpenApiInferenceModelsResponse>(API_PATHS.inferenceModels, {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  inferenceGenerate(body: InferenceGenerateRequest) {
    return request<OpenApiInferenceGenerateResponse>(API_PATHS.inferenceGenerate, {
      method: "POST",
      body: JSON.stringify(body),
    });
  },
};
