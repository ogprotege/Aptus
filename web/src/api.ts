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
  ProfileRequest,
  ProfileResponse,
  TrainingPlan,
  ValidateRequest,
  ValidationReport,
} from "./types";

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

function planRequest(facts: FactDraft): PlanRequest {
  const device = facts.hardware.devices[0];
  if (!device) throw new Error("At least one hardware device is required before planning.");
  return {
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
    const payload = await request<Record<string, unknown>>("/api/v1/bootstrap", { signal });
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
    } as BootstrapResponse;
  },

  async profile(facts: FactDraft) {
    const body: ProfileRequest = {
      dataset_path: facts.dataset.source_path,
      ...(facts.dataset.sample_limit ? { sample_limit: facts.dataset.sample_limit } : {}),
      ...(facts.target.sequence_length ? { sequence_length: facts.target.sequence_length } : {}),
    };
    const payload = await request<Record<string, unknown>>("/api/v1/profile", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return normalizeProfile(payload);
  },

  inspectModel(modelId: string, revision: string) {
    return request<ModelInspectionResponse>("/api/v1/models/inspect", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId, revision }),
    });
  },

  async plan(facts: FactDraft) {
    const body = planRequest(facts);
    try {
      const payload = await request<Record<string, unknown>>("/api/v1/plan", {
        method: "POST",
        body: JSON.stringify(body),
      });
      return normalizePlan(payload);
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
    const body: CompileRequest = { plan_id: plan.plan_id, output_dir: outputDir };
    const payload = await request<CompileResponse | ValidationReport>("/api/v1/compile", {
      method: "POST",
      body: JSON.stringify(body),
    });
    if ("bundle_dir" in payload) return payload as CompileResponse;
    return {
      bundle_dir: outputDir,
      files: payload.checked_files ?? [],
      report: payload,
    };
  },

  validate(
    bundleDir: string,
    level: ValidateRequest["level"],
    run: boolean,
  ) {
    const body: ValidateRequest = { bundle_dir: bundleDir, level, run };
    return request<ValidationReport>("/api/v1/validate", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  async createJob(body: JobRequest) {
    const payload = await request<Record<string, unknown>>("/api/v1/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return normalizeJob(payload);
  },

  async getJob(id: string, signal?: AbortSignal) {
    const payload = await request<Record<string, unknown>>(
      `/api/v1/jobs/${encodeURIComponent(id)}`,
      { signal },
    );
    return normalizeJob(payload);
  },

  async cancelJob(id: string) {
    const payload = await request<Record<string, unknown>>(
      `/api/v1/jobs/${encodeURIComponent(id)}/cancel`,
      { method: "POST" },
    );
    return normalizeJob(payload);
  },

  async hardware() {
    const envelope = await request<HardwareProbeResponse>("/api/v1/hardware");
    if (envelope.status === "unavailable" || !envelope.hardware) {
      throw new Error(envelope.error ?? "Hardware inspection is unavailable on this Aptus host.");
    }
    return envelope.hardware;
  },

  platform() {
    return request<ApplePlatformResponse>("/api/v1/platform");
  },

  runtimes() {
    return request<RuntimeInventory>("/api/v1/runtimes");
  },

  inferenceServices() {
    return request<Record<string, unknown>>("/api/v1/inference/services");
  },

  inferenceModels(body: InferenceServiceRequest) {
    return request<Record<string, unknown>>("/api/v1/inference/models", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  inferenceGenerate(body: InferenceGenerateRequest) {
    return request<Record<string, unknown>>("/api/v1/inference/generate", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },
};
