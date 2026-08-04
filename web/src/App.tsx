import { useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { api, ApiError } from "./api";
import {
  DESKTOP_WORKBENCH_READY_MARKER,
  getDesktopBridge,
} from "./desktopBridge";
import { FitLedger } from "./components/FitLedger";
import { ProjectHistory } from "./components/ProjectHistory";
import { summarizeHardwareProbe } from "./lib/hardware";
import {
  applyPlanDerivedModelFacts,
  applyProviderModelInspection,
} from "./lib/modelInspection";
import {
  buildModelPolicyPresentation,
  validationReportMatchesBinding,
} from "./lib/modelPolicy";
import type { ValidationReportBindingIdentity } from "./lib/modelPolicy";
import {
  MobileStageBar,
  WorkflowRail,
  WORKFLOW_STAGES,
} from "./components/WorkflowRail";
import {
  EMPTY_DRAFT,
  EXAMPLE_BUNDLE,
  EXAMPLE_DRAFT,
  EXAMPLE_JOB,
  EXAMPLE_PLAN,
  EXAMPLE_PROFILE,
  EXAMPLE_REPORT,
} from "./demo";
import { CompareStage } from "./stages/CompareStage";
import { CompileStage } from "./stages/CompileStage";
import { FactsStage } from "./stages/FactsStage";
import { RunStage } from "./stages/RunStage";
import { ValidateStage } from "./stages/ValidateStage";
import type {
  BootstrapResponse,
  CandidatePlan,
  CompileResponse,
  FactDraft,
  InputProfile,
  Job,
  MethodDescriptor,
  ModelInspectionReceipt,
  ModelInspectionResponse,
  PlanView,
  ProjectDetail,
  ProjectRevisionSummary,
  ProjectSummary,
  TrainingPlan,
  ValidationReport,
  ValidateRequest,
  WorkflowStage,
} from "./types";

type ConnectionState = "connecting" | "connected" | "unavailable";
const ACTIVE_JOB_STATES = new Set(["queued", "running", "cancelling"]);

function isActiveJob(job: Job | null): job is Job {
  return Boolean(job && ACTIVE_JOB_STATES.has(job.state));
}

function isBoundTrainingPlan(plan: PlanView | null): plan is TrainingPlan {
  return Boolean(plan && "schema_version" in plan);
}

function validationRank(state: string | undefined): number {
  return [
    "invalid",
    "contract-pass",
    "static-pass",
    "dependency-pass",
    "model-data-pass",
    "measured-preflight-pass",
    "pilot-pass",
    "execution-approved",
    "measured-run-pass",
  ].indexOf(state ?? "invalid");
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function restoredDraft(
  plan: TrainingPlan,
  bundle: CompileResponse | null,
  projectName?: string,
): FactDraft {
  const model = plan.model ?? {};
  const dataset = plan.dataset ?? {};
  const hardware = plan.hardware ?? {};
  const target = plan.target ?? {};
  const hardwareProvenance = hardware.provenance as Record<string, unknown> | undefined;
  const hardwareWasMeasured = hardwareProvenance?.kind === "measured";
  const devices = Array.isArray(hardware.devices)
    ? hardware.devices as Array<Record<string, unknown>>
    : [];
  const bytesToGiB = (value: unknown) => {
    const numeric = numberValue(value);
    return numeric === null ? null : numeric / 1024 ** 3;
  };
  const source = typeof dataset.source_path === "string" ? dataset.source_path : "";
  const sourcePath = source && !source.startsWith("/") && bundle
    ? `${bundle.bundle_dir.replace(/\/$/, "")}/${source}`
    : source;
  const finiteMinimum = (values: Array<number | null>) => {
    const present = values.filter((value): value is number => value !== null);
    return present.length ? Math.min(...present) : null;
  };
  const limitingDevice = devices.length
    ? {
          name: devices.length === 1
          ? String(devices[0]?.name ?? "Restored CUDA GPU")
          : `Distributed limiting profile across ${devices.length} restored GPUs`,
        backend: devices.every(
          (device) => device.backend === devices[0]?.backend,
        )
          ? String(devices[0]?.backend ?? "cuda")
          : "unknown",
        total_vram_gib: finiteMinimum(devices.map((device) => bytesToGiB(device.total_vram_bytes))),
        free_vram_gib: finiteMinimum(devices.map((device) => bytesToGiB(device.free_vram_bytes))),
        supports_bf16: devices.every((device) => device.supports_bf16 === true),
        supports_8bit: devices.every((device) => device.supports_8bit === true),
        supports_4bit: devices.every((device) => device.supports_4bit === true),
      }
    : null;
  return {
    project_name: projectName
      ?? (typeof plan.plan_id === "string" ? plan.plan_id : "Restored plan"),
    model: {
      model_id: String(model.model_id ?? ""),
      revision: String(model.revision ?? ""),
      family: String(model.family ?? ""),
      parameters_b: numberValue(model.parameters) === null ? null : Number(model.parameters) / 1_000_000_000,
      hidden_size: numberValue(model.hidden_size),
      layers: numberValue(model.layers),
      context_length: numberValue(model.context_length),
      intermediate_size: numberValue(model.intermediate_size),
      license_name: String(model.license_name ?? ""),
      training_allowed: model.training_allowed === true,
      model_type: typeof model.model_type === "string" ? model.model_type : null,
      architecture: typeof model.architecture === "string" ? model.architecture : null,
      quantization_bits: numberValue(model.quantization_bits),
      quantization_layout:
        typeof model.quantization_layout === "object" && model.quantization_layout !== null
          ? structuredClone(model.quantization_layout) as FactDraft["model"]["quantization_layout"]
          : null,
      moe:
        typeof model.moe === "object" && model.moe !== null
          ? structuredClone(model.moe) as FactDraft["model"]["moe"]
          : null,
      active_parameters_b:
        numberValue(model.active_parameters) === null
          ? null
          : Number(model.active_parameters) / 1_000_000_000,
      sparse_layer_count: numberValue(model.sparse_layer_count),
    },
    dataset: {
      source_path: sourcePath,
      format: String(dataset.source_format ?? "jsonl"),
      schema_name: String(dataset.schema_name ?? "text"),
      tokenizer_id: String(model.tokenizer_id ?? model.model_id ?? ""),
      sample_limit: numberValue(dataset.sampled_examples),
    },
    hardware: {
      discovery: hardwareWasMeasured ? "local-scan" : "manual",
      gpu_count: devices.length || 1,
      devices: limitingDevice
        ? [limitingDevice]
        : structuredClone(EMPTY_DRAFT.hardware.devices),
      host_ram_gib: bytesToGiB(hardware.host_ram_bytes),
      host_ram_free_gib: bytesToGiB(hardware.host_ram_free_bytes),
      reserve_per_device_gib: bytesToGiB(hardware.reserve_per_device_bytes),
      disk_free_gib: bytesToGiB(hardware.disk_free_bytes),
    },
    target: {
      task: String(target.task ?? "sft"),
      objective: ["quality", "memory", "speed"].includes(String(target.objective))
        ? String(target.objective) as FactDraft["target"]["objective"]
        : "quality",
      sequence_length: numberValue(target.sequence_length),
      effective_batch_size: numberValue(target.effective_batch_size),
      max_epochs: numberValue(target.max_epochs),
      method_preference: String(target.method_preference ?? ""),
      runtime: String(
        target.training_runtime
          ?? plan.recommended?.runtime_contract?.training_runtime
          ?? "transformers-peft-cuda",
      ),
      evaluation_fraction: numberValue(target.evaluation_fraction) ?? 0.1,
      packing: target.packing === true,
      checkpoint_steps: numberValue(target.checkpoint_steps) ?? 100,
    },
  };
}

function restoredHardwareWasMeasured(plan: TrainingPlan): boolean {
  const provenance = plan.hardware?.provenance as Record<string, unknown> | undefined;
  return provenance?.kind === "measured";
}

function restoredProfile(plan: TrainingPlan): InputProfile {
  const dataset = plan.dataset ?? {};
  return {
    ...dataset,
    facts: [
      {
        key: "dataset_hash",
        label: "Dataset fingerprint",
        value: String(dataset.source_sha256 ?? ""),
        provenance: "measured",
        source: "Restored compiled bundle",
      },
      {
        key: "example_count",
        label: "Examples",
        value: numberValue(dataset.example_count),
        provenance: "measured",
        source: "Restored compiled bundle",
      },
      {
        key: "sequence_p95",
        label: "Sequence p95",
        value: numberValue(dataset.sequence_p95),
        unit: "tokens",
        provenance: dataset.measurement === "tokenizer-measured" ? "measured" : "inferred",
        source: "Restored compiled bundle",
      },
    ],
    warnings: ["These facts were restored from the compiled artifact. Edit them only to begin a new plan."],
  };
}

function freshDraft(): FactDraft {
  return structuredClone(EMPTY_DRAFT);
}

function inspectionReceiptFromPlan(plan: PlanView | null): ModelInspectionReceipt | null {
  return isBoundTrainingPlan(plan) && plan.inspection_receipt
    ? structuredClone(plan.inspection_receipt)
    : null;
}

function mergeDefaults(current: FactDraft, defaults: Partial<FactDraft>): FactDraft {
  return {
    ...current,
    ...defaults,
    model: { ...current.model, ...defaults.model },
    dataset: { ...current.dataset, ...defaults.dataset },
    hardware: {
      ...current.hardware,
      ...defaults.hardware,
      devices: defaults.hardware?.devices ?? current.hardware.devices,
    },
    target: { ...current.target, ...defaults.target },
  };
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Aptus could not complete this action.";
}

function outputSlug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "") || "training-bundle";
}

export default function App() {
  const embeddedInDesktop = getDesktopBridge() !== null;
  const [stage, setStage] = useState<WorkflowStage>("facts");
  const [draft, setDraft] = useState<FactDraft>(freshDraft);
  const [profile, setProfile] = useState<InputProfile | null>(null);
  const [plan, setPlan] = useState<PlanView | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<CandidatePlan | null>(null);
  const [bundle, setBundle] = useState<CompileResponse | null>(null);
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [bootstrapReady, setBootstrapReady] = useState(false);
  const [serviceVersion, setServiceVersion] = useState<string | undefined>();
  const [demoMode, setDemoMode] = useState(false);
  const [hardwareScanned, setHardwareScanned] = useState(false);
  const [modelInspection, setModelInspection] = useState<ModelInspectionResponse | null>(null);
  const [inspectionReceipt, setInspectionReceipt] = useState<ModelInspectionReceipt | null>(null);
  const [methodCatalog, setMethodCatalog] = useState<MethodDescriptor[]>([]);
  const [outputDir, setOutputDir] = useState("");
  const [validationLevel, setValidationLevel] = useState<ValidateRequest["level"]>("static");
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [currentProject, setCurrentProject] = useState<ProjectDetail | null>(null);
  const [projectHistory, setProjectHistory] = useState<ProjectRevisionSummary[]>([]);
  const bundleRef = useRef<CompileResponse | null>(null);
  const draftVersionRef = useRef(0);
  const workbenchReadyReportedRef = useRef(false);

  useEffect(() => {
    bundleRef.current = bundle;
  }, [bundle]);

  const applyJobUpdate = (nextJob: Job) => {
    setJob(nextJob);
    const currentBundle = bundleRef.current;
    const reportMatchesBundle = Boolean(
      currentBundle
      && nextJob.bundle_dir
      && currentBundle.bundle_dir === nextJob.bundle_dir,
    );
    if (!currentBundle || !reportMatchesBundle) return;
    let nextBundle = currentBundle;
    if (nextJob.project_id && nextJob.project_revision_id) {
      nextBundle = {
        ...nextBundle,
        project_id: nextJob.project_id,
        project_revision_id: nextJob.project_revision_id,
      };
    }
    if (nextJob.validation_report) {
      setReport(nextJob.validation_report);
      nextBundle = { ...nextBundle, report: nextJob.validation_report };
    }
    if (nextJob.validation_report_error) {
      setReport(null);
      nextBundle = { ...nextBundle, report: undefined };
      setError(nextJob.validation_report_error);
    }
    bundleRef.current = nextBundle;
    setBundle(nextBundle);
  };

  useEffect(() => {
    const controller = new AbortController();
    const bootstrapDraftVersion = draftVersionRef.current;
    api.bootstrap(controller.signal)
      .then((bootstrap: BootstrapResponse) => {
        setConnection("connected");
        setServiceVersion(bootstrap.service?.version ?? bootstrap.version);
        if (Array.isArray(bootstrap.capabilities?.method_catalog)) {
          setMethodCatalog(bootstrap.capabilities.method_catalog);
        }
        setProjects(bootstrap.projects ?? []);
        setCurrentProject(bootstrap.project ?? null);
        setProjectHistory(bootstrap.project_history ?? []);
        const restoreWorkspace = draftVersionRef.current === bootstrapDraftVersion;
        if (bootstrap.defaults && restoreWorkspace) {
          const defaults = bootstrap.defaults;
          setDraft((current) => {
            const merged = mergeDefaults(current, defaults);
            return {
              ...merged,
              dataset: {
                ...merged.dataset,
                sample_limit: defaults.sample_limit ?? merged.dataset.sample_limit,
              },
              hardware: {
                ...merged.hardware,
                devices: defaults.backend
                  ? [{ ...merged.hardware.devices[0], backend: defaults.backend }]
                  : merged.hardware.devices,
                reserve_per_device_gib: defaults.reserve_gib ?? merged.hardware.reserve_per_device_gib,
              },
              target: {
                ...merged.target,
                runtime: defaults.training_runtime ?? merged.target.runtime,
                task: defaults.task ?? merged.target.task,
                packing: defaults.packing ?? merged.target.packing,
              },
            };
          });
        }
        if (bootstrap.plan && restoreWorkspace) {
          setPlan(bootstrap.plan);
          setInspectionReceipt(inspectionReceiptFromPlan(bootstrap.plan));
          setSelectedCandidate(bootstrap.plan.recommended ?? bootstrap.plan.candidates[0] ?? null);
        }
        if (bootstrap.bundle && restoreWorkspace) {
          bundleRef.current = bootstrap.bundle;
          setBundle(bootstrap.bundle);
          setReport(bootstrap.bundle.report ?? null);
        }
        if (bootstrap.plan && restoreWorkspace) {
          setDraft(restoredDraft(
            bootstrap.plan,
            bootstrap.bundle ?? null,
            bootstrap.project?.name,
          ));
          setProfile(restoredProfile(bootstrap.plan));
          setHardwareScanned(restoredHardwareWasMeasured(bootstrap.plan));
          setNotice(
            bootstrap.bundle
              ? "Restored the latest validated local artifact and its bound facts."
            : "Restored the latest local project revision. Training authorization was not restored.",
          );
        }
        if (bootstrap.replan_required && restoreWorkspace) {
          setError(bootstrap.replan_required.message);
        }
        if (bootstrap.job) {
          applyJobUpdate(bootstrap.job);
          if (ACTIVE_JOB_STATES.has(bootstrap.job.state)) setStage("run");
        }
        setBootstrapReady(true);
      })
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        if (caught instanceof Error) setError(caught.message);
        setConnection("unavailable");
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!bootstrapReady || workbenchReadyReportedRef.current) return;
    const desktopBridge = getDesktopBridge();
    if (!desktopBridge) return;

    workbenchReadyReportedRef.current = true;
    void desktopBridge.reportWorkbenchReady().catch(() => {
      workbenchReadyReportedRef.current = false;
    });
  }, [bootstrapReady]);

  useEffect(() => {
    document.querySelector<HTMLElement>("#stage-heading")?.focus();
  }, [stage]);

  useEffect(() => {
    if (demoMode || !isActiveJob(job)) return;
    let stopped = false;
    let timer: number | undefined;
    let controller: AbortController | null = null;
    const pollingErrorPrefix = `Could not refresh active job ${job.id}.`;
    const poll = async () => {
      controller = new AbortController();
      try {
        applyJobUpdate(await api.getJob(job.id, controller.signal));
        setError((current) => current?.startsWith(pollingErrorPrefix) ? null : current);
      } catch (caught) {
        if (!(caught instanceof DOMException && caught.name === "AbortError")) {
          setError(`${pollingErrorPrefix} The displayed state may be stale; use Refresh job to retry.`);
        }
      } finally {
        if (!stopped) timer = window.setTimeout(() => void poll(), 2000);
      }
    };
    timer = window.setTimeout(() => void poll(), 2000);
    return () => {
      stopped = true;
      controller?.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [demoMode, job?.id, job?.state]);

  const activeReport = report ?? bundle?.report ?? null;
  const validationBinding: ValidationReportBindingIdentity | null = isBoundTrainingPlan(plan)
    && plan.recommended
    && typeof plan.recommended.candidate_id === "string"
    && typeof plan.model?.revision === "string"
    ? {
        planId: plan.plan_id,
        candidateId: plan.recommended.candidate_id,
        modelRevision: plan.model.revision,
      }
    : null;
  const boundActiveReport = validationReportMatchesBinding(
    activeReport,
    validationBinding,
  ) ? activeReport : null;
  const completed = useMemo(() => {
    const values = new Set<WorkflowStage>();
    if (profile) values.add("facts");
    if (plan) values.add("compare");
    if (bundle) values.add("compile");
    if (validationRank(boundActiveReport?.state) >= validationRank("static-pass")) {
      values.add("validate");
    }
    if (
      job?.state === "completed"
      && (job.mode === "train" || job.action === "train")
      && bundle
      && boundActiveReport
      && job.bundle_dir === bundle.bundle_dir
    ) values.add("run");
    return values;
  }, [profile, plan, bundle, boundActiveReport, job]);

  const selected = selectedCandidate ?? plan?.recommended ?? plan?.candidates[0] ?? null;
  const modelPolicyPresentation = useMemo(() => {
    const decision = plan?.model_policy_decision
      ?? inspectionReceipt?.decision
      ?? null;
    if (!decision) return null;
    const source = plan?.model_policy_decision_source ?? "provider-inspection";
    const planModel = plan?.model ?? null;
    const modelId = typeof planModel?.model_id === "string"
      ? planModel.model_id
      : draft.model.model_id;
    const revision = typeof planModel?.revision === "string"
      ? planModel.revision
      : draft.model.revision;
    return buildModelPolicyPresentation({
      decision,
      source,
      candidate: plan ? selected : null,
      report: plan ? activeReport : null,
      modelId,
      revision,
      planId: isBoundTrainingPlan(plan) ? plan.plan_id : null,
    });
  }, [
    plan,
    inspectionReceipt,
    selected,
    activeReport,
    draft.model.model_id,
    draft.model.revision,
  ]);
  const currentStageLabel = WORKFLOW_STAGES.find((item) => item.id === stage)?.label ?? "Facts";
  const activeJob = isActiveJob(job);

  const selectStage = (nextStage: WorkflowStage) => {
    if (nextStage === stage) {
      document.querySelector<HTMLElement>("#stage-heading")?.focus();
      return;
    }
    setStage(nextStage);
  };

  const blockMutationDuringRun = (action: string): boolean => {
    if (!activeJob) return false;
    setError(`Job ${job.id} is ${job.phase ?? job.state}. ${action} is blocked until the local GPU job reaches a terminal state.`);
    setStage("run");
    return true;
  };

  const updateDraft: Dispatch<SetStateAction<FactDraft>> = (action) => {
    if (blockMutationDuringRun("Changing facts")) return;
    if (busy !== null) {
      setError(`Aptus is still completing ${busy}. Wait for it to finish before changing facts.`);
      return;
    }
    draftVersionRef.current += 1;
    setDraft((current) => {
      const next = typeof action === "function" ? action(current) : action;
      return {
        ...next,
        model: {
          ...next.model,
          active_parameters_b: null,
          sparse_layer_count: null,
        },
      };
    });
    setProfile(null);
    setPlan(null);
    setSelectedCandidate(null);
    setBundle(null);
    setReport(null);
    setJob(null);
    setDemoMode(false);
    setHardwareScanned(false);
  };

  const beginAction = (name: string) => {
    setBusy(name);
    setError(null);
    setNotice(null);
  };

  const finishAction = () => setBusy(null);

  const refreshProjectSurface = async (projectId: string) => {
    const [nextProjects, nextProject, nextHistory] = await Promise.all([
      api.listProjects(),
      api.getProject(projectId),
      api.projectHistory(projectId),
    ]);
    setProjects(nextProjects);
    setCurrentProject(nextProject);
    setProjectHistory(nextHistory);
  };

  const handleProfile = async () => {
    if (demoMode) {
      setError("Clear the labeled example before calling the profiler.");
      return;
    }
    if (blockMutationDuringRun("Profiling new inputs")) return;
    beginAction("profile");
    const requestDraftVersion = draftVersionRef.current;
    try {
      const response = await api.profile(draft);
      if (draftVersionRef.current !== requestDraftVersion) {
        setNotice("Facts changed while profiling. Aptus discarded the older response; profile the current facts again.");
        return;
      }
      setProfile(response);
      setPlan(null);
      setSelectedCandidate(null);
      setBundle(null);
      setReport(null);
      setJob(null);
      setDemoMode(false);
      setConnection("connected");
      setNotice("Input profile complete. Review the evidence, then compare strategies.");
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      finishAction();
    }
  };

  const handleModelInspect = async () => {
    if (demoMode) {
      setError("Clear the labeled example before inspecting a real model.");
      return;
    }
    if (blockMutationDuringRun("Inspecting and replacing model facts")) return;
    const modelId = draft.model.model_id.trim();
    const revision = draft.model.revision.trim();
    if (!modelId || !revision) {
      setError("Enter a model ID and a revision before asking the provider to inspect it.");
      return;
    }
    beginAction("inspect-model");
    const requestDraftVersion = draftVersionRef.current;
    try {
      const inspection = await api.inspectModel(modelId, revision);
      if (inspection.status !== "ok" || !inspection.facts || !inspection.resolved_revision) {
        setModelInspection(inspection);
        throw new Error(inspection.error ?? "The provider did not return revision-bound model facts.");
      }
      const receipt = inspection.inspection_receipt;
      if (
        !receipt
        || receipt.schema_version !== "aptus.model-inspection-receipt.v1"
        || receipt.model_id !== modelId
        || receipt.resolved_revision.toLowerCase() !== inspection.resolved_revision.toLowerCase()
      ) {
        throw new Error(
          "The provider inspection did not return a receipt bound to this model and resolved revision.",
        );
      }
      if (draftVersionRef.current !== requestDraftVersion) {
        setNotice("Model facts changed during inspection. Aptus did not apply the older provider response.");
        return;
      }
      setModelInspection(inspection);
      setInspectionReceipt(structuredClone(receipt));
      draftVersionRef.current += 1;
      setDraft((current) => ({
        ...current,
        model: applyProviderModelInspection(current.model, inspection),
      }));
      setProfile(null);
      setPlan(null);
      setSelectedCandidate(null);
      setBundle(null);
      setReport(null);
      setJob(null);
      setConnection("connected");
      setNotice("Provider-declared architecture facts were applied and the revision was pinned. Parameter count and training permission remain your explicit facts.");
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      finishAction();
    }
  };

  const handlePlan = async () => {
    if (demoMode) {
      setError("Clear the labeled example before creating a real plan.");
      return;
    }
    if (blockMutationDuringRun("Creating a new plan")) return;
    beginAction("plan");
    const requestDraftVersion = draftVersionRef.current;
    try {
      const projectId = currentProject?.project_id ?? plan?.project_id ?? null;
      const nextPlan = inspectionReceipt
        ? await api.plan(draft, projectId, inspectionReceipt)
        : await api.plan(draft, projectId);
      if (draftVersionRef.current !== requestDraftVersion) {
        setNotice("Facts changed while planning. Aptus discarded the older response; compare the current facts again.");
        return;
      }
      setPlan(nextPlan);
      setDraft((current) => ({
        ...current,
        model: applyPlanDerivedModelFacts(current.model, nextPlan),
      }));
      setSelectedCandidate(nextPlan.recommended ?? nextPlan.candidates[0] ?? null);
      setBundle(null);
      setReport(null);
      setJob(null);
      setDemoMode(false);
      setConnection("connected");
      setNotice("Feasibility comparison complete.");
      setOutputDir((current) => current || `./aptus-output/${outputSlug(draft.project_name)}`);
      setStage("compare");
      if (nextPlan.project_id) {
        try {
          await refreshProjectSurface(nextPlan.project_id);
        } catch {
          setNotice("Feasibility comparison complete. Project history will refresh the next time Aptus opens.");
        }
      }
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      finishAction();
    }
  };

  const handleCompile = async () => {
    if (!isBoundTrainingPlan(plan)) return;
    if (demoMode) {
      setError("Clear the labeled example before compiling artifacts.");
      return;
    }
    if (blockMutationDuringRun("Compiling another bundle")) return;
    setStage("compile");
    beginAction("compile");
    try {
      if (!outputDir.trim()) throw new Error("Choose a bundle output directory before compilation.");
      const nextBundle = await api.compileBundle(plan, outputDir.trim());
      bundleRef.current = nextBundle;
      setBundle(nextBundle);
      setReport(nextBundle.report ?? null);
      setJob(null);
      setDemoMode(false);
      setConnection("connected");
      setNotice(`Bundle compiled with ${nextBundle.files.length} artifacts.`);
      const projectId = nextBundle.project_id ?? currentProject?.project_id;
      if (projectId) {
        try { await refreshProjectSurface(projectId); } catch { /* Primary compile result remains valid. */ }
      }
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      finishAction();
    }
  };

  const handleValidate = async () => {
    if (!bundle) return;
    if (demoMode) {
      setError("Clear the labeled example before validating an artifact.");
      return;
    }
    if (blockMutationDuringRun("Revalidating the active artifact")) return;
    setStage("validate");
    beginAction("validate");
    try {
      if (!bundle.project_id || !bundle.project_revision_id) {
        throw new Error(
          "Validation requires the bundle's exact project and project revision identity.",
        );
      }
      const nextReport = await api.validate(
        bundle.bundle_dir,
        validationLevel,
        false,
        bundle.project_id,
        bundle.project_revision_id,
      );
      const nextBundle = {
        ...bundle,
        report: nextReport,
        project_id: nextReport.project_id,
        project_revision_id: nextReport.project_revision_id,
      };
      bundleRef.current = nextBundle;
      setBundle(nextBundle);
      setReport(nextReport);
      setDemoMode(false);
      setConnection("connected");
      if (nextReport.state === "invalid") {
        setError("Validation failed. Review the findings before continuing.");
      } else {
        setNotice(`Validation finished with state ${nextReport.state ?? "unknown"}.`);
      }
      const projectId = nextReport.project_id ?? currentProject?.project_id;
      if (projectId) {
        try { await refreshProjectSurface(projectId); } catch { /* Primary validation result remains valid. */ }
      }
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      finishAction();
    }
  };

  const handleCreateJob = async (
    mode: "dependency" | "model-data" | "preflight" | "pilot" | "train",
  ) => {
    if (!bundle) return;
    if (blockMutationDuringRun("Starting another job")) return;
    beginAction("job");
    try {
      if (!bundle.project_id || !bundle.project_revision_id) {
        throw new Error(
          "Job creation requires the bundle's exact project and project revision identity.",
        );
      }
      const nextJob = await api.createJob({
        bundle_dir: bundle.bundle_dir,
        project_id: bundle.project_id,
        expected_project_revision_id: bundle.project_revision_id,
        action: mode,
        confirm_full_train: mode === "train",
      });
      applyJobUpdate(nextJob);
      setDemoMode(false);
      setConnection("connected");
      setNotice(`Job ${nextJob.id} entered state ${nextJob.state}.`);
      const projectId = nextJob.project_id ?? currentProject?.project_id;
      if (projectId) {
        try { await refreshProjectSurface(projectId); } catch { /* Primary job result remains valid. */ }
      }
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      finishAction();
    }
  };

  const handleCancelJob = async () => {
    if (!job || demoMode) return;
    beginAction("cancel-job");
    try {
      applyJobUpdate(await api.cancelJob(job.id));
      setConnection("connected");
      setNotice(`Cancellation requested for ${job.id}.`);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      finishAction();
    }
  };

  const handleRefreshJob = async () => {
    if (!job || demoMode) return;
    beginAction("refresh-job");
    try {
      applyJobUpdate(await api.getJob(job.id));
      setConnection("connected");
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      finishAction();
    }
  };

  const handleHardwareScan = async () => {
    if (demoMode) {
      setError("Clear the labeled example before scanning hardware.");
      return;
    }
    if (blockMutationDuringRun("Scanning and replacing hardware facts")) return;
    beginAction("hardware");
    try {
      const probe = await api.hardware();
      const measuredHardware = summarizeHardwareProbe(probe, draft.hardware);
      const backend = measuredHardware.devices[0]?.backend ?? "unknown";
      updateDraft((current) => ({
        ...current,
        hardware: summarizeHardwareProbe(probe, current.hardware),
        target: {
          ...current.target,
          runtime:
            backend === "mps"
              ? "mlx-lm"
              : current.target.runtime === "mlx-lm" || current.target.runtime === "pytorch-mps"
                ? "transformers-peft-cuda"
                : current.target.runtime,
        },
      }));
      setHardwareScanned(true);
      setConnection("connected");
      setNotice(
        backend === "mps"
          ? "Apple Silicon was measured as one shared memory system. Aptus will compare MLX-LM LoRA and QLoRA candidates conservatively. Measured preflight remains a bounded smoke. A passing uninterrupted pilot authorizes an explicitly confirmed full-duration run from scratch; resume is not supported."
          : "Hardware measured on this Aptus host. Single-device rows bind the strongest method-compatible GPU. Distributed rows use limiting VRAM and capabilities shared by every scanned GPU.",
      );
    } catch (caught) {
      setError(`${errorMessage(caught)} Enter a manual hardware profile instead.`);
    } finally {
      finishAction();
    }
  };

  const handleRecoverProject = async (projectId: string, revisionId: string) => {
    if (blockMutationDuringRun("Recovering a project revision")) return;
    if (demoMode) {
      setError("Clear the labeled example before recovering a real project revision.");
      return;
    }
    beginAction("recover-project");
    try {
      const recovered = await api.recoverProjectRevision(projectId, revisionId);
      if (recovered.training_authorization_current !== false) {
        throw new Error("Aptus rejected the recovery because its authorization boundary was unclear.");
      }
      const bootstrap = await api.bootstrap();
      setProjects(bootstrap.projects ?? []);
      setCurrentProject(bootstrap.project ?? null);
      setProjectHistory(bootstrap.project_history ?? []);
      setConnection("connected");
      setDemoMode(false);
      setModelInspection(null);
      setPlan(bootstrap.plan ?? null);
      setInspectionReceipt(inspectionReceiptFromPlan(bootstrap.plan ?? null));
      setSelectedCandidate(
        bootstrap.plan?.recommended ?? bootstrap.plan?.candidates[0] ?? null,
      );
      bundleRef.current = bootstrap.bundle ?? null;
      setBundle(bootstrap.bundle ?? null);
      setReport(bootstrap.bundle?.report ?? null);
      setJob(bootstrap.job ?? null);
      if (bootstrap.plan) {
        draftVersionRef.current += 1;
        setDraft(restoredDraft(
          bootstrap.plan,
          bootstrap.bundle ?? null,
          bootstrap.project?.name,
        ));
        setProfile(restoredProfile(bootstrap.plan));
        setHardwareScanned(restoredHardwareWasMeasured(bootstrap.plan));
      }
      setStage(
        bootstrap.job && ACTIVE_JOB_STATES.has(bootstrap.job.state)
          ? "run"
          : bootstrap.bundle
            ? "validate"
            : bootstrap.plan
              ? "compare"
              : "facts",
      );
      setNotice(
        `Recovered revision ${recovered.revision.ordinal} as a new immutable revision. Training authorization was not restored. Revalidate current evidence and confirm training again.`,
      );
    } catch (caught) {
      setError(errorMessage(caught));
      throw caught;
    } finally {
      finishAction();
    }
  };

  const loadExample = () => {
    if (blockMutationDuringRun("Loading example data")) return;
    if (busy !== null) {
      setError(`Aptus is still completing ${busy}. Wait before loading the example.`);
      return;
    }
    draftVersionRef.current += 1;
    setDraft(structuredClone(EXAMPLE_DRAFT));
    setProfile(structuredClone(EXAMPLE_PROFILE));
    setPlan(structuredClone(EXAMPLE_PLAN));
    setSelectedCandidate(structuredClone(EXAMPLE_PLAN.recommended));
    setBundle(structuredClone(EXAMPLE_BUNDLE));
    setReport(structuredClone(EXAMPLE_REPORT));
    setJob(structuredClone(EXAMPLE_JOB));
    setDemoMode(true);
    setHardwareScanned(false);
    setModelInspection(null);
    setInspectionReceipt(null);
    setOutputDir("./aptus-output/example-support-adapter");
    setError(null);
    setNotice("Labeled example loaded. No model inspection, compilation, validation, or execution ran.");
    setStage("facts");
  };

  const clearExample = () => {
    if (blockMutationDuringRun("Clearing the workspace")) return;
    if (busy !== null) {
      setError(`Aptus is still completing ${busy}. Wait before clearing the workspace.`);
      return;
    }
    draftVersionRef.current += 1;
    setDraft(freshDraft());
    setProfile(null);
    setPlan(null);
    setSelectedCandidate(null);
    setBundle(null);
    setReport(null);
    setJob(null);
    setDemoMode(false);
    setHardwareScanned(false);
    setModelInspection(null);
    setInspectionReceipt(null);
    setOutputDir("");
    setError(null);
    setNotice("Example cleared. Enter facts for a real plan.");
    setStage("facts");
  };

  const handleNewProject = () => {
    if (blockMutationDuringRun("Starting a new project")) return;
    if (busy !== null || !bootstrapReady) return;
    draftVersionRef.current += 1;
    setCurrentProject(null);
    setProjectHistory([]);
    setDraft((current) => ({ ...current, project_name: "" }));
    setProfile(null);
    setPlan(null);
    setSelectedCandidate(null);
    bundleRef.current = null;
    setBundle(null);
    setReport(null);
    setJob(null);
    setDemoMode(false);
    setHardwareScanned(false);
    setModelInspection(null);
    setOutputDir("");
    setError(null);
    setNotice("New project started. Review or edit the facts before creating its first plan.");
    setStage("facts");
  };

  return (
    <div className={`app-shell${embeddedInDesktop ? " is-desktop-host" : ""}`}>
      <WorkflowRail
        current={stage}
        completed={completed}
        projectName={draft.project_name}
        connection={connection}
        serviceVersion={serviceVersion}
        runState={job?.phase ?? job?.state}
        projects={projects}
        currentProject={currentProject}
        projectHistory={projectHistory}
        projectActionsDisabled={busy !== null || activeJob || demoMode}
        onRecoverProject={handleRecoverProject}
        onSelect={selectStage}
      />

      <main
        id="main-content"
        className="workbench"
        aria-busy={busy !== null}
        data-aptus-workbench-ready={bootstrapReady ? DESKTOP_WORKBENCH_READY_MARKER : undefined}
      >
        <MobileStageBar current={stage} completed={completed} runState={job?.phase ?? job?.state} onSelect={selectStage} />

        <div className="compact-project-history">
          <ProjectHistory
            projects={projects}
            currentProject={currentProject}
            currentHistory={projectHistory}
            disabled={busy !== null || activeJob || demoMode}
            onRecover={handleRecoverProject}
          />
        </div>

        <div className="workbench-topline">
          <span>{currentStageLabel}</span>
          <span>{draft.project_name || "Untitled plan"}</span>
          {demoMode ? <strong>Example workspace</strong> : null}
          <button
            type="button"
            className="button button-quiet new-project-action"
            disabled={!bootstrapReady || busy !== null || activeJob}
            onClick={handleNewProject}
          >
            New Project
          </button>
        </div>

        {demoMode ? (
          <div className="example-banner" role="note">
            <strong>Example workspace</strong>
            <span>Every displayed result is labeled example data. No inspection, planning, compilation, validation, or training ran.</span>
          </div>
        ) : null}

        {connection === "unavailable" && !demoMode ? (
          <div className="connection-banner" role="status">
            <div>
              <strong>The local planner API is unavailable.</strong>
              <span>Start the Aptus API to profile real inputs, or inspect the clearly labeled example workspace.</span>
            </div>
            <button type="button" className="button button-secondary" onClick={loadExample}>Load labeled example</button>
          </div>
        ) : null}

        <div className="message-region" aria-live="polite" aria-atomic="true">
          {error ? (
            <div className="error-banner" role="alert">
              <div><strong>Action blocked</strong><span>{error}</span></div>
              <button type="button" className="dismiss-button" onClick={() => setError(null)} aria-label="Dismiss error">×</button>
            </div>
          ) : notice ? (
            <div className="notice-banner" role="status">
              <span className="notice-check" aria-hidden="true">✓</span>
              <span>{notice}</span>
              <button type="button" className="dismiss-button" onClick={() => setNotice(null)} aria-label="Dismiss message">×</button>
            </div>
          ) : null}
        </div>

        {stage !== "facts" ? (
          <details className="inline-fit-ledger">
            <summary>Inspect per-device memory fit</summary>
            <FitLedger candidate={selected} example={demoMode} compact />
          </details>
        ) : null}

        <div className="stage-content">
          {stage === "facts" ? (
            <FactsStage
              draft={draft}
              setDraft={updateDraft}
              profile={profile}
              busy={busy}
              demoMode={demoMode}
              onLoadExample={loadExample}
              onClearExample={clearExample}
              onProfile={handleProfile}
              onModelInspect={handleModelInspect}
              onInvalidateModelInspection={() => {
                setModelInspection(null);
                setInspectionReceipt(null);
              }}
              onPlan={handlePlan}
              onHardwareScan={handleHardwareScan}
              hardwareScanned={hardwareScanned}
              modelInspection={modelInspection}
              modelPolicyPresentation={modelPolicyPresentation}
              methodCatalog={methodCatalog}
            />
          ) : null}
          {stage === "compare" ? (
            <CompareStage
              plan={plan}
              selected={selected}
              busy={busy}
              demoMode={demoMode}
              modelPolicyPresentation={modelPolicyPresentation}
              onInspectCandidate={setSelectedCandidate}
              onCompile={handleCompile}
              onReturnToFacts={() => setStage("facts")}
            />
          ) : null}
          {stage === "compile" ? (
            <CompileStage
              plan={isBoundTrainingPlan(plan) ? plan : null}
              bundle={bundle}
              busy={busy}
              demoMode={demoMode}
              onCompile={handleCompile}
              onValidate={handleValidate}
              onReturnToCompare={() => setStage("compare")}
              outputDir={outputDir}
              onOutputDirChange={setOutputDir}
            />
          ) : null}
          {stage === "validate" ? (
            <ValidateStage
              bundle={bundle}
              report={report}
              reportBinding={validationBinding}
              busy={busy}
              demoMode={demoMode}
              onValidate={handleValidate}
              onOpenRun={() => setStage("run")}
              onReturnToCompile={() => setStage("compile")}
              validationLevel={validationLevel}
              onValidationLevelChange={setValidationLevel}
            />
          ) : null}
          {stage === "run" ? (
            <RunStage
              bundle={bundle}
              report={report}
              reportBinding={validationBinding}
              job={job}
              busy={busy}
              demoMode={demoMode}
              onCreateJob={handleCreateJob}
              onRefreshJob={handleRefreshJob}
              onCancelJob={handleCancelJob}
              onReturnToValidate={() => setStage("validate")}
            />
          ) : null}
        </div>
      </main>

      <aside className="fit-inspector" aria-label="Inspected candidate memory evidence">
        <FitLedger candidate={selected} example={demoMode} />
      </aside>
    </div>
  );
}
