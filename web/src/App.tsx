import { useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { api, ApiError } from "./api";
import { FitLedger } from "./components/FitLedger";
import { summarizeHardwareProbe } from "./lib/hardware";
import { applyProviderModelInspection } from "./lib/modelInspection";
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
  ModelInspectionResponse,
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

function restoredDraft(plan: TrainingPlan, bundle: CompileResponse): FactDraft {
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
  const sourcePath = source && !source.startsWith("/")
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
    project_name: typeof plan.plan_id === "string" ? plan.plan_id : "Restored plan",
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
      runtime: "transformers-peft",
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
  const [stage, setStage] = useState<WorkflowStage>("facts");
  const [draft, setDraft] = useState<FactDraft>(freshDraft);
  const [profile, setProfile] = useState<InputProfile | null>(null);
  const [plan, setPlan] = useState<TrainingPlan | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<CandidatePlan | null>(null);
  const [bundle, setBundle] = useState<CompileResponse | null>(null);
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [serviceVersion, setServiceVersion] = useState<string | undefined>();
  const [demoMode, setDemoMode] = useState(false);
  const [hardwareScanned, setHardwareScanned] = useState(false);
  const [modelInspection, setModelInspection] = useState<ModelInspectionResponse | null>(null);
  const [methodCatalog, setMethodCatalog] = useState<MethodDescriptor[]>([]);
  const [outputDir, setOutputDir] = useState("");
  const [validationLevel, setValidationLevel] = useState<ValidateRequest["level"]>("static");
  const bundleRef = useRef<CompileResponse | null>(null);
  const draftVersionRef = useRef(0);

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
    if (nextJob.validation_report && reportMatchesBundle) {
      setReport(nextJob.validation_report);
      setBundle((current) =>
        current ? { ...current, report: nextJob.validation_report } : current,
      );
    }
    if (nextJob.validation_report_error && reportMatchesBundle) {
      setReport(null);
      setBundle((current) =>
        current ? { ...current, report: undefined } : current,
      );
      setError(nextJob.validation_report_error);
    }
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
                reserve_per_device_gib: defaults.reserve_gib ?? merged.hardware.reserve_per_device_gib,
              },
              target: {
                ...merged.target,
                task: defaults.task ?? merged.target.task,
                packing: defaults.packing ?? merged.target.packing,
              },
            };
          });
        }
        if (bootstrap.plan && restoreWorkspace) {
          setPlan(bootstrap.plan);
          setSelectedCandidate(bootstrap.plan.recommended ?? bootstrap.plan.candidates[0] ?? null);
        }
        if (bootstrap.bundle && restoreWorkspace) {
          bundleRef.current = bootstrap.bundle;
          setBundle(bootstrap.bundle);
          setReport(bootstrap.bundle.report ?? null);
        }
        if (bootstrap.plan && bootstrap.bundle && restoreWorkspace) {
          setDraft(restoredDraft(bootstrap.plan, bootstrap.bundle));
          setProfile(restoredProfile(bootstrap.plan));
          setHardwareScanned(restoredHardwareWasMeasured(bootstrap.plan));
          setNotice("Restored the latest validated local artifact and its bound facts.");
        }
        if (bootstrap.job) {
          applyJobUpdate(bootstrap.job);
          if (ACTIVE_JOB_STATES.has(bootstrap.job.state)) setStage("run");
        }
      })
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        setConnection("unavailable");
      });
    return () => controller.abort();
  }, []);

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
  const completed = useMemo(() => {
    const values = new Set<WorkflowStage>();
    if (profile) values.add("facts");
    if (plan) values.add("compare");
    if (bundle) values.add("compile");
    if (validationRank(activeReport?.state) >= validationRank("static-pass")) {
      values.add("validate");
    }
    if (
      job?.state === "completed"
      && (job.mode === "train" || job.action === "train")
      && bundle
      && job.bundle_dir === bundle.bundle_dir
    ) values.add("run");
    return values;
  }, [profile, plan, bundle, activeReport?.state, job]);

  const selected = selectedCandidate ?? plan?.recommended ?? plan?.candidates[0] ?? null;
  const currentStageLabel = WORKFLOW_STAGES.find((item) => item.id === stage)?.label ?? "Facts";
  const activeJob = isActiveJob(job);

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
    setDraft((current) =>
      typeof action === "function" ? action(current) : action,
    );
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
      setModelInspection(inspection);
      if (inspection.status !== "ok" || !inspection.facts || !inspection.resolved_revision) {
        throw new Error(inspection.error ?? "The provider did not return revision-bound model facts.");
      }
      if (draftVersionRef.current !== requestDraftVersion) {
        setNotice("Model facts changed during inspection. Aptus did not apply the older provider response.");
        return;
      }
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
      const nextPlan = await api.plan(draft);
      if (draftVersionRef.current !== requestDraftVersion) {
        setNotice("Facts changed while planning. Aptus discarded the older response; compare the current facts again.");
        return;
      }
      setPlan(nextPlan);
      setSelectedCandidate(nextPlan.recommended ?? nextPlan.candidates[0] ?? null);
      setBundle(null);
      setReport(null);
      setJob(null);
      setDemoMode(false);
      setConnection("connected");
      setNotice("Feasibility comparison complete.");
      setOutputDir((current) => current || `./aptus-output/${outputSlug(draft.project_name)}`);
      setStage("compare");
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      finishAction();
    }
  };

  const handleCompile = async () => {
    if (!plan) return;
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
      setBundle(nextBundle);
      setReport(nextBundle.report ?? null);
      setJob(null);
      setDemoMode(false);
      setConnection("connected");
      setNotice(`Bundle compiled with ${nextBundle.files.length} artifacts.`);
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
      const nextReport = await api.validate(bundle.bundle_dir, validationLevel, false);
      setReport(nextReport);
      setDemoMode(false);
      setConnection("connected");
      if (nextReport.state === "invalid") {
        setError("Validation failed. Review the findings before continuing.");
      } else {
        setNotice(`Validation finished with state ${nextReport.state ?? "unknown"}.`);
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
      const nextJob = await api.createJob({
        bundle_dir: bundle.bundle_dir,
        action: mode,
        confirm_full_train: mode === "train",
      });
      applyJobUpdate(nextJob);
      setDemoMode(false);
      setConnection("connected");
      setNotice(`Job ${nextJob.id} entered state ${nextJob.state}.`);
    } catch (caught) {
      if (mode === "train") {
        setReport((current) => current ? {
          ...current,
          authorization_current: false,
          authorization_error: "The latest training launch was rejected by the server's atomic pilot-binding and capacity admission.",
        } : current);
      }
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
      }));
      setHardwareScanned(true);
      setConnection("connected");
      setNotice(
        backend === "mps"
          ? "Apple Silicon and its shared unified-memory pool were measured. The current CUDA compiler will mark execution methods unsupported until Aptus has a separately validated MLX backend."
          : "Hardware measured on this Aptus host. Single-device rows bind the strongest method-compatible GPU. Distributed rows use limiting VRAM and capabilities shared by every scanned GPU.",
      );
    } catch (caught) {
      setError(`${errorMessage(caught)} Enter a manual hardware profile instead.`);
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
    setOutputDir("");
    setError(null);
    setNotice("Example cleared. Enter facts for a real plan.");
    setStage("facts");
  };

  return (
    <div className="app-shell">
      <WorkflowRail
        current={stage}
        completed={completed}
        projectName={draft.project_name}
        connection={connection}
        serviceVersion={serviceVersion}
        runState={job?.phase ?? job?.state}
        onSelect={setStage}
      />

      <main id="main-content" className="workbench" aria-busy={busy !== null}>
        <MobileStageBar current={stage} completed={completed} runState={job?.phase ?? job?.state} onSelect={setStage} />

        <div className="workbench-topline">
          <span>{currentStageLabel}</span>
          <span>{draft.project_name || "Untitled plan"}</span>
          {demoMode ? <strong>Example workspace</strong> : null}
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
              onInvalidateModelInspection={() => setModelInspection(null)}
              onPlan={handlePlan}
              onHardwareScan={handleHardwareScan}
              hardwareScanned={hardwareScanned}
              modelInspection={modelInspection}
              methodCatalog={methodCatalog}
            />
          ) : null}
          {stage === "compare" ? (
            <CompareStage
              plan={plan}
              selected={selected}
              busy={busy}
              demoMode={demoMode}
              onSelectCandidate={setSelectedCandidate}
              onCompile={handleCompile}
              onReturnToFacts={() => setStage("facts")}
            />
          ) : null}
          {stage === "compile" ? (
            <CompileStage
              plan={plan}
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
