import { useState, type Dispatch, type FormEvent, type SetStateAction } from "react";
import type {
  FactDraft,
  InputProfile,
  MethodDescriptor,
  ModelInspectionResponse,
} from "../types";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { ExpertTopologyRail } from "../components/ExpertTopologyRail";
import { ModelPolicyPanel } from "../components/ModelPolicyPanel";
import { StageHeader } from "../components/StageHeader";
import { getDesktopBridge } from "../desktopBridge";
import type { ModelPolicyPresentation } from "../lib/modelPolicy";

interface FactsStageProps {
  draft: FactDraft;
  setDraft: Dispatch<SetStateAction<FactDraft>>;
  profile: InputProfile | null;
  busy: string | null;
  demoMode: boolean;
  onLoadExample: () => void;
  onClearExample: () => void;
  onProfile: () => Promise<void>;
  onModelInspect: () => Promise<void>;
  onInvalidateModelInspection: () => void;
  onPlan: () => Promise<void>;
  onHardwareScan: () => Promise<void>;
  hardwareScanned: boolean;
  modelInspection: ModelInspectionResponse | null;
  modelPolicyPresentation: ModelPolicyPresentation | null;
  methodCatalog: MethodDescriptor[];
}

function numberValue(value: string): number | null {
  return value === "" ? null : Number(value);
}

export function FactsStage({
  draft,
  setDraft,
  profile,
  busy,
  demoMode,
  onLoadExample,
  onClearExample,
  onProfile,
  onModelInspect,
  onInvalidateModelInspection,
  onPlan,
  onHardwareScan,
  hardwareScanned,
  modelInspection,
  modelPolicyPresentation,
  methodCatalog,
}: FactsStageProps) {
  const desktopBridge = getDesktopBridge();
  const [datasetPickerError, setDatasetPickerError] = useState<string | null>(null);

  const updateModel = <K extends keyof FactDraft["model"]>(
    key: K,
    value: FactDraft["model"][K],
  ) => {
    const preservesInspection = key === "parameters_b" || key === "training_allowed";
    if (!preservesInspection) {
      onInvalidateModelInspection();
    }
    setDraft((current) => ({
      ...current,
      model: preservesInspection
        ? {
            ...current.model,
            [key]: value,
            active_parameters_b: null,
            sparse_layer_count: null,
          }
        : {
            ...current.model,
            [key]: value,
            model_type: null,
            architecture: null,
            quantization_bits: null,
            quantization_layout: null,
            moe: null,
            active_parameters_b: null,
            sparse_layer_count: null,
          },
    }));
  };

  const updateDataset = <K extends keyof FactDraft["dataset"]>(
    key: K,
    value: FactDraft["dataset"][K],
  ) => setDraft((current) => ({
    ...current,
    dataset: { ...current.dataset, [key]: value },
  }));

  const updateHardware = <K extends keyof FactDraft["hardware"]>(
    key: K,
    value: FactDraft["hardware"][K],
  ) => setDraft((current) => ({
    ...current,
    hardware: { ...current.hardware, [key]: value },
  }));

  const updateDevice = <K extends keyof FactDraft["hardware"]["devices"][number]>(
    key: K,
    value: FactDraft["hardware"]["devices"][number][K],
  ) => setDraft((current) => ({
    ...current,
    hardware: {
      ...current.hardware,
      devices: [{ ...current.hardware.devices[0], [key]: value }],
    },
  }));

  const updateBackend = (backend: string) => setDraft((current) => ({
    ...current,
    hardware: {
      ...current.hardware,
      reserve_per_device_gib:
        backend === "mps"
          ? Math.max(current.hardware.reserve_per_device_gib ?? 0, 8)
          : current.hardware.reserve_per_device_gib,
      devices: [{ ...current.hardware.devices[0], backend }],
    },
    target: {
      ...current.target,
      runtime: backend === "mps" ? "mlx-lm" : "transformers-peft-cuda",
    },
  }));

  const updateTarget = <K extends keyof FactDraft["target"]>(
    key: K,
    value: FactDraft["target"][K],
  ) => setDraft((current) => ({
    ...current,
    target: { ...current.target, [key]: value },
  }));

  const submitProfile = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void onProfile();
  };

  const chooseDataset = async () => {
    if (!desktopBridge) return;

    setDatasetPickerError(null);
    try {
      const path = await desktopBridge.pickDataset();
      if (path) updateDataset("source_path", path);
    } catch {
      setDatasetPickerError("Aptus could not open the data picker. Enter an absolute path instead.");
    }
  };

  const profileFacts = profile?.facts ?? [];
  const inspectedQuantizationLayout = modelInspection?.facts?.quantization_layout;
  const inspectedOverrideCount = inspectedQuantizationLayout?.module_overrides.length ?? 0;
  const inspectedModelFacts = modelInspection?.status === "ok" && modelInspection.facts
    ? [
        { key: "family", label: "Aptus catalog family", value: modelInspection.facts.family },
        { key: "model_type", label: "Provider model type", value: modelInspection.facts.model_type },
        {
          key: "architectures",
          label: "Provider architectures",
          value: Array.isArray(modelInspection.facts.architectures)
            ? modelInspection.facts.architectures.join(", ")
            : modelInspection.facts.architecture,
        },
        { key: "hidden_size", label: "Hidden size", value: modelInspection.facts.hidden_size },
        { key: "intermediate_size", label: "Intermediate size", value: modelInspection.facts.intermediate_size },
        { key: "layers", label: "Layers", value: modelInspection.facts.layers },
        { key: "context_length", label: "Context", value: modelInspection.facts.context_length },
        { key: "license_name", label: "License label", value: modelInspection.facts.license_name },
        {
          key: "quantization_layout",
          label: "Provider quantization layout",
          value: inspectedQuantizationLayout
            ? [
                `${inspectedQuantizationLayout.default_bits}-bit`,
                `group ${inspectedQuantizationLayout.default_group_size};`,
                `${inspectedOverrideCount} ${inspectedOverrideCount === 1 ? "override" : "overrides"}`,
              ].join(" ")
            : null,
        },
      ].filter((fact) => fact.value !== null && fact.value !== undefined)
    : [];
  const selectableMethods = methodCatalog.filter((method) => method.selectable);
  const selectedBackend = draft.hardware.devices[0]?.backend ?? "cuda";
  const unifiedMemory = selectedBackend === "mps";
  const backendSelectableMethods = selectableMethods.filter((method) =>
    method.supported_backends.includes(selectedBackend),
  );
  const preferenceOptions = selectableMethods.length
    ? selectableMethods.map((method) => ({
        value: method.method_id,
        label: `Prefer ${method.display_name} if feasible`,
      }))
    : [
        { value: "full", label: "Prefer full fine-tuning on an objective tie" },
        { value: "lora", label: "Prefer LoRA if feasible" },
        { value: "int8-lora", label: "Prefer 8-bit LoRA if feasible" },
        { value: "qlora", label: "Prefer QLoRA if feasible" },
      ];

  return (
    <>
      <StageHeader
        eyebrow="Stage 1 · Evidence intake"
        title="Explicit facts in. Plan-bound bundle out."
        lede="Name the model, data, hardware, and training target. Aptus profiles what it can measure and keeps every assumption visible."
        meta={
          <div className="header-actions">
            {demoMode ? (
              <button type="button" className="button button-quiet" onClick={onClearExample}>
                Clear example
              </button>
            ) : (
              <button type="button" className="button button-quiet" onClick={onLoadExample}>
                Load labeled example
              </button>
            )}
          </div>
        }
      />

      <form className="facts-form" onSubmit={submitProfile}>
        <div className="project-name-field">
          <label htmlFor="project-name">Plan name</label>
          <input
            id="project-name"
            type="text"
            required
            value={draft.project_name}
            onChange={(event) => setDraft((current) => ({ ...current, project_name: event.target.value }))}
            placeholder="Customer-support adapter"
          />
          <small>Aptus uses this name only to suggest the bundle directory.</small>
        </div>

        <div className="fact-grid">
          <fieldset className="fact-panel">
            <legend>
              <span><span className="fact-index">M</span> Model</span>
              <ProvenanceBadge kind={demoMode ? "example" : "user-supplied"} />
            </legend>
            <p className="fact-intro">Pin the exact artifact and state whether training is permitted.</p>
            <div className="field full-field">
              <label htmlFor="model-id">Model ID</label>
              <input id="model-id" required value={draft.model.model_id} onChange={(event) => updateModel("model_id", event.target.value)} placeholder="organization/model-name" />
            </div>
            <div className="field full-field">
              <label htmlFor="revision">Immutable revision</label>
              <input id="revision" required pattern="[a-fA-F0-9]{40,64}" value={draft.model.revision} onChange={(event) => updateModel("revision", event.target.value)} placeholder="Commit hash, tag, or branch to inspect" aria-describedby="revision-help" />
              <small id="revision-help">Inspect can resolve a branch or tag. Planning requires the returned 40–64 character commit hash.</small>
            </div>
            <button
              type="button"
              className="button button-secondary model-inspect-button"
              disabled={busy !== null || demoMode || !draft.model.model_id.trim() || !draft.model.revision.trim()}
              onClick={() => void onModelInspect()}
            >
              {busy === "inspect-model" ? "Inspecting pinned model…" : "Inspect and pin model"}
            </button>
            {modelInspection ? (
              <section className={`model-inspection-result inspection-${modelInspection.status}`} aria-labelledby="model-inspection-title">
                <div className="inspection-heading">
                  <div>
                    <p className="eyebrow">Provider inspection</p>
                    <h3 id="model-inspection-title">
                      {modelInspection.status === "ok" ? "Revision-bound facts applied" : "Provider facts were not applied"}
                    </h3>
                  </div>
                  <ProvenanceBadge kind={modelInspection.status === "ok" ? "provider-declared" : "unknown"} />
                </div>
                {modelInspection.resolved_revision ? (
                  <p className="inspection-revision">Resolved commit <code>{modelInspection.resolved_revision}</code></p>
                ) : null}
                {inspectedModelFacts.length ? (
                  <dl className="inspection-facts">
                    {inspectedModelFacts.map((fact) => {
                      const provenance = modelInspection.provenance?.[fact.key];
                      return (
                        <div key={fact.key}>
                          <dt>{fact.label}</dt>
                          <dd>{String(fact.value)}</dd>
                          <small>{provenance?.source ?? "Provider response"}</small>
                        </div>
                      );
                    })}
                  </dl>
                ) : null}
                {modelInspection.error ? <p className="job-error">{modelInspection.error}</p> : null}
                {modelInspection.warnings?.length ? (
                  <ul className="plain-list inspection-warnings" aria-label="Model inspection warnings">
                    {modelInspection.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}
                  </ul>
                ) : null}
                <p className="inspection-boundary">Total parameter count and training permission never come from this inspection. Enter and confirm both yourself.</p>
              </section>
            ) : null}
            {modelPolicyPresentation ? (
              <ModelPolicyPanel presentation={modelPolicyPresentation} />
            ) : null}
            {draft.model.moe ? (
              <ExpertTopologyRail
                topology={draft.model.moe}
                totalParametersB={draft.model.parameters_b}
                activeParametersB={draft.model.active_parameters_b}
                sparseLayerCount={draft.model.sparse_layer_count}
                quantizationBits={draft.model.quantization_bits}
              />
            ) : null}
            <div className="field-row">
              <div className="field">
                <label htmlFor="family">Architecture family</label>
                <input id="family" required value={draft.model.family} onChange={(event) => updateModel("family", event.target.value)} placeholder="llama" />
                <small>Catalog families: llama, mistral, gemma, gemma4, qwen, and exact inspected qwen3_moe and gemma4_moe checkpoints. A named family is not runtime support.</small>
              </div>
              <div className="field">
                <label htmlFor="parameters">Total resident parameters</label>
                <div className="unit-input"><input id="parameters" type="number" required min="0.01" step="0.01" value={draft.model.parameters_b ?? ""} onChange={(event) => updateModel("parameters_b", numberValue(event.target.value))} /><span>B</span></div>
                <small>MoE checkpoints keep all weights resident. Active parameters do not replace this total.</small>
              </div>
            </div>
            <div className="field-row three-fields">
              <div className="field">
                <label htmlFor="hidden-size">Hidden size</label>
                <input id="hidden-size" type="number" required min="1" value={draft.model.hidden_size ?? ""} onChange={(event) => updateModel("hidden_size", numberValue(event.target.value))} />
              </div>
              <div className="field">
                <label htmlFor="layers">Layers</label>
                <input id="layers" type="number" required min="1" value={draft.model.layers ?? ""} onChange={(event) => updateModel("layers", numberValue(event.target.value))} />
              </div>
              <div className="field">
                <label htmlFor="context-length">Context</label>
                <input id="context-length" type="number" required min="1" value={draft.model.context_length ?? ""} onChange={(event) => updateModel("context_length", numberValue(event.target.value))} />
              </div>
            </div>
            <div className="field full-field">
              <label htmlFor="intermediate-size">Intermediate size</label>
              <input id="intermediate-size" type="number" min="1" value={draft.model.intermediate_size ?? ""} onChange={(event) => updateModel("intermediate_size", numberValue(event.target.value))} placeholder="Required for MLP adapter targets" />
            </div>
            <div className="field full-field">
              <label htmlFor="license-name">License</label>
              <input id="license-name" required value={draft.model.license_name} onChange={(event) => updateModel("license_name", event.target.value)} placeholder="License name and version" />
            </div>
            <label className="check-row">
              <input type="checkbox" checked={draft.model.training_allowed} onChange={(event) => updateModel("training_allowed", event.target.checked)} />
              <span>
                <strong>I confirmed this model permits the intended training.</strong>
                <small>Aptus blocks planning without explicit confirmation.</small>
              </span>
            </label>
          </fieldset>

          <fieldset className="fact-panel">
            <legend>
              <span><span className="fact-index">D</span> Dataset</span>
              <ProvenanceBadge kind={demoMode ? "example" : profile ? "measured" : "unknown"} />
            </legend>
            <p className="fact-intro">Point Aptus to real local data so it can inspect shape and length.</p>
            <div className="field full-field">
              <label htmlFor="dataset-path">Dataset path</label>
              <div className="native-path-control">
                <input id="dataset-path" required value={draft.dataset.source_path} onChange={(event) => updateDataset("source_path", event.target.value)} placeholder="/absolute/path/training.jsonl" aria-describedby="dataset-path-help" />
                {desktopBridge ? (
                  <button type="button" className="button button-secondary native-path-button" disabled={demoMode} onClick={() => void chooseDataset()}>
                    Choose file
                  </button>
                ) : null}
              </div>
              <small id="dataset-path-help">
                {desktopBridge
                  ? "Choose a local JSONL, JSON, CSV, or text file, or enter its absolute path."
                  : "The local API reads this path. A browser upload is not implied."}
              </small>
              {datasetPickerError ? <small className="native-action-error" role="alert">{datasetPickerError}</small> : null}
            </div>
            <p className="fact-boundary">
              Aptus detects JSONL, JSON, CSV, or text and validates each row's
              schema. The model-data gate resolves the pinned model tokenizer.
            </p>
            <div className="field full-field">
              <label htmlFor="sample-limit">Length-stat sample size</label>
              <input id="sample-limit" type="number" min="1" value={draft.dataset.sample_limit ?? ""} onChange={(event) => updateDataset("sample_limit", numberValue(event.target.value))} placeholder="Blank uses the 512-row default" />
              <small>Every row is schema checked. This deterministic sample bounds only the profiling length statistics.</small>
            </div>
          </fieldset>

          <fieldset className="fact-panel">
            <legend>
              <span><span className="fact-index">H</span> Hardware</span>
              <ProvenanceBadge kind={demoMode ? "example" : hardwareScanned ? "measured" : "user-supplied"} />
            </legend>
            <p className="fact-intro">Measure the intended training host, then let each runtime apply its own memory and capability rules.</p>
            <div className="field full-field">
              <label htmlFor="hardware-discovery">Source</label>
              <select id="hardware-discovery" value={draft.hardware.discovery} onChange={(event) => updateHardware("discovery", event.target.value as FactDraft["hardware"]["discovery"])}>
                <option value="manual">Enter a hardware profile</option>
                <option value="local-scan">Ask the local Aptus runner to scan</option>
              </select>
              {draft.hardware.discovery === "local-scan" ? <small>The API host will be re-scanned during planning. Use local scan only when it is the intended training host.</small> : null}
            </div>
        <button type="button" className="button button-secondary hardware-scan-button" disabled={busy !== null || demoMode} onClick={() => void onHardwareScan()}>
              {busy === "hardware" ? "Scanning this Aptus host…" : "Scan this Aptus host"}
            </button>
            {hardwareScanned ? <p className="example-inline measured-inline">Measured on this Aptus host. Confirm this is the intended execution machine.</p> : null}
            {unifiedMemory ? (
              <p className="fact-boundary">
                Apple Silicon shares one memory pool between the CPU and GPU. This is not dedicated VRAM. Aptus evaluates MLX-LM LoRA and QLoRA with a separate estimator, then runs a bounded measured preflight. A passing uninterrupted pilot authorizes an explicitly confirmed full-duration run from scratch. Resume is not supported.
              </p>
            ) : null}
            <div className="field-row">
              <div className="field">
                <label htmlFor="backend">Backend</label>
                <select id="backend" value={draft.hardware.devices[0]?.backend ?? "cuda"} onChange={(event) => updateBackend(event.target.value)}>
                  <option value="cuda">NVIDIA CUDA</option>
                  <option value="mps">Apple Metal</option>
                  <option value="rocm" disabled>ROCm · not supported in v0.2</option>
                  <option value="cpu" disabled>CPU · not supported in v0.2</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="device-vram">{unifiedMemory ? "Unified memory pool" : "VRAM per device"}</label>
                <div className="unit-input"><input id="device-vram" type="number" required min="0.1" step="0.1" value={draft.hardware.devices[0]?.total_vram_gib ?? ""} onChange={(event) => updateDevice("total_vram_gib", numberValue(event.target.value))} /><span>GiB</span></div>
              </div>
            </div>
            <div className="field-row">
              <div className="field">
                <label htmlFor="gpu-count">Accelerator count</label>
                <input id="gpu-count" type="number" required min="1" value={draft.hardware.gpu_count} onChange={(event) => updateHardware("gpu_count", Number(event.target.value))} />
              </div>
              <div className="field">
                <label htmlFor="disk-free">Free disk</label>
                <div className="unit-input"><input id="disk-free" type="number" min="0.1" step="0.1" value={draft.hardware.disk_free_gib ?? ""} onChange={(event) => updateHardware("disk_free_gib", numberValue(event.target.value))} /><span>GiB</span></div>
              </div>
            </div>
            <div className="field-row">
              <div className="field">
                <label htmlFor="device-free-vram">{unifiedMemory ? "Measured memory headroom" : "Free VRAM now"}</label>
                <div className="unit-input"><input id="device-free-vram" type="number" min="0.1" step="0.1" value={draft.hardware.devices[0]?.free_vram_gib ?? ""} onChange={(event) => updateDevice("free_vram_gib", numberValue(event.target.value))} placeholder="Optional" /><span>GiB</span></div>
              </div>
              <div className="field">
                <label htmlFor="host-ram-free">Free host RAM now</label>
                <div className="unit-input"><input id="host-ram-free" type="number" min="0.1" step="0.1" value={draft.hardware.host_ram_free_gib ?? ""} onChange={(event) => updateHardware("host_ram_free_gib", numberValue(event.target.value))} placeholder="Optional" /><span>GiB</span></div>
              </div>
            </div>
            <div className="field-row">
              <div className="field">
                <label htmlFor="host-ram">Host RAM</label>
                <div className="unit-input"><input id="host-ram" type="number" required min="1" step="1" value={draft.hardware.host_ram_gib ?? ""} onChange={(event) => updateHardware("host_ram_gib", numberValue(event.target.value))} /><span>GiB</span></div>
              </div>
              <div className="field">
                <label htmlFor="device-reserve">Reserve per device</label>
                <div className="unit-input"><input id="device-reserve" type="number" required min="0" step="0.5" value={draft.hardware.reserve_per_device_gib ?? ""} onChange={(event) => updateHardware("reserve_per_device_gib", numberValue(event.target.value))} /><span>GiB</span></div>
              </div>
            </div>
            {unifiedMemory ? (
              <div className="mps-capability-boundary" role="note" aria-label="Apple Silicon capability boundary">
                <strong>Apple capability rules are runtime-specific.</strong>
                <span>CUDA BF16 and bitsandbytes flags do not apply to MLX. Unknown live headroom stays unknown, and every proposed MLX-LM run remains pilot-required.</span>
              </div>
            ) : (
            <div className="capability-checks" role="group" aria-label="Device capabilities">
              <label className="check-row compact-check">
                <input type="checkbox" checked={draft.hardware.devices[0]?.supports_bf16 ?? false} onChange={(event) => updateDevice("supports_bf16", event.target.checked)} />
                <span><strong>BF16 supported</strong></span>
              </label>
              <label className="check-row compact-check">
                <input type="checkbox" checked={draft.hardware.devices[0]?.supports_8bit ?? false} onChange={(event) => updateDevice("supports_8bit", event.target.checked)} />
                <span><strong>8-bit backend supported</strong></span>
              </label>
              <label className="check-row compact-check">
                <input type="checkbox" checked={draft.hardware.devices[0]?.supports_4bit ?? false} onChange={(event) => updateDevice("supports_4bit", event.target.checked)} />
                <span><strong>4-bit backend supported</strong></span>
              </label>
            </div>
            )}
          </fieldset>

          <fieldset className="fact-panel">
            <legend>
              <span><span className="fact-index">T</span> Target</span>
              <ProvenanceBadge kind={demoMode ? "example" : "declared"} />
            </legend>
            <p className="fact-intro">State the outcome. Method choice remains subordinate to feasibility.</p>
            <div className="field full-field">
              <label htmlFor="task">Training task</label>
              <select id="task" value={draft.target.task} onChange={(event) => updateTarget("task", event.target.value)}>
                <option value="sft">Supervised fine-tuning</option>
              </select>
            </div>
            <fieldset className="choice-fieldset">
              <legend>Primary objective</legend>
              <div className="segmented-control">
                {([
                  ["quality", "Prefer higher-fidelity methods (full → LoRA → …)"],
                  ["memory", "Prefer lower memory"],
                  ["speed", "Prefer fewer steps / faster setups"],
                ] as const).map(([objective, label]) => (
                  <label key={objective}>
                    <input type="radio" name="objective" value={objective} checked={draft.target.objective === objective} onChange={() => updateTarget("objective", objective)} />
                    <span>{label}</span>
                  </label>
                ))}
              </div>
            </fieldset>
            <div className="field-row three-fields">
              <div className="field">
                <label htmlFor="sequence-length">Sequence</label>
                <input id="sequence-length" type="number" required min="1" value={draft.target.sequence_length ?? ""} onChange={(event) => updateTarget("sequence_length", numberValue(event.target.value))} />
              </div>
              <div className="field">
                <label htmlFor="effective-batch">Effective batch</label>
                <input id="effective-batch" type="number" required min="1" value={draft.target.effective_batch_size ?? ""} onChange={(event) => updateTarget("effective_batch_size", numberValue(event.target.value))} />
              </div>
              <div className="field">
                <label htmlFor="epochs">Max epochs</label>
                <input id="epochs" type="number" required min="1" value={draft.target.max_epochs ?? ""} onChange={(event) => updateTarget("max_epochs", numberValue(event.target.value))} />
              </div>
            </div>
            <div className="field full-field">
              <label htmlFor="method-preference">Tie-break method preference</label>
              <select id="method-preference" value={draft.target.method_preference} onChange={(event) => updateTarget("method_preference", event.target.value)}>
                <option value="">No preference. Compare feasible methods.</option>
                {preferenceOptions.map((method) => (
                  <option key={method.value} value={method.value}>{method.label}</option>
                ))}
              </select>
              <small>The primary objective ranks first. A preference never reverses it or overrides a failed gate.</small>
            </div>
            {methodCatalog.length ? (
              <details className="method-readiness-board">
                <summary>
                  <span>Inspect method readiness</span>
                  <small>
                    {selectableMethods.length} compiler paths · {backendSelectableMethods.length} available on {selectedBackend.toUpperCase()} · {methodCatalog.length - selectableMethods.length} held behind research gates
                  </small>
                </summary>
                <div
                  className="method-readiness-list"
                  role="region"
                  aria-label="Fine-tuning method readiness"
                  tabIndex={0}
                >
                  {methodCatalog.map((method) => {
                    const availableOnBackend = method.selectable
                      && method.supported_backends.includes(selectedBackend);
                    return (
                    <article key={method.method_id} className={`method-readiness-row lifecycle-${method.lifecycle}`}>
                      <header>
                        <div>
                          <strong>{method.display_name}</strong>
                          <code>{method.method_id}</code>
                        </div>
                        <span>
                          {method.lifecycle === "gated-executable"
                            ? availableOnBackend
                              ? "Executable behind gates"
                              : `Unavailable on ${selectedBackend.toUpperCase()}`
                            : method.lifecycle === "experimental"
                              ? "Experimental"
                              : "Research only"}
                        </span>
                      </header>
                      <p>{method.summary}</p>
                      <dl>
                        <div><dt>Trainable object</dt><dd>{method.parameter_scope}</dd></div>
                        <div><dt>Base storage</dt><dd>{method.base_storage}</dd></div>
                      </dl>
                      {method.blocker ? <p className="method-blocker"><strong>Blocked:</strong> {method.blocker}</p> : null}
                      {method.selectable && !availableOnBackend ? (
                        <p className="method-blocker"><strong>Backend gate:</strong> This compiler supports {method.supported_backends.join(", ") || "no released backend"}, not {selectedBackend}.</p>
                      ) : null}
                      <p className="method-next-gate"><strong>Required proof:</strong> {method.pilot_requirement}</p>
                    </article>
                    );
                  })}
                </div>
              </details>
            ) : null}
            <div className="field-row">
              <div className="field evidence-caution">
                <label htmlFor="evaluation-fraction">Evaluation fraction</label>
                <input id="evaluation-fraction" type="number" required min="0" max="0.99" step="0.01" value={draft.target.evaluation_fraction} onChange={(event) => updateTarget("evaluation_fraction", Number(event.target.value))} aria-describedby="evaluation-fraction-help" />
                <small id="evaluation-fraction-help">Train/validation split only. This is not a quality-evaluation contract and does not decide eval pass.</small>
              </div>
              <div className="field">
                <label htmlFor="checkpoint-steps">Checkpoint interval</label>
                <input id="checkpoint-steps" type="number" required min="1" value={draft.target.checkpoint_steps} onChange={(event) => updateTarget("checkpoint_steps", Number(event.target.value))} />
              </div>
            </div>
            <details className="method-readiness-board">
              <summary><span>Phase 3 execution controls</span><small>Identity-bound steps, seeds, and batch arithmetic</small></summary>
              <div className="field-row three-fields">
                <div className="field">
                  <label htmlFor="optimizer-steps">Optimizer steps</label>
                  <input id="optimizer-steps" type="number" min="1" value={draft.target.optimizer_steps ?? ""} onChange={(event) => updateTarget("optimizer_steps", numberValue(event.target.value))} placeholder="Epoch-controlled if blank" />
                </div>
                <div className="field">
                  <label htmlFor="split-seed">Split seed</label>
                  <input id="split-seed" type="number" min="0" value={draft.target.split_seed} onChange={(event) => updateTarget("split_seed", Number(event.target.value))} />
                </div>
                <div className="field">
                  <label htmlFor="training-seed">Training seed</label>
                  <input id="training-seed" type="number" min="0" value={draft.target.training_seed} onChange={(event) => {
                    const value = Number(event.target.value);
                    setDraft((current) => ({
                      ...current,
                      target: { ...current.target, training_seed: value, data_order_seed: 1_000_000 + value },
                    }));
                  }} />
                </div>
              </div>
              <div className="field-row three-fields">
                <div className="field">
                  <label htmlFor="data-order-seed">Data-order seed</label>
                  <input id="data-order-seed" type="number" min="0" value={draft.target.data_order_seed} onChange={(event) => updateTarget("data_order_seed", Number(event.target.value))} />
                </div>
                <div className="field">
                  <label htmlFor="micro-batch-size">Micro-batch</label>
                  <input id="micro-batch-size" type="number" min="1" value={draft.target.micro_batch_size ?? ""} onChange={(event) => updateTarget("micro_batch_size", numberValue(event.target.value))} placeholder="Planner-derived" />
                </div>
                <div className="field">
                  <label htmlFor="gradient-accumulation-steps">Accumulation</label>
                  <input id="gradient-accumulation-steps" type="number" min="1" value={draft.target.gradient_accumulation_steps ?? ""} onChange={(event) => updateTarget("gradient_accumulation_steps", numberValue(event.target.value))} placeholder="Planner-derived" />
                </div>
              </div>
              <p className="fact-boundary">Micro-batch and accumulation must be supplied together. Full CUDA training rejects unbound step or seed overrides.</p>
            </details>
            <label className="check-row">
              <input type="checkbox" checked={false} disabled />
              <span><strong>Sequence packing · not supported in v0.2</strong><small>The masking compiler rejects packing until its loss-boundary rules are implemented.</small></span>
            </label>
            <div className="field full-field">
              <label htmlFor="bundle-runtime">Bundle runtime</label>
              <select
                id="bundle-runtime"
                value={draft.target.runtime}
                onChange={(event) => updateTarget("runtime", event.target.value)}
              >
                {unifiedMemory ? (
                  <>
                    <option value="mlx-lm">MLX-LM · native Apple training</option>
                    <option value="pytorch-mps" disabled>
                      PyTorch MPS · known compatibility runtime, compiler unavailable
                    </option>
                  </>
                ) : (
                  <option value="transformers-peft-cuda">Transformers + PEFT · CUDA</option>
                )}
              </select>
              <small>
                {unifiedMemory
                  ? "MLX-LM is the primary Apple runtime. PyTorch MPS is detected separately but cannot be selected until Aptus ships a compiler binding."
                  : "CUDA bundles use the pinned Transformers, PEFT, and Accelerate compiler."}
              </small>
            </div>
          </fieldset>
        </div>

        {profile ? (
          <section className="profile-strip" aria-labelledby="profile-strip-title">
            <div>
              <p className="eyebrow">Profile available</p>
              <h2 id="profile-strip-title">Review the measured facts</h2>
            </div>
            {profileFacts.length ? (
              <dl>
                {profileFacts.slice(0, 4).map((fact, index) => (
                  <div key={`${fact.key ?? fact.label ?? "fact"}-${index}`}>
                    <dt>{fact.label ?? fact.key ?? "Profile fact"}</dt>
                    <dd>
                      <span>{String(fact.value ?? "Unknown")}{fact.unit ? ` ${fact.unit}` : ""}</span>
                      <ProvenanceBadge kind={fact.provenance} />
                    </dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p>The API returned a profile. Open Compare to inspect the resulting plan.</p>
            )}
          </section>
        ) : null}

        <div className="sticky-actions">
          <div>
            <strong>{profile ? "Dataset profiled" : "Profile the dataset before planning"}</strong>
            <span>{profile ? "Aptus can now compare supported strategies." : "Model and hardware facts remain explicit attestations."}</span>
          </div>
          <div className="action-buttons">
          <button type="submit" className="button button-secondary" disabled={busy !== null || demoMode}>
              {busy === "profile" ? "Profiling…" : "Profile dataset"}
            </button>
          <button type="button" className="button button-primary" disabled={!profile || busy !== null || demoMode} onClick={() => void onPlan()}>
              {busy === "plan" ? "Comparing…" : "Compare strategies"}
            </button>
          </div>
        </div>
      </form>
    </>
  );
}
