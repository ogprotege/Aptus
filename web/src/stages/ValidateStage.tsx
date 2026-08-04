import type { CompileResponse, ValidateRequest, ValidationReport } from "../types";
import { EmptyStage } from "../components/EmptyStage";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { StageHeader } from "../components/StageHeader";
import { ValidationGates } from "../components/ValidationGates";
import {
  validationReportMatchesBinding,
} from "../lib/modelPolicy";
import type { ValidationReportBindingIdentity } from "../lib/modelPolicy";
import { canStartAction, formatBytes } from "../lib/plan";

interface ValidateStageProps {
  bundle: CompileResponse | null;
  report: ValidationReport | null;
  reportBinding: ValidationReportBindingIdentity | null;
  busy: string | null;
  demoMode: boolean;
  onValidate: () => Promise<void>;
  onOpenRun: () => void;
  onReturnToCompile: () => void;
  validationLevel: ValidateRequest["level"];
  onValidationLevelChange: (value: ValidateRequest["level"]) => void;
}

export function ValidateStage({
  bundle,
  report,
  reportBinding,
  busy,
  demoMode,
  onValidate,
  onOpenRun,
  onReturnToCompile,
  validationLevel,
  onValidationLevelChange,
}: ValidateStageProps) {
  if (!bundle) {
    return (
      <>
        <StageHeader eyebrow="Stage 4 · Validation" title="Prove the artifact before execution." lede="A validation claim needs a bundle to inspect." />
        <EmptyStage title="No bundle to validate" actionLabel="Open compiler" onAction={onReturnToCompile}>
          Compile the recommended plan into a versioned bundle first.
        </EmptyStage>
      </>
    );
  }

  const activeReport = report ?? bundle.report ?? null;
  const reportBound = validationReportMatchesBinding(activeReport, reportBinding);
  const runActionsReady = reportBound
    && canStartAction(activeReport?.state, "dependency");
  const bindings = Object.entries(activeReport?.bindings ?? {});
  const preflightMetrics = activeReport?.preflight_metrics;
  const pilotMetrics = activeReport?.pilot_metrics;
  const mlxRuntime = bundle.runtime_contract?.training_runtime === "mlx-lm"
    || pilotMetrics?.training_runtime === "mlx-lm";
  const mlxAdmission = pilotMetrics?.unified_memory_admission;
  const mlxTargetBinding = pilotMetrics?.trainable_target_binding;
  const mlxReload = pilotMetrics?.reload_evidence;
  const pilotRunId = pilotMetrics?.pilot_run_id ?? pilotMetrics?.run_id;
  const phaseOne = pilotMetrics?.phase_one;
  const phaseTwo = pilotMetrics?.phase_two_resumed;
  const checkpointContracts = [
    { label: "Phase 1 checkpoint", contract: pilotMetrics?.phase_one_checkpoint },
    { label: "Phase 2 checkpoint", contract: pilotMetrics?.phase_two_checkpoint },
  ];

  return (
    <>
      <StageHeader
        eyebrow="Stage 4 · Validation"
        title="Prove the artifact before execution."
        lede={mlxRuntime
          ? "Each gate establishes a specific claim. Static success does not imply model fit, and the bounded MLX preflight smoke does not authorize full-duration training."
          : "Each gate establishes a specific claim. Static success does not imply model fit, and a synthetic method step does not imply calibrated VRAM."}
        meta={demoMode ? <ProvenanceBadge kind="example" label="Example gates" /> : undefined}
      />

      {activeReport ? (
        <>
          {!reportBound ? (
            <section className="blocked-panel" role="status">
              <h2>This report does not belong to the compiled recommendation.</h2>
              <p>Revalidate the current bundle before opening Run. Plan, candidate, and model revision identities must all match.</p>
            </section>
          ) : null}
          <ValidationGates report={activeReport} />
          {activeReport.runtime_evidence?.length ? (
            <section className="runtime-evidence" aria-labelledby="runtime-evidence-title">
              <p className="eyebrow">Recorded observations</p>
              <h2 id="runtime-evidence-title">Runtime evidence</h2>
              <ul className="plain-list evidence-list">
                {activeReport.runtime_evidence.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
              </ul>
            </section>
          ) : null}
          {activeReport.artifact_fingerprint ? (
            <div className="fingerprint-block">
              <span>Artifact fingerprint</span>
              <code>{activeReport.artifact_fingerprint}</code>
            </div>
          ) : null}
          {activeReport.validation_level || activeReport.validator_version || bindings.length ? (
            <section className="attestation-panel" aria-labelledby="attestation-title">
              <p className="eyebrow">{reportBound ? "Bound validation evidence" : "Unbound validation evidence"}</p>
              <h2 id="attestation-title">{reportBound
                ? "Attestation bound to this artifact"
                : "Attestation from another artifact"}</h2>
              <dl className="attestation-summary">
                <div><dt>Level</dt><dd>{activeReport.validation_level ?? "Not recorded"}</dd></div>
                <div><dt>Validator</dt><dd>{activeReport.validator_version ?? "Not recorded"}</dd></div>
                <div><dt>Validated at</dt><dd>{activeReport.validated_at ?? "Not recorded"}</dd></div>
                {pilotMetrics?.checkpoint_continuation_observed !== undefined ? <div><dt>Checkpoint continuation observed</dt><dd>{pilotMetrics.checkpoint_continuation_observed ? "Yes" : "No"}</dd></div> : null}
                {pilotMetrics?.execution_semantics ? <div><dt>Execution</dt><dd>{pilotMetrics.execution_semantics === "uninterrupted" ? "Uninterrupted from scratch" : pilotMetrics.execution_semantics}</dd></div> : null}
                {pilotMetrics?.resume_supported !== undefined ? <div><dt>Resume</dt><dd>{pilotMetrics.resume_supported ? "Supported" : "Unavailable for this runtime"}</dd></div> : null}
                {pilotMetrics?.measured_checkpoint_bytes !== undefined ? <div><dt>Largest pilot checkpoint</dt><dd>{formatBytes(pilotMetrics.measured_checkpoint_bytes)}</dd></div> : null}
                {pilotMetrics?.measured_final_export_bytes !== undefined ? <div><dt>Largest pilot export</dt><dd>{formatBytes(pilotMetrics.measured_final_export_bytes)}</dd></div> : null}
                {pilotRunId ? <div><dt>Pilot run ID</dt><dd><code>{pilotRunId}</code></dd></div> : null}
              </dl>
              {preflightMetrics ? (
                <div className="pilot-metric-grid preflight-metric-grid">
                  <article>
                    <h3>{mlxRuntime ? "Bounded MLX preflight smoke" : "Measured synthetic preflight"}</h3>
                    <dl>
                      <div className="manifest-row"><dt>Candidate</dt><dd><code>{preflightMetrics.candidate_id ?? "Not recorded"}</code></dd></div>
                      <div><dt>Method</dt><dd>{preflightMetrics.method ?? "Not recorded"}</dd></div>
                      <div><dt>Precision</dt><dd>{preflightMetrics.precision ?? "Not recorded"}</dd></div>
                      <div><dt>Quantization</dt><dd>{preflightMetrics.quantization ?? "None"}</dd></div>
                      <div><dt>Distribution</dt><dd>{preflightMetrics.distribution ?? "Not recorded"}</dd></div>
                      <div><dt>World size</dt><dd>{preflightMetrics.world_size ?? "Not recorded"}</dd></div>
                      <div><dt>{mlxRuntime ? "Peak MLX memory" : "Peak CUDA memory"}</dt><dd>{formatBytes(mlxRuntime ? preflightMetrics.measured_peak_bytes : preflightMetrics.measured_peak_cuda_bytes)}</dd></div>
                      <div className="manifest-row"><dt>Scope</dt><dd>{preflightMetrics.scope ?? "Not recorded"}</dd></div>
                    </dl>
                  </article>
                </div>
              ) : null}
              {bindings.length ? (
                <dl className="binding-list">
                  {bindings.map(([name, value]) => (
                    <div key={name}><dt>{name.replace(/_/g, " ")}</dt><dd><code>{value}</code></dd></div>
                  ))}
                </dl>
              ) : <p>No cryptographic bindings were recorded.</p>}
              {phaseOne || phaseTwo ? (
                <div className="pilot-metric-grid">
                  {[{ label: "Phase 1", metrics: phaseOne }, { label: "Phase 2 resumed", metrics: phaseTwo }].map(({ label, metrics }) => metrics ? (
                    <article key={label}>
                      <h3>{label}</h3>
                      <dl>
                        <div><dt>Global step</dt><dd>{String(metrics.global_step ?? "Not recorded")}</dd></div>
                        <div><dt>Train loss</dt><dd>{String(metrics.train_loss ?? "Not recorded")}</dd></div>
                        <div><dt>Peak CUDA bytes</dt><dd>{String(metrics.measured_peak_cuda_bytes ?? "Not recorded")}</dd></div>
                        <div><dt>Reserved CUDA bytes</dt><dd>{String(metrics.measured_reserved_cuda_bytes ?? "Not recorded")}</dd></div>
                      </dl>
                    </article>
                  ) : null)}
                </div>
              ) : null}
              {mlxRuntime && pilotMetrics ? (
                <div className="pilot-metric-grid">
                  <article>
                    <h3>Uninterrupted MLX pilot</h3>
                    <dl>
                      <div><dt>Optimizer updates</dt><dd>{pilotMetrics.completed_optimizer_updates ?? "Not recorded"}</dd></div>
                      <div><dt>Finite train loss</dt><dd>{pilotMetrics.finite_train_loss === true ? "Verified" : "Not recorded"}</dd></div>
                      <div><dt>Finite validation loss</dt><dd>{pilotMetrics.finite_validation_loss === true ? "Verified" : "Not recorded"}</dd></div>
                      <div><dt>Peak unified memory</dt><dd>{formatBytes(pilotMetrics.measured_peak_bytes)}</dd></div>
                    </dl>
                  </article>
                  <article>
                    <h3>Bound adapter evidence</h3>
                    <dl>
                      <div><dt>Exact target census</dt><dd>{mlxTargetBinding?.adapter_target_instance_count !== undefined && mlxTargetBinding.expected_adapter_target_instance_count !== undefined ? `${mlxTargetBinding.adapter_target_instance_count} / ${mlxTargetBinding.expected_adapter_target_instance_count} adapter instances` : "Not recorded"}</dd></div>
                      <div><dt>Trainable tensors</dt><dd>{mlxTargetBinding?.trainable_tensor_count ?? "Not recorded"}</dd></div>
                      <div><dt>Immutable adapter artifacts</dt><dd>{pilotMetrics.adapter_manifest ? `${pilotMetrics.adapter_manifest.length} manifest-bound files` : "Not recorded"}</dd></div>
                      <div><dt>Fresh-process reload</dt><dd>{mlxReload?.fresh_process_observed === true ? "Verified" : "Not recorded"}</dd></div>
                      <div><dt>Generated verification tokens</dt><dd>{mlxReload?.generation_tokens ?? "Not recorded"}</dd></div>
                    </dl>
                  </article>
                  <article>
                    <h3>Live unified-memory admission</h3>
                    <dl>
                      <div><dt>Available at admission</dt><dd>{formatBytes(mlxAdmission?.available_unified_memory_bytes)}</dd></div>
                      <div><dt>Required including reserve</dt><dd>{formatBytes(mlxAdmission?.required_available_bytes)}</dd></div>
                      <div><dt>Aptus reserve</dt><dd>{formatBytes(mlxAdmission?.reserve_bytes)}</dd></div>
                    </dl>
                  </article>
                </div>
              ) : null}
              {checkpointContracts.some(({ contract }) => contract) ? (
                <div className="pilot-metric-grid checkpoint-contract-grid">
                  {checkpointContracts.map(({ label, contract }) => contract ? (
                    <article key={label}>
                      <h3>{label}</h3>
                      <dl>
                        <div><dt>Total bytes</dt><dd>{formatBytes(contract.total_bytes)}</dd></div>
                        <div><dt>Manifest files</dt><dd>{contract.files?.length ?? "Not recorded"}</dd></div>
                        <div className="manifest-row"><dt>Manifest SHA-256</dt><dd><code>{contract.manifest_sha256 ?? "Not recorded"}</code></dd></div>
                      </dl>
                    </article>
                  ) : null)}
                </div>
              ) : null}
            </section>
          ) : null}
        </>
      ) : (
        <section className="compile-ready-panel">
          <div className="compiler-glyph" aria-hidden="true"><span>bundle/</span><i>→</i><span>report.json</span></div>
          <h2>This bundle has no validation report yet.</h2>
          <p>Run the validator to check the plan contract, generated files, dependencies, and available runtime gates.</p>
        </section>
      )}

      <section className="validation-controls" aria-labelledby="validation-controls-title">
        <div>
          <p className="eyebrow">Next validation claim</p>
          <h2 id="validation-controls-title">Choose the gate explicitly.</h2>
        </div>
        <div className="field">
          <label htmlFor="validation-level">Validation level</label>
          <select id="validation-level" value={validationLevel} onChange={(event) => onValidationLevelChange(event.target.value as ValidateRequest["level"])}>
            <option value="contract">Contract</option>
            <option value="static">Static</option>
          </select>
        </div>
        <p className="fact-boundary">
          Dependency, model-data, measured-preflight, and pilot checks run as
          cancellable jobs in the Run stage. They are never executed inside a
          synchronous API request.
        </p>
      </section>

      <div className="sticky-actions">
        <div>
          <strong>{runActionsReady ? "Static gate passed" : "Run actions remain gated"}</strong>
          <span>{runActionsReady
            ? "Open Run to start with the dependency gate."
            : activeReport && !reportBound
              ? "Revalidate this exact plan, candidate, and model revision before creating a runtime job."
              : "Pass static validation before creating a runtime job."}</span>
        </div>
        <div className="action-buttons">
          <button type="button" className="button button-secondary" disabled={busy !== null || demoMode} onClick={() => void onValidate()}>
            {busy === "validate" ? "Validating…" : activeReport ? "Run validation again" : "Run validation"}
          </button>
          <button type="button" className="button button-primary" disabled={!runActionsReady} onClick={onOpenRun}>Open run actions</button>
        </div>
      </div>
    </>
  );
}
