import type { CompileResponse, ValidateRequest, ValidationReport } from "../types";
import { EmptyStage } from "../components/EmptyStage";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { StageHeader } from "../components/StageHeader";
import { ValidationGates } from "../components/ValidationGates";
import { canStartAction, formatBytes } from "../lib/plan";

interface ValidateStageProps {
  bundle: CompileResponse | null;
  report: ValidationReport | null;
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
  const runActionsReady = canStartAction(activeReport?.state, "dependency");
  const bindings = Object.entries(activeReport?.bindings ?? {});
  const preflightMetrics = activeReport?.preflight_metrics;
  const pilotMetrics = activeReport?.pilot_metrics;
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
        lede="Each gate establishes a specific claim. Static success does not imply model fit, and a synthetic method step does not imply calibrated VRAM."
        meta={demoMode ? <ProvenanceBadge kind="example" label="Example gates" /> : undefined}
      />

      {activeReport ? (
        <>
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
              <p className="eyebrow">Bound validation evidence</p>
              <h2 id="attestation-title">Attestation bound to this artifact</h2>
              <dl className="attestation-summary">
                <div><dt>Level</dt><dd>{activeReport.validation_level ?? "Not recorded"}</dd></div>
                <div><dt>Validator</dt><dd>{activeReport.validator_version ?? "Not recorded"}</dd></div>
                <div><dt>Validated at</dt><dd>{activeReport.validated_at ?? "Not recorded"}</dd></div>
                {pilotMetrics ? <div><dt>Checkpoint continuation observed</dt><dd>{pilotMetrics.checkpoint_continuation_observed === true ? "Yes" : "No"}</dd></div> : null}
                {pilotMetrics?.measured_checkpoint_bytes !== undefined ? <div><dt>Largest pilot checkpoint</dt><dd>{formatBytes(pilotMetrics.measured_checkpoint_bytes)}</dd></div> : null}
                {pilotMetrics?.measured_final_export_bytes !== undefined ? <div><dt>Largest pilot export</dt><dd>{formatBytes(pilotMetrics.measured_final_export_bytes)}</dd></div> : null}
                {pilotMetrics?.pilot_run_id ? <div><dt>Pilot run ID</dt><dd><code>{pilotMetrics.pilot_run_id}</code></dd></div> : null}
              </dl>
              {preflightMetrics ? (
                <div className="pilot-metric-grid preflight-metric-grid">
                  <article>
                    <h3>Measured synthetic preflight</h3>
                    <dl>
                      <div className="manifest-row"><dt>Candidate</dt><dd><code>{preflightMetrics.candidate_id ?? "Not recorded"}</code></dd></div>
                      <div><dt>Method</dt><dd>{preflightMetrics.method ?? "Not recorded"}</dd></div>
                      <div><dt>Precision</dt><dd>{preflightMetrics.precision ?? "Not recorded"}</dd></div>
                      <div><dt>Quantization</dt><dd>{preflightMetrics.quantization ?? "None"}</dd></div>
                      <div><dt>Distribution</dt><dd>{preflightMetrics.distribution ?? "Not recorded"}</dd></div>
                      <div><dt>World size</dt><dd>{preflightMetrics.world_size ?? "Not recorded"}</dd></div>
                      <div><dt>Peak CUDA memory</dt><dd>{formatBytes(preflightMetrics.measured_peak_cuda_bytes)}</dd></div>
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
          <span>{runActionsReady ? "Open Run to start with the dependency gate." : "Pass static validation before creating a runtime job."}</span>
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
