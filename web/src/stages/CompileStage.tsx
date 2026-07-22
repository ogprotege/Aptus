import type { CompileResponse, TrainingPlan } from "../types";
import { ArtifactTree } from "../components/ArtifactTree";
import { EmptyStage } from "../components/EmptyStage";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { StageHeader } from "../components/StageHeader";
import { StatusBadge } from "../components/StatusBadge";

interface CompileStageProps {
  plan: TrainingPlan | null;
  bundle: CompileResponse | null;
  busy: string | null;
  demoMode: boolean;
  onCompile: () => Promise<void>;
  onValidate: () => Promise<void>;
  onReturnToCompare: () => void;
  outputDir: string;
  onOutputDirChange: (value: string) => void;
}

export function CompileStage({
  plan,
  bundle,
  busy,
  demoMode,
  onCompile,
  onValidate,
  onReturnToCompare,
  outputDir,
  onOutputDirChange,
}: CompileStageProps) {
  if (!plan) {
    return (
      <>
        <StageHeader eyebrow="Stage 3 · Artifact compilation" title="Compile the plan, not a guess." lede="Aptus emits code only after a feasible strategy exists." />
        <EmptyStage title="No plan to compile" actionLabel="Open comparison" onAction={onReturnToCompare}>
          Compare strategies and produce a recommended plan first.
        </EmptyStage>
      </>
    );
  }

  return (
    <>
      <StageHeader
        eyebrow="Stage 3 · Artifact compilation"
        title="Compile the plan, not a guess."
        lede="The bundle keeps executable code separate from user facts, pins its dependencies, and carries its validation contract."
        meta={demoMode ? <ProvenanceBadge kind="example" label="Example bundle" /> : bundle?.report?.state ? <StatusBadge state={bundle.report.state} /> : undefined}
      />
      <p className="fact-boundary dependency-contract-note" role="note">
        <code>requirements.txt</code> contains exact direct package pins for the selected method. It is not a complete transitive lock.
      </p>

      {!bundle ? (
        <section className="compile-ready-panel">
          <div className="compiler-glyph" aria-hidden="true">
            <span>plan.json</span><i>→</i><span>bundle/</span>
          </div>
          <h2>The plan is ready for deterministic output.</h2>
          <p>Aptus will write a fresh bundle directory, generate the runtime files, and return a manifest plus its first validation report.</p>
          <div className="field compile-output-field">
            <label htmlFor="bundle-output-dir">Bundle output directory</label>
            <input id="bundle-output-dir" required value={outputDir} onChange={(event) => onOutputDirChange(event.target.value)} placeholder="./aptus-output/customer-support-adapter" />
            <small>The target must be empty so Aptus cannot overwrite an existing bundle.</small>
          </div>
          <button type="button" className="button button-primary" disabled={busy !== null || demoMode || !outputDir.trim()} onClick={() => void onCompile()}>
            {busy === "compile" ? "Compiling…" : "Compile training bundle"}
          </button>
        </section>
      ) : (
        <>
          <section className="bundle-summary" aria-labelledby="bundle-summary-title">
            <div>
              <p className="eyebrow">Bundle directory</p>
              <h2 id="bundle-summary-title">Compilation returned {bundle.files.length} artifacts.</h2>
              <code>{bundle.bundle_dir}</code>
            </div>
            <dl>
              <div>
                <dt>Archive</dt>
                <dd>{bundle.archive_path ?? "Not produced"}</dd>
              </div>
              <div>
                <dt>Initial validation</dt>
                <dd>{bundle.report?.state ?? "Not returned"}</dd>
              </div>
              <div>
                <dt>Fingerprint</dt>
                <dd className="truncate-value">{bundle.report?.artifact_fingerprint ?? "Not returned"}</dd>
              </div>
            </dl>
          </section>
          <ArtifactTree bundleDir={bundle.bundle_dir} files={bundle.files} />
          <section className="bundle-contract" aria-labelledby="bundle-contract-title">
            <p className="eyebrow">Generated-code boundary</p>
            <h2 id="bundle-contract-title">Facts remain in the plan.</h2>
            <p>The generated training entrypoint consumes the typed plan. Model identifiers and dataset paths should not be interpolated into executable source.</p>
          </section>
        </>
      )}

      <div className="sticky-actions">
        <div>
          <strong>{bundle ? "Bundle compiled" : "Compilation has not run"}</strong>
          <span>{bundle ? "Run the validation gates before execution." : "Choose a fresh output directory. Aptus will not overwrite a nonempty bundle."}</span>
        </div>
        <div className="action-buttons">
          <button type="button" className="button button-quiet" onClick={onReturnToCompare}>Review plan</button>
          {bundle ? (
            <button type="button" className="button button-primary" disabled={busy !== null || demoMode} onClick={() => void onValidate()}>
              {busy === "validate" ? "Validating…" : "Run validation"}
            </button>
          ) : null}
        </div>
      </div>
    </>
  );
}
