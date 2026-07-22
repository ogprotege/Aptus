import { useState } from "react";
import type { CompileResponse, TrainingPlan } from "../types";
import { ArtifactTree } from "../components/ArtifactTree";
import { EmptyStage } from "../components/EmptyStage";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { StageHeader } from "../components/StageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { getDesktopBridge } from "../desktopBridge";

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
  const desktopBridge = getDesktopBridge();
  const [desktopActionError, setDesktopActionError] = useState<string | null>(null);

  const chooseOutputDirectory = async () => {
    if (!desktopBridge) return;

    setDesktopActionError(null);
    try {
      const path = await desktopBridge.pickOutputDirectory();
      if (path) onOutputDirChange(path);
    } catch {
      setDesktopActionError("Aptus could not open the folder picker. Enter an absolute path instead.");
    }
  };

  const revealPath = async (path: string) => {
    if (!desktopBridge) return;

    setDesktopActionError(null);
    try {
      await desktopBridge.revealInFinder(path);
    } catch {
      setDesktopActionError("Finder could not reveal that item. Confirm that it still exists.");
    }
  };

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
            <div className="native-path-control">
              <input id="bundle-output-dir" required value={outputDir} onChange={(event) => onOutputDirChange(event.target.value)} placeholder="./aptus-output/customer-support-adapter" aria-describedby="bundle-output-help" />
              {desktopBridge ? (
                <button type="button" className="button button-secondary native-path-button" disabled={busy !== null || demoMode} onClick={() => void chooseOutputDirectory()}>
                  Choose folder
                </button>
              ) : null}
            </div>
            <small id="bundle-output-help">The target must be empty so Aptus cannot overwrite an existing bundle.</small>
          </div>
          {desktopActionError ? <p className="native-action-error" role="alert">{desktopActionError}</p> : null}
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
              {desktopBridge ? (
                <div className="native-reveal-actions" role="group" aria-label="Finder actions">
                  <button type="button" className="button button-secondary" disabled={demoMode} onClick={() => void revealPath(bundle.bundle_dir)}>
                    Show bundle in Finder
                  </button>
                  {bundle.archive_path ? (
                    <button type="button" className="button button-quiet" disabled={demoMode} onClick={() => void revealPath(bundle.archive_path as string)}>
                      Show archive in Finder
                    </button>
                  ) : null}
                </div>
              ) : null}
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
          {desktopActionError ? <p className="native-action-error" role="alert">{desktopActionError}</p> : null}
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
