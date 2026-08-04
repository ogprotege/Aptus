import { useEffect, useState } from "react";
import type { CompileResponse, Job, ValidationReport } from "../types";
import { EmptyStage } from "../components/EmptyStage";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { RunConsole } from "../components/RunConsole";
import { StageHeader } from "../components/StageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { getDesktopBridge } from "../desktopBridge";
import { validationReportMatchesBinding } from "../lib/modelPolicy";
import type { ValidationReportBindingIdentity } from "../lib/modelPolicy";
import { canStartAction, nextForwardAction } from "../lib/plan";

interface RunStageProps {
  bundle: CompileResponse | null;
  report: ValidationReport | null;
  reportBinding: ValidationReportBindingIdentity | null;
  job: Job | null;
  busy: string | null;
  demoMode: boolean;
  onCreateJob: (mode: "dependency" | "model-data" | "preflight" | "pilot" | "train") => Promise<void>;
  onRefreshJob: () => Promise<void>;
  onCancelJob: () => Promise<void>;
  onReturnToValidate: () => void;
}

function quotePosixShellArgument(value: string): string {
  return `'${value.replace(/'/g, `'"'"'`)}'`;
}

export function RunStage({
  bundle,
  report,
  reportBinding,
  job,
  busy,
  demoMode,
  onCreateJob,
  onRefreshJob,
  onCancelJob,
  onReturnToValidate,
}: RunStageProps) {
  const desktopBridge = getDesktopBridge();
  const [mode, setMode] = useState<"dependency" | "model-data" | "preflight" | "pilot" | "train">("dependency");
  const [confirmed, setConfirmed] = useState(false);
  const activeReport = report ?? bundle?.report ?? null;
  const boundReport = validationReportMatchesBinding(activeReport, reportBinding)
    ? activeReport
    : null;
  const passing = canStartAction(boundReport?.state, "dependency");
  const activeJob = Boolean(job && ["queued", "running", "cancelling"].includes(job.state));
  const displayJobState = job ? job.phase ?? job.state : null;
  const jobMatchesBundle = Boolean(
    job && bundle && job.bundle_dir && job.bundle_dir === bundle.bundle_dir,
  );
  const macDesktop = desktopBridge?.platform === "macos";
  const selectedRuntime = bundle?.runtime_contract?.training_runtime;
  const mlxRuntime = selectedRuntime === "mlx-lm";
  const localAppleRuntime = macDesktop
    && (selectedRuntime === "mlx-lm" || selectedRuntime === "pytorch-mps");
  const desktopHandoff = macDesktop && !localAppleRuntime;

  useEffect(() => {
    setConfirmed(false);
  }, [job?.id]);

  useEffect(() => {
    if (!activeJob) {
      setMode(nextForwardAction(boundReport?.state));
      setConfirmed(false);
    }
  }, [activeJob, boundReport?.state, job?.id]);

  const chooseMode = (nextMode: "dependency" | "model-data" | "preflight" | "pilot" | "train") => {
    setMode(nextMode);
    setConfirmed(false);
  };

  const startJob = () => {
    void onCreateJob(mode);
  };

  const jobMonitor = job ? (
    <>
      <RunConsole job={job} example={demoMode} />
      {!demoMode ? (
        <>
          <div className="console-actions">
            <button type="button" className="button button-secondary" disabled={busy !== null} onClick={() => void onRefreshJob()}>
              {busy === "refresh-job" ? "Refreshing…" : "Refresh job"}
            </button>
            {activeJob ? (
              <button
                type="button"
                className="button button-danger"
                disabled={busy !== null || job.state === "cancelling" || job.cancellable === false}
                onClick={() => void onCancelJob()}
              >
                {busy === "cancel-job" || job.state === "cancelling" ? "Cancelling…" : "Cancel job"}
              </button>
            ) : null}
          </div>
          {activeJob && job.cancellation_note ? <p className="job-owner-note">{job.cancellation_note}</p> : null}
        </>
      ) : null}
    </>
  ) : desktopHandoff ? null : (
    <section className="run-empty-log">
      <h2>No job has started.</h2>
      <p>Select a mode and complete the preflight. Aptus will poll the persisted job log and show it here.</p>
    </section>
  );

  if ((!bundle || !passing || !jobMatchesBundle) && activeJob && job) {
    return (
      <>
        <StageHeader
          eyebrow="Stage 5 · Execution"
          title="The active job remains observable."
          lede="Artifact evidence is missing or no longer authorizes a new submission. Aptus keeps the persisted job state and log visible until it becomes terminal."
          meta={<StatusBadge state={displayJobState ?? job.state} />}
        />
        <section className="blocked-panel" role="status">
          <h2>New jobs are blocked.</h2>
          <p>Monitor this job first. Then restore or revalidate its bundle before starting another action.</p>
        </section>
        {jobMonitor}
      </>
    );
  }

  if (!bundle || !passing) {
    return (
      <>
        <StageHeader eyebrow="Stage 5 · Execution" title="Run only what the evidence supports." lede="Aptus requires static-pass or stronger evidence before it creates a job." />
        <EmptyStage title="Run preflight is blocked" actionLabel="Review validation" onAction={onReturnToValidate}>
          Compile the bundle and pass the required validation gate first.
        </EmptyStage>
      </>
    );
  }

  const trainNeedsConfirmation = mode === "train";
  const targetBundle = quotePosixShellArgument(
    `./${bundle.bundle_dir.split("/").filter(Boolean).at(-1) ?? "aptus-bundle"}`,
  );
  const handoffCommands = [
    `aptus run ${targetBundle} --action dependency`,
    `aptus run ${targetBundle} --action model-data`,
    `aptus run ${targetBundle} --action preflight`,
    `aptus run ${targetBundle} --action pilot`,
    `aptus run ${targetBundle} --action train --confirm-full-train`,
  ].join("\n");

  return (
    <>
      <StageHeader
        eyebrow="Stage 5 · Execution"
        title="Run only what the evidence supports."
        lede={mlxRuntime
          ? "Choose the smallest execution that answers the next question. Measured preflight is a bounded MLX smoke. The pilot then proves uninterrupted LoRA or QLoRA updates and a fresh-process adapter reload before a confirmed full run from scratch."
          : "Choose the smallest execution that answers the next question. Run dependency, model-data, and synthetic method checks, then observe checkpoint continuation with the bounded real-data pilot."}
        meta={demoMode ? <ProvenanceBadge kind="example" label="Example run" /> : job ? <StatusBadge state={displayJobState ?? job.state} /> : undefined}
      />

      <section className="run-preflight" aria-labelledby="run-preflight-title">
        <div>
          <p className="eyebrow">Execution boundary</p>
          <h2 id="run-preflight-title">Run preflight</h2>
          {desktopHandoff ? (
            <p className="desktop-handoff-intro">
              This bundle targets CUDA. Transfer <code>{bundle.bundle_dir}</code> to the intended NVIDIA host before measured preflight, pilot, or training.
            </p>
          ) : (
            <p>
              The next forward action is selected automatically. You can still rerun an earlier passed gate. The {localAppleRuntime ? `${selectedRuntime} runtime on this Mac` : "local runtime"} executes the compiled bundle at <code>{bundle.bundle_dir}</code>.
            </p>
          )}
          <p className="fact-boundary dependency-contract-note" role="note">
            <code>requirements.txt</code> contains exact direct package pins for the selected method. It is not a complete transitive lock.
            {mlxRuntime ? " This MLX path executes LoRA and QLoRA only. DoRA, full-parameter training, and resume are not supported." : null}
          </p>
        </div>
        {desktopHandoff ? (
          <section className="target-host-handoff" aria-labelledby="target-host-handoff-title">
            <p className="eyebrow">Target-host handoff</p>
            <h3 id="target-host-handoff-title">Continue on the CUDA machine.</h3>
            <p>
              The macOS app never submits CUDA work locally, including plans that describe a remote CUDA machine. Copy the complete bundle to that host, install Aptus there, then run each gate in order.
            </p>
            <pre className="handoff-commands" aria-label="CUDA host commands"><code>{handoffCommands}</code></pre>
          </section>
        ) : (
          <>
            <fieldset>
              <legend>Run mode</legend>
              <div className="run-mode-grid">
                <label>
                  <input type="radio" name="run-mode" value="dependency" checked={mode === "dependency"} onChange={() => chooseMode("dependency")} />
                  <span><strong>Dependency check</strong><small>Verify the exact direct package pins in this bundle.</small></span>
                </label>
                <label>
                  <input type="radio" name="run-mode" value="model-data" checked={mode === "model-data"} onChange={() => chooseMode("model-data")} />
                  <span><strong>Model and data</strong><small>Inspect the pinned model and tokenize every canonical row.</small></span>
                </label>
                <label>
                  <input type="radio" name="run-mode" value="preflight" checked={mode === "preflight"} onChange={() => chooseMode("preflight")} />
                  <span><strong>Measured preflight</strong><small>{mlxRuntime ? "Run dependency, model-data, and a bounded MLX adapter smoke. This is not pilot evidence." : "Run dependency, model-data, and synthetic runtime-specific method checks."}</small></span>
                </label>
                <label>
                  <input type="radio" name="run-mode" value="pilot" checked={mode === "pilot"} onChange={() => chooseMode("pilot")} />
                  <span><strong>{mlxRuntime ? "Uninterrupted MLX pilot" : "Measured pilot"}</strong><small>{mlxRuntime ? "Prove at least two optimizer updates, exact adapter targets, live memory admission, bound artifacts, and a 1–4 token fresh-process reload." : "Run step 1, manifest a checkpoint, start a fresh process, and continue through step 2."}</small></span>
                </label>
                <label>
                  <input type="radio" name="run-mode" value="train" checked={mode === "train"} onChange={() => chooseMode("train")} />
                  <span><strong>Training job</strong><small>{mlxRuntime ? "After pilot-pass, run the compiled full training duration for LoRA or QLoRA from the pinned base model. Resume is unavailable." : "Use the pinned model and local dataset."}</small></span>
                </label>
              </div>
            </fieldset>

            {trainNeedsConfirmation ? (
              <>
                <label className="check-row run-confirmation">
                  <input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />
                  <span>
                    <strong>{mlxRuntime ? "I reviewed the exact plan, paths, resource estimate, warnings, pilot metrics, and bindings, and I confirm a full-duration MLX LoRA or QLoRA run from scratch." : "I reviewed the exact plan, paths, resource estimate, warnings, pilot metrics, and attestation bindings."}</strong>
                    <small>{mlxRuntime ? "Training receives a unique run-ID-bound directory and starts from the pinned base model. Resume is unavailable, so an interrupted run must start again." : "Training receives a unique run-ID-bound directory. Aptus verifies the file tree at completion, but does not make the directory immutable. Full-training resume remains fail-closed."}</small>
                  </span>
                </label>
                <p className="fact-boundary submit-admission-note">
                  {mlxRuntime
                    ? "The server performs authoritative admission when you submit. It acquires the host lease, verifies the pilot bindings, and checks live unified-memory headroom, the Aptus reserve, and disk. A stale browser snapshot never authorizes training."
                    : "The server performs the authoritative admission atomically when you submit: it acquires the host lease, deeply verifies the pilot bindings, and probes current VRAM, host RAM, and disk. A stale browser snapshot never authorizes training."}
                </p>
              </>
            ) : null}

            <div className="preflight-action">
              <div>
                <span>Validation state</span>
                <StatusBadge state={boundReport?.state ?? "unknown"} />
              </div>
              <button
                type="button"
                className="button button-primary"
                disabled={busy !== null || demoMode || Boolean(activeJob) || !canStartAction(boundReport?.state, mode) || (trainNeedsConfirmation && !confirmed)}
                onClick={startJob}
              >
                {busy === "job" ? "Starting…" : mode === "train" ? (mlxRuntime ? "Start full MLX training" : "Start training") : mode === "pilot" ? (mlxRuntime ? "Run uninterrupted pilot" : "Run measured pilot") : mode === "preflight" ? "Run measured preflight" : mode === "model-data" ? "Inspect model and data" : "Check dependencies"}
              </button>
            </div>
            {demoMode ? <p className="example-inline">Execution is disabled for example data. Clear the example and profile real inputs to create a job.</p> : null}
            {activeJob && job ? <p className="example-inline">Job {job.id} is active for this local user and host. V0.2 permits one Aptus execution job at a time across state roots.</p> : null}
            {!demoMode && !canStartAction(boundReport?.state, mode) ? (
              <p className="example-inline">
                {mode === "dependency"
                  ? "Pass static validation before checking the runtime dependencies."
                  : mode === "model-data"
                    ? "Complete the dependency check before inspecting the model and data."
                    : mode === "preflight"
                      ? "Complete model-data validation before running measured preflight."
                      : mode === "pilot"
                        ? "Select Measured preflight and complete it before starting the pilot."
                        : mlxRuntime
                          ? "Complete the uninterrupted MLX pilot and fresh-process adapter reload before starting full training."
                          : "Select Measured pilot and observe both checkpoint-continuation phases before starting training."}
              </p>
            ) : null}
            {mlxRuntime && mode === "pilot" && canStartAction(boundReport?.state, mode) ? (
              <p className="example-inline">
                The pilot starts from the pinned base model and does not create a resume point. A pass authorizes an explicitly confirmed full-duration run from scratch.
              </p>
            ) : null}
          </>
        )}
      </section>

      {jobMonitor}
    </>
  );
}
