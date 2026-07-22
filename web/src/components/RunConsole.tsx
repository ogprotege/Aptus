import type { Job } from "../types";
import { formatBytes } from "../lib/plan";
import { StatusBadge } from "./StatusBadge";

interface RunConsoleProps {
  job: Job;
  example?: boolean;
}

export function RunConsole({ job, example = false }: RunConsoleProps) {
  const lines = Array.isArray(job.log) ? job.log : job.log.split(/\r?\n/);
  const displayState = job.phase ?? job.state;
  const capacity = job.prelaunch_capacity_check;
  const completion = job.completion_attestation;
  const reportMeasuredRun = objectValue(job.validation_report?.measured_run);
  const reportFinalExport = objectValue(job.validation_report?.final_export);
  const measuredRun = completion?.measured_run ?? reportMeasuredRun;
  const finalExport = completion?.final_export ?? reportFinalExport;
  const finalExportManifest = finalExportManifestPath(job, finalExport);
  const integrity = job.artifact_integrity;
  const hasCompletionEvidence = Boolean(
    job.run_output_dir || capacity || completion || measuredRun || finalExport || integrity,
  );

  return (
    <section className="run-console" aria-labelledby="run-console-title">
      <header>
        <div>
          <p className="eyebrow">{example ? "Labeled example output" : `Job ${job.id}`}</p>
          <h2 id="run-console-title">Run console</h2>
        </div>
        <StatusBadge state={displayState} />
      </header>
      {job.log_path || job.created_at || job.started_at || job.finished_at || job.completed_at ? (
        <dl className="job-record-grid" aria-label="Persisted job record">
          {job.log_path ? <div className="wide-job-record"><dt>Persisted log path</dt><dd><code>{job.log_path}</code></dd></div> : null}
          {job.created_at ? <div><dt>Created</dt><dd><time dateTime={job.created_at}>{job.created_at}</time></dd></div> : null}
          {job.started_at ? <div><dt>Started</dt><dd><time dateTime={job.started_at}>{job.started_at}</time></dd></div> : null}
          {job.finished_at || job.completed_at ? (
            <div><dt>Finished</dt><dd><time dateTime={job.finished_at ?? job.completed_at ?? undefined}>{job.finished_at ?? job.completed_at}</time></dd></div>
          ) : null}
        </dl>
      ) : null}
      <pre role="log" aria-live="polite" aria-label={`${job.mode} job output`}>
        <code>{lines.join("\n") || "Waiting for the first log line…"}</code>
      </pre>
      {job.error ? <p className="job-error" role="alert">{job.error}</p> : null}
      {hasCompletionEvidence ? (
        <section className="run-evidence" aria-labelledby="run-evidence-title">
          <div className="run-evidence-heading">
            <div>
              <p className="eyebrow">Bound execution evidence</p>
              <h3 id="run-evidence-title">Run artifact and capacity record</h3>
            </div>
            {integrity?.status ? <StatusBadge state={integrity.status} /> : null}
          </div>

          {job.run_output_dir ? (
            <div className="run-output-record">
              <span>Run output directory</span>
              <code>{job.run_output_dir}</code>
              <p>
                This directory is unique to run ID <code>{job.run_id ?? job.id}</code>. {completion || integrity
                  ? "Aptus deeply verified its bound file tree at completion. Later polling checks required paths only."
                  : "Aptus will deeply verify its bound file tree before marking a successful process complete."} The directory is not immutable.
              </p>
            </div>
          ) : null}

          {measuredRun || finalExport ? (
            <dl className="run-evidence-grid" aria-label="Final run metrics">
              {numberMetric(measuredRun, "global_step") !== null ? <div><dt>Global step</dt><dd>{numberMetric(measuredRun, "global_step")}</dd></div> : null}
              {numberMetric(measuredRun, "train_loss") !== null ? <div><dt>Train loss</dt><dd>{numberMetric(measuredRun, "train_loss")}</dd></div> : null}
              {numberMetric(measuredRun, "eval_loss") !== null ? <div><dt>Evaluation loss</dt><dd>{numberMetric(measuredRun, "eval_loss")}</dd></div> : null}
              {stringMetric(measuredRun, "distribution") ? <div><dt>Distribution</dt><dd>{stringMetric(measuredRun, "distribution")}</dd></div> : null}
              {numberMetric(measuredRun, "world_size") !== null ? <div><dt>World size</dt><dd>{numberMetric(measuredRun, "world_size")}</dd></div> : null}
              {peakCudaMetric(measuredRun, "measured_peak_cuda_bytes") !== null ? <div><dt>Peak CUDA allocated</dt><dd>{formatBytes(peakCudaMetric(measuredRun, "measured_peak_cuda_bytes"))}</dd></div> : null}
              {peakCudaMetric(measuredRun, "measured_reserved_cuda_bytes") !== null ? <div><dt>Peak CUDA reserved</dt><dd>{formatBytes(peakCudaMetric(measuredRun, "measured_reserved_cuda_bytes"))}</dd></div> : null}
              {completion?.state ? <div><dt>Completion state</dt><dd>{completion.state}</dd></div> : null}
              {numberMetric(finalExport, "total_bytes") !== null ? <div><dt>Final export</dt><dd>{formatBytes(numberMetric(finalExport, "total_bytes"))}</dd></div> : null}
              {stringMetric(finalExport, "verification_level") ? <div><dt>Export verification</dt><dd>{stringMetric(finalExport, "verification_level")}</dd></div> : null}
              {stringMetric(finalExport, "path") ? <div className="wide-evidence"><dt>Final export path</dt><dd><code>{stringMetric(finalExport, "path")}</code></dd></div> : null}
              {finalExportManifest ? <div className="wide-evidence"><dt>Final export manifest</dt><dd><code>{finalExportManifest}</code></dd></div> : null}
              {stringMetric(measuredRun, "metrics_sha256") ? <div className="wide-evidence"><dt>Metrics SHA-256</dt><dd><code>{stringMetric(measuredRun, "metrics_sha256")}</code></dd></div> : null}
              {stringMetric(finalExport, "manifest_sha256") ? <div className="wide-evidence"><dt>Export manifest SHA-256</dt><dd><code>{stringMetric(finalExport, "manifest_sha256")}</code></dd></div> : null}
              {completion?.measured_run_completed_at ? <div><dt>Verified at</dt><dd>{completion.measured_run_completed_at}</dd></div> : null}
            </dl>
          ) : null}

          {capacity ? (
            <div className="capacity-record">
              <h4>Submit-time capacity admission</h4>
              <dl className="run-evidence-grid">
                <div><dt>Checked at</dt><dd>{String(capacity.checked_at ?? "Not recorded")}</dd></div>
                <div><dt>Required free VRAM</dt><dd>{formatBytes(numericValue(capacity.required_free_cuda_bytes))}</dd></div>
                <div><dt>Observed free VRAM</dt><dd>{formatByteArray(capacity.free_cuda_bytes)}</dd></div>
                <div><dt>Required host RAM</dt><dd>{formatBytes(numericValue(capacity.required_host_ram_bytes))}</dd></div>
                <div><dt>Observed free host RAM</dt><dd>{formatBytes(numericValue(capacity.host_ram_free_bytes))}</dd></div>
                <div><dt>Required output disk</dt><dd>{formatBytes(numericValue(capacity.required_training_output_disk_bytes))}</dd></div>
                <div><dt>Observed free disk</dt><dd>{formatBytes(numericValue(capacity.free_disk_bytes))}</dd></div>
              </dl>
            </div>
          ) : null}

          {integrity ? (
            <div className="integrity-record">
              <strong>Artifact integrity: {integrity.status ?? "Not recorded"}</strong>
              {integrity.verified_at ? <span>Deep verification completed at {integrity.verified_at}.</span> : null}
              <p>Polling performs a presence check. This status describes the completion snapshot and does not prevent later mutation. Request deep verification before treating a later copy as current.</p>
              {integrity.missing_paths?.length ? (
                <ul className="plain-list">
                  {integrity.missing_paths.map((path) => <li key={path}><code>{path}</code></li>)}
                </ul>
              ) : null}
            </div>
          ) : null}
        </section>
      ) : null}
      <footer>
        <span>Mode <strong>{job.mode}</strong></span>
        <span>Phase <strong>{displayState}</strong></span>
        {job.phase && job.phase !== job.state ? <span>Record state <strong>{job.state}</strong></span> : null}
        <span>
          Return code <strong>{job.return_code ?? "pending"}</strong>
        </span>
      </footer>
    </section>
  );
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function numericValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function numberMetric(record: Record<string, unknown> | null | undefined, key: string): number | null {
  return numericValue(record?.[key]);
}

function stringMetric(record: Record<string, unknown> | null | undefined, key: string): string | null {
  const value = record?.[key];
  return typeof value === "string" && value ? value : null;
}

function peakCudaMetric(record: Record<string, unknown> | null | undefined, key: string): number | null {
  const direct = numberMetric(record, key);
  if (direct !== null) return direct;
  const perRank = record?.per_rank_cuda_peaks;
  if (!Array.isArray(perRank)) return null;
  const values = perRank
    .map(objectValue)
    .map((rank) => numericValue(rank?.[key]))
    .filter((value): value is number => value !== null);
  return values.length ? Math.max(...values) : null;
}

function formatByteArray(value: unknown): string {
  if (!Array.isArray(value)) return "Not measured";
  const observed = value.map(numericValue).filter((item): item is number => item !== null);
  return observed.length ? observed.map(formatBytes).join(" · ") : "Not measured";
}

function finalExportManifestPath(
  job: Job,
  finalExport: Record<string, unknown> | null | undefined,
): string | null {
  const explicit = stringMetric(finalExport, "manifest_path");
  if (explicit) return explicit;
  const runOutput = job.run_output_dir?.replace(/[\\/]+$/, "");
  if (runOutput) {
    const separator = runOutput.includes("\\") && !runOutput.includes("/") ? "\\" : "/";
    return `${runOutput}${separator}final-export.json`;
  }
  const exportDirectory = stringMetric(finalExport, "path")?.replace(/[\\/]+$/, "");
  if (!exportDirectory) return null;
  const match = exportDirectory.match(/^(.*)[\\/]final$/);
  const parent = match?.[1] ?? exportDirectory;
  const separator = parent.includes("\\") && !parent.includes("/") ? "\\" : "/";
  return `${parent}${separator}final-export.json`;
}
