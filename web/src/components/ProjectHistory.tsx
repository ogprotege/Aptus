import { useEffect, useId, useState } from "react";
import { api } from "../api";
import type {
  ProjectDetail,
  ProjectRevision,
  ProjectRevisionSummary,
  ProjectSummary,
} from "../types";

interface ProjectHistoryProps {
  projects: ProjectSummary[];
  currentProject: ProjectDetail | null;
  currentHistory: ProjectRevisionSummary[];
  disabled?: boolean;
  onRecover: (projectId: string, revisionId: string) => Promise<void>;
}

function revisionReason(reason: string): string {
  const known: Record<string, string> = {
    "plan-created": "Plan created",
    "bundle-compiled": "Bundle compiled",
    "bundle-validated": "Validation recorded",
    "job-submitted": "Job submitted",
    "legacy-import": "Legacy workspace imported",
  };
  if (reason.startsWith("recovered-from:")) return "Historical revision recovered";
  return known[reason] ?? reason.replaceAll("-", " ");
}

function revisionTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function ProjectHistory({
  projects,
  currentProject,
  currentHistory,
  disabled = false,
  onRecover,
}: ProjectHistoryProps) {
  const projectSelectId = useId();
  const historyBodyId = useId();
  const [open, setOpen] = useState(false);
  const [viewedProject, setViewedProject] = useState<ProjectDetail | null>(currentProject);
  const [history, setHistory] = useState(currentHistory);
  const [revision, setRevision] = useState<ProjectRevision | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setViewedProject(currentProject);
    setHistory(currentHistory);
    setRevision(null);
    setError(null);
  }, [currentProject, currentHistory]);

  const browseProject = async (projectId: string) => {
    if (!projectId) return;
    setBusy("project");
    setError(null);
    setRevision(null);
    try {
      const [nextProject, nextHistory] = await Promise.all([
        api.getProject(projectId),
        api.projectHistory(projectId),
      ]);
      setViewedProject(nextProject);
      setHistory(nextHistory);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Aptus could not load this project history.");
    } finally {
      setBusy(null);
    }
  };

  const inspectRevision = async (summary: ProjectRevisionSummary) => {
    if (!viewedProject) return;
    setBusy(summary.revision_id);
    setError(null);
    try {
      setRevision(await api.projectRevision(viewedProject.project_id, summary.revision_id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Aptus could not load this revision.");
    } finally {
      setBusy(null);
    }
  };

  const recoverRevision = async () => {
    if (!viewedProject || !revision) return;
    setBusy("recover");
    setError(null);
    try {
      await onRecover(viewedProject.project_id, revision.revision_id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Aptus could not recover this revision.");
    } finally {
      setBusy(null);
    }
  };

  const latestRevisionId = viewedProject?.latest_revision_id ?? null;
  const revisionIsCurrent = Boolean(
    viewedProject?.project_id === currentProject?.project_id
      && revision?.revision_id === latestRevisionId,
  );

  return (
    <section className={`project-history${open ? " is-open" : ""}`}>
      <button
        type="button"
        className="project-history-toggle"
        aria-expanded={open}
        aria-controls={historyBodyId}
        onClick={() => setOpen((current) => !current)}
      >
        <span className="project-history-disclosure" aria-hidden="true">{open ? "−" : "+"}</span>
        <span>Project history</span>
        <small>{currentProject?.revision_count ?? 0} revisions</small>
      </button>

      {open ? <div id={historyBodyId} className="project-history-body">
        <label htmlFor={projectSelectId}>Browse project</label>
        <select
          id={projectSelectId}
          value={viewedProject?.project_id ?? ""}
          disabled={disabled || busy !== null || projects.length === 0}
          onChange={(event) => void browseProject(event.target.value)}
        >
          {!viewedProject && projects.length ? <option value="">Choose a project</option> : null}
          {projects.length === 0 ? <option value="">No saved projects</option> : null}
          {projects.map((project) => (
            <option key={project.project_id} value={project.project_id}>
              {project.name}
            </option>
          ))}
        </select>

        {viewedProject ? (
          <div className="project-history-heading">
            <strong>{viewedProject.name}</strong>
            <small>{viewedProject.revision_count} immutable revisions</small>
          </div>
        ) : (
          <p className="project-history-empty">Create a plan to begin a local project history.</p>
        )}

        {history.length ? (
          <ol className="revision-list" aria-label="Project revisions">
            {history.map((item) => (
              <li key={item.revision_id}>
                <button
                  type="button"
                  aria-pressed={revision?.revision_id === item.revision_id}
                  disabled={disabled || busy !== null}
                  onClick={() => void inspectRevision(item)}
                >
                  <span>
                    <strong>Revision {item.ordinal}</strong>
                    <small>{revisionReason(item.reason)}</small>
                  </span>
                  <span>
                    <strong>{item.validation_state ?? "Not validated"}</strong>
                    <small>{revisionTime(item.created_at)}</small>
                  </span>
                </button>
              </li>
            ))}
          </ol>
        ) : null}

        {revision ? (
          <section className="revision-detail" aria-label={`Revision ${revision.ordinal} detail`}>
            <div>
              <strong>{revisionReason(revision.reason)}</strong>
              <code>{revision.revision_id}</code>
            </div>
            <dl>
              <div><dt>Plan</dt><dd>{revision.plan_id ?? "No plan"}</dd></div>
              <div><dt>Jobs</dt><dd>{revision.job_ids.length}</dd></div>
            </dl>
            <p>
              Recovery creates a new revision from verified local artifacts. It never restores training authorization. Revalidate current evidence and confirm training again.
            </p>
            <button
              type="button"
              className="button button-secondary project-recover-button"
              disabled={disabled || busy !== null || revisionIsCurrent}
              onClick={() => void recoverRevision()}
            >
              {revisionIsCurrent ? "Current revision" : busy === "recover" ? "Recovering…" : "Recover as new revision"}
            </button>
          </section>
        ) : null}

        {busy === "project" ? <p className="project-history-status" role="status">Loading project history…</p> : null}
        {error ? <p className="project-history-error" role="alert">{error}</p> : null}
      </div> : null}
    </section>
  );
}
