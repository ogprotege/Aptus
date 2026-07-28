import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ProjectDetail, ProjectRevision, ProjectRevisionSummary, ProjectSummary } from "../types";

const { getProjectMock, historyMock, revisionMock } = vi.hoisted(() => ({
  getProjectMock: vi.fn(),
  historyMock: vi.fn(),
  revisionMock: vi.fn(),
}));

vi.mock("../api", () => ({
  api: {
    getProject: getProjectMock,
    projectHistory: historyMock,
    projectRevision: revisionMock,
  },
}));

import { ProjectHistory } from "./ProjectHistory";

const projectId = `project_${"a".repeat(32)}`;
const revisionId = `revision_${"b".repeat(32)}`;
const latestRevisionId = `revision_${"c".repeat(32)}`;

const history: ProjectRevisionSummary[] = [{
  revision_id: revisionId,
  ordinal: 1,
  created_at: "2026-07-27T12:00:00Z",
  reason: "plan-created",
  plan_id: "plan_one",
  selected_candidate_id: "candidate_one",
  bundle_dir: null,
  validation_state: null,
  job_count: 0,
}];

const project: ProjectDetail = {
  schema_version: "aptus.project.v1",
  project_id: projectId,
  name: "Parish corpus adapter",
  created_at: "2026-07-27T12:00:00Z",
  updated_at: "2026-07-27T12:05:00Z",
  latest_revision_id: latestRevisionId,
  revision_count: 2,
  latest_revision: null,
};

const revision: ProjectRevision = {
  schema_version: "aptus.project-revision.v1",
  revision_id: revisionId,
  project_id: projectId,
  parent_revision_id: null,
  ordinal: 1,
  created_at: "2026-07-27T12:00:00Z",
  reason: "plan-created",
  plan_id: "plan_one",
  selected_candidate_id: "candidate_one",
  job_ids: [],
  training_authorization: {
    current: false,
    reason: "Training authorization is never durable.",
  },
  content_sha256: "d".repeat(64),
};

beforeEach(() => {
  getProjectMock.mockReset();
  historyMock.mockReset();
  revisionMock.mockReset();
  revisionMock.mockResolvedValue(revision);
});

describe("ProjectHistory", () => {
  it("inspects a revision and states the recovery authorization boundary", async () => {
    const onRecover = vi.fn(async () => undefined);
    render(
      <ProjectHistory
        projects={[project]}
        currentProject={project}
        currentHistory={history}
        onRecover={onRecover}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Project history/i }));
    fireEvent.click(screen.getByRole("button", { name: /Revision 1/i }));

    expect(await screen.findByText(/never restores training authorization/i)).toBeInTheDocument();
    expect(screen.getByText(/Revalidate current evidence and confirm training again/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Recover as new revision" }));

    await waitFor(() => expect(onRecover).toHaveBeenCalledWith(projectId, revisionId));
  });

  it("browses another saved project without changing the current workspace", async () => {
    const other = {
      ...project,
      project_id: `project_${"e".repeat(32)}`,
      name: "Second project",
      revision_count: 0,
      latest_revision_id: null,
    } satisfies ProjectSummary;
    getProjectMock.mockResolvedValue({ ...other, latest_revision: null });
    historyMock.mockResolvedValue([]);

    render(
      <ProjectHistory
        projects={[project, other]}
        currentProject={project}
        currentHistory={history}
        onRecover={vi.fn(async () => undefined)}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Project history/i }));
    fireEvent.change(screen.getByLabelText("Browse project"), {
      target: { value: other.project_id },
    });

    await waitFor(() => expect(getProjectMock).toHaveBeenCalledWith(other.project_id));
    expect(historyMock).toHaveBeenCalledWith(other.project_id);
    await waitFor(() => expect(screen.getAllByText("Second project")).toHaveLength(2));
  });
});
