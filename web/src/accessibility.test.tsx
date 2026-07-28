import axe, { type Result } from "axe-core";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProjectHistory } from "./components/ProjectHistory";
import type {
  ProjectDetail,
  ProjectRevision,
  ProjectRevisionSummary,
  ProjectSummary,
} from "./types";

const { bootstrapMock, projectRevisionMock } = vi.hoisted(() => ({
  bootstrapMock: vi.fn(),
  projectRevisionMock: vi.fn(),
}));

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    api: {
      ...actual.api,
      bootstrap: bootstrapMock,
      projectRevision: projectRevisionMock,
    },
  };
});

import App from "./App";

function violationSummary(violations: Result[]): string {
  return violations.map((violation) => {
    const targets = violation.nodes
      .flatMap((node) => node.target.map(String))
      .join(", ");
    return `${violation.id}: ${violation.help} (${targets})`;
  }).join("\n");
}

async function expectAccessible(root: Element): Promise<void> {
  const results = await axe.run(root, {
    rules: {
      // jsdom does not calculate layout or rendered colors. Browser-level
      // contrast remains a manual and visual-regression gate.
      "color-contrast": { enabled: false },
    },
  });
  expect(results.violations, violationSummary(results.violations)).toEqual([]);
}

function projectFixtures(): {
  project: ProjectDetail;
  projects: ProjectSummary[];
  history: ProjectRevisionSummary[];
  historicalRevision: ProjectRevision;
} {
  const projectId = `project_${"a".repeat(32)}`;
  const oldRevisionId = `revision_${"b".repeat(32)}`;
  const currentRevisionId = `revision_${"c".repeat(32)}`;
  const history: ProjectRevisionSummary[] = [
    {
      revision_id: oldRevisionId,
      ordinal: 1,
      created_at: "2026-07-26T12:00:00Z",
      reason: "plan-created",
      plan_id: "plan_old",
      validation_state: null,
      job_count: 0,
    },
    {
      revision_id: currentRevisionId,
      ordinal: 2,
      created_at: "2026-07-27T12:00:00Z",
      reason: "bundle-validated",
      plan_id: "plan_current",
      validation_state: "static-pass",
      job_count: 1,
    },
  ];
  const project: ProjectDetail = {
    schema_version: "aptus.project.v1",
    project_id: projectId,
    name: "Support adapter",
    created_at: "2026-07-26T12:00:00Z",
    updated_at: "2026-07-27T12:00:00Z",
    latest_revision_id: currentRevisionId,
    revision_count: 2,
    latest: history[1],
    latest_revision: null,
  };
  return {
    project,
    projects: [project],
    history,
    historicalRevision: {
      schema_version: "aptus.project-revision.v1",
      project_id: projectId,
      revision_id: oldRevisionId,
      ordinal: 1,
      created_at: "2026-07-26T12:00:00Z",
      reason: "plan-created",
      facts: {},
      plan_id: "plan_old",
      plan_snapshot: null,
      selected_candidate_id: null,
      bundle: null,
      validation: null,
      job_ids: [],
      training_authorization: {
        current: false,
        reason: "Authorization is never restored from project history.",
      },
      content_sha256: "d".repeat(64),
    },
  };
}

beforeEach(() => {
  bootstrapMock.mockReset();
  bootstrapMock.mockRejectedValue(new Error("planner unavailable"));
  projectRevisionMock.mockReset();
});

describe("workbench accessibility acceptance", () => {
  it("has no actionable axe violations in disconnected and labeled-example stages", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    const unavailable = await screen.findByText("The local planner API is unavailable.");
    await expectAccessible(container);

    const connectionBanner = unavailable.closest(".connection-banner");
    expect(connectionBanner).not.toBeNull();
    await user.click(within(connectionBanner as HTMLElement).getByRole("button", {
      name: "Load labeled example",
    }));
    await screen.findByText(
      "Every displayed result is labeled example data. No inspection, planning, compilation, validation, or training ran.",
    );

    const stageNavigation = screen.getByRole("navigation", { name: "Plan stages" });
    for (const name of ["Facts", "Compare", "Compile", "Validate", "Run"]) {
      await user.click(within(stageNavigation).getByRole("button", { name: new RegExp(`^${name}`) }));
      await waitFor(() => expect(screen.getByRole("heading", { level: 1 })).toHaveFocus());
      await expectAccessible(container);
    }
  }, 15_000);

  it("keeps the stage controls in workflow order and moves focus to each selected stage", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("The local planner API is unavailable.");

    const navigation = screen.getByRole("navigation", { name: "Plan stages" });
    const stages = within(navigation).getAllByRole("button");
    expect(stages.map((button) => button.textContent)).toEqual([
      expect.stringMatching(/^1Facts/),
      expect.stringMatching(/^2Compare/),
      expect.stringMatching(/^3Compile/),
      expect.stringMatching(/^4Validate/),
      expect.stringMatching(/^5Run/),
    ]);

    stages[0].focus();
    for (let index = 1; index < stages.length; index += 1) {
      await user.tab();
      expect(stages[index]).toHaveFocus();
    }

    for (const stage of stages) {
      stage.focus();
      await user.keyboard("{Enter}");
      await waitFor(() => expect(stage).toHaveAttribute("aria-current", "step"));
      expect(screen.getByRole("heading", { level: 1 })).toHaveFocus();
    }
  });

  it("exposes project history controls in a usable keyboard sequence", async () => {
    const user = userEvent.setup();
    const { project, projects, history, historicalRevision } = projectFixtures();
    const onRecover = vi.fn(async () => undefined);
    projectRevisionMock.mockResolvedValue(historicalRevision);

    const { container } = render(
      <ProjectHistory
        projects={projects}
        currentProject={project}
        currentHistory={history}
        onRecover={onRecover}
      />,
    );

    const historyToggle = screen.getByRole("button", { name: /Project history/ });
    historyToggle.focus();
    await user.keyboard("{Enter}");
    expect(historyToggle).toHaveAttribute("aria-expanded", "true");

    await user.tab();
    expect(screen.getByRole("combobox", { name: "Browse project" })).toHaveFocus();
    await user.tab();
    const revisions = screen.getAllByRole("button", { name: /Revision \d/ });
    expect(revisions[0]).toHaveFocus();
    await user.keyboard("{Enter}");
    await screen.findByRole("region", { name: "Revision 1 detail" });
    expect(revisions[0]).toHaveAttribute("aria-pressed", "true");

    await user.tab();
    expect(revisions[1]).toHaveFocus();
    await user.tab();
    const recover = screen.getByRole("button", { name: "Recover as new revision" });
    expect(recover).toHaveFocus();
    await user.keyboard("{Enter}");
    await waitFor(() => expect(onRecover).toHaveBeenCalledWith(
      project.project_id,
      historicalRevision.revision_id,
    ));
    await expectAccessible(container);
  });
});
