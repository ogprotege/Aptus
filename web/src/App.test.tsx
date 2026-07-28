import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AptusDesktopBridge } from "./desktopBridge";

const {
  bootstrapMock,
  hardwareMock,
  profileMock,
  planMock,
  compileBundleMock,
  validateMock,
  createJobMock,
  listProjectsMock,
  getProjectMock,
  projectHistoryMock,
} = vi.hoisted(() => ({
  bootstrapMock: vi.fn(),
  hardwareMock: vi.fn(),
  profileMock: vi.fn(),
  planMock: vi.fn(),
  compileBundleMock: vi.fn(),
  validateMock: vi.fn(),
  createJobMock: vi.fn(),
  listProjectsMock: vi.fn(),
  getProjectMock: vi.fn(),
  projectHistoryMock: vi.fn(),
}));

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    api: {
      ...actual.api,
      bootstrap: bootstrapMock,
      hardware: hardwareMock,
      profile: profileMock,
      plan: planMock,
      compileBundle: compileBundleMock,
      validate: validateMock,
      createJob: createJobMock,
      listProjects: listProjectsMock,
      getProject: getProjectMock,
      projectHistory: projectHistoryMock,
    },
  };
});

import App from "./App";
import { EXAMPLE_PLAN } from "./demo";

function installDesktopBridge(): AptusDesktopBridge {
  const bridge: AptusDesktopBridge = {
    platform: "macos",
    reportWorkbenchReady: vi.fn(async () => undefined),
    pickDataset: vi.fn(async () => null),
    pickOutputDirectory: vi.fn(async () => null),
    revealInFinder: vi.fn(async () => undefined),
  };
  window.aptusDesktop = bridge;
  return bridge;
}

beforeEach(() => {
  bootstrapMock.mockReset();
  hardwareMock.mockReset();
  profileMock.mockReset();
  planMock.mockReset();
  compileBundleMock.mockReset();
  validateMock.mockReset();
  createJobMock.mockReset();
  listProjectsMock.mockReset();
  getProjectMock.mockReset();
  projectHistoryMock.mockReset();
});

afterEach(() => {
  delete window.aptusDesktop;
});

describe("desktop workbench readiness", () => {
  it("reports ready only after authenticated bootstrap commits the stable marker", async () => {
    let resolveBootstrap: ((value: { service: { version: string } }) => void) | undefined;
    bootstrapMock.mockReturnValue(new Promise((resolve) => {
      resolveBootstrap = resolve;
    }));
    const bridge = installDesktopBridge();
    vi.mocked(bridge.reportWorkbenchReady).mockImplementation(async () => {
      expect(document.querySelector("[data-aptus-workbench-ready='aptus-workbench-v1']"))
        .toBeInTheDocument();
    });

    render(<App />);

    expect(document.querySelector(".app-shell.is-desktop-host")).toBeInTheDocument();
    expect(bridge.reportWorkbenchReady).not.toHaveBeenCalled();
    expect(document.querySelector("[data-aptus-workbench-ready]")).toBeNull();

    resolveBootstrap?.({ service: { version: "0.2.0" } });

    await waitFor(() => expect(bridge.reportWorkbenchReady).toHaveBeenCalledOnce());
    expect(document.querySelector("[data-aptus-workbench-ready='aptus-workbench-v1']"))
      .toBeInTheDocument();
  });

  it("does not report ready when authenticated bootstrap fails", async () => {
    bootstrapMock.mockRejectedValue(new Error("session rejected"));
    const bridge = installDesktopBridge();

    render(<App />);

    await screen.findByText("The local planner API is unavailable.");
    expect(bridge.reportWorkbenchReady).not.toHaveBeenCalled();
    expect(document.querySelector("[data-aptus-workbench-ready]")).toBeNull();
  });

  it("describes Apple execution as uninterrupted with no resume after a local scan", async () => {
    bootstrapMock.mockResolvedValue({ service: { version: "0.2.0" } });
    hardwareMock.mockResolvedValue({
      status: "ok",
      scope: "local-measured",
      devices: [{
        name: "Apple M5 Pro (shared unified memory)",
        backend: "mps",
        total_vram_bytes: 64 * 1024 ** 3,
        free_vram_bytes: null,
        supports_bf16: false,
        supports_8bit: false,
        supports_4bit: false,
      }],
      host_ram_bytes: 64 * 1024 ** 3,
      host_ram_free_bytes: 40 * 1024 ** 3,
      reserve_per_device_bytes: 8 * 1024 ** 3,
      disk_free_bytes: 200 * 1024 ** 3,
    });

    render(<App />);

    const scanButton = await screen.findByRole("button", { name: "Scan this Aptus host" });
    fireEvent.click(scanButton);

    const notice = await screen.findByText(
      "Apple Silicon was measured as one shared memory system. Aptus will compare MLX-LM LoRA and QLoRA candidates conservatively. Measured preflight remains a bounded smoke. A passing uninterrupted pilot authorizes an explicitly confirmed full-duration run from scratch; resume is not supported.",
    );
    expect(notice).toBeInTheDocument();
    expect(screen.queryByText(/pilot and full-run approval remain fail-closed/i)).not.toBeInTheDocument();
  });

  it("retains the bootstrap project id when the user replans", async () => {
    const projectId = `project_${"a".repeat(32)}`;
    const project = {
      schema_version: "aptus.project.v1",
      project_id: projectId,
      name: "Retained project",
      created_at: "2026-07-27T12:00:00Z",
      updated_at: "2026-07-27T12:01:00Z",
      latest_revision_id: `revision_${"b".repeat(32)}`,
      revision_count: 1,
      latest_revision: null,
    };
    const plan = {
      schema_version: "aptus.training-plan.v2",
      plan_id: "plan_retained",
      project_id: projectId,
      model: {},
      dataset: {},
      hardware: {},
      target: {},
      recommended: null,
      candidates: [],
      warnings: [],
      rationale: [],
    };
    bootstrapMock.mockResolvedValue({
      service: { version: "0.2.0" },
      projects: [project],
      project,
      project_history: [],
      plan,
    });
    planMock.mockResolvedValue(plan);
    listProjectsMock.mockResolvedValue([project]);
    getProjectMock.mockResolvedValue(project);
    projectHistoryMock.mockResolvedValue([]);

    render(<App />);

    const compareButton = await screen.findByRole("button", { name: "Compare strategies" });
    fireEvent.click(compareButton);

    await waitFor(() => expect(planMock).toHaveBeenCalledWith(
      expect.objectContaining({ project_name: "Retained project" }),
      projectId,
    ));
  });

  it("starts a new project boundary before edited facts are planned", async () => {
    const previousProjectId = `project_${"a".repeat(32)}`;
    const previousRevisionId = `revision_${"b".repeat(32)}`;
    const previousProject = {
      schema_version: "aptus.project.v1",
      project_id: previousProjectId,
      name: "Previous project",
      created_at: "2026-07-27T12:00:00Z",
      updated_at: "2026-07-27T12:01:00Z",
      latest_revision_id: previousRevisionId,
      revision_count: 1,
      latest_revision: null,
    };
    const restoredPlan = {
      ...structuredClone(EXAMPLE_PLAN),
      plan_id: "plan_previous",
      project_id: previousProjectId,
      project_revision_id: previousRevisionId,
    };
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      projects: [previousProject],
      project: previousProject,
      project_history: [],
      plan: restoredPlan,
    });
    profileMock.mockResolvedValue({ facts: [], warnings: [] });
    planMock.mockResolvedValue({
      ...restoredPlan,
      plan_id: "plan_new",
      project_id: `project_${"c".repeat(32)}`,
      project_revision_id: `revision_${"d".repeat(32)}`,
    });
    listProjectsMock.mockResolvedValue([]);
    getProjectMock.mockResolvedValue(previousProject);
    projectHistoryMock.mockResolvedValue([]);

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "New Project" }));
    expect(screen.getByText(/new project started/i)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Plan name"), {
      target: { value: "Independent project" },
    });
    const profileButton = screen.getByRole("button", { name: "Profile dataset" });
    fireEvent.submit(profileButton.closest("form") as HTMLFormElement);
    await waitFor(() => expect(profileMock).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole("button", { name: "Compare strategies" }));

    await waitFor(() => expect(planMock).toHaveBeenCalledWith(
      expect.objectContaining({ project_name: "Independent project" }),
      null,
    ));
  });

  it("propagates each project revision through validation and sequential jobs", async () => {
    const projectId = `project_${"a".repeat(32)}`;
    const planRevisionId = `revision_${"b".repeat(32)}`;
    const bundleRevisionId = `revision_${"c".repeat(32)}`;
    const validationRevisionId = `revision_${"d".repeat(32)}`;
    const dependencyRevisionId = `revision_${"e".repeat(32)}`;
    const modelDataRevisionId = `revision_${"f".repeat(32)}`;
    const project = {
      schema_version: "aptus.project.v1",
      project_id: projectId,
      name: "Sequential project",
      created_at: "2026-07-27T12:00:00Z",
      updated_at: "2026-07-27T12:01:00Z",
      latest_revision_id: planRevisionId,
      revision_count: 1,
      latest_revision: null,
    };
    const plan = {
      ...structuredClone(EXAMPLE_PLAN),
      plan_id: "plan_sequential",
      project_id: projectId,
      project_revision_id: planRevisionId,
    };
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      projects: [project],
      project,
      project_history: [],
      plan,
    });
    compileBundleMock.mockResolvedValue({
      bundle_dir: "/tmp/sequential-bundle",
      archive_path: "/tmp/sequential-bundle.zip",
      files: [],
      report: { state: "static-pass", findings: [] },
      project_id: projectId,
      project_revision_id: bundleRevisionId,
    });
    validateMock.mockResolvedValue({
      state: "static-pass",
      findings: [],
      project_id: projectId,
      project_revision_id: validationRevisionId,
    });
    createJobMock
      .mockResolvedValueOnce({
        id: "job_dependency",
        state: "completed",
        action: "dependency",
        mode: "dependency",
        bundle_dir: "/tmp/sequential-bundle",
        log: "",
        return_code: 0,
        validation_report: { state: "dependency-pass", findings: [] },
        project_id: projectId,
        project_revision_id: dependencyRevisionId,
      })
      .mockResolvedValueOnce({
        id: "job_model_data",
        state: "completed",
        action: "model-data",
        mode: "model-data",
        bundle_dir: "/tmp/sequential-bundle",
        log: "",
        return_code: 0,
        validation_report: { state: "model-data-pass", findings: [] },
        project_id: projectId,
        project_revision_id: modelDataRevisionId,
      });
    listProjectsMock.mockResolvedValue([project]);
    getProjectMock.mockResolvedValue(project);
    projectHistoryMock.mockResolvedValue([]);

    render(<App />);

    const compileStageLabel = await screen.findByText("Build the bundle");
    fireEvent.click(compileStageLabel.closest("button") as HTMLButtonElement);
    fireEvent.change(screen.getByLabelText("Bundle output directory"), {
      target: { value: "/tmp/sequential-bundle" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Compile training bundle" }));
    await waitFor(() => expect(compileBundleMock).toHaveBeenCalledOnce());
    fireEvent.click(await screen.findByRole("button", { name: "Run validation" }));
    await waitFor(() => expect(validateMock).toHaveBeenCalledWith(
      "/tmp/sequential-bundle",
      "static",
      false,
      projectId,
      bundleRevisionId,
    ));
    fireEvent.click(await screen.findByRole("button", { name: "Open run actions" }));
    fireEvent.click(await screen.findByRole("button", { name: "Check dependencies" }));
    await waitFor(() => expect(createJobMock).toHaveBeenNthCalledWith(1, {
      bundle_dir: "/tmp/sequential-bundle",
      project_id: projectId,
      expected_project_revision_id: validationRevisionId,
      action: "dependency",
      confirm_full_train: false,
    }));
    fireEvent.click(await screen.findByRole("button", { name: "Inspect model and data" }));
    await waitFor(() => expect(createJobMock).toHaveBeenNthCalledWith(2, {
      bundle_dir: "/tmp/sequential-bundle",
      project_id: projectId,
      expected_project_revision_id: dependencyRevisionId,
      action: "model-data",
      confirm_full_train: false,
    }));
  });
});
