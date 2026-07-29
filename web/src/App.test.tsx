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
import { EXAMPLE_DRAFT, EXAMPLE_PLAN } from "./demo";
import type { FactDraft, TrainingPlan } from "./types";

const QWEN3_REVISION = "d".repeat(40);
const REVIEWED_QWEN3_LAYOUT = {
  default_bits: 4,
  default_group_size: 64,
  module_overrides: Array.from({ length: 48 }, (_, index) => ({
    module_path: `model.layers.${index}.mlp.gate`,
    bits: 8,
    group_size: 64,
  })).sort((left, right) => left.module_path.localeCompare(right.module_path)),
};

function exactQwen3MoEDraft(): FactDraft {
  const draft = structuredClone(EXAMPLE_DRAFT);
  draft.project_name = "Qwen3 MoE pilot";
  draft.model = {
    ...draft.model,
    model_id: "Qwen/Qwen3-30B-A3B",
    revision: QWEN3_REVISION,
    family: "qwen3_moe",
    parameters_b: 30.5,
    hidden_size: 2048,
    layers: 48,
    context_length: 32768,
    intermediate_size: 768,
    model_type: "qwen3_moe",
    architecture: "Qwen3MoeForCausalLM",
    quantization_bits: 4,
    quantization_layout: structuredClone(REVIEWED_QWEN3_LAYOUT),
    active_parameters_b: null,
    sparse_layer_count: null,
    moe: {
      expert_count: 128,
      experts_per_token: 8,
      expert_intermediate_size: 768,
      decoder_sparse_step: 1,
      mlp_only_layers: [],
      shared_expert_intermediate_size: null,
    },
  };
  draft.hardware.devices[0] = {
    ...draft.hardware.devices[0],
    name: "Apple M5 Pro (shared unified memory)",
    backend: "mps",
    total_vram_gib: 64,
    free_vram_gib: 40,
  };
  draft.target.runtime = "mlx-lm";
  draft.target.method_preference = "qlora";
  return draft;
}

function exactQwen3MoEPlan(): TrainingPlan {
  const recommended = {
    candidate_id: "qlora-single",
    method: "qlora",
    distribution: "single",
    status: "conditional",
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"],
    runtime_contract: {
      schema_version: "aptus.runtime-contract.v1",
      compute_backend: "mps",
      training_runtime: "mlx-lm",
      compiler_id: "mlx-lm.qlora.v1",
      estimator_id: "aptus-memory-mlx-v2",
      evidence_requirement: "pilot-required",
      export_kind: "mlx-lm-adapter",
    },
  };
  return {
    schema_version: "aptus.training-plan.v3",
    plan_id: "plan_qwen3_moe",
    model: {
      model_id: "Qwen/Qwen3-30B-A3B",
      revision: QWEN3_REVISION,
      family: "qwen3_moe",
      parameters: 30_500_000_000,
      hidden_size: 2048,
      layers: 48,
      context_length: 32768,
      intermediate_size: 768,
      license_name: "apache-2.0",
      training_allowed: true,
      tokenizer_id: "Qwen/Qwen3-30B-A3B",
      model_type: "qwen3_moe",
      architecture: "Qwen3MoeForCausalLM",
      quantization_bits: 4,
      quantization_layout: REVIEWED_QWEN3_LAYOUT,
      active_parameters: 3_300_000_000,
      sparse_layer_count: 48,
      moe: {
        expert_count: 128,
        experts_per_token: 8,
        expert_intermediate_size: 768,
        decoder_sparse_step: 1,
        mlp_only_layers: [],
        shared_expert_intermediate_size: null,
      },
    },
    dataset: {
      source_path: "/data/example-support.jsonl",
      source_format: "jsonl",
      schema_name: "text",
      sampled_examples: 1000,
    },
    hardware: {
      devices: [{
        name: "Apple M5 Pro (shared unified memory)",
        backend: "mps",
        total_vram_bytes: 64 * 1024 ** 3,
        free_vram_bytes: 40 * 1024 ** 3,
        supports_bf16: false,
        supports_8bit: false,
        supports_4bit: false,
      }],
      host_ram_bytes: 64 * 1024 ** 3,
      host_ram_free_bytes: 40 * 1024 ** 3,
      reserve_per_device_bytes: 8 * 1024 ** 3,
      disk_free_bytes: 200 * 1024 ** 3,
    },
    target: {
      task: "sft",
      objective: "memory",
      sequence_length: 512,
      effective_batch_size: 1,
      max_epochs: 1,
      method_preference: "qlora",
      training_runtime: "mlx-lm",
      evaluation_fraction: 0.1,
      packing: false,
      checkpoint_steps: 100,
    },
    recommended,
    candidates: [recommended],
    warnings: [],
    rationale: [],
  };
}

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
  it("keeps plan-derived MoE facts and compatibility across planning and restore", async () => {
    const plan = exactQwen3MoEPlan();
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      defaults: exactQwen3MoEDraft(),
      projects: [],
      project: null,
      project_history: [],
    });
    profileMock.mockResolvedValue({ facts: [], warnings: [] });
    planMock.mockResolvedValue(plan);

    const { unmount } = render(<App />);

    const profileButton = await screen.findByRole("button", { name: "Profile dataset" });
    fireEvent.submit(profileButton.closest("form") as HTMLFormElement);
    await waitFor(() => expect(profileMock).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole("button", { name: "Compare strategies" }));
    await waitFor(() => expect(planMock).toHaveBeenCalledOnce());
    fireEvent.click(await screen.findByRole("button", { name: "Edit facts" }));

    expect(await screen.findByRole("heading", { name: "Exact MoE path recognized" }))
      .toBeInTheDocument();
    expect(screen.getByText(
      "mlx-lm supports qlora on single. The current v3 plan binds this exact topology to single-device MLX-LM QLoRA with attention-only adapters. Measured preflight and a real-model pilot remain mandatory.",
    )).toBeInTheDocument();
    expect(screen.getByText("3.3B")).toBeInTheDocument();
    expect(screen.getByText("48")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Total resident parameters"), {
      target: { value: "31" },
    });
    await waitFor(() => {
      expect(screen.queryByText("3.3B")).not.toBeInTheDocument();
      expect(screen.getAllByText("Derived during planning")).toHaveLength(2);
    });

    unmount();
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      projects: [],
      project: null,
      project_history: [],
      plan,
    });
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Exact MoE path recognized" }))
      .toBeInTheDocument();
    expect(screen.getByText(
      "mlx-lm supports qlora on single. The current v3 plan binds this exact topology to single-device MLX-LM QLoRA with attention-only adapters. Measured preflight and a real-model pilot remain mandatory.",
    )).toBeInTheDocument();
    expect(screen.getByText("3.3B")).toBeInTheDocument();
    expect(screen.getByText("48")).toBeInTheDocument();
  });

  it("preserves a legacy project but requires replan before restore", async () => {
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      projects: [],
      project: null,
      project_history: [],
      plan: null,
      replan_required: {
        status: "replan_required",
        plan_id: "plan_legacy",
        found_schema: "aptus.training-plan.v2",
        required_schema: "aptus.training-plan.v3",
        source: "project-revision",
        message: "This saved plan predates the current executable contract.",
      },
    });

    render(<App />);

    expect(await screen.findByText(
      "This saved plan predates the current executable contract.",
    )).toBeInTheDocument();
    expect(screen.queryByText(/Restored the latest local project revision/))
      .not.toBeInTheDocument();
  });

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
      schema_version: "aptus.training-plan.v3",
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
