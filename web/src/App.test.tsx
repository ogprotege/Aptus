import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AptusDesktopBridge } from "./desktopBridge";

const {
  bootstrapMock,
  hardwareMock,
  inspectModelMock,
  profileMock,
  planMock,
  compileBundleMock,
  validateMock,
  createJobMock,
  listProjectsMock,
  getProjectMock,
  projectHistoryMock,
  projectRevisionMock,
  recoverProjectRevisionMock,
} = vi.hoisted(() => ({
  bootstrapMock: vi.fn(),
  hardwareMock: vi.fn(),
  inspectModelMock: vi.fn(),
  profileMock: vi.fn(),
  planMock: vi.fn(),
  compileBundleMock: vi.fn(),
  validateMock: vi.fn(),
  createJobMock: vi.fn(),
  listProjectsMock: vi.fn(),
  getProjectMock: vi.fn(),
  projectHistoryMock: vi.fn(),
  projectRevisionMock: vi.fn(),
  recoverProjectRevisionMock: vi.fn(),
}));

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    api: {
      ...actual.api,
      bootstrap: bootstrapMock,
      hardware: hardwareMock,
      inspectModel: inspectModelMock,
      profile: profileMock,
      plan: planMock,
      compileBundle: compileBundleMock,
      validate: validateMock,
      createJob: createJobMock,
      listProjects: listProjectsMock,
      getProject: getProjectMock,
      projectHistory: projectHistoryMock,
      projectRevision: projectRevisionMock,
      recoverProjectRevision: recoverProjectRevisionMock,
    },
  };
});

import App from "./App";
import { EXAMPLE_DRAFT, EXAMPLE_PLAN } from "./demo";
import type {
  CandidatePlan,
  FactDraft,
  ModelInspectionReceipt,
  ModelPolicyPath,
  TrainingPlan,
} from "./types";

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

function inspectionReceipt(
  modelId: string,
  revision: string,
): ModelInspectionReceipt {
  return {
    schema_version: "aptus.model-inspection-receipt.v1",
    receipt_id: `receipt_${"a".repeat(20)}`,
    model_id: modelId,
    resolved_revision: revision,
    observed_facts_sha256: "b".repeat(64),
    decision: {
      schema_version: "aptus.model-compatibility.v2",
      decision_id: `compat_${"c".repeat(20)}`,
      subject_facts_sha256: "d".repeat(64),
      kind: "family-recognized",
      family: "llama",
      policy_id: null,
      policy_version: null,
      paths: [],
      reason_codes: ["family-recognized"],
      evidence_ids: [],
      reason: "The dense family is recognized without an artifact-specific policy.",
    },
    provenance_summary: [{
      field: "family",
      kind: "provider-declared",
      source: "Provider config",
      observed_at: "2026-07-29T12:00:00+00:00",
      resolved_revision: revision,
    }],
    provenance_requirement: null,
    provenance_requirement_met: false,
    evaluated_at: "2026-07-29T12:00:00+00:00",
  };
}

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
  const decisionId = `compat_${"a".repeat(20)}`;
  const subjectDigest = "b".repeat(64);
  const runtimeContract: ModelPolicyPath["runtime_contract"] = {
    schema_version: "aptus.runtime-contract.v1",
    compute_backend: "mps",
    training_runtime: "mlx-lm",
    compiler_id: "mlx-lm.qlora.v1",
    estimator_id: "aptus-memory-mlx-v2",
    evidence_requirement: "pilot-required",
    export_kind: "mlx-lm-adapter",
  };
  const recommended: CandidatePlan = {
    candidate_id: `cand_${"e".repeat(20)}`,
    model_policy_decision_id: decisionId,
    policy_binding: {
      schema_version: "aptus.model-policy-binding.v1" as const,
      decision_id: decisionId,
      subject_facts_sha256: subjectDigest,
      policy_id: "model.qwen3-moe.mlx-qlora",
      policy_version: "1.0.0",
      path_id: "mlx-lm.qlora.single.attention-qkvo.v1",
      source: "user-attested" as const,
      inspection_receipt_id: null,
      reason_codes: ["exact-reviewed-artifact", "pilot-not-yet-proven"],
      evidence_ids: ["policy.qwen3-moe.mlx-qlora.v1"],
    },
    method: "qlora",
    distribution: "single",
    status: "conditional",
    feasible: true,
    rejection_reasons: [],
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"],
    runtime_contract: runtimeContract,
  };
  return {
    schema_version: "aptus.training-plan.v5",
    plan_id: `plan_${"d".repeat(20)}`,
    model_policy_snapshot_sha256: "a".repeat(64),
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
    recommendation_rationale: [],
    model_policy_decision: {
      schema_version: "aptus.model-compatibility.v2",
      decision_id: decisionId,
      subject_facts_sha256: subjectDigest,
      kind: "path-matched",
      family: "qwen3_moe",
      policy_id: "model.qwen3-moe.mlx-qlora",
      policy_version: "1.0.0",
      paths: [{
        path_id: "mlx-lm.qlora.single.attention-qkvo.v1",
        method: "qlora",
        distribution: "single",
        adapter_profile_id: "attention-qkvo.v1",
        target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"],
        runtime_contract: runtimeContract,
        required_validation_levels: ["model-data", "measured-preflight", "pilot"],
        evidence_ids: ["policy.qwen3-moe.mlx-qlora.v1"],
      }],
      reason_codes: ["exact-reviewed-artifact", "pilot-not-yet-proven"],
      evidence_ids: ["policy.qwen3-moe.mlx-qlora.v1"],
      reason: "Exact reviewed test fixture.",
    },
    model_policy_decision_source: "user-attested",
    inspection_receipt: null,
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
  inspectModelMock.mockReset();
  profileMock.mockReset();
  planMock.mockReset();
  compileBundleMock.mockReset();
  validateMock.mockReset();
  createJobMock.mockReset();
  listProjectsMock.mockReset();
  getProjectMock.mockReset();
  projectHistoryMock.mockReset();
  projectRevisionMock.mockReset();
  recoverProjectRevisionMock.mockReset();
});

afterEach(() => {
  delete window.aptusDesktop;
  vi.unstubAllGlobals();
});

describe("desktop workbench readiness", () => {
  it("does not silently downgrade an unreceipted provider inspection", async () => {
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      defaults: structuredClone(EXAMPLE_DRAFT),
      projects: [],
      project: null,
      project_history: [],
    });
    inspectModelMock.mockResolvedValue({
      status: "ok",
      model_id: EXAMPLE_DRAFT.model.model_id,
      requested_revision: EXAMPLE_DRAFT.model.revision,
      resolved_revision: "b".repeat(40),
      facts: { family: "llama" },
    });

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Inspect and pin model" }));
    expect(await screen.findByText(
      "The provider inspection did not return a receipt bound to this model and resolved revision.",
    )).toBeInTheDocument();
    expect(screen.getByLabelText("Immutable revision"))
      .toHaveValue(EXAMPLE_DRAFT.model.revision);
    expect(screen.queryByRole("heading", { name: "Revision-bound facts applied" }))
      .not.toBeInTheDocument();
  });

  it("retains an inspection receipt across user attestations and clears it on identity edits", async () => {
    const resolvedRevision = "b".repeat(40);
    const receipt = inspectionReceipt(EXAMPLE_DRAFT.model.model_id, resolvedRevision);
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      defaults: structuredClone(EXAMPLE_DRAFT),
      projects: [],
      project: null,
      project_history: [],
    });
    inspectModelMock.mockResolvedValue({
      status: "ok",
      model_id: EXAMPLE_DRAFT.model.model_id,
      requested_revision: EXAMPLE_DRAFT.model.revision,
      resolved_revision: resolvedRevision,
      facts: {
        architecture: "LlamaForCausalLM",
        architectures: ["LlamaForCausalLM"],
        model_type: "llama",
        family: "llama",
        hidden_size: 4096,
        intermediate_size: 11008,
        layers: 32,
        context_length: 4096,
        license_name: "Example license entry",
      },
      inspection_receipt: receipt,
      warnings: [],
    });
    profileMock.mockResolvedValue({ facts: [], warnings: [] });
    planMock.mockResolvedValue(structuredClone(EXAMPLE_PLAN));

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Inspect and pin model" }));
    await screen.findByRole("heading", { name: "Revision-bound facts applied" });
    fireEvent.change(screen.getByLabelText("Total resident parameters"), {
      target: { value: "7.5" },
    });
    const trainingPermission = screen.getByRole("checkbox", {
      name: /I confirmed this model permits the intended training/i,
    });
    fireEvent.click(trainingPermission);
    fireEvent.click(trainingPermission);
    fireEvent.submit(
      screen.getByRole("button", { name: "Profile dataset" }).closest("form") as HTMLFormElement,
    );
    await waitFor(() => expect(profileMock).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole("button", { name: "Compare strategies" }));

    await waitFor(() => expect(planMock).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        model: expect.objectContaining({
          revision: resolvedRevision,
          parameters_b: 7.5,
          training_allowed: true,
        }),
      }),
      null,
      receipt,
    ));

    fireEvent.click(await screen.findByRole("button", { name: "Edit facts" }));
    fireEvent.change(screen.getByLabelText("Architecture family"), {
      target: { value: "mistral" },
    });
    fireEvent.submit(
      screen.getByRole("button", { name: "Profile dataset" }).closest("form") as HTMLFormElement,
    );
    await waitFor(() => expect(profileMock).toHaveBeenCalledTimes(2));
    fireEvent.click(screen.getByRole("button", { name: "Compare strategies" }));

    await waitFor(() => expect(planMock).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        model: expect.objectContaining({ family: "mistral" }),
      }),
      null,
    ));
  });

  it("restores the inspection receipt from a v5 plan during bootstrap", async () => {
    const plan = exactQwen3MoEPlan();
    const receipt = inspectionReceipt("Qwen/Qwen3-30B-A3B", QWEN3_REVISION);
    plan.inspection_receipt = receipt;
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      projects: [],
      project: null,
      project_history: [],
      plan,
    });
    planMock.mockResolvedValue(plan);

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Compare strategies" }));
    await waitFor(() => expect(planMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: expect.objectContaining({
          model_id: "Qwen/Qwen3-30B-A3B",
          revision: QWEN3_REVISION,
        }),
      }),
      null,
      receipt,
    ));
  });

  it("restores the recovered plan receipt before the next comparison", async () => {
    const projectId = `project_${"a".repeat(32)}`;
    const currentRevisionId = `revision_${"b".repeat(32)}`;
    const historicalRevisionId = `revision_${"c".repeat(32)}`;
    const recoveredRevisionId = `revision_${"d".repeat(32)}`;
    const project = {
      schema_version: "aptus.project.v1",
      project_id: projectId,
      name: "Receipt recovery",
      created_at: "2026-07-29T10:00:00Z",
      updated_at: "2026-07-29T11:00:00Z",
      latest_revision_id: currentRevisionId,
      revision_count: 2,
      latest_revision: null,
    };
    const historicalSummary = {
      revision_id: historicalRevisionId,
      ordinal: 1,
      created_at: "2026-07-29T10:00:00Z",
      reason: "plan-created",
      plan_id: "plan_historical",
      job_count: 0,
    };
    const restoredPlan = exactQwen3MoEPlan();
    const receipt = inspectionReceipt("Qwen/Qwen3-30B-A3B", QWEN3_REVISION);
    restoredPlan.inspection_receipt = receipt;
    bootstrapMock
      .mockResolvedValueOnce({
        api_contract_version: "aptus.api.v1",
        service: { version: "0.2.0" },
        projects: [project],
        project,
        project_history: [historicalSummary],
        plan: exactQwen3MoEPlan(),
      })
      .mockResolvedValueOnce({
        api_contract_version: "aptus.api.v1",
        service: { version: "0.2.0" },
        projects: [project],
        project,
        project_history: [],
        plan: restoredPlan,
      });
    projectRevisionMock.mockResolvedValue({
      schema_version: "aptus.project-revision.v1",
      revision_id: historicalRevisionId,
      project_id: projectId,
      parent_revision_id: null,
      ordinal: 1,
      created_at: "2026-07-29T10:00:00Z",
      reason: "plan-created",
      plan_id: "plan_historical",
      facts: {},
      plan_snapshot: null,
      bundle: null,
      validation: null,
      job_ids: [],
      training_authorization: { current: false, reason: "Recovery clears authorization." },
      content_sha256: "e".repeat(64),
    });
    recoverProjectRevisionMock.mockResolvedValue({
      status: "recovered",
      project_id: projectId,
      revision: {
        revision_id: recoveredRevisionId,
        ordinal: 3,
      },
      training_authorization_current: false,
    });
    planMock.mockResolvedValue(restoredPlan);

    render(<App />);

    fireEvent.click((await screen.findAllByRole("button", { name: /Project history/i }))[0]);
    fireEvent.click(screen.getByRole("button", { name: /Revision 1/i }));
    fireEvent.click(await screen.findByRole("button", { name: "Recover as new revision" }));
    await screen.findByText(/Recovered revision 3 as a new immutable revision/i);
    fireEvent.click(screen.getByRole("button", { name: "Edit facts" }));
    fireEvent.click(screen.getByRole("button", { name: "Compare strategies" }));

    await waitFor(() => expect(planMock).toHaveBeenCalledWith(
      expect.any(Object),
      projectId,
      receipt,
    ));
  });

  it("keeps plan-derived MoE facts and the server policy presentation across planning and restore", async () => {
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

    expect(await screen.findByRole("heading", { name: "Pinned MoE topology" }))
      .toBeInTheDocument();
    const plannedMatch = screen.getByRole("article", { name: "Model-policy match" });
    const plannedPath = screen.getByRole("article", { name: "Selected candidate path" });
    expect(within(plannedMatch).getByText("Exact reviewed test fixture.")).toBeInTheDocument();
    expect(within(plannedPath).getByText("mlx-lm.qlora.single.attention-qkvo.v1"))
      .toBeInTheDocument();
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

    expect(await screen.findByRole("heading", { name: "Pinned MoE topology" }))
      .toBeInTheDocument();
    const restoredMatch = screen.getByRole("article", { name: "Model-policy match" });
    const restoredPath = screen.getByRole("article", { name: "Selected candidate path" });
    expect(within(restoredMatch).getByText("Exact reviewed test fixture.")).toBeInTheDocument();
    expect(within(restoredPath).getByText("mlx-lm.qlora.single.attention-qkvo.v1"))
      .toBeInTheDocument();
    expect(screen.getByText("3.3B")).toBeInTheDocument();
    expect(screen.getByText("48")).toBeInTheDocument();
  });

  it("projects only the candidate currently inspected in Compare", async () => {
    const plan = exactQwen3MoEPlan();
    const alternative: CandidatePlan = {
      ...plan.recommended,
      candidate_id: `cand_${"f".repeat(20)}`,
      policy_binding: null,
      method: "lora",
      status: "feasible",
      target_modules: ["q_proj", "v_proj"],
      runtime_contract: {
        schema_version: "aptus.runtime-contract.v1",
        compute_backend: "cuda",
        training_runtime: "transformers-peft-cuda",
        compiler_id: "transformers.peft-lora.v1",
        estimator_id: "aptus-memory-v2",
        evidence_requirement: "pilot-required",
        export_kind: "peft-adapter-safetensors",
      },
    };
    plan.candidates = [plan.recommended, alternative];
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      projects: [],
      project: null,
      project_history: [],
      plan,
    });

    render(<App />);

    const compareButtons = await screen.findAllByRole("button", { name: /Compare.*Resolve feasibility/i });
    fireEvent.click(compareButtons[0]);
    const initialPath = screen.getByRole("article", { name: "Selected candidate path" });
    expect(within(initialPath).getByText("Bound")).toBeInTheDocument();
    expect(within(initialPath).getByText("mlx-lm.qlora.single.attention-qkvo.v1"))
      .toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", {
      name: /Inspect LoRA candidate evidence/i,
    })[0]);

    const alternativePath = screen.getByRole("article", { name: "Selected candidate path" });
    expect(within(alternativePath).getByText("Unbound")).toBeInTheDocument();
    expect(within(alternativePath).getByText("transformers-peft-cuda"))
      .toBeInTheDocument();
    expect(within(alternativePath).queryByText("mlx-lm.qlora.single.attention-qkvo.v1"))
      .not.toBeInTheDocument();
  });

  it("does not consume or unlock stages from missing or wrong validation bindings", async () => {
    for (const bindings of [
      undefined,
      {
        plan_id: `plan_${"f".repeat(20)}`,
        candidate_id: `cand_${"e".repeat(20)}`,
        model_revision: QWEN3_REVISION,
      },
      {
        plan_id: `plan_${"d".repeat(20)}`,
        candidate_id: `cand_${"e".repeat(20)}`,
        model_revision: "f".repeat(40),
      },
    ]) {
      const plan = exactQwen3MoEPlan();
      bootstrapMock.mockResolvedValueOnce({
        api_contract_version: "aptus.api.v1",
        service: { version: "0.2.0" },
        projects: [],
        project: null,
        project_history: [],
        plan,
        bundle: {
          bundle_dir: "/tmp/wrong-binding",
          files: [],
          report: {
            state: "pilot-pass",
            authorization_status: "current",
            authorization_current: true,
            bindings,
          },
        },
      });
      const rendered = render(<App />);
      const evidence = await screen.findByRole("article", { name: "Evidence readiness" });
      expect(within(evidence).getByText("Evidence required")).toBeInTheDocument();
      expect(within(evidence).getByText("Not bound")).toBeInTheDocument();
      expect(within(evidence).queryByText("Admission active")).not.toBeInTheDocument();
      const validateStage = (await screen.findAllByRole("button", {
        name: /Validate.*Pass the gates/i,
      }))[0];
      expect(within(validateStage).queryByText("Complete.")).not.toBeInTheDocument();
      fireEvent.click(validateStage);
      expect(await screen.findByText(/does not belong to the compiled recommendation/i))
        .toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Open run actions" })).toBeDisabled();
      rendered.unmount();
    }
  });

  it("does not suggest impossible validation actions for a no-feasible row", async () => {
    const success = exactQwen3MoEPlan();
    const rejected: CandidatePlan = {
      ...success.recommended,
      status: "infeasible",
      feasible: false,
      rejection_reasons: ["The point estimate exceeds available memory."],
    };
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      projects: [],
      project: null,
      project_history: [],
      plan: {
        no_feasible_plan: true,
        recommended: null,
        candidates: [rejected],
        model: {
          model_id: "Qwen/Qwen3-30B-A3B",
          revision: QWEN3_REVISION,
        },
        model_policy_decision: success.model_policy_decision,
        model_policy_decision_source: success.model_policy_decision_source,
        inspection_receipt: null,
        warnings: ["No candidate passed every hard gate."],
        rationale: ["No candidate passed every hard gate."],
        recommendation_rationale: ["No candidate passed every hard gate."],
      },
    });

    render(<App />);
    fireEvent.click((await screen.findAllByRole("button", {
      name: /Compare.*Resolve feasibility/i,
    }))[0]);

    const evidence = screen.getByRole("article", { name: "Evidence readiness" });
    expect(within(evidence).getByText("Not applicable")).toBeInTheDocument();
    expect(within(evidence).getByText(/rejected candidate cannot advance/i)).toBeInTheDocument();
    expect(within(evidence).queryByText(/validate this candidate|run the .*gate/i))
      .not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Compile recommended bundle" })).toBeDisabled();
  });

  it("surfaces bootstrap policy-version skew without hydrating a policy panel", async () => {
    bootstrapMock.mockRejectedValue(new Error(
      "Unsupported model policy decision contract \"aptus.model-compatibility.v3\". Update Aptus so the workbench and local service use the same contract.",
    ));

    render(<App />);

    expect(await screen.findByText(/unsupported model policy decision contract.*update Aptus/i))
      .toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Model policy" })).not.toBeInTheDocument();
  });

  it("surfaces plan, receipt, and binding schema skew from the real bootstrap decoder", async () => {
    const actualApi = await vi.importActual<typeof import("./api")>("./api");
    bootstrapMock.mockImplementation((signal) => actualApi.api.bootstrap(signal));
    const cases: Array<[RegExp, () => Record<string, unknown>]> = [
      [
        /requires aptus\.training-plan\.v5/i,
        () => ({ ...structuredClone(exactQwen3MoEPlan()), schema_version: "aptus.training-plan.v6" }),
      ],
      [
        /invalid model policy binding.*unsupported schema version/i,
        () => {
          const plan = structuredClone(exactQwen3MoEPlan());
          plan.recommended.policy_binding!.schema_version = "aptus.model-policy-binding.v2" as never;
          return plan;
        },
      ],
      [
        /invalid model inspection receipt.*unsupported schema version/i,
        () => {
          const plan = structuredClone(exactQwen3MoEPlan());
          const receiptId = `receipt_${"a".repeat(20)}`;
          plan.model_policy_decision_source = "provider-inspection";
          plan.recommended.policy_binding = {
            ...plan.recommended.policy_binding!,
            source: "provider-inspection",
            inspection_receipt_id: receiptId,
          };
          plan.inspection_receipt = {
            schema_version: "aptus.model-inspection-receipt.v2" as never,
            receipt_id: receiptId,
            model_id: plan.model.model_id,
            resolved_revision: plan.model.revision,
            observed_facts_sha256: "f".repeat(64),
            decision: plan.model_policy_decision,
            provenance_summary: [{
              field: "family",
              kind: "provider-declared",
              source: "Provider config",
              observed_at: "2026-08-04T12:00:00+00:00",
              resolved_revision: plan.model.revision,
            }],
            provenance_requirement: "provider-declared",
            provenance_requirement_met: true,
            evaluated_at: "2026-08-04T12:00:00+00:00",
          };
          return plan;
        },
      ],
    ];

    for (const [message, makePlan] of cases) {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
        api_contract_version: "aptus.api.v1",
        service: { version: "0.2.0" },
        projects: [],
        project: null,
        project_history: [],
        plan: makePlan(),
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })));
      const rendered = render(<App />);
      expect(await screen.findByRole("alert")).toHaveTextContent(message);
      rendered.unmount();
    }
  });

  it("keeps admission evidence unchanged when train submission fails", async () => {
    const plan = exactQwen3MoEPlan();
    const projectId = `project_${"a".repeat(32)}`;
    const revisionId = `revision_${"b".repeat(32)}`;
    const bindings = {
      plan_id: plan.plan_id,
      candidate_id: plan.recommended.candidate_id,
      model_revision: plan.model.revision,
    };
    bootstrapMock.mockResolvedValue({
      api_contract_version: "aptus.api.v1",
      service: { version: "0.2.0" },
      projects: [],
      project: null,
      project_history: [],
      plan,
      bundle: {
        bundle_dir: "/tmp/failed-admission",
        files: [],
        runtime_contract: plan.recommended.runtime_contract,
        project_id: projectId,
        project_revision_id: revisionId,
        report: {
          state: "pilot-pass",
          bindings,
          authorization_status: "blocked",
          authorization_current: false,
          authorization_error: "Original bound admission reason.",
        },
      },
    });
    createJobMock.mockRejectedValue(new Error("Host lease is already held."));

    render(<App />);
    const runStage = (await screen.findAllByRole("button", {
      name: /Run.*Execute with evidence/i,
    }))[0];
    fireEvent.click(runStage);
    await waitFor(() => expect(screen.getByLabelText(/Training job/i)).toBeChecked());
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: "Start full MLX training" }));
    expect(await screen.findByText("Host lease is already held.")).toBeInTheDocument();

    const compareStage = screen.getAllByRole("button", {
      name: /Compare.*Resolve feasibility/i,
    })[0];
    fireEvent.click(compareStage);
    expect(await screen.findByText("Original bound admission reason.")).toBeInTheDocument();
    expect(screen.queryByText(/latest training launch was rejected/i)).not.toBeInTheDocument();
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
        required_schema: "aptus.training-plan.v5",
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
    expect(screen.getByText("session rejected")).toBeInTheDocument();
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
      schema_version: "aptus.training-plan.v5",
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
    const reportBindings = {
      plan_id: plan.plan_id,
      candidate_id: plan.recommended.candidate_id,
      model_revision: plan.model.revision,
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
      report: { state: "static-pass", findings: [], bindings: reportBindings },
      project_id: projectId,
      project_revision_id: bundleRevisionId,
    });
    validateMock.mockResolvedValue({
      state: "static-pass",
      findings: [],
      bindings: reportBindings,
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
        validation_report: {
          state: "dependency-pass",
          findings: [],
          bindings: reportBindings,
        },
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
        validation_report: {
          state: "model-data-pass",
          findings: [],
          bindings: reportBindings,
        },
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
