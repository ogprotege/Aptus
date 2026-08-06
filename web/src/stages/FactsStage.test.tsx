import { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EXAMPLE_DRAFT } from "../demo";
import type { AptusDesktopBridge } from "../desktopBridge";
import type { MethodDescriptor } from "../types";
import { FactsStage } from "./FactsStage";

const methods: MethodDescriptor[] = [
  {
    schema_version: "aptus.method-descriptor.v1",
    method_id: "lora",
    display_name: "LoRA",
    summary: "Frozen base plus low-rank adapters.",
    lifecycle: "gated-executable",
    selectable: true,
    parameter_scope: "frozen-base-plus-adapter",
    parameterization: "lora",
    base_storage: "unquantized",
    compiler_id: "transformers.peft-lora.v2",
    export_kind: "peft-adapter-safetensors",
    supported_backends: ["cuda", "mps"],
    supported_runtimes: ["transformers-peft-cuda", "mlx-lm"],
    supported_distributions: ["single"],
    evidence_ids: ["method.lora.paper"],
    pilot_requirement: "A bounded pilot is required.",
    runtime_bindings: [
      {
        schema_version: "aptus.runtime-binding.v1",
        training_runtime: "transformers-peft-cuda",
        compute_backend: "cuda",
        compiler_id: "transformers.peft-lora.v2",
        estimator_id: "aptus-memory-v2",
        export_kind: "peft-adapter-safetensors",
        supported_distributions: ["single"],
        evidence_requirement: "pilot-required",
      },
      {
        schema_version: "aptus.runtime-binding.v1",
        training_runtime: "mlx-lm",
        compute_backend: "mps",
        compiler_id: "mlx-lm.lora.v1",
        estimator_id: "aptus-memory-mlx-v2",
        export_kind: "mlx-lm-adapter",
        supported_distributions: ["single"],
        evidence_requirement: "pilot-required",
      },
    ],
  },
  {
    schema_version: "aptus.method-descriptor.v1",
    method_id: "bitfit",
    display_name: "BitFit",
    summary: "Updates an inspected set of existing biases.",
    lifecycle: "experimental",
    selectable: false,
    parameter_scope: "selected-existing-biases",
    parameterization: "bias-only",
    base_storage: "unquantized",
    compiler_id: null,
    export_kind: null,
    supported_backends: [],
    supported_distributions: [],
    evidence_ids: ["method.bitfit.paper"],
    blocker: "The exact architecture may expose no eligible bias tensors.",
    pilot_requirement: "A non-empty trainable census is required.",
    runtime_bindings: [],
  },
];

afterEach(() => {
  delete window.aptusDesktop;
});

function FactsHarness() {
  const [draft, setDraft] = useState(structuredClone(EXAMPLE_DRAFT));
  return (
    <FactsStage
      draft={draft}
      setDraft={setDraft}
      profile={null}
      busy={null}
      demoMode={false}
      onLoadExample={vi.fn()}
      onClearExample={vi.fn()}
      onProfile={vi.fn(async () => undefined)}
      onModelInspect={vi.fn(async () => undefined)}
      onInvalidateModelInspection={vi.fn()}
      onPlan={vi.fn(async () => undefined)}
      onHardwareScan={vi.fn(async () => undefined)}
      hardwareScanned={false}
      modelInspection={null}
      modelPolicyPresentation={null}
      methodCatalog={methods}
    />
  );
}

describe("FactsStage", () => {
  it("shows exact MoE topology and resident-memory truth", () => {
    const draft = structuredClone(EXAMPLE_DRAFT);
    draft.model = {
      ...draft.model,
      family: "qwen3_moe",
      parameters_b: 30.5,
      model_type: "qwen3_moe",
      architecture: "Qwen3MoeForCausalLM",
      quantization_bits: 4,
      active_parameters_b: 3.3,
      sparse_layer_count: 48,
      moe: {
        expert_count: 128,
        experts_per_token: 8,
        expert_intermediate_size: 768,
        decoder_sparse_step: 1,
        mlp_only_layers: [],
        shared_expert_intermediate_size: null,
      },
    };
    draft.hardware.devices[0].backend = "mps";
    draft.target.runtime = "mlx-lm";

    render(
      <FactsStage
        draft={draft}
        setDraft={vi.fn()}
        profile={null}
        busy={null}
        demoMode={false}
        onLoadExample={vi.fn()}
        onClearExample={vi.fn()}
        onProfile={vi.fn(async () => undefined)}
        onModelInspect={vi.fn(async () => undefined)}
        onInvalidateModelInspection={vi.fn()}
        onPlan={vi.fn(async () => undefined)}
        onHardwareScan={vi.fn(async () => undefined)}
        hardwareScanned
        modelInspection={{
          status: "ok",
          model_id: "Qwen/Qwen3-30B-A3B",
          requested_revision: "main",
          resolved_revision: "d".repeat(40),
          facts: {
            family: "qwen3_moe",
            model_type: "qwen3_moe",
            architecture: "Qwen3MoeForCausalLM",
            quantization_layout: {
              default_bits: 4,
              default_group_size: 64,
              module_overrides: [{
                module_path: "model.layers.0.mlp.gate",
                bits: 8,
                group_size: 64,
              }],
            },
          },
          compatibility: {
            status: "conditional",
            family: "qwen3_moe",
            supported_runtime: "mlx-lm",
            compute_backend: "mps",
            supported_methods: ["qlora"],
            distribution: "single",
            evidence_requirement: "pilot-required",
            adapter_profile_id: "attention-qkvo.v1",
            reason: "Exact model-data and pilot evidence are required.",
          },
        }}
        modelPolicyPresentation={null}
        methodCatalog={methods}
      />,
    );

    expect(screen.getByRole("heading", { name: "Pinned MoE topology" })).toBeInTheDocument();
    expect(screen.getByText("Any 8 of 128 routed experts")).toBeInTheDocument();
    expect(screen.getByText(/router selects any 8 of 128 routed experts for each token/i)).toBeInTheDocument();
    expect(screen.getByText("30.5B")).toBeInTheDocument();
    expect(screen.getByText("3.3B")).toBeInTheDocument();
    expect(screen.getByText("4-bit group 64; 1 override")).toBeInTheDocument();
    expect(screen.getByText(/all checkpoint weights must remain resident/i)).toBeInTheDocument();
  });

  it("clears stale provider topology when an inspected model fact changes", async () => {
    const invalidate = vi.fn();

    function MoEHarness() {
      const [draft, setDraft] = useState(() => {
        const next = structuredClone(EXAMPLE_DRAFT);
        next.model = {
          ...next.model,
          family: "qwen3_moe",
          model_type: "qwen3_moe",
          architecture: "Qwen3MoeForCausalLM",
          quantization_bits: 4,
          quantization_layout: {
            default_bits: 4,
            default_group_size: 64,
            module_overrides: [],
          },
          active_parameters_b: 3.3,
          sparse_layer_count: 48,
          moe: {
            expert_count: 128,
            experts_per_token: 8,
            expert_intermediate_size: 768,
            decoder_sparse_step: 1,
            mlp_only_layers: [],
          },
        };
        return next;
      });
      return (
        <>
          <FactsStage
            draft={draft}
            setDraft={setDraft}
            profile={null}
            busy={null}
            demoMode={false}
            onLoadExample={vi.fn()}
            onClearExample={vi.fn()}
            onProfile={vi.fn(async () => undefined)}
            onModelInspect={vi.fn(async () => undefined)}
            onInvalidateModelInspection={invalidate}
            onPlan={vi.fn(async () => undefined)}
            onHardwareScan={vi.fn(async () => undefined)}
            hardwareScanned={false}
            modelInspection={null}
            modelPolicyPresentation={null}
            methodCatalog={methods}
          />
          <output data-testid="quantization-layout-state">
            {JSON.stringify(draft.model.quantization_layout)}
          </output>
        </>
      );
    }

    render(<MoEHarness />);
    expect(screen.getByRole("heading", { name: "Pinned MoE topology" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Total resident parameters"), {
      target: { value: "31" },
    });

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Pinned MoE topology" })).toBeInTheDocument();
      expect(screen.queryByText("3.3B")).not.toBeInTheDocument();
      expect(screen.getAllByText("Derived during planning")).toHaveLength(2);
    });
    expect(screen.getByLabelText("Architecture family")).toHaveValue("qwen3_moe");
    expect(screen.getByTestId("quantization-layout-state")).not.toHaveTextContent("null");
    expect(invalidate).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText("Architecture family"), {
      target: { value: "llama" },
    });

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: "Pinned MoE topology" })).not.toBeInTheDocument();
      expect(screen.getByTestId("quantization-layout-state")).toHaveTextContent("null");
    });
    expect(invalidate).toHaveBeenCalledOnce();
  });

  it("uses the API method registry and explains Apple unified memory", () => {
    const draft = structuredClone(EXAMPLE_DRAFT);
    draft.hardware.devices[0] = {
      ...draft.hardware.devices[0],
      name: "Apple M5 Pro (shared unified memory)",
      backend: "mps",
      total_vram_gib: 64,
      free_vram_gib: 48,
      supports_bf16: false,
      supports_8bit: false,
      supports_4bit: false,
    };
    draft.target.runtime = "mlx-lm";
    render(
      <FactsStage
        draft={draft}
        setDraft={vi.fn()}
        profile={null}
        busy={null}
        demoMode={false}
        onLoadExample={vi.fn()}
        onClearExample={vi.fn()}
        onProfile={vi.fn(async () => undefined)}
        onModelInspect={vi.fn(async () => undefined)}
        onInvalidateModelInspection={vi.fn()}
        onPlan={vi.fn(async () => undefined)}
        onHardwareScan={vi.fn(async () => undefined)}
        hardwareScanned
        modelInspection={null}
        modelPolicyPresentation={null}
        methodCatalog={methods}
      />,
    );

    expect(
      screen.getByRole("option", { name: "Prefer LoRA if feasible" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /BitFit/ })).not.toBeInTheDocument();
    const readinessSummary = screen.getByText("Inspect method readiness");
    const readinessDetails = readinessSummary.closest("details");
    expect(readinessDetails).not.toHaveAttribute("open");
    fireEvent.click(readinessSummary);
    expect(readinessDetails).toHaveAttribute("open");
    expect(screen.getByText(/1 compiler paths · 1 available on MPS/i)).toBeInTheDocument();
    expect(screen.getByText("Executable behind gates")).toBeInTheDocument();
    expect(screen.getByText("BitFit")).toBeInTheDocument();
    expect(screen.getByText("Experimental")).toBeInTheDocument();
    expect(screen.getByText(/shares one memory pool/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Unified memory pool")).toHaveValue(64);
    expect(screen.getByLabelText("Measured memory headroom")).toHaveValue(48);
    expect(screen.getByText("Apple capability rules are runtime-specific.")).toBeInTheDocument();
    expect(screen.getByLabelText("Bundle runtime")).toHaveValue("mlx-lm");
    expect(
      screen.getByRole("option", {
        name: /PyTorch MPS · known compatibility runtime, compiler unavailable/i,
      }),
    ).toBeDisabled();
    expect(screen.getByText(/cannot be selected until Aptus ships a compiler binding/i)).toBeInTheDocument();
    expect(screen.queryByLabelText("BF16 supported")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("8-bit backend supported")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("4-bit backend supported")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Choose file" })).not.toBeInTheDocument();
  });

  it("uses the native data picker only in the desktop host", async () => {
    const bridge: AptusDesktopBridge = {
      platform: "macos",
      reportWorkbenchReady: vi.fn(async () => undefined),
      pickDataset: vi.fn(async () => "/Users/wilson/training.jsonl"),
      pickOutputDirectory: vi.fn(async () => null),
      revealInFinder: vi.fn(async () => undefined),
    };
    window.aptusDesktop = bridge;
    render(<FactsHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Choose file" }));

    await waitFor(() => {
      expect(screen.getByLabelText("Dataset path")).toHaveValue("/Users/wilson/training.jsonl");
    });
    expect(bridge.pickDataset).toHaveBeenCalledOnce();
    expect(screen.getByText(/Choose a local JSONL, JSON, CSV, or text file/i)).toBeInTheDocument();
  });

  it("keeps the current data path when the native picker is cancelled", async () => {
    const bridge: AptusDesktopBridge = {
      platform: "macos",
      reportWorkbenchReady: vi.fn(async () => undefined),
      pickDataset: vi.fn(async () => null),
      pickOutputDirectory: vi.fn(async () => null),
      revealInFinder: vi.fn(async () => undefined),
    };
    window.aptusDesktop = bridge;
    render(<FactsHarness />);
    const originalPath = EXAMPLE_DRAFT.dataset.source_path;

    fireEvent.click(screen.getByRole("button", { name: "Choose file" }));

    await waitFor(() => expect(bridge.pickDataset).toHaveBeenCalledOnce());
    expect(screen.getByLabelText("Dataset path")).toHaveValue(originalPath);
  });
});
