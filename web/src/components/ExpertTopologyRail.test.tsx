import axe from "axe-core";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ModelCompatibility } from "../types";
import { ExpertTopologyRail } from "./ExpertTopologyRail";

const CONDITIONAL_COMPATIBILITY = {
  status: "conditional",
  family: "qwen3_moe",
  supported_runtime: "mlx-lm",
  supported_methods: ["qlora"],
  distribution: "single",
  evidence_requirement: "pilot-required",
  adapter_scope: "attention-only",
  reason:
    "This exact mixed-precision Qwen3 MoE artifact can enter the single-device MLX-LM QLoRA path with attention-only adapters. Measured preflight and a real-model pilot remain mandatory.",
} satisfies ModelCompatibility;

const TOPOLOGY = {
  expert_count: 128,
  experts_per_token: 8,
  expert_intermediate_size: 768,
  decoder_sparse_step: 1,
  mlp_only_layers: [],
  shared_expert_intermediate_size: 768,
};

function expectConditionalEvidence(copy: HTMLElement) {
  expect(copy.textContent).toContain(CONDITIONAL_COMPATIBILITY.supported_runtime);
  for (const method of CONDITIONAL_COMPATIBILITY.supported_methods) {
    expect(copy.textContent).toContain(method);
  }
  expect(copy.textContent).toContain(CONDITIONAL_COMPATIBILITY.distribution);
  expect(copy.textContent).toContain(CONDITIONAL_COMPATIBILITY.reason);
}

describe("ExpertTopologyRail", () => {
  it("exposes the topology and support state without accessibility violations", async () => {
    const { container } = render(
      <ExpertTopologyRail
        topology={{
          expert_count: 128,
          experts_per_token: 8,
          expert_intermediate_size: 768,
          decoder_sparse_step: 1,
          mlp_only_layers: [],
          shared_expert_intermediate_size: 768,
        }}
        totalParametersB={30.5}
        activeParametersB={3.3}
        sparseLayerCount={48}
        quantizationBits={4}
        compatibility={{
          status: "conditional",
          family: "qwen3_moe",
          supported_runtime: "mlx-lm",
          supported_methods: ["qlora"],
          distribution: "single",
          evidence_requirement: "pilot-required",
          reason: "Exact pilot evidence is required.",
        }}
        selectedRuntime="mlx-lm"
      />,
    );

    const results = await axe.run(container, {
      rules: { "color-contrast": { enabled: false } },
    });
    expect(results.violations).toEqual([]);
  });

  it("states the runtime, method, placement, and pilot boundary when the target runtime matches", () => {
    render(
      <ExpertTopologyRail
        topology={TOPOLOGY}
        totalParametersB={30.5}
        activeParametersB={3.3}
        sparseLayerCount={48}
        quantizationBits={4}
        compatibility={CONDITIONAL_COMPATIBILITY}
        selectedRuntime="mlx-lm"
      />,
    );

    const copy = screen.getByText(/mlx-lm supports/i);
    expectConditionalEvidence(copy);
  });

  it("states the runtime, method, placement, and pilot boundary when the target runtime differs", () => {
    render(
      <ExpertTopologyRail
        topology={TOPOLOGY}
        totalParametersB={30.5}
        activeParametersB={3.3}
        sparseLayerCount={48}
        quantizationBits={4}
        compatibility={CONDITIONAL_COMPATIBILITY}
        selectedRuntime="transformers-peft-cuda"
      />,
    );

    const copy = screen.getByText(/conditional path requires/i);
    expectConditionalEvidence(copy);
    expect(copy.textContent).toContain("transformers-peft-cuda");
  });
});
