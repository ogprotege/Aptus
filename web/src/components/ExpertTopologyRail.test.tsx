import axe from "axe-core";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MALFORMED_COMPATIBILITY_REASON } from "../lib/modelCompatibility";
import type { ModelCompatibility } from "../types";
import { ExpertTopologyRail } from "./ExpertTopologyRail";

const CONDITIONAL_COMPATIBILITY = {
  status: "conditional",
  family: "qwen3_moe",
  supported_runtime: "mlx-lm",
  compute_backend: "mps",
  supported_methods: ["qlora"],
  distribution: "single",
  evidence_requirement: "pilot-required",
  adapter_profile_id: "attention-qkvo.v1",
  reason:
    "The model identity, mixed-precision layout, routed-expert topology, and attention-only q/k/v/o target policy match the reviewed Qwen3 MoE slice. Measured preflight and a real-model pilot remain mandatory.",
} satisfies ModelCompatibility;

const TOPOLOGY = {
  expert_count: 128,
  experts_per_token: 8,
  expert_intermediate_size: 768,
  decoder_sparse_step: 1,
  mlp_only_layers: [],
  shared_expert_intermediate_size: 768,
};

describe("ExpertTopologyRail", () => {
  it("exposes the topology and path eligibility without accessibility violations", async () => {
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
          compute_backend: "mps",
          supported_methods: ["qlora"],
          distribution: "single",
          evidence_requirement: "pilot-required",
          adapter_profile_id: "attention-qkvo.v1",
          reason: "Exact pilot evidence is required.",
        }}
        selectedRuntime="mlx-lm"
        selectedBackend="mps"
      />,
    );

    const results = await axe.run(container, {
      rules: { "color-contrast": { enabled: false } },
    });
    expect(results.violations).toEqual([]);
  });

  it("states the exact reviewed path and evidence boundary when the selected target matches", () => {
    render(
      <ExpertTopologyRail
        topology={TOPOLOGY}
        totalParametersB={30.5}
        activeParametersB={3.3}
        sparseLayerCount={48}
        quantizationBits={4}
        compatibility={CONDITIONAL_COMPATIBILITY}
        selectedRuntime="mlx-lm"
        selectedBackend="mps"
      />,
    );

    const copy = screen.getByText(/eligible for the reviewed pilot path/i);
    expect(copy.textContent).toBe(
      "This artifact is eligible for the reviewed pilot path: runtime mlx-lm, "
      + "backend mps, method qlora, placement single, adapter profile attention-qkvo.v1. "
      + "Evidence requirement: pilot-required. "
      + CONDITIONAL_COMPATIBILITY.reason,
    );
  });

  it("states the full reviewed path and selected target when the runtime differs", () => {
    render(
      <ExpertTopologyRail
        topology={TOPOLOGY}
        totalParametersB={30.5}
        activeParametersB={3.3}
        sparseLayerCount={48}
        quantizationBits={4}
        compatibility={CONDITIONAL_COMPATIBILITY}
        selectedRuntime="transformers-peft-cuda"
        selectedBackend="cuda"
      />,
    );

    const copy = screen.getByText(/reviewed pilot path requires/i);
    expect(copy.textContent).toBe(
      "The reviewed pilot path requires runtime mlx-lm, backend mps, method qlora, "
      + "placement single, and adapter profile attention-qkvo.v1. "
      + "The selected target uses runtime transformers-peft-cuda and backend cuda; "
      + "it does not match this path. Evidence requirement: pilot-required. "
      + CONDITIONAL_COMPATIBILITY.reason,
    );
  });

  it("fails the selected-target match when only the backend differs", () => {
    render(
      <ExpertTopologyRail
        topology={TOPOLOGY}
        totalParametersB={30.5}
        activeParametersB={3.3}
        sparseLayerCount={48}
        quantizationBits={4}
        compatibility={CONDITIONAL_COMPATIBILITY}
        selectedRuntime="mlx-lm"
        selectedBackend="cuda"
      />,
    );

    expect(screen.getByText("Target mismatch")).toBeInTheDocument();
    expect(screen.getByText(/reviewed pilot path requires/i).textContent).toBe(
      "The reviewed pilot path requires runtime mlx-lm, backend mps, method qlora, "
      + "placement single, and adapter profile attention-qkvo.v1. "
      + "The selected target uses runtime mlx-lm and backend cuda; "
      + "it does not match this path. Evidence requirement: pilot-required. "
      + CONDITIONAL_COMPATIBILITY.reason,
    );
  });

  it("fails closed when conditional evidence is incomplete or contradictory", () => {
    const malformedCompatibility = {
      status: "conditional",
      family: "qwen3_moe",
      supported_runtime: null,
      compute_backend: "mps",
      supported_methods: ["qlora"],
      distribution: null,
      evidence_requirement: "implementation-required",
      adapter_profile_id: "attention-qkvo.v1",
      reason: "Decoy evidence mentions mlx-lm, qlora, and single.",
    } as unknown as ModelCompatibility;

    render(
      <ExpertTopologyRail
        topology={TOPOLOGY}
        totalParametersB={30.5}
        activeParametersB={3.3}
        sparseLayerCount={48}
        quantizationBits={4}
        compatibility={malformedCompatibility}
        selectedRuntime="mlx-lm"
        selectedBackend="mps"
      />,
    );

    expect(screen.getByText("Unsupported")).toBeInTheDocument();
    expect(screen.getByText(MALFORMED_COMPATIBILITY_REASON)).toBeInTheDocument();
    expect(
      screen.queryByText(/eligible for the reviewed pilot path|reviewed pilot path requires/i),
    ).not.toBeInTheDocument();
  });
});
