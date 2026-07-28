import axe from "axe-core";
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ExpertTopologyRail } from "./ExpertTopologyRail";

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
});
