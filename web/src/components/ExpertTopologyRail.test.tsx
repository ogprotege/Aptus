import axe from "axe-core";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ExpertTopologyRail } from "./ExpertTopologyRail";

const TOPOLOGY = {
  expert_count: 128,
  experts_per_token: 8,
  expert_intermediate_size: 768,
  decoder_sparse_step: 1,
  mlp_only_layers: [],
  shared_expert_intermediate_size: 768,
};

describe("ExpertTopologyRail", () => {
  it("exposes the pinned topology without accessibility violations", async () => {
    const { container } = render(
      <ExpertTopologyRail
        topology={TOPOLOGY}
        totalParametersB={30.5}
        activeParametersB={3.3}
        sparseLayerCount={48}
        quantizationBits={4}
      />,
    );

    const results = await axe.run(container, {
      rules: { "color-contrast": { enabled: false } },
    });
    expect(results.violations).toEqual([]);
  });

  it("keeps resident weights separate from routed per-token activity", () => {
    render(
      <ExpertTopologyRail
        topology={TOPOLOGY}
        totalParametersB={30.5}
        activeParametersB={3.3}
        sparseLayerCount={48}
        quantizationBits={4}
      />,
    );

    expect(screen.getByRole("heading", { name: "Pinned MoE topology" })).toBeInTheDocument();
    expect(screen.getByText("30.5B")).toBeInTheDocument();
    expect(screen.getByText("3.3B")).toBeInTheDocument();
    expect(screen.getByText(/all checkpoint weights must remain resident/i)).toBeInTheDocument();
    expect(screen.getByText(/active parameters.*never reduce the base-weight memory budget/i)).toBeInTheDocument();
  });

  it("describes routed and shared expert execution without making a policy claim", () => {
    render(
      <ExpertTopologyRail
        topology={TOPOLOGY}
        totalParametersB={30.5}
        activeParametersB={null}
        sparseLayerCount={null}
        quantizationBits={4}
      />,
    );

    expect(screen.getAllByText(/any 8 of 128 routed experts/i)).toHaveLength(2);
    expect(screen.getByText(/shared expert path also runs for each token/i)).toBeInTheDocument();
    expect(screen.queryByText(/eligible for the reviewed pilot path/i)).not.toBeInTheDocument();
  });
});
