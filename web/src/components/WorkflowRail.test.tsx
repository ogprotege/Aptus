import axe from "axe-core";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { WorkflowRail } from "./WorkflowRail";

function renderRail(overrides: Partial<Parameters<typeof WorkflowRail>[0]> = {}) {
  return render(
    <WorkflowRail
      current="compare"
      completed={new Set(["facts"])}
      projectName="Customer-support adapter"
      connection="connected"
      serviceVersion="1"
      projects={[]}
      currentProject={null}
      projectHistory={[]}
      onRecoverProject={vi.fn(async () => {})}
      onSelect={vi.fn()}
      {...overrides}
    />,
  );
}

describe("WorkflowRail", () => {
  it("announces completed and current stage state to assistive technology", () => {
    renderRail();
    const facts = screen.getByRole("button", { name: /Facts.*Complete\./ });
    expect(facts).toBeInTheDocument();
    const compare = screen.getByRole("button", { name: /Compare/ });
    expect(compare).toHaveAttribute("aria-current", "step");
    expect(compare).not.toHaveAccessibleName(/Complete\./);
  });

  it("announces a failed run stage as needing attention", () => {
    renderRail({ current: "run", runState: "failed" });
    const run = screen.getByRole("button", { name: /Run.*Needs attention\./ });
    expect(run).toBeInTheDocument();
  });

  it("has no accessibility violations", async () => {
    const { container } = renderRail();
    const results = await axe.run(container, {
      rules: { "color-contrast": { enabled: false } },
    });
    expect(results.violations).toEqual([]);
  });
});
