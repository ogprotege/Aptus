import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EXAMPLE_PLAN } from "../demo";
import { CandidateComparison } from "./CandidateComparison";

describe("CandidateComparison", () => {
  it("uses a semantic table and lets the user inspect an alternative", () => {
    const onInspect = vi.fn();
    render(
      <CandidateComparison
        candidates={EXAMPLE_PLAN.candidates}
        recommended={EXAMPLE_PLAN.recommended}
        inspected={EXAMPLE_PLAN.recommended}
        onInspect={onInspect}
      />,
    );
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText(/Compilation still uses the recommendation/i)).toBeInTheDocument();
    const qloraButtons = screen.getAllByRole("button", { name: /Inspect QLoRA candidate evidence/i });
    fireEvent.click(qloraButtons[0]);
    expect(onInspect).toHaveBeenCalledWith(
      EXAMPLE_PLAN.candidates.find(
        (candidate) => candidate.method === "qlora" && candidate.distribution === "single",
      ),
    );
  });

  it("shows required host RAM and disk for every candidate in GiB", () => {
    const GiB = 1024 ** 3;
    const candidate = {
      ...EXAMPLE_PLAN.candidates[0],
      id: "capacity-candidate",
      candidate_id: "capacity-candidate",
      required_host_ram_bytes: 16 * GiB,
      required_disk_bytes: 32 * GiB,
    };
    render(
      <CandidateComparison
        candidates={[candidate]}
        recommended={candidate}
        inspected={candidate}
        onInspect={vi.fn()}
      />,
    );

    expect(screen.getByRole("columnheader", { name: "Host RAM required" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Disk required" })).toBeInTheDocument();
    expect(screen.getAllByText("16.0 GiB")).toHaveLength(2);
    expect(screen.getAllByText("32.0 GiB")).toHaveLength(2);
  });
});
