import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EXAMPLE_PLAN } from "../demo";
import { FitLedger } from "./FitLedger";

describe("FitLedger", () => {
  it("names the fit claim and exposes component values as text", () => {
    render(<FitLedger candidate={EXAMPLE_PLAN.recommended} example />);
    expect(screen.getByRole("heading", { name: "The Fit Ledger" })).toBeInTheDocument();
    expect(screen.getByText(/headroom/i)).toBeInTheDocument();
    expect(screen.getByText("Base weights")).toBeInTheDocument();
    expect(screen.getByText(/No hardware inspection ran/i)).toBeInTheDocument();
    expect(screen.getByText("Per-device VRAM feasibility")).toBeInTheDocument();
    expect(screen.getByText(/usable per-device VRAM/i)).toBeInTheDocument();
  });

  it("uses unified-memory language for an MLX-LM candidate", () => {
    const mlxCandidate = {
      ...EXAMPLE_PLAN.recommended!,
      runtime_contract: {
        schema_version: "aptus.runtime-contract.v1",
        compute_backend: "mlx",
        training_runtime: "mlx-lm",
        compiler_id: "aptus-mlx-lm-v1",
        estimator_id: "aptus-memory-mlx-v1",
        evidence_requirement: "measured-pilot",
        export_kind: "mlx-adapter",
      },
    };

    render(<FitLedger candidate={mlxCandidate} />);

    expect(screen.getByText("Unified-memory feasibility")).toBeInTheDocument();
    expect(screen.getByText(/usable unified-memory headroom/i)).toBeInTheDocument();
    expect(screen.getByText("Aptus reserve")).toBeInTheDocument();
    expect(screen.queryByText(/usable per-device VRAM/i)).not.toBeInTheDocument();
  });

  it("provides a directed empty state", () => {
    render(<FitLedger candidate={null} />);
    expect(screen.getByText(/Compare strategies to calculate memory fit/i)).toBeInTheDocument();
  });
});
