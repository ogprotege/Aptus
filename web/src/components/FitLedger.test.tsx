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
  });

  it("provides a directed empty state", () => {
    render(<FitLedger candidate={null} />);
    expect(screen.getByText(/Compare strategies to calculate memory fit/i)).toBeInTheDocument();
  });
});
