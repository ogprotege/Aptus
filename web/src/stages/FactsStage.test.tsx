import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EXAMPLE_DRAFT } from "../demo";
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
    supported_backends: ["cuda"],
    supported_distributions: ["single"],
    evidence_ids: ["method.lora.paper"],
    pilot_requirement: "A bounded pilot is required.",
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
  },
];

describe("FactsStage", () => {
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
    expect(screen.getByText(/1 compiler paths · 0 available on MPS/i)).toBeInTheDocument();
    expect(screen.getByText("Unavailable on MPS")).toBeInTheDocument();
    expect(screen.getByText("BitFit")).toBeInTheDocument();
    expect(screen.getByText("Experimental")).toBeInTheDocument();
    expect(screen.getByText(/shares one unified-memory pool/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Unified memory pool")).toHaveValue(64);
    expect(screen.getByLabelText("Available unified memory")).toHaveValue(48);
    expect(screen.getByText("MLX capabilities are not measured yet.")).toBeInTheDocument();
    expect(screen.queryByLabelText("BF16 supported")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("8-bit backend supported")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("4-bit backend supported")).not.toBeInTheDocument();
  });
});
