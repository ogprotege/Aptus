import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ValidateStage } from "./ValidateStage";
import type { CompileResponse, ValidationReport } from "../types";

describe("ValidateStage", () => {
  it("renders attestation bindings and both pilot phases", () => {
    const report: ValidationReport = {
      state: "pilot-pass",
      validation_level: "pilot",
      validator_version: "aptus-portable-validator-v2",
      validated_at: "2026-07-21T12:00:00+00:00",
      findings: [],
      bindings: {
        plan_id: "plan_bound",
        pilot_metrics: "metrics_digest",
      },
      preflight_metrics: {
        schema_version: "aptus.preflight-metrics.v1",
        candidate_id: "cand_bound",
        method: "qlora",
        precision: "bf16",
        quantization: "nf4-double-quant",
        distribution: "single",
        world_size: 1,
        measured_peak_cuda_bytes: 2 * 1024 ** 3,
        scope: "synthetic-method-preflight-not-model-data-pilot",
      },
      pilot_metrics: {
        checkpoint_continuation_observed: true,
        measured_checkpoint_bytes: 1024 ** 3,
        phase_one_checkpoint: {
          total_bytes: 1024 ** 3,
          manifest_sha256: "phase-one-digest",
          files: [{ path: "optimizer.pt" }],
        },
        phase_two_checkpoint: {
          total_bytes: 2 * 1024 ** 3,
          manifest_sha256: "phase-two-digest",
          files: [{ path: "optimizer.pt" }, { path: "trainer_state.json" }],
        },
        phase_one: { global_step: 1, train_loss: 2.4, measured_peak_cuda_bytes: 100 },
        phase_two_resumed: { global_step: 2, train_loss: 2.3, measured_peak_cuda_bytes: 110 },
      },
    };
    const bundle: CompileResponse = {
      bundle_dir: "/tmp/bundle",
      files: [],
      report,
    };

    render(
      <ValidateStage
        bundle={bundle}
        report={report}
        busy={null}
        demoMode={false}
        onValidate={vi.fn(async () => undefined)}
        onOpenRun={vi.fn()}
        onReturnToCompile={vi.fn()}
        validationLevel="static"
        onValidationLevelChange={vi.fn()}
      />,
    );

    expect(screen.getByText("Attestation bound to this artifact")).toBeInTheDocument();
    expect(screen.getByText("aptus-portable-validator-v2")).toBeInTheDocument();
    expect(screen.getByText("plan_bound")).toBeInTheDocument();
    expect(screen.getByText("Phase 1")).toBeInTheDocument();
    expect(screen.getByText("Phase 2 resumed")).toBeInTheDocument();
    expect(screen.getByText("Yes")).toBeInTheDocument();
    expect(screen.getByText("phase-one-digest")).toBeInTheDocument();
    expect(screen.getByText("phase-two-digest")).toBeInTheDocument();
    expect(screen.getByText("Measured synthetic preflight")).toBeInTheDocument();
    expect(screen.getByText("cand_bound")).toBeInTheDocument();
    expect(screen.getByText("nf4-double-quant")).toBeInTheDocument();
    expect(screen.getAllByText("2.00 GiB").length).toBeGreaterThan(0);
  });

  it("opens Run from static-pass so the dependency gate can run", () => {
    const onOpenRun = vi.fn();
    const report: ValidationReport = { state: "static-pass", findings: [] };
    render(
      <ValidateStage
        bundle={{ bundle_dir: "/tmp/bundle", files: [], report }}
        report={report}
        busy={null}
        demoMode={false}
        onValidate={vi.fn(async () => undefined)}
        onOpenRun={onOpenRun}
        onReturnToCompile={vi.fn()}
        validationLevel="static"
        onValidationLevelChange={vi.fn()}
      />,
    );

    const button = screen.getByRole("button", { name: "Open run actions" });
    expect(button).toBeEnabled();
    fireEvent.click(button);
    expect(onOpenRun).toHaveBeenCalledOnce();
  });
});
