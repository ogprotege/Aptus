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

  it("renders the uninterrupted MLX pilot contract without implying resume", () => {
    const report: ValidationReport = {
      state: "pilot-pass",
      validation_level: "pilot",
      validator_version: "aptus-validator-mlx-v1",
      validated_at: "2026-07-22T12:00:00+00:00",
      findings: [],
      bindings: {
        plan_id: "plan_mlx",
        pilot_metrics: "mlx-metrics-digest",
      },
      preflight_metrics: {
        schema_version: "aptus.runtime-metrics.v1",
        candidate_id: "cand_mlx_qlora",
        method: "qlora",
        precision: "bf16",
        quantization: "mlx-int4",
        distribution: "single",
        world_size: 1,
        training_runtime: "mlx-lm",
        measured_peak_bytes: 3 * 1024 ** 3,
        memory_metric_backend: "mlx",
        scope: "bounded-compiler-smoke-not-pilot-evidence",
      },
      pilot_metrics: {
        training_runtime: "mlx-lm",
        compute_backend: "mps",
        scope: "uninterrupted-pilot",
        action: "pilot",
        execution_semantics: "uninterrupted",
        resume_supported: false,
        run_id: "pilot_mlx_001",
        completed_optimizer_updates: 2,
        finite_train_loss: true,
        finite_validation_loss: true,
        measured_peak_bytes: 4 * 1024 ** 3,
        memory_metric_backend: "mlx",
        unified_memory_admission: {
          available_unified_memory_bytes: 48 * 1024 ** 3,
          required_available_bytes: 16 * 1024 ** 3,
          reserve_bytes: 8 * 1024 ** 3,
        },
        trainable_target_binding: {
          expected_adapter_target_instance_count: 24,
          adapter_target_instance_count: 24,
          trainable_tensor_count: 48,
        },
        adapter_manifest: [
          { path: "adapter_config.json", size_bytes: 512, sha256: "config-digest" },
          { path: "adapters.safetensors", size_bytes: 4096, sha256: "adapter-digest" },
        ],
        reload_evidence: {
          fresh_process_observed: true,
          generation_max_tokens: 4,
          generation_tokens: 3,
          execution_semantics: "uninterrupted",
          resume_supported: false,
        },
      },
    };
    const bundle: CompileResponse = {
      bundle_dir: "/tmp/mlx-bundle",
      files: [],
      runtime_contract: {
        schema_version: "aptus.runtime-contract.v1",
        compute_backend: "mps",
        training_runtime: "mlx-lm",
        compiler_id: "mlx-lm.qlora.v1",
        estimator_id: "aptus-memory-mlx-v2",
        evidence_requirement: "pilot-required",
        export_kind: "mlx-lm-adapter",
      },
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

    expect(screen.getByText("Uninterrupted from scratch")).toBeInTheDocument();
    expect(screen.getByText("Unavailable for this runtime")).toBeInTheDocument();
    expect(screen.queryByText("Checkpoint continuation observed")).not.toBeInTheDocument();
    expect(screen.getByText("Bounded MLX preflight smoke")).toBeInTheDocument();
    expect(screen.getByText("Uninterrupted MLX pilot")).toBeInTheDocument();
    expect(screen.getByText("24 / 24 adapter instances")).toBeInTheDocument();
    expect(screen.getByText("2 manifest-bound files")).toBeInTheDocument();
    expect(screen.getAllByText("Verified").length).toBeGreaterThanOrEqual(3);
    expect(screen.getByText("48.0 GiB")).toBeInTheDocument();
    expect(screen.getByText("16.0 GiB")).toBeInTheDocument();
    expect(screen.getByText("8.00 GiB")).toBeInTheDocument();
    expect(screen.getByText("pilot_mlx_001")).toBeInTheDocument();
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
