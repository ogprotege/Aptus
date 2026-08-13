import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RunStage } from "./RunStage";
import type { AptusDesktopBridge } from "../desktopBridge";
import type { CompileResponse, Job, ValidationReport } from "../types";

const reportBinding = {
  planId: "plan_aaaaaaaaaaaaaaaaaaaa",
  candidateId: "candidate-bound",
  modelRevision: "a".repeat(40),
};
const reportBindings = {
  plan_id: reportBinding.planId,
  candidate_id: reportBinding.candidateId,
  model_revision: reportBinding.modelRevision,
};
const callbacks = {
  reportBinding,
  onCreateJob: vi.fn(async () => undefined),
  onRefreshJob: vi.fn(async () => undefined),
  onCancelJob: vi.fn(async () => undefined),
  onReturnToValidate: vi.fn(),
};

const bundle: CompileResponse = { bundle_dir: "/tmp/bundle", files: [] };
const mlxBundle: CompileResponse = {
  ...bundle,
  runtime_contract: {
    schema_version: "aptus.runtime-contract.v1",
    compute_backend: "mps",
    training_runtime: "mlx-lm",
    compiler_id: "mlx-lm.lora.v1",
    estimator_id: "aptus-memory-mlx-v2",
    evidence_requirement: "pilot-required",
    export_kind: "mlx-lm-adapter",
  },
};
const pilotReport: ValidationReport = {
  state: "pilot-pass",
  findings: [],
  bindings: reportBindings,
  authorization_status: "blocked",
  authorization_current: false,
  authorization_error: "Cached authorization is stale.",
};

afterEach(() => {
  delete window.aptusDesktop;
});

describe("RunStage", () => {
  it("hands every desktop execution plan off to the CUDA host", () => {
    const bridge: AptusDesktopBridge = {
      platform: "macos",
      reportWorkbenchReady: vi.fn(async () => undefined),
      pickDataset: vi.fn(async () => null),
      pickOutputDirectory: vi.fn(async () => null),
      revealInFinder: vi.fn(async () => undefined),
    };
    window.aptusDesktop = bridge;
    render(
      <RunStage
        bundle={bundle}
        report={pilotReport}
        job={null}
        busy={null}
        demoMode={false}
        {...callbacks}
      />,
    );

    expect(screen.getByText("Continue on the CUDA machine.")).toBeInTheDocument();
    expect(screen.getByText(/never submits CUDA work locally/i)).toBeInTheDocument();
    expect(screen.getByLabelText("CUDA host commands")).toHaveTextContent("--action pilot");
    expect(screen.queryByRole("radio")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start training" })).not.toBeInTheDocument();
    expect(screen.queryByText("No job has started.")).not.toBeInTheDocument();
  });

  it("keeps an MLX-LM bundle executable inside the Mac app", async () => {
    window.aptusDesktop = {
      platform: "macos",
      reportWorkbenchReady: vi.fn(async () => undefined),
      pickDataset: vi.fn(async () => null),
      pickOutputDirectory: vi.fn(async () => null),
      revealInFinder: vi.fn(async () => undefined),
    };
    render(
      <RunStage
        bundle={mlxBundle}
        report={{ state: "static-pass", findings: [], bindings: reportBindings }}
        job={null}
        busy={null}
        demoMode={false}
        {...callbacks}
      />,
    );

    expect(screen.queryByText("Continue on the CUDA machine.")).not.toBeInTheDocument();
    expect(screen.getByText(/mlx-lm runtime on this Mac/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Dependency check/i)).toBeChecked();
    expect(screen.getByRole("button", { name: "Check dependencies" })).toBeEnabled();
    expect(screen.getByRole("note")).toHaveTextContent("DoRA, full-parameter training, and resume are not supported");
  });

  it("runs the MLX pilot as uninterrupted evidence instead of a continuation diagnostic", async () => {
    const onCreateJob = vi.fn(async () => undefined);
    window.aptusDesktop = {
      platform: "macos",
      reportWorkbenchReady: vi.fn(async () => undefined),
      pickDataset: vi.fn(async () => null),
      pickOutputDirectory: vi.fn(async () => null),
      revealInFinder: vi.fn(async () => undefined),
    };
    render(
      <RunStage
        bundle={mlxBundle}
        report={{ state: "measured-preflight-pass", findings: [], bindings: reportBindings }}
        job={null}
        busy={null}
        demoMode={false}
        {...callbacks}
        onCreateJob={onCreateJob}
      />,
    );

    await waitFor(() => expect(screen.getByLabelText(/Uninterrupted MLX pilot/i)).toBeChecked());
    expect(screen.getByText(/at least two optimizer updates/i)).toBeInTheDocument();
    expect(screen.getByText(/does not create a resume point/i)).toBeInTheDocument();
    expect(screen.queryByText(/expected to stop/i)).not.toBeInTheDocument();
    const pilotButton = screen.getByRole("button", { name: "Run uninterrupted pilot" });
    expect(pilotButton).toBeEnabled();
    fireEvent.click(pilotButton);
    expect(onCreateJob).toHaveBeenCalledWith("pilot");
  });

  it("enables an explicitly confirmed full MLX run from scratch after pilot-pass", async () => {
    const onCreateJob = vi.fn(async () => undefined);
    window.aptusDesktop = {
      platform: "macos",
      reportWorkbenchReady: vi.fn(async () => undefined),
      pickDataset: vi.fn(async () => null),
      pickOutputDirectory: vi.fn(async () => null),
      revealInFinder: vi.fn(async () => undefined),
    };
    render(
      <RunStage
        bundle={mlxBundle}
        report={{ state: "pilot-pass", findings: [], bindings: reportBindings }}
        job={null}
        busy={null}
        demoMode={false}
        {...callbacks}
        onCreateJob={onCreateJob}
      />,
    );

    await waitFor(() => expect(screen.getByLabelText(/Training job/i)).toBeChecked());
    expect(screen.getByText(/confirm a full-duration MLX LoRA or QLoRA run from scratch/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Resume is unavailable/i)).toHaveLength(2);
    const trainButton = screen.getByRole("button", { name: "Start full MLX training" });
    expect(trainButton).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox"));
    expect(trainButton).toBeEnabled();
    fireEvent.click(trainButton);
    expect(onCreateJob).toHaveBeenCalledWith("train");
  });

  it("quotes the transferred bundle name in desktop shell commands", () => {
    window.aptusDesktop = {
      platform: "macos",
      reportWorkbenchReady: vi.fn(async () => undefined),
      pickDataset: vi.fn(async () => null),
      pickOutputDirectory: vi.fn(async () => null),
      revealInFinder: vi.fn(async () => undefined),
    };
    render(
      <RunStage
        bundle={{ ...bundle, bundle_dir: "/tmp/Wilson's training bundle" }}
        report={pilotReport}
        job={null}
        busy={null}
        demoMode={false}
        {...callbacks}
      />,
    );

    expect(screen.getByLabelText("CUDA host commands")).toHaveTextContent(
      `aptus run './Wilson'"'"'s training bundle' --action dependency`,
    );
  });

  it("keeps a foreign active job visible when artifact evidence is unavailable", () => {
    const job: Job = {
      id: "job_foreign",
      state: "running",
      mode: "pilot",
      log: "still running",
      return_code: null,
      cancellable: false,
      cancellation_note: "Another live Aptus process owns this job.",
    };
    render(
      <RunStage
        bundle={null}
        report={null}
        job={job}
        busy={null}
        demoMode={false}
        {...callbacks}
      />,
    );

    expect(screen.getByText("The active job remains observable.")).toBeInTheDocument();
    expect(screen.getByText("still running")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel job" })).toBeDisabled();
    expect(screen.getByText(job.cancellation_note!)).toBeInTheDocument();
  });

  it("allows confirmed training to reach atomic server admission without a cached authorization flag", async () => {
    const onCreateJob = vi.fn(async () => undefined);
    const { rerender } = render(
      <RunStage
        bundle={bundle}
        report={pilotReport}
        job={null}
        busy={null}
        demoMode={false}
        {...callbacks}
        onCreateJob={onCreateJob}
      />,
    );
    await waitFor(() => expect(screen.getByLabelText(/Training job/i)).toBeChecked());
    fireEvent.click(screen.getByLabelText(/Training job/i));
    expect(screen.getByText(/server performs the authoritative admission atomically/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox"));
    expect(screen.getByRole("button", { name: "Start training" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));
    expect(onCreateJob).toHaveBeenCalledWith("train");

    const nextJob: Job = {
      id: "job_new",
      state: "running",
      phase: "verifying",
      mode: "train",
      bundle_dir: "/tmp/bundle",
      log: "termination requested",
      return_code: 0,
      cancellable: false,
    };
    rerender(
      <RunStage
        bundle={bundle}
        report={pilotReport}
        job={nextJob}
        busy={null}
        demoMode={false}
        {...callbacks}
        onCreateJob={onCreateJob}
      />,
    );

    expect(screen.getByText("termination requested")).toBeInTheDocument();
    expect(screen.getAllByText("verifying").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Cancel job" })).toBeDisabled();
    await waitFor(() => expect(screen.getByRole("checkbox")).not.toBeChecked());
  });

  it.each([
    ["missing", undefined],
    ["wrong", { ...reportBindings, candidate_id: "candidate-other" }],
  ])("blocks a pilot-pass report with %s binding identities", (_label, bindings) => {
    const onCreateJob = vi.fn(async () => undefined);
    render(
      <RunStage
        bundle={bundle}
        report={{ state: "pilot-pass", findings: [], bindings }}
        job={null}
        busy={null}
        demoMode={false}
        {...callbacks}
        onCreateJob={onCreateJob}
      />,
    );

    expect(screen.getByText("Run preflight is blocked")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Start training/i })).not.toBeInTheDocument();
    expect(onCreateJob).not.toHaveBeenCalled();
  });

  it("renders the run-bound completion, capacity, and historical integrity record", () => {
    const job: Job = {
      id: "job_complete",
      state: "completed",
      phase: "completed",
      mode: "train",
      bundle_dir: "/tmp/bundle",
      run_id: "run_complete",
      run_output_dir: "/tmp/bundle/runs/run_complete",
      log_path: "/tmp/aptus/jobs/job_complete.log",
      created_at: "2026-07-21T11:59:58Z",
      started_at: "2026-07-21T12:00:00Z",
      finished_at: "2026-07-21T13:00:01Z",
      log: "training complete",
      return_code: 0,
      prelaunch_capacity_check: {
        checked_at: "2026-07-21T12:00:00Z",
        required_free_cuda_bytes: 8 * 1024 ** 3,
        free_cuda_bytes: [12 * 1024 ** 3],
        required_host_ram_bytes: 16 * 1024 ** 3,
        host_ram_free_bytes: 24 * 1024 ** 3,
        required_training_output_disk_bytes: 20 * 1024 ** 3,
        free_disk_bytes: 100 * 1024 ** 3,
      },
      completion_attestation: {
        state: "measured-run-pass",
        measured_run_completed_at: "2026-07-21T13:00:00Z",
        measured_run: {
          global_step: 100,
          distribution: "single",
          world_size: 1,
          train_loss: 2.1,
          eval_loss: 2.4,
        },
        final_export: {
          path: "/tmp/bundle/runs/run_complete/final",
          total_bytes: 2 * 1024 ** 3,
          manifest_sha256: "final-export-digest",
          verification_level: "structural-file-tree",
        },
      },
      artifact_integrity: {
        status: "verified-at-completion-not-rehashed",
        verified_at: "2026-07-21T13:00:00Z",
        missing_paths: [],
        note: "Polling checks presence only.",
      },
    };
    render(
      <RunStage
        bundle={bundle}
        report={pilotReport}
        job={job}
        busy={null}
        demoMode={false}
        {...callbacks}
      />,
    );

    expect(screen.getByText("/tmp/bundle/runs/run_complete")).toBeInTheDocument();
    expect(screen.getByText("/tmp/aptus/jobs/job_complete.log")).toBeInTheDocument();
    expect(screen.getByText("2026-07-21T11:59:58Z")).toBeInTheDocument();
    expect(screen.getAllByText("2026-07-21T12:00:00Z").length).toBeGreaterThan(0);
    expect(screen.getByText("2026-07-21T13:00:01Z")).toBeInTheDocument();
    expect(screen.getByText("training complete")).toBeInTheDocument();
    expect(screen.getByText("/tmp/bundle/runs/run_complete/final-export.json")).toBeInTheDocument();
    expect(screen.getByText(/directory is not immutable/i)).toBeInTheDocument();
    expect(screen.getAllByText("verified at completion not rehashed").length).toBeGreaterThan(0);
    expect(screen.getByText("final-export-digest")).toBeInTheDocument();
    expect(screen.getByText("Submit-time capacity admission")).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent("exact direct package pins");
    expect(screen.getByRole("note")).toHaveTextContent("not a complete transitive lock");
    expect(screen.getByText("Train loss")).toBeInTheDocument();
    expect(screen.getByText("Split evaluation loss")).toBeInTheDocument();
    expect(screen.queryByText(/^Eval loss$/i)).not.toBeInTheDocument();
    expect(
      screen.getByText(/Train loss and split evaluation loss are not an evaluation pass/i),
    ).toBeInTheDocument();
  });
});
