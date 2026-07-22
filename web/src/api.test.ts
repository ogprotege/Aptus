import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { EXAMPLE_DRAFT } from "./demo";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("typed API client", () => {
  it("normalizes restored plan and job payloads during bootstrap", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            plan: {
              plan_id: "plan_restored",
              hardware: { reserve_per_device_bytes: 0, devices: [] },
              recommended: {
                candidate_id: "cand_restored",
                method: "lora",
                memory: { point_estimate_bytes: 10, upper_estimate_bytes: 12 },
              },
              candidates: [],
              warnings: [],
              recommendation_rationale: ["restored"],
            },
            bundle: { bundle_dir: "/tmp/restored", files: [] },
            job: { job_id: "job_restored", state: "cancelling", action: "pilot" },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    const bootstrap = await api.bootstrap();

    expect(bootstrap.plan?.recommended?.id).toBe("cand_restored");
    expect(bootstrap.plan?.rationale).toEqual(["restored"]);
    expect(bootstrap.job?.id).toBe("job_restored");
    expect(bootstrap.job?.state).toBe("cancelling");
  });

  it("translates the UI fact draft into the strict v0.2 plan request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          schema_version: "aptus.training-plan.v2",
          plan_id: "plan_example",
          recommended: null,
          candidates: [],
          warnings: [],
          recommendation_rationale: [],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.plan(EXAMPLE_DRAFT);

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(String(init.body));
    expect(url).toBe("/api/v1/plan");
    expect(body).not.toHaveProperty("facts");
    expect(body.model.model_id).toBe(EXAMPLE_DRAFT.model.model_id);
    expect(body.hardware.gpu_count).toBe(1);
    expect(body.hardware.discovery).toBe("manual");
    expect(body.hardware.free_vram_gib).toBe(24);
    expect(body.hardware.host_ram_free_gib).toBe(48);
    expect(body.hardware.supports_8bit).toBe(true);
    expect(body.target.task).toBe("sft");
    expect(body.target.evaluation_fraction).toBe(0.1);
    expect(body.dataset_path).toBe(EXAMPLE_DRAFT.dataset.source_path);
  });

  it("normalizes the persisted job record without replacing the log tail with its path", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            job_id: "job_123",
            state: "completed",
            action: "pilot",
            log: "/tmp/aptus/jobs/job_123.log",
            log_tail: "phase one\nphase two",
            created_at: "2026-07-21T12:00:00Z",
            started_at: "2026-07-21T12:00:01Z",
            finished_at: "2026-07-21T12:00:02Z",
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );
    const job = await api.createJob({
      bundle_dir: "/tmp/bundle",
      action: "pilot",
      confirm_full_train: false,
    });
    expect(job.id).toBe("job_123");
    expect(job.mode).toBe("pilot");
    expect(job.log).toBe("phase one\nphase two");
    expect(job.log_path).toBe("/tmp/aptus/jobs/job_123.log");
    expect(job.created_at).toBe("2026-07-21T12:00:00Z");
    expect(job.started_at).toBe("2026-07-21T12:00:01Z");
    expect(job.finished_at).toBe("2026-07-21T12:00:02Z");
  });

  it("requests provider model facts without sending user permission or parameter claims", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          status: "ok",
          model_id: "org/model",
          requested_revision: "main",
          resolved_revision: "a".repeat(40),
          facts: { family: "llama", parameters: null, training_allowed: null },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.inspectModel("org/model", "main");

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/v1/models/inspect");
    expect(JSON.parse(String(init.body))).toEqual({ model_id: "org/model", revision: "main" });
  });

  it("returns the validation report attached to a refreshed job", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            id: "job_123",
            state: "completed",
            action: "preflight",
            validation_report: {
              state: "measured-preflight-pass",
              findings: [],
            },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    const job = await api.getJob("job_123");

    expect(job.validation_report?.state).toBe("measured-preflight-pass");
  });

  it("cancels a job through the bound job endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ id: "job_123", state: "cancelled", action: "pilot" }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const job = await api.cancelJob("job_123");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/jobs/job_123/cancel",
      expect.objectContaining({ method: "POST" }),
    );
    expect(job.state).toBe("cancelled");
  });

  it("labels estimated token statistics as inferred evidence", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            source_sha256: "a".repeat(64),
            example_count: 3,
            sequence_p95: 20,
            truncation_rate: 0.1,
            measurement: "estimated",
            provenance: { kind: "measured", source: "local-file" },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    const profile = await api.profile(EXAMPLE_DRAFT);

    expect(profile.facts?.find((fact) => fact.key === "dataset_hash")?.provenance).toBe("measured");
    expect(profile.facts?.find((fact) => fact.key === "sequence_p95")?.provenance).toBe("inferred");
  });

  it("preserves rejected candidates when no strategy is feasible", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: "no_feasible_plan",
            message: "No candidate passed every hard gate.",
            candidates: [{
              candidate_id: "cand_rejected",
              method: "qlora",
              status: "infeasible",
              rejection_reasons: ["Even the point estimate exceeds usable per-device VRAM."],
              memory: { point_estimate_bytes: 12, upper_estimate_bytes: 15 },
            }],
          }),
          { status: 422, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    const plan = await api.plan(EXAMPLE_DRAFT);

    expect(plan.recommended).toBeNull();
    expect(plan.candidates[0]?.id).toBe("cand_rejected");
    expect(plan.warnings[0]).toContain("No candidate");
  });

  it("uses the limiting returned GPU in the Fit Ledger normalization", async () => {
    const GiB = 1024 ** 3;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            plan_id: "plan_example",
            hardware: {
              reserve_per_device_bytes: 2 * GiB,
              devices: [
                { total_vram_bytes: 24 * GiB, free_vram_bytes: 22 * GiB },
                { total_vram_bytes: 16 * GiB, free_vram_bytes: 9 * GiB },
              ],
            },
            recommended: {
              candidate_id: "cand_example",
              method: "lora",
              status: "feasible",
              memory: { point_estimate_bytes: 4 * GiB, upper_estimate_bytes: 5 * GiB },
            },
            candidates: [],
            warnings: [],
            recommendation_rationale: [],
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    const plan = await api.plan(EXAMPLE_DRAFT);

    expect(plan.recommended?.memory?.limit_bytes).toBe(7 * GiB);
    expect(plan.recommended?.memory?.device_total_bytes).toBe(16 * GiB);
  });

  it("uses each candidate's bound devices for its Fit Ledger capacity", async () => {
    const GiB = 1024 ** 3;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            plan_id: "plan_selected_device",
            hardware: {
              reserve_per_device_bytes: 2 * GiB,
              devices: [
                { total_vram_bytes: 24 * GiB, free_vram_bytes: 22 * GiB },
                { total_vram_bytes: 16 * GiB, free_vram_bytes: 9 * GiB },
              ],
            },
            recommended: {
              candidate_id: "cand_selected_device",
              method: "lora",
              status: "feasible",
              device_indices: [0],
              memory: { point_estimate_bytes: 4 * GiB, upper_estimate_bytes: 5 * GiB },
            },
            candidates: [],
            warnings: [],
            recommendation_rationale: [],
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    const plan = await api.plan(EXAMPLE_DRAFT);

    expect(plan.recommended?.memory?.limit_bytes).toBe(20 * GiB);
    expect(plan.recommended?.memory?.device_total_bytes).toBe(24 * GiB);
  });
});
