import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { EXAMPLE_DRAFT } from "./demo";

const REVIEWED_QWEN3_LAYOUT = {
  default_bits: 4,
  default_group_size: 64,
  module_overrides: Array.from({ length: 48 }, (_, index) => ({
    module_path: `model.layers.${index}.mlp.gate`,
    bits: 8,
    group_size: 64,
  })).sort((left, right) => left.module_path.localeCompare(right.module_path)),
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("typed API client", () => {
  it("accepts the supported API contract and rejects a future one before hydration", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            api_contract_version: "aptus.api.v1",
            service: { name: "aptus", version: "0.2.0" },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            api_contract_version: "aptus.api.v2",
            service: { name: "aptus", version: "9.0.0" },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ service: { name: "aptus", version: "0.2.0" } }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.bootstrap()).resolves.toMatchObject({
      api_contract_version: "aptus.api.v1",
    });
    await expect(api.bootstrap()).rejects.toThrow(
      /unsupported Aptus API contract.*aptus\.api\.v2.*requires aptus\.api\.v1/i,
    );
    await expect(api.bootstrap()).rejects.toThrow(
      /missing or unsupported Aptus API contract.*requires aptus\.api\.v1/i,
    );
  });

  it("normalizes restored plan and job payloads during bootstrap", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            api_contract_version: "aptus.api.v1",
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

  it("rejects unknown method lifecycles at the bootstrap boundary", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            api_contract_version: "aptus.api.v1",
            capabilities: {
              method_catalog: [
                {
                  schema_version: "aptus.method-descriptor.v1",
                  method_id: "future-method",
                  display_name: "Future method",
                  summary: "Unknown lifecycle fixture.",
                  lifecycle: "secretly-executable",
                  selectable: false,
                  parameter_scope: "unknown",
                  parameterization: "unknown",
                  base_storage: "unknown",
                  compiler_id: null,
                  export_kind: null,
                  supported_backends: [],
                  supported_distributions: [],
                  evidence_ids: ["fixture"],
                  pilot_requirement: "Unknown",
                  blocker: "Unknown",
                },
              ],
            },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    await expect(api.bootstrap()).rejects.toThrow(/violates its API contract/i);
  });

  it("translates the UI fact draft and retained project into the strict plan request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          schema_version: "aptus.training-plan.v3",
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

    const projectId = `project_${"a".repeat(32)}`;
    await api.plan(EXAMPLE_DRAFT, projectId);

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(String(init.body));
    expect(url).toBe("/api/v1/plan");
    expect(body).not.toHaveProperty("facts");
    expect(body.project_id).toBe(projectId);
    expect(body.project_name).toBe(EXAMPLE_DRAFT.project_name);
    expect(body.model.model_id).toBe(EXAMPLE_DRAFT.model.model_id);
    expect(body.model).not.toHaveProperty("moe");
    expect(body.model).not.toHaveProperty("model_type");
    expect(body.model).not.toHaveProperty("quantization_bits");
    expect(body.model).not.toHaveProperty("quantization_layout");
    expect(body.hardware.gpu_count).toBe(1);
    expect(body.hardware.discovery).toBe("manual");
    expect(body.hardware.free_vram_gib).toBe(24);
    expect(body.hardware.host_ram_free_gib).toBe(48);
    expect(body.hardware.supports_8bit).toBe(true);
    expect(body.target.task).toBe("sft");
    expect(body.target.evaluation_fraction).toBe(0.1);
    expect(body.dataset_path).toBe(EXAMPLE_DRAFT.dataset.source_path);
  });

  it("sends the exact MoE topology while omitting derived plan facts", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          schema_version: "aptus.training-plan.v3",
          plan_id: "plan_moe",
          recommended: null,
          candidates: [],
          warnings: [],
          recommendation_rationale: [],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const draft = structuredClone(EXAMPLE_DRAFT);
    draft.model = {
      ...draft.model,
      family: "qwen3_moe",
      model_type: "qwen3_moe",
      architecture: "Qwen3MoeForCausalLM",
      quantization_bits: 4,
      quantization_layout: REVIEWED_QWEN3_LAYOUT,
      parameters_b: 30.5,
      active_parameters_b: 3.3,
      sparse_layer_count: 48,
      moe: {
        expert_count: 128,
        experts_per_token: 8,
        expert_intermediate_size: 768,
        decoder_sparse_step: 1,
        mlp_only_layers: [],
        shared_expert_intermediate_size: 768,
      },
    };

    await api.plan(draft);

    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(body.model).toMatchObject({
      family: "qwen3_moe",
      model_type: "qwen3_moe",
      architecture: "Qwen3MoeForCausalLM",
      quantization_bits: 4,
      quantization_layout: REVIEWED_QWEN3_LAYOUT,
      moe: {
        expert_count: 128,
        experts_per_token: 8,
        expert_intermediate_size: 768,
        decoder_sparse_step: 1,
        mlp_only_layers: [],
        shared_expert_intermediate_size: 768,
      },
    });
    expect(body.model).not.toHaveProperty("active_parameters");
    expect(body.model).not.toHaveProperty("active_parameters_b");
    expect(body.model).not.toHaveProperty("sparse_layer_count");
  });

  it("binds compile and validation mutations to the exact project revision", async () => {
    const projectId = `project_${"a".repeat(32)}`;
    const planRevisionId = `revision_${"b".repeat(32)}`;
    const bundleRevisionId = `revision_${"c".repeat(32)}`;
    const validationRevisionId = `revision_${"d".repeat(32)}`;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        bundle_dir: "/tmp/bundle",
        archive_path: "/tmp/bundle.zip",
        files: [],
        runtime_contract: null,
        report: { state: "static-pass" },
        project_id: projectId,
        project_revision_id: bundleRevisionId,
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        state: "static-pass",
        project_id: projectId,
        project_revision_id: validationRevisionId,
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }));
    vi.stubGlobal("fetch", fetchMock);
    const plan = {
      schema_version: "aptus.training-plan.v3",
      plan_id: "plan_exact",
      project_id: projectId,
      project_revision_id: planRevisionId,
      recommended: null,
      candidates: [],
      warnings: [],
      rationale: [],
    };

    const bundle = await api.compileBundle(plan, "/tmp/bundle");
    await api.validate(
      bundle.bundle_dir,
      "static",
      false,
      bundle.project_id,
      bundle.project_revision_id,
    );

    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      plan_id: "plan_exact",
      project_id: projectId,
      expected_project_revision_id: planRevisionId,
    });
    expect(JSON.parse(String(fetchMock.mock.calls[1]?.[1]?.body))).toMatchObject({
      bundle_dir: "/tmp/bundle",
      project_id: projectId,
      expected_project_revision_id: bundleRevisionId,
    });
  });

  it("refuses compilation when a plan has no project revision identity", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.compileBundle({
      schema_version: "aptus.training-plan.v3",
      plan_id: "plan_unbound",
      recommended: null,
      candidates: [],
      warnings: [],
      rationale: [],
    }, "/tmp/bundle")).rejects.toThrow(/exact project and project revision/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("uses typed project history, revision detail, and recovery endpoints", async () => {
    const projectId = `project_${"b".repeat(32)}`;
    const revisionId = `revision_${"c".repeat(32)}`;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify([]), {
        status: 200,
        headers: { "content-type": "application/json" },
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        schema_version: "aptus.project-revision.v1",
        project_id: projectId,
        revision_id: revisionId,
        ordinal: 1,
        created_at: "2026-07-27T12:00:00Z",
        reason: "plan-created",
        job_ids: [],
        training_authorization: { current: false, reason: "Never durable." },
        content_sha256: "d".repeat(64),
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        status: "recovered",
        project_id: projectId,
        revision: {
          schema_version: "aptus.project-revision.v1",
          project_id: projectId,
          revision_id: `revision_${"e".repeat(32)}`,
          ordinal: 2,
          created_at: "2026-07-27T12:01:00Z",
          reason: `recovered-from:${revisionId}`,
          job_ids: [],
          training_authorization: { current: false, reason: "Never durable." },
          content_sha256: "f".repeat(64),
        },
        training_authorization_current: false,
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }));
    vi.stubGlobal("fetch", fetchMock);

    await api.projectHistory(projectId);
    const revision = await api.projectRevision(projectId, revisionId);
    const recovery = await api.recoverProjectRevision(projectId, revisionId);

    expect(revision.training_authorization.current).toBe(false);
    expect(recovery.training_authorization_current).toBe(false);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(`/api/v1/projects/${projectId}/revisions`);
    expect(fetchMock.mock.calls[1]?.[0]).toBe(`/api/v1/projects/${projectId}/revisions/${revisionId}`);
    expect(fetchMock.mock.calls[2]).toEqual([
      `/api/v1/projects/${projectId}/recover`,
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ revision_id: revisionId }),
      }),
    ]);
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
      project_id: `project_${"a".repeat(32)}`,
      expected_project_revision_id: `revision_${"b".repeat(32)}`,
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
    const fetchMock = vi.mocked(fetch);
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      project_id: `project_${"a".repeat(32)}`,
      expected_project_revision_id: `revision_${"b".repeat(32)}`,
    });
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
