import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { EXAMPLE_DRAFT } from "./demo";
import type { ModelInspectionReceipt } from "./types";

const REVIEWED_QWEN3_LAYOUT = {
  default_bits: 4,
  default_group_size: 64,
  module_overrides: Array.from({ length: 48 }, (_, index) => ({
    module_path: `model.layers.${index}.mlp.gate`,
    bits: 8,
    group_size: 64,
  })).sort((left, right) => left.module_path.localeCompare(right.module_path)),
};

function inspectionReceipt(
  modelId = EXAMPLE_DRAFT.model.model_id,
  revision = EXAMPLE_DRAFT.model.revision,
): ModelInspectionReceipt {
  return {
    schema_version: "aptus.model-inspection-receipt.v1",
    receipt_id: `receipt_${"a".repeat(20)}`,
    model_id: modelId,
    resolved_revision: revision,
    observed_facts_sha256: "b".repeat(64),
    decision: {
      schema_version: "aptus.model-compatibility.v2",
      decision_id: `compat_${"c".repeat(20)}`,
      subject_facts_sha256: "d".repeat(64),
      kind: "family-recognized",
      family: "llama",
      policy_id: null,
      policy_version: null,
      paths: [],
      reason_codes: ["family-recognized"],
      evidence_ids: [],
      reason: "The dense family is recognized without an artifact-specific policy.",
    },
    provenance_summary: [{
      field: "family",
      kind: "provider-declared",
      source: "Provider config",
      observed_at: "2026-07-29T12:00:00+00:00",
      resolved_revision: revision,
    }],
    provenance_requirement: null,
    provenance_requirement_met: false,
    evaluated_at: "2026-07-29T12:00:00+00:00",
  };
}

function trainingPlanResponse(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  const decision = (overrides.model_policy_decision ?? inspectionReceipt().decision) as Record<string, unknown>;
  const candidateOverrides = (
    overrides.recommended && typeof overrides.recommended === "object"
      ? overrides.recommended
      : {}
  ) as Record<string, unknown>;
  const recommended = {
    candidate_id: `cand_${"1".repeat(20)}`,
    model_policy_decision_id: decision.decision_id,
    policy_binding: null,
    method: "lora",
    distribution: "single",
    status: "feasible",
    feasible: true,
    rejection_reasons: [],
    target_modules: ["q_proj"],
    runtime_contract: {
      schema_version: "aptus.runtime-contract.v1",
      compute_backend: "mps",
      training_runtime: "mlx-lm",
      compiler_id: null,
      estimator_id: "aptus-memory-mlx-v2",
      evidence_requirement: "implementation-required",
      export_kind: null,
    },
    ...candidateOverrides,
  };
  const rawCandidates = Array.isArray(overrides.candidates)
    ? overrides.candidates
    : [recommended];
  const candidates = rawCandidates.map((item, index) => ({
    candidate_id: `cand_${String(index + 1).padStart(20, "0")}`,
    model_policy_decision_id: decision.decision_id,
    policy_binding: null,
    method: "lora",
    distribution: "single",
    status: "feasible",
    feasible: true,
    rejection_reasons: [],
    target_modules: ["q_proj"],
    runtime_contract: {
      schema_version: "aptus.runtime-contract.v1",
      compute_backend: "mps",
      training_runtime: "mlx-lm",
      compiler_id: null,
      estimator_id: "aptus-memory-mlx-v2",
      evidence_requirement: "implementation-required",
      export_kind: null,
    },
    ...(item as Record<string, unknown>),
  }));
  const listedRecommended = candidates.find(
    (item) => item.candidate_id === recommended.candidate_id,
  ) ?? { ...candidates[0], ...recommended };
  if (!candidates.some((item) => item.candidate_id === listedRecommended.candidate_id)) {
    candidates[0] = listedRecommended;
  }
  return {
    ...overrides,
    schema_version: "aptus.training-plan.v5",
    plan_id: overrides.plan_id ?? `plan_${"2".repeat(20)}`,
    model_policy_snapshot_sha256:
      overrides.model_policy_snapshot_sha256 ?? "a".repeat(64),
    model: overrides.model ?? {
      model_id: EXAMPLE_DRAFT.model.model_id,
      revision: EXAMPLE_DRAFT.model.revision,
    },
    recommended: listedRecommended,
    candidates,
    warnings: overrides.warnings ?? [],
    recommendation_rationale: overrides.recommendation_rationale ?? [],
    model_policy_decision: decision,
    model_policy_decision_source:
      overrides.model_policy_decision_source ?? "user-attested",
    inspection_receipt: overrides.inspection_receipt ?? null,
  };
}

function boundTrainingPlanResponse(): Record<string, unknown> {
  const decision = {
    schema_version: "aptus.model-compatibility.v2",
    decision_id: `compat_${"7".repeat(20)}`,
    subject_facts_sha256: "8".repeat(64),
    kind: "path-matched",
    family: "future_sparse_family",
    policy_id: "model.future-sparse.mlx-lora",
    policy_version: "2.0.0",
    paths: [{
      path_id: "future-sparse.mlx-lora.single",
      method: "lora",
      distribution: "single",
      adapter_profile_id: "attention-qkvo.v1",
      target_modules: ["q_proj", "k_proj"],
      runtime_contract: {
        schema_version: "aptus.runtime-contract.v1",
        compute_backend: "mps",
        training_runtime: "mlx-lm",
        compiler_id: "mlx-lm.lora.v1",
        estimator_id: "aptus-memory-mlx-v2",
        evidence_requirement: "pilot-required",
        export_kind: "mlx-lm-adapter",
      },
      required_validation_levels: ["model-data", "measured-preflight", "pilot"],
      evidence_ids: ["runtime.future-sparse.mlx-lora.v1"],
    }],
    reason_codes: ["exact-reviewed-artifact", "pilot-not-yet-proven"],
    evidence_ids: ["runtime.future-sparse.mlx-lora.v1"],
    reason: "The pinned artifact matches a server-registered path.",
  };
  const candidate = {
    candidate_id: `cand_${"7".repeat(20)}`,
    model_policy_decision_id: decision.decision_id,
    policy_binding: {
      schema_version: "aptus.model-policy-binding.v1",
      decision_id: decision.decision_id,
      subject_facts_sha256: decision.subject_facts_sha256,
      policy_id: decision.policy_id,
      policy_version: decision.policy_version,
      path_id: decision.paths[0].path_id,
      source: "user-attested",
      inspection_receipt_id: null,
      reason_codes: decision.reason_codes,
      evidence_ids: decision.evidence_ids,
    },
    method: "lora",
    distribution: "single",
    status: "conditional",
    feasible: true,
    rejection_reasons: [],
    target_modules: ["q_proj", "k_proj"],
    runtime_contract: decision.paths[0].runtime_contract,
  };
  return trainingPlanResponse({
    model_policy_decision: decision,
    recommended: candidate,
    candidates: [candidate],
  });
}

function methodDescriptor(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    schema_version: "aptus.method-descriptor.v1",
    method_id: "lora",
    display_name: "LoRA",
    summary: "Train low-rank adapters.",
    lifecycle: "gated-executable",
    selectable: true,
    parameter_scope: "frozen-base-plus-adapter",
    parameterization: "lora",
    base_storage: "unquantized",
    compiler_id: "transformers.peft-lora.v2",
    export_kind: "peft-adapter-safetensors",
    supported_backends: ["cuda"],
    supported_distributions: ["single", "ddp", "fsdp"],
    evidence_ids: ["method.lora.paper"],
    pilot_requirement: "A bounded pilot is mandatory.",
    blocker: null,
    runtime_bindings: [
      {
        schema_version: "aptus.runtime-binding.v1",
        training_runtime: "transformers-peft-cuda",
        compute_backend: "cuda",
        compiler_id: "transformers.peft-lora.v2",
        estimator_id: "aptus-memory-v2",
        export_kind: "peft-adapter-safetensors",
        supported_distributions: ["single", "ddp", "fsdp"],
        evidence_requirement: "pilot-required",
      },
    ],
    ...overrides,
  };
}

function jobResponse(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    schema_version: "aptus.job-record.v1",
    id: "job_123",
    job_id: "job_123",
    state: "queued",
    action: "pilot",
    bundle_dir: "/tmp/bundle",
    created_at: "2026-07-21T12:00:00Z",
    ...overrides,
  };
}

function profileResponse(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    source_path: "/tmp/train.jsonl",
    source_sha256: "a".repeat(64),
    source_format: "jsonl",
    schema_name: "messages",
    example_count: 3,
    total_estimated_tokens: 45,
    sequence_p50: 12,
    sequence_p95: 20,
    sequence_max: 24,
    measurement: "estimated",
    warnings: [],
    schema_counts: { messages: 3 },
    sampled_examples: 3,
    sample_indices: [0, 1, 2],
    duplicate_count: 0,
    empty_count: 0,
    truncation_count: 1,
    truncation_rate: 1 / 3,
    source_size_bytes: 128,
    canonical_size_bytes: 120,
    max_canonical_row_bytes: 40,
    bundle_path: null,
    provenance: { kind: "measured", source: "local-file" },
    ...overrides,
  };
}

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
            plan: trainingPlanResponse({
              plan_id: `plan_${"3".repeat(20)}`,
              hardware: { reserve_per_device_bytes: 0, devices: [] },
              recommended: {
                candidate_id: `cand_${"3".repeat(20)}`,
                method: "lora",
                memory: { point_estimate_bytes: 10, upper_estimate_bytes: 12 },
              },
              candidates: [],
              warnings: [],
              recommendation_rationale: ["restored"],
            }),
            bundle: {
              bundle_dir: "/tmp/restored",
              archive_path: null,
              files: [],
              runtime_contract: null,
              report: null,
            },
            job: jobResponse({
              id: "job_restored",
              job_id: "job_restored",
              state: "cancelling",
            }),
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    const bootstrap = await api.bootstrap();

    expect(bootstrap.plan?.recommended?.id).toBe(`cand_${"3".repeat(20)}`);
    expect(bootstrap.plan?.rationale).toEqual(["restored"]);
    expect(bootstrap.bundle?.report).toBeNull();
    expect(bootstrap.job?.id).toBe("job_restored");
    expect(bootstrap.job?.state).toBe("cancelling");
  });

  it("enforces method descriptor versions and lifecycle invariants", async () => {
    const accepted = methodDescriptor();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({
          api_contract_version: "aptus.api.v1",
          capabilities: { method_catalog: [accepted] },
        }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    await expect(api.bootstrap()).resolves.toMatchObject({
      capabilities: { method_catalog: [{ method_id: "lora" }] },
    });

    const cases: Array<(descriptor: Record<string, unknown>) => void> = [
      (descriptor) => { descriptor.schema_version = "aptus.method-descriptor.v2"; },
      (descriptor) => { descriptor.lifecycle = "secretly-executable"; },
      (descriptor) => { descriptor.lifecycle = "experimental"; },
      (descriptor) => { descriptor.supported_backends = []; },
      (descriptor) => { descriptor.runtime_bindings = []; },
      (descriptor) => {
        const bindings = descriptor.runtime_bindings as Array<Record<string, unknown>>;
        bindings[0].schema_version = "aptus.runtime-binding.v2";
      },
      (descriptor) => {
        Object.assign(descriptor, {
          lifecycle: "experimental",
          selectable: false,
          compiler_id: null,
          export_kind: null,
          supported_backends: [],
          supported_distributions: [],
          blocker: null,
        });
      },
    ];

    for (const mutate of cases) {
      const descriptor = methodDescriptor();
      mutate(descriptor);
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify({
            api_contract_version: "aptus.api.v1",
            capabilities: { method_catalog: [descriptor] },
          }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      );
      await expect(api.bootstrap()).rejects.toThrow(
        /method descriptor.*(contract|blocker)/i,
      );
    }
  });

  it("translates the UI fact draft and retained project into the strict plan request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify(trainingPlanResponse({
          plan_id: `plan_${"4".repeat(20)}`,
        })),
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
    expect(body).not.toHaveProperty("inspection_receipt");
  });

  it("rejects v5 plan responses with missing provenance links", async () => {
    const cases: Array<(payload: Record<string, unknown>) => void> = [
      (payload) => { delete payload.model_policy_decision; },
      (payload) => { delete payload.inspection_receipt; },
      (payload) => {
        const candidate = (payload.candidates as Array<Record<string, unknown>>)[0];
        delete candidate.policy_binding;
      },
    ];

    for (const mutate of cases) {
      const payload = trainingPlanResponse();
      mutate(payload);
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      );

      await expect(api.plan(EXAMPLE_DRAFT)).rejects.toThrow(
        /plan.*provenance|plan response|candidate|model policy decision/i,
      );
    }
  });

  it("rejects plan responses with a missing or malformed policy snapshot digest", async () => {
    const cases: Array<(payload: Record<string, unknown>) => void> = [
      (payload) => { delete payload.model_policy_snapshot_sha256; },
      (payload) => { payload.model_policy_snapshot_sha256 = 7; },
      (payload) => { payload.model_policy_snapshot_sha256 = "A".repeat(64); },
      (payload) => { payload.model_policy_snapshot_sha256 = "a".repeat(63); },
      (payload) => { payload.model_policy_snapshot_sha256 = "g".repeat(64); },
    ];

    for (const mutate of cases) {
      const payload = trainingPlanResponse();
      mutate(payload);
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      );

      await expect(api.plan(EXAMPLE_DRAFT)).rejects.toThrow(
        /model policy snapshot digest/i,
      );
    }
  });

  it("deeply decodes policy paths and candidate bindings at plan ingress", async () => {
    const accepted = boundTrainingPlanResponse();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(accepted), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );

    await expect(api.plan(EXAMPLE_DRAFT)).resolves.toMatchObject({
      model_policy_decision: { policy_id: "model.future-sparse.mlx-lora" },
      recommended: {
        policy_binding: { path_id: "future-sparse.mlx-lora.single" },
      },
    });

    const cases: Array<(payload: Record<string, unknown>) => void> = [
      (payload) => {
        const decision = payload.model_policy_decision as Record<string, unknown>;
        decision.schema_version = "aptus.model-compatibility.v3";
      },
      (payload) => {
        const decision = payload.model_policy_decision as { paths: Array<Record<string, unknown>> };
        const runtime = decision.paths[0].runtime_contract as Record<string, unknown>;
        runtime.browser_policy_hint = "trust-me";
      },
      (payload) => {
        const candidate = (payload.candidates as Array<Record<string, unknown>>)[0];
        const binding = candidate.policy_binding as Record<string, unknown>;
        binding.subject_facts_sha256 = "0".repeat(64);
      },
      (payload) => {
        const candidate = (payload.candidates as Array<Record<string, unknown>>)[0];
        candidate.method = "qlora";
      },
    ];

    for (const mutate of cases) {
      const payload = boundTrainingPlanResponse();
      mutate(payload);
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      );
      await expect(api.plan(EXAMPLE_DRAFT)).rejects.toThrow(/policy|contract|path/i);
    }
  });

  it("binds successful plans to the submitted model subject and policy source", async () => {
    const cases: Array<{
      payload: Record<string, unknown>;
      receipt?: ModelInspectionReceipt;
      pattern: RegExp;
    }> = [
      {
        payload: trainingPlanResponse({
          model: {
            model_id: "other/model",
            revision: EXAMPLE_DRAFT.model.revision,
          },
        }),
        pattern: /model subject differs/i,
      },
      {
        payload: trainingPlanResponse({
          model: {
            model_id: EXAMPLE_DRAFT.model.model_id,
            revision: "f".repeat(40),
          },
        }),
        pattern: /model subject differs/i,
      },
      {
        payload: trainingPlanResponse(),
        receipt: inspectionReceipt(),
        pattern: /policy source differs/i,
      },
      {
        payload: trainingPlanResponse({
          model_policy_decision_source: "provider-inspection",
          inspection_receipt: inspectionReceipt(),
        }),
        pattern: /policy source differs/i,
      },
    ];

    for (const testCase of cases) {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(testCase.payload), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      );
      await expect(api.plan(EXAMPLE_DRAFT, null, testCase.receipt)).rejects.toThrow(
        testCase.pattern,
      );
    }
  });

  it("rejects malformed unbound tuples and exact-path null-binding downgrades", async () => {
    const malformed = trainingPlanResponse();
    const malformedCandidate = (malformed.candidates as Array<Record<string, unknown>>)[0];
    malformedCandidate.distribution = { label: "single" };
    const downgraded = boundTrainingPlanResponse();
    const downgradedCandidate = (downgraded.candidates as Array<Record<string, unknown>>)[0];
    downgradedCandidate.policy_binding = null;
    (downgraded.recommended as Record<string, unknown>).policy_binding = null;

    for (const payload of [malformed, downgraded]) {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      );
      await expect(api.plan(EXAMPLE_DRAFT)).rejects.toThrow(/plan candidate|policy path/i);
    }
  });

  it("rejects a successful response whose canonical recommendation is not viable", async () => {
    const payload = trainingPlanResponse();
    for (const candidate of [
      payload.recommended as Record<string, unknown>,
      (payload.candidates as Array<Record<string, unknown>>)[0],
    ]) {
      candidate.status = "infeasible";
      candidate.feasible = false;
      candidate.rejection_reasons = ["Rejected by a hard gate."];
    }
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );

    await expect(api.plan(EXAMPLE_DRAFT)).rejects.toThrow(/recommendation must be viable/i);
  });

  it("requires the complete decoded recommendation to equal its listed candidate", async () => {
    const planningExtras = {
      memory: {
        expected_bytes: 6 * 1024 ** 3,
        limit_bytes: 12 * 1024 ** 3,
        device_total_bytes: 16 * 1024 ** 3,
      },
      device_indices: [0, 1],
      batches: {
        micro_batch_size: 2,
        gradient_accumulation_steps: 4,
        effective_batch_size: 8,
      },
      precision: "bf16",
    };
    const matching = trainingPlanResponse();
    const listed = (matching.candidates as Array<Record<string, unknown>>)[0];
    Object.assign(listed, structuredClone(planningExtras));
    matching.recommended = Object.fromEntries(
      Object.entries(structuredClone(listed)).reverse(),
    );
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(matching), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    await expect(api.plan(EXAMPLE_DRAFT)).resolves.toMatchObject({
      recommended: planningExtras,
    });

    for (const omitted of ["memory", "device_indices", "batches", "precision"]) {
      const payload = trainingPlanResponse();
      const candidate = (payload.candidates as Array<Record<string, unknown>>)[0];
      Object.assign(candidate, structuredClone(planningExtras));
      const recommendation = structuredClone(candidate);
      delete recommendation[omitted];
      payload.recommended = recommendation;
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      );
      await expect(api.plan(EXAMPLE_DRAFT)).rejects.toThrow(
        /recommendation differs from its listed candidate/i,
      );
    }

    const reordered = trainingPlanResponse();
    const reorderedListed = (reordered.candidates as Array<Record<string, unknown>>)[0];
    Object.assign(reorderedListed, structuredClone(planningExtras));
    reordered.recommended = {
      ...structuredClone(reorderedListed),
      device_indices: [1, 0],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(reordered), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    await expect(api.plan(EXAMPLE_DRAFT)).rejects.toThrow(
      /recommendation differs from its listed candidate/i,
    );
  });

  it("forwards a retained inspection receipt without adding one to manual plans", async () => {
    const receipt = inspectionReceipt();
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify(trainingPlanResponse({
          plan_id: `plan_${"5".repeat(20)}`,
          model_policy_decision_source: "provider-inspection",
          inspection_receipt: receipt,
        })),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.plan(EXAMPLE_DRAFT, null, receipt);

    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(body.inspection_receipt).toEqual(receipt);
  });

  it("sends the exact MoE topology while omitting derived plan facts", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify(trainingPlanResponse({
          plan_id: `plan_${"6".repeat(20)}`,
        })),
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
      schema_version: "aptus.training-plan.v5",
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

  it("requires complete live compile responses", async () => {
    const plan = {
      plan_id: `plan_${"a".repeat(20)}`,
      project_id: `project_${"b".repeat(32)}`,
      project_revision_id: `revision_${"c".repeat(32)}`,
    };
    const response = {
      bundle_dir: "/tmp/bundle",
      archive_path: "/tmp/bundle.zip",
      files: ["plan.json"],
      runtime_contract: null,
      report: { state: "static-pass" },
      project_id: plan.project_id,
      project_revision_id: `revision_${"d".repeat(32)}`,
    } as Record<string, unknown>;
    const cases: Array<{
      mutate: (payload: Record<string, unknown>) => void;
      pattern: RegExp;
    }> = [
      {
        mutate: (payload) => { delete payload.report; },
        pattern: /compile response requires a validation report/i,
      },
      {
        mutate: (payload) => { payload.report = null; },
        pattern: /compile response requires a validation report/i,
      },
      {
        mutate: (payload) => { payload.files = { path: "plan.json" }; },
        pattern: /bundle files must be a list/i,
      },
      {
        mutate: (payload) => { payload.project_revision_id = null; },
        pattern: /bundle project revision id/i,
      },
    ];

    for (const testCase of cases) {
      const payload = structuredClone(response);
      testCase.mutate(payload);
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      );
      await expect(api.compileBundle(plan, "/tmp/bundle")).rejects.toThrow(
        testCase.pattern,
      );
    }
  });

  it("refuses compilation when a plan has no project revision identity", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.compileBundle({
      schema_version: "aptus.training-plan.v5",
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
          JSON.stringify(jobResponse({
            state: "completed",
            log: "/tmp/aptus/jobs/job_123.log",
            log_tail: "phase one\nphase two",
            started_at: "2026-07-21T12:00:01Z",
            finished_at: "2026-07-21T12:00:02Z",
          })),
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

  it("rejects malformed persisted job contracts before hydration", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(jobResponse({ bundle_dir: "" })), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    await expect(api.getJob("job_123")).resolves.toMatchObject({
      id: "job_123",
      bundle_dir: "",
    });

    const cases: Array<{
      mutate: (payload: Record<string, unknown>) => void;
      pattern: RegExp;
    }> = [
      {
        mutate: (payload) => { payload.schema_version = "aptus.job-record.v2"; },
        pattern: /unsupported job contract/i,
      },
      {
        mutate: (payload) => { delete payload.id; },
        pattern: /job id.*non-empty/i,
      },
      {
        mutate: (payload) => { payload.job_id = "job_other"; },
        pattern: /job whose ids disagree/i,
      },
      {
        mutate: (payload) => { payload.bundle_dir = null; },
        pattern: /job bundle directory must be text/i,
      },
      {
        mutate: (payload) => { payload.state = { value: "running" }; },
        pattern: /job state.*non-empty/i,
      },
      {
        mutate: (payload) => { payload.log_tail = ["valid", 7]; },
        pattern: /job log tail/i,
      },
      {
        mutate: (payload) => { payload.validation_report = "static-pass"; },
        pattern: /job validation report must be an object/i,
      },
    ];

    for (const testCase of cases) {
      const payload = jobResponse();
      testCase.mutate(payload);
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      );
      await expect(api.getJob("job_123")).rejects.toThrow(testCase.pattern);
    }
  });

  it("requests provider model facts without sending user permission or parameter claims", async () => {
    const receipt = inspectionReceipt("org/model", "a".repeat(40));
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          status: "ok",
          model_id: "org/model",
          requested_revision: "main",
          resolved_revision: "a".repeat(40),
          facts: { family: "llama", parameters: null, training_allowed: null },
          compatibility: {
            status: "recognized",
            family: "llama",
            supported_runtime: null,
            compute_backend: null,
            supported_methods: [],
            distribution: null,
            evidence_requirement: "pilot-required",
            adapter_profile_id: null,
            reason: "The dense family is recognized.",
          },
          provenance: {},
          inspection_receipt: receipt,
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const inspection = await api.inspectModel("org/model", "main");

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/v1/models/inspect");
    expect(JSON.parse(String(init.body))).toEqual({ model_id: "org/model", revision: "main" });
    expect(inspection.inspection_receipt).toEqual(receipt);
  });

  it("does not require or normalize the retired compatibility projection", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            status: "ok",
            model_id: "org/model",
            requested_revision: "main",
            resolved_revision: "a".repeat(40),
            facts: { family: "llama", parameters: null, training_allowed: null },
            compatibility: {
              status: "conditional",
              family: "qwen3_moe",
              supported_runtime: null,
              compute_backend: "mps",
              supported_methods: ["qlora"],
              distribution: "single",
              evidence_requirement: "pilot-required",
              adapter_profile_id: "attention-qkvo.v1",
              reason: "Incomplete producer data.",
            },
            provenance: {},
            inspection_receipt: inspectionReceipt("org/model", "a".repeat(40)),
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    const inspection = await api.inspectModel("org/model", "main");

    expect(inspection.inspection_receipt?.decision.kind).toBe("family-recognized");
    expect(inspection.compatibility).toMatchObject({
      status: "conditional",
      reason: "Incomplete producer data.",
    });
  });

  it("rejects inspection decisions that the workbench contract cannot decode", async () => {
    const receipt = inspectionReceipt("org/model", "a".repeat(40));
    const futureReceipt = structuredClone(receipt) as unknown as Record<string, unknown>;
    const decision = futureReceipt.decision as Record<string, unknown>;
    decision.schema_version = "aptus.model-compatibility.v3";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({
          status: "ok",
          model_id: "org/model",
          requested_revision: "main",
          resolved_revision: "a".repeat(40),
          facts: { family: "llama", parameters: null, training_allowed: null },
          compatibility: {
            status: "recognized",
            family: "llama",
            supported_runtime: null,
            compute_backend: null,
            supported_methods: [],
            distribution: null,
            evidence_requirement: "pilot-required",
            adapter_profile_id: null,
            reason: "The dense family is recognized.",
          },
          provenance: {},
          inspection_receipt: futureReceipt,
        }), { status: 200, headers: { "content-type": "application/json" } }),
      ),
    );

    await expect(api.inspectModel("org/model", "main")).rejects.toThrow(
      /unsupported model policy decision contract.*update aptus/i,
    );
  });

  it("returns the validation report attached to a refreshed job", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify(jobResponse({
            state: "completed",
            action: "preflight",
            validation_report: {
              state: "measured-preflight-pass",
              findings: [],
            },
          })),
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
        JSON.stringify(jobResponse({ state: "cancelled" })),
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

  it("distinguishes estimated and tokenizer-measured profile evidence", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(profileResponse()), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(profileResponse({
          measurement: "tokenizer-measured",
          provenance: { kind: "provider-declared", source: "cached-metadata" },
        })), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const estimated = await api.profile(EXAMPLE_DRAFT);
    const measured = await api.profile(EXAMPLE_DRAFT);

    expect(estimated.facts?.find((fact) => fact.key === "dataset_hash")).toMatchObject({
      value: "aaaaaaaa…aaaa",
      provenance: "measured",
      source: "local-file",
    });
    expect(estimated.facts?.find((fact) => fact.key === "sequence_p95")?.provenance).toBe("inferred");
    expect(measured.facts?.find((fact) => fact.key === "dataset_hash")?.provenance).toBe("inferred");
    expect(measured.facts?.find((fact) => fact.key === "sequence_p95")?.provenance).toBe("measured");
  });

  it("rejects incomplete or malformed profile contracts", async () => {
    const cases: Array<{
      mutate: (payload: Record<string, unknown>) => void;
      pattern: RegExp;
    }> = [
      {
        mutate: (payload) => { delete payload.total_estimated_tokens; },
        pattern: /total_estimated_tokens.*positive integer/i,
      },
      {
        mutate: (payload) => { payload.measurement = "provider-guessed"; },
        pattern: /unknown measurement kind/i,
      },
      {
        mutate: (payload) => {
          payload.provenance = { kind: "future", source: "local-file" };
        },
        pattern: /invalid provenance/i,
      },
      {
        mutate: (payload) => { payload.sequence_p95 = 4; },
        pattern: /percentiles are out of order/i,
      },
      {
        mutate: (payload) => { payload.sample_indices = [0, "1"]; },
        pattern: /sample indices/i,
      },
    ];

    for (const testCase of cases) {
      const payload = profileResponse();
      testCase.mutate(payload);
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      );
      await expect(api.profile(EXAMPLE_DRAFT)).rejects.toThrow(testCase.pattern);
    }
  });

  it("preserves rejected candidates when no strategy is feasible", async () => {
    const decision = inspectionReceipt().decision;
    const candidateId = `cand_${"9".repeat(20)}`;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: "no_feasible_plan",
            message: "No candidate passed every hard gate.",
            model: {
              model_id: EXAMPLE_DRAFT.model.model_id,
              revision: EXAMPLE_DRAFT.model.revision,
            },
            model_policy_decision: decision,
            model_policy_decision_source: "user-attested",
            inspection_receipt: null,
            candidates: [{
              candidate_id: candidateId,
              model_policy_decision_id: decision.decision_id,
              policy_binding: null,
              method: "qlora",
              distribution: "single",
              status: "infeasible",
              feasible: false,
              rejection_reasons: ["Even the point estimate exceeds usable per-device VRAM."],
              target_modules: ["q_proj"],
              runtime_contract: {
                schema_version: "aptus.runtime-contract.v1",
                compute_backend: "mps",
                training_runtime: "mlx-lm",
                compiler_id: null,
                estimator_id: "aptus-memory-mlx-v2",
                evidence_requirement: "implementation-required",
                export_kind: null,
              },
              memory: { point_estimate_bytes: 12, upper_estimate_bytes: 15 },
            }],
          }),
          { status: 422, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    const plan = await api.plan(EXAMPLE_DRAFT);

    expect(plan.recommended).toBeNull();
    expect(plan.candidates[0]?.id).toBe(candidateId);
    expect(plan.model_policy_decision).toEqual(decision);
    expect(plan.model_policy_decision_source).toBe("user-attested");
    expect(plan.warnings[0]).toContain("No candidate");
  });

  it("rejects no-feasible responses that omit or mismatch the policy chain", async () => {
    const decision = inspectionReceipt().decision;
    const base = {
      error: "no_feasible_plan",
      message: "No candidate passed every hard gate.",
      model: {
        model_id: EXAMPLE_DRAFT.model.model_id,
        revision: EXAMPLE_DRAFT.model.revision,
      },
      model_policy_decision: decision,
      model_policy_decision_source: "user-attested",
      inspection_receipt: null,
      candidates: [{
        candidate_id: `cand_${"9".repeat(20)}`,
        model_policy_decision_id: decision.decision_id,
        policy_binding: null,
        method: "qlora",
        distribution: "single",
        status: "infeasible",
        feasible: false,
        rejection_reasons: ["Rejected."],
        target_modules: ["q_proj"],
        runtime_contract: {
          schema_version: "aptus.runtime-contract.v1",
          compute_backend: "mps",
          training_runtime: "mlx-lm",
          compiler_id: null,
          estimator_id: "aptus-memory-mlx-v2",
          evidence_requirement: "implementation-required",
          export_kind: null,
        },
      }],
    };
    const cases = [
      { ...base, model_policy_decision: undefined },
      { ...base, model_policy_decision_source: "browser-inferred" },
      { ...base, message: " " },
      { ...base, browser_recovery_hint: "trust me" },
      { ...base, model: undefined },
      {
        ...base,
        model: { ...base.model, model_id: "other/model" },
      },
      {
        ...base,
        model: { ...base.model, revision: "f".repeat(40) },
      },
      {
        ...base,
        model: { ...base.model, model_id: " " },
      },
      {
        ...base,
        model: { ...base.model, revision: "main" },
      },
      { ...base, candidates: [base.candidates[0], { ...base.candidates[0] }] },
      {
        ...base,
        candidates: [{
          ...base.candidates[0],
          status: "feasible",
          feasible: true,
          rejection_reasons: [],
        }],
      },
      {
        ...base,
        candidates: [{ ...base.candidates[0], rejection_reasons: [] }],
      },
      {
        ...base,
        candidates: [{ ...base.candidates[0], runtime_contract: null }],
      },
      {
        ...base,
        candidates: [{ ...base.candidates[0], model_policy_decision_id: `compat_${"0".repeat(20)}` }],
      },
    ];

    for (const payload of cases) {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(payload), {
            status: 422,
            headers: { "content-type": "application/json" },
          }),
        ),
      );
      await expect(api.plan(EXAMPLE_DRAFT)).rejects.toThrow(
        /policy|candidate|message|shape|duplicate|model|revision/i,
      );
    }
  });

  it("binds provider-backed no-feasible responses to the submitted receipt and artifact", async () => {
    const requestReceipt = inspectionReceipt();
    const candidate = {
      candidate_id: `cand_${"8".repeat(20)}`,
      model_policy_decision_id: requestReceipt.decision.decision_id,
      policy_binding: null,
      method: "lora",
      distribution: "single",
      status: "infeasible",
      feasible: false,
      rejection_reasons: ["Rejected by memory admission."],
      target_modules: ["q_proj"],
      runtime_contract: {
        schema_version: "aptus.runtime-contract.v1",
        compute_backend: "mps",
        training_runtime: "mlx-lm",
        compiler_id: null,
        estimator_id: "aptus-memory-mlx-v2",
        evidence_requirement: "implementation-required",
        export_kind: null,
      },
    };
    const base = {
      error: "no_feasible_plan",
      message: "No candidate passed every hard gate.",
      model: {
        model_id: EXAMPLE_DRAFT.model.model_id,
        revision: EXAMPLE_DRAFT.model.revision,
      },
      candidates: [candidate],
      model_policy_decision: requestReceipt.decision,
      model_policy_decision_source: "provider-inspection",
      inspection_receipt: requestReceipt,
    };
    const wrongModel = structuredClone(requestReceipt);
    wrongModel.model_id = "other/model";
    const wrongRevision = structuredClone(requestReceipt);
    wrongRevision.resolved_revision = "f".repeat(40);
    wrongRevision.provenance_summary[0].resolved_revision = "f".repeat(40);
    const wrongReceipt = structuredClone(requestReceipt);
    wrongReceipt.receipt_id = `receipt_${"f".repeat(20)}`;
    const cases = [
      { ...base, model_policy_decision_source: "user-attested", inspection_receipt: null },
      { ...base, inspection_receipt: wrongModel },
      { ...base, inspection_receipt: wrongRevision },
      { ...base, inspection_receipt: wrongReceipt },
    ];

    for (const payload of cases) {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify(payload), {
            status: 422,
            headers: { "content-type": "application/json" },
          }),
        ),
      );
      await expect(api.plan(EXAMPLE_DRAFT, null, requestReceipt)).rejects.toThrow(
        /source|receipt|model ID|revision/i,
      );
    }
  });

  it("rejects a no-feasible exact-path row with a downgraded null binding", async () => {
    const success = boundTrainingPlanResponse();
    const candidate = structuredClone(
      (success.candidates as Array<Record<string, unknown>>)[0],
    );
    candidate.policy_binding = null;
    candidate.status = "infeasible";
    candidate.feasible = false;
    candidate.rejection_reasons = ["Rejected by memory admission."];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({
          error: "no_feasible_plan",
          message: "No candidate passed every hard gate.",
          model: {
            model_id: EXAMPLE_DRAFT.model.model_id,
            revision: EXAMPLE_DRAFT.model.revision,
          },
          candidates: [candidate],
          model_policy_decision: success.model_policy_decision,
          model_policy_decision_source: "user-attested",
          inspection_receipt: null,
        }), {
          status: 422,
          headers: { "content-type": "application/json" },
        }),
      ),
    );

    await expect(api.plan(EXAMPLE_DRAFT)).rejects.toThrow(
      /exactly matches a policy path.*cannot omit/i,
    );
  });

  it("uses the limiting returned GPU in the Fit Ledger normalization", async () => {
    const GiB = 1024 ** 3;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify(trainingPlanResponse({
            plan_id: `plan_${"7".repeat(20)}`,
            hardware: {
              reserve_per_device_bytes: 2 * GiB,
              devices: [
                { total_vram_bytes: 24 * GiB, free_vram_bytes: 22 * GiB },
                { total_vram_bytes: 16 * GiB, free_vram_bytes: 9 * GiB },
              ],
            },
            recommended: {
              candidate_id: `cand_${"7".repeat(20)}`,
              method: "lora",
              status: "feasible",
              memory: { point_estimate_bytes: 4 * GiB, upper_estimate_bytes: 5 * GiB },
            },
            candidates: [],
          })),
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
          JSON.stringify(trainingPlanResponse({
            plan_id: `plan_${"8".repeat(20)}`,
            hardware: {
              reserve_per_device_bytes: 2 * GiB,
              devices: [
                { total_vram_bytes: 24 * GiB, free_vram_bytes: 22 * GiB },
                { total_vram_bytes: 16 * GiB, free_vram_bytes: 9 * GiB },
              ],
            },
            recommended: {
              candidate_id: `cand_${"8".repeat(20)}`,
              method: "lora",
              status: "feasible",
              device_indices: [0],
              memory: { point_estimate_bytes: 4 * GiB, upper_estimate_bytes: 5 * GiB },
            },
            candidates: [],
          })),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    const plan = await api.plan(EXAMPLE_DRAFT);

    expect(plan.recommended?.memory?.limit_bytes).toBe(20 * GiB);
    expect(plan.recommended?.memory?.device_total_bytes).toBe(24 * GiB);
  });
});
