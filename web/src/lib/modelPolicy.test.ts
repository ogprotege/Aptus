import { describe, expect, it } from "vitest";
import type {
  CandidatePlan,
  ModelPolicyBinding,
  ModelPolicyDecision,
  ValidationReport,
} from "../types";
import {
  buildModelPolicyPresentation,
  decodeModelInspectionReceipt,
  decodePlanCandidate,
  decodeModelPolicyBinding,
  decodeModelPolicyDecision,
  decodeValidationReport,
} from "./modelPolicy";

const DECISION_ID = `compat_${"a".repeat(20)}`;
const CANDIDATE_ID = `cand_${"b".repeat(20)}`;
const SUBJECT_DIGEST = "c".repeat(64);
const PLAN_ID = `plan_${"9".repeat(20)}`;
const REVISION = "e".repeat(40);

const DECISION: ModelPolicyDecision = {
  schema_version: "aptus.model-compatibility.v2",
  decision_id: DECISION_ID,
  subject_facts_sha256: SUBJECT_DIGEST,
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
  reason: "The pinned future sparse artifact matches a server-registered path.",
};

const BINDING: ModelPolicyBinding = {
  schema_version: "aptus.model-policy-binding.v1",
  decision_id: DECISION_ID,
  subject_facts_sha256: SUBJECT_DIGEST,
  policy_id: "model.future-sparse.mlx-lora",
  policy_version: "2.0.0",
  path_id: "future-sparse.mlx-lora.single",
  source: "provider-inspection",
  inspection_receipt_id: `receipt_${"d".repeat(20)}`,
  reason_codes: ["exact-reviewed-artifact", "pilot-not-yet-proven"],
  evidence_ids: ["runtime.future-sparse.mlx-lora.v1"],
};

const RECEIPT = {
  schema_version: "aptus.model-inspection-receipt.v1",
  receipt_id: BINDING.inspection_receipt_id,
  model_id: "provider/future-sparse",
  resolved_revision: "e".repeat(40),
  observed_facts_sha256: "f".repeat(64),
  decision: DECISION,
  provenance_summary: [{
    field: "family",
    kind: "provider-declared",
    source: "Provider config",
    observed_at: "2026-08-04T12:00:00+00:00",
    resolved_revision: "e".repeat(40),
  }],
  provenance_requirement: "provider-declared",
  provenance_requirement_met: true,
  evaluated_at: "2026-08-04T12:00:00+00:00",
};

const CANDIDATE: CandidatePlan = {
  candidate_id: CANDIDATE_ID,
  model_policy_decision_id: DECISION_ID,
  policy_binding: BINDING,
  method: "lora",
  distribution: "single",
  status: "conditional",
  feasible: true,
  rejection_reasons: [],
  target_modules: ["q_proj", "k_proj"],
  runtime_contract: DECISION.paths[0].runtime_contract,
};

function reportBindings(
  candidateId = CANDIDATE_ID,
  planId = PLAN_ID,
  revision = REVISION,
): Record<string, string> {
  return {
    plan_id: planId,
    candidate_id: candidateId,
    model_revision: revision,
  };
}

describe("model policy decision boundary", () => {
  it("decodes an unknown-to-the-browser policy without reconstructing its predicates", () => {
    expect(decodeModelPolicyDecision(structuredClone(DECISION))).toEqual(DECISION);
    expect(decodeModelPolicyBinding(structuredClone(BINDING), {
      decision: DECISION,
      source: "provider-inspection",
      inspectionReceiptId: BINDING.inspection_receipt_id,
      candidate: CANDIDATE,
    })).toEqual(BINDING);
  });

  it("accepts every current decision kind without hard-coding a policy family", () => {
    const variants = [
      DECISION,
      {
        ...DECISION,
        kind: "family-recognized",
        policy_id: null,
        policy_version: null,
        paths: [],
        reason_codes: ["family-recognized"],
        evidence_ids: [],
      },
      {
        ...DECISION,
        kind: "blocked",
        paths: [],
        reason_codes: ["identity-mismatch"],
      },
      {
        ...DECISION,
        kind: "unknown",
        family: null,
        policy_id: null,
        policy_version: null,
        paths: [],
        reason_codes: ["no-policy-match"],
        evidence_ids: [],
      },
    ];

    expect(variants.map((variant) => decodeModelPolicyDecision(variant).kind)).toEqual([
      "path-matched",
      "family-recognized",
      "blocked",
      "unknown",
    ]);
  });

  it("rejects malformed decision and path shapes before UI hydration", () => {
    const cases: unknown[] = [
      { ...DECISION, unexpected_policy_hint: true },
      { ...DECISION, subject_facts_sha256: "A".repeat(64) },
      {
        ...DECISION,
        paths: [{
          ...DECISION.paths[0],
          target_modules: ["q_proj", "q_proj"],
        }],
      },
      {
        ...DECISION,
        paths: [{
          ...DECISION.paths[0],
          runtime_contract: {
            ...DECISION.paths[0].runtime_contract,
            compute_backend: "cuda",
          },
        }],
      },
      {
        ...DECISION,
        kind: "family-recognized",
      },
    ];

    for (const value of cases) {
      expect(() => decodeModelPolicyDecision(value)).toThrow(/invalid model policy/i);
    }
  });

  it("decodes receipts and rejects decision or binding identity drift", () => {
    expect(decodeModelInspectionReceipt(structuredClone(RECEIPT), {
      modelId: RECEIPT.model_id,
      resolvedRevision: RECEIPT.resolved_revision,
      decision: DECISION,
    })).toEqual(RECEIPT);

    expect(() => decodeModelInspectionReceipt({
      ...RECEIPT,
      decision: {
        ...DECISION,
        paths: [{ ...DECISION.paths[0], target_modules: ["v_proj"] }],
      },
    }, { decision: DECISION })).toThrow(/receipt.*decision differs/i);

    for (const binding of [
      { ...BINDING, subject_facts_sha256: "0".repeat(64) },
      { ...BINDING, path_id: "future-sparse.other" },
      { ...BINDING, reason_codes: ["exact-reviewed-artifact"] },
      { ...BINDING, evidence_ids: ["runtime.unrelated.v1"] },
    ]) {
      expect(() => decodeModelPolicyBinding(binding, {
        decision: DECISION,
        source: "provider-inspection",
        inspectionReceiptId: BINDING.inspection_receipt_id,
        candidate: CANDIDATE,
      })).toThrow(/invalid model policy binding/i);
    }

    for (const receipt of [
      { ...RECEIPT, provenance_requirement: null },
      { ...RECEIPT, provenance_requirement_met: false },
    ]) {
      expect(() => decodeModelInspectionReceipt(receipt)).toThrow(
        /receipt provenance requirement/i,
      );
    }
    for (const modelId of ["owner/model/extra", "../model", "owner--name/model", "owner/model.git"]) {
      expect(() => decodeModelInspectionReceipt({ ...RECEIPT, model_id: modelId })).toThrow(
        /receipt model ID/i,
      );
    }

    expect(() => decodeModelInspectionReceipt({
      ...RECEIPT,
      provenance_summary: RECEIPT.provenance_summary.map((item) => ({
        ...item,
        kind: "inferred",
      })),
    })).toThrow(/provider-declared observation/i);
  });

  it("decodes every candidate execution tuple and rejects a null binding downgrade", () => {
    expect(decodePlanCandidate(structuredClone(CANDIDATE), {
      decision: DECISION,
      source: "provider-inspection",
      inspectionReceiptId: BINDING.inspection_receipt_id,
    })).toMatchObject({ candidate_id: CANDIDATE_ID, policy_binding: BINDING });

    expect(() => decodePlanCandidate({ ...CANDIDATE, distribution: {} }, {
      decision: DECISION,
      source: "provider-inspection",
      inspectionReceiptId: BINDING.inspection_receipt_id,
    })).toThrow(/plan candidate distribution/i);
    expect(() => decodePlanCandidate({ ...CANDIDATE, policy_binding: null }, {
      decision: DECISION,
      source: "provider-inspection",
      inspectionReceiptId: BINDING.inspection_receipt_id,
    })).toThrow(/exactly matches a policy path.*cannot omit/i);
  });

  it("separates model-policy match, the selected candidate path, and evidence readiness", () => {
    const report: ValidationReport = {
      state: "dependency-pass",
      bindings: reportBindings(),
    };

    const presentation = buildModelPolicyPresentation({
      decision: DECISION,
      source: "provider-inspection",
      candidate: CANDIDATE,
      report,
      modelId: "provider/future-sparse",
      revision: REVISION,
      planId: PLAN_ID,
    });

    expect(presentation.artifactMatch).toMatchObject({
      state: "path-matched",
      policyId: "model.future-sparse.mlx-lora",
      policyVersion: "2.0.0",
      source: "provider-inspection",
    });
    expect(presentation.selectedPath).toMatchObject({
      state: "bound",
      candidateId: CANDIDATE_ID,
      bindingPathId: "future-sparse.mlx-lora.single",
      runtime: "mlx-lm",
      backend: "mps",
      method: "lora",
      distribution: "single",
    });
    expect(presentation.evidenceReadiness).toMatchObject({
      state: "validation-required",
      currentState: "dependency-pass",
      nextAction: "model-data",
      reportBoundToSelectedCandidate: true,
    });
  });

  it("keeps a null policy binding explicit while still presenting the candidate runtime path", () => {
    const candidate = {
      ...CANDIDATE,
      candidate_id: `cand_${"e".repeat(20)}`,
      status: "feasible",
      policy_binding: null,
      method: "qlora",
      runtime_contract: {
        ...CANDIDATE.runtime_contract!,
        training_runtime: "transformers-peft-cuda",
        compute_backend: "cuda",
        compiler_id: "transformers.peft-qlora.v2",
        estimator_id: "aptus-memory-v2",
        export_kind: "peft-adapter-safetensors",
      },
    } satisfies CandidatePlan;

    const presentation = buildModelPolicyPresentation({
      decision: DECISION,
      source: "user-attested",
      candidate,
      report: null,
      modelId: "provider/future-sparse",
      revision: "f".repeat(40),
      planId: PLAN_ID,
    });

    expect(presentation.selectedPath).toMatchObject({
      state: "unbound",
      runtime: "transformers-peft-cuda",
      backend: "cuda",
      method: "qlora",
    });
    expect(presentation.selectedPath.requiredValidationLevels).toEqual([]);
    expect(presentation.evidenceReadiness).toMatchObject({
      state: "not-applicable",
      nextAction: null,
      requiredValidationLevels: [],
    });
  });

  it("keeps mismatched evidence unbound and reports denied admission", () => {
    const otherCandidate = buildModelPolicyPresentation({
      decision: DECISION,
      source: "provider-inspection",
      candidate: CANDIDATE,
      report: {
        state: "pilot-pass",
        authorization_status: "current",
        authorization_current: true,
        bindings: reportBindings(`cand_${"8".repeat(20)}`),
      },
      modelId: "provider/future-sparse",
      revision: REVISION,
      planId: PLAN_ID,
    });
    expect(otherCandidate.evidenceReadiness).toMatchObject({
      state: "validation-required",
      currentState: null,
      reportBoundToSelectedCandidate: false,
    });

    const stale = buildModelPolicyPresentation({
      decision: DECISION,
      source: "provider-inspection",
      candidate: CANDIDATE,
      report: {
        state: "pilot-pass",
        authorization_status: "blocked",
        authorization_current: false,
        authorization_error: "The host policy changed; replan_required.",
        bindings: reportBindings(),
      },
      modelId: "provider/future-sparse",
      revision: REVISION,
      planId: PLAN_ID,
    });
    expect(stale.evidenceReadiness).toMatchObject({
      state: "authorization-blocked",
      currentState: "pilot-pass",
      reportBoundToSelectedCandidate: true,
    });
  });

  it("advances evidence readiness only through exact candidate-bound reports", () => {
    const states = [
      [null, "validation-required", "model-data"],
      ["dependency-pass", "validation-required", "model-data"],
      ["model-data-pass", "validation-required", "measured-preflight"],
      ["measured-preflight-pass", "validation-required", "pilot"],
      ["pilot-pass", "validation-complete", null],
    ] as const;

    for (const [reportState, expectedState, nextAction] of states) {
      const presentation = buildModelPolicyPresentation({
        decision: DECISION,
        source: "provider-inspection",
        candidate: CANDIDATE,
        report: reportState
          ? {
              state: reportState,
              bindings: reportBindings(),
            }
          : null,
        modelId: "provider/future-sparse",
        revision: REVISION,
        planId: PLAN_ID,
      });
      expect(presentation.evidenceReadiness).toMatchObject({
        state: expectedState,
        nextAction,
      });
    }
  });

  it("separates deferred admission, missing admission, and active authorization", () => {
    const cases = [
      [undefined, undefined, undefined, "validation-complete"],
      ["deferred", false, "Capacity will be checked later.", "admission-deferred"],
      ["current", true, undefined, "authorized"],
    ] as const;
    for (const [status, authorization, reason, state] of cases) {
      const presentation = buildModelPolicyPresentation({
        decision: DECISION,
        source: "provider-inspection",
        candidate: CANDIDATE,
        report: {
          state: "pilot-pass",
          bindings: reportBindings(),
          ...(status === undefined ? {} : { authorization_status: status }),
          ...(authorization === undefined ? {} : { authorization_current: authorization }),
          ...(reason === undefined ? {} : { authorization_error: reason }),
        },
        modelId: RECEIPT.model_id,
        revision: REVISION,
        planId: PLAN_ID,
      });
      expect(presentation.evidenceReadiness.state).toBe(state);
    }
  });

  it("keeps implementation blockers distinct from invalid validation evidence", () => {
    const implementationRuntime = {
      ...DECISION.paths[0].runtime_contract,
      compiler_id: null,
      evidence_requirement: "implementation-required" as const,
      export_kind: null,
    };
    const implementationDecision: ModelPolicyDecision = {
      ...DECISION,
      paths: [{ ...DECISION.paths[0], runtime_contract: implementationRuntime }],
    };
    const implementation = buildModelPolicyPresentation({
      decision: implementationDecision,
      source: "provider-inspection",
      candidate: { ...CANDIDATE, runtime_contract: implementationRuntime },
      report: null,
      modelId: RECEIPT.model_id,
      revision: REVISION,
      planId: PLAN_ID,
    });
    expect(implementation.evidenceReadiness.state).toBe("implementation-blocked");

    const invalid = buildModelPolicyPresentation({
      decision: DECISION,
      source: "provider-inspection",
      candidate: CANDIDATE,
      report: { state: "invalid", bindings: reportBindings() },
      modelId: RECEIPT.model_id,
      revision: REVISION,
      planId: PLAN_ID,
    });
    expect(invalid.evidenceReadiness.state).toBe("invalid");
  });

  it("ignores reports bound to the wrong plan or immutable model revision", () => {
    for (const bindings of [
      reportBindings(CANDIDATE_ID, `plan_${"8".repeat(20)}`),
      reportBindings(CANDIDATE_ID, PLAN_ID, "f".repeat(40)),
    ]) {
      const presentation = buildModelPolicyPresentation({
        decision: DECISION,
        source: "provider-inspection",
        candidate: CANDIDATE,
        report: {
          state: "pilot-pass",
          authorization_status: "current",
          authorization_current: true,
          bindings,
        },
        modelId: RECEIPT.model_id,
        revision: REVISION,
        planId: PLAN_ID,
      });
      expect(presentation.evidenceReadiness).toMatchObject({
        state: "validation-required",
        reportBoundToSelectedCandidate: false,
        authorizationCurrent: null,
      });
    }
  });

  it("runtime-decodes report state, bindings, and authorization coherence", () => {
    expect(decodeValidationReport({
      state: "pilot-pass",
      bindings: reportBindings(),
      authorization_status: "current",
      authorization_current: true,
      authorization_error: null,
    })).toMatchObject({
      state: "pilot-pass",
      authorization_status: "current",
      authorization_current: true,
    });

    for (const report of [
      { state: "future-pass", bindings: reportBindings() },
      { state: "pilot-pass", bindings: { ...reportBindings(), plan_id: {} } },
      { state: "pilot-pass", authorization_current: true },
      { state: "pilot-pass", authorization_status: "current" },
      {
        state: "dependency-pass",
        authorization_status: "current",
        authorization_current: true,
      },
      {
        state: "pilot-pass",
        authorization_status: "deferred",
        authorization_current: false,
      },
      {
        state: "pilot-pass",
        authorization_status: "blocked",
        authorization_current: true,
        authorization_error: "No longer admitted.",
      },
      { state: "pilot-pass", authorization_error: "Unscoped failure." },
    ]) {
      expect(() => decodeValidationReport(report)).toThrow(/validation report/i);
    }
  });

  it("rejects orphaned capacity evidence while preserving coherent and null capacity", () => {
    const capacity = {
      checked_at: "2026-08-04T12:00:00+00:00",
      free_disk_bytes: 100 * 1024 ** 3,
    };
    const authorized = {
      state: "pilot-pass",
      authorization_status: "current",
      authorization_current: true,
      authorization_error: null,
      prelaunch_capacity_check: capacity,
    };
    expect(decodeValidationReport(authorized)).toMatchObject({
      authorization_status: "current",
      prelaunch_capacity_check: capacity,
    });
    expect(decodeValidationReport({
      state: "static-pass",
      prelaunch_capacity_check: null,
    })).toMatchObject({ prelaunch_capacity_check: null });

    const orphaned = structuredClone(authorized) as Record<string, unknown>;
    delete orphaned.authorization_status;
    delete orphaned.authorization_current;
    delete orphaned.authorization_error;
    expect(() => decodeValidationReport(orphaned)).toThrow(
      /authorization fields require a typed authorization status/i,
    );
  });

  it("distinguishes an unsupported decision contract from a genuine blocked decision", () => {
    expect(() => decodeModelPolicyDecision({
      ...DECISION,
      schema_version: "aptus.model-compatibility.v3",
    })).toThrow(/unsupported model policy decision contract.*update aptus/i);

    const blocked = decodeModelPolicyDecision({
      ...DECISION,
      kind: "blocked",
      paths: [],
      reason_codes: ["identity-mismatch"],
      reason: "The pinned artifact does not match the registered identity.",
    });
    expect(blocked.kind).toBe("blocked");
  });
});
