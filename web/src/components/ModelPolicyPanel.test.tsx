import axe from "axe-core";
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type {
  CandidatePlan,
  ModelPolicyBinding,
  ModelPolicyDecision,
  ValidationReport,
} from "../types";
import {
  buildModelPolicyPresentation,
  decodeModelPolicyDecision,
} from "../lib/modelPolicy";
import { ModelPolicyPanel } from "./ModelPolicyPanel";

const DECISION_ID = `compat_${"a".repeat(20)}`;
const CANDIDATE_ID = `cand_${"b".repeat(20)}`;
const SUBJECT_DIGEST = "c".repeat(64);
const FUTURE_POLICY_ID = "model.future-sparse.mlx-lora";
const PLAN_ID = `plan_${"9".repeat(20)}`;
const REVISION = "e".repeat(40);

const DECISION: ModelPolicyDecision = {
  schema_version: "aptus.model-compatibility.v2",
  decision_id: DECISION_ID,
  subject_facts_sha256: SUBJECT_DIGEST,
  kind: "path-matched",
  family: "future_sparse_family",
  policy_id: FUTURE_POLICY_ID,
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
  policy_id: FUTURE_POLICY_ID,
  policy_version: "2.0.0",
  path_id: "future-sparse.mlx-lora.single",
  source: "provider-inspection",
  inspection_receipt_id: `receipt_${"d".repeat(20)}`,
  reason_codes: ["exact-reviewed-artifact", "pilot-not-yet-proven"],
  evidence_ids: ["runtime.future-sparse.mlx-lora.v1"],
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

function presentation(
  candidate: CandidatePlan | null = CANDIDATE,
  report: ValidationReport | null = null,
  decision: ModelPolicyDecision = DECISION,
) {
  return buildModelPolicyPresentation({
    decision,
    source: "provider-inspection",
    candidate,
    report,
    modelId: "provider/future-sparse",
    revision: REVISION,
    planId: PLAN_ID,
  });
}

function reportBindings(candidateId = CANDIDATE_ID): Record<string, string> {
  return {
    plan_id: PLAN_ID,
    candidate_id: candidateId,
    model_revision: REVISION,
  };
}

describe("ModelPolicyPanel", () => {
  it("renders the three server-owned records and an unfamiliar policy ID accessibly", async () => {
    const report: ValidationReport = {
      state: "dependency-pass",
      bindings: reportBindings(),
    };
    const { container } = render(
      <ModelPolicyPanel presentation={presentation(CANDIDATE, report)} />,
    );

    const match = screen.getByRole("article", { name: "Model-policy match" });
    const path = screen.getByRole("article", { name: "Selected candidate path" });
    const evidence = screen.getByRole("article", { name: "Evidence readiness" });

    expect(screen.getAllByRole("article")).toHaveLength(3);
    expect(within(match).getByText(FUTURE_POLICY_ID)).toBeInTheDocument();
    expect(within(match).getByText("v2.0.0")).toBeInTheDocument();
    expect(within(path).getByText("Bound")).toBeInTheDocument();
    expect(within(path).getByText("future-sparse.mlx-lora.single")).toBeInTheDocument();
    expect(within(path).getByText("mlx-lm")).toBeInTheDocument();
    expect(within(evidence).getByText("Evidence required")).toBeInTheDocument();
    expect(within(evidence).getByText("Model data")).toBeInTheDocument();

    const results = await axe.run(container, {
      rules: { "color-contrast": { enabled: false } },
    });
    expect(results.violations).toEqual([]);
  });

  it("renders a blocked decision while unsupported contracts still fail before presentation", () => {
    const blocked = decodeModelPolicyDecision({
      ...DECISION,
      kind: "blocked",
      paths: [],
      reason_codes: ["identity-mismatch"],
      reason: "The pinned artifact does not match the registered identity.",
    });
    render(<ModelPolicyPanel presentation={presentation(null, null, blocked)} />);

    const match = screen.getByRole("article", { name: "Model-policy match" });
    const path = screen.getByRole("article", { name: "Selected candidate path" });
    const evidence = screen.getByRole("article", { name: "Evidence readiness" });
    expect(within(match).getByText("Blocked")).toBeInTheDocument();
    expect(within(match).getByText(blocked.reason)).toBeInTheDocument();
    expect(within(match).getByText(/supported pinned artifact/i)).toBeInTheDocument();
    expect(within(path).getByText("Not selected")).toBeInTheDocument();
    expect(within(path).getByText(/select a candidate to review/i)).toBeInTheDocument();
    expect(within(evidence).getByText("Not applicable")).toBeInTheDocument();
    expect(screen.queryByText(/unsupported model policy decision contract/i)).not.toBeInTheDocument();

    expect(() => decodeModelPolicyDecision({
      ...DECISION,
      schema_version: "aptus.model-compatibility.v3",
    })).toThrow(/unsupported model policy decision contract/i);
  });

  it("shows an unbound candidate contract without claiming a policy path", () => {
    const unboundCandidate = {
      ...CANDIDATE,
      candidate_id: `cand_${"f".repeat(20)}`,
      policy_binding: null,
      method: "qlora",
      runtime_contract: {
        ...CANDIDATE.runtime_contract!,
        compute_backend: "cuda",
        training_runtime: "transformers-peft-cuda",
        compiler_id: "transformers.peft-qlora.v2",
        estimator_id: "aptus-memory-v2",
        export_kind: "peft-adapter-safetensors",
      },
    } satisfies CandidatePlan;
    render(<ModelPolicyPanel presentation={presentation(unboundCandidate)} />);

    const path = screen.getByRole("article", { name: "Selected candidate path" });
    expect(within(path).getByText("Unbound")).toBeInTheDocument();
    expect(within(path).getByText("No registered path")).toBeInTheDocument();
    expect(within(path).getByText("Transformers-peft-cuda", { exact: false })).toBeInTheDocument();
    expect(within(path).getByText("QLoRA")).toBeInTheDocument();
    expect(within(path).getByText(/no policy-path claim applies/i)).toBeInTheDocument();
  });

  it("does not present a different candidate's report as evidence", () => {
    render(
      <ModelPolicyPanel
        presentation={presentation(CANDIDATE, {
          state: "pilot-pass",
          authorization_status: "current",
          authorization_current: true,
          bindings: reportBindings(`cand_${"8".repeat(20)}`),
        })}
      />,
    );

    const evidence = screen.getByRole("article", { name: "Evidence readiness" });
    expect(within(evidence).getByText("Evidence required")).toBeInTheDocument();
    expect(within(evidence).getByText(/no validation report is bound/i)).toBeInTheDocument();
    expect(within(evidence).getByText("Not bound")).toBeInTheDocument();
    expect(within(evidence).getByText(/only a report bound to it can authorize/i)).toBeInTheDocument();
    expect(within(evidence).queryByText("Pilot pass")).not.toBeInTheDocument();
  });

  it("shows ordinary pilot admission as deferred without stale or replan copy", () => {
    render(
      <ModelPolicyPanel
        presentation={presentation(CANDIDATE, {
          state: "pilot-pass",
          authorization_status: "deferred",
          authorization_current: false,
          authorization_error: "Launch admission is intentionally deferred until submission.",
          bindings: reportBindings(),
        })}
      />,
    );

    const evidence = screen.getByRole("article", { name: "Evidence readiness" });
    expect(within(evidence).getByText("Admission deferred")).toBeInTheDocument();
    expect(within(evidence).getByText("Deferred")).toBeInTheDocument();
    expect(within(evidence).getByText(/intentionally deferred until submission/i)).toBeInTheDocument();
    expect(within(evidence).queryByText(/stale|create a new plan/i)).not.toBeInTheDocument();
  });

  it("does not label passing evidence with missing authorization as ready", () => {
    render(
      <ModelPolicyPanel
        presentation={presentation(CANDIDATE, {
          state: "pilot-pass",
          bindings: reportBindings(),
        })}
      />,
    );

    const evidence = screen.getByRole("article", { name: "Evidence readiness" });
    expect(within(evidence).getByText("Evidence complete")).toBeInTheDocument();
    expect(within(evidence).getByText("Not checked")).toBeInTheDocument();
    expect(within(evidence).getByText(/launch admission has not been checked/i)).toBeInTheDocument();
    expect(within(evidence).queryByText("Ready")).not.toBeInTheDocument();
  });
});
