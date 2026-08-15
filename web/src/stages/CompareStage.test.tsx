import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { CandidatePlan, NoFeasibleComparisonPlan } from "../types";
import { CompareStage } from "./CompareStage";

const decisionId = `compat_${"b".repeat(20)}`;
const subjectDigest = "a".repeat(64);

const rejected: CandidatePlan = {
  candidate_id: `cand_${"c".repeat(20)}`,
  model_policy_decision_id: decisionId,
  policy_binding: null,
  method: "lora",
  distribution: "single",
  status: "infeasible",
  feasible: false,
  rejection_reasons: ["Usable per-device memory is unknown."],
  target_modules: ["q_proj"],
  pareto_frontier: undefined,
  runtime_contract: {
    schema_version: "aptus.runtime-contract.v1",
    compute_backend: "cuda",
    training_runtime: "transformers-peft-cuda",
    compiler_id: "transformers.peft-lora.v2",
    estimator_id: "aptus-memory-v2",
    evidence_requirement: "pilot-required",
    export_kind: "peft-adapter-safetensors",
  },
};

const noFeasiblePlan: NoFeasibleComparisonPlan = {
  no_feasible_plan: true,
  recommended: null,
  candidates: [rejected],
  model_policy_decision: {
    schema_version: "aptus.model-compatibility.v2",
    decision_id: decisionId,
    subject_facts_sha256: subjectDigest,
    kind: "blocked",
    family: "llama",
    policy_id: null,
    policy_version: null,
    paths: [],
    reason_codes: [],
    evidence_ids: [],
    reason: "No viable path in this fixture.",
  },
  model_policy_decision_source: "user-attested",
  inspection_receipt: null,
  model: {
    model_id: "example/model",
    revision: "a".repeat(40),
    family: "llama",
    parameters: 1_000_000_000,
    hidden_size: 2048,
    layers: 24,
    context_length: 4096,
    license_name: "apache-2.0",
    training_allowed: true,
  },
  warnings: [],
  rationale: [],
  recommendation_rationale: [],
};

describe("CompareStage claim language", () => {
  it("does not call a missing recommendation a safe plan", () => {
    render(
      <CompareStage
        plan={noFeasiblePlan}
        selected={rejected}
        busy={null}
        demoMode={false}
        modelPolicyPresentation={null}
        onInspectCandidate={vi.fn()}
        onSelectCandidate={vi.fn(async () => undefined)}
        onCompile={vi.fn(async () => undefined)}
        onReturnToFacts={vi.fn()}
      />,
    );

    expect(
      screen.getByText("Aptus did not find a viable strategy under these facts."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/safe plan/i)).not.toBeInTheDocument();
  });

  it("does not treat a missing Pareto flag as No", () => {
    render(
      <CompareStage
        plan={noFeasiblePlan}
        selected={rejected}
        busy={null}
        demoMode={false}
        modelPolicyPresentation={null}
        onInspectCandidate={vi.fn()}
        onSelectCandidate={vi.fn(async () => undefined)}
        onCompile={vi.fn(async () => undefined)}
        onReturnToFacts={vi.fn()}
      />,
    );

    const frontier = screen.getByText(/ranking frontier/i).closest("div");
    expect(frontier).not.toBeNull();
    expect(frontier).toHaveTextContent("Not supplied");
    expect(frontier).not.toHaveTextContent(/^No$/);
  });
});
