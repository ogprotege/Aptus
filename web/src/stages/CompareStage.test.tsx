import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { CandidatePlan, NoFeasibleComparisonPlan, TrainingPolicy } from "../types";
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

const trainingPolicy: TrainingPolicy = {
  schema_version: "aptus.training-policy.v1",
  policy_version: "aptus-training-policy-v1",
  knobs: [
    {
      name: "rank",
      value: "16",
      prior_kind: "objective-and-token-volume-prior",
      rationale:
        "Adapter rank 16 is the Aptus v0.2 objective and dataset-volume prior, not a tuned optimum.",
    },
    {
      name: "learning_rate",
      value: "0.0002",
      prior_kind: "method-class-prior",
      rationale:
        "Learning rate 0.0002 is an Aptus v0.2 method-class prior, not a tuned optimum.",
    },
  ],
  non_claims: ["These knobs are not a prediction of model quality."],
};

/** Path Alpha presentation: 4 rows, 1 epoch → supervision prior (conditional). */
const pathAlphaTrainingPolicy: TrainingPolicy = {
  schema_version: "aptus.training-policy.v1",
  policy_version: "aptus-training-policy-v1",
  knobs: [
    {
      name: "rank",
      value: "16",
      prior_kind: "objective-and-token-volume-prior",
      rationale:
        "Adapter rank 16 is the Aptus v0.2 objective and dataset-volume prior, not a tuned optimum.",
    },
    {
      name: "epochs",
      value: "1",
      prior_kind: "method-class-prior",
      rationale:
        "Dataset example_count is below the instruction-SFT supervision prior of 100 rows; this is not a justified domain adaptation.",
    },
    {
      name: "dataset_size",
      value: "4",
      prior_kind: "method-class-prior",
      rationale:
        "Dataset example_count is below the instruction-SFT supervision prior of 100 rows; this is not a justified domain adaptation.",
    },
  ],
  non_claims: ["These knobs are not a prediction of model quality."],
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

const unsupported: CandidatePlan = {
  ...rejected,
  candidate_id: `cand_${"d".repeat(20)}`,
  status: "unsupported",
  rejection_reasons: ["The method registry does not list this distribution."],
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

  it("keeps unsupported candidates visible with the blocked evidence class", () => {
    render(
      <CompareStage
        plan={{ ...noFeasiblePlan, candidates: [unsupported] }}
        selected={unsupported}
        busy={null}
        demoMode={false}
        modelPolicyPresentation={null}
        onInspectCandidate={vi.fn()}
        onSelectCandidate={vi.fn(async () => undefined)}
        onCompile={vi.fn(async () => undefined)}
        onReturnToFacts={vi.fn()}
      />,
    );
    expect(screen.getAllByText(/unsupported/).length).toBeGreaterThan(0);
    expect(document.querySelector(".evidence-blocked")).not.toBeNull();
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

  it("shows training-knob priors without optimal claims", () => {
    render(
      <CompareStage
        plan={{ ...noFeasiblePlan, training_policy: trainingPolicy }}
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

    const region = screen.getByRole("region", { name: "Why these training knobs" });
    expect(region).toHaveTextContent(/prior/i);
    expect(region).not.toHaveTextContent(/optimal/i);
  });

  it("shows supervision prior for 4-row / 1-epoch Path Alpha fixture", () => {
    render(
      <CompareStage
        plan={{ ...noFeasiblePlan, training_policy: pathAlphaTrainingPolicy }}
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

    const region = screen.getByRole("region", { name: "Why these training knobs" });
    expect(region).toHaveTextContent(/below the instruction-SFT supervision prior of 100 rows/i);
    expect(region).toHaveTextContent("Dataset size");
    expect(region).toHaveTextContent("Epochs");
    expect(region).not.toHaveTextContent(/this dataset will produce a sycophant/i);
    expect(region).not.toHaveTextContent(/3 epochs is optimal/i);
    expect(region).not.toHaveTextContent(/optimal/i);
  });
});
