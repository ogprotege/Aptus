import { describe, expect, it } from "vitest";
import type { CandidatePlan } from "../types";
import {
  canStartAction,
  expectedMemory,
  formatBytes,
  isPassingValidation,
  memoryComponents,
  nextForwardAction,
  upperMemory,
} from "./plan";

describe("plan memory normalization", () => {
  const candidate: CandidatePlan = {
    candidate_id: `cand_${"a".repeat(20)}`,
    model_policy_decision_id: `compat_${"b".repeat(20)}`,
    policy_binding: null,
    method: "lora",
    distribution: "single",
    status: "conditional",
    feasible: true,
    rejection_reasons: [],
    target_modules: ["q_proj"],
    runtime_contract: {
      schema_version: "aptus.runtime-contract.v1",
      compute_backend: "mps",
      training_runtime: "mlx-lm",
      compiler_id: "mlx-lm.lora.v1",
      estimator_id: "aptus-memory-mlx-v2",
      evidence_requirement: "pilot-required",
      export_kind: "mlx-lm-adapter",
    },
    memory: {
      point_estimate_bytes: 10 * 1024 ** 3,
      upper_estimate_bytes: 12 * 1024 ** 3,
      base_weights_bytes: 8 * 1024 ** 3,
      activations_bytes: 2 * 1024 ** 3,
      component_upper_bounds: {
        base_weights_bytes: 8 * 1024 ** 3,
        activations_bytes: 3 * 1024 ** 3,
        uncertainty_bytes: 1 * 1024 ** 3,
      },
    },
  };

  it("reads v2 point and upper estimates without combining them", () => {
    expect(expectedMemory(candidate)).toBe(10 * 1024 ** 3);
    expect(upperMemory(candidate)).toBe(12 * 1024 ** 3);
    expect(formatBytes(upperMemory(candidate))).toBe("12.0 GiB");
  });

  it("keeps point components and the named upper uncertainty", () => {
    const components = memoryComponents(candidate);
    expect(components.find((item) => item.label === "Base weights")?.upper_bytes).toBe(8 * 1024 ** 3);
    expect(components.find((item) => item.label === "Uncertainty")?.upper_bytes).toBe(1 * 1024 ** 3);
  });
});

describe("execution gates", () => {
  it("keeps runtime gates sequential before it allows a full training job", () => {
    expect(canStartAction("static-pass", "dependency")).toBe(true);
    expect(canStartAction("static-pass", "preflight")).toBe(false);
    expect(canStartAction("model-data-pass", "preflight")).toBe(true);
    expect(canStartAction("static-pass", "train")).toBe(false);
    expect(canStartAction("execution-approved", "train")).toBe(true);
    expect(nextForwardAction("static-pass")).toBe("dependency");
    expect(nextForwardAction("dependency-pass")).toBe("model-data");
    expect(nextForwardAction("model-data-pass")).toBe("preflight");
    expect(nextForwardAction("measured-preflight-pass")).toBe("pilot");
    expect(nextForwardAction("pilot-pass")).toBe("train");
  });

  it("fails closed for generic or legacy report states", () => {
    for (const state of ["valid", "passed", "environment-pass", "smoke-pass"]) {
      expect(isPassingValidation(state)).toBe(false);
      expect(canStartAction(state, "preflight")).toBe(false);
      expect(canStartAction(state, "train")).toBe(false);
    }
  });
});
