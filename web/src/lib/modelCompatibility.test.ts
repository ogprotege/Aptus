import { describe, expect, it } from "vitest";
import {
  MALFORMED_COMPATIBILITY_REASON,
  normalizeModelCompatibility,
} from "./modelCompatibility";

const CONDITIONAL = {
  status: "conditional",
  family: "qwen3_moe",
  supported_runtime: "mlx-lm",
  compute_backend: "mps",
  supported_methods: ["qlora"],
  distribution: "single",
  evidence_requirement: "pilot-required",
  adapter_profile_id: "attention-qkvo.v1",
  reason: "A measured pilot remains required.",
};

const RECOGNIZED = {
  status: "recognized",
  family: "llama",
  supported_runtime: null,
  compute_backend: null,
  supported_methods: [],
  distribution: null,
  evidence_requirement: "pilot-required",
  adapter_profile_id: null,
  reason: "The planner decides the executable path.",
};

const UNSUPPORTED = {
  status: "unsupported",
  family: null,
  supported_runtime: null,
  compute_backend: null,
  supported_methods: [],
  distribution: null,
  evidence_requirement: "implementation-required",
  adapter_profile_id: null,
  reason: "No reviewed policy matches this model.",
};

describe("normalizeModelCompatibility", () => {
  it("preserves every coherent status variant", () => {
    for (const value of [CONDITIONAL, RECOGNIZED, UNSUPPORTED]) {
      expect(normalizeModelCompatibility(value)).toEqual(value);
    }
  });

  it("keeps absent compatibility evidence unknown", () => {
    expect(normalizeModelCompatibility(null)).toBeNull();
    expect(normalizeModelCompatibility(undefined)).toBeNull();
  });

  it("turns incomplete or contradictory evidence into canonical unsupported data", () => {
    const malformed = [
      { ...CONDITIONAL, supported_runtime: null },
      { ...CONDITIONAL, supported_runtime: "" },
      { ...CONDITIONAL, supported_runtime: "not-a-runtime" },
      { ...CONDITIONAL, supported_runtime: " mlx-lm" },
      { ...CONDITIONAL, compute_backend: null },
      { ...CONDITIONAL, compute_backend: "not-a-backend" },
      { ...CONDITIONAL, compute_backend: "cuda" },
      { ...CONDITIONAL, supported_methods: [] },
      { ...CONDITIONAL, supported_methods: [""] },
      { ...CONDITIONAL, supported_methods: ["not-a-method"] },
      { ...CONDITIONAL, supported_methods: ["qlora", "qlora"] },
      {
        ...CONDITIONAL,
        supported_runtime: "transformers-peft-cuda",
        compute_backend: "cuda",
        supported_methods: ["full"],
      },
      { ...CONDITIONAL, distribution: null },
      { ...CONDITIONAL, distribution: "" },
      { ...CONDITIONAL, distribution: "not-a-placement" },
      { ...CONDITIONAL, evidence_requirement: "implementation-required" },
      { ...CONDITIONAL, adapter_profile_id: null },
      { ...CONDITIONAL, adapter_profile_id: "" },
      { ...CONDITIONAL, adapter_profile_id: "not-a-profile" },
      { ...CONDITIONAL, family: "" },
      { ...CONDITIONAL, family: "   " },
      { ...CONDITIONAL, family: " qwen3_moe" },
      { ...CONDITIONAL, reason: "" },
      { ...CONDITIONAL, reason: "\t" },
      { ...CONDITIONAL, reason: " Pilot required. " },
      { ...CONDITIONAL, extra: true },
      { ...RECOGNIZED, supported_runtime: "mlx-lm" },
      { ...RECOGNIZED, compute_backend: "mps" },
      { ...RECOGNIZED, supported_methods: ["lora"] },
      { ...RECOGNIZED, distribution: "single" },
      { ...RECOGNIZED, adapter_profile_id: "attention-qkvo.v1" },
      { ...UNSUPPORTED, evidence_requirement: "pilot-required" },
      { ...UNSUPPORTED, compute_backend: "mps" },
      { ...UNSUPPORTED, adapter_profile_id: "attention-qkvo.v1" },
    ];

    for (const value of malformed) {
      expect(normalizeModelCompatibility(value)).toEqual({
        status: "unsupported",
        family: null,
        supported_runtime: null,
        compute_backend: null,
        supported_methods: [],
        distribution: null,
        evidence_requirement: "implementation-required",
        adapter_profile_id: null,
        reason: MALFORMED_COMPATIBILITY_REASON,
      });
    }
  });
});
