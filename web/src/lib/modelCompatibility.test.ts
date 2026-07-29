import { describe, expect, it } from "vitest";
import {
  MALFORMED_COMPATIBILITY_REASON,
  normalizeModelCompatibility,
} from "./modelCompatibility";

const CONDITIONAL = {
  status: "conditional",
  family: "qwen3_moe",
  supported_runtime: "mlx-lm",
  supported_methods: ["qlora"],
  distribution: "single",
  evidence_requirement: "pilot-required",
  adapter_scope: "attention-only",
  reason: "A measured pilot remains required.",
};

const RECOGNIZED = {
  status: "recognized",
  family: "llama",
  supported_runtime: null,
  supported_methods: [],
  distribution: null,
  evidence_requirement: "pilot-required",
  adapter_scope: null,
  reason: "The planner decides the executable path.",
};

const UNSUPPORTED = {
  status: "unsupported",
  family: null,
  supported_runtime: null,
  supported_methods: [],
  distribution: null,
  evidence_requirement: "implementation-required",
  adapter_scope: null,
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
      { ...CONDITIONAL, supported_methods: [] },
      { ...CONDITIONAL, supported_methods: [""] },
      { ...CONDITIONAL, distribution: null },
      { ...CONDITIONAL, distribution: "" },
      { ...CONDITIONAL, evidence_requirement: "implementation-required" },
      { ...CONDITIONAL, adapter_scope: null },
      { ...CONDITIONAL, adapter_scope: "" },
      { ...CONDITIONAL, family: "" },
      { ...CONDITIONAL, reason: "" },
      { ...CONDITIONAL, extra: true },
      { ...RECOGNIZED, supported_runtime: "mlx-lm" },
      { ...UNSUPPORTED, evidence_requirement: "pilot-required" },
    ];

    for (const value of malformed) {
      expect(normalizeModelCompatibility(value)).toEqual({
        status: "unsupported",
        family: null,
        supported_runtime: null,
        supported_methods: [],
        distribution: null,
        evidence_requirement: "implementation-required",
        adapter_scope: null,
        reason: MALFORMED_COMPATIBILITY_REASON,
      });
    }
  });
});
