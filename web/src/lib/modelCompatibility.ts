import type { ModelCompatibility } from "../types";

const COMPATIBILITY_KEYS = new Set([
  "status",
  "family",
  "supported_runtime",
  "supported_methods",
  "distribution",
  "evidence_requirement",
  "adapter_scope",
  "reason",
]);

export const MALFORMED_COMPATIBILITY_REASON =
  "Aptus received incomplete or contradictory compatibility evidence. Execution support is blocked until the model is inspected again.";

function malformedCompatibility(): ModelCompatibility {
  return {
    status: "unsupported",
    family: null,
    supported_runtime: null,
    supported_methods: [],
    distribution: null,
    evidence_requirement: "implementation-required",
    adapter_scope: null,
    reason: MALFORMED_COMPATIBILITY_REASON,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function hasExactContractKeys(value: Record<string, unknown>): boolean {
  const keys = Object.keys(value);
  return keys.length === COMPATIBILITY_KEYS.size
    && keys.every((key) => COMPATIBILITY_KEYS.has(key));
}

function hasNoExecutionClaim(value: Record<string, unknown>): boolean {
  return value.supported_runtime === null
    && Array.isArray(value.supported_methods)
    && value.supported_methods.length === 0
    && value.distribution === null
    && value.adapter_scope === null;
}

export function normalizeModelCompatibility(value: unknown): ModelCompatibility | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (!isRecord(value) || !hasExactContractKeys(value) || !isNonEmptyString(value.reason)) {
    return malformedCompatibility();
  }

  if (
    value.status === "conditional"
    && isNonEmptyString(value.family)
    && isNonEmptyString(value.supported_runtime)
    && Array.isArray(value.supported_methods)
    && value.supported_methods.length > 0
    && value.supported_methods.every(isNonEmptyString)
    && isNonEmptyString(value.distribution)
    && value.evidence_requirement === "pilot-required"
    && isNonEmptyString(value.adapter_scope)
  ) {
    return value as ModelCompatibility;
  }

  if (
    value.status === "recognized"
    && isNonEmptyString(value.family)
    && value.evidence_requirement === "pilot-required"
    && hasNoExecutionClaim(value)
  ) {
    return value as ModelCompatibility;
  }

  if (
    value.status === "unsupported"
    && (value.family === null || isNonEmptyString(value.family))
    && value.evidence_requirement === "implementation-required"
    && hasNoExecutionClaim(value)
  ) {
    return value as ModelCompatibility;
  }

  return malformedCompatibility();
}
