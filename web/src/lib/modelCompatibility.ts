import type { ModelCompatibility } from "../types";

type ConditionalCompatibility = Extract<ModelCompatibility, { status: "conditional" }>;

const COMPATIBILITY_KEYS = new Set([
  "status",
  "family",
  "supported_runtime",
  "compute_backend",
  "supported_methods",
  "distribution",
  "evidence_requirement",
  "adapter_profile_id",
  "reason",
]);

const KNOWN_RUNTIME_IDS = {
  "transformers-peft-cuda": true,
  "mlx-lm": true,
  "pytorch-mps": true,
} as const satisfies Record<ConditionalCompatibility["supported_runtime"], true>;

const KNOWN_METHOD_IDS = {
  full: true,
  lora: true,
  "int8-lora": true,
  qlora: true,
} as const satisfies Record<ConditionalCompatibility["supported_methods"][number], true>;

const KNOWN_DISTRIBUTION_IDS = {
  single: true,
  ddp: true,
  fsdp: true,
} as const satisfies Record<ConditionalCompatibility["distribution"], true>;

const KNOWN_BACKEND_IDS = {
  cuda: true,
  rocm: true,
  mps: true,
  cpu: true,
} as const satisfies Record<ConditionalCompatibility["compute_backend"], true>;

const KNOWN_ADAPTER_PROFILE_IDS = {
  "attention-qkvo.v1": true,
} as const satisfies Record<ConditionalCompatibility["adapter_profile_id"], true>;

const ADAPTER_METHOD_IDS = {
  lora: true,
  "int8-lora": true,
  qlora: true,
} as const satisfies Partial<
  Record<ConditionalCompatibility["supported_methods"][number], true>
>;

const RUNTIME_BACKEND_IDS = {
  "transformers-peft-cuda": "cuda",
  "mlx-lm": "mps",
  "pytorch-mps": "mps",
} as const satisfies Record<
  ConditionalCompatibility["supported_runtime"],
  ConditionalCompatibility["compute_backend"]
>;

export const MALFORMED_COMPATIBILITY_REASON =
  "Aptus received incomplete or contradictory compatibility evidence. Execution support is blocked until the model is inspected again.";

function malformedCompatibility(): ModelCompatibility {
  return {
    status: "unsupported",
    family: null,
    supported_runtime: null,
    compute_backend: null,
    supported_methods: [],
    distribution: null,
    evidence_requirement: "implementation-required",
    adapter_profile_id: null,
    reason: MALFORMED_COMPATIBILITY_REASON,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value === value.trim();
}

function hasKnownId<T extends string>(
  value: unknown,
  knownIds: Readonly<Record<T, true>>,
): value is T {
  return typeof value === "string"
    && Object.prototype.hasOwnProperty.call(knownIds, value);
}

function hasKnownUniqueMethods(
  value: unknown,
): value is ConditionalCompatibility["supported_methods"] {
  return Array.isArray(value)
    && value.length > 0
    && value.every((method) => hasKnownId(method, KNOWN_METHOD_IDS))
    && value.every((method) => hasKnownId(method, ADAPTER_METHOD_IDS))
    && new Set(value).size === value.length;
}

function hasKnownRuntimeBackend(
  runtime: unknown,
  backend: unknown,
): boolean {
  return hasKnownId(runtime, KNOWN_RUNTIME_IDS)
    && hasKnownId(backend, KNOWN_BACKEND_IDS)
    && RUNTIME_BACKEND_IDS[runtime] === backend;
}

function hasExactContractKeys(value: Record<string, unknown>): boolean {
  const keys = Object.keys(value);
  return keys.length === COMPATIBILITY_KEYS.size
    && keys.every((key) => COMPATIBILITY_KEYS.has(key));
}

function hasNoExecutionClaim(value: Record<string, unknown>): boolean {
  return value.supported_runtime === null
    && value.compute_backend === null
    && Array.isArray(value.supported_methods)
    && value.supported_methods.length === 0
    && value.distribution === null
    && value.adapter_profile_id === null;
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
    && hasKnownRuntimeBackend(value.supported_runtime, value.compute_backend)
    && hasKnownUniqueMethods(value.supported_methods)
    && hasKnownId(value.distribution, KNOWN_DISTRIBUTION_IDS)
    && value.evidence_requirement === "pilot-required"
    && hasKnownId(value.adapter_profile_id, KNOWN_ADAPTER_PROFILE_IDS)
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
