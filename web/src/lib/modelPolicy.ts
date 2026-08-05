import type {
  CandidatePlan,
  ModelInspectionReceipt,
  ModelPolicyBinding,
  ModelPolicyBindingSource,
  ModelPolicyDecision,
  ModelPolicyDecisionKind,
  ModelPolicyPath,
  ValidationReport,
} from "../types";

const DECISION_KEYS = [
  "schema_version",
  "decision_id",
  "subject_facts_sha256",
  "kind",
  "family",
  "policy_id",
  "policy_version",
  "paths",
  "reason_codes",
  "evidence_ids",
  "reason",
] as const;

const PATH_KEYS = [
  "path_id",
  "method",
  "distribution",
  "adapter_profile_id",
  "target_modules",
  "runtime_contract",
  "required_validation_levels",
  "evidence_ids",
] as const;

const RUNTIME_KEYS = [
  "schema_version",
  "compute_backend",
  "training_runtime",
  "compiler_id",
  "estimator_id",
  "evidence_requirement",
  "export_kind",
] as const;

const BINDING_KEYS = [
  "schema_version",
  "decision_id",
  "subject_facts_sha256",
  "policy_id",
  "policy_version",
  "path_id",
  "source",
  "inspection_receipt_id",
  "reason_codes",
  "evidence_ids",
] as const;

const RECEIPT_KEYS = [
  "schema_version",
  "receipt_id",
  "model_id",
  "resolved_revision",
  "observed_facts_sha256",
  "decision",
  "provenance_summary",
  "provenance_requirement",
  "provenance_requirement_met",
  "evaluated_at",
] as const;

const PROVENANCE_KEYS = [
  "field",
  "kind",
  "source",
  "observed_at",
  "resolved_revision",
] as const;

const DECISION_KINDS = new Set<ModelPolicyDecisionKind>([
  "path-matched",
  "family-recognized",
  "blocked",
  "unknown",
]);
const METHODS = new Set<ModelPolicyPath["method"]>([
  "full",
  "lora",
  "int8-lora",
  "qlora",
]);
const DISTRIBUTIONS = new Set<ModelPolicyPath["distribution"]>([
  "single",
  "ddp",
  "fsdp",
]);
const ADAPTER_PROFILES = new Set<NonNullable<ModelPolicyPath["adapter_profile_id"]>>([
  "attention-qkvo.v1",
  "dense-causal-lm.v1",
]);
const BACKENDS = new Set<ModelPolicyPath["runtime_contract"]["compute_backend"]>([
  "cuda",
  "rocm",
  "mps",
  "cpu",
]);
const TRAINING_RUNTIMES = new Set<ModelPolicyPath["runtime_contract"]["training_runtime"]>([
  "transformers-peft-cuda",
  "mlx-lm",
  "pytorch-mps",
]);
const EVIDENCE_REQUIREMENTS = new Set<ModelPolicyPath["runtime_contract"]["evidence_requirement"]>([
  "pilot-required",
  "implementation-required",
]);
const VALIDATION_LEVELS = new Set<ModelPolicyValidationLevel>([
  "model-data",
  "measured-preflight",
  "pilot",
]);
const REASON_CODES = new Set<ModelPolicyDecision["reason_codes"][number]>([
  "exact-reviewed-artifact",
  "reviewed-runtime-path",
  "pilot-not-yet-proven",
  "invalid-compatibility-facts",
  "identity-mismatch",
  "layer-count-mismatch",
  "quantization-layout-mismatch",
  "topology-incomplete",
  "dense-topology-required",
  "shared-expert-unsupported",
  "four-bit-required",
  "family-recognized",
  "unreviewed-sparse-model",
  "no-policy-match",
]);

const VALIDATION_STATE_RANK = new Map<string, number>([
  ["invalid", 0],
  ["contract-pass", 1],
  ["static-pass", 2],
  ["dependency-pass", 3],
  ["model-data-pass", 4],
  ["measured-preflight-pass", 5],
  ["pilot-pass", 6],
  ["execution-approved", 7],
  ["measured-run-pass", 8],
]);

const VALIDATION_LEVEL_STATE = {
  "model-data": "model-data-pass",
  "measured-preflight": "measured-preflight-pass",
  pilot: "pilot-pass",
} as const;

const CANDIDATE_STATUSES = new Set([
  "feasible",
  "conditional",
  "infeasible",
  "unsupported",
] as const);

const VALIDATION_STATES = new Set([
  "invalid",
  "unsupported",
  "contract-pass",
  "static-pass",
  "dependency-pass",
  "model-data-pass",
  "measured-preflight-pass",
  "pilot-pass",
  "execution-approved",
  "measured-run-pass",
] as const);

const AUTHORIZABLE_VALIDATION_STATES = new Set([
  "pilot-pass",
  "execution-approved",
  "measured-run-pass",
]);

const AUTHORIZATION_STATUSES = new Set([
  "current",
  "deferred",
  "blocked",
] as const);

export type ModelPolicyValidationLevel =
  | "model-data"
  | "measured-preflight"
  | "pilot";

export interface ModelPolicyArtifactMatchPresentation {
  state: ModelPolicyDecisionKind;
  decisionId: string;
  subjectFactsSha256: string;
  modelId: string;
  revision: string;
  family: string | null;
  policyId: string | null;
  policyVersion: string | null;
  source: ModelPolicyBindingSource;
  reasonCodes: string[];
  evidenceIds: string[];
  reason: string;
}

export interface ModelPolicySelectedPathPresentation {
  state: "not-selected" | "bound" | "unbound";
  candidateId: string | null;
  decisionId: string;
  bindingPathId: string | null;
  source: ModelPolicyBindingSource | null;
  runtime: string | null;
  backend: string | null;
  method: string | null;
  distribution: string | null;
  adapterProfileId: string | null;
  targetModules: string[];
  requiredValidationLevels: ModelPolicyValidationLevel[];
  evidenceIds: string[];
}

export interface ModelPolicyEvidenceReadinessPresentation {
  state:
    | "not-applicable"
    | "validation-required"
    | "validation-complete"
    | "admission-deferred"
    | "authorized"
    | "authorization-blocked"
    | "implementation-blocked"
    | "invalid";
  currentState: string | null;
  nextAction: ModelPolicyValidationLevel | null;
  requiredValidationLevels: ModelPolicyValidationLevel[];
  reportBoundToSelectedCandidate: boolean;
  candidateRejected: boolean;
  authorizationStatus: "current" | "deferred" | "blocked" | null;
  authorizationCurrent: boolean | null;
  blocker: string | null;
}

export interface ValidationReportBindingIdentity {
  planId: string;
  candidateId: string;
  modelRevision: string;
}

export interface ModelPolicyPresentation {
  artifactMatch: ModelPolicyArtifactMatchPresentation;
  selectedPath: ModelPolicySelectedPathPresentation;
  evidenceReadiness: ModelPolicyEvidenceReadinessPresentation;
}

interface BindingDecodeContext {
  decision: ModelPolicyDecision;
  source: ModelPolicyBindingSource;
  inspectionReceiptId: string | null;
  candidate?: CandidatePlan;
}

interface ReceiptDecodeContext {
  modelId?: string;
  resolvedRevision?: string;
  decision?: ModelPolicyDecision;
}

interface BuildModelPolicyPresentationInput {
  decision: ModelPolicyDecision;
  source: ModelPolicyBindingSource;
  candidate: CandidatePlan | null;
  report: ValidationReport | null;
  modelId: string;
  revision: string;
  planId?: string | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function invalid(label: string, detail: string): never {
  throw new Error(`Aptus returned an invalid ${label}: ${detail}`);
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) invalid(label, "expected an object.");
  return value;
}

function requireExactKeys(
  value: Record<string, unknown>,
  keys: readonly string[],
  label: string,
): void {
  const expected = new Set(keys);
  const missing = keys.filter((key) => !(key in value));
  const extra = Object.keys(value).filter((key) => !expected.has(key));
  if (missing.length || extra.length) {
    invalid(
      label,
      [
        missing.length ? `missing ${missing.join(", ")}` : "",
        extra.length ? `unexpected ${extra.join(", ")}` : "",
      ].filter(Boolean).join("; ") + ".",
    );
  }
}

function requireUnpaddedText(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0 || value.trim() !== value) {
    invalid(label, "expected non-empty unpadded text.");
  }
  return value;
}

function requireNullableUnpaddedText(value: unknown, label: string): string | null {
  return value === null ? null : requireUnpaddedText(value, label);
}

function requirePattern(value: unknown, pattern: RegExp, label: string): string {
  if (typeof value !== "string" || !pattern.test(value)) {
    invalid(label, "value does not match the required identity format.");
  }
  return value;
}

function requireProviderModelId(value: unknown, label: string): string {
  const modelId = requirePattern(
    value,
    /^(?:[A-Za-z0-9][A-Za-z0-9._-]*\/)?[A-Za-z0-9][A-Za-z0-9._-]*$/,
    label,
  );
  if (modelId.includes("..") || modelId.includes("--") || modelId.endsWith(".git")) {
    invalid(label, "expected a provider repository identifier.");
  }
  return modelId;
}

function requireEnum<T extends string>(
  value: unknown,
  allowed: ReadonlySet<T>,
  label: string,
): T {
  if (typeof value !== "string" || !allowed.has(value as T)) {
    invalid(label, `unknown value ${JSON.stringify(value)}.`);
  }
  return value as T;
}

function requireTextArray(
  value: unknown,
  label: string,
  options: {
    nonEmpty?: boolean;
    allowed?: ReadonlySet<string>;
  } = {},
): string[] {
  if (!Array.isArray(value)) invalid(label, "expected a list.");
  const result = value.map((item, index) => {
    const text = requireUnpaddedText(item, `${label} item ${index}`);
    if (options.allowed && !options.allowed.has(text)) {
      invalid(label, `unknown value ${JSON.stringify(text)}.`);
    }
    return text;
  });
  if (options.nonEmpty && result.length === 0) {
    invalid(label, "expected at least one item.");
  }
  if (new Set(result).size !== result.length) {
    invalid(label, "items must be unique.");
  }
  return result;
}

function sameTextArray(left: readonly string[] | undefined, right: readonly string[]): boolean {
  return Boolean(
    left
    && left.length === right.length
    && left.every((item, index) => item === right[index]),
  );
}

function decodeRuntimeContract(value: unknown, pathLabel: string): ModelPolicyPath["runtime_contract"] {
  const label = `${pathLabel} runtime contract`;
  const record = requireRecord(value, label);
  requireExactKeys(record, RUNTIME_KEYS, label);
  if (record.schema_version !== "aptus.runtime-contract.v1") {
    invalid(label, "unsupported schema version.");
  }
  const computeBackend = requireEnum(record.compute_backend, BACKENDS, `${label} backend`);
  const trainingRuntime = requireEnum(
    record.training_runtime,
    TRAINING_RUNTIMES,
    `${label} training runtime`,
  );
  const expectedBackend = {
    "transformers-peft-cuda": "cuda",
    "mlx-lm": "mps",
    "pytorch-mps": "mps",
  }[trainingRuntime];
  if (computeBackend !== expectedBackend) {
    invalid(label, `${trainingRuntime} requires ${expectedBackend} compute.`);
  }
  const compilerId = requireNullableUnpaddedText(record.compiler_id, `${label} compiler ID`);
  const estimatorId = requireUnpaddedText(record.estimator_id, `${label} estimator ID`);
  const evidenceRequirement = requireEnum(
    record.evidence_requirement,
    EVIDENCE_REQUIREMENTS,
    `${label} evidence requirement`,
  );
  const exportKind = requireNullableUnpaddedText(record.export_kind, `${label} export kind`);
  if (evidenceRequirement === "pilot-required" && (!compilerId || !exportKind)) {
    invalid(label, "pilot-gated paths require compiler and export identities.");
  }
  return {
    schema_version: "aptus.runtime-contract.v1",
    compute_backend: computeBackend,
    training_runtime: trainingRuntime,
    compiler_id: compilerId,
    estimator_id: estimatorId,
    evidence_requirement: evidenceRequirement,
    export_kind: exportKind,
  };
}

interface CandidateDecodeContext {
  decision: ModelPolicyDecision;
  source: ModelPolicyBindingSource;
  inspectionReceiptId: string | null;
  requireRejected?: boolean;
}

function candidateMatchesPath(
  candidate: CandidatePlan,
  path: ModelPolicyPath,
): boolean {
  return candidate.method === path.method
    && candidate.distribution === path.distribution
    && sameTextArray(candidate.target_modules, path.target_modules)
    && runtimeContractsMatch(candidate.runtime_contract, path.runtime_contract);
}

export function decodePlanCandidate(
  value: unknown,
  context: CandidateDecodeContext,
): CandidatePlan {
  const label = "plan candidate";
  const record = requireRecord(value, label);
  const candidateId = requirePattern(
    record.candidate_id,
    /^cand_[0-9a-f]{20}$/,
    `${label} ID`,
  );
  const decisionId = requirePattern(
    record.model_policy_decision_id,
    /^compat_[0-9a-f]{20}$/,
    `${label} decision ID`,
  );
  if (decisionId !== context.decision.decision_id) {
    invalid(label, "decision link differs from the response policy decision.");
  }
  if (!("policy_binding" in record)) {
    invalid(label, "an explicit nullable policy binding is required.");
  }
  const method = requireEnum(record.method, METHODS, `${label} method`);
  const distribution = requireEnum(
    record.distribution,
    DISTRIBUTIONS,
    `${label} distribution`,
  );
  const status = requireEnum(record.status, CANDIDATE_STATUSES, `${label} status`);
  if (typeof record.feasible !== "boolean") {
    invalid(label, "feasible must be boolean.");
  }
  const feasible = record.feasible;
  const statusIsFeasible = status === "feasible" || status === "conditional";
  if (feasible !== statusIsFeasible) {
    invalid(label, "status and feasibility disagree.");
  }
  if (context.requireRejected && (feasible || statusIsFeasible)) {
    invalid(label, "no-feasible-plan rows must be rejected.");
  }
  const rejectionReasons = requireTextArray(
    record.rejection_reasons,
    `${label} rejection reasons`,
  );
  if (context.requireRejected && rejectionReasons.length === 0) {
    invalid(label, "no-feasible-plan rows require a rejection reason.");
  }
  const targetModules = requireTextArray(
    record.target_modules,
    `${label} target modules`,
  );
  const runtimeContract = decodeRuntimeContract(
    record.runtime_contract,
    `${label} execution`,
  );
  const candidate = {
    ...record,
    candidate_id: candidateId,
    id: candidateId,
    model_policy_decision_id: decisionId,
    method,
    distribution,
    status,
    feasible,
    rejection_reasons: rejectionReasons,
    target_modules: targetModules,
    runtime_contract: runtimeContract,
    policy_binding: null,
  } as CandidatePlan;
  const binding = record.policy_binding === null
    ? null
    : decodeModelPolicyBinding(record.policy_binding, {
        decision: context.decision,
        source: context.source,
        inspectionReceiptId: context.inspectionReceiptId,
        candidate,
      });
  if (
    binding === null
    && context.decision.paths.some((path) => candidateMatchesPath(candidate, path))
  ) {
    invalid(
      label,
      "an execution tuple that exactly matches a policy path cannot omit its binding.",
    );
  }
  return { ...candidate, policy_binding: binding };
}

export function decodeValidationReport(value: unknown): ValidationReport {
  const label = "validation report";
  const record = requireRecord(value, label);
  const state = requireEnum(record.state, VALIDATION_STATES, `${label} state`);
  let bindings: Record<string, string> | undefined;
  if (record.bindings !== undefined) {
    const rawBindings = requireRecord(record.bindings, `${label} bindings`);
    bindings = Object.fromEntries(
      Object.entries(rawBindings).map(([key, item]) => [
        requireUnpaddedText(key, `${label} binding name`),
        requireUnpaddedText(item, `${label} binding ${key}`),
      ]),
    );
  }
  const authorizationStatus = record.authorization_status === undefined
    || record.authorization_status === null
    ? null
    : requireEnum(
        record.authorization_status,
        AUTHORIZATION_STATUSES,
        `${label} authorization status`,
      );
  const authorizationCurrent = record.authorization_current === undefined
    || record.authorization_current === null
    ? null
    : record.authorization_current;
  if (authorizationCurrent !== null && typeof authorizationCurrent !== "boolean") {
    invalid(label, "authorization_current must be boolean when supplied.");
  }
  const authorizationError = record.authorization_error === undefined
    || record.authorization_error === null
    ? null
    : requireUnpaddedText(record.authorization_error, `${label} authorization reason`);
  const prelaunchCapacityCheck = record.prelaunch_capacity_check === undefined
    || record.prelaunch_capacity_check === null
    ? record.prelaunch_capacity_check
    : requireRecord(
        record.prelaunch_capacity_check,
        `${label} prelaunch capacity check`,
      );
  if (authorizationStatus === null) {
    if (
      authorizationCurrent !== null
      || authorizationError !== null
      || prelaunchCapacityCheck != null
    ) {
      invalid(label, "authorization fields require a typed authorization status.");
    }
  } else if (authorizationCurrent === null) {
    invalid(label, "authorization status requires authorization_current.");
  } else if (authorizationStatus === "current") {
    if (
      authorizationCurrent !== true
      || !AUTHORIZABLE_VALIDATION_STATES.has(state)
      || authorizationError !== null
    ) {
      invalid(label, "current authorization requires qualifying evidence, a true current flag, and no error.");
    }
  } else if (authorizationCurrent !== false || authorizationError === null) {
    invalid(label, "deferred or blocked authorization requires a false current flag and a reason.");
  }
  return {
    ...record,
    state,
    ...(bindings ? { bindings } : {}),
    ...(record.authorization_status === undefined
      ? {}
      : { authorization_status: authorizationStatus }),
    ...(record.authorization_current === undefined
      ? {}
      : { authorization_current: authorizationCurrent }),
    ...(record.authorization_error === undefined
      ? {}
      : { authorization_error: authorizationError }),
    ...(record.prelaunch_capacity_check === undefined
      ? {}
      : { prelaunch_capacity_check: prelaunchCapacityCheck }),
  } as ValidationReport;
}

function decodeModelPolicyPath(value: unknown, index: number): ModelPolicyPath {
  const label = `model policy path ${index}`;
  const record = requireRecord(value, label);
  requireExactKeys(record, PATH_KEYS, label);
  const method = requireEnum(record.method, METHODS, `${label} method`);
  const adapterProfile = record.adapter_profile_id === null
    ? null
    : requireEnum(record.adapter_profile_id, ADAPTER_PROFILES, `${label} adapter profile`);
  if ((method === "full") !== (adapterProfile === null)) {
    invalid(label, "adapter methods require a profile and full tuning forbids one.");
  }
  const requiredValidationLevels = requireTextArray(
    record.required_validation_levels,
    `${label} validation levels`,
    { nonEmpty: true, allowed: VALIDATION_LEVELS },
  ) as ModelPolicyValidationLevel[];
  return {
    path_id: requireUnpaddedText(record.path_id, `${label} ID`),
    method,
    distribution: requireEnum(record.distribution, DISTRIBUTIONS, `${label} distribution`),
    adapter_profile_id: adapterProfile,
    target_modules: requireTextArray(record.target_modules, `${label} target modules`, {
      nonEmpty: true,
    }),
    runtime_contract: decodeRuntimeContract(record.runtime_contract, label),
    required_validation_levels: requiredValidationLevels,
    evidence_ids: requireTextArray(record.evidence_ids, `${label} evidence IDs`, {
      nonEmpty: true,
    }),
  };
}

export function decodeModelPolicyDecision(value: unknown): ModelPolicyDecision {
  const record = requireRecord(value, "model policy decision");
  if (record.schema_version !== "aptus.model-compatibility.v2") {
    throw new Error(
      `Unsupported model policy decision contract ${JSON.stringify(record.schema_version)}. `
      + "Update Aptus so the workbench and local service use the same contract.",
    );
  }
  requireExactKeys(record, DECISION_KEYS, "model policy decision");
  const kind = requireEnum(record.kind, DECISION_KINDS, "model policy decision kind");
  const family = requireNullableUnpaddedText(record.family, "model policy decision family");
  const policyId = requireNullableUnpaddedText(record.policy_id, "model policy decision policy ID");
  const policyVersion = record.policy_version === null
    ? null
    : requirePattern(
        record.policy_version,
        /^[0-9]+\.[0-9]+\.[0-9]+$/,
        "model policy decision policy version",
      );
  if ((policyId === null) !== (policyVersion === null)) {
    invalid("model policy decision", "policy ID and version must be supplied together.");
  }
  if (!Array.isArray(record.paths)) {
    invalid("model policy decision paths", "expected a list.");
  }
  const paths = record.paths.map(decodeModelPolicyPath);
  if (new Set(paths.map((path) => path.path_id)).size !== paths.length) {
    invalid("model policy decision paths", "path IDs must be unique.");
  }
  if (kind === "path-matched") {
    if (!family || !policyId || paths.length === 0) {
      invalid(
        "model policy decision",
        "path-matched decisions require family, policy identity, and a path.",
      );
    }
  } else if (paths.length > 0) {
    invalid("model policy decision", "only path-matched decisions may carry paths.");
  }
  if (kind === "family-recognized" && !family) {
    invalid("model policy decision", "family-recognized decisions require a family.");
  }
  if ((kind === "family-recognized" || kind === "unknown") && policyId !== null) {
    invalid("model policy decision", "unregistered decisions cannot claim a policy identity.");
  }

  return {
    schema_version: "aptus.model-compatibility.v2",
    decision_id: requirePattern(
      record.decision_id,
      /^compat_[0-9a-f]{20}$/,
      "model policy decision ID",
    ),
    subject_facts_sha256: requirePattern(
      record.subject_facts_sha256,
      /^[0-9a-f]{64}$/,
      "model policy subject digest",
    ),
    kind,
    family,
    policy_id: policyId,
    policy_version: policyVersion,
    paths,
    reason_codes: requireTextArray(record.reason_codes, "model policy reason codes", {
      nonEmpty: true,
      allowed: REASON_CODES,
    }) as ModelPolicyDecision["reason_codes"],
    evidence_ids: requireTextArray(record.evidence_ids, "model policy evidence IDs"),
    reason: requireUnpaddedText(record.reason, "model policy decision reason"),
  };
}

function runtimeContractsMatch(
  left: CandidatePlan["runtime_contract"],
  right: ModelPolicyPath["runtime_contract"],
): boolean {
  return Boolean(
    left
    && left.schema_version === right.schema_version
    && left.compute_backend === right.compute_backend
    && left.training_runtime === right.training_runtime
    && left.compiler_id === right.compiler_id
    && left.estimator_id === right.estimator_id
    && left.evidence_requirement === right.evidence_requirement
    && left.export_kind === right.export_kind,
  );
}

export function decodeModelPolicyBinding(
  value: unknown,
  context: BindingDecodeContext,
): ModelPolicyBinding {
  const label = "model policy binding";
  const record = requireRecord(value, label);
  requireExactKeys(record, BINDING_KEYS, label);
  if (record.schema_version !== "aptus.model-policy-binding.v1") {
    invalid(label, "unsupported schema version.");
  }
  const decisionId = requirePattern(record.decision_id, /^compat_[0-9a-f]{20}$/, `${label} decision ID`);
  const subjectDigest = requirePattern(record.subject_facts_sha256, /^[0-9a-f]{64}$/, `${label} subject digest`);
  const policyId = requireUnpaddedText(record.policy_id, `${label} policy ID`);
  const policyVersion = requirePattern(
    record.policy_version,
    /^[0-9]+\.[0-9]+\.[0-9]+$/,
    `${label} policy version`,
  );
  const pathId = requireUnpaddedText(record.path_id, `${label} path ID`);
  const source = requireEnum(
    record.source,
    new Set<ModelPolicyBindingSource>(["provider-inspection", "user-attested"]),
    `${label} source`,
  );
  const inspectionReceiptId = record.inspection_receipt_id === null
    ? null
    : requirePattern(
        record.inspection_receipt_id,
        /^receipt_[0-9a-f]{20}$/,
        `${label} receipt ID`,
      );
  if (source === "provider-inspection" && inspectionReceiptId === null) {
    invalid(label, "provider-inspection bindings require a receipt ID.");
  }
  if (source === "user-attested" && inspectionReceiptId !== null) {
    invalid(label, "user-attested bindings cannot claim a receipt ID.");
  }
  const decision = context.decision;
  const path = decision.paths.find((candidate) => candidate.path_id === pathId);
  if (
    decisionId !== decision.decision_id
    || subjectDigest !== decision.subject_facts_sha256
    || policyId !== decision.policy_id
    || policyVersion !== decision.policy_version
    || source !== context.source
    || inspectionReceiptId !== context.inspectionReceiptId
    || !path
  ) {
    invalid(label, "identity, source, receipt, or path differs from the plan decision.");
  }
  if (context.candidate) {
    const candidate = context.candidate;
    if (
      candidate.model_policy_decision_id !== decisionId
      || candidate.method !== path.method
      || candidate.distribution !== path.distribution
      || !sameTextArray(candidate.target_modules, path.target_modules)
      || !runtimeContractsMatch(candidate.runtime_contract, path.runtime_contract)
    ) {
      invalid(label, "selected path differs from its candidate execution contract.");
    }
  }
  const reasonCodes = requireTextArray(record.reason_codes, `${label} reason codes`, {
    nonEmpty: true,
    allowed: REASON_CODES,
  }) as ModelPolicyBinding["reason_codes"];
  const evidenceIds = requireTextArray(record.evidence_ids, `${label} evidence IDs`, {
    nonEmpty: true,
  });
  const expectedEvidenceIds = [...new Set([...decision.evidence_ids, ...path.evidence_ids])];
  if (
    !sameTextArray(reasonCodes, decision.reason_codes)
    || !sameTextArray(evidenceIds, expectedEvidenceIds)
  ) {
    invalid(label, "reason codes or evidence IDs differ from the selected policy path.");
  }
  return {
    schema_version: "aptus.model-policy-binding.v1",
    decision_id: decisionId,
    subject_facts_sha256: subjectDigest,
    policy_id: policyId,
    policy_version: policyVersion,
    path_id: pathId,
    source,
    inspection_receipt_id: inspectionReceiptId,
    reason_codes: reasonCodes,
    evidence_ids: evidenceIds,
  };
}

function decisionsShareIdentity(
  left: ModelPolicyDecision,
  right: ModelPolicyDecision,
): boolean {
  return left.decision_id === right.decision_id
    && left.subject_facts_sha256 === right.subject_facts_sha256
    && left.kind === right.kind
    && left.family === right.family
    && left.policy_id === right.policy_id
    && left.policy_version === right.policy_version
    && JSON.stringify(left.paths) === JSON.stringify(right.paths)
    && sameTextArray(left.reason_codes, right.reason_codes)
    && sameTextArray(left.evidence_ids, right.evidence_ids);
}

function requireTimestamp(value: unknown, label: string): string {
  const text = requireUnpaddedText(value, label);
  if (!/^\d{4}-\d{2}-\d{2}T.+(?:Z|[+-]\d{2}:\d{2})$/.test(text) || Number.isNaN(Date.parse(text))) {
    invalid(label, "expected an ISO-8601 timestamp with a timezone.");
  }
  return text;
}

export function decodeModelInspectionReceipt(
  value: unknown,
  context: ReceiptDecodeContext = {},
): ModelInspectionReceipt {
  const label = "model inspection receipt";
  const record = requireRecord(value, label);
  requireExactKeys(record, RECEIPT_KEYS, label);
  if (record.schema_version !== "aptus.model-inspection-receipt.v1") {
    invalid(label, "unsupported schema version.");
  }
  const modelId = requireProviderModelId(record.model_id, `${label} model ID`);
  const resolvedRevision = requirePattern(
    record.resolved_revision,
    /^[0-9a-fA-F]{40,64}$/,
    `${label} resolved revision`,
  );
  const decision = decodeModelPolicyDecision(record.decision);
  if (context.modelId !== undefined && modelId !== context.modelId) {
    invalid(label, "model ID differs from the inspection response.");
  }
  if (
    context.resolvedRevision !== undefined
    && resolvedRevision.toLowerCase() !== context.resolvedRevision.toLowerCase()
  ) {
    invalid(label, "revision differs from the inspection response.");
  }
  if (context.decision && !decisionsShareIdentity(decision, context.decision)) {
    invalid(label, "decision differs from the plan policy decision.");
  }
  if (!Array.isArray(record.provenance_summary) || record.provenance_summary.length === 0) {
    invalid(`${label} provenance`, "expected at least one observation.");
  }
  const provenanceSummary = record.provenance_summary.map((item, index) => {
    const itemLabel = `${label} provenance item ${index}`;
    const provenance = requireRecord(item, itemLabel);
    requireExactKeys(provenance, PROVENANCE_KEYS, itemLabel);
    const provenanceRevision = requirePattern(
      provenance.resolved_revision,
      /^[0-9a-fA-F]{40,64}$/,
      `${itemLabel} revision`,
    );
    if (provenanceRevision.toLowerCase() !== resolvedRevision.toLowerCase()) {
      invalid(itemLabel, "revision differs from the receipt.");
    }
    return {
      field: requireUnpaddedText(provenance.field, `${itemLabel} field`),
      kind: requireEnum(
        provenance.kind,
        new Set<ModelInspectionReceipt["provenance_summary"][number]["kind"]>([
          "provider-declared",
          "inferred",
        ]),
        `${itemLabel} kind`,
      ),
      source: requireUnpaddedText(provenance.source, `${itemLabel} source`),
      observed_at: requireTimestamp(provenance.observed_at, `${itemLabel} observed time`),
      resolved_revision: provenanceRevision,
    };
  });
  const fields = provenanceSummary.map((item) => item.field);
  if (
    new Set(fields).size !== fields.length
    || fields.some((field, index) => index > 0 && fields[index - 1] > field)
  ) {
    invalid(`${label} provenance`, "fields must be sorted and unique.");
  }
  const hasProviderDeclaredProvenance = provenanceSummary.some(
    (item) => item.kind === "provider-declared",
  );
  if (!hasProviderDeclaredProvenance) {
    invalid(`${label} provenance`, "expected at least one provider-declared observation.");
  }
  const provenanceRequirement = record.provenance_requirement === null
    ? null
    : requireEnum(
        record.provenance_requirement,
        new Set<NonNullable<ModelInspectionReceipt["provenance_requirement"]>>([
          "provider-declared",
        ]),
        `${label} provenance requirement`,
      );
  if (typeof record.provenance_requirement_met !== "boolean") {
    invalid(`${label} provenance requirement`, "met flag must be boolean.");
  }
  if (record.provenance_requirement_met && provenanceRequirement === null) {
    invalid(`${label} provenance requirement`, "a met requirement must name its kind.");
  }
  if (
    record.provenance_requirement_met
    && provenanceRequirement === "provider-declared"
    && !hasProviderDeclaredProvenance
  ) {
    invalid(
      `${label} provenance requirement`,
      "provider-declared provenance cannot be satisfied by inferred observations.",
    );
  }
  if (
    decision.kind === "path-matched"
    && (
      provenanceRequirement !== "provider-declared"
      || record.provenance_requirement_met !== true
    )
  ) {
    invalid(
      `${label} provenance requirement`,
      "path-matched provider decisions require satisfied provider-declared provenance.",
    );
  }
  return {
    schema_version: "aptus.model-inspection-receipt.v1",
    receipt_id: requirePattern(record.receipt_id, /^receipt_[0-9a-f]{20}$/, `${label} ID`),
    model_id: modelId,
    resolved_revision: resolvedRevision,
    observed_facts_sha256: requirePattern(
      record.observed_facts_sha256,
      /^[0-9a-f]{64}$/,
      `${label} observed-facts digest`,
    ),
    decision,
    provenance_summary: provenanceSummary,
    provenance_requirement: provenanceRequirement,
    provenance_requirement_met: record.provenance_requirement_met,
    evaluated_at: requireTimestamp(record.evaluated_at, `${label} evaluation time`),
  };
}

function policyPathForBinding(
  decision: ModelPolicyDecision,
  binding: ModelPolicyBinding | null,
): ModelPolicyPath | null {
  return binding
    ? decision.paths.find((path) => path.path_id === binding.path_id) ?? null
    : null;
}

function nextRequiredLevel(
  levels: readonly ModelPolicyValidationLevel[],
  currentState: string | null,
): ModelPolicyValidationLevel | null {
  const currentRank = VALIDATION_STATE_RANK.get(currentState ?? "invalid") ?? 0;
  return levels.find(
    (level) => currentRank < (VALIDATION_STATE_RANK.get(VALIDATION_LEVEL_STATE[level]) ?? 0),
  ) ?? null;
}

export function validationReportMatchesBinding(
  report: ValidationReport | null | undefined,
  identity: ValidationReportBindingIdentity | null | undefined,
): boolean {
  return Boolean(
    report
    && identity
    && report.bindings?.plan_id === identity.planId
    && report.bindings.candidate_id === identity.candidateId
    && report.bindings.model_revision === identity.modelRevision,
  );
}

export function buildModelPolicyPresentation({
  decision,
  source,
  candidate,
  report,
  modelId,
  revision,
  planId,
}: BuildModelPolicyPresentationInput): ModelPolicyPresentation {
  const artifactMatch: ModelPolicyArtifactMatchPresentation = {
    state: decision.kind,
    decisionId: decision.decision_id,
    subjectFactsSha256: decision.subject_facts_sha256,
    modelId,
    revision,
    family: decision.family,
    policyId: decision.policy_id,
    policyVersion: decision.policy_version,
    source,
    reasonCodes: [...decision.reason_codes],
    evidenceIds: [...decision.evidence_ids],
    reason: decision.reason,
  };

  if (!candidate) {
    return {
      artifactMatch,
      selectedPath: {
        state: "not-selected",
        candidateId: null,
        decisionId: decision.decision_id,
        bindingPathId: null,
        source: null,
        runtime: null,
        backend: null,
        method: null,
        distribution: null,
        adapterProfileId: null,
        targetModules: [],
        requiredValidationLevels: [],
        evidenceIds: [],
      },
      evidenceReadiness: {
        state: "not-applicable",
        currentState: null,
        nextAction: null,
        requiredValidationLevels: [],
        reportBoundToSelectedCandidate: false,
        candidateRejected: false,
        authorizationStatus: null,
        authorizationCurrent: null,
        blocker: null,
      },
    };
  }

  const binding = candidate.policy_binding;
  const path = policyPathForBinding(decision, binding);
  const requiredValidationLevels = path ? [...path.required_validation_levels] : [];
  const reportStateKnown = Boolean(
    report?.state && VALIDATION_STATES.has(report.state as never),
  );
  const reportBoundToSelectedCandidate = Boolean(
    path
    && planId
    && reportStateKnown
    && validationReportMatchesBinding(report, {
      planId,
      candidateId: candidate.candidate_id,
      modelRevision: revision,
    }),
  );
  const currentState = reportBoundToSelectedCandidate ? report?.state ?? null : null;
  const authorizationStatus = reportBoundToSelectedCandidate
    ? report?.authorization_status ?? null
    : null;
  const authorizationCurrent = reportBoundToSelectedCandidate
    ? report?.authorization_current ?? null
    : null;
  const candidateRejected = candidate.feasible === false
    || candidate.status === "infeasible"
    || candidate.status === "unsupported";
  const nextAction = candidateRejected
    ? null
    : nextRequiredLevel(requiredValidationLevels, currentState);
  const implementationBlocked = candidate.runtime_contract?.evidence_requirement === "implementation-required";
  const invalidEvidence = currentState === "invalid" || currentState === "unsupported";
  const evidenceState: ModelPolicyEvidenceReadinessPresentation["state"] = !path
    ? "not-applicable"
    : candidateRejected
      ? "not-applicable"
      : implementationBlocked
        ? "implementation-blocked"
        : invalidEvidence
          ? "invalid"
          : nextAction !== null
            ? "validation-required"
            : authorizationStatus === "current"
              ? "authorized"
              : authorizationStatus === "deferred"
                ? "admission-deferred"
                : authorizationStatus === "blocked"
                  ? "authorization-blocked"
                  : "validation-complete";
  const blocker = implementationBlocked
    ? "This runtime path requires an implementation before evidence can authorize execution."
    : invalidEvidence
      ? "The bound validation report is invalid or unsupported."
      : authorizationStatus === "deferred" || authorizationStatus === "blocked"
        ? report?.authorization_error ?? "Launch admission was not granted."
        : null;

  return {
    artifactMatch,
    selectedPath: {
      state: binding ? "bound" : "unbound",
      candidateId: candidate.candidate_id,
      decisionId: candidate.model_policy_decision_id,
      bindingPathId: binding?.path_id ?? null,
      source: binding?.source ?? null,
      runtime: candidate.runtime_contract?.training_runtime ?? null,
      backend: candidate.runtime_contract?.compute_backend ?? null,
      method: candidate.method ?? null,
      distribution: candidate.distribution ?? null,
      adapterProfileId: path?.adapter_profile_id ?? null,
      targetModules: [...(candidate.target_modules ?? [])],
      requiredValidationLevels,
      evidenceIds: [...(binding?.evidence_ids ?? [])],
    },
    evidenceReadiness: {
      state: evidenceState,
      currentState,
      nextAction,
      requiredValidationLevels,
      reportBoundToSelectedCandidate,
      candidateRejected,
      authorizationStatus,
      authorizationCurrent,
      blocker,
    },
  };
}
