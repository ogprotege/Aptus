import type {
  CandidatePlan,
  CompileResponse,
  FactDraft,
  InputProfile,
  Job,
  ModelPolicyDecision,
  TrainingPlan,
  ValidationReport,
} from "./types";

const GiB = 1024 ** 3;
const EXAMPLE_POLICY_DECISION_ID = `compat_${"e".repeat(20)}`;

const EXAMPLE_POLICY_DECISION: ModelPolicyDecision = {
  schema_version: "aptus.model-compatibility.v2",
  decision_id: EXAMPLE_POLICY_DECISION_ID,
  subject_facts_sha256: "e".repeat(64),
  kind: "family-recognized",
  family: "llama",
  policy_id: null,
  policy_version: null,
  paths: [],
  reason_codes: ["family-recognized"],
  evidence_ids: [],
  reason: "The labeled example uses a recognized dense model family without a registered artifact-specific policy.",
};

export const EMPTY_DRAFT: FactDraft = {
  project_name: "",
  model: {
    model_id: "",
    revision: "",
    family: "",
    parameters_b: null,
    hidden_size: null,
    layers: null,
    context_length: null,
    intermediate_size: null,
    license_name: "",
    training_allowed: false,
    model_type: null,
    architecture: null,
    quantization_bits: null,
    quantization_layout: null,
    moe: null,
    active_parameters_b: null,
    sparse_layer_count: null,
  },
  dataset: {
    source_path: "",
    format: "jsonl",
    schema_name: "text",
    tokenizer_id: "",
    sample_limit: null,
  },
  hardware: {
    discovery: "manual",
    gpu_count: 1,
    devices: [
      {
        name: "",
        backend: "cuda",
        total_vram_gib: null,
        free_vram_gib: null,
        supports_bf16: false,
        supports_8bit: false,
        supports_4bit: false,
      },
    ],
    host_ram_gib: null,
    host_ram_free_gib: null,
    reserve_per_device_gib: 2,
    disk_free_gib: null,
  },
  target: {
    task: "sft",
    objective: "quality",
    sequence_length: null,
    effective_batch_size: 16,
    max_epochs: 3,
    method_preference: "",
    runtime: "transformers-peft-cuda",
    evaluation_fraction: 0.1,
    packing: false,
    checkpoint_steps: 100,
  },
};

export const EXAMPLE_DRAFT: FactDraft = {
  project_name: "Example support adapter",
  model: {
    model_id: "meta-llama/example-7b",
    revision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    family: "llama",
    parameters_b: 7,
    hidden_size: 4096,
    layers: 32,
    context_length: 4096,
    intermediate_size: 11008,
    license_name: "Example license entry",
    training_allowed: true,
    model_type: null,
    architecture: null,
    quantization_bits: null,
    quantization_layout: null,
    moe: null,
    active_parameters_b: null,
    sparse_layer_count: null,
  },
  dataset: {
    source_path: "/data/example-support.jsonl",
    format: "jsonl",
    schema_name: "text",
    tokenizer_id: "meta-llama/example-7b",
    sample_limit: 1000,
  },
  hardware: {
    discovery: "manual",
    gpu_count: 1,
    devices: [
      {
        name: "Example CUDA GPU",
        backend: "cuda",
        total_vram_gib: 24,
        free_vram_gib: 24,
        supports_bf16: true,
        supports_8bit: true,
        supports_4bit: true,
      },
    ],
    host_ram_gib: 64,
    host_ram_free_gib: 48,
    reserve_per_device_gib: 2,
    disk_free_gib: 500,
  },
  target: {
    task: "sft",
    objective: "quality",
    sequence_length: 128,
    effective_batch_size: 8,
    max_epochs: 1,
    method_preference: "",
    runtime: "transformers-peft-cuda",
    evaluation_fraction: 0.1,
    packing: false,
    checkpoint_steps: 100,
  },
};

export const EXAMPLE_PROFILE: InputProfile = {
  facts: [
    {
      key: "dataset_hash",
      label: "Dataset fingerprint",
      value: "da7d…fb0a",
      provenance: "example",
      source: "Labeled interface example",
      confidence: "high",
    },
    {
      key: "sequence_p95",
      label: "Sequence p95",
      value: 29,
      unit: "tokens",
      provenance: "example",
      source: "Labeled interface example",
      confidence: "low",
    },
    {
      key: "limiting_vram",
      label: "Usable per-device VRAM",
      value: 22,
      unit: "GiB",
      provenance: "example",
      source: "24 GiB minus 2 GiB reserve",
      confidence: "high",
    },
  ],
  warnings: [
    "Example values illustrate the interface. Aptus did not inspect a real model, dataset, or device.",
  ],
};

const loraComponents = [
  { label: "Base weights", expected_bytes: 14_000_000_000 },
  { label: "Adapter weights", expected_bytes: 192_937_984 },
  { label: "Adapter gradients", expected_bytes: 192_937_984 },
  { label: "Optimizer states", expected_bytes: 385_875_968 },
  { label: "Activations", expected_bytes: 805_306_368 },
  { label: "Temporary overhead", expected_bytes: 1_120_000_000 },
  { label: "Safety margin", expected_bytes: 2_504_558_746 },
];

const qloraComponents = [
  { label: "Base weights", expected_bytes: 3_500_000_000 },
  { label: "Quantization metadata", expected_bytes: 111_125_000 },
  { label: "Adapter weights", expected_bytes: 192_937_984 },
  { label: "Adapter gradients", expected_bytes: 192_937_984 },
  { label: "Optimizer states", expected_bytes: 385_875_968 },
  { label: "Activations", expected_bytes: 805_306_368 },
  { label: "Temporary overhead", expected_bytes: 1_073_741_824 },
  { label: "Safety margin", expected_bytes: 939_288_769 },
];

const fullComponents = [
  { label: "Base weights", expected_bytes: 14_000_000_000 },
  { label: "Gradients", expected_bytes: 14_000_000_000 },
  { label: "Optimizer states", expected_bytes: 56_000_000_000 },
  { label: "Activations", expected_bytes: 805_306_368 },
  { label: "Temporary overhead", expected_bytes: 2_800_000_000 },
];

function exampleCandidate(
  method: "full" | "lora" | "int8-lora" | "qlora",
  distribution: "single" | "ddp" | "fsdp",
  status: "feasible" | "infeasible" | "unsupported",
  expectedBytes: number,
  upperBytes: number,
  rejectionReasons: string[] = [],
): CandidatePlan {
  const isQuantized = method === "int8-lora" || method === "qlora";
  const components = method === "full"
    ? fullComponents
    : method === "lora"
      ? loraComponents
      : qloraComponents;
  const evidenceId = method === "qlora"
    ? "method.qlora.paper"
    : method === "full"
      ? "method.full.transformers"
      : "method.lora.paper";
  const requiredDiskBytes = method === "full"
    ? 250_000_000_000
    : method === "lora"
      ? 24_000_000_000
      : 22_000_000_000;
  return {
    id: `example-${method}-${distribution}`,
    candidate_id: `example-${method}-${distribution}`,
    method,
    distribution,
    status,
    feasible: status === "feasible",
    precision: "bf16",
    quantization: method === "qlora"
      ? "nf4-double-quant"
      : method === "int8-lora"
        ? "int8-bitsandbytes"
        : null,
    batches: {
      micro_batch_size: 8,
      gradient_accumulation_steps: 1,
      effective_batch_size: 8,
    },
    memory: {
      expected_bytes: expectedBytes,
      upper_bytes: upperBytes,
      limit_bytes: 22 * GiB,
      device_total_bytes: 24 * GiB,
      components,
    },
    assumptions: [
      "The memory estimate is an uncalibrated planning example.",
      "The fit check uses the candidate's bound device set, never aggregate VRAM.",
    ],
    evidence: [evidenceId, "estimate.memory.v2"],
    confidence: "uncalibrated-pilot-required",
    rejection_reasons: rejectionReasons,
    rank: method === "full" ? 0 : 16,
    alpha: method === "full" ? 0 : 32,
    learning_rate: method === "full" ? 0.00002 : 0.0002,
    world_size: distribution === "single" ? 1 : 1,
    target_modules: method === "full" ? [] : ["q_proj", "k_proj", "v_proj", "o_proj"],
    pareto_frontier: status === "feasible" && distribution === "single",
    ranking_basis: [
      "Example quality objective policy.",
      "No model-quality or throughput value was fabricated.",
    ],
    required_host_ram_bytes: 15_400_000_000,
    required_disk_bytes: requiredDiskBytes,
    checkpoint_retention_bytes: method === "full" ? 210_000_000_000 : 1_250_000_000,
    final_export_bytes: method === "full" ? 14_000_000_000 : 86_000_000,
    model_policy_decision_id: EXAMPLE_POLICY_DECISION_ID,
    policy_binding: null,
    quantized: isQuantized,
  };
}

const EXAMPLE_CANDIDATES: CandidatePlan[] = [
  exampleCandidate("full", "single", "infeasible", 86_805_306_368, 104_000_000_000, ["Even the point estimate exceeds usable per-device VRAM."]),
  exampleCandidate("full", "ddp", "unsupported", 86_805_306_368, 104_000_000_000, ["DDP requires at least two CUDA devices."]),
  exampleCandidate("full", "fsdp", "unsupported", 45_000_000_000, 58_000_000_000, ["FSDP requires at least two CUDA devices."]),
  exampleCandidate("lora", "single", "feasible", 19_201_617_050, 20_669_890_560),
  exampleCandidate("lora", "ddp", "unsupported", 19_201_617_050, 20_669_890_560, ["DDP requires at least two CUDA devices."]),
  exampleCandidate("lora", "fsdp", "unsupported", 12_900_000_000, 15_400_000_000, ["FSDP requires at least two CUDA devices."]),
  exampleCandidate("int8-lora", "single", "feasible", 10_700_000_000, 12_400_000_000),
  exampleCandidate("int8-lora", "ddp", "unsupported", 10_700_000_000, 12_400_000_000, ["DDP requires at least two CUDA devices."]),
  exampleCandidate("int8-lora", "fsdp", "unsupported", 10_700_000_000, 12_400_000_000, ["8-bit LoRA with FSDP is unsupported in v0.2."]),
  exampleCandidate("qlora", "single", "feasible", 7_201_213_897, 8_697_308_774),
  exampleCandidate("qlora", "ddp", "unsupported", 7_201_213_897, 8_697_308_774, ["DDP requires at least two CUDA devices."]),
  exampleCandidate("qlora", "fsdp", "unsupported", 7_201_213_897, 8_697_308_774, ["QLoRA with FSDP is unsupported in v0.2."]),
];

const EXAMPLE_RECOMMENDED = EXAMPLE_CANDIDATES[3];

export const EXAMPLE_PLAN: TrainingPlan = {
  schema_version: "aptus.training-plan.v5",
  plan_id: "plan_eeeeeeeeeeeeeeeeeeee",
  model_policy_snapshot_sha256: "e".repeat(64),
  example: true,
  model_policy_decision: EXAMPLE_POLICY_DECISION,
  model_policy_decision_source: "user-attested",
  inspection_receipt: null,
  recommended: EXAMPLE_RECOMMENDED,
  candidates: EXAMPLE_CANDIDATES,
  warnings: [
    "Example data only. No claim about real model fit or expected quality is being made.",
  ],
  rationale: [
    "The example chooses LoRA because full fine-tuning is infeasible and the quality policy prefers LoRA among the viable adapter methods.",
    "QLoRA remains the lower-memory viable alternative.",
  ],
  recommendation_rationale: [
    "The example chooses LoRA because full fine-tuning is infeasible and the quality policy prefers LoRA among the viable adapter methods.",
    "QLoRA remains the lower-memory viable alternative.",
  ],
  evidence_records: [
    {
      evidence_id: "method.full.transformers",
      claim: "Full causal-language-model training uses the Transformers trainer path.",
      source: "https://huggingface.co/docs/transformers/index",
      source_kind: "official-documentation",
      scope: "Runtime implementation prior, not a fit result.",
      confidence: "documented",
    },
    {
      evidence_id: "method.lora.paper",
      claim: "LoRA freezes base weights and trains low-rank adapter matrices.",
      source: "https://arxiv.org/abs/2106.09685",
      source_kind: "primary-paper",
      scope: "Method definition, not a universal rank or quality claim.",
      confidence: "documented",
    },
    {
      evidence_id: "method.qlora.paper",
      claim: "QLoRA combines a quantized base with trainable LoRA adapters.",
      source: "https://arxiv.org/abs/2305.14314",
      source_kind: "primary-paper",
      scope: "Method definition, not a hardware fit guarantee.",
      confidence: "documented",
    },
    {
      evidence_id: "estimate.memory.v2",
      claim: "Aptus applies its named analytical memory estimator.",
      source: "docs/methodology/memory-estimation.md",
      source_kind: "aptus-methodology",
      scope: "Uncalibrated point estimate and heuristic upper envelope.",
      confidence: "uncalibrated",
    },
  ],
};

export const EXAMPLE_REPORT: ValidationReport = {
  state: "pilot-pass",
  validation_level: "pilot",
  validator_version: "aptus-portable-validator-v2-example",
  validated_at: "2026-07-21T12:00:00+00:00",
  authorization_current: true,
  artifact_fingerprint: "example-a1d6e415ca3deefe26474d4f1967c3861081630e36b22a20b9c5d95a12b69249",
  bindings: {
    bundle: "example-bundle-digest",
    dataset: "example-dataset-digest",
    plan_id: "plan_eeeeeeeeeeeeeeeeeeee",
    candidate_id: "example-lora-single",
    model_revision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    environment: "example-environment-digest",
    hardware: "example-hardware-digest",
    pilot_metrics: "example-pilot-metrics-digest",
  },
  pilot_metrics: {
    checkpoint_continuation_observed: true,
    pilot_run_id: "pilot_example0001",
    pilot_run_dir: "runs/pilot_example0001",
    measured_checkpoint_bytes: 1_460_000_000,
    measured_final_export_bytes: 86_000_000,
    phase_one_checkpoint: {
      total_bytes: 1_420_000_000,
      manifest_sha256: "example-phase-one-checkpoint-manifest",
      files: [{ path: "trainer_state.json" }, { path: "optimizer.pt" }],
    },
    phase_two_checkpoint: {
      total_bytes: 1_460_000_000,
      manifest_sha256: "example-phase-two-checkpoint-manifest",
      files: [{ path: "trainer_state.json" }, { path: "optimizer.pt" }],
    },
    phase_one: {
      global_step: 1,
      train_loss: 2.41,
      measured_peak_cuda_bytes: 19_100_000_000,
      measured_reserved_cuda_bytes: 19_600_000_000,
    },
    phase_two_resumed: {
      global_step: 2,
      train_loss: 2.29,
      measured_peak_cuda_bytes: 19_200_000_000,
      measured_reserved_cuda_bytes: 19_700_000_000,
    },
  },
  gates: [
    { label: "Plan contract", state: "passed", detail: "Example gate result" },
    { label: "Generated Python", state: "passed", detail: "Example gate result" },
    { label: "Dependency set", state: "passed", detail: "Example gate result" },
    { label: "Model and data", state: "passed", detail: "Example gate result" },
    { label: "Synthetic method preflight", state: "passed", detail: "Example gate result" },
    { label: "Exact model and data pilot", state: "passed", detail: "Example two-phase checkpoint-continuation result" },
  ],
  findings: [],
  checked_files: [
    "README.md",
    "plan.json",
    "plan_contract.py",
    "requirements.txt",
    "config/accelerate.yaml",
    "config/trainer.json",
    "preflight.py",
    "train.py",
    "validate.py",
  ],
  runtime_evidence: [
    "Example only: the plan and dependency contract passed.",
    "Example only: a synthetic selected-method optimizer step passed.",
    "Example only: pilot phase one saved checkpoint-1 and a fresh phase resumed step 2.",
  ],
};

export const EXAMPLE_BUNDLE: CompileResponse = {
  bundle_dir: "/output/example-aptus-bundle",
  archive_path: "/output/example-aptus-bundle.zip",
  files: [
    "README.md",
    "bundle-manifest.json",
    "candidates.json",
    "config/accelerate.yaml",
    "config/trainer.json",
    "decision-report.md",
    "evidence.jsonl",
    "plan.json",
    "plan_contract.py",
    "preflight.py",
    "profiles/dataset.json",
    "profiles/hardware.json",
    "profiles/model.json",
    "requirements.txt",
    "runbook.md",
    "train.py",
    "validate.py",
  ],
  report: EXAMPLE_REPORT,
};

export const EXAMPLE_JOB: Job = {
  id: "example-job",
  bundle_dir: "/output/example-aptus-bundle",
  state: "completed",
  phase: "completed",
  mode: "pilot",
  log: [
    "$ aptus run /output/example-aptus-bundle --action pilot",
    "Example output only. No command ran from this interface.",
    "Example pilot saved checkpoint-1 and resumed step 2.",
  ],
  return_code: 0,
  validation_report: EXAMPLE_REPORT,
};
