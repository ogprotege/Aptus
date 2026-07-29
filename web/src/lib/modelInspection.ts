import type {
  InspectedMoETopology,
  ModelCompatibility,
  ModelFacts,
  ModelInspectionResponse,
  MoETopology,
  QuantizationLayout,
  TrainingPlan,
} from "../types";

const QWEN3_MOE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function matchesCurrentModel(
  model: Record<string, unknown>,
  current: ModelFacts,
): boolean {
  return model.model_id === current.model_id && model.revision === current.revision;
}

function completeMoETopology(
  value: InspectedMoETopology | null | undefined,
): MoETopology | null {
  if (!value || typeof value !== "object") return null;
  const positiveInteger = (candidate: unknown): candidate is number =>
    typeof candidate === "number" && Number.isInteger(candidate) && candidate > 0;
  if (
    !positiveInteger(value.expert_count)
    || !positiveInteger(value.experts_per_token)
    || value.experts_per_token > value.expert_count
    || !positiveInteger(value.expert_intermediate_size)
    || !positiveInteger(value.decoder_sparse_step)
    || !Array.isArray(value.mlp_only_layers)
    || value.mlp_only_layers.some(
      (layer, index, layers) =>
        !Number.isInteger(layer)
        || layer < 0
        || (index > 0 && layers[index - 1] >= layer),
    )
    || (
      value.shared_expert_intermediate_size !== null
      && value.shared_expert_intermediate_size !== undefined
      && !positiveInteger(value.shared_expert_intermediate_size)
    )
  ) {
    return null;
  }
  return {
    expert_count: value.expert_count,
    experts_per_token: value.experts_per_token,
    expert_intermediate_size: value.expert_intermediate_size,
    decoder_sparse_step: value.decoder_sparse_step,
    mlp_only_layers: [...value.mlp_only_layers],
    shared_expert_intermediate_size: value.shared_expert_intermediate_size ?? null,
  };
}

function completeQuantizationLayout(
  value: QuantizationLayout | null | undefined,
): QuantizationLayout | null {
  if (!value || typeof value !== "object" || !Array.isArray(value.module_overrides)) {
    return null;
  }
  const validBits = (candidate: unknown): candidate is number =>
    typeof candidate === "number"
    && Number.isInteger(candidate)
    && candidate >= 1
    && candidate <= 16;
  const positiveInteger = (candidate: unknown): candidate is number =>
    typeof candidate === "number" && Number.isInteger(candidate) && candidate > 0;
  if (
    !validBits(value.default_bits)
    || !positiveInteger(value.default_group_size)
    || value.module_overrides.some(
      (override) =>
        !override
        || typeof override.module_path !== "string"
        || !override.module_path
        || !validBits(override.bits)
        || !positiveInteger(override.group_size),
    )
  ) {
    return null;
  }
  return {
    default_bits: value.default_bits,
    default_group_size: value.default_group_size,
    module_overrides: value.module_overrides.map((override) => ({ ...override })),
  };
}

function isReviewedQwen3MoELayout(value: unknown, layerCount: unknown): boolean {
  if (
    !isRecord(value)
    || typeof layerCount !== "number"
    || !Number.isInteger(layerCount)
    || layerCount <= 0
  ) {
    return false;
  }
  const overrides = value.module_overrides;
  if (
    value.default_bits !== 4
    || value.default_group_size !== 64
    || !Array.isArray(overrides)
    || overrides.length !== layerCount
  ) {
    return false;
  }
  const expectedPaths = Array.from(
    { length: layerCount },
    (_, index) => `model.layers.${index}.mlp.gate`,
  ).sort();
  return overrides.every((override, index) =>
    isRecord(override)
    && override.module_path === expectedPaths[index]
    && override.bits === 8
    && override.group_size === 64,
  );
}

export function applyProviderModelInspection(
  current: ModelFacts,
  inspection: ModelInspectionResponse,
): ModelFacts {
  if (inspection.status !== "ok" || !inspection.facts || !inspection.resolved_revision) {
    throw new Error(inspection.error ?? "The provider did not return revision-bound model facts.");
  }
  const facts = inspection.facts;
  const moe = completeMoETopology(facts.moe);
  const quantizationLayout = completeQuantizationLayout(facts.quantization_layout);
  return {
    ...current,
    revision: inspection.resolved_revision,
    family: typeof facts.family === "string" ? facts.family : current.family,
    hidden_size: typeof facts.hidden_size === "number" ? facts.hidden_size : current.hidden_size,
    intermediate_size: typeof facts.intermediate_size === "number" ? facts.intermediate_size : current.intermediate_size,
    layers: typeof facts.layers === "number" ? facts.layers : current.layers,
    context_length: typeof facts.context_length === "number" ? facts.context_length : current.context_length,
    license_name: typeof facts.license_name === "string" ? facts.license_name : current.license_name,
    model_type: typeof facts.model_type === "string" ? facts.model_type : null,
    architecture: typeof facts.architecture === "string" ? facts.architecture : null,
    quantization_bits: typeof facts.quantization_bits === "number" ? facts.quantization_bits : null,
    quantization_layout: quantizationLayout,
    moe,
    active_parameters_b: null,
    sparse_layer_count: null,
  };
}

export function applyPlanDerivedModelFacts(
  current: ModelFacts,
  plan: TrainingPlan,
): ModelFacts {
  const model = plan.model;
  if (
    plan.schema_version !== "aptus.training-plan.v3"
    || !isRecord(model)
    || !matchesCurrentModel(model, current)
  ) {
    return current;
  }

  const activeParameters = model.active_parameters;
  const sparseLayerCount = model.sparse_layer_count;
  const totalParameters = model.parameters;
  const layerCount = model.layers;
  return {
    ...current,
    active_parameters_b:
      typeof activeParameters === "number"
      && Number.isFinite(activeParameters)
      && activeParameters > 0
      && typeof totalParameters === "number"
      && Number.isFinite(totalParameters)
      && activeParameters <= totalParameters
        ? activeParameters / 1_000_000_000
        : null,
    sparse_layer_count:
      typeof sparseLayerCount === "number"
      && Number.isInteger(sparseLayerCount)
      && sparseLayerCount >= 0
      && typeof layerCount === "number"
      && Number.isInteger(layerCount)
      && sparseLayerCount <= layerCount
        ? sparseLayerCount
        : null,
  };
}

export function moeCompatibilityFromPlan(
  plan: TrainingPlan | null,
  current: ModelFacts,
): ModelCompatibility | null {
  if (plan?.schema_version !== "aptus.training-plan.v3" || !current.moe) return null;
  const model = plan.model;
  if (
    !isRecord(model)
    || !matchesCurrentModel(model, current)
    || model.family !== "qwen3_moe"
    || model.model_type !== "qwen3_moe"
    || model.architecture !== "Qwen3MoeForCausalLM"
    || model.quantization_bits !== 4
    || !isRecord(model.moe)
  ) {
    return null;
  }

  const candidate = plan.recommended;
  const runtime = candidate?.runtime_contract;
  const exactTargets = Array.isArray(candidate?.target_modules)
    && candidate.target_modules.length === QWEN3_MOE_TARGETS.length
    && candidate.target_modules.every((target, index) => target === QWEN3_MOE_TARGETS[index]);
  const exactConditionalPath = Boolean(
    candidate
    && candidate.status === "conditional"
    && candidate.method === "qlora"
    && candidate.distribution === "single"
    && isReviewedQwen3MoELayout(model.quantization_layout, model.layers)
    && runtime?.compute_backend === "mps"
    && runtime.training_runtime === "mlx-lm"
    && runtime.compiler_id === "mlx-lm.qlora.v1"
    && runtime.estimator_id === "aptus-memory-mlx-v2"
    && runtime.evidence_requirement === "pilot-required"
    && runtime.export_kind === "mlx-lm-adapter"
    && exactTargets,
  );

  return exactConditionalPath
    ? {
        status: "conditional",
        family: "qwen3_moe",
        supported_runtime: "mlx-lm",
        compute_backend: "mps",
        supported_methods: ["qlora"],
        distribution: "single",
        evidence_requirement: "pilot-required",
        adapter_profile_id: "attention-qkvo.v1",
        reason: "The current v3 plan preserves the reviewed model identity, quantization layout, topology, MLX-LM runtime contract, and attention-only q/k/v/o target set. Measured preflight and a real-model pilot remain mandatory.",
      }
    : {
        status: "unsupported",
        family: "qwen3_moe",
        supported_runtime: null,
        compute_backend: null,
        supported_methods: [],
        distribution: null,
        evidence_requirement: "implementation-required",
        adapter_profile_id: null,
        reason: "The current plan does not bind this topology to the exact conditional Qwen3 MoE path.",
      };
}
