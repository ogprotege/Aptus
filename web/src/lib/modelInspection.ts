import type {
  InspectedMoETopology,
  ModelFacts,
  ModelInspectionResponse,
  MoETopology,
  PlanView,
  QuantizationLayout,
} from "../types";

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
  plan: PlanView,
): ModelFacts {
  if (!("schema_version" in plan)) return current;
  const model = plan.model;
  if (
    plan.schema_version !== "aptus.training-plan.v6"
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
