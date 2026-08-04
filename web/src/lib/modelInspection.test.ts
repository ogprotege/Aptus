import { describe, expect, it } from "vitest";
import { EXAMPLE_DRAFT, EXAMPLE_PLAN } from "../demo";
import type { TrainingPlan } from "../types";
import {
  applyPlanDerivedModelFacts,
  applyProviderModelInspection,
} from "./modelInspection";

const QWEN3_REVISION = "d".repeat(40);
const REVIEWED_QWEN3_LAYOUT = {
  default_bits: 4,
  default_group_size: 64,
  module_overrides: Array.from({ length: 48 }, (_, index) => ({
    module_path: `model.layers.${index}.mlp.gate`,
    bits: 8,
    group_size: 64,
  })).sort((left, right) => left.module_path.localeCompare(right.module_path)),
};

function exactQwen3MoEPlan(): TrainingPlan {
  const plan = structuredClone(EXAMPLE_PLAN);
  plan.model = {
      model_id: "Qwen/Qwen3-30B-A3B",
      revision: QWEN3_REVISION,
      family: "qwen3_moe",
      model_type: "qwen3_moe",
      architecture: "Qwen3MoeForCausalLM",
      quantization_bits: 4,
      quantization_layout: REVIEWED_QWEN3_LAYOUT,
      parameters: 30_500_000_000,
      layers: 48,
      active_parameters: 3_300_000_000,
      sparse_layer_count: 48,
      moe: {
        expert_count: 128,
        experts_per_token: 8,
        expert_intermediate_size: 768,
        decoder_sparse_step: 1,
        mlp_only_layers: [],
        shared_expert_intermediate_size: null,
      },
  };
  return plan;
}

describe("provider model inspection", () => {
  it("applies only provider-declared architecture facts and the resolved revision", () => {
    const current = structuredClone(EXAMPLE_DRAFT.model);
    current.parameters_b = 8.03;
    current.training_allowed = true;

    const merged = applyProviderModelInspection(current, {
      status: "ok",
      model_id: current.model_id,
      requested_revision: "main",
      resolved_revision: "b".repeat(40),
      facts: {
        architecture: "MistralForCausalLM",
        architectures: ["MistralForCausalLM"],
        model_type: "mistral",
        family: "mistral",
        hidden_size: 5120,
        intermediate_size: 14336,
        layers: 40,
        context_length: 8192,
        license_name: "apache-2.0",
        parameters: null,
        training_allowed: null,
      },
    });

    expect(merged).toMatchObject({
      revision: "b".repeat(40),
      family: "mistral",
      hidden_size: 5120,
      intermediate_size: 14336,
      layers: 40,
      context_length: 8192,
      license_name: "apache-2.0",
    });
    expect(merged.parameters_b).toBe(8.03);
    expect(merged.training_allowed).toBe(true);
    expect(merged.model_id).toBe(current.model_id);
  });

  it("applies the canonical family while retaining exact provider identity", () => {
    const current = structuredClone(EXAMPLE_DRAFT.model);

    const merged = applyProviderModelInspection(current, {
      status: "ok",
      model_id: current.model_id,
      requested_revision: "main",
      resolved_revision: "c".repeat(40),
      facts: {
        architecture: "Qwen2ForCausalLM",
        architectures: ["Qwen2ForCausalLM"],
        model_type: "qwen2",
        family: "qwen",
      },
    });

    expect(merged.family).toBe("qwen");
    expect(merged.model_type).toBe("qwen2");
    expect(merged.architecture).toBe("Qwen2ForCausalLM");
    expect(merged).not.toHaveProperty("architectures");
  });

  it("applies an exact MoE topology without replacing user attestations", () => {
    const current = structuredClone(EXAMPLE_DRAFT.model);
    current.parameters_b = 30.5;
    current.training_allowed = true;

    const merged = applyProviderModelInspection(current, {
      status: "ok",
      model_id: "Qwen/Qwen3-30B-A3B",
      requested_revision: "main",
      resolved_revision: QWEN3_REVISION,
      facts: {
        architecture: "Qwen3MoeForCausalLM",
        architectures: ["Qwen3MoeForCausalLM"],
        model_type: "qwen3_moe",
        family: "qwen3_moe",
        hidden_size: 2048,
        layers: 48,
        context_length: 32768,
        quantization_bits: 4,
        quantization_layout: REVIEWED_QWEN3_LAYOUT,
        moe: {
          expert_count: 128,
          experts_per_token: 8,
          expert_intermediate_size: 768,
          decoder_sparse_step: 1,
          mlp_only_layers: [],
          shared_expert_intermediate_size: null,
        },
      },
    });

    expect(merged).toMatchObject({
      family: "qwen3_moe",
      model_type: "qwen3_moe",
      architecture: "Qwen3MoeForCausalLM",
      quantization_bits: 4,
      quantization_layout: REVIEWED_QWEN3_LAYOUT,
      active_parameters_b: null,
      sparse_layer_count: null,
      moe: {
        expert_count: 128,
        experts_per_token: 8,
        mlp_only_layers: [],
      },
    });
    expect(merged.parameters_b).toBe(30.5);
    expect(merged.training_allowed).toBe(true);
    expect(merged.quantization_layout).not.toBe(REVIEWED_QWEN3_LAYOUT);
  });

  it("applies derived MoE facts only from the matching v5 plan", () => {
    const current = structuredClone(EXAMPLE_DRAFT.model);
    current.model_id = "Qwen/Qwen3-30B-A3B";
    current.revision = QWEN3_REVISION;
    current.family = "qwen3_moe";
    current.model_type = "qwen3_moe";
    current.architecture = "Qwen3MoeForCausalLM";
    current.quantization_bits = 4;
    current.quantization_layout = structuredClone(REVIEWED_QWEN3_LAYOUT);
    current.moe = {
      expert_count: 128,
      experts_per_token: 8,
      expert_intermediate_size: 768,
      decoder_sparse_step: 1,
      mlp_only_layers: [],
      shared_expert_intermediate_size: null,
    };
    const plan = exactQwen3MoEPlan();

    const merged = applyPlanDerivedModelFacts(current, plan);

    expect(merged.active_parameters_b).toBe(3.3);
    expect(merged.sparse_layer_count).toBe(48);
  });

  it("does not apply plan-derived MoE facts across a model revision boundary", () => {
    const current = structuredClone(EXAMPLE_DRAFT.model);
    current.model_id = "Qwen/Qwen3-30B-A3B";
    current.revision = "e".repeat(40);
    current.active_parameters_b = null;
    current.sparse_layer_count = null;
    current.moe = {
      expert_count: 128,
      experts_per_token: 8,
      expert_intermediate_size: 768,
      decoder_sparse_step: 1,
      mlp_only_layers: [],
    };

    expect(applyPlanDerivedModelFacts(current, exactQwen3MoEPlan())).toBe(current);
  });

  it("does not apply an incomplete provider topology", () => {
    const current = structuredClone(EXAMPLE_DRAFT.model);
    current.moe = {
      expert_count: 64,
      experts_per_token: 4,
      expert_intermediate_size: 512,
      decoder_sparse_step: 1,
      mlp_only_layers: [],
    };

    const merged = applyProviderModelInspection(current, {
      status: "ok",
      model_id: "provider/incomplete-moe",
      requested_revision: "main",
      resolved_revision: "e".repeat(40),
      facts: {
        architecture: "UnknownMoeForCausalLM",
        model_type: "unknown_moe",
        family: "unknown_moe",
        moe: {
          expert_count: 64,
          experts_per_token: null,
          expert_intermediate_size: 512,
          decoder_sparse_step: 1,
          mlp_only_layers: [],
        },
      },
      compatibility: {
        status: "unsupported",
        family: "unknown_moe",
        supported_runtime: null,
        compute_backend: null,
        supported_methods: [],
        distribution: null,
        evidence_requirement: "implementation-required",
        adapter_profile_id: null,
        reason: "The provider topology is incomplete.",
      },
    });

    expect(merged.moe).toBeNull();
  });
});
