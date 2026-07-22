import { describe, expect, it } from "vitest";
import { EXAMPLE_DRAFT } from "../demo";
import { applyProviderModelInspection } from "./modelInspection";

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

  it("applies the canonical family without copying raw provider identifiers into plan fields", () => {
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
    expect(merged).not.toHaveProperty("model_type");
    expect(merged).not.toHaveProperty("architecture");
    expect(merged).not.toHaveProperty("architectures");
  });
});
