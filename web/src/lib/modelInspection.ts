import type { ModelFacts, ModelInspectionResponse } from "../types";

export function applyProviderModelInspection(
  current: ModelFacts,
  inspection: ModelInspectionResponse,
): ModelFacts {
  if (inspection.status !== "ok" || !inspection.facts || !inspection.resolved_revision) {
    throw new Error(inspection.error ?? "The provider did not return revision-bound model facts.");
  }
  const facts = inspection.facts;
  return {
    ...current,
    revision: inspection.resolved_revision,
    family: typeof facts.family === "string" ? facts.family : current.family,
    hidden_size: typeof facts.hidden_size === "number" ? facts.hidden_size : current.hidden_size,
    intermediate_size: typeof facts.intermediate_size === "number" ? facts.intermediate_size : current.intermediate_size,
    layers: typeof facts.layers === "number" ? facts.layers : current.layers,
    context_length: typeof facts.context_length === "number" ? facts.context_length : current.context_length,
    license_name: typeof facts.license_name === "string" ? facts.license_name : current.license_name,
  };
}
