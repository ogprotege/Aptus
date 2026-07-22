import type { ProvenanceKind } from "../types";

const LABELS: Record<ProvenanceKind, string> = {
  measured: "Measured",
  "provider-declared": "Provider declared",
  "user-attested": "User attested",
  declared: "Declared",
  inferred: "Inferred",
  "user-supplied": "User supplied",
  unknown: "Unknown",
  example: "Example data",
};

interface ProvenanceBadgeProps {
  kind: ProvenanceKind;
  label?: string;
}

export function ProvenanceBadge({ kind, label }: ProvenanceBadgeProps) {
  return (
    <span className={`provenance-badge provenance-${kind}`}>
      <span className="provenance-dot" aria-hidden="true" />
      {label ?? LABELS[kind]}
    </span>
  );
}
