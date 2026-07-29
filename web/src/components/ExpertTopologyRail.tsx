import { useId } from "react";
import { StatusBadge } from "./StatusBadge";
import type { ModelCompatibility, MoETopology } from "../types";
import { normalizeModelCompatibility } from "../lib/modelCompatibility";

interface ExpertTopologyRailProps {
  topology: MoETopology;
  totalParametersB: number | null;
  activeParametersB?: number | null;
  sparseLayerCount?: number | null;
  quantizationBits?: number | null;
  compatibility?: ModelCompatibility | null;
  selectedRuntime: string;
  selectedBackend: string;
}

function parameterLabel(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "Derived during planning";
  }
  return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value)}B`;
}

export function ExpertTopologyRail({
  topology,
  totalParametersB,
  activeParametersB,
  sparseLayerCount,
  quantizationBits,
  compatibility,
  selectedRuntime,
  selectedBackend,
}: ExpertTopologyRailProps) {
  const titleId = useId();
  const captionId = useId();
  const normalizedCompatibility = normalizeModelCompatibility(compatibility);
  const supportedRuntime = normalizedCompatibility?.supported_runtime ?? null;
  const supportedBackend = normalizedCompatibility?.compute_backend ?? null;
  const status = normalizedCompatibility?.status ?? "unknown";
  const pathMatches = status === "conditional"
    && supportedRuntime === selectedRuntime
    && supportedBackend === selectedBackend;
  const title = status === "conditional"
    ? pathMatches
      ? "Exact MoE path recognized"
      : "MoE path recognized; selected target differs"
    : status === "unsupported"
      ? "MoE topology recognized; execution unsupported"
      : "Pinned MoE topology";
  const statusLabel = status === "conditional" && pathMatches
    ? "Pilot required"
    : status === "conditional"
      ? "Target mismatch"
      : status === "unsupported"
        ? "Unsupported"
        : "Support unknown";
  const routedText = `Any ${topology.experts_per_token} of ${topology.expert_count} routed experts`;
  const supportedMethodValues = normalizedCompatibility?.supported_methods ?? [];
  const supportedMethods = supportedMethodValues.join(", ") || "the allowlisted method";
  const methodLabel = supportedMethodValues.length === 1 ? "method" : "methods";
  const supportedDistribution = normalizedCompatibility?.distribution ?? "the allowlisted placement";
  const adapterProfileId = normalizedCompatibility?.adapter_profile_id ?? "the allowlisted adapter profile";
  const evidenceRequirement = normalizedCompatibility?.evidence_requirement ?? "pilot-required";
  const pilotBoundary = normalizedCompatibility?.reason ?? "Every exact bundle must still pass its pilot.";

  return (
    <section className="moe-topology-panel" aria-labelledby={titleId}>
      <header className="moe-topology-heading">
        <div>
          <p className="eyebrow">Sparse model contract</p>
          <h3 id={titleId}>{title}</h3>
        </div>
        <StatusBadge state={status === "conditional" && !pathMatches ? "warning" : status} label={statusLabel} />
      </header>

      <figure className="expert-topology" aria-describedby={captionId}>
        <div className="expert-topology-rail" aria-hidden="true">
          <span className="expert-node token-node">Token</span>
          <span className="expert-arrow">→</span>
          <span className="expert-node router-node">Router</span>
          <span className="expert-arrow">→</span>
          <span className="expert-destinations">
            <span className="expert-node expert-bank">{routedText}</span>
            {topology.shared_expert_intermediate_size ? (
              <span className="expert-node shared-expert">
                Shared expert path · width {topology.shared_expert_intermediate_size.toLocaleString()}
              </span>
            ) : null}
          </span>
        </div>
        <figcaption id={captionId}>
          The router selects any {topology.experts_per_token} of {topology.expert_count} routed experts for each token.
          {topology.shared_expert_intermediate_size
            ? " The shared expert path also runs for each token."
            : " No shared expert path is declared."}
        </figcaption>
      </figure>

      <dl className="moe-topology-facts">
        <div>
          <dt>Total resident parameters</dt>
          <dd>{totalParametersB === null ? "Required" : parameterLabel(totalParametersB)}</dd>
        </div>
        <div>
          <dt>Active per token</dt>
          <dd>{parameterLabel(activeParametersB)}</dd>
        </div>
        <div>
          <dt>Sparse layers</dt>
          <dd>{sparseLayerCount ?? "Derived during planning"}</dd>
        </div>
        <div>
          <dt>Expert width</dt>
          <dd>{topology.expert_intermediate_size.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Checkpoint precision</dt>
          <dd>{quantizationBits ? `${quantizationBits}-bit` : "Not declared"}</dd>
        </div>
        <div>
          <dt>Sparse cadence</dt>
          <dd>Every {topology.decoder_sparse_step} layer{topology.decoder_sparse_step === 1 ? "" : "s"}</dd>
        </div>
      </dl>

      {status === "conditional" ? (
        <p className={`moe-support-copy${pathMatches ? "" : " support-mismatch"}`}>
          {pathMatches
            ? `This artifact is eligible for the reviewed pilot path: runtime ${supportedRuntime}, backend ${supportedBackend}, ${methodLabel} ${supportedMethods}, placement ${supportedDistribution}, adapter profile ${adapterProfileId}. Evidence requirement: ${evidenceRequirement}. ${pilotBoundary}`
            : `The reviewed pilot path requires runtime ${supportedRuntime}, backend ${supportedBackend}, ${methodLabel} ${supportedMethods}, placement ${supportedDistribution}, and adapter profile ${adapterProfileId}. The selected target uses runtime ${selectedRuntime} and backend ${selectedBackend}; it does not match this path. Evidence requirement: ${evidenceRequirement}. ${pilotBoundary}`}
        </p>
      ) : normalizedCompatibility?.reason ? (
        <p className="moe-support-copy support-mismatch">{normalizedCompatibility.reason}</p>
      ) : null}

      <p className="moe-memory-boundary">
        All checkpoint weights must remain resident. Active parameters describe per-token computation and never reduce the base-weight memory budget.
      </p>
    </section>
  );
}
