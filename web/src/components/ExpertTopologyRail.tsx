import { useId } from "react";
import type { MoETopology } from "../types";

interface ExpertTopologyRailProps {
  topology: MoETopology;
  totalParametersB: number | null;
  activeParametersB?: number | null;
  sparseLayerCount?: number | null;
  quantizationBits?: number | null;
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
}: ExpertTopologyRailProps) {
  const titleId = useId();
  const captionId = useId();
  const routedText = `Any ${topology.experts_per_token} of ${topology.expert_count} routed experts`;

  return (
    <section className="moe-topology-panel" aria-labelledby={titleId}>
      <header className="moe-topology-heading">
        <div>
          <p className="eyebrow">Sparse model contract</p>
          <h3 id={titleId}>Pinned MoE topology</h3>
        </div>
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

      <p className="moe-memory-boundary">
        All checkpoint weights must remain resident. Active parameters describe per-token computation and never reduce the base-weight memory budget.
      </p>
    </section>
  );
}
