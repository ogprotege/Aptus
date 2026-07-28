import type { CandidatePlan, EvidenceRecord, TrainingPlan } from "../types";
import { CandidateComparison } from "../components/CandidateComparison";
import { EmptyStage } from "../components/EmptyStage";
import { ProvenanceBadge } from "../components/ProvenanceBadge";
import { StageHeader } from "../components/StageHeader";
import { StatusBadge } from "../components/StatusBadge";
import {
  candidateMemoryLanguage,
  candidateStatus,
  formatBytes,
  formatMethod,
  memoryLimit,
  planRationale,
  upperMemory,
} from "../lib/plan";

interface CompareStageProps {
  plan: TrainingPlan | null;
  selected: CandidatePlan | null;
  busy: string | null;
  demoMode: boolean;
  onInspectCandidate: (candidate: CandidatePlan) => void;
  onCompile: () => Promise<void>;
  onReturnToFacts: () => void;
}

export function CompareStage({
  plan,
  selected,
  busy,
  demoMode,
  onInspectCandidate,
  onCompile,
  onReturnToFacts,
}: CompareStageProps) {
  if (!plan) {
    return (
      <>
        <StageHeader
          eyebrow="Stage 2 · Feasibility"
          title="Compare predicted fit."
          lede="Aptus needs a dataset profile and explicit model, hardware, and target facts before it can reject unsupported strategies and rank viable ones."
        />
        <EmptyStage title="No plan to compare" actionLabel="Return to facts" onAction={onReturnToFacts}>
          Profile the dataset and supply the model, hardware, and target facts. Then ask Aptus to compare supported strategies.
        </EmptyStage>
      </>
    );
  }

  const recommended = plan.recommended;
  const upper = upperMemory(recommended);
  const limit = memoryLimit(recommended);
  const headroom = upper !== null && limit !== null ? limit - upper : null;
  const rationale = planRationale(plan);
  const inspected = selected ?? recommended;
  const assumptions = inspected?.assumptions ?? plan.assumptions ?? [];
  const evidence = inspected?.evidence ?? plan.evidence ?? [];
  const rejectionReasons = inspected?.rejection_reasons ?? [];
  const evidenceById = new Map<string, EvidenceRecord>(
    (plan.evidence_records ?? []).map((record) => [record.evidence_id, record]),
  );
  const recommendedMemoryLanguage = candidateMemoryLanguage(recommended);

  return (
    <>
      <StageHeader
        eyebrow="Stage 2 · Feasibility"
        title="Compare predicted fit."
        lede="Hard constraints decide what can run. Your objective ranks only the strategies that survive."
        meta={demoMode ? <ProvenanceBadge kind="example" label="Example comparison" /> : <StatusBadge state={recommended ? candidateStatus(recommended) : "infeasible"} />}
      />

      {recommended ? (
        <section className="recommendation-panel" aria-labelledby="recommendation-title">
          <div className="recommendation-main">
            <p className="eyebrow">Recommended under these facts</p>
            <h2 id="recommendation-title">
              {formatMethod(recommended.method)}
              <span>{recommended.precision ?? "precision unknown"}</span>
              <span>{recommended.quantization ?? "no quantization"}</span>
            </h2>
            <p>{rationale[0] ?? "Aptus returned this candidate as the recommended viable plan."}</p>
          </div>
          <dl className="recommendation-metrics">
            <div>
              <dt>Heuristic upper envelope</dt>
              <dd>{formatBytes(upper)}</dd>
            </div>
            <div>
              <dt>{recommendedMemoryLanguage.recommendationLabel}</dt>
              <dd>{formatBytes(limit)}</dd>
            </div>
            <div>
              <dt>Headroom</dt>
              <dd className={headroom !== null && headroom < 0 ? "negative-value" : undefined}>
                {headroom === null ? "Not supplied" : formatBytes(Math.abs(headroom))}{headroom !== null && headroom < 0 ? " over" : ""}
              </dd>
            </div>
            <div>
              <dt>Confidence</dt>
              <dd>{recommended.confidence ?? "Not supplied"}</dd>
            </div>
          </dl>
        </section>
      ) : (
        <section className="blocked-panel" role="alert">
          <StatusBadge state="infeasible" label="No viable strategy" />
          <h2>Aptus did not find a safe plan.</h2>
          <p>Review each rejected candidate below. Change the target or hardware facts, then compare again.</p>
        </section>
      )}

      <CandidateComparison
        candidates={plan.candidates}
        recommended={recommended}
        inspected={selected}
        onInspect={onInspectCandidate}
      />

      {inspected ? (
        <section className="candidate-gate-panel" aria-labelledby="candidate-gate-title">
          <p className="eyebrow">Inspected candidate gate result</p>
          <h2 id="candidate-gate-title">{formatMethod(inspected.method)} · {inspected.distribution ?? "distribution not supplied"}</h2>
          <dl className="candidate-contract-grid">
            <div><dt>World size</dt><dd>{inspected.world_size ?? "Not supplied"}</dd></div>
            <div><dt>Rank / alpha</dt><dd>{inspected.rank ?? "N/A"} / {inspected.alpha ?? "N/A"}</dd></div>
            <div><dt>Learning rate</dt><dd>{inspected.learning_rate ?? "Not supplied"}</dd></div>
            <div><dt>Pareto frontier</dt><dd>{inspected.pareto_frontier === true ? "Yes" : "No"}</dd></div>
            <div><dt>Required host RAM</dt><dd>{formatBytes(inspected.required_host_ram_bytes)}</dd></div>
            <div><dt>Required disk</dt><dd>{formatBytes(inspected.required_disk_bytes)}</dd></div>
            <div><dt>Checkpoint retention</dt><dd>{formatBytes(inspected.checkpoint_retention_bytes)}</dd></div>
            <div><dt>Final export estimate</dt><dd>{formatBytes(inspected.final_export_bytes)}</dd></div>
            <div className="wide-contract-value"><dt>Target modules</dt><dd>{inspected.target_modules?.join(", ") || "Full model or not supplied"}</dd></div>
          </dl>
          {inspected.ranking_basis?.length ? (
            <div className="ranking-basis">
              <strong>Ranking basis</strong>
              <ul className="plain-list">
                {inspected.ranking_basis.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
              </ul>
            </div>
          ) : null}
          {rejectionReasons.length ? (
            <ul className="plain-list warning-list">
              {rejectionReasons.map((reason, index) => <li key={`${reason}-${index}`}>{reason}</li>)}
            </ul>
          ) : (
            <p>No hard-gate rejection was returned for this candidate.</p>
          )}
        </section>
      ) : null}

      <div className="evidence-grid">
        <section aria-labelledby="rationale-title">
          <p className="eyebrow">Decision trace</p>
          <h2 id="rationale-title">Why Aptus chose this</h2>
          {rationale.length ? (
            <ol className="trace-list">
              {rationale.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
            </ol>
          ) : <p>No recommendation rationale was returned.</p>}
        </section>
        <section aria-labelledby="assumptions-title">
          <p className="eyebrow">Limits on the claim</p>
          <h2 id="assumptions-title">Inspected candidate assumptions</h2>
          {assumptions.length ? <ul className="plain-list amber-list">{assumptions.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul> : <p>No assumptions were returned.</p>}
        </section>
        <section aria-labelledby="evidence-title">
          <p className="eyebrow">Source basis</p>
          <h2 id="evidence-title">Inspected candidate evidence</h2>
          {evidence.length ? (
            <ul className="plain-list evidence-list">
              {evidence.map((item, index) => {
                const record = evidenceById.get(item);
                if (!record) return <li key={`${item}-${index}`}><code>{item}</code></li>;
                const sourceIsLink = /^https?:\/\//.test(record.source);
                return (
                  <li className="evidence-record" key={record.evidence_id}>
                    <code>{record.evidence_id}</code>
                    <span>{record.claim}</span>
                    {sourceIsLink ? (
                      <a href={record.source} target="_blank" rel="noreferrer">Open source</a>
                    ) : (
                      <small>{record.source}</small>
                    )}
                    <small>{record.scope}</small>
                  </li>
                );
              })}
            </ul>
          ) : <p>No evidence labels were returned.</p>}
        </section>
        <section aria-labelledby="warnings-title">
          <p className="eyebrow">Needs attention</p>
          <h2 id="warnings-title">Warnings</h2>
          {plan.warnings.length ? <ul className="plain-list warning-list">{plan.warnings.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul> : <p>No warnings were returned.</p>}
        </section>
      </div>

      <div className="sticky-actions">
        <div>
          <strong>{recommended ? "Recommended plan ready to compile" : "Compilation blocked"}</strong>
          <span>{recommended ? "Inspecting an alternative only changes the evidence shown. Compilation uses the recommended candidate above." : "A recommended viable candidate is required."}</span>
        </div>
        <div className="action-buttons">
          <button type="button" className="button button-quiet" onClick={onReturnToFacts}>Edit facts</button>
          <button type="button" className="button button-primary" disabled={!recommended || busy !== null || demoMode} onClick={() => void onCompile()}>
            {busy === "compile" ? "Compiling…" : "Compile recommended bundle"}
          </button>
        </div>
      </div>
    </>
  );
}
