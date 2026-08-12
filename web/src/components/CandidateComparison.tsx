import type { CandidatePlan } from "../types";
import {
  candidateBatches,
  candidateStatus,
  expectedMemory,
  formatBytes,
  formatMethod,
  upperMemory,
} from "../lib/plan";
import { fitStatusLabel } from "../lib/refusal";
import { StatusBadge } from "./StatusBadge";

interface CandidateComparisonProps {
  candidates: CandidatePlan[];
  recommended: CandidatePlan | null;
  inspected: CandidatePlan | null;
  onInspect: (candidate: CandidatePlan) => void;
}

function candidateKey(candidate: CandidatePlan, index: number): string {
  return candidate.id ?? `${candidate.method}-${candidate.distribution ?? "default"}-${index}`;
}

function isSameCandidate(left: CandidatePlan | null, right: CandidatePlan): boolean {
  if (!left) return false;
  if (left.id && right.id) return left.id === right.id;
  return left.method === right.method && left.distribution === right.distribution;
}

export function CandidateComparison({
  candidates,
  recommended,
  inspected,
  onInspect,
}: CandidateComparisonProps) {
  return (
    <section className="comparison-section" aria-labelledby="candidate-title">
      <div className="section-heading-row">
        <div>
          <p className="eyebrow">Feasibility before preference</p>
          <h2 id="candidate-title">Candidate comparison</h2>
        </div>
        <p>Inspect a strategy to review its evidence. Compilation still uses the recommendation.</p>
      </div>

      <div className="candidate-table-wrap">
        <table className="candidate-table">
          <caption className="sr-only">
            Fine-tuning strategy feasibility, device memory, host memory, disk, precision, quantization, and batching comparison
          </caption>
          <thead>
            <tr>
              <th scope="col">Strategy</th>
              <th scope="col">Fit</th>
              <th scope="col">Point / heuristic upper</th>
              <th scope="col">Host RAM required</th>
              <th scope="col">Disk required</th>
              <th scope="col">Precision</th>
              <th scope="col">Quantization</th>
              <th scope="col">Batch</th>
              <th scope="col">Distribution</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((candidate, index) => {
              const batches = candidateBatches(candidate);
              const isRecommended = isSameCandidate(recommended, candidate);
              const isInspected = isSameCandidate(inspected, candidate);
              return (
                <tr key={candidateKey(candidate, index)} className={isInspected ? "is-inspected" : undefined}>
                  <th scope="row">
                    <button
                      type="button"
                      className="candidate-select"
                      aria-label={`Inspect ${formatMethod(candidate.method)} candidate evidence`}
                      aria-pressed={isInspected}
                      onClick={() => onInspect(candidate)}
                    >
                      <span>{formatMethod(candidate.method)}</span>
                      {isRecommended ? <small>Recommended</small> : null}
                      <small className="candidate-inspect-label">
                        {isInspected ? "Inspecting" : "Inspect"}
                      </small>
                    </button>
                  </th>
                  <td>
                    <StatusBadge
                      state={candidateStatus(candidate)}
                      label={fitStatusLabel(candidateStatus(candidate))}
                    />
                  </td>
                  <td className="mono-cell">
                    {formatBytes(expectedMemory(candidate))}
                    <small>{formatBytes(upperMemory(candidate))} heuristic upper</small>
                  </td>
                  <td className="mono-cell">{formatBytes(candidate.required_host_ram_bytes)}</td>
                  <td className="mono-cell">{formatBytes(candidate.required_disk_bytes)}</td>
                  <td>{candidate.precision ?? "Unknown"}</td>
                  <td>{candidate.quantization ?? "None"}</td>
                  <td className="mono-cell">
                    {batches.micro_batch_size ?? "?"} × {batches.gradient_accumulation_steps ?? "?"}
                    <small>{batches.effective_batch_size ?? "?"} effective</small>
                  </td>
                  <td>{candidate.distribution ?? "Not supplied"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="candidate-cards" aria-label="Candidate comparison cards">
        {candidates.map((candidate, index) => {
          const batches = candidateBatches(candidate);
          const isRecommended = isSameCandidate(recommended, candidate);
          const isInspected = isSameCandidate(inspected, candidate);
          return (
            <button
              type="button"
              key={`${candidateKey(candidate, index)}-card`}
              className={`candidate-card${isInspected ? " is-inspected" : ""}`}
              aria-label={`Inspect ${formatMethod(candidate.method)} candidate evidence`}
              aria-pressed={isInspected}
              onClick={() => onInspect(candidate)}
            >
              <span className="candidate-card-title">
                <strong>{formatMethod(candidate.method)}</strong>
                <StatusBadge
                  state={candidateStatus(candidate)}
                  label={fitStatusLabel(candidateStatus(candidate))}
                />
              </span>
              {isRecommended ? <span className="recommended-label">Recommended</span> : null}
              <dl>
                <div><dt>Point estimate</dt><dd>{formatBytes(expectedMemory(candidate))}</dd></div>
                <div><dt>Heuristic upper</dt><dd>{formatBytes(upperMemory(candidate))}</dd></div>
                <div><dt>Host RAM required</dt><dd>{formatBytes(candidate.required_host_ram_bytes)}</dd></div>
                <div><dt>Disk required</dt><dd>{formatBytes(candidate.required_disk_bytes)}</dd></div>
                <div><dt>Precision</dt><dd>{candidate.precision ?? "Unknown"}</dd></div>
                <div><dt>Quantization</dt><dd>{candidate.quantization ?? "None"}</dd></div>
                <div><dt>Batch</dt><dd>{batches.micro_batch_size ?? "?"} × {batches.gradient_accumulation_steps ?? "?"}</dd></div>
              </dl>
              <span className="candidate-card-action">
                {isInspected ? "Inspecting evidence" : "Inspect evidence"}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
