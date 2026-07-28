import type { CandidatePlan } from "../types";
import {
  candidateStatus,
  candidateMemoryLanguage,
  expectedMemory,
  formatBytes,
  formatMethod,
  memoryComponents,
  memoryLimit,
  upperMemory,
} from "../lib/plan";
import { StatusBadge } from "./StatusBadge";

interface FitLedgerProps {
  candidate: CandidatePlan | null;
  example?: boolean;
  compact?: boolean;
}

export function FitLedger({ candidate, example = false, compact = false }: FitLedgerProps) {
  const components = memoryComponents(candidate);
  const expected = expectedMemory(candidate);
  const upper = upperMemory(candidate);
  const limit = memoryLimit(candidate);
  const deviceTotal = candidate?.memory?.device_total_bytes ?? limit;
  const scale = Math.max(upper ?? 0, deviceTotal ?? 0, 1) * 1.06;
  const fits = upper !== null && limit !== null ? upper <= limit : null;
  const headroom =
    upper !== null && limit !== null ? Math.max(limit - upper, 0) : null;
  const fitLinePosition = limit ? Math.min((limit / scale) * 100, 100) : null;
  const reserveStart = limit ? Math.min((limit / scale) * 100, 100) : 0;
  const reserveHeight =
    deviceTotal && limit
      ? Math.max(((deviceTotal - limit) / scale) * 100, 0)
      : 0;
  const memoryLanguage = candidateMemoryLanguage(candidate);

  const description = candidate
    ? `${formatMethod(candidate.method)} has a ${formatBytes(upper)} heuristic upper envelope against ${formatBytes(limit)} ${memoryLanguage.budgetDescription}. Host staging memory is tracked separately.`
    : "No candidate is selected. Compare strategies to calculate memory fit.";

  return (
    <section className={`fit-ledger${compact ? " fit-ledger-compact" : ""}`} aria-labelledby={compact ? "compact-fit-title" : "fit-title"}>
      <header className="ledger-header">
        <div>
          <p className="eyebrow">{memoryLanguage.eyebrow}</p>
          <h2 id={compact ? "compact-fit-title" : "fit-title"}>The Fit Ledger</h2>
        </div>
        {candidate ? <StatusBadge state={candidateStatus(candidate)} /> : null}
      </header>

      {example ? (
        <p className="example-inline">Example values. No hardware inspection ran.</p>
      ) : null}

      {!candidate ? (
        <div className="ledger-empty">
          <span className="empty-glyph" aria-hidden="true">↕</span>
          <p>{description}</p>
        </div>
      ) : (
        <>
          <div className="ledger-summary" role="status">
            <strong>{fits === true ? `${formatBytes(headroom)} predicted headroom` : fits === false ? "Heuristic upper envelope crosses the fit line" : "Fit limit not supplied"}</strong>
            <span>
              {formatBytes(expected)} point estimate · {formatBytes(upper)} heuristic upper · {formatBytes(limit)} {memoryLanguage.budgetDescription}
            </span>
          </div>

          <div className="ledger-chart" role="img" aria-label={description}>
            {reserveHeight > 0 ? (
              <div
                className="reserve-zone"
                style={{ bottom: `${reserveStart}%`, height: `${reserveHeight}%` }}
                aria-hidden="true"
              />
            ) : null}
            <div className="ledger-stack" aria-hidden="true">
              {components.map((component, index) => {
                const visualBytes = component.upper_bytes ?? component.expected_bytes;
                return (
                  <div
                    key={`${component.key ?? component.label}-${index}`}
                    className={`ledger-segment segment-${index % 8}`}
                    style={{ height: `${(visualBytes / scale) * 100}%`, minHeight: visualBytes > 0 ? 4 : 0 }}
                    title={`${component.label}: ${formatBytes(component.expected_bytes)} point estimate${component.upper_bytes !== undefined ? `, ${formatBytes(component.upper_bytes)} heuristic upper` : ""}`}
                  />
                );
              })}
            </div>
            {fitLinePosition !== null ? (
              <div className="fit-line" style={{ bottom: `${fitLinePosition}%` }} aria-hidden="true">
                <span>{memoryLanguage.fitLineLabel} {formatBytes(limit)}</span>
              </div>
            ) : null}
          </div>

          <ul className="ledger-key" aria-label="Memory components">
            {components.map((component, index) => (
              <li key={`${component.key ?? component.label}-key-${index}`}>
                <span className={`key-swatch segment-${index % 8}`} aria-hidden="true" />
                <span>{component.label}</span>
                <strong>{formatBytes(component.expected_bytes)}{component.upper_bytes !== undefined && component.upper_bytes !== component.expected_bytes ? ` / ${formatBytes(component.upper_bytes)}` : ""}</strong>
              </li>
            ))}
            {deviceTotal && limit && deviceTotal > limit ? (
              <li>
                <span className="key-swatch reserve-swatch" aria-hidden="true" />
                <span>{memoryLanguage.reserveLabel}</span>
                <strong>{formatBytes(deviceTotal - limit)}</strong>
              </li>
            ) : null}
          </ul>
        </>
      )}
    </section>
  );
}
