interface StatusBadgeProps {
  state: string;
  label?: string;
}

export function StatusBadge({ state, label }: StatusBadgeProps) {
  const normalized = state.toLowerCase();
  const tone = ["passed", "complete", "completed", "feasible", "contract-pass", "static-pass", "dependency-pass", "model-data-pass", "measured-preflight-pass", "pilot-pass", "execution-approved", "measured-run-pass", "verified-at-completion"].includes(
    normalized,
  )
    ? "positive"
    : ["failed", "invalid", "infeasible", "unsupported", "error", "cancelled", "missing-since-completion"].includes(normalized)
      ? "negative"
      : ["running", "cancelling", "profiling", "compiling", "validating", "verifying"].includes(normalized)
        ? "active"
        : ["warning", "blocked", "unknown", "conditional", "verified-at-completion-not-rehashed"].includes(normalized)
          ? "warning"
          : ["omitted", "absent", "no-last-call"].includes(normalized)
            ? "omitted"
            : "neutral";

  return (
    <span className={`status-badge status-${tone}`}>
      <span className="status-dot" aria-hidden="true" />
      {label ?? state.replace(/-/g, " ")}
    </span>
  );
}
