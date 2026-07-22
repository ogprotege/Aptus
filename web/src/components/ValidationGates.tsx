import type { ValidationGate, ValidationReport } from "../types";
import { StatusBadge } from "./StatusBadge";

function derivedGates(report: ValidationReport): ValidationGate[] {
  if (report.gates?.length) return report.gates;
  const state = report.state ?? "pending";
  const states = ["contract-pass", "static-pass", "dependency-pass", "model-data-pass", "measured-preflight-pass", "pilot-pass", "execution-approved", "measured-run-pass"];
  const rank = states.indexOf(state);
  const errorCodes = (report.findings ?? [])
    .filter((finding) => finding.severity === "error")
    .map((finding) => finding.code ?? "");
  const hasCode = (...prefixes: string[]) =>
    errorCodes.some((code) => prefixes.some((prefix) => code.startsWith(prefix)));
  const structureFailed = hasCode(
    "MISSING_FILE",
    "PLAN_",
    "MANIFEST_",
    "DEPENDENCY_SET_MISMATCH",
    "TRAINER_CONFIG_MISMATCH",
    "USER_VALUE_EMBEDDED",
  );
  const staticFailed = hasCode("PYTHON_PARSE_ERROR", "UNRESOLVED_TEMPLATE");
  const runtimeFailed = hasCode("RUNTIME_VALIDATION_FAILED");
  const structureState = structureFailed || (state === "invalid" && !staticFailed && !runtimeFailed)
    ? "failed"
    : rank >= 0 || staticFailed || runtimeFailed
      ? "passed"
      : "pending";
  const staticState = structureFailed
    ? "blocked"
    : staticFailed
      ? "failed"
      : rank >= 1 || runtimeFailed
        ? "passed"
        : "pending";
  return [
    {
      label: "Bundle structure",
      state: structureState,
      detail: report.checked_files?.length
        ? `${report.checked_files.length} required files checked.`
        : "Waiting for a file manifest.",
    },
    {
      label: "Static contracts",
      state: staticState,
      detail: "Plan semantics, Python syntax, templates, and source boundaries. Installed dependencies are a later gate.",
    },
    {
      label: "Dependency environment",
      state: structureFailed || staticFailed ? "blocked" : rank >= 2 ? "passed" : runtimeFailed ? "failed" : "pending",
      detail: "Direct pinned package versions are installed. Binary and runtime compatibility remain untested.",
    },
    {
      label: "Model and data bindings",
      state: rank >= 3 ? "passed" : "pending",
      detail: "Pinned model revision, dataset hash, tokenizer, and transform contract.",
    },
    {
      label: "Measured preflight",
      state: rank >= 4 ? "passed" : "pending",
      detail: "Bounded synthetic method and selected-runtime check. This is not planned-model fit.",
    },
    {
      label: "Exact model and data pilot",
      state: rank >= 5 ? "passed" : "pending",
      detail: "Pinned model and data steps with finite loss, bound artifacts, and the selected runtime's reload proof.",
    },
  ];
}

interface ValidationGatesProps {
  report: ValidationReport;
}

export function ValidationGates({ report }: ValidationGatesProps) {
  const gates = derivedGates(report);
  return (
    <section className="validation-gates" aria-labelledby="validation-gates-title">
      <div className="section-heading-row compact-heading">
        <div>
          <p className="eyebrow">Proof before execution</p>
          <h2 id="validation-gates-title">Validation gates</h2>
        </div>
        <StatusBadge state={report.state ?? "pending"} />
      </div>

      <ol className="gate-list">
        {gates.map((gate, index) => (
          <li key={gate.id ?? `${gate.label}-${index}`}>
            <span className="gate-index" aria-hidden="true">{index + 1}</span>
            <span className="gate-copy">
              <strong>{gate.label}</strong>
              {gate.detail ? <small>{gate.detail}</small> : null}
            </span>
            <StatusBadge state={gate.state} />
          </li>
        ))}
      </ol>

      {report.findings?.length ? (
        <div className="finding-list">
          <h3>Findings</h3>
          <ul>
            {report.findings.map((finding, index) => (
              <li key={`${finding.code ?? "finding"}-${index}`} data-severity={finding.severity}>
                <StatusBadge state={finding.severity} />
                <span>
                  <strong>{finding.code ?? "Validation finding"}</strong>
                  <span>{finding.message}</span>
                  {finding.path ? <code>{finding.path}</code> : null}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="no-findings">No validation findings were returned.</p>
      )}
    </section>
  );
}
