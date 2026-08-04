import { useId, type ReactNode } from "react";
import type {
  ModelPolicyArtifactMatchPresentation,
  ModelPolicyEvidenceReadinessPresentation,
  ModelPolicyPresentation,
  ModelPolicySelectedPathPresentation,
  ModelPolicyValidationLevel,
} from "../lib/modelPolicy";
import { formatMethod } from "../lib/plan";
import { StatusBadge } from "./StatusBadge";

export interface ModelPolicyPanelProps {
  presentation: ModelPolicyPresentation;
}

interface BadgePresentation {
  state: string;
  label: string;
}

function words(value: string): string {
  const text = value.replace(/-/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function sourceLabel(source: ModelPolicyArtifactMatchPresentation["source"]): string {
  return source === "provider-inspection" ? "Provider inspection" : "User attested";
}

function levelLabel(level: ModelPolicyValidationLevel): string {
  return level === "model-data"
    ? "Model data"
    : level === "measured-preflight"
      ? "Measured preflight"
      : "Pilot";
}

function artifactBadge(
  match: ModelPolicyArtifactMatchPresentation,
): BadgePresentation {
  switch (match.state) {
    case "path-matched":
      return { state: "contract-pass", label: "Path matched" };
    case "family-recognized":
      return { state: "conditional", label: "Family only" };
    case "blocked":
      return { state: "blocked", label: "Blocked" };
    case "unknown":
      return { state: "unknown", label: "No policy match" };
  }
}

function selectedPathBadge(
  selectedPath: ModelPolicySelectedPathPresentation,
): BadgePresentation {
  switch (selectedPath.state) {
    case "bound":
      return { state: "contract-pass", label: "Bound" };
    case "unbound":
      return { state: "warning", label: "Unbound" };
    case "not-selected":
      return { state: "unknown", label: "Not selected" };
  }
}

function evidenceBadge(
  readiness: ModelPolicyEvidenceReadinessPresentation,
): BadgePresentation {
  switch (readiness.state) {
    case "authorized":
      return { state: "contract-pass", label: "Admission active" };
    case "validation-complete":
      return { state: "conditional", label: "Evidence complete" };
    case "admission-deferred":
      return { state: "conditional", label: "Admission deferred" };
    case "validation-required":
      return { state: "warning", label: "Evidence required" };
    case "authorization-blocked":
      return { state: "blocked", label: "Admission blocked" };
    case "implementation-blocked":
      return { state: "blocked", label: "Blocked" };
    case "invalid":
      return { state: "blocked", label: "Invalid evidence" };
    case "not-applicable":
      return { state: "unknown", label: "Not applicable" };
  }
}

function CodeList({ values, empty }: { values: readonly string[]; empty: string }) {
  if (values.length === 0) return <span>{empty}</span>;
  return (
    <ul className="model-policy-token-list">
      {values.map((value) => <li key={value}><code>{value}</code></li>)}
    </ul>
  );
}

function PolicyDetail({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function ArtifactMatchRecord({
  match,
  headingId,
}: {
  match: ModelPolicyArtifactMatchPresentation;
  headingId: string;
}) {
  const badge = artifactBadge(match);
  return (
    <article className="model-policy-record" aria-labelledby={headingId}>
      <header>
        <h3 id={headingId}>Model-policy match</h3>
        <StatusBadge state={badge.state} label={badge.label} />
      </header>

      <p className="model-policy-summary">{match.reason}</p>

      <dl className="model-policy-details">
        <PolicyDetail label="Model"><code>{match.modelId}</code></PolicyDetail>
        <PolicyDetail label="Revision"><code>{match.revision}</code></PolicyDetail>
        <PolicyDetail label="Family">
          {match.family ? <code>{match.family}</code> : "Not recognized"}
        </PolicyDetail>
        <PolicyDetail label="Policy">
          {match.policyId ? (
            <span className="model-policy-identity">
              <code>{match.policyId}</code>
              {match.policyVersion ? <span>v{match.policyVersion}</span> : null}
            </span>
          ) : "No registered policy"}
        </PolicyDetail>
        <PolicyDetail label="Decision source">{sourceLabel(match.source)}</PolicyDetail>
        <PolicyDetail label="Decision ID"><code>{match.decisionId}</code></PolicyDetail>
        <PolicyDetail label="Subject facts"><code>{match.subjectFactsSha256}</code></PolicyDetail>
        <PolicyDetail label="Reason codes">
          <CodeList values={match.reasonCodes} empty="None supplied" />
        </PolicyDetail>
        <PolicyDetail label="Decision evidence">
          <CodeList values={match.evidenceIds} empty="None supplied" />
        </PolicyDetail>
      </dl>

      {match.state === "blocked" ? (
        <p className="model-policy-next-step" role="note">
          <strong>Next step</strong>
          <span>Use a supported pinned artifact or correct its model facts, then create a new plan.</span>
        </p>
      ) : null}
    </article>
  );
}

function selectedPathSummary(selectedPath: ModelPolicySelectedPathPresentation): string {
  switch (selectedPath.state) {
    case "bound":
      return "The selected candidate is bound to the server-issued path below.";
    case "unbound":
      return "This candidate has an execution contract, but no registered model-policy path is bound to it.";
    case "not-selected":
      return "No candidate is selected, so there is no execution path to bind.";
  }
}

function SelectedPathRecord({
  selectedPath,
  headingId,
}: {
  selectedPath: ModelPolicySelectedPathPresentation;
  headingId: string;
}) {
  const badge = selectedPathBadge(selectedPath);
  return (
    <article className="model-policy-record" aria-labelledby={headingId}>
      <header>
        <h3 id={headingId}>Selected candidate path</h3>
        <StatusBadge state={badge.state} label={badge.label} />
      </header>

      <p className="model-policy-summary">{selectedPathSummary(selectedPath)}</p>

      <dl className="model-policy-details">
        <PolicyDetail label="Candidate">
          {selectedPath.candidateId ? <code>{selectedPath.candidateId}</code> : "None selected"}
        </PolicyDetail>
        <PolicyDetail label="Decision link"><code>{selectedPath.decisionId}</code></PolicyDetail>
        <PolicyDetail label="Bound path">
          {selectedPath.bindingPathId ? <code>{selectedPath.bindingPathId}</code> : "No registered path"}
        </PolicyDetail>
        <PolicyDetail label="Binding source">
          {selectedPath.source ? sourceLabel(selectedPath.source) : "Not bound"}
        </PolicyDetail>
        <PolicyDetail label="Runtime">
          {selectedPath.runtime ? <code>{selectedPath.runtime}</code> : "Not supplied"}
        </PolicyDetail>
        <PolicyDetail label="Backend">
          {selectedPath.backend ? <code>{selectedPath.backend}</code> : "Not supplied"}
        </PolicyDetail>
        <PolicyDetail label="Method">
          {selectedPath.method ? formatMethod(selectedPath.method) : "Not supplied"}
        </PolicyDetail>
        <PolicyDetail label="Distribution">
          {selectedPath.distribution ? words(selectedPath.distribution) : "Not supplied"}
        </PolicyDetail>
        <PolicyDetail label="Adapter profile">
          {selectedPath.adapterProfileId
            ? <code>{selectedPath.adapterProfileId}</code>
            : "Not applicable"}
        </PolicyDetail>
        <PolicyDetail label="Target modules">
          <CodeList values={selectedPath.targetModules} empty="None supplied" />
        </PolicyDetail>
        <PolicyDetail label="Path evidence">
          <CodeList values={selectedPath.evidenceIds} empty="None bound" />
        </PolicyDetail>
      </dl>

      {selectedPath.state === "not-selected" ? (
        <p className="model-policy-next-step" role="note">
          <strong>Next step</strong>
          <span>Select a candidate to review its execution path.</span>
        </p>
      ) : selectedPath.state === "unbound" ? (
        <p className="model-policy-next-step" role="note">
          <strong>Boundary</strong>
          <span>No policy-path claim applies to this candidate.</span>
        </p>
      ) : null}
    </article>
  );
}

function evidenceSummary(
  selectedPath: ModelPolicySelectedPathPresentation,
  readiness: ModelPolicyEvidenceReadinessPresentation,
): string {
  if (selectedPath.state === "not-selected") {
    return "Evidence readiness starts after a candidate is selected.";
  }
  if (readiness.candidateRejected) {
    return "This rejected candidate cannot advance to validation or launch admission.";
  }
  if (
    readiness.state === "validation-required"
    && readiness.requiredValidationLevels.length > 0
    && !readiness.reportBoundToSelectedCandidate
  ) {
    return "No validation report is bound to the selected candidate.";
  }
  switch (readiness.state) {
    case "authorized":
      return "Required validation evidence passed and launch admission is active.";
    case "validation-complete":
      return "Required validation evidence passed; launch admission has not been checked.";
    case "admission-deferred":
      return readiness.blocker
        ?? "Required validation evidence passed; launch admission is checked at submission.";
    case "validation-required":
      return readiness.nextAction
        ? `${levelLabel(readiness.nextAction)} evidence is required next.`
        : "Validation evidence is still required for this candidate.";
    case "authorization-blocked":
      return readiness.blocker ?? "Launch admission was not granted.";
    case "implementation-blocked":
      return readiness.blocker ?? "This path cannot currently gather authorizing evidence.";
    case "invalid":
      return readiness.blocker ?? "The bound validation evidence is invalid.";
    case "not-applicable":
      return "This candidate has no model-policy-specific validation gates.";
  }
}

function evidenceNextStep(
  selectedPath: ModelPolicySelectedPathPresentation,
  readiness: ModelPolicyEvidenceReadinessPresentation,
): string | null {
  if (selectedPath.state === "not-selected") {
    return "Select a candidate before evaluating evidence.";
  }
  if (readiness.candidateRejected || selectedPath.state === "unbound") {
    return null;
  }
  if (readiness.state === "implementation-blocked") {
    return "Choose an implemented runtime path, then create a new plan.";
  }
  if (readiness.state === "invalid") {
    return "Review the validation findings before attempting launch admission.";
  }
  if (readiness.state === "authorization-blocked") {
    return "Review the host admission reason before submitting training again.";
  }
  if (
    readiness.state === "validation-complete"
    || readiness.state === "admission-deferred"
  ) {
    return "Submit full training when ready; Aptus will perform atomic launch admission.";
  }
  if (
    readiness.state === "validation-required"
    && readiness.requiredValidationLevels.length > 0
    && !readiness.reportBoundToSelectedCandidate
  ) {
    return "Validate this candidate; only a report bound to it can authorize execution.";
  }
  if (readiness.state === "validation-required" && readiness.nextAction) {
    return `Run the ${levelLabel(readiness.nextAction).toLowerCase()} gate next.`;
  }
  return null;
}

function authorizationValue(
  readiness: ModelPolicyEvidenceReadinessPresentation,
): ReactNode {
  if (readiness.authorizationStatus === "current") {
    return <StatusBadge state="contract-pass" label="Current" />;
  }
  if (readiness.authorizationStatus === "deferred") {
    return <StatusBadge state="conditional" label="Deferred" />;
  }
  if (readiness.authorizationStatus === "blocked") {
    return <StatusBadge state="blocked" label="Not active" />;
  }
  return <StatusBadge state="unknown" label="Not checked" />;
}

function EvidenceReadinessRecord({
  selectedPath,
  readiness,
  headingId,
}: {
  selectedPath: ModelPolicySelectedPathPresentation;
  readiness: ModelPolicyEvidenceReadinessPresentation;
  headingId: string;
}) {
  const badge = evidenceBadge(readiness);
  const nextStep = evidenceNextStep(selectedPath, readiness);
  return (
    <article className="model-policy-record" aria-labelledby={headingId}>
      <header>
        <h3 id={headingId}>Evidence readiness</h3>
        <StatusBadge state={badge.state} label={badge.label} />
      </header>

      <p className="model-policy-summary">{evidenceSummary(selectedPath, readiness)}</p>

      <dl className="model-policy-details">
        <PolicyDetail label="Selected-candidate report">
          {readiness.reportBoundToSelectedCandidate
            ? <StatusBadge state={readiness.currentState ?? "unknown"} label={readiness.currentState ? words(readiness.currentState) : "State unavailable"} />
            : "Not bound"}
        </PolicyDetail>
        <PolicyDetail label="Required levels">
          {readiness.requiredValidationLevels.length > 0
            ? readiness.requiredValidationLevels.map(levelLabel).join(" → ")
            : "None"}
        </PolicyDetail>
        <PolicyDetail label="Next gate">
          {readiness.nextAction ? levelLabel(readiness.nextAction) : "None"}
        </PolicyDetail>
        <PolicyDetail label="Host authorization">
          {authorizationValue(readiness)}
        </PolicyDetail>
      </dl>

      {nextStep ? (
        <p className="model-policy-next-step" role="note">
          <strong>Next step</strong>
          <span>{nextStep}</span>
        </p>
      ) : null}
    </article>
  );
}

export function ModelPolicyPanel({ presentation }: ModelPolicyPanelProps) {
  const panelId = useId();
  const titleId = `${panelId}-title`;
  const matchId = `${panelId}-match`;
  const selectedPathId = `${panelId}-selected-path`;
  const evidenceId = `${panelId}-evidence`;

  return (
    <section className="model-policy-panel" aria-labelledby={titleId}>
      <header className="model-policy-panel-header">
        <div>
          <p className="eyebrow">Server-owned compatibility</p>
          <h2 id={titleId}>Model policy</h2>
        </div>
        <p>
          Aptus keeps the artifact decision, candidate binding, and validation evidence
          as separate records.
        </p>
      </header>

      <div className="model-policy-records">
        <ArtifactMatchRecord match={presentation.artifactMatch} headingId={matchId} />
        <SelectedPathRecord selectedPath={presentation.selectedPath} headingId={selectedPathId} />
        <EvidenceReadinessRecord
          selectedPath={presentation.selectedPath}
          readiness={presentation.evidenceReadiness}
          headingId={evidenceId}
        />
      </div>
    </section>
  );
}
