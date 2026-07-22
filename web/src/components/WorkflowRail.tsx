import type { WorkflowStage } from "../types";
import { AptusMark } from "./AptusMark";

export const WORKFLOW_STAGES: Array<{
  id: WorkflowStage;
  label: string;
  description: string;
}> = [
  { id: "facts", label: "Facts", description: "Inspect the inputs" },
  { id: "compare", label: "Compare", description: "Resolve feasibility" },
  { id: "compile", label: "Compile", description: "Build the bundle" },
  { id: "validate", label: "Validate", description: "Pass the gates" },
  { id: "run", label: "Run", description: "Execute with evidence" },
];

interface WorkflowRailProps {
  current: WorkflowStage;
  completed: Set<WorkflowStage>;
  projectName: string;
  connection: "connecting" | "connected" | "unavailable";
  serviceVersion?: string;
  runState?: string;
  onSelect: (stage: WorkflowStage) => void;
}

export function WorkflowRail({
  current,
  completed,
  projectName,
  connection,
  serviceVersion,
  runState,
  onSelect,
}: WorkflowRailProps) {
  return (
    <aside className="workflow-rail" aria-label="Aptus workflow">
      <div className="brand-lockup">
        <AptusMark className="brand-mark" />
        <span>
          <strong>Aptus</strong>
          <small>Fine-tuning workbench</small>
        </span>
      </div>

      <div className="project-chip">
        <span>Current plan</span>
        <strong>{projectName || "Untitled plan"}</strong>
      </div>

      <nav aria-label="Plan stages">
        <ol className="stage-list">
          {WORKFLOW_STAGES.map((stage, index) => {
            const isCurrent = stage.id === current;
            const isComplete = completed.has(stage.id);
            const isTerminalRun = stage.id === "run" && ["failed", "cancelled"].includes(runState ?? "");
            return (
              <li key={stage.id}>
                <button
                  type="button"
                  className={`stage-button${isCurrent ? " is-current" : ""}${isComplete ? " is-complete" : ""}${isTerminalRun ? " is-terminal" : ""}`}
                  aria-current={isCurrent ? "step" : undefined}
                  onClick={() => onSelect(stage.id)}
                >
                  <span className="stage-number" aria-hidden="true">
                    {isComplete ? "✓" : isTerminalRun ? "!" : index + 1}
                  </span>
                  <span>
                    <strong>{stage.label}</strong>
                    <small>{stage.description}</small>
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </nav>

      <div className="rail-connection" data-state={connection}>
        <span className="connection-light" aria-hidden="true" />
        <span>
          <strong>
            {connection === "connected"
              ? "Planner connected"
              : connection === "connecting"
                ? "Connecting"
                : "Planner unavailable"}
          </strong>
          <small>{serviceVersion ? `API ${serviceVersion}` : "Local API"}</small>
        </span>
      </div>
    </aside>
  );
}

interface MobileStageBarProps {
  current: WorkflowStage;
  completed: Set<WorkflowStage>;
  onSelect: (stage: WorkflowStage) => void;
  runState?: string;
}

export function MobileStageBar({ current, completed, onSelect, runState }: MobileStageBarProps) {
  const currentIndex = WORKFLOW_STAGES.findIndex((stage) => stage.id === current);
  return (
    <div className="mobile-stage-bar">
      <label htmlFor="mobile-stage-select">
        Stage {currentIndex + 1} of {WORKFLOW_STAGES.length}
      </label>
      <select
        id="mobile-stage-select"
        value={current}
        onChange={(event) => onSelect(event.target.value as WorkflowStage)}
      >
        {WORKFLOW_STAGES.map((stage) => (
          <option key={stage.id} value={stage.id}>
            {completed.has(stage.id) ? "✓ " : stage.id === "run" && ["failed", "cancelled"].includes(runState ?? "") ? "! " : ""}
            {stage.label}
          </option>
        ))}
      </select>
    </div>
  );
}
