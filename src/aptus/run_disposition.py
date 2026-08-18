"""Operator-attested run disposition (presentation only; not quality).

Pure build/parse of ``aptus.run-disposition.v1``. No I/O. Callers persist and
attach at the job/API/CLI boundary. Aptus never infers the kind.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


DISPOSITION_SCHEMA_VERSION = "aptus.run-disposition.v1"
DISPOSITION_SOURCE = "operator-attested"
DISPOSITION_KINDS = frozenset({"use", "done", "stop"})
EVALUATION_DECISIONS = frozenset({"pass", "fail", "abstain", "omitted"})

DISPOSITION_NON_CLAIMS = (
    "Training finished is not this decision.",
    "Training loss is not this decision.",
    "Gold exact-match is not general model quality.",
    "This is not a 0.2 ship, freeze, or stop.",
)

_NEXT_STEP_BY_KIND: dict[str, tuple[str, str]] = {
    "use": ("load-adapter", "Load this adapter"),
    "done": ("none", "I'm finished training this."),
    "stop": ("none", "Don't use this. Don't train this again."),
}


@dataclass(frozen=True)
class DispositionEvidence:
    validation_state: str | None
    run_correction_kind: str | None
    evaluation_decision: str

    def to_primitive(self) -> dict[str, object]:
        return {
            "validation_state": self.validation_state,
            "run_correction_kind": self.run_correction_kind,
            "evaluation_decision": self.evaluation_decision,
        }


@dataclass(frozen=True)
class DispositionNextStep:
    action: str
    label: str

    def to_primitive(self) -> dict[str, object]:
        return {"action": self.action, "label": self.label}


@dataclass(frozen=True)
class RunDisposition:
    kind: str
    job_id: str
    plan_id: str
    candidate_id: str
    run_id: str | None
    attested_at: str
    previous_kind: str | None
    evidence: DispositionEvidence
    operator_next_step: DispositionNextStep
    schema_version: str = DISPOSITION_SCHEMA_VERSION
    source: str = DISPOSITION_SOURCE
    non_claims: tuple[str, ...] = DISPOSITION_NON_CLAIMS

    def to_primitive(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "job_id": self.job_id,
            "plan_id": self.plan_id,
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "attested_at": self.attested_at,
            "previous_kind": self.previous_kind,
            "source": self.source,
            "evidence": self.evidence.to_primitive(),
            "operator_next_step": self.operator_next_step.to_primitive(),
            "non_claims": list(self.non_claims),
        }


def next_step_for_kind(kind: str) -> tuple[str, str]:
    """Return ``(action, label)`` for a disposition kind."""

    try:
        return _NEXT_STEP_BY_KIND[kind]
    except KeyError as exc:
        raise ValueError(
            f"kind must be one of: {', '.join(sorted(DISPOSITION_KINDS))}."
        ) from exc


def build_run_disposition(
    *,
    kind: str,
    job_id: str,
    plan_id: str,
    candidate_id: str,
    run_id: str | None,
    attested_at: str,
    previous_kind: str | None,
    validation_state: str | None,
    run_correction_kind: str | None,
    evaluation_decision: str | None,
) -> RunDisposition:
    """Build a pure operator-attested run disposition object."""

    if kind not in DISPOSITION_KINDS:
        raise ValueError(
            f"kind must be one of: {', '.join(sorted(DISPOSITION_KINDS))}."
        )
    if previous_kind is not None and previous_kind not in DISPOSITION_KINDS:
        raise ValueError(
            "previous_kind must be null or one of: "
            f"{', '.join(sorted(DISPOSITION_KINDS))}."
        )
    decision = "omitted" if evaluation_decision is None else evaluation_decision
    if decision not in EVALUATION_DECISIONS:
        raise ValueError(
            "evaluation_decision must be one of: "
            f"{', '.join(sorted(EVALUATION_DECISIONS))}."
        )
    action, label = next_step_for_kind(kind)
    return RunDisposition(
        kind=kind,
        job_id=job_id,
        plan_id=plan_id,
        candidate_id=candidate_id,
        run_id=run_id,
        attested_at=attested_at,
        previous_kind=previous_kind,
        evidence=DispositionEvidence(
            validation_state=validation_state,
            run_correction_kind=run_correction_kind,
            evaluation_decision=decision,
        ),
        operator_next_step=DispositionNextStep(action=action, label=label),
    )


def run_disposition_from_primitive(payload: Mapping[str, Any]) -> RunDisposition:
    """Parse and validate an ``aptus.run-disposition.v1`` mapping."""

    if not isinstance(payload, Mapping):
        raise ValueError("Run disposition must be a JSON object.")
    if payload.get("schema_version") != DISPOSITION_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {DISPOSITION_SCHEMA_VERSION}.")
    if payload.get("source") != DISPOSITION_SOURCE:
        raise ValueError(f"source must be {DISPOSITION_SOURCE!r}.")

    kind = payload.get("kind")
    if kind not in DISPOSITION_KINDS:
        raise ValueError(
            f"kind must be one of: {', '.join(sorted(DISPOSITION_KINDS))}."
        )

    previous_kind = payload.get("previous_kind")
    if previous_kind is not None and previous_kind not in DISPOSITION_KINDS:
        raise ValueError(
            "previous_kind must be null or one of: "
            f"{', '.join(sorted(DISPOSITION_KINDS))}."
        )

    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be an object.")
    evaluation_decision = evidence.get("evaluation_decision")
    if evaluation_decision not in EVALUATION_DECISIONS:
        raise ValueError(
            "evaluation_decision must be one of: "
            f"{', '.join(sorted(EVALUATION_DECISIONS))}."
        )

    next_step = payload.get("operator_next_step")
    if not isinstance(next_step, Mapping):
        raise ValueError("operator_next_step must be an object.")
    action = next_step.get("action")
    label = next_step.get("label")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("operator_next_step.action must be a non-empty string.")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("operator_next_step.label must be a non-empty string.")
    expected_action, expected_label = next_step_for_kind(kind)
    if action != expected_action or label != expected_label:
        raise ValueError(
            f"operator_next_step for kind {kind!r} must be "
            f"action={expected_action!r}, label={expected_label!r}."
        )

    non_claims = payload.get("non_claims")
    if not isinstance(non_claims, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in non_claims
    ):
        raise ValueError("non_claims must be a list of non-empty strings.")
    missing_non_claims = [
        item for item in DISPOSITION_NON_CLAIMS if item not in non_claims
    ]
    if missing_non_claims:
        raise ValueError("Run disposition is missing required non_claims.")

    job_id = payload.get("job_id")
    plan_id = payload.get("plan_id")
    candidate_id = payload.get("candidate_id")
    attested_at = payload.get("attested_at")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a non-empty string.")
    if not isinstance(plan_id, str) or not plan_id.strip():
        raise ValueError("plan_id must be a non-empty string.")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty string.")
    if not isinstance(attested_at, str) or not attested_at.strip():
        raise ValueError("attested_at must be a non-empty string.")

    run_id = payload.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError("run_id must be a non-empty string or null.")

    validation_state = evidence.get("validation_state")
    if validation_state is not None and not isinstance(validation_state, str):
        raise ValueError("validation_state must be a string or null.")
    run_correction_kind = evidence.get("run_correction_kind")
    if run_correction_kind is not None and not isinstance(run_correction_kind, str):
        raise ValueError("run_correction_kind must be a string or null.")

    return RunDisposition(
        kind=kind,
        job_id=job_id,
        plan_id=plan_id,
        candidate_id=candidate_id,
        run_id=run_id,
        attested_at=attested_at,
        previous_kind=previous_kind,
        evidence=DispositionEvidence(
            validation_state=validation_state,
            run_correction_kind=run_correction_kind,
            evaluation_decision=evaluation_decision,
        ),
        operator_next_step=DispositionNextStep(action=action, label=label),
        non_claims=tuple(str(item) for item in non_claims),
    )
