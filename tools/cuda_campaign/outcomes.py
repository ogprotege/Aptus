"""Exact native-outcome and evidence-status profiles for managed CUDA runs.

The campaign has two independent terminal axes.  ``native_outcome`` describes
what Aptus did; ``evidence_status`` describes whether the attempt was captured
well enough to support protocol claims.  A completely captured refusal or
cancellation can therefore be ``protocol-valid``, while a native pass can be
``capture-invalid``.  Publication eligibility is deliberately narrower: only
``passed`` intersected with ``protocol-valid`` is eligible.

This module validates the frozen five-action managed sequence without changing
the raw-manifest vocabulary.  It also owns the narrowly scoped exception used
by :func:`tools.cuda_campaign.contracts.validate_event_ledger`: a runtime start
may remain unmatched only when a terminal non-pass profile closes the command,
and a cancellation or timeout additionally has the exact safety chain.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    NATIVE_OUTCOMES,
    REASON_CODES,
    ContractError,
    validate_event_ledger,
)


MANAGED_ACTION_ORDER = (
    "dependency",
    "model-data",
    "preflight",
    "pilot",
    "train",
)
CONDITIONING_ACTION_ORDER = MANAGED_ACTION_ORDER[:-1]
RUNTIME_BOUNDARY_ORDER = (
    ("pilot.phase-started", "pilot-phase-1", "pilot"),
    ("pilot.phase-finished", "pilot-phase-1", "pilot"),
    ("pilot.phase-started", "pilot-phase-2", "pilot"),
    ("pilot.phase-finished", "pilot-phase-2", "pilot"),
    ("training.started", "training", "train"),
    ("export.started", "final-export", "train"),
    ("export.finished", "final-export", "train"),
    ("training.finished", "training", "train"),
    ("verification.started", "parent-verification", "train"),
    ("verification.finished", "parent-verification", "train"),
)
RUNTIME_EVENT_TYPES = frozenset(item[0] for item in RUNTIME_BOUNDARY_ORDER)
CANCELLATION_EVENT_ORDER = (
    "safety.triggered",
    "cancellation.requested",
    "process-group.terminated",
    "lease.reconciled",
)
DEADLINE_REASON_CODE = "EMERGENCY_DEADLINE_EXCEEDED"

_ACTION_LABEL = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
_STARTED_ACTION_FIELDS = frozenset(
    {
        "label",
        "action",
        "job_id",
        "native_outcome",
        "reason_code",
        "terminal",
        "capture_reason_code",
    }
)
_SAFETY_EVENT_TYPES = frozenset(CANCELLATION_EVENT_ORDER)
_RUNTIME_STARTED_TYPES = frozenset(
    event_type
    for event_type, _phase, _action in RUNTIME_BOUNDARY_ORDER
    if event_type.endswith("started")
)


class OutcomeProfileError(ValueError):
    """A managed terminal ledger does not match one exact outcome profile."""


@dataclass(frozen=True)
class ManagedOutcomeProfile:
    """Validated independent terminal axes and their stopping boundary."""

    native_outcome: str
    evidence_status: str
    reason_code: str
    stopping_action: str
    started_action_count: int
    runtime_boundary_count: int
    sequence_profile: str

    @property
    def publication_eligible(self) -> bool:
        return (
            self.native_outcome == "passed" and self.evidence_status == "protocol-valid"
        )


@dataclass(frozen=True)
class _ActionResult:
    label: str
    action: str
    job_id: str | None
    native_outcome: str
    reason_code: str
    terminal: bool
    capture_reason_code: str


def is_publication_eligible(native_outcome: str, evidence_status: str) -> bool:
    """Return the frozen publication intersection, rejecting unknown values."""

    if native_outcome not in NATIVE_OUTCOMES:
        raise OutcomeProfileError("native outcome is invalid")
    if evidence_status not in {"protocol-valid", "capture-invalid"}:
        raise OutcomeProfileError("started evidence status is invalid")
    return native_outcome == "passed" and evidence_status == "protocol-valid"


def _parse_configured_actions(value: Any) -> tuple[tuple[str, str], ...]:
    if type(value) is not list or len(value) not in {
        len(CONDITIONING_ACTION_ORDER),
        len(MANAGED_ACTION_ORDER),
    }:
        raise OutcomeProfileError(
            "configured actions are not a frozen conditioning or measured sequence"
        )
    expected_order = MANAGED_ACTION_ORDER[: len(value)]
    parsed: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if type(item) is not dict:
            raise OutcomeProfileError("configured action is not an exact object")
        label = item.get("label")
        action = item.get("action")
        if (
            not isinstance(label, str)
            or _ACTION_LABEL.fullmatch(label) is None
            or action != expected_order[index]
        ):
            raise OutcomeProfileError("configured action order or label is invalid")
        parsed.append((label, action))
    if len({label for label, _action in parsed}) != len(parsed):
        raise OutcomeProfileError("configured action labels are not unique")
    return tuple(parsed)


def _parse_started_actions(
    value: Any,
    configured: tuple[tuple[str, str], ...],
) -> tuple[_ActionResult, ...]:
    if type(value) is not list or not 1 <= len(value) <= len(configured):
        raise OutcomeProfileError("started actions are not a non-empty frozen prefix")
    parsed: list[_ActionResult] = []
    for index, item in enumerate(value):
        if type(item) is not dict or set(item) != _STARTED_ACTION_FIELDS:
            raise OutcomeProfileError(
                "started action has the wrong exact outcome fields"
            )
        label = item["label"]
        action = item["action"]
        job_id = item["job_id"]
        native_outcome = item["native_outcome"]
        reason_code = item["reason_code"]
        terminal = item["terminal"]
        capture_reason_code = item["capture_reason_code"]
        if (label, action) != configured[index]:
            raise OutcomeProfileError("started actions are reordered or inserted")
        if job_id is not None and (
            not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None
        ):
            raise OutcomeProfileError("started action job identity is invalid")
        if native_outcome not in NATIVE_OUTCOMES:
            raise OutcomeProfileError("started action native outcome is invalid")
        if reason_code not in REASON_CODES or capture_reason_code not in REASON_CODES:
            raise OutcomeProfileError("started action reason code is invalid")
        if type(terminal) is not bool:
            raise OutcomeProfileError("started action terminal flag is invalid")
        if native_outcome == "passed":
            if reason_code != "NONE" or not terminal or job_id is None:
                raise OutcomeProfileError("a passed action is not an exact job pass")
        else:
            if reason_code == "NONE":
                raise OutcomeProfileError("a non-pass action lacks its reason")
            if native_outcome != "unknown" and not terminal:
                raise OutcomeProfileError("a known non-pass action is not terminal")
            if native_outcome in {"refused", "guard-blocked"} and job_id is not None:
                raise OutcomeProfileError("pre-submit refusal or guard must be jobless")
            if native_outcome in {"failed", "cancelled", "timed-out"} and (
                job_id is None
            ):
                raise OutcomeProfileError("post-submit non-pass lacks its job identity")
        parsed.append(
            _ActionResult(
                label=label,
                action=action,
                job_id=job_id,
                native_outcome=native_outcome,
                reason_code=reason_code,
                terminal=terminal,
                capture_reason_code=capture_reason_code,
            )
        )
    if any(item.native_outcome != "passed" for item in parsed[:-1]):
        raise OutcomeProfileError("an action appears after the stopping action")
    return tuple(parsed)


def _require_terminal_axes(
    summary: Mapping[str, Any],
    actions: tuple[_ActionResult, ...],
    configured: tuple[tuple[str, str], ...],
) -> tuple[str, str, str]:
    native_outcome = summary.get("native_outcome")
    evidence_status = summary.get("evidence_status")
    reason_code = summary.get("reason_code")
    capture_reason_code = summary.get("capture_reason_code")
    if native_outcome not in NATIVE_OUTCOMES:
        raise OutcomeProfileError("summary native outcome is invalid")
    if evidence_status not in {"protocol-valid", "capture-invalid"}:
        raise OutcomeProfileError("summary evidence status is invalid")
    if reason_code not in REASON_CODES:
        raise OutcomeProfileError("summary reason code is invalid")
    if capture_reason_code not in REASON_CODES:
        raise OutcomeProfileError("summary capture reason code is invalid")
    capture_reasons = [
        capture_reason_code,
        *(item.capture_reason_code for item in actions),
    ]
    capture_is_clean = all(item == "NONE" for item in capture_reasons)
    if (evidence_status == "protocol-valid") != capture_is_clean:
        raise OutcomeProfileError(
            "summary evidence status contradicts its capture reason codes"
        )
    final = actions[-1]
    if (native_outcome, reason_code) != (
        final.native_outcome,
        final.reason_code,
    ):
        raise OutcomeProfileError(
            "summary disposition differs from the stopping action"
        )
    if native_outcome == "passed":
        if len(actions) != len(configured) or reason_code != "NONE":
            raise OutcomeProfileError("a pass does not complete its frozen sequence")
    elif reason_code == "NONE":
        raise OutcomeProfileError("a non-pass summary lacks its reason")
    if native_outcome == "refused" and reason_code != "APTUS_ADMISSION_REFUSAL":
        raise OutcomeProfileError("an admission refusal has the wrong reason")
    if native_outcome == "timed-out" and reason_code != DEADLINE_REASON_CODE:
        raise OutcomeProfileError("a timeout must be caused by the emergency deadline")
    if native_outcome == "cancelled" and reason_code == DEADLINE_REASON_CODE:
        raise OutcomeProfileError("the emergency deadline maps to timed-out")
    stopped_early = summary.get("stopped_early")
    if type(stopped_early) is not bool or stopped_early != (native_outcome != "passed"):
        raise OutcomeProfileError("summary stopped_early disagrees with the outcome")
    return native_outcome, evidence_status, reason_code


def _runtime_rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in records if row["event_type"] in RUNTIME_EVENT_TYPES]


def _validate_runtime_shape(
    records: Sequence[Mapping[str, Any]],
    *,
    action_results: tuple[_ActionResult, ...] | None,
    native_outcome: str,
    reason_code: str,
    evidence_status: str | None = None,
    conditioning: bool = False,
) -> list[Mapping[str, Any]]:
    runtime = _runtime_rows(records)
    expected_boundary_count = 4 if conditioning else len(RUNTIME_BOUNDARY_ORDER)
    if len(runtime) > expected_boundary_count:
        raise OutcomeProfileError("runtime boundary sequence contains an insertion")
    job_by_action: dict[str, str] = {}
    if action_results is not None:
        job_by_action = {
            item.action: item.job_id
            for item in action_results
            if item.job_id is not None
        }
    observed_jobs: dict[str, str] = {}
    open_boundaries: list[tuple[str, str, str, str]] = []
    matching_start = {
        "pilot.phase-finished": "pilot.phase-started",
        "training.finished": "training.started",
        "export.finished": "export.started",
        "verification.finished": "verification.started",
    }
    state = "pilot-phase-1-start"
    propagated_failure: tuple[str, str] | None = None
    exact_state = {
        "pilot-phase-1-start": (
            "pilot.phase-started",
            "pilot-phase-1",
            "pilot",
        ),
        "pilot-phase-1-finish": (
            "pilot.phase-finished",
            "pilot-phase-1",
            "pilot",
        ),
        "pilot-phase-2-start": (
            "pilot.phase-started",
            "pilot-phase-2",
            "pilot",
        ),
        "pilot-phase-2-finish": (
            "pilot.phase-finished",
            "pilot-phase-2",
            "pilot",
        ),
        "training-start": ("training.started", "training", "train"),
        "export-finish": ("export.finished", "final-export", "train"),
        "training-finish": ("training.finished", "training", "train"),
        "verification-start": (
            "verification.started",
            "parent-verification",
            "train",
        ),
        "verification-finish": (
            "verification.finished",
            "parent-verification",
            "train",
        ),
    }
    for row in runtime:
        if state == "training-body":
            allowed = {
                ("export.started", "final-export", "train"),
                ("training.finished", "training", "train"),
            }
        elif state == "training-finish-after-export-failure":
            allowed = {("training.finished", "training", "train")}
        elif state == "terminal":
            allowed = set()
        else:
            allowed = {exact_state[state]}
        if (
            (row["event_type"], row["phase"], row["action"]) not in allowed
            or row["subject_kind"] != "aptus-job"
            or row["observation_kind"] != "emitted"
            or row["exit_code"] is not None
            or not isinstance(row["subject_id"], str)
        ):
            raise OutcomeProfileError("runtime boundary state is reordered or misbound")
        subject_id = row["subject_id"]
        expected_action = str(row["action"])
        expected_type = str(row["event_type"])
        expected_phase = str(row["phase"])
        expected_job_id = job_by_action.get(expected_action)
        if expected_job_id is not None and subject_id != expected_job_id:
            raise OutcomeProfileError("runtime boundary uses the wrong action job")
        prior_job = observed_jobs.setdefault(expected_action, subject_id)
        if prior_job != subject_id:
            raise OutcomeProfileError("runtime action mixes job identities")
        if expected_type.endswith("started"):
            if row["native_outcome"] is not None or row["reason_code"] != "NONE":
                raise OutcomeProfileError("runtime start is spuriously terminal")
            open_boundaries.append(
                (expected_type, expected_phase, expected_action, subject_id)
            )
            state = {
                "pilot-phase-1-start": "pilot-phase-1-finish",
                "pilot-phase-2-start": "pilot-phase-2-finish",
                "training-start": "training-body",
                "verification-start": "verification-finish",
                "training-body": "export-finish",
            }[state]
            continue
        wanted_start = matching_start[expected_type]
        wanted = (wanted_start, expected_phase, expected_action, subject_id)
        if not open_boundaries or open_boundaries[-1] != wanted:
            raise OutcomeProfileError("runtime finish does not close its exact start")
        open_boundaries.pop()
        row_outcome = row["native_outcome"]
        row_reason = row["reason_code"]
        if row_outcome == "passed" and row_reason == "NONE":
            if state == "training-body":
                raise OutcomeProfileError(
                    "training cannot pass before the final export closes"
                )
            if propagated_failure is not None:
                raise OutcomeProfileError(
                    "an outer runtime boundary contradicts its failed child"
                )
            state = {
                "pilot-phase-1-finish": "pilot-phase-2-start",
                "pilot-phase-2-finish": (
                    "terminal" if conditioning else "training-start"
                ),
                "export-finish": "training-finish",
                "training-finish": "verification-start",
                "verification-finish": "terminal",
            }[state]
            continue
        expected_terminal = (
            "cancelled"
            if native_outcome in {"cancelled", "timed-out"}
            else native_outcome
        )
        if (
            row_outcome != expected_terminal
            or row_reason != reason_code
            or native_outcome not in {"failed", "cancelled", "timed-out"}
        ):
            raise OutcomeProfileError(
                "runtime terminal boundary contradicts the outcome"
            )
        if state == "export-finish":
            propagated_failure = (str(row_outcome), str(row_reason))
            state = "training-finish-after-export-failure"
        elif state == "training-finish-after-export-failure":
            if propagated_failure != (row_outcome, row_reason):
                raise OutcomeProfileError(
                    "an outer runtime boundary contradicts its failed child"
                )
            propagated_failure = None
            state = "terminal"
        else:
            state = "terminal"

    if native_outcome == "passed":
        if (
            len(runtime) != expected_boundary_count
            or open_boundaries
            or state != "terminal"
        ):
            raise OutcomeProfileError(
                "a pass requires the exact ten runtime boundaries"
            )
    elif action_results is not None:
        stopping_action = action_results[-1].action
        if stopping_action in MANAGED_ACTION_ORDER[:3] and runtime:
            raise OutcomeProfileError("runtime starts before the pilot action")
        if stopping_action == "pilot" and len(runtime) > 4:
            raise OutcomeProfileError(
                "runtime continues after the pilot stopping action"
            )
        if stopping_action == "train" and len(runtime) < 4:
            raise OutcomeProfileError("train starts without a completed pilot runtime")
        if native_outcome in {"refused", "guard-blocked"}:
            expected_count = 4 if stopping_action == "train" else 0
            if len(runtime) != expected_count:
                raise OutcomeProfileError(
                    "a pre-submit stop contains stopping-action runtime evidence"
                )
    if (
        state == "training-finish-after-export-failure"
        and evidence_status == "protocol-valid"
    ):
        raise OutcomeProfileError(
            "a protocol-valid export failure lacks its enclosing training finish"
        )
    return runtime


def _action_command_pairs(
    records: Sequence[Mapping[str, Any]],
    actions: tuple[_ActionResult, ...],
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    commands = [
        row
        for row in records
        if row["event_type"] in {"command.started", "command.finished"}
    ]
    if len(commands) != len(actions) * 2:
        raise OutcomeProfileError("command boundaries do not match the started prefix")
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for index, result in enumerate(actions):
        started, finished = commands[index * 2 : index * 2 + 2]
        identity = (
            result.label,
            result.action,
            "managed-action",
            result.label,
        )
        if (
            started["event_type"] != "command.started"
            or finished["event_type"] != "command.finished"
            or (
                started["phase"],
                started["action"],
                started["subject_kind"],
                started["subject_id"],
            )
            != identity
            or (
                finished["phase"],
                finished["action"],
                finished["subject_kind"],
                finished["subject_id"],
            )
            != identity
            or started["exit_code"] is not None
            or started["native_outcome"] is not None
            or started["reason_code"] != "NONE"
            or finished["native_outcome"] != result.native_outcome
            or finished["reason_code"] != result.reason_code
            or started["sequence"] >= finished["sequence"]
        ):
            raise OutcomeProfileError(
                "command boundary is reordered, spoofed, or terminal"
            )
        if result.native_outcome == "passed" and finished["exit_code"] != 0:
            raise OutcomeProfileError("passed command does not have exit code zero")
        if result.native_outcome in {"refused", "guard-blocked"} and (
            finished["exit_code"] is not None
        ):
            raise OutcomeProfileError("jobless pre-submit stop has an exit code")
        pairs.append((started, finished))
    return tuple(pairs)


def _validate_job_observations(
    records: Sequence[Mapping[str, Any]],
    actions: tuple[_ActionResult, ...],
    pairs: tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...],
    *,
    evidence_status: str,
) -> None:
    job_scoped_types = (
        RUNTIME_EVENT_TYPES | _SAFETY_EVENT_TYPES | {"job.state-observed"}
    )
    pair_by_action = {
        result.action: (result, started, finished)
        for result, (started, finished) in zip(actions, pairs)
    }
    for row in records:
        if row["event_type"] != "job.state-observed":
            continue
        binding = pair_by_action.get(row["action"])
        if binding is None:
            raise OutcomeProfileError(
                "job state belongs to an action that never started"
            )
        result, started, finished = binding
        if (
            result.job_id is None
            or row["subject_kind"] != "aptus-job"
            or row["subject_id"] != result.job_id
            or not started["sequence"] < row["sequence"] < finished["sequence"]
        ):
            raise OutcomeProfileError("job state lies outside its exact action command")
    for result, (started, finished) in zip(actions, pairs):
        scoped = [
            row
            for row in records
            if started["sequence"] < row["sequence"] < finished["sequence"]
            and row["event_type"] in job_scoped_types
            and row["action"] == result.action
        ]
        if result.job_id is None:
            if any(row["subject_kind"] == "aptus-job" for row in scoped):
                raise OutcomeProfileError(
                    "a jobless action contains Aptus job evidence"
                )
            continue
        if any(
            row["subject_kind"] == "aptus-job" and row["subject_id"] != result.job_id
            for row in scoped
        ):
            raise OutcomeProfileError("an action mixes Aptus job identities")
        observations = [
            row
            for row in scoped
            if row["event_type"] == "job.state-observed"
            and row["subject_kind"] == "aptus-job"
            and row["subject_id"] == result.job_id
        ]
        if any(row["observation_kind"] != "observed" for row in observations):
            raise OutcomeProfileError("job state is not observed evidence")
        terminals = [row for row in observations if row["native_outcome"] is not None]
        if len(terminals) > 1 or (terminals and terminals[-1] is not observations[-1]):
            raise OutcomeProfileError(
                "job states continue after a terminal observation"
            )
        if terminals:
            expected_job_outcome = (
                "cancelled"
                if result.native_outcome == "timed-out"
                else result.native_outcome
            )
            terminal = terminals[0]
            if (
                result.native_outcome == "unknown"
                or terminal["native_outcome"] != expected_job_outcome
                or terminal["reason_code"] != result.reason_code
            ):
                raise OutcomeProfileError(
                    "observed terminal job state contradicts action"
                )
        terminal_required = result.native_outcome in {"passed", "failed"} and (
            result.native_outcome == "passed" or evidence_status == "protocol-valid"
        )
        if terminal_required:
            if not terminals:
                raise OutcomeProfileError(
                    "action lacks its observed terminal job state"
                )


def _safety_rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in records if row["event_type"] in _SAFETY_EVENT_TYPES]


def _validate_exact_cancellation_chain(
    records: Sequence[Mapping[str, Any]],
    *,
    action: str,
    phase: str,
    job_id: str,
    native_outcome: str,
    reason_code: str,
    after_sequence: int,
    before_sequence: int | None,
) -> None:
    chain = _safety_rows(records)
    if [row["event_type"] for row in chain] != list(CANCELLATION_EVENT_ORDER):
        raise OutcomeProfileError(
            "cancellation safety chain is incomplete or reordered"
        )
    for index, row in enumerate(chain):
        if (
            row["phase"] != phase
            or row["action"] != action
            or row["subject_kind"] != "aptus-job"
            or row["subject_id"] != job_id
            or row["reason_code"] != reason_code
            or row["exit_code"] is not None
            or row["observation_kind"] != "observed"
            or row["native_outcome"] != (None if index == 0 else "cancelled")
            or row["sequence"] <= after_sequence
            or (before_sequence is not None and row["sequence"] >= before_sequence)
        ):
            raise OutcomeProfileError("cancellation safety chain is misbound")
    if native_outcome == "timed-out":
        if reason_code != DEADLINE_REASON_CODE:
            raise OutcomeProfileError("non-deadline safety reason cannot be timed-out")
    elif native_outcome == "cancelled":
        if reason_code == DEADLINE_REASON_CODE:
            raise OutcomeProfileError("deadline cancellation must be timed-out")
    else:
        raise OutcomeProfileError("cancellation chain closes the wrong native outcome")


def _validate_safety_profile(
    records: Sequence[Mapping[str, Any]],
    *,
    actions: tuple[_ActionResult, ...],
    pairs: tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...],
    runtime: Sequence[Mapping[str, Any]],
    native_outcome: str,
    reason_code: str,
) -> None:
    final = actions[-1]
    started, finished = pairs[-1]
    safety = _safety_rows(records)
    if native_outcome in {"cancelled", "timed-out"}:
        if final.job_id is None:
            raise OutcomeProfileError("cancellation profile lacks a job")
        last_runtime_sequence = (
            runtime[-1]["sequence"] if runtime else started["sequence"]
        )
        _validate_exact_cancellation_chain(
            records,
            action=final.action,
            phase=final.label,
            job_id=final.job_id,
            native_outcome=native_outcome,
            reason_code=reason_code,
            after_sequence=last_runtime_sequence,
            before_sequence=finished["sequence"],
        )
        return
    if native_outcome == "guard-blocked":
        if len(safety) != 1:
            raise OutcomeProfileError(
                "pre-submit guard requires one exact safety trigger"
            )
        trigger = safety[0]
        if (
            trigger["event_type"] != "safety.triggered"
            or trigger["phase"] != final.label
            or trigger["action"] != final.action
            or trigger["subject_kind"] != "managed-action"
            or trigger["subject_id"] != final.label
            or trigger["reason_code"] != reason_code
            or trigger["native_outcome"] is not None
            or trigger["exit_code"] is not None
            or trigger["observation_kind"] != "observed"
            or not started["sequence"] < trigger["sequence"] < finished["sequence"]
        ):
            raise OutcomeProfileError("pre-submit safety trigger is misbound")
        return
    if safety:
        raise OutcomeProfileError("non-cancellation outcome contains safety milestones")


def _validate_terminal_envelope(
    records: Sequence[Mapping[str, Any]],
    *,
    native_outcome: str,
    reason_code: str,
) -> None:
    run_id = records[0]["experiment_run_id"]
    for event_type in ("harness.finished", "seal.started"):
        matching = [row for row in records if row["event_type"] == event_type]
        if len(matching) != 1:
            raise OutcomeProfileError(f"ledger lacks one exact {event_type}")
        row = matching[0]
        if (
            row["subject_kind"] != "experiment-run"
            or row["subject_id"] != run_id
            or row["native_outcome"] != native_outcome
            or row["reason_code"] != reason_code
        ):
            raise OutcomeProfileError(
                "terminal harness envelope contradicts the summary"
            )


def validate_managed_sequence_outcome(
    summary: Mapping[str, Any],
    ledger_records: Sequence[Mapping[str, Any]],
) -> ManagedOutcomeProfile:
    """Validate one exact five-action native/evidence terminal profile.

    ``summary`` is the managed sequence summary produced by the capture harness.
    Additional summary fields remain forward-compatible, but every field used
    for the terminal profile is required and validated fail closed.
    """

    if type(summary) is not dict:
        raise OutcomeProfileError("managed sequence summary is not an object")
    configured = _parse_configured_actions(summary.get("configured_actions"))
    conditioning = len(configured) == len(CONDITIONING_ACTION_ORDER)
    actions = _parse_started_actions(summary.get("started_actions"), configured)
    native_outcome, evidence_status, reason_code = _require_terminal_axes(
        summary, actions, configured
    )
    try:
        ledger = validate_event_ledger(ledger_records)
    except ContractError as error:
        raise OutcomeProfileError(
            "managed event ledger is structurally invalid"
        ) from error
    _validate_terminal_envelope(
        ledger,
        native_outcome=native_outcome,
        reason_code=reason_code,
    )
    pairs = _action_command_pairs(ledger, actions)
    runtime = _validate_runtime_shape(
        ledger,
        action_results=actions,
        native_outcome=native_outcome,
        reason_code=reason_code,
        evidence_status=evidence_status,
        conditioning=conditioning,
    )
    pair_by_action = {result.action: pair for result, pair in zip(actions, pairs)}
    for row in runtime:
        try:
            action_started, action_finished = pair_by_action[str(row["action"])]
        except KeyError:
            raise OutcomeProfileError(
                "runtime boundary belongs to an action that never started"
            ) from None
        if not (
            action_started["sequence"] < row["sequence"] < action_finished["sequence"]
        ):
            raise OutcomeProfileError(
                "runtime boundary lies outside its exact action command"
            )
    _validate_job_observations(
        ledger,
        actions,
        pairs,
        evidence_status=evidence_status,
    )
    _validate_safety_profile(
        ledger,
        actions=actions,
        pairs=pairs,
        runtime=runtime,
        native_outcome=native_outcome,
        reason_code=reason_code,
    )
    return ManagedOutcomeProfile(
        native_outcome=native_outcome,
        evidence_status=evidence_status,
        reason_code=reason_code,
        stopping_action=actions[-1].action,
        started_action_count=len(actions),
        runtime_boundary_count=len(runtime),
        sequence_profile="conditioning" if conditioning else "measured",
    )


def validate_unmatched_runtime_terminal_prefix(
    records: Sequence[Mapping[str, Any]],
    unmatched_starts: Sequence[Mapping[str, Any]],
) -> None:
    """Authorize the generic ledger's sole incomplete-runtime exception.

    The caller has already validated row schemas, global sequence/time, and the
    harness envelope.  This function intentionally does not call
    :func:`validate_event_ledger`, avoiding recursion when invoked from that
    validator.
    """

    if not unmatched_starts or any(
        row["event_type"] not in _RUNTIME_STARTED_TYPES for row in unmatched_starts
    ):
        raise OutcomeProfileError("only runtime-start boundaries may remain unmatched")
    final = records[-1]
    harness = records[-3]
    native_outcome = final["native_outcome"]
    reason_code = final["reason_code"]
    run_id = final["experiment_run_id"]
    if (
        native_outcome not in {"failed", "cancelled", "timed-out", "unknown"}
        or reason_code == "NONE"
        or harness["native_outcome"] != native_outcome
        or harness["reason_code"] != reason_code
        or harness["subject_kind"] != "experiment-run"
        or harness["subject_id"] != run_id
        or final["subject_kind"] != "experiment-run"
        or final["subject_id"] != run_id
    ):
        raise OutcomeProfileError("unmatched runtime start lacks a terminal non-pass")
    commands = [
        row
        for row in records
        if row["event_type"] in {"command.started", "command.finished"}
    ]
    if (
        len(commands) < 2
        or len(commands) % 2
        or commands[-2]["event_type"] != "command.started"
        or (commands[-1]["event_type"] != "command.finished")
    ):
        raise OutcomeProfileError("unmatched runtime start lacks its stopping command")
    command_pairs: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for index in range(0, len(commands), 2):
        started = commands[index]
        finished = commands[index + 1]
        action_index = index // 2
        if action_index >= len(MANAGED_ACTION_ORDER):
            raise OutcomeProfileError("unmatched runtime is not in a managed prefix")
        expected_action = MANAGED_ACTION_ORDER[action_index]
        identity = (
            started["phase"],
            started["action"],
            started["subject_kind"],
            started["subject_id"],
        )
        if (
            started["event_type"] != "command.started"
            or finished["event_type"] != "command.finished"
            or started["action"] != expected_action
            or identity
            != (
                finished["phase"],
                finished["action"],
                finished["subject_kind"],
                finished["subject_id"],
            )
            or started["subject_kind"] != "managed-action"
            or started["native_outcome"] is not None
            or started["reason_code"] != "NONE"
            or started["exit_code"] is not None
            or (
                index < len(commands) - 2
                and (
                    finished["native_outcome"] != "passed"
                    or finished["reason_code"] != "NONE"
                    or finished["exit_code"] != 0
                )
            )
        ):
            raise OutcomeProfileError("unmatched runtime is not in a managed prefix")
        command_pairs[expected_action] = (started, finished)
    command_start, command_finish = commands[-2:]
    if (
        command_finish["native_outcome"] != native_outcome
        or command_finish["reason_code"] != reason_code
        or command_finish["action"] not in {"pilot", "train"}
        or command_start["action"] != command_finish["action"]
        or command_start["phase"] != command_finish["phase"]
        or command_start["subject_kind"] != "managed-action"
        or command_finish["subject_kind"] != "managed-action"
        or command_start["subject_id"] != command_finish["subject_id"]
    ):
        raise OutcomeProfileError("stopping command does not close the runtime prefix")
    runtime = _validate_runtime_shape(
        records,
        action_results=None,
        native_outcome=native_outcome,
        reason_code=reason_code,
    )
    action_job_ids: dict[str, str] = {}
    job_scoped_types = (
        RUNTIME_EVENT_TYPES | _SAFETY_EVENT_TYPES | {"job.state-observed"}
    )
    for row in records:
        if row["event_type"] not in job_scoped_types or row["subject_kind"] != (
            "aptus-job"
        ):
            continue
        action = row["action"]
        pair = command_pairs.get(action)
        subject_id = row["subject_id"]
        if (
            pair is None
            or not isinstance(subject_id, str)
            or not pair[0]["sequence"] < row["sequence"] < pair[1]["sequence"]
        ):
            raise OutcomeProfileError("job evidence lies outside its managed command")
        prior_job_id = action_job_ids.setdefault(str(action), subject_id)
        if prior_job_id != subject_id:
            raise OutcomeProfileError("managed command mixes Aptus job identities")
    active_runtime: list[Mapping[str, Any]] = []
    for row in runtime:
        if row["event_type"].endswith("started"):
            active_runtime.append(row)
        else:
            if not active_runtime:
                raise OutcomeProfileError("runtime finish lacks an active start")
            active_runtime.pop()
    open_sequences = {row["sequence"] for row in unmatched_starts}
    if (
        open_sequences != {row["sequence"] for row in active_runtime}
        or runtime[-1]["sequence"] >= command_finish["sequence"]
        or runtime[-1]["action"] != command_finish["action"]
        or not command_start["sequence"] < runtime[-1]["sequence"]
    ):
        raise OutcomeProfileError("unmatched runtime start is outside its command")
    if native_outcome in {"cancelled", "timed-out"}:
        job_id = runtime[-1]["subject_id"]
        if not isinstance(job_id, str):
            raise OutcomeProfileError("runtime cancellation lacks a job identity")
        _validate_exact_cancellation_chain(
            records,
            action=str(command_finish["action"]),
            phase=str(command_finish["phase"]),
            job_id=job_id,
            native_outcome=native_outcome,
            reason_code=reason_code,
            after_sequence=runtime[-1]["sequence"],
            before_sequence=command_finish["sequence"],
        )
    elif _safety_rows(records):
        raise OutcomeProfileError(
            "non-cancellation runtime prefix has safety milestones"
        )


__all__ = [
    "CANCELLATION_EVENT_ORDER",
    "DEADLINE_REASON_CODE",
    "MANAGED_ACTION_ORDER",
    "ManagedOutcomeProfile",
    "OutcomeProfileError",
    "RUNTIME_BOUNDARY_ORDER",
    "RUNTIME_EVENT_TYPES",
    "is_publication_eligible",
    "validate_managed_sequence_outcome",
    "validate_unmatched_runtime_terminal_prefix",
]
