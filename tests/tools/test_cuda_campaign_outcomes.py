from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any

from tools.cuda_campaign.contracts import (
    SCHEMA_VERSIONS,
    ContractError,
    validate_event_ledger,
)
from tools.cuda_campaign.outcomes import (
    MANAGED_ACTION_ORDER,
    OutcomeProfileError,
    RUNTIME_BOUNDARY_ORDER,
    is_publication_eligible,
    validate_managed_sequence_outcome,
)


XRUN_ID = "xrun_" + "a" * 32
WALL_TIME = "2026-08-08T12:00:00+00:00"


def _job_id(index: int) -> str:
    return "job_" + f"{index + 1:x}" * 32


def _row(event_type: str, **values: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSIONS["event_ledger_row"],
        "sequence": 0,
        "experiment_run_id": XRUN_ID,
        "monotonic_ns": 0,
        "wall_time_utc": WALL_TIME,
        "event_type": event_type,
        "phase": None,
        "action": None,
        "subject_kind": None,
        "subject_id": None,
        "observation_kind": "observed",
        "source_reported_at_utc": None,
        "exit_code": None,
        "native_outcome": None,
        "reason_code": "NONE",
        **values,
    }


def _runtime_row(
    event_type: str,
    phase: str,
    action: str,
    job_id: str,
    *,
    native_outcome: str | None = None,
    reason_code: str = "NONE",
) -> dict[str, Any]:
    return _row(
        event_type,
        phase=phase,
        action=action,
        subject_kind="aptus-job",
        subject_id=job_id,
        observation_kind="emitted",
        native_outcome=native_outcome,
        reason_code=reason_code,
    )


def _runtime_prefix(
    count: int,
    *,
    terminal_outcome: str | None = None,
    terminal_reason: str = "NONE",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (event_type, phase, action) in enumerate(RUNTIME_BOUNDARY_ORDER[:count]):
        outcome = None
        reason = "NONE"
        if event_type.endswith("finished"):
            outcome = "passed"
        if index == count - 1 and terminal_outcome is not None:
            outcome = terminal_outcome
            reason = terminal_reason
        rows.append(
            _runtime_row(
                event_type,
                phase,
                action,
                _job_id(3 if action == "pilot" else 4),
                native_outcome=outcome,
                reason_code=reason,
            )
        )
    return rows


def _command_start(action: str) -> dict[str, Any]:
    return _row(
        "command.started",
        phase=action,
        action=action,
        subject_kind="managed-action",
        subject_id=action,
    )


def _command_finish(
    action: str,
    outcome: str,
    reason: str,
) -> dict[str, Any]:
    return _row(
        "command.finished",
        phase=action,
        action=action,
        subject_kind="managed-action",
        subject_id=action,
        exit_code=(0 if outcome == "passed" else None),
        native_outcome=outcome,
        reason_code=reason,
    )


def _job_observation(
    action: str,
    index: int,
    outcome: str | None,
    reason: str = "NONE",
) -> dict[str, Any]:
    return _row(
        "job.state-observed",
        phase=action,
        action=action,
        subject_kind="aptus-job",
        subject_id=_job_id(index),
        native_outcome=outcome,
        reason_code=reason,
    )


def _finish_ledger(
    body: list[dict[str, Any]],
    *,
    outcome: str,
    reason: str,
) -> list[dict[str, Any]]:
    rows = [
        _row("clock.mapping"),
        _row(
            "harness.started",
            subject_kind="experiment-run",
            subject_id=XRUN_ID,
        ),
        *body,
        _row(
            "harness.finished",
            subject_kind="experiment-run",
            subject_id=XRUN_ID,
            native_outcome=outcome,
            reason_code=reason,
        ),
        _row("clock.mapping"),
        _row(
            "seal.started",
            subject_kind="experiment-run",
            subject_id=XRUN_ID,
            native_outcome=outcome,
            reason_code=reason,
        ),
    ]
    for sequence, row in enumerate(rows):
        row["sequence"] = sequence
        row["monotonic_ns"] = sequence + 1
    return rows


def _action_summary(
    action: str,
    index: int,
    *,
    outcome: str,
    reason: str,
    jobless: bool = False,
    evidence_status: str = "protocol-valid",
) -> dict[str, Any]:
    return {
        "label": action,
        "action": action,
        "job_id": None if jobless else _job_id(index),
        "native_outcome": outcome,
        "reason_code": reason,
        "terminal": outcome != "unknown",
        "capture_reason_code": (
            "NONE"
            if evidence_status == "protocol-valid"
            else "MISSING_REQUIRED_EVIDENCE"
        ),
    }


def _summary(
    started: list[dict[str, Any]],
    *,
    outcome: str,
    reason: str,
    evidence_status: str = "protocol-valid",
) -> dict[str, Any]:
    return {
        "record_kind": "aptus-cuda-campaign-managed-sequence-v1",
        "configured_actions": [
            {"label": action, "action": action} for action in MANAGED_ACTION_ORDER
        ],
        "started_actions": started,
        "native_outcome": outcome,
        "reason_code": reason,
        "evidence_status": evidence_status,
        "capture_reason_code": (
            "NONE"
            if evidence_status == "protocol-valid"
            else "MISSING_REQUIRED_EVIDENCE"
        ),
        "stopped_early": outcome != "passed",
    }


def _pass_action(action: str, index: int) -> list[dict[str, Any]]:
    rows = [_command_start(action)]
    if action == "pilot":
        rows.extend(_runtime_prefix(4))
    elif action == "train":
        rows.extend(_runtime_prefix(10)[4:])
    rows.extend(
        (
            _job_observation(action, index, "passed"),
            _command_finish(action, "passed", "NONE"),
        )
    )
    return rows


def _passing_case(
    *, evidence_status: str = "protocol-valid"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    body: list[dict[str, Any]] = []
    started: list[dict[str, Any]] = []
    for index, action in enumerate(MANAGED_ACTION_ORDER):
        body.extend(_pass_action(action, index))
        started.append(
            _action_summary(
                action,
                index,
                outcome="passed",
                reason="NONE",
                evidence_status=evidence_status,
            )
        )
    return (
        _summary(
            started,
            outcome="passed",
            reason="NONE",
            evidence_status=evidence_status,
        ),
        _finish_ledger(body, outcome="passed", reason="NONE"),
    )


def _negative_case(
    *,
    stop_action: str,
    outcome: str,
    reason: str,
    runtime_count: int = 0,
    evidence_status: str = "protocol-valid",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stop_index = MANAGED_ACTION_ORDER.index(stop_action)
    body: list[dict[str, Any]] = []
    started: list[dict[str, Any]] = []
    for index, action in enumerate(MANAGED_ACTION_ORDER[:stop_index]):
        body.extend(_pass_action(action, index))
        started.append(_action_summary(action, index, outcome="passed", reason="NONE"))

    body.append(_command_start(stop_action))
    jobless = outcome in {"refused", "guard-blocked"} or outcome == "unknown"
    if outcome == "guard-blocked":
        body.append(
            _row(
                "safety.triggered",
                phase=stop_action,
                action=stop_action,
                subject_kind="managed-action",
                subject_id=stop_action,
                reason_code=reason,
            )
        )
    if not jobless:
        body.append(_job_observation(stop_action, stop_index, None))
        prefix = _runtime_prefix(runtime_count)
        body.extend(prefix[4:] if stop_action == "train" else prefix)
        if outcome in {"cancelled", "timed-out"}:
            for chain_index, event_type in enumerate(
                (
                    "safety.triggered",
                    "cancellation.requested",
                    "process-group.terminated",
                    "lease.reconciled",
                )
            ):
                body.append(
                    _row(
                        event_type,
                        phase=stop_action,
                        action=stop_action,
                        subject_kind="aptus-job",
                        subject_id=_job_id(stop_index),
                        native_outcome=(None if chain_index == 0 else "cancelled"),
                        reason_code=reason,
                    )
                )
        elif outcome == "failed":
            body.append(_job_observation(stop_action, stop_index, "failed", reason))
    body.append(_command_finish(stop_action, outcome, reason))
    started.append(
        _action_summary(
            stop_action,
            stop_index,
            outcome=outcome,
            reason=reason,
            jobless=jobless,
            evidence_status=evidence_status,
        )
    )
    return (
        _summary(
            started,
            outcome=outcome,
            reason=reason,
            evidence_status=evidence_status,
        ),
        _finish_ledger(body, outcome=outcome, reason=reason),
    )


def _resequence(rows: list[dict[str, Any]]) -> None:
    for sequence, row in enumerate(rows):
        row["sequence"] = sequence
        row["monotonic_ns"] = sequence + 1


class ManagedOutcomeProfileTests(unittest.TestCase):
    def test_pass_requires_all_five_actions_and_exact_ten_runtime_boundaries(
        self,
    ) -> None:
        summary, ledger = _passing_case()
        profile = validate_managed_sequence_outcome(summary, ledger)
        self.assertEqual(profile.started_action_count, 5)
        self.assertEqual(profile.runtime_boundary_count, 10)
        self.assertEqual(profile.stopping_action, "train")
        self.assertTrue(profile.publication_eligible)

        missing = deepcopy(ledger)
        missing.pop(
            next(
                index
                for index, row in enumerate(missing)
                if row["event_type"] == "verification.finished"
            )
        )
        _resequence(missing)
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(summary, missing)

    def test_conditioning_pass_ends_after_the_exact_four_action_pilot(self) -> None:
        body: list[dict[str, Any]] = []
        started: list[dict[str, Any]] = []
        for index, action in enumerate(MANAGED_ACTION_ORDER[:-1]):
            body.extend(_pass_action(action, index))
            started.append(
                _action_summary(action, index, outcome="passed", reason="NONE")
            )
        summary = _summary(started, outcome="passed", reason="NONE")
        summary["configured_actions"] = summary["configured_actions"][:-1]
        ledger = _finish_ledger(body, outcome="passed", reason="NONE")

        profile = validate_managed_sequence_outcome(summary, ledger)

        self.assertEqual(profile.sequence_profile, "conditioning")
        self.assertEqual(profile.started_action_count, 4)
        self.assertEqual(profile.runtime_boundary_count, 4)
        self.assertEqual(profile.stopping_action, "pilot")
        self.assertTrue(profile.publication_eligible)

        summary["configured_actions"] = summary["configured_actions"][:-1]
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(summary, ledger)

    def test_native_and_evidence_axes_are_independent_but_publication_is_not(
        self,
    ) -> None:
        guarded_summary, guarded_ledger = _negative_case(
            stop_action="preflight",
            outcome="guard-blocked",
            reason="UNRELATED_GPU_ACTIVITY",
        )
        guarded = validate_managed_sequence_outcome(guarded_summary, guarded_ledger)
        self.assertEqual(guarded.evidence_status, "protocol-valid")
        self.assertFalse(guarded.publication_eligible)

        pass_summary, pass_ledger = _passing_case(evidence_status="capture-invalid")
        passed = validate_managed_sequence_outcome(pass_summary, pass_ledger)
        self.assertEqual(passed.native_outcome, "passed")
        self.assertFalse(passed.publication_eligible)
        self.assertTrue(is_publication_eligible("passed", "protocol-valid"))
        self.assertFalse(is_publication_eligible("failed", "protocol-valid"))
        with self.assertRaises(OutcomeProfileError):
            is_publication_eligible("passed", "not-started")

    def test_evidence_status_requires_exact_capture_reason_consistency(self) -> None:
        summary, ledger = _passing_case()

        dirty_action = deepcopy(summary)
        dirty_action["started_actions"][2]["capture_reason_code"] = (
            "MISSING_REQUIRED_EVIDENCE"
        )
        with self.assertRaisesRegex(OutcomeProfileError, "capture reason"):
            validate_managed_sequence_outcome(dirty_action, ledger)

        dirty_summary = deepcopy(summary)
        dirty_summary["capture_reason_code"] = "STREAM_CAPTURE_FAILURE"
        with self.assertRaisesRegex(OutcomeProfileError, "capture reason"):
            validate_managed_sequence_outcome(dirty_summary, ledger)

        invalid_capture = deepcopy(summary)
        invalid_capture["evidence_status"] = "capture-invalid"
        with self.assertRaisesRegex(OutcomeProfileError, "capture reason"):
            validate_managed_sequence_outcome(invalid_capture, ledger)

        summary_only_reason = deepcopy(summary)
        summary_only_reason["evidence_status"] = "capture-invalid"
        summary_only_reason["capture_reason_code"] = "STREAM_CAPTURE_FAILURE"
        profile = validate_managed_sequence_outcome(summary_only_reason, ledger)
        self.assertEqual(profile.evidence_status, "capture-invalid")

        malformed = deepcopy(summary_only_reason)
        malformed["capture_reason_code"] = "NOT_A_REASON"
        with self.assertRaisesRegex(OutcomeProfileError, "capture reason"):
            validate_managed_sequence_outcome(malformed, ledger)

    def test_capture_invalid_never_authorizes_a_contradictory_terminal_job(
        self,
    ) -> None:
        summary, ledger = _negative_case(
            stop_action="pilot",
            outcome="failed",
            reason="PROCESS_EXIT_NONZERO",
            runtime_count=1,
            evidence_status="capture-invalid",
        )
        terminal = next(
            row
            for row in ledger
            if row["event_type"] == "job.state-observed"
            and row["action"] == "pilot"
            and row["native_outcome"] is not None
        )
        terminal["native_outcome"] = "passed"
        terminal["reason_code"] = "NONE"
        with self.assertRaisesRegex(OutcomeProfileError, "contradicts action"):
            validate_managed_sequence_outcome(summary, ledger)

    def test_started_action_entries_reject_unknown_keys(self) -> None:
        summary, ledger = _passing_case()
        changed = deepcopy(summary)
        changed["started_actions"][0]["unexpected"] = True
        with self.assertRaisesRegex(OutcomeProfileError, "wrong exact"):
            validate_managed_sequence_outcome(changed, ledger)

        configured_extra = deepcopy(summary)
        configured_extra["configured_actions"][0]["submit_kwargs"] = {}
        validate_managed_sequence_outcome(configured_extra, ledger)

    def test_all_six_nonpass_profiles_validate_as_exact_stopping_prefixes(self) -> None:
        cases = (
            ("preflight", "refused", "APTUS_ADMISSION_REFUSAL", 0),
            ("pilot", "failed", "PROCESS_EXIT_NONZERO", 1),
            ("train", "cancelled", "CUDA_XID", 6),
            ("train", "timed-out", "EMERGENCY_DEADLINE_EXCEEDED", 5),
            ("preflight", "guard-blocked", "UNRELATED_GPU_ACTIVITY", 0),
            ("dependency", "unknown", "OWNERSHIP_UNCERTAIN", 0),
        )
        for stop_action, outcome, reason, runtime_count in cases:
            with self.subTest(outcome=outcome):
                summary, ledger = _negative_case(
                    stop_action=stop_action,
                    outcome=outcome,
                    reason=reason,
                    runtime_count=runtime_count,
                )
                profile = validate_managed_sequence_outcome(summary, ledger)
                self.assertEqual(profile.native_outcome, outcome)
                self.assertEqual(profile.stopping_action, stop_action)
                self.assertFalse(profile.publication_eligible)

    def test_training_failure_before_export_closes_its_runtime_boundary(self) -> None:
        summary, ledger = _negative_case(
            stop_action="train",
            outcome="failed",
            reason="CUDA_OOM",
            runtime_count=5,
        )
        training_started = next(
            index
            for index, row in enumerate(ledger)
            if row["event_type"] == "training.started"
        )
        ledger.insert(
            training_started + 1,
            _runtime_row(
                "training.finished",
                "training",
                "train",
                _job_id(4),
                native_outcome="failed",
                reason_code="CUDA_OOM",
            ),
        )
        _resequence(ledger)
        profile = validate_managed_sequence_outcome(summary, ledger)
        self.assertEqual(profile.native_outcome, "failed")
        self.assertEqual(profile.runtime_boundary_count, 6)

        impossible_pass = deepcopy(ledger)
        training_finished = next(
            row for row in impossible_pass if row["event_type"] == "training.finished"
        )
        training_finished["native_outcome"] = "passed"
        training_finished["reason_code"] = "NONE"
        with self.assertRaisesRegex(OutcomeProfileError, "before the final export"):
            validate_managed_sequence_outcome(summary, impossible_pass)

    def test_export_failure_closes_both_nested_runtime_boundaries(self) -> None:
        summary, ledger = _negative_case(
            stop_action="train",
            outcome="failed",
            reason="EXPORT_VERIFICATION_FAILURE",
            runtime_count=7,
        )
        export_finished = next(
            index
            for index, row in enumerate(ledger)
            if row["event_type"] == "export.finished"
        )
        ledger[export_finished]["native_outcome"] = "failed"
        ledger[export_finished]["reason_code"] = "EXPORT_VERIFICATION_FAILURE"
        ledger.insert(
            export_finished + 1,
            _runtime_row(
                "training.finished",
                "training",
                "train",
                _job_id(4),
                native_outcome="failed",
                reason_code="EXPORT_VERIFICATION_FAILURE",
            ),
        )
        _resequence(ledger)
        profile = validate_managed_sequence_outcome(summary, ledger)
        self.assertEqual(profile.runtime_boundary_count, 8)

        contradictory = deepcopy(ledger)
        training_finished = next(
            row for row in contradictory if row["event_type"] == "training.finished"
        )
        training_finished["reason_code"] = "PROCESS_EXIT_NONZERO"
        with self.assertRaisesRegex(OutcomeProfileError, "contradicts"):
            validate_managed_sequence_outcome(summary, contradictory)

        missing_outer = [
            deepcopy(row) for row in ledger if row["event_type"] != "training.finished"
        ]
        _resequence(missing_outer)
        with self.assertRaisesRegex(OutcomeProfileError, "enclosing training finish"):
            validate_managed_sequence_outcome(summary, missing_outer)

        capture_invalid = deepcopy(summary)
        capture_invalid["evidence_status"] = "capture-invalid"
        capture_invalid["capture_reason_code"] = "MISSING_REQUIRED_EVIDENCE"
        capture_invalid["started_actions"][-1]["capture_reason_code"] = (
            "MISSING_REQUIRED_EVIDENCE"
        )
        profile = validate_managed_sequence_outcome(capture_invalid, missing_outer)
        self.assertEqual(profile.evidence_status, "capture-invalid")

        continued = deepcopy(ledger)
        command_finish = next(
            index
            for index, row in enumerate(continued)
            if row["event_type"] == "command.finished" and row["action"] == "train"
        )
        continued.insert(
            command_finish,
            _runtime_row(
                "verification.started",
                "parent-verification",
                "train",
                _job_id(4),
            ),
        )
        _resequence(continued)
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(summary, continued)

    def test_unmatched_runtime_start_is_allowed_only_for_exact_terminal_nonpass(
        self,
    ) -> None:
        summary, ledger = _negative_case(
            stop_action="pilot",
            outcome="failed",
            reason="PROCESS_EXIT_NONZERO",
            runtime_count=1,
        )
        self.assertEqual(validate_event_ledger(ledger), ledger)
        validate_managed_sequence_outcome(summary, ledger)

        passing_terminal = deepcopy(ledger)
        for row in passing_terminal:
            if row["event_type"] in {
                "command.finished",
                "harness.finished",
                "seal.started",
            }:
                row["native_outcome"] = "passed"
                row["reason_code"] = "NONE"
                if row["event_type"] == "command.finished":
                    row["exit_code"] = 0
        with self.assertRaisesRegex(ContractError, "unauthorized incomplete"):
            validate_event_ledger(passing_terminal)

        skipped_prefix = [
            deepcopy(row)
            for row in ledger
            if row["action"] not in {"dependency", "model-data", "preflight"}
        ]
        _resequence(skipped_prefix)
        with self.assertRaisesRegex(ContractError, "unauthorized incomplete"):
            validate_event_ledger(skipped_prefix)

    def test_refusal_and_pre_submit_guard_are_jobless_exact_profiles(self) -> None:
        for outcome, reason in (
            ("refused", "APTUS_ADMISSION_REFUSAL"),
            ("guard-blocked", "HOST_RAM_FLOOR"),
        ):
            summary, ledger = _negative_case(
                stop_action="dependency",
                outcome=outcome,
                reason=reason,
            )
            validate_managed_sequence_outcome(summary, ledger)
            injected = deepcopy(ledger)
            command_finish = next(
                index
                for index, row in enumerate(injected)
                if row["event_type"] == "command.finished"
            )
            injected.insert(
                command_finish,
                _job_observation("dependency", 0, None),
            )
            _resequence(injected)
            with self.assertRaises(OutcomeProfileError):
                validate_managed_sequence_outcome(summary, injected)

    def test_reorder_insertion_and_later_runtime_are_rejected(self) -> None:
        summary, ledger = _passing_case()
        reordered = deepcopy(summary)
        reordered["started_actions"][1], reordered["started_actions"][2] = (
            reordered["started_actions"][2],
            reordered["started_actions"][1],
        )
        with self.assertRaisesRegex(OutcomeProfileError, "reordered|inserted"):
            validate_managed_sequence_outcome(reordered, ledger)

        inserted = deepcopy(ledger)
        runtime_index = next(
            index
            for index, row in enumerate(inserted)
            if row["event_type"] == "training.started"
        )
        inserted.insert(runtime_index + 1, deepcopy(inserted[runtime_index]))
        _resequence(inserted)
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(summary, inserted)

        refused, stopped = _negative_case(
            stop_action="pilot",
            outcome="refused",
            reason="APTUS_ADMISSION_REFUSAL",
        )
        finish_index = next(
            index
            for index, row in enumerate(stopped)
            if row["event_type"] == "command.finished" and row["action"] == "pilot"
        )
        stopped.insert(
            finish_index,
            _runtime_row(
                "pilot.phase-started",
                "pilot-phase-1",
                "pilot",
                _job_id(3),
            ),
        )
        stopped.insert(
            finish_index + 1,
            _runtime_row(
                "pilot.phase-finished",
                "pilot-phase-1",
                "pilot",
                _job_id(3),
                native_outcome="passed",
            ),
        )
        _resequence(stopped)
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(refused, stopped)

        failed, later_action = _negative_case(
            stop_action="pilot",
            outcome="failed",
            reason="PROCESS_EXIT_NONZERO",
            runtime_count=1,
        )
        harness_index = next(
            index
            for index, row in enumerate(later_action)
            if row["event_type"] == "harness.finished"
        )
        later_action[harness_index:harness_index] = [
            _command_start("train"),
            _command_finish("train", "failed", "PROCESS_EXIT_NONZERO"),
        ]
        _resequence(later_action)
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(failed, later_action)

    def test_cancellation_chain_rejects_reorder_reason_and_deadline_mismatch(
        self,
    ) -> None:
        summary, ledger = _negative_case(
            stop_action="train",
            outcome="cancelled",
            reason="CUDA_XID",
            runtime_count=6,
        )
        validate_managed_sequence_outcome(summary, ledger)
        chain_indices = [
            index
            for index, row in enumerate(ledger)
            if row["event_type"]
            in {
                "safety.triggered",
                "cancellation.requested",
                "process-group.terminated",
                "lease.reconciled",
            }
        ]

        reordered = deepcopy(ledger)
        first, second = chain_indices[1:3]
        reordered[first], reordered[second] = reordered[second], reordered[first]
        _resequence(reordered)
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(summary, reordered)

        wrong_reason = deepcopy(ledger)
        wrong_reason[chain_indices[-1]]["reason_code"] = "CUDA_OOM"
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(summary, wrong_reason)

        wrong_outcome = deepcopy(ledger)
        wrong_outcome[chain_indices[-1]]["native_outcome"] = "failed"
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(summary, wrong_outcome)

        wrong_mapping = deepcopy(summary)
        wrong_mapping["native_outcome"] = "timed-out"
        wrong_mapping["started_actions"][-1]["native_outcome"] = "timed-out"
        with self.assertRaisesRegex(OutcomeProfileError, "deadline"):
            validate_managed_sequence_outcome(wrong_mapping, ledger)

    def test_cancellation_chain_follows_last_emitted_boundary_without_fake_finish(
        self,
    ) -> None:
        summary, ledger = _negative_case(
            stop_action="train",
            outcome="timed-out",
            reason="EMERGENCY_DEADLINE_EXCEEDED",
            runtime_count=6,
        )
        runtime = [
            row
            for row in ledger
            if row["event_type"] in {item[0] for item in RUNTIME_BOUNDARY_ORDER}
        ]
        self.assertEqual(runtime[-1]["event_type"], "export.started")
        self.assertNotIn("export.finished", [row["event_type"] for row in runtime])
        trigger = next(row for row in ledger if row["event_type"] == "safety.triggered")
        self.assertGreater(trigger["sequence"], runtime[-1]["sequence"])
        validate_managed_sequence_outcome(summary, ledger)

        trigger_before_runtime = deepcopy(ledger)
        trigger_index = next(
            index
            for index, row in enumerate(trigger_before_runtime)
            if row["event_type"] == "safety.triggered"
        )
        trigger_row = trigger_before_runtime.pop(trigger_index)
        runtime_index = next(
            index
            for index, row in enumerate(trigger_before_runtime)
            if row["event_type"] == "export.started"
        )
        trigger_before_runtime.insert(runtime_index, trigger_row)
        _resequence(trigger_before_runtime)
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(summary, trigger_before_runtime)

        missing = [
            deepcopy(row)
            for row in ledger
            if row["event_type"] != "process-group.terminated"
        ]
        _resequence(missing)
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(summary, missing)

    def test_summary_and_terminal_envelope_must_agree(self) -> None:
        summary, ledger = _negative_case(
            stop_action="preflight",
            outcome="refused",
            reason="APTUS_ADMISSION_REFUSAL",
        )
        changed = deepcopy(summary)
        changed["reason_code"] = "POLICY_REPLAN_REQUIRED"
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(changed, ledger)

        changed_ledger = deepcopy(ledger)
        changed_ledger[-1]["native_outcome"] = "unknown"
        changed_ledger[-1]["reason_code"] = "UNKNOWN_TERMINAL_STATE"
        with self.assertRaises(OutcomeProfileError):
            validate_managed_sequence_outcome(summary, changed_ledger)


if __name__ == "__main__":
    unittest.main()
