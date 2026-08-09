from __future__ import annotations

import copy
import json
import multiprocessing
import os
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aptus.generation import (
    create_bundle_archive,
    generate_bundle,
    validate_bundle_archive_bytes,
)
from tools.cuda_campaign import admission
from tools.cuda_campaign.admission import (
    ACTIVATION_FILE_NAMES,
    ACTIVATION_SEAL_NAME,
    ExecutionProposal,
    FrozenResourceBudget,
    InjectedAdmissionAuthority,
    PlannedSlotContext,
    RunProposal,
    authority_snapshot,
    construct_admission_observation,
    evaluate_pre_slot_admission,
)
from tools.cuda_campaign import storage
from tools.cuda_campaign.contracts import (
    EventLedgerWriter,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    compact_canonical_json_bytes,
    deterministic_id,
    sha256_bytes,
)
from tools.cuda_campaign.monitoring import (
    GIB,
    construct_telemetry_sample,
    summarize_telemetry,
    validate_cooldown,
)
from tools.cuda_campaign.outcomes import is_publication_eligible
from tools.cuda_campaign.qualification import (
    REQUIRED_QUALIFYING_ARTIFACT_ROLES,
    build_segment_summaries,
)
from tools.cuda_campaign.storage import (
    AppendOnlyReceiptStore,
    ArtifactIntegrityError,
    EvidenceStorageError,
    RawArtifactWriter,
    ReceiptChainError,
    RetrievalError,
    add_calendar_months_utc,
    copy_sealed_artifact,
    evaluate_retention_state,
    retention_deadline_utc,
    retrieve_sealed_artifact,
    verify_copy_equality,
    verify_sealed_artifact,
    write_capture_failure_receipt,
    write_sealed_capture_failure_artifact,
)
from tests.tools.test_cuda_campaign_phase4 import (
    baseline_path,
    host_observation,
    phase4_configuration,
    phase4_records,
)
from tests.aptus.helpers import make_plan
from tools.cuda_campaign.phase4 import (
    PHASE4_IDLE_SAMPLES_NAME,
    PHASE4_SOURCE_FREEZE_NAME,
    PHASE4_SOURCE_FREEZE_SEAL_NAME,
    _create_phase4_source_freeze_for_test,
    _test_phase4_boundary,
    _validate_retained_phase4_source_freeze_for_test,
)


ARTIFACT_ID = "artifact_" + "a" * 32
PROTOCOL_CAMPAIGN, PROTOCOL_COHORT, PROTOCOL_CELL = phase4_records()


def _canonical_test_bundle() -> tuple[dict, dict, bytes]:
    """Compile one deterministic real static bundle for authority fixtures."""

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle = root / "bundle"
        plan = make_plan(root)
        provenance = plan.dataset.provenance
        if provenance is None:  # pragma: no cover - the test helper is measured.
            raise RuntimeError("Storage bundle fixture lacks dataset provenance.")
        plan = replace(
            plan,
            dataset=replace(
                plan.dataset,
                provenance=replace(
                    provenance,
                    observed_at="2026-08-08T12:00:00+00:00",
                ),
            ),
        )
        generate_bundle(plan, bundle)
        archive = create_bundle_archive(bundle, root / "bundle.zip")
        return (
            json.loads((bundle / "plan.json").read_text(encoding="utf-8")),
            json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8")),
            archive.read_bytes(),
        )


_PLAN_PAYLOAD, _BUNDLE_MANIFEST_PAYLOAD, _BUNDLE_ARCHIVE_BYTES = (
    _canonical_test_bundle()
)
_PLAN_BYTES = canonical_json_bytes(_PLAN_PAYLOAD)
_BUNDLE_MANIFEST_BYTES = canonical_json_bytes(_BUNDLE_MANIFEST_PAYLOAD)
_BUNDLE_FINGERPRINT = sha256_bytes(_BUNDLE_MANIFEST_BYTES)
_SLOT_IDENTITY = {
    "schema_version": "aptus.experiment-attempt-slot.v1",
    "comparison_cohort_id": PROTOCOL_COHORT["comparison_cohort_id"],
    "comparison_cell_id": PROTOCOL_CELL["comparison_cell_id"],
    "block": 0,
    "ordinal": 1,
    "role": "anchor",
    "order_position": 0,
    "scheduled_seed": 17,
}
SLOT_ID = deterministic_id("slot_", _SLOT_IDENTITY)
RUN_ID = "xrun_" + "c" * 32
OTHER_RUN_ID = "xrun_" + "d" * 32
JOB_ID = "job_" + "1" * 32
QUALIFYING_ACTIONS = ("dependency", "model-data", "preflight", "pilot", "train")
QUALIFYING_JOB_IDS = (JOB_ID,) + tuple(
    "job_" + f"{index:032x}" for index in range(2, len(QUALIFYING_ACTIONS) + 1)
)
ENTRY_ID = "entry_" + "d" * 32
COPY_ID = "copy_" + "e" * 32
DOMAIN_ID = "domain_" + "f" * 32
RECEIPT_ONE = "receipt_" + "1" * 32
RECEIPT_TWO = "receipt_" + "2" * 32
RETAIN_UNTIL = "2028-08-08T12:00:00+00:00"


def private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def writer_at(root: Path, name: str = "artifact") -> RawArtifactWriter:
    return RawArtifactWriter(
        root / name,
        protected_artifact_id=ARTIFACT_ID,
        record_kind="legacy-recovery",
        identity_bindings={"recovery_fixture": "storage-test"},
        capture_tool={"name": "storage-test", "version": "v1"},
        source_bindings={"source_commit": "1" * 40},
        provisional_retain_not_before_utc=RETAIN_UNTIL,
        required_role_bindings={"job-log": ENTRY_ID},
    )


class _ProtocolFixtureWriter(RawArtifactWriter):
    """Keep visibly nonproduction Phase-4 bytes inside the test process."""

    def seal(self) -> dict:
        with patch.object(
            storage,
            "validate_retained_phase4_source_freeze",
            _validate_retained_phase4_source_freeze_for_test,
        ):
            return super().seal()


def sealed_artifact(root: Path, name: str = "artifact") -> tuple[Path, dict]:
    writer = writer_at(root, name)
    writer.write_payload(
        b"complete job output\n",
        "jobs/job.log",
        role="job-log",
        media_type="text/plain",
        entry_id=ENTRY_ID,
        captured_at_utc="2026-08-08T12:00:00+00:00",
    )
    result = writer.seal()
    return writer.directory, result


def command_writer(
    root: Path,
    name: str,
    *,
    run_id: str = RUN_ID,
    artifact_id: str = ARTIFACT_ID,
) -> RawArtifactWriter:
    entry_ids = {
        "command-record": f"entry_{name}_command",
        "command-output": f"entry_{name}_output",
        "event-ledger": f"entry_{name}_ledger",
    }
    writer = RawArtifactWriter(
        root / name,
        protected_artifact_id=artifact_id,
        record_kind="experiment-run",
        identity_bindings={
            "attempt_slot_id": SLOT_ID,
            "experiment_run_id": run_id,
            "capture_kind": "command",
            "capture_status": "complete",
        },
        capture_tool={"name": "storage-test", "version": "v1"},
        source_bindings={"source_commit": "1" * 40},
        provisional_retain_not_before_utc=RETAIN_UNTIL,
        required_role_bindings=entry_ids,
    )
    for role, relative_path, payload in (
        ("command-record", "command/record.json", b"{}\n"),
        ("command-output", "command/output.bin", b"output\n"),
        ("event-ledger", "events/events.jsonl", b"{}\n"),
    ):
        writer.write_payload(
            payload,
            relative_path,
            role=role,
            entry_id=entry_ids[role],
        )
    return writer


def _seal_with_process_gate(
    vault_text: str,
    name: str,
    artifact_id: str,
    started: object,
    entered_duplicate_check: object,
    release_duplicate_check: object,
    results: object,
) -> None:
    """Seal in a child process while exposing the duplicate-check boundary."""

    writer = command_writer(Path(vault_text), name, artifact_id=artifact_id)
    original = storage._assert_experiment_run_is_unsealed_elsewhere

    def gated_duplicate_check(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        entered_duplicate_check.set()
        if not release_duplicate_check.wait(timeout=10):
            raise RuntimeError("process seal gate timed out")

    started.set()
    try:
        with patch.object(
            storage,
            "_assert_experiment_run_is_unsealed_elsewhere",
            side_effect=gated_duplicate_check,
        ):
            writer.seal()
    except BaseException as error:
        results.put(("error", type(error).__name__))
    else:
        results.put(("sealed", name))


_DEFAULT = object()
_WALL_TIME = "2026-08-08T12:00:00+00:00"


def protocol_event_records(
    *,
    experiment_run_id: str = RUN_ID,
) -> list[dict]:
    ledger = EventLedgerWriter(experiment_run_id)

    def append(monotonic_ns: int, event_type: str, **values: object) -> None:
        ledger.append(
            monotonic_ns=monotonic_ns,
            wall_time_utc=_WALL_TIME,
            event_type=event_type,
            **values,
        )

    append(
        0, "clock.mapping", subject_kind="experiment-run", subject_id=experiment_run_id
    )
    append(
        0,
        "harness.started",
        subject_kind="experiment-run",
        subject_id=experiment_run_id,
    )
    append(
        1_000_000_000,
        "telemetry.started",
        subject_kind="experiment-run",
        subject_id=experiment_run_id,
    )
    runtime_by_action = {
        "pilot": (
            ("pilot.phase-started", "pilot-phase-1"),
            ("pilot.phase-finished", "pilot-phase-1"),
            ("pilot.phase-started", "pilot-phase-2"),
            ("pilot.phase-finished", "pilot-phase-2"),
        ),
        "train": (
            ("training.started", "training"),
            ("export.started", "final-export"),
            ("export.finished", "final-export"),
            ("training.finished", "training"),
            ("verification.started", "parent-verification"),
            ("verification.finished", "parent-verification"),
        ),
    }
    for action, job_id in zip(QUALIFYING_ACTIONS, QUALIFYING_JOB_IDS):
        append(
            1_000_000_000,
            "command.started",
            phase=action,
            action=action,
            subject_kind="managed-action",
            subject_id=action,
        )
        for event_type, phase in runtime_by_action.get(action, ()):
            append(
                1_000_000_000,
                event_type,
                phase=phase,
                action=action,
                subject_kind="aptus-job",
                subject_id=job_id,
                observation_kind="emitted",
                native_outcome=(None if event_type.endswith("started") else "passed"),
            )
        append(
            1_000_000_000,
            "job.state-observed",
            phase=action,
            action=action,
            subject_kind="aptus-job",
            subject_id=job_id,
            native_outcome="passed",
        )
        append(
            1_000_000_000,
            "command.finished",
            phase=action,
            action=action,
            subject_kind="managed-action",
            subject_id=action,
            exit_code=0,
            native_outcome="passed",
        )
    append(
        2_000_000_000,
        "cooldown.started",
        phase="cooldown",
        subject_kind="experiment-run",
        subject_id=experiment_run_id,
    )
    append(
        121_000_000_000,
        "cooldown.finished",
        phase="cooldown",
        subject_kind="experiment-run",
        subject_id=experiment_run_id,
        native_outcome="passed",
    )
    append(
        121_000_000_000,
        "telemetry.stopped",
        subject_kind="experiment-run",
        subject_id=experiment_run_id,
    )
    append(
        122_000_000_000,
        "harness.finished",
        subject_kind="experiment-run",
        subject_id=experiment_run_id,
        exit_code=0,
        native_outcome="passed",
    )
    append(
        122_000_000_000,
        "clock.mapping",
        subject_kind="experiment-run",
        subject_id=experiment_run_id,
    )
    append(
        122_000_000_000,
        "seal.started",
        subject_kind="experiment-run",
        subject_id=experiment_run_id,
        exit_code=0,
        native_outcome="passed",
    )
    return list(ledger.records)


def protocol_runtime_journal_records(
    ledger: list[dict] | None = None,
) -> dict[str, list[dict]]:
    records = protocol_event_records() if ledger is None else ledger
    journals = {"pilot": [], "train": []}
    runtime_types = {
        "pilot.phase-started",
        "pilot.phase-finished",
        "training.started",
        "training.finished",
        "export.started",
        "export.finished",
        "verification.started",
        "verification.finished",
    }
    for row in records:
        if row["event_type"] not in runtime_types:
            continue
        action = row["action"]
        journals[action].append(
            {
                "schema_version": "aptus.cuda-campaign-runtime-boundary.v1",
                "experiment_run_id": row["experiment_run_id"],
                "job_id": row["subject_id"],
                "monotonic_ns": row["monotonic_ns"],
                "wall_time_utc": row["wall_time_utc"],
                "event_type": row["event_type"],
                "phase": row["phase"],
                "action": action,
                "native_outcome": row["native_outcome"],
                "reason_code": row["reason_code"],
            }
        )
    return journals


def _test_bundle_archive() -> bytes:
    return _BUNDLE_ARCHIVE_BYTES


def _artifact_payload_bytes(value: object) -> bytes:
    return value if isinstance(value, bytes) else canonical_json_bytes(value)


def protocol_required_artifact_payloads() -> dict[str, object]:
    payloads: dict[str, object] = {
        role: {"record_kind": role, "status": "passed"}
        for role in sorted(REQUIRED_QUALIFYING_ARTIFACT_ROLES)
    }
    payloads["plan"] = _PLAN_PAYLOAD
    payloads["bundle-manifest"] = _BUNDLE_MANIFEST_PAYLOAD
    payloads["bundle-archive"] = _test_bundle_archive()
    return payloads


def protocol_telemetry_records(*, experiment_run_id: str = RUN_ID) -> list[dict]:
    total = 8 * GIB
    free = 7 * GIB
    probe = {
        "gpu": {
            "uuid": "GPU-protected-storage-test",
            "memory_used": {"value": str(total - free), "unit": "B"},
            "memory_free": {"value": str(free), "unit": "B"},
            "memory_reserved": {"value": "0", "unit": "B"},
            "memory_total": {"value": str(total), "unit": "B"},
            "utilization_percent": 0.0,
            "temperature_c": 35.0,
            "power_draw_w": 20.0,
            "power_limit_w": 130.0,
            "graphics_clock_mhz": 210.0,
            "memory_clock_mhz": 405.0,
            "performance_state": "P8",
            "throttle_reasons": [],
            "throttle_state": "0x0000000000000000",
            "xid_errors": [],
            "reset_detected": False,
            "device_lost": False,
            "hardware_error": False,
            "compute_processes": [],
        },
        "host": {
            "mem_available_bytes": 48 * GIB,
            "swap_used_bytes": 0,
            "swap_read_bytes": 0,
            "swap_write_bytes": 0,
            "load_1m": 0.25,
            "filesystem_free_bytes": 200 * GIB,
            "managed_process_rss_bytes": 256 * 1024**2,
            "managed_process_cpu_seconds": 1.5,
            "managed_process_read_bytes": 1024,
            "managed_process_write_bytes": 2048,
            "disk_growth_bytes": 4096,
            "aptus_lease_active": False,
            "cpu_temperature": {
                "status": "unsupported",
                "value": None,
                "reason_code": "NOT_CONFIGURED",
            },
            "nvme_temperature": {
                "status": "unsupported",
                "value": None,
                "reason_code": "NOT_CONFIGURED",
            },
        },
    }
    return [
        construct_telemetry_sample(
            sequence=slot,
            experiment_run_id=experiment_run_id,
            scheduled_slot=slot,
            scheduled_monotonic_ns=1_000_000_000 + slot * 1_000_000_000,
            observed_monotonic_ns=1_000_000_000 + slot * 1_000_000_000,
            wall_time_utc=_WALL_TIME,
            probe_reading=probe,
            collector={
                "healthy": True,
                "status_code": None,
                "probe_duration_ns": 1000,
            },
            watchdog={
                "healthy": True,
                "heartbeat_monotonic_ns": (1_000_000_000 + slot * 1_000_000_000),
                "ownership_certain": True,
            },
        )
        for slot in range(121)
    ]


def protocol_sequence_summary() -> dict:
    return {
        "record_kind": "aptus-cuda-campaign-managed-sequence-v1",
        "experiment_run_id": RUN_ID,
        "attempt_slot_id": SLOT_ID,
        "configured_actions": [
            {
                "label": action,
                "action": action,
                "supervision_timeout_seconds": 5,
                "submit_kwargs": {},
            }
            for action in QUALIFYING_ACTIONS
        ],
        "started_actions": [
            {
                "label": action,
                "action": action,
                "job_id": job_id,
                "native_outcome": "passed",
                "reason_code": "NONE",
                "terminal": True,
                "capture_reason_code": "NONE",
            }
            for action, job_id in zip(QUALIFYING_ACTIONS, QUALIFYING_JOB_IDS)
        ],
        "native_outcome": "passed",
        "reason_code": "NONE",
        "evidence_status": "protocol-valid",
        "capture_reason_code": "NONE",
        "telemetry_required": True,
        "telemetry_test_override": False,
        "telemetry_healthy": True,
        "stopped_early": False,
        "five_action_duration_ns": 0,
    }


def protocol_negative_outcome_case(
    outcome: str,
    reason: str,
    *,
    stop_action: str = "preflight",
    runtime_count: int = 0,
) -> tuple[dict, list[dict], list[dict]]:
    from tests.tools.test_cuda_campaign_outcomes import _negative_case

    summary, ledger = _negative_case(
        stop_action=stop_action,
        outcome=outcome,
        reason=reason,
        runtime_count=runtime_count,
    )
    for row in ledger:
        row["experiment_run_id"] = RUN_ID
        if row.get("subject_kind") == "experiment-run":
            row["subject_id"] = RUN_ID
    telemetry_started = {
        "schema_version": "aptus.experiment-event.v1",
        "sequence": 0,
        "experiment_run_id": RUN_ID,
        "monotonic_ns": 0,
        "wall_time_utc": _WALL_TIME,
        "event_type": "telemetry.started",
        "phase": None,
        "action": None,
        "subject_kind": "experiment-run",
        "subject_id": RUN_ID,
        "observation_kind": "observed",
        "source_reported_at_utc": None,
        "exit_code": None,
        "native_outcome": None,
        "reason_code": "NONE",
    }
    telemetry_stopped = {
        **telemetry_started,
        "event_type": "telemetry.stopped",
        "native_outcome": "passed",
    }
    ledger.insert(2, telemetry_started)
    harness_finished_index = next(
        index
        for index, row in enumerate(ledger)
        if row["event_type"] == "harness.finished"
    )
    ledger.insert(harness_finished_index, telemetry_stopped)
    for sequence, row in enumerate(ledger):
        row["sequence"] = sequence
        row["monotonic_ns"] = sequence + 1
    telemetry_start_ns = next(
        row["monotonic_ns"]
        for row in ledger
        if row["event_type"] == "telemetry.started"
    )
    sample = protocol_telemetry_records()[0]
    sample.update(
        sequence=0,
        experiment_run_id=RUN_ID,
        scheduled_slot=0,
        scheduled_monotonic_ns=telemetry_start_ns,
        observed_monotonic_ns=telemetry_start_ns,
    )
    sample["watchdog"]["heartbeat_monotonic_ns"] = telemetry_start_ns
    command_boundaries = [
        row
        for row in ledger
        if row["event_type"] in {"command.started", "command.finished"}
    ]
    five_action_duration_ns = (
        command_boundaries[-1]["monotonic_ns"] - command_boundaries[0]["monotonic_ns"]
        if len(summary["started_actions"]) == len(QUALIFYING_ACTIONS)
        and len(command_boundaries) == 2 * len(QUALIFYING_ACTIONS)
        else None
    )
    summary.update(
        experiment_run_id=RUN_ID,
        attempt_slot_id=SLOT_ID,
        configured_actions=[
            {
                "label": action,
                "action": action,
                "supervision_timeout_seconds": 5,
                "submit_kwargs": {},
            }
            for action in QUALIFYING_ACTIONS
        ],
        telemetry_required=True,
        telemetry_test_override=False,
        telemetry_healthy=True,
        five_action_duration_ns=five_action_duration_ns,
    )
    return summary, ledger, [sample]


def protocol_attempt_slot_record() -> dict:
    return {
        "schema_version": "aptus.experiment-attempt-slot.v1",
        "attempt_slot_id": SLOT_ID,
        "comparison_cohort_id": PROTOCOL_COHORT["comparison_cohort_id"],
        "comparison_cell_id": PROTOCOL_CELL["comparison_cell_id"],
        "block": 0,
        "ordinal": 1,
        "role": "anchor",
        "order_position": 0,
        "scheduled_seed": 17,
        "slot_status": "started",
        "execution_configuration_id": protocol_execution_configuration_record()[
            "execution_configuration_id"
        ],
        "experiment_run_id": RUN_ID,
        "native_outcome": "passed",
        "evidence_status": "protocol-valid",
        "reason_code": "NONE",
    }


def protocol_execution_configuration_record() -> dict:
    proposal = protocol_planned_slot_context().execution_proposal
    identity = {
        "schema_version": "aptus.experiment-execution-configuration.v1",
        "comparison_cell_id": PROTOCOL_CELL["comparison_cell_id"],
        "exact_behavior_values": dict(proposal.exact_behavior_values),
        "split_seed": proposal.split_seed,
        "training_seed": proposal.training_seed,
        "data_order_seed": proposal.data_order_seed,
        "plan_id": proposal.plan_id,
        "candidate_id": proposal.candidate_id,
        "bundle_fingerprint": proposal.bundle_fingerprint,
    }
    return {
        **identity,
        "execution_configuration_id": deterministic_id("exec_", identity),
        "emergency_deadline_seconds": proposal.emergency_deadline_seconds,
    }


def protocol_telemetry_configuration() -> dict:
    return dict(phase4_configuration())


def protocol_telemetry_summary(telemetry: list[dict], ledger: list[dict]) -> dict:
    start_ns = next(
        row["monotonic_ns"]
        for row in ledger
        if row["event_type"] == "telemetry.started"
    )
    stop_ns = next(
        row["monotonic_ns"]
        for row in ledger
        if row["event_type"] == "telemetry.stopped"
    )
    summary = summarize_telemetry(telemetry, start_ns, stop_ns)
    present = {sample["scheduled_slot"] for sample in telemetry}
    summary["missing_scheduled_slots"] = [
        slot for slot in range(summary["expected_sample_count"]) if slot not in present
    ]
    allow_open_terminal_prefix = any(
        row["event_type"] == "harness.finished" and row["native_outcome"] != "passed"
        for row in ledger
    )
    return {
        "record_kind": "aptus-cuda-campaign-telemetry-summary-v1",
        "experiment_run_id": RUN_ID,
        "telemetry": summary,
        "segments": build_segment_summaries(
            telemetry,
            ledger,
            allow_open_terminal_prefix=allow_open_terminal_prefix,
        ),
    }


_PHASE4_FIXTURE_LOCK = threading.Lock()
_PHASE4_FIXTURE: tuple[dict[str, object], dict[str, object]] | None = None


def protocol_phase4_authority() -> tuple[dict[str, object], dict[str, object]]:
    global _PHASE4_FIXTURE
    with _PHASE4_FIXTURE_LOCK:
        if _PHASE4_FIXTURE is None:
            with tempfile.TemporaryDirectory() as temporary:
                root = private_directory(Path(temporary) / "root")
                repository = private_directory(root / "repository")
                boundary = _test_phase4_boundary(
                    source_binding=PROTOCOL_CELL["source_binding"],
                    host_observation=host_observation(),
                )
                result = _create_phase4_source_freeze_for_test(
                    directory=root / "phase4",
                    repository_root=repository,
                    campaign=PROTOCOL_CAMPAIGN,
                    comparison_cohort=PROTOCOL_COHORT,
                    comparison_cell=PROTOCOL_CELL,
                    telemetry_configuration=protocol_telemetry_configuration(),
                    telemetry_samples_path=baseline_path(root),
                    _trusted_boundary=boundary,
                )
                payloads: dict[str, object] = {
                    "campaign-record": PROTOCOL_CAMPAIGN,
                    "comparison-cohort-record": PROTOCOL_COHORT,
                    "comparison-cell-record": PROTOCOL_CELL,
                    "phase4-source-freeze": (
                        result.directory / PHASE4_SOURCE_FREEZE_NAME
                    ).read_bytes(),
                    "phase4-source-freeze-seal": (
                        result.directory / PHASE4_SOURCE_FREEZE_SEAL_NAME
                    ).read_bytes(),
                    "phase4-idle-baseline-samples": (
                        result.directory / PHASE4_IDLE_SAMPLES_NAME
                    ).read_bytes(),
                }
                _PHASE4_FIXTURE = (payloads, dict(result.baseline_binding))
        payloads, baseline = _PHASE4_FIXTURE
        return copy.deepcopy(payloads), copy.deepcopy(baseline)


def protocol_idle_baseline_binding() -> dict:
    return protocol_phase4_authority()[1]


def protocol_planned_slot_context() -> PlannedSlotContext:
    baseline = protocol_idle_baseline_binding()
    planned_slot = {
        **_SLOT_IDENTITY,
        "attempt_slot_id": SLOT_ID,
        "slot_status": "planned-not-started",
        "execution_configuration_id": None,
        "experiment_run_id": None,
        "native_outcome": None,
        "evidence_status": "not-started",
        "reason_code": "PRIOR_STOP_RULE",
    }
    budget = FrozenResourceBudget(
        plan_id=_PLAN_PAYLOAD["plan_id"],
        candidate_id=_PLAN_PAYLOAD["recommended"]["candidate_id"],
        bundle_fingerprint=_BUNDLE_FINGERPRINT,
        comparison_cell_id=PROTOCOL_CELL["comparison_cell_id"],
        attempt_slot_id=SLOT_ID,
        exact_artifact_bytes=1,
        plan_required_disk_bytes=1,
        largest_pilot_checkpoint_bytes=1,
        final_export_bytes=1,
        expected_copied_output_bytes=1,
        expected_log_bytes=1,
        expected_telemetry_bytes=1,
        plan_required_host_ram_bytes=1,
    )
    behavior = {
        "emergency_deadline_seconds": 600,
        "remaining_disk_budget_bytes": 10_000,
        "resource_budget_sha256": budget.sha256,
        "phase4_binding_sha256": sha256_bytes(compact_canonical_json_bytes(baseline)),
    }
    return PlannedSlotContext(
        campaign=PROTOCOL_CAMPAIGN,
        comparison_cohort=PROTOCOL_COHORT,
        comparison_cell=PROTOCOL_CELL,
        planned_attempt_slot=planned_slot,
        execution_proposal=ExecutionProposal(
            exact_behavior_values=behavior,
            plan_id=_PLAN_PAYLOAD["plan_id"],
            candidate_id=_PLAN_PAYLOAD["recommended"]["candidate_id"],
            bundle_fingerprint=_BUNDLE_FINGERPRINT,
            split_seed=424242,
            training_seed=17,
            data_order_seed=1_000_017,
            emergency_deadline_seconds=600,
        ),
        run_proposal=RunProposal(
            working_directory="/protected/bundle",
            fresh_state_root="/protected/state",
            bundle_path="/protected/bundle",
            output_path="/protected/bundle/runs",
            bundle_manifest_sha256=sha256_bytes(_BUNDLE_MANIFEST_BYTES),
            archive_sha256=sha256_bytes(_test_bundle_archive()),
        ),
        phase4_binding=baseline,
        resource_budget=budget,
    )


def _protocol_admission_probe() -> dict[str, object]:
    return {
        "gpu": {
            "uuid": "GPU-protected-storage-test",
            "memory_used": {"value": str(1 * GIB), "unit": "B"},
            "memory_free": {"value": str(7 * GIB), "unit": "B"},
            "memory_reserved": {"value": "0", "unit": "B"},
            "memory_total": {"value": str(8 * GIB), "unit": "B"},
            "utilization_percent": 0.0,
            "temperature_c": 35.0,
            "power_draw_w": 20.0,
            "power_limit_w": 130.0,
            "graphics_clock_mhz": 210.0,
            "memory_clock_mhz": 405.0,
            "performance_state": "P8",
            "throttle_reasons": [],
            "throttle_state": "0x0000000000000000",
            "xid_errors": [],
            "reset_detected": False,
            "device_lost": False,
            "hardware_error": False,
            "compute_processes": [],
        },
        "host": {
            "mem_available_bytes": 48 * GIB,
            "swap_used_bytes": 0,
            "swap_read_bytes": 0,
            "swap_write_bytes": 0,
            "load_1m": 0.25,
            "filesystem_free_bytes": 200 * GIB,
            "managed_process_rss_bytes": 0,
            "managed_process_cpu_seconds": 0.0,
            "managed_process_read_bytes": 0,
            "managed_process_write_bytes": 0,
            "disk_growth_bytes": 0,
            "aptus_lease_active": False,
            "cpu_temperature": {
                "status": "unsupported",
                "value": None,
                "reason_code": "NOT_CONFIGURED",
            },
            "nvme_temperature": {
                "status": "unsupported",
                "value": None,
                "reason_code": "NOT_CONFIGURED",
            },
        },
    }


_ACTIVATION_FIXTURE_LOCK = threading.Lock()
_ACTIVATION_FIXTURE: dict[str, object] | None = None


def protocol_activation_authority() -> dict[str, object]:
    global _ACTIVATION_FIXTURE
    with _ACTIVATION_FIXTURE_LOCK:
        if _ACTIVATION_FIXTURE is None:
            context = protocol_planned_slot_context()
            authority = InjectedAdmissionAuthority(lambda: dict(context.phase4_binding))
            _, authority_digest = authority_snapshot(authority)
            observations = []
            for index in range(120):
                timestamp = index * 1_000_000_000
                observations.append(
                    construct_admission_observation(
                        sequence=index,
                        admission_context_sha256=context.sha256,
                        current_authority_sha256=authority_digest,
                        scheduled_slot=index,
                        scheduled_monotonic_ns=timestamp,
                        observed_monotonic_ns=timestamp,
                        wall_time_utc=_WALL_TIME,
                        probe_reading=_protocol_admission_probe(),
                        collector={
                            "healthy": True,
                            "status_code": None,
                            "probe_duration_ns": 1000,
                        },
                        watchdog={
                            "healthy": True,
                            "heartbeat_monotonic_ns": timestamp,
                            "ownership_certain": True,
                        },
                    )
                )
            result = evaluate_pre_slot_admission(
                context,
                observations,
                authority=authority,
                acquisition_started_monotonic_ns=0,
                acquisition_finished_monotonic_ns=119_000_000_000,
            )
            if not result.admitted:  # pragma: no cover - fixture invariant.
                raise AssertionError(result.decision["reason_codes"])
            decision = dict(result.decision)
            decision["production_authority"] = True
            decision["production_observations"] = True
            execution, run, started, activation_record = (
                admission._build_activation_records(
                    SimpleNamespace(context=context, decision=decision),
                    authority_sha256=authority_digest,
                    experiment_run_id=RUN_ID,
                    activated_monotonic_ns=120_000_000_000,
                    activated_at_utc=_WALL_TIME,
                    production_qualifying=True,
                )
            )
            values = {
                "admission-decision.json": decision,
                "admission-observations.json": observations,
                "execution-configuration.json": execution,
                "experiment-run-template.json": run,
                "started-identity-template.json": started,
                "activation-decision.json": activation_record,
            }
            payloads = {
                name: canonical_json_bytes(value) for name, value in values.items()
            }
            seal = {
                "schema_version": "aptus.cuda-campaign-activation-seal.v1",
                "activation_decision_sha256": sha256_bytes(
                    payloads["activation-decision.json"]
                ),
                "files": [
                    {
                        "name": name,
                        "size_bytes": len(payloads[name]),
                        "sha256": sha256_bytes(payloads[name]),
                    }
                    for name in ACTIVATION_FILE_NAMES
                ],
            }
            payloads[ACTIVATION_SEAL_NAME] = canonical_json_bytes(seal)
            role_by_name = {
                "admission-decision.json": "activation-admission-decision",
                "admission-observations.json": "activation-admission-observations",
                "execution-configuration.json": ("activation-execution-configuration"),
                "experiment-run-template.json": ("activation-experiment-run-template"),
                "started-identity-template.json": (
                    "activation-started-identity-template"
                ),
                "activation-decision.json": "activation-decision",
                ACTIVATION_SEAL_NAME: "activation-seal",
            }
            _ACTIVATION_FIXTURE = {
                "planned-slot-context": context.record(),
                **{role_by_name[name]: payload for name, payload in payloads.items()},
            }
        return copy.deepcopy(_ACTIVATION_FIXTURE)


def protocol_cooldown_summary(telemetry: list[dict]) -> dict:
    cooldown = validate_cooldown(
        telemetry[1:], protocol_idle_baseline_binding()["summary"]
    )
    if not cooldown.valid:  # pragma: no cover - fixture construction invariant.
        raise AssertionError(cooldown.reason_codes)
    return {
        "record_kind": "aptus-cuda-campaign-cooldown-summary-v1",
        "experiment_run_id": RUN_ID,
        "valid": True,
        "reason_codes": [],
        "summary": dict(cooldown.summary),
    }


def protocol_terminal_job_record(index: int, action: str, job_id: str) -> dict:
    queued = index * 100 + 1
    return {
        "id": job_id,
        "job_id": job_id,
        "action": action,
        "state": "completed",
        "run_id": None,
        "return_code": 0,
        "monotonic_clock_binding": (
            "linux-boot-sha256:"
            + protocol_idle_baseline_binding()["current_boot_id_sha256"]
        ),
        "queued_monotonic_ns": queued,
        "child_process_started_monotonic_ns": queued + 1,
        "child_process_finished_monotonic_ns": queued + 2,
        "terminal_monotonic_ns": queued + 3,
    }


def protocol_terminal_job_projection(index: int, action: str, job_id: str) -> dict:
    record = protocol_terminal_job_record(index, action, job_id)
    return {
        "job_id": job_id,
        "run_id": None,
        "action": action,
        "state": "completed",
        "return_code": 0,
        "monotonic_clock_binding": record["monotonic_clock_binding"],
        "queued_monotonic_ns": record["queued_monotonic_ns"],
        "child_process_started_monotonic_ns": record[
            "child_process_started_monotonic_ns"
        ],
        "child_process_finished_monotonic_ns": record[
            "child_process_finished_monotonic_ns"
        ],
        "terminal_monotonic_ns": record["terminal_monotonic_ns"],
        "child_runtime_duration_ns": 1,
        "queue_to_terminal_duration_ns": 3,
    }


def protocol_outcome_terminal_record(index: int, started: dict) -> dict:
    job_id = started["job_id"]
    if not isinstance(job_id, str):
        raise TypeError("outcome terminal record requires a job ID")
    record = protocol_terminal_job_record(index, started["action"], job_id)
    if started["native_outcome"] == "failed":
        record.update(state="failed", return_code=9)
    elif started["native_outcome"] in {"cancelled", "timed-out"}:
        record.update(
            state="cancelled",
            return_code=None,
            cancel_reason_code=started["reason_code"],
        )
    return record


def protocol_outcome_terminal_projection(index: int, started: dict) -> dict:
    record = protocol_outcome_terminal_record(index, started)
    child_started = record["child_process_started_monotonic_ns"]
    child_finished = record["child_process_finished_monotonic_ns"]
    return {
        "job_id": record["job_id"],
        "run_id": record["run_id"],
        "action": record["action"],
        "state": record["state"],
        "return_code": record["return_code"],
        "monotonic_clock_binding": record["monotonic_clock_binding"],
        "queued_monotonic_ns": record["queued_monotonic_ns"],
        "child_process_started_monotonic_ns": child_started,
        "child_process_finished_monotonic_ns": child_finished,
        "terminal_monotonic_ns": record["terminal_monotonic_ns"],
        "child_runtime_duration_ns": child_finished - child_started,
        "queue_to_terminal_duration_ns": (
            record["terminal_monotonic_ns"] - record["queued_monotonic_ns"]
        ),
    }


def protocol_outcome_timing_summary(jobs: list[dict]) -> dict | None:
    if not jobs:
        return None
    return {
        "monotonic_clock_binding": jobs[0]["monotonic_clock_binding"],
        "submitted_jobs_span_ns": (
            jobs[-1]["terminal_monotonic_ns"] - jobs[0]["queued_monotonic_ns"]
        ),
        "child_runtime_duration_ns_by_action": {
            item["action"]: item["child_runtime_duration_ns"] for item in jobs
        },
        "queue_to_terminal_duration_ns_by_action": {
            item["action"]: item["queue_to_terminal_duration_ns"] for item in jobs
        },
    }


def protocol_terminal_timing_summary() -> dict:
    return {
        "monotonic_clock_binding": (
            "linux-boot-sha256:"
            + protocol_idle_baseline_binding()["current_boot_id_sha256"]
        ),
        "submitted_jobs_span_ns": 403,
        "child_runtime_duration_ns_by_action": {
            action: 1 for action in QUALIFYING_ACTIONS
        },
        "queue_to_terminal_duration_ns_by_action": {
            action: 3 for action in QUALIFYING_ACTIONS
        },
    }


def protocol_experiment_run_record(
    evidence_role_sha256: dict[str, str] | None = None,
    *,
    bundle_manifest_sha256: str | None = None,
    idle_baseline_sha256: str | None = None,
) -> dict:
    artifact_payloads = protocol_required_artifact_payloads()
    if evidence_role_sha256 is None:
        telemetry = protocol_telemetry_records()
        ledger = protocol_event_records()
        phase4_payloads, _baseline = protocol_phase4_authority()
        values = {
            "attempt-slot-record": protocol_attempt_slot_record(),
            "execution-configuration-record": (
                protocol_execution_configuration_record()
            ),
            "idle-baseline-binding": protocol_idle_baseline_binding(),
            "telemetry-configuration": protocol_telemetry_configuration(),
            "telemetry-summary": protocol_telemetry_summary(telemetry, ledger),
            "cooldown-summary": protocol_cooldown_summary(telemetry),
            **artifact_payloads,
            **phase4_payloads,
            **protocol_activation_authority(),
        }
        evidence_role_sha256 = {
            role: sha256_bytes(_artifact_payload_bytes(value))
            for role, value in values.items()
        }
    if bundle_manifest_sha256 is None:
        bundle_manifest_sha256 = sha256_bytes(
            canonical_json_bytes(artifact_payloads["bundle-manifest"])
        )
    if idle_baseline_sha256 is None:
        idle_baseline_sha256 = sha256_bytes(
            canonical_json_bytes(protocol_idle_baseline_binding())
        )
    template = json.loads(
        protocol_activation_authority()["activation-experiment-run-template"]
    )
    return {
        **template,
        "exact_argv": ["aptus", "dependency"],
        "observed_host_state": {
            **template["observed_host_state"],
            "idle_baseline_sha256": idle_baseline_sha256,
        },
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "aptus_job_ids": list(QUALIFYING_JOB_IDS),
        "aptus_run_ids": [],
        "terminal_evidence": {
            "native_outcome": "passed",
            "evidence_status": "protocol-valid",
            "reason_code": "NONE",
            "jobs": [
                protocol_terminal_job_projection(index, action, job_id)
                for index, (action, job_id) in enumerate(
                    zip(QUALIFYING_ACTIONS, QUALIFYING_JOB_IDS)
                )
            ],
            "timing": protocol_terminal_timing_summary(),
            "evidence_role_sha256": dict(evidence_role_sha256),
        },
    }


def protocol_managed_writer(
    root: Path,
    name: str,
    *,
    ledger: object = _DEFAULT,
    telemetry: object = _DEFAULT,
    summary: object = _DEFAULT,
    terminal: object = _DEFAULT,
    experiment_record: object = _DEFAULT,
    attempt_record: object = _DEFAULT,
    execution_record: object = _DEFAULT,
    idle_baseline: object = _DEFAULT,
    telemetry_configuration: object = _DEFAULT,
    telemetry_summary: object = _DEFAULT,
    cooldown_summary: object = _DEFAULT,
    runtime_journals: object = _DEFAULT,
    required_artifacts: object = _DEFAULT,
    activation_authority: object = _DEFAULT,
    outcome_fixture: bool = False,
    terminal_label: str = "dependency",
    include_experiment_record: bool = True,
) -> RawArtifactWriter:
    default_ledger = protocol_event_records()
    default_telemetry = protocol_telemetry_records()
    ledger_value = default_ledger if ledger is _DEFAULT else ledger
    telemetry_value = default_telemetry if telemetry is _DEFAULT else telemetry
    summary_value = protocol_sequence_summary() if summary is _DEFAULT else summary
    attempt_value = (
        protocol_attempt_slot_record() if attempt_record is _DEFAULT else attempt_record
    )
    execution_value = (
        protocol_execution_configuration_record()
        if execution_record is _DEFAULT
        else execution_record
    )
    authority_values, phase4_baseline = protocol_phase4_authority()
    activation_values = (
        protocol_activation_authority()
        if activation_authority is _DEFAULT
        else activation_authority
    )
    if type(activation_values) is not dict:
        raise TypeError("activation_authority test fixture must be a dictionary")
    idle_baseline_value = (
        phase4_baseline if idle_baseline is _DEFAULT else idle_baseline
    )
    configuration_value = (
        protocol_telemetry_configuration()
        if telemetry_configuration is _DEFAULT
        else telemetry_configuration
    )
    telemetry_summary_value = (
        protocol_telemetry_summary(
            telemetry_value if outcome_fixture else default_telemetry,
            ledger_value if outcome_fixture else default_ledger,
        )
        if telemetry_summary is _DEFAULT
        else telemetry_summary
    )
    cooldown_summary_value = (
        protocol_cooldown_summary(default_telemetry)
        if cooldown_summary is _DEFAULT
        else cooldown_summary
    )
    runtime_journal_values = (
        protocol_runtime_journal_records(default_ledger)
        if runtime_journals is _DEFAULT
        else runtime_journals
    )
    required_artifact_values = (
        protocol_required_artifact_payloads()
        if required_artifacts is _DEFAULT
        else required_artifacts
    )
    started_rows = list(
        summary_value["started_actions"]
        if outcome_fixture
        else protocol_sequence_summary()["started_actions"]
    )
    if outcome_fixture:
        pilot_passed = any(
            row["action"] == "pilot" and row["native_outcome"] == "passed"
            for row in started_rows
        )
        required_artifact_values = {
            role: value
            for role, value in required_artifact_values.items()
            if role
            not in {
                "training-metrics",
                "final-export-manifest",
                *(set() if pilot_passed else {"pilot-metrics"}),
            }
        }
        runtime_journal_values = {
            action: records
            for action, records in protocol_runtime_journal_records(
                ledger_value
            ).items()
            if any(
                row["action"] == action and row["job_id"] is not None
                for row in started_rows
            )
        }
        attempt_value = {
            **protocol_attempt_slot_record(),
            "native_outcome": summary_value["native_outcome"],
            "evidence_status": summary_value["evidence_status"],
            "reason_code": summary_value["reason_code"],
        }
    retained_values = {
        "attempt-slot-record": attempt_value,
        "execution-configuration-record": execution_value,
        "idle-baseline-binding": idle_baseline_value,
        "telemetry-configuration": configuration_value,
        "telemetry-summary": telemetry_summary_value,
        **required_artifact_values,
        **authority_values,
        **activation_values,
    }
    if not outcome_fixture:
        retained_values["cooldown-summary"] = cooldown_summary_value
    embedded_digests = {
        role: sha256_bytes(_artifact_payload_bytes(value))
        for role, value in retained_values.items()
    }
    if experiment_record is _DEFAULT:
        run_value = protocol_experiment_run_record(
            embedded_digests,
            bundle_manifest_sha256=sha256_bytes(
                canonical_json_bytes(required_artifact_values["bundle-manifest"])
            ),
            idle_baseline_sha256=sha256_bytes(
                canonical_json_bytes(idle_baseline_value)
            ),
        )
        if outcome_fixture:
            terminal_projections = [
                protocol_outcome_terminal_projection(index, row)
                for index, row in enumerate(started_rows)
                if row["job_id"] is not None and row["terminal"] is True
            ]
            run_value.update(
                aptus_job_ids=[
                    row["job_id"] for row in started_rows if row["job_id"] is not None
                ],
                aptus_run_ids=[],
                terminal_evidence={
                    "native_outcome": summary_value["native_outcome"],
                    "evidence_status": summary_value["evidence_status"],
                    "reason_code": summary_value["reason_code"],
                    "jobs": terminal_projections,
                    "timing": protocol_outcome_timing_summary(terminal_projections),
                    "evidence_role_sha256": dict(embedded_digests),
                },
            )
    else:
        run_value = experiment_record
    source_role_digests = {
        **embedded_digests,
        "experiment-run-record": sha256_bytes(canonical_json_bytes(run_value)),
    }
    role_bindings: dict[str, str | list[str]] = {
        "event-ledger": "entry_protocol_ledger",
        "telemetry": "entry_protocol_telemetry",
        "sequence-summary": "entry_protocol_summary",
        "terminal-job-record": [
            f"entry_protocol_terminal_{index}"
            for index, row in enumerate(started_rows)
            if row["job_id"] is not None and row["terminal"] is True
        ],
        "job-log": [
            f"entry_protocol_log_{index}"
            for index, row in enumerate(started_rows)
            if row["job_id"] is not None
        ],
        "action-submission-record": [
            f"entry_protocol_submission_{index}"
            for index, row in enumerate(started_rows)
            if row["job_id"] is None
        ],
        "attempt-slot-record": "entry_protocol_slot",
        "execution-configuration-record": "entry_protocol_execution",
        "idle-baseline-binding": "entry_protocol_idle_baseline",
        "telemetry-configuration": "entry_protocol_configuration",
        "telemetry-summary": "entry_protocol_telemetry_summary",
        "campaign-record": "entry_protocol_campaign",
        "comparison-cohort-record": "entry_protocol_cohort",
        "comparison-cell-record": "entry_protocol_cell",
        "phase4-source-freeze": "entry_protocol_phase4_freeze",
        "phase4-source-freeze-seal": "entry_protocol_phase4_seal",
        "phase4-idle-baseline-samples": "entry_protocol_phase4_samples",
        "runtime-boundary-journal": [
            f"entry_protocol_{action}_runtime_boundaries"
            for action in runtime_journal_values
        ],
    }
    if not outcome_fixture:
        role_bindings["cooldown-summary"] = "entry_protocol_cooldown_summary"
    for optional_role in (
        "terminal-job-record",
        "job-log",
        "action-submission-record",
        "runtime-boundary-journal",
    ):
        if not role_bindings[optional_role]:
            role_bindings.pop(optional_role)
    for role in activation_values:
        role_bindings[role] = f"entry_protocol_{role}"
    for role in required_artifact_values:
        role_bindings[role] = f"entry_protocol_{role}"
    if include_experiment_record:
        role_bindings["experiment-run-record"] = "entry_protocol_run"
    writer = _ProtocolFixtureWriter(
        root / name,
        protected_artifact_id="artifact_" + "9" * 32,
        record_kind="experiment-run",
        identity_bindings={
            "campaign_id": PROTOCOL_CAMPAIGN["campaign_id"],
            "comparison_cohort_id": PROTOCOL_COHORT["comparison_cohort_id"],
            "comparison_cell_id": PROTOCOL_CELL["comparison_cell_id"],
            "attempt_slot_id": SLOT_ID,
            "experiment_run_id": RUN_ID,
            "execution_configuration_id": protocol_execution_configuration_record()[
                "execution_configuration_id"
            ],
            "capture_kind": "managed-sequence",
            "capture_status": "complete",
            "evidence_status": "protocol-valid",
            "capture_reason_code": "NONE",
        },
        capture_tool={"name": "storage-test", "version": "v1"},
        source_bindings={
            "source_commit": "1" * 40,
            "execution_configuration_sha256": embedded_digests[
                "execution-configuration-record"
            ],
            "idle_baseline_binding_sha256": embedded_digests["idle-baseline-binding"],
            "campaign_sha256": embedded_digests["campaign-record"],
            "comparison_cohort_sha256": embedded_digests["comparison-cohort-record"],
            "comparison_cell_sha256": embedded_digests["comparison-cell-record"],
            "phase4_source_freeze_sha256": embedded_digests["phase4-source-freeze"],
            "phase4_source_freeze_seal_sha256": embedded_digests[
                "phase4-source-freeze-seal"
            ],
            "phase4_idle_baseline_samples_sha256": embedded_digests[
                "phase4-idle-baseline-samples"
            ],
            "planned_slot_context_sha256": protocol_planned_slot_context().sha256,
            "planned_attempt_slot_sha256": sha256_bytes(
                canonical_json_bytes(
                    dict(protocol_planned_slot_context().planned_attempt_slot)
                )
            ),
            "admission_decision_sha256": embedded_digests[
                "activation-admission-decision"
            ],
            "admission_observations_sha256": embedded_digests[
                "activation-admission-observations"
            ],
            "activation_decision_sha256": embedded_digests["activation-decision"],
            "started_identity_template_sha256": embedded_digests[
                "activation-started-identity-template"
            ],
            "experiment_run_template_sha256": embedded_digests[
                "activation-experiment-run-template"
            ],
            "activation_provenance_sha256_by_role": {
                role: embedded_digests[role] for role in sorted(activation_values)
            },
            "evidence_role_sha256": source_role_digests,
        },
        provisional_retain_not_before_utc=RETAIN_UNTIL,
        required_role_bindings=role_bindings,
    )
    writer.write_payload(
        canonical_jsonl_bytes(ledger_value),
        "events/events.jsonl",
        role="event-ledger",
        media_type="application/x-ndjson",
        entry_id="entry_protocol_ledger",
    )
    writer.write_payload(
        canonical_jsonl_bytes(telemetry_value),
        "telemetry/samples.jsonl",
        role="telemetry",
        media_type="application/x-ndjson",
        entry_id="entry_protocol_telemetry",
    )
    writer.write_payload(
        canonical_json_bytes(summary_value),
        "sequence/summary.json",
        role="sequence-summary",
        media_type="application/json",
        entry_id="entry_protocol_summary",
    )
    for index, started in enumerate(started_rows):
        action = started["action"]
        label = terminal_label if index == 0 else started["label"]
        if started["job_id"] is None:
            record = {
                "record_kind": (
                    "aptus-cuda-campaign-submission-refusal-v1"
                    if started["native_outcome"] == "refused"
                    else "aptus-cuda-campaign-pre-submit-guard-v1"
                    if started["native_outcome"] == "guard-blocked"
                    else "aptus-cuda-campaign-ambiguous-submission-failure-v1"
                ),
                "action_label": started["label"],
                "action": action,
                "native_outcome": started["native_outcome"],
                "reason_code": started["reason_code"],
            }
            if started["native_outcome"] == "refused":
                record["exception_type"] = "JobPrerequisiteError"
            writer.write_payload(
                canonical_json_bytes(record),
                f"actions/{label}/submission.json",
                role="action-submission-record",
                media_type="application/json",
                entry_id=f"entry_protocol_submission_{index}",
            )
            continue
        terminal_value = (
            terminal
            if index == 0 and terminal is not _DEFAULT
            else protocol_outcome_terminal_record(index, started)
        )
        writer.write_payload(
            canonical_json_bytes(terminal_value),
            f"actions/{label}/terminal.json",
            role="terminal-job-record",
            media_type="application/json",
            entry_id=f"entry_protocol_terminal_{index}",
        )
        writer.write_payload(
            f"complete {action} log\n".encode(),
            f"actions/{label}/full.log",
            role="job-log",
            media_type="text/plain",
            entry_id=f"entry_protocol_log_{index}",
        )
    for action, records in runtime_journal_values.items():
        writer.write_payload(
            canonical_jsonl_bytes(records),
            f"actions/{action}/runtime-boundaries.jsonl",
            role="runtime-boundary-journal",
            media_type="application/x-ndjson",
            entry_id=f"entry_protocol_{action}_runtime_boundaries",
        )
    for role, value in required_artifact_values.items():
        payload = _artifact_payload_bytes(value)
        writer.write_payload(
            payload,
            (
                "selected/bundle-archive.zip"
                if role == "bundle-archive"
                else f"selected/{role}.json"
            ),
            role=role,
            media_type=(
                "application/zip" if role == "bundle-archive" else "application/json"
            ),
            entry_id=f"entry_protocol_{role}",
        )
    for role, value, path, entry_id in (
        (
            "attempt-slot-record",
            attempt_value,
            "identity/final-slot.json",
            "entry_protocol_slot",
        ),
        (
            "execution-configuration-record",
            execution_value,
            "identity/execution.json",
            "entry_protocol_execution",
        ),
        (
            "idle-baseline-binding",
            idle_baseline_value,
            "identity/idle-baseline-binding.json",
            "entry_protocol_idle_baseline",
        ),
        (
            "telemetry-configuration",
            configuration_value,
            "telemetry/configuration.json",
            "entry_protocol_configuration",
        ),
        (
            "telemetry-summary",
            telemetry_summary_value,
            "summaries/telemetry.json",
            "entry_protocol_telemetry_summary",
        ),
        (
            "cooldown-summary",
            cooldown_summary_value,
            "summaries/cooldown.json",
            "entry_protocol_cooldown_summary",
        ),
    ):
        if outcome_fixture and role == "cooldown-summary":
            continue
        writer.write_payload(
            canonical_json_bytes(value),
            path,
            role=role,
            media_type="application/json",
            entry_id=entry_id,
        )
    for role, value, path, entry_id, media_type in (
        (
            "campaign-record",
            authority_values["campaign-record"],
            "authority/campaign.json",
            "entry_protocol_campaign",
            "application/json",
        ),
        (
            "comparison-cohort-record",
            authority_values["comparison-cohort-record"],
            "authority/comparison-cohort.json",
            "entry_protocol_cohort",
            "application/json",
        ),
        (
            "comparison-cell-record",
            authority_values["comparison-cell-record"],
            "authority/comparison-cell.json",
            "entry_protocol_cell",
            "application/json",
        ),
        (
            "phase4-source-freeze",
            authority_values["phase4-source-freeze"],
            "authority/phase4-source-freeze.json",
            "entry_protocol_phase4_freeze",
            "application/json",
        ),
        (
            "phase4-source-freeze-seal",
            authority_values["phase4-source-freeze-seal"],
            "authority/PHASE4-SEALED.json",
            "entry_protocol_phase4_seal",
            "application/json",
        ),
        (
            "phase4-idle-baseline-samples",
            authority_values["phase4-idle-baseline-samples"],
            "authority/idle-baseline-samples.jsonl",
            "entry_protocol_phase4_samples",
            "application/x-ndjson",
        ),
    ):
        writer.write_payload(
            _artifact_payload_bytes(value),
            path,
            role=role,
            media_type=media_type,
            entry_id=entry_id,
        )
    activation_paths = {
        "planned-slot-context": "activation/planned-slot-context.json",
        "activation-admission-decision": "activation/admission-decision.json",
        "activation-admission-observations": ("activation/admission-observations.json"),
        "activation-execution-configuration": (
            "activation/execution-configuration.json"
        ),
        "activation-experiment-run-template": (
            "activation/experiment-run-template.json"
        ),
        "activation-started-identity-template": (
            "activation/started-identity-template.json"
        ),
        "activation-decision": "activation/activation-decision.json",
        "activation-seal": "activation/ACTIVATED.json",
    }
    for role, value in activation_values.items():
        writer.write_payload(
            _artifact_payload_bytes(value),
            activation_paths[role],
            role=role,
            media_type="application/json",
            entry_id=f"entry_protocol_{role}",
        )
    if include_experiment_record:
        writer.write_payload(
            canonical_json_bytes(run_value),
            "identity/experiment-run.json",
            role="experiment-run-record",
            media_type="application/json",
            entry_id="entry_protocol_run",
        )
    return writer


class RawArtifactStorageTests(unittest.TestCase):
    def test_storage_bundle_fixture_is_real_and_deterministic(self) -> None:
        plan, manifest, archive = _canonical_test_bundle()

        self.assertEqual(plan, _PLAN_PAYLOAD)
        self.assertEqual(manifest, _BUNDLE_MANIFEST_PAYLOAD)
        self.assertEqual(archive, _BUNDLE_ARCHIVE_BYTES)
        inventory = validate_bundle_archive_bytes(archive)
        self.assertEqual(
            inventory["plan.json"]["sha256"],
            sha256_bytes(_PLAN_BYTES),
        )
        self.assertIn("train.py", inventory)
        self.assertIn("campaign_events.py", inventory)

    def test_final_inventory_rewalk_rejects_seal_and_verify_toctou(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = writer_at(vault, "seal-race")
            writer.write_payload(
                b"payload\n", "jobs/job.log", role="job-log", entry_id=ENTRY_ID
            )
            original = storage._assert_exact_artifact_inventory
            calls = 0

            def inject_after_rewalk(
                root: Path, expected: dict[str, tuple[int, str]]
            ) -> None:
                nonlocal calls
                original(root, expected)
                calls += 1
                if calls == 1:
                    extra = root / "late-extra.bin"
                    extra.write_bytes(b"late\n")
                    extra.chmod(0o600)

            with patch.object(
                storage,
                "_assert_exact_artifact_inventory",
                side_effect=inject_after_rewalk,
            ):
                with self.assertRaisesRegex(ArtifactIntegrityError, "inventory"):
                    writer.seal()
            self.assertFalse((writer.directory / "SEALED.json").exists())

        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            artifact, _result = sealed_artifact(vault, "verify-race")
            original_hash = storage._hash_pinned_evidence_file
            injected = False

            def inject_during_hash(path: Path) -> tuple[int, str]:
                nonlocal injected
                result = original_hash(path)
                if path.name == "job.log" and not injected:
                    injected = True
                    extra = artifact / "late-verify-extra.bin"
                    extra.write_bytes(b"late\n")
                    extra.chmod(0o600)
                return result

            with patch.object(
                storage,
                "_hash_pinned_evidence_file",
                side_effect=inject_during_hash,
            ):
                with self.assertRaisesRegex(ArtifactIntegrityError, "inventory"):
                    verify_sealed_artifact(artifact)

    def test_pinned_evidence_reads_reject_links_and_path_swaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_bytes(b"{}\n")
            source.chmod(0o600)
            symlink = root / "symlink.json"
            symlink.symlink_to(source)
            hardlink = root / "hardlink.json"
            os.link(source, hardlink)

            for reader in (
                storage._read_pinned_evidence_bytes,
                storage._hash_pinned_evidence_file,
            ):
                with self.subTest(reader=reader.__name__, attack="symlink"):
                    with self.assertRaises(ArtifactIntegrityError):
                        reader(symlink)
                with self.subTest(reader=reader.__name__, attack="hardlink"):
                    with self.assertRaises(ArtifactIntegrityError):
                        reader(hardlink)

            real_open = os.open
            for index, reader in enumerate(
                (
                    storage._read_pinned_evidence_bytes,
                    storage._hash_pinned_evidence_file,
                )
            ):
                victim = root / f"victim-{index}.json"
                replacement = root / f"replacement-{index}.json"
                held = root / f"held-{index}.json"
                victim.write_bytes(b"original\n")
                replacement.write_bytes(b"replacement\n")
                victim.chmod(0o600)
                replacement.chmod(0o600)
                swapped = False

                def swap_then_open(
                    target: object, flags: int, mode: int = 0o777
                ) -> int:
                    nonlocal swapped
                    if Path(target) == victim and not swapped:
                        swapped = True
                        victim.rename(held)
                        replacement.rename(victim)
                    return real_open(target, flags, mode)

                with self.subTest(reader=reader.__name__, attack="path-swap"):
                    with patch(
                        "tools.cuda_campaign.storage.os.open",
                        side_effect=swap_then_open,
                    ):
                        with self.assertRaisesRegex(
                            ArtifactIntegrityError, "identity changed"
                        ):
                            reader(victim)

    def test_fsync_rejects_same_and_different_byte_path_replacements(self) -> None:
        for replacement_payload in (b"original\n", b"replacement\n"):
            with self.subTest(replacement_payload=replacement_payload):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    victim = root / "victim.bin"
                    replacement = root / "replacement.bin"
                    held = root / "held.bin"
                    victim.write_bytes(b"original\n")
                    replacement.write_bytes(replacement_payload)
                    victim.chmod(0o600)
                    replacement.chmod(0o600)
                    real_open = os.open
                    swapped = False

                    def swap_then_open(
                        target: object, flags: int, mode: int = 0o777
                    ) -> int:
                        nonlocal swapped
                        if Path(target) == victim and not swapped:
                            swapped = True
                            victim.rename(held)
                            replacement.rename(victim)
                        return real_open(target, flags, mode)

                    with patch(
                        "tools.cuda_campaign.storage.os.open",
                        side_effect=swap_then_open,
                    ):
                        with self.assertRaisesRegex(
                            ArtifactIntegrityError, "identity changed"
                        ):
                            storage._fsync_regular_file(victim)

    def test_seal_is_canonical_private_and_deeply_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            artifact, result = sealed_artifact(vault)

            self.assertEqual(result["protected_artifact_id"], ARTIFACT_ID)
            self.assertEqual(result["file_count"], 1)
            self.assertEqual(
                result["manifest"]["files"][0]["relative_path"],
                "jobs/job.log",
            )
            self.assertEqual(
                (artifact / "raw-manifest.json").read_bytes(),
                storage.canonical_json_bytes(result["manifest"]),
            )
            self.assertEqual(
                stat.S_IMODE(artifact.stat().st_mode),
                0o700,
            )
            self.assertEqual(
                stat.S_IMODE((artifact / "jobs").stat().st_mode),
                0o700,
            )
            for path in (
                artifact / "jobs/job.log",
                artifact / "raw-manifest.json",
                artifact / "SEALED.json",
            ):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            self.assertEqual(
                verify_sealed_artifact(artifact)["raw_manifest_sha256"],
                result["raw_manifest_sha256"],
            )

    def test_payload_rejects_traversal_symlink_and_hardlink_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vault = private_directory(root / "vault")
            source = root / "source.log"
            source.write_text("source\n", encoding="utf-8")
            writer = writer_at(vault)

            with self.assertRaises(ValueError):
                writer.write_payload(
                    b"escape", "../escape", role="job-log", entry_id=ENTRY_ID
                )

            if os.name == "posix":
                symlink = root / "source-link.log"
                symlink.symlink_to(source)
                with self.assertRaisesRegex(EvidenceStorageError, "non-symlink"):
                    writer.copy_payload(
                        symlink,
                        "jobs/symlink.log",
                        role="job-log",
                    )

                hardlink = root / "source-hardlink.log"
                os.link(source, hardlink)
                with self.assertRaisesRegex(EvidenceStorageError, "hardlinked"):
                    writer.copy_payload(
                        source,
                        "jobs/hardlink.log",
                        role="job-log",
                    )

            self.assertFalse((vault / "escape").exists())

    def test_no_clobber_applies_to_payload_artifact_and_reseal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = writer_at(vault)
            writer.write_payload(
                b"one",
                "jobs/job.log",
                role="job-log",
                entry_id=ENTRY_ID,
            )
            with self.assertRaises(FileExistsError):
                writer.write_payload(
                    b"two",
                    "jobs/job.log",
                    role="job-log",
                )
            writer.seal()
            with self.assertRaises(FileExistsError):
                writer.seal()
            with self.assertRaises(FileExistsError):
                writer_at(vault)
            self.assertEqual((vault / "artifact/jobs/job.log").read_bytes(), b"one")

    def test_interrupted_seal_is_not_complete_and_cannot_be_resealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = writer_at(vault)
            writer.write_payload(
                b"partial",
                "jobs/job.log",
                role="job-log",
                entry_id=ENTRY_ID,
            )
            original = storage._write_exclusive_control_bytes

            def interrupt(
                path: Path, payload: bytes, *, mode: int = 0o600
            ) -> tuple[int, int]:
                if path.name == "SEALED.json":
                    raise OSError("simulated interruption")
                return original(path, payload, mode=mode)

            with patch.object(
                storage, "_write_exclusive_control_bytes", side_effect=interrupt
            ):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    writer.seal()

            self.assertTrue((writer.directory / "raw-manifest.json").is_file())
            self.assertFalse((writer.directory / "SEALED.json").exists())
            with self.assertRaises(FileExistsError):
                writer.seal()
            with self.assertRaises(ArtifactIntegrityError):
                verify_sealed_artifact(writer.directory)

    def test_post_seal_mutation_extra_file_and_hardlink_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vault = private_directory(root / "vault")
            artifact, _ = sealed_artifact(vault)
            payload = artifact / "jobs/job.log"
            payload.write_bytes(b"changed\n")
            payload.chmod(0o600)
            with self.assertRaises(ArtifactIntegrityError):
                verify_sealed_artifact(artifact)

            artifact_two, _ = sealed_artifact(vault, "artifact-two")
            extra = artifact_two / "unexpected.txt"
            extra.write_text("unexpected", encoding="utf-8")
            extra.chmod(0o600)
            with self.assertRaises(ArtifactIntegrityError):
                verify_sealed_artifact(artifact_two)

            if os.name == "posix":
                artifact_three, _ = sealed_artifact(vault, "artifact-three")
                os.link(
                    artifact_three / "jobs/job.log",
                    root / "external-hardlink.log",
                )
                with self.assertRaisesRegex(ArtifactIntegrityError, "hardlinked"):
                    verify_sealed_artifact(artifact_three)

    def test_copy_is_byte_identical_and_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = private_directory(root / "source-vault")
            copy_root = private_directory(root / "copy-vault")
            source, source_result = sealed_artifact(source_root)
            destination = copy_root / "artifact-copy"

            copied = copy_sealed_artifact(source, destination)

            self.assertEqual(
                copied["raw_manifest_sha256"],
                source_result["raw_manifest_sha256"],
            )
            self.assertEqual(
                verify_copy_equality(source, destination)["verification_result"],
                "passed",
            )
            with self.assertRaisesRegex(ArtifactIntegrityError, "distinct root"):
                verify_copy_equality(source, source)
            with self.assertRaisesRegex(
                EvidenceStorageError, "inside its sealed source"
            ):
                copy_sealed_artifact(source, source / "nested-copy")
            self.assertFalse((source / "nested-copy").exists())
            copied_payload = destination / "jobs/job.log"
            copied_payload.write_bytes(b"copy changed")
            copied_payload.chmod(0o600)
            with self.assertRaises(ArtifactIntegrityError):
                verify_copy_equality(source, destination)

    def test_copy_equality_rejects_exact_content_root_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = private_directory(root / "source-vault")
            copy_root = private_directory(root / "copy-vault")
            source, _source_result = sealed_artifact(source_root)
            destination = copy_root / "artifact-copy"
            replacement = copy_root / "artifact-replacement"
            copy_sealed_artifact(source, destination)
            copy_sealed_artifact(source, replacement)
            held = source_root / "artifact-held"
            real_verify = storage.verify_sealed_artifact
            swapped = False

            def swap_after_semantic_verification(path: Path) -> dict:
                nonlocal swapped
                result = real_verify(path)
                if path == source and not swapped:
                    swapped = True
                    source.rename(held)
                    replacement.rename(source)
                return result

            with patch.object(
                storage,
                "verify_sealed_artifact",
                side_effect=swap_after_semantic_verification,
            ):
                with self.assertRaisesRegex(
                    ArtifactIntegrityError, "directory (?:path )?changed"
                ):
                    verify_copy_equality(source, destination)

    def test_experiment_run_rejects_incomplete_or_contradictory_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            incomplete = RawArtifactWriter(
                vault / "incomplete",
                protected_artifact_id=ARTIFACT_ID,
                record_kind="experiment-run",
                identity_bindings={
                    "attempt_slot_id": SLOT_ID,
                    "experiment_run_id": RUN_ID,
                },
                capture_tool={"name": "storage-test", "version": "v1"},
                source_bindings={"source_commit": "1" * 40},
                provisional_retain_not_before_utc=RETAIN_UNTIL,
                required_role_bindings={"job-log": ENTRY_ID},
            )
            incomplete.write_payload(
                b"job\n", "job/full.log", role="job-log", entry_id=ENTRY_ID
            )
            with self.assertRaisesRegex(EvidenceStorageError, "event-ledger"):
                incomplete.seal()
            self.assertFalse((incomplete.directory / "raw-manifest.json").exists())

            contradictory = command_writer(
                vault,
                "contradictory",
                run_id="xrun_" + "1" * 32,
                artifact_id="artifact_" + "1" * 32,
            )
            contradictory.write_payload(
                b"extra job log\n",
                "job/full.log",
                role="job-log",
                entry_id="entry_contradictory_job_log",
            )
            with self.assertRaisesRegex(
                EvidenceStorageError, "one complete capture profile"
            ):
                contradictory.seal()

    def test_experiment_run_requires_every_core_entry_to_be_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = command_writer(vault, "unbound")
            writer.write_payload(
                b"unbound duplicate output\n",
                "command/other-output.bin",
                role="command-output",
                entry_id="entry_unbound_extra_output",
            )

            with self.assertRaisesRegex(
                EvidenceStorageError, "exactly 1 'command-output'"
            ):
                writer.seal()
            self.assertFalse((writer.directory / "raw-manifest.json").exists())

    def test_managed_sequence_requires_one_record_for_each_job_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = RawArtifactWriter(
                vault / "managed-sequence",
                protected_artifact_id=ARTIFACT_ID,
                record_kind="experiment-run",
                identity_bindings={
                    "attempt_slot_id": SLOT_ID,
                    "experiment_run_id": RUN_ID,
                    "capture_kind": "managed-sequence",
                    "capture_status": "complete",
                },
                capture_tool={"name": "storage-test", "version": "v1"},
                source_bindings={"source_commit": "1" * 40},
                provisional_retain_not_before_utc=RETAIN_UNTIL,
                required_role_bindings={
                    "job-log": ["entry_job_log_1", "entry_job_log_2"],
                    "terminal-job-record": [
                        "entry_terminal_1",
                        "entry_terminal_2",
                    ],
                    "sequence-summary": "entry_sequence_summary",
                    "event-ledger": "entry_sequence_ledger",
                },
            )
            for index in (1, 2):
                writer.write_payload(
                    f"job {index}\n".encode(),
                    f"jobs/{index}/full.log",
                    role="job-log",
                    entry_id=f"entry_job_log_{index}",
                )
                writer.write_payload(
                    b"{}\n",
                    f"jobs/{index}/terminal.json",
                    role="terminal-job-record",
                    entry_id=f"entry_terminal_{index}",
                )
            writer.write_payload(
                b"{}\n",
                "sequence/summary.json",
                role="sequence-summary",
                entry_id="entry_sequence_summary",
            )
            writer.write_payload(
                b"{}\n",
                "events/events.jsonl",
                role="event-ledger",
                entry_id="entry_sequence_ledger",
            )

            result = writer.seal()

            self.assertEqual(result["file_count"], 6)
            self.assertEqual(
                result["manifest"]["required_role_bindings"]["job-log"],
                ["entry_job_log_1", "entry_job_log_2"],
            )

    def test_managed_sequence_rejects_job_record_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = RawArtifactWriter(
                vault / "managed-sequence",
                protected_artifact_id=ARTIFACT_ID,
                record_kind="experiment-run",
                identity_bindings={
                    "attempt_slot_id": SLOT_ID,
                    "experiment_run_id": RUN_ID,
                    "capture_kind": "managed-sequence",
                },
                capture_tool={"name": "storage-test", "version": "v1"},
                source_bindings={"source_commit": "1" * 40},
                provisional_retain_not_before_utc=RETAIN_UNTIL,
                required_role_bindings={
                    "job-log": ["entry_job_log_1", "entry_job_log_2"],
                    "terminal-job-record": "entry_terminal_1",
                    "sequence-summary": "entry_sequence_summary",
                    "event-ledger": "entry_sequence_ledger",
                },
            )
            for index in (1, 2):
                writer.write_payload(
                    b"log\n",
                    f"jobs/{index}/full.log",
                    role="job-log",
                    entry_id=f"entry_job_log_{index}",
                )
            writer.write_payload(
                b"{}\n",
                "jobs/1/terminal.json",
                role="terminal-job-record",
                entry_id="entry_terminal_1",
            )
            writer.write_payload(
                b"{}\n",
                "sequence/summary.json",
                role="sequence-summary",
                entry_id="entry_sequence_summary",
            )
            writer.write_payload(
                b"{}\n",
                "events/events.jsonl",
                role="event-ledger",
                entry_id="entry_sequence_ledger",
            )

            with self.assertRaisesRegex(
                EvidenceStorageError, "one terminal or last-observed"
            ):
                writer.seal()

    def test_managed_admission_refusal_is_complete_without_a_job_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = RawArtifactWriter(
                vault / "managed-refusal",
                protected_artifact_id=ARTIFACT_ID,
                record_kind="experiment-run",
                identity_bindings={
                    "attempt_slot_id": SLOT_ID,
                    "experiment_run_id": RUN_ID,
                    "capture_kind": "managed-job",
                    "capture_status": "complete",
                },
                capture_tool={"name": "storage-test", "version": "v1"},
                source_bindings={"source_commit": "1" * 40},
                provisional_retain_not_before_utc=RETAIN_UNTIL,
                required_role_bindings={
                    "action-submission-record": "entry_submission",
                    "sequence-summary": "entry_sequence_summary",
                    "event-ledger": "entry_sequence_ledger",
                },
            )
            for payload, path, role, entry_id in (
                (
                    b"{}\n",
                    "job/submission.json",
                    "action-submission-record",
                    "entry_submission",
                ),
                (
                    b"{}\n",
                    "sequence/summary.json",
                    "sequence-summary",
                    "entry_sequence_summary",
                ),
                (
                    b"{}\n",
                    "events/events.jsonl",
                    "event-ledger",
                    "entry_sequence_ledger",
                ),
            ):
                writer.write_payload(payload, path, role=role, entry_id=entry_id)

            result = writer.seal()

            self.assertEqual(result["file_count"], 3)

    def test_protocol_valid_managed_run_is_semantically_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = protocol_managed_writer(vault, "protocol-valid")

            result = writer.seal()

            self.assertEqual(result["file_count"], 43)
            self.assertEqual(
                result["manifest"]["required_role_bindings"]["experiment-run-record"],
                "entry_protocol_run",
            )
            self.assertEqual(
                set(result["manifest"]["required_role_bindings"])
                & REQUIRED_QUALIFYING_ARTIFACT_ROLES,
                REQUIRED_QUALIFYING_ARTIFACT_ROLES,
            )
            self.assertEqual(
                len(
                    result["manifest"]["required_role_bindings"][
                        "runtime-boundary-journal"
                    ]
                ),
                2,
            )
            with patch.object(
                storage,
                "validate_retained_phase4_source_freeze",
                _validate_retained_phase4_source_freeze_for_test,
            ):
                self.assertEqual(
                    verify_sealed_artifact(writer.directory)["raw_manifest_sha256"],
                    result["raw_manifest_sha256"],
                )

    def test_all_seven_native_outcomes_seal_and_deep_verify(self) -> None:
        cases = (
            ("passed", "NONE", None, 0),
            ("refused", "APTUS_ADMISSION_REFUSAL", "preflight", 0),
            ("guard-blocked", "HOST_RAM_FLOOR", "dependency", 0),
            ("failed", "PROCESS_EXIT_NONZERO", "pilot", 1),
            ("cancelled", "CUDA_XID", "train", 5),
            (
                "timed-out",
                "EMERGENCY_DEADLINE_EXCEEDED",
                "train",
                5,
            ),
            ("unknown", "OWNERSHIP_UNCERTAIN", "dependency", 0),
        )
        for outcome, reason, stop_action, runtime_count in cases:
            with (
                self.subTest(outcome=outcome),
                tempfile.TemporaryDirectory() as temporary,
            ):
                vault = private_directory(Path(temporary) / "vault")
                if outcome == "passed":
                    writer = protocol_managed_writer(vault, outcome)
                else:
                    summary, ledger, telemetry = protocol_negative_outcome_case(
                        outcome,
                        reason,
                        stop_action=stop_action or "preflight",
                        runtime_count=runtime_count,
                    )
                    writer = protocol_managed_writer(
                        vault,
                        outcome,
                        summary=summary,
                        ledger=ledger,
                        telemetry=telemetry,
                        outcome_fixture=True,
                    )

                result = writer.seal()

                with patch.object(
                    storage,
                    "validate_retained_phase4_source_freeze",
                    _validate_retained_phase4_source_freeze_for_test,
                ):
                    verified = verify_sealed_artifact(writer.directory)
                self.assertEqual(
                    verified["raw_manifest_sha256"],
                    result["raw_manifest_sha256"],
                )
                run_record = json.loads(
                    (writer.directory / "identity/experiment-run.json").read_text(
                        encoding="utf-8"
                    )
                )
                terminal = run_record["terminal_evidence"]
                self.assertEqual(terminal["native_outcome"], outcome)
                self.assertEqual(terminal["evidence_status"], "protocol-valid")
                self.assertEqual(terminal["reason_code"], reason)
                self.assertEqual(
                    is_publication_eligible(outcome, "protocol-valid"),
                    outcome == "passed",
                )
                if outcome != "passed":
                    self.assertNotIn(
                        "cooldown-summary",
                        result["manifest"]["required_role_bindings"],
                    )

    def test_retained_activation_rejects_tamper_omission_and_role_swap(self) -> None:
        tampered = protocol_activation_authority()
        tampered["activation-decision"] = (
            _artifact_payload_bytes(tampered["activation-decision"]) + b" "
        )
        omitted = protocol_activation_authority()
        omitted.pop("activation-seal")
        swapped = protocol_activation_authority()
        (
            swapped["activation-decision"],
            swapped["activation-started-identity-template"],
        ) = (
            swapped["activation-started-identity-template"],
            swapped["activation-decision"],
        )

        for name, activation_values in (
            ("tampered", tampered),
            ("omitted", omitted),
            ("role-swapped", swapped),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                vault = private_directory(Path(temporary) / "vault")
                writer = protocol_managed_writer(
                    vault,
                    f"activation-{name}",
                    activation_authority=activation_values,
                )

                with self.assertRaisesRegex(
                    EvidenceStorageError, "activation|Activation|role"
                ):
                    writer.seal()

        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = protocol_managed_writer(vault, "activation-authority-bridge")
            writer.source_bindings["idle_baseline_binding_sha256"] = "0" * 64

            with self.assertRaisesRegex(
                EvidenceStorageError,
                "retained campaign or Phase-4 authority",
            ):
                writer.seal()

    def test_protocol_valid_run_rejects_empty_core_payloads(self) -> None:
        cases = (
            ("ledger", {"ledger": [{}]}),
            ("telemetry", {"telemetry": [{}]}),
            ("summary", {"summary": {}}),
            ("terminal", {"terminal": {}}),
            ("experiment-record", {"experiment_record": {}}),
            ("attempt-record", {"attempt_record": {}}),
            ("execution-record", {"execution_record": {}}),
            ("idle-baseline", {"idle_baseline": {}}),
            ("telemetry-configuration", {"telemetry_configuration": {}}),
            ("telemetry-summary", {"telemetry_summary": {}}),
            ("cooldown-summary", {"cooldown_summary": {}}),
        )
        for name, overrides in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                vault = private_directory(Path(temporary) / "vault")
                writer = protocol_managed_writer(vault, name, **overrides)

                with self.assertRaises(EvidenceStorageError):
                    writer.seal()

                self.assertFalse((writer.directory / "raw-manifest.json").exists())

    def test_protocol_valid_run_rejects_cross_identity_spoofing(self) -> None:
        wrong_job_id = "job_" + "2" * 32
        wrong_run_id = "run_" + "3" * 32
        cases: list[tuple[str, dict[str, object]]] = []

        wrong_summary_run = protocol_sequence_summary()
        wrong_summary_run["experiment_run_id"] = OTHER_RUN_ID
        cases.append(("summary-run", {"summary": wrong_summary_run}))

        wrong_summary_job = protocol_sequence_summary()
        wrong_summary_job["started_actions"][0]["job_id"] = wrong_job_id
        cases.append(("summary-job", {"summary": wrong_summary_job}))

        wrong_summary_action = protocol_sequence_summary()
        wrong_summary_action["configured_actions"][0]["action"] = "pilot"
        wrong_summary_action["started_actions"][0]["action"] = "pilot"
        cases.append(("summary-action", {"summary": wrong_summary_action}))

        wrong_terminal_job = {
            "id": wrong_job_id,
            "job_id": wrong_job_id,
            "action": "dependency",
            "state": "completed",
            "run_id": None,
        }
        cases.append(("terminal-job", {"terminal": wrong_terminal_job}))

        wrong_terminal_action = {
            "id": JOB_ID,
            "job_id": JOB_ID,
            "action": "pilot",
            "state": "completed",
            "run_id": None,
        }
        cases.append(("terminal-action", {"terminal": wrong_terminal_action}))

        wrong_record_slot = protocol_experiment_run_record()
        wrong_record_slot["attempt_slot_id"] = "slot_" + "4" * 20
        cases.append(("run-record-slot", {"experiment_record": wrong_record_slot}))

        wrong_record_config = protocol_experiment_run_record()
        wrong_record_config["execution_configuration_id"] = "exec_" + "5" * 20
        cases.append(
            ("run-record-configuration", {"experiment_record": wrong_record_config})
        )

        wrong_record_job = protocol_experiment_run_record()
        wrong_record_job["aptus_job_ids"] = [wrong_job_id]
        cases.append(("run-record-job", {"experiment_record": wrong_record_job}))

        wrong_record_run = protocol_experiment_run_record()
        wrong_record_run["aptus_run_ids"] = [wrong_run_id]
        cases.append(("run-record-run", {"experiment_record": wrong_record_run}))

        cases.append(
            (
                "ledger-run",
                {"ledger": protocol_event_records(experiment_run_id=OTHER_RUN_ID)},
            )
        )
        wrong_ledger_job = protocol_event_records()
        dependency_state = next(
            row
            for row in wrong_ledger_job
            if row["event_type"] == "job.state-observed"
            and row["action"] == "dependency"
        )
        dependency_state["subject_id"] = wrong_job_id
        cases.append(("ledger-job", {"ledger": wrong_ledger_job}))
        cases.append(
            (
                "telemetry-run",
                {
                    "telemetry": protocol_telemetry_records(
                        experiment_run_id=OTHER_RUN_ID
                    )
                },
            )
        )

        for name, overrides in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                vault = private_directory(Path(temporary) / "vault")
                writer = protocol_managed_writer(vault, name, **overrides)

                with self.assertRaises(EvidenceStorageError):
                    writer.seal()

    def test_protocol_valid_run_rejects_swapped_action_paths(self) -> None:
        summary = protocol_sequence_summary()
        summary["configured_actions"][0]["label"] = "other-action"
        summary["started_actions"][0]["label"] = "other-action"
        ledger = protocol_event_records()
        for row in ledger:
            if row["phase"] == "dependency":
                row["phase"] = "other-action"
            if row["subject_kind"] == "managed-action" and row["subject_id"] == (
                "dependency"
            ):
                row["subject_id"] = "other-action"
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = protocol_managed_writer(
                vault,
                "swapped-label",
                summary=summary,
                ledger=ledger,
                terminal_label="dependency",
            )

            with self.assertRaisesRegex(EvidenceStorageError, "exact terminal"):
                writer.seal()

    def test_protocol_valid_run_rejects_telemetry_sequence_and_schedule_spoofs(
        self,
    ) -> None:
        wrong_sequence = protocol_telemetry_records()
        wrong_sequence[1]["sequence"] = 2
        wrong_schedule = protocol_telemetry_records()
        wrong_schedule[1]["scheduled_monotonic_ns"] = 1_500_000_000
        cases = (("sequence", wrong_sequence), ("schedule", wrong_schedule))
        for name, telemetry in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                vault = private_directory(Path(temporary) / "vault")
                writer = protocol_managed_writer(
                    vault, f"telemetry-{name}", telemetry=telemetry
                )

                with self.assertRaises(EvidenceStorageError):
                    writer.seal()

    def test_qualifying_run_requires_frozen_internal_runtime_boundaries(self) -> None:
        swapped_phases = protocol_event_records()
        for row in swapped_phases:
            if row["phase"] == "pilot-phase-1":
                row["phase"] = "pilot-phase-2"
            elif row["phase"] == "pilot-phase-2":
                row["phase"] = "pilot-phase-1"

        mislabeled_command = protocol_event_records()
        pilot_command = next(
            row
            for row in mislabeled_command
            if row["event_type"] == "command.started" and row["action"] == "pilot"
        )
        pilot_command["phase"] = "pilot-phase-1"

        wrong_runtime_job = protocol_event_records()
        pilot_runtime = next(
            row
            for row in wrong_runtime_job
            if row["event_type"] == "pilot.phase-started"
        )
        pilot_runtime["subject_id"] = QUALIFYING_JOB_IDS[-1]

        for name, ledger in (
            ("swapped-phases", swapped_phases),
            ("mislabeled-command", mislabeled_command),
            ("wrong-runtime-job", wrong_runtime_job),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                vault = private_directory(Path(temporary) / "vault")
                writer = protocol_managed_writer(vault, name, ledger=ledger)

                with self.assertRaises(EvidenceStorageError):
                    writer.seal()

    def test_qualifying_runtime_journals_are_exact_and_match_the_ledger(self) -> None:
        mutated = protocol_runtime_journal_records()
        mutated["pilot"][0]["wall_time_utc"] = "2026-08-08T12:00:01+00:00"
        omitted = protocol_runtime_journal_records()
        omitted["train"].pop()
        extra = protocol_runtime_journal_records()
        extra["pilot"].append(dict(extra["pilot"][-1]))
        reordered = protocol_runtime_journal_records()
        reordered["pilot"][0], reordered["pilot"][1] = (
            reordered["pilot"][1],
            reordered["pilot"][0],
        )
        missing_journal = protocol_runtime_journal_records()
        missing_journal.pop("train")

        for name, journals in (
            ("mutated", mutated),
            ("omitted-boundary", omitted),
            ("extra-boundary", extra),
            ("reordered", reordered),
            ("missing-journal", missing_journal),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                vault = private_directory(Path(temporary) / "vault")
                writer = protocol_managed_writer(
                    vault,
                    f"runtime-journal-{name}",
                    runtime_journals=journals,
                )

                with self.assertRaisesRegex(EvidenceStorageError, "runtime|journal"):
                    writer.seal()

    def test_qualifying_run_requires_every_single_semantic_role_binding(self) -> None:
        for role in (
            "attempt-slot-record",
            "execution-configuration-record",
            "idle-baseline-binding",
            "telemetry-configuration",
            "telemetry-summary",
            "cooldown-summary",
            *sorted(REQUIRED_QUALIFYING_ARTIFACT_ROLES),
        ):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                vault = private_directory(Path(temporary) / "vault")
                writer = protocol_managed_writer(vault, role)
                writer.required_role_bindings.pop(role)

                with self.assertRaisesRegex(EvidenceStorageError, role):
                    writer.seal()

    def test_qualifying_artifact_roles_reject_omission_and_substitution(self) -> None:
        omitted = protocol_required_artifact_payloads()
        omitted.pop("pilot-metrics")
        substituted = protocol_required_artifact_payloads()
        substituted["plan-substitute"] = substituted.pop("plan")

        for name, artifacts, missing_role in (
            ("omitted", omitted, "pilot-metrics"),
            ("substituted", substituted, "plan"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                vault = private_directory(Path(temporary) / "vault")
                writer = protocol_managed_writer(
                    vault,
                    f"artifact-role-{name}",
                    required_artifacts=artifacts,
                )

                with self.assertRaisesRegex(EvidenceStorageError, missing_role):
                    writer.seal()

    def test_cooldown_is_recomputed_from_retained_idle_baseline(self) -> None:
        hostile_baseline = protocol_idle_baseline_binding()
        hostile_baseline["summary"].update(
            gpu_temperature_median_c=30.0,
            gpu_temperature_p95_c=31.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = protocol_managed_writer(
                vault, "baseline-relative-failure", idle_baseline=hostile_baseline
            )

            with self.assertRaisesRegex(
                EvidenceStorageError,
                "Activation provenance|activated template|Phase-4 source",
            ):
                writer.seal()

        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = protocol_managed_writer(vault, "baseline-source-digest")
            writer.source_bindings["idle_baseline_binding_sha256"] = "0" * 64

            with self.assertRaisesRegex(
                EvidenceStorageError,
                "Activation provenance|activated template|run provenance",
            ):
                writer.seal()

        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            run = protocol_experiment_run_record()
            run["observed_host_state"]["idle_baseline_sha256"] = "0" * 64
            writer = protocol_managed_writer(
                vault, "baseline-run-digest", experiment_record=run
            )

            with self.assertRaisesRegex(
                EvidenceStorageError, "activated template|run provenance"
            ):
                writer.seal()

    def test_qualifying_run_rejects_digest_and_derived_summary_spoofs(self) -> None:
        telemetry_summary = protocol_telemetry_summary(
            protocol_telemetry_records(), protocol_event_records()
        )
        telemetry_summary["telemetry"]["coverage"] = 0.5
        cooldown_summary = protocol_cooldown_summary(protocol_telemetry_records())
        cooldown_summary["summary"]["sample_count"] = 119
        attempt_record = protocol_attempt_slot_record()
        attempt_record["execution_configuration_id"] = "exec_" + "5" * 20
        experiment_record = protocol_experiment_run_record()
        experiment_record["terminal_evidence"]["evidence_role_sha256"][
            "telemetry-summary"
        ] = "0" * 64
        cases = (
            ("telemetry-summary", {"telemetry_summary": telemetry_summary}),
            ("cooldown-summary", {"cooldown_summary": cooldown_summary}),
            ("attempt-record", {"attempt_record": attempt_record}),
            ("embedded-digest", {"experiment_record": experiment_record}),
        )
        for name, overrides in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                vault = private_directory(Path(temporary) / "vault")
                writer = protocol_managed_writer(vault, name, **overrides)

                with self.assertRaises(EvidenceStorageError):
                    writer.seal()

        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = protocol_managed_writer(vault, "manifest-digest")
            writer.source_bindings["evidence_role_sha256"]["telemetry-summary"] = (
                "0" * 64
            )

            with self.assertRaisesRegex(EvidenceStorageError, "source bindings"):
                writer.seal()

        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = protocol_managed_writer(vault, "selected-artifact-digest")
            writer.source_bindings["evidence_role_sha256"]["plan"] = "0" * 64

            with self.assertRaisesRegex(EvidenceStorageError, "source bindings"):
                writer.seal()

        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            run = protocol_experiment_run_record()
            run["bundle_manifest_sha256"] = "0" * 64
            writer = protocol_managed_writer(
                vault, "bundle-manifest-semantic-digest", experiment_record=run
            )

            with self.assertRaisesRegex(
                EvidenceStorageError, "activated template|bundle manifest"
            ):
                writer.seal()

    def test_protocol_valid_run_requires_bound_experiment_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = protocol_managed_writer(
                vault, "missing-run-record", include_experiment_record=False
            )

            with self.assertRaisesRegex(EvidenceStorageError, "experiment-run-record"):
                writer.seal()

    def test_capture_invalid_core_payloads_remain_syntactically_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = protocol_managed_writer(
                vault,
                "capture-invalid-empty",
                ledger=[{}],
                telemetry=[{}],
                summary={},
                terminal={},
                experiment_record={},
            )
            writer.identity_bindings.update(
                evidence_status="capture-invalid",
                capture_reason_code="MISSING_REQUIRED_EVIDENCE",
            )

            result = writer.seal()

            self.assertEqual(result["file_count"], 43)

        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            writer = RawArtifactWriter(
                vault / "capture-invalid-noncanonical",
                protected_artifact_id=ARTIFACT_ID,
                record_kind="experiment-run",
                identity_bindings={
                    "attempt_slot_id": SLOT_ID,
                    "experiment_run_id": RUN_ID,
                    "capture_kind": "command",
                    "capture_status": "complete",
                    "evidence_status": "capture-invalid",
                    "capture_reason_code": "MISSING_REQUIRED_EVIDENCE",
                },
                capture_tool={"name": "storage-test", "version": "v1"},
                source_bindings={"source_commit": "1" * 40},
                provisional_retain_not_before_utc=RETAIN_UNTIL,
                required_role_bindings={
                    "command-record": "entry_noncanonical_command",
                    "command-output": "entry_noncanonical_output",
                    "event-ledger": "entry_noncanonical_ledger",
                },
            )
            writer.write_payload(
                b"{ }\n",
                "command/record.json",
                role="command-record",
                entry_id="entry_noncanonical_command",
            )
            writer.write_payload(
                b"output\n",
                "command/output.bin",
                role="command-output",
                entry_id="entry_noncanonical_output",
            )
            writer.write_payload(
                b"{}\n",
                "events/events.jsonl",
                role="event-ledger",
                entry_id="entry_noncanonical_ledger",
            )

            with self.assertRaisesRegex(EvidenceStorageError, "not canonical"):
                writer.seal()

    def test_concurrent_duplicate_experiment_run_seal_has_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            first = command_writer(vault, "first", artifact_id="artifact_" + "1" * 32)
            second = command_writer(vault, "second", artifact_id="artifact_" + "2" * 32)
            barrier = threading.Barrier(2)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(
                        lambda writer: self._seal_or_error(writer, barrier),
                        (first, second),
                    )
                )

            self.assertEqual(sum(result == "sealed" for result in results), 1)
            self.assertEqual(sum(result == "duplicate" for result in results), 1)
            self.assertEqual(
                sum(
                    (path / "SEALED.json").is_file()
                    for path in (first.directory, second.directory)
                ),
                1,
            )
            lock = vault / ".experiment-run-seals.lock"
            self.assertEqual(stat.S_IMODE(lock.stat().st_mode), 0o600)

    def test_replaced_seal_lock_cannot_split_cross_process_authority(self) -> None:
        if os.name != "posix" or storage.fcntl is None:
            self.skipTest("POSIX process locks are required")
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("A fork process context is required")

        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            context = multiprocessing.get_context("fork")
            started_first = context.Event()
            entered_first = context.Event()
            release_first = context.Event()
            started_second = context.Event()
            entered_second = context.Event()
            release_second = context.Event()
            results = context.Queue()
            first = context.Process(
                target=_seal_with_process_gate,
                args=(
                    str(vault),
                    "first-process",
                    "artifact_" + "1" * 32,
                    started_first,
                    entered_first,
                    release_first,
                    results,
                ),
            )
            second = context.Process(
                target=_seal_with_process_gate,
                args=(
                    str(vault),
                    "second-process",
                    "artifact_" + "2" * 32,
                    started_second,
                    entered_second,
                    release_second,
                    results,
                ),
            )
            first.start()
            try:
                self.assertTrue(started_first.wait(timeout=5))
                self.assertTrue(entered_first.wait(timeout=5))
                lock = vault / ".experiment-run-seals.lock"
                displaced_lock = vault / ".displaced-experiment-seal-lock"
                lock.rename(displaced_lock)
                replacement_descriptor = os.open(
                    lock,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                os.fchmod(replacement_descriptor, 0o600)
                os.close(replacement_descriptor)
                storage._fsync_directory(vault)

                second.start()
                self.assertTrue(started_second.wait(timeout=5))
                self.assertFalse(entered_second.wait(timeout=0.5))
            finally:
                release_first.set()
                release_second.set()
                first.join(timeout=10)
                if second.pid is not None:
                    second.join(timeout=10)
                for process in (first, second):
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)

            self.assertEqual(first.exitcode, 0)
            self.assertEqual(second.exitcode, 0)
            outcomes = [results.get(timeout=5), results.get(timeout=5)]
            self.assertEqual(sum(item[0] == "sealed" for item in outcomes), 1)
            self.assertEqual(sum(item[0] == "error" for item in outcomes), 1)
            sealed_directories = [
                path
                for path in (
                    vault / "first-process",
                    vault / "second-process",
                )
                if (path / "SEALED.json").is_file()
            ]
            self.assertEqual(len(sealed_directories), 1)
            verify_sealed_artifact(sealed_directories[0])

    def test_experiment_seal_rejects_symlink_or_hardlinked_lock(self) -> None:
        if os.name != "posix":
            self.skipTest("POSIX link protections are not available")
        for attack in ("symlink", "hardlink"):
            with self.subTest(attack=attack):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    vault = private_directory(root / "vault")
                    target = root / "outside.lock"
                    target.write_bytes(b"")
                    target.chmod(0o600)
                    lock = vault / ".experiment-run-seals.lock"
                    if attack == "symlink":
                        lock.symlink_to(target)
                    else:
                        os.link(target, lock)
                    writer = command_writer(vault, "attacked")

                    with self.assertRaises(EvidenceStorageError):
                        writer.seal()

                    self.assertFalse((writer.directory / "raw-manifest.json").exists())
                    self.assertEqual(target.read_bytes(), b"")

    @staticmethod
    def _seal_or_error(writer: RawArtifactWriter, barrier: threading.Barrier) -> str:
        barrier.wait()
        try:
            writer.seal()
        except EvidenceStorageError as error:
            if "already has a sealed artifact" not in str(error):
                raise
            return "duplicate"
        return "sealed"

    def test_failed_attempt_may_create_one_sealed_fallback_but_not_reseal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vault = private_directory(Path(temporary) / "vault")
            normal = command_writer(vault, "normal")
            original = storage._write_exclusive_control_bytes

            def interrupt(
                path: Path, payload: bytes, *, mode: int = 0o600
            ) -> tuple[int, int]:
                if path.parent == normal.directory and path.name == "SEALED.json":
                    raise OSError("simulated interruption")
                return original(path, payload, mode=mode)

            with patch.object(
                storage, "_write_exclusive_control_bytes", side_effect=interrupt
            ):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    normal.seal()

            fallback = vault / "normal.capture-failure"
            result = write_sealed_capture_failure_artifact(
                fallback,
                protected_artifact_id=ARTIFACT_ID,
                attempt_slot_id=SLOT_ID,
                experiment_run_id=RUN_ID,
                reason_code="SEAL_FAILURE",
                available_files=[],
                missing_fields=["SEALED.json"],
                recoverable_locator="opaque-private-locator-1",
                capture_tool={"name": "storage-test", "version": "v1"},
                source_bindings={"source_commit": "1" * 40},
                provisional_retain_not_before_utc=RETAIN_UNTIL,
            )
            self.assertEqual(
                result["manifest"]["identity_bindings"]["capture_status"], "failed"
            )
            self.assertTrue((fallback / "SEALED.json").is_file())

            retry = command_writer(
                vault,
                "retry",
                artifact_id="artifact_" + "3" * 32,
            )
            with self.assertRaisesRegex(
                EvidenceStorageError, "already has a sealed artifact"
            ):
                retry.seal()


class CaptureFailureAndReceiptTests(unittest.TestCase):
    def test_capture_failure_receipt_is_canonical_private_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fallback = private_directory(root / "fallback")
            path = fallback / "capture-failure.json"
            available = [
                {
                    "entry_id": ENTRY_ID,
                    "role": "job-log",
                    "relative_path": "jobs/job.log",
                    "media_type": "text/plain",
                    "size_bytes": 4,
                    "sha256": storage.sha256_bytes(b"log\n"),
                    "captured_at_utc": "2026-08-08T12:00:00+00:00",
                }
            ]

            receipt = write_capture_failure_receipt(
                path,
                protected_artifact_id=ARTIFACT_ID,
                attempt_slot_id=SLOT_ID,
                experiment_run_id=RUN_ID,
                reason_code="SEAL_FAILURE",
                available_files=available,
                missing_fields=["seal", "raw_manifest"],
                recoverable_locator="opaque-private-locator-1",
            )

            self.assertEqual(path.read_bytes(), storage.canonical_json_bytes(receipt))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                write_capture_failure_receipt(
                    path,
                    protected_artifact_id=ARTIFACT_ID,
                    attempt_slot_id=SLOT_ID,
                    experiment_run_id=RUN_ID,
                    reason_code="SEAL_FAILURE",
                    available_files=available,
                    missing_fields=["seal"],
                    recoverable_locator=None,
                )

    def test_capture_failure_fallback_is_sealed_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fallback = private_directory(root / "fallback")
            result = write_sealed_capture_failure_artifact(
                fallback / "failed-run",
                protected_artifact_id=ARTIFACT_ID,
                attempt_slot_id=SLOT_ID,
                experiment_run_id=RUN_ID,
                reason_code="SEAL_FAILURE",
                available_files=[],
                missing_fields=["raw-manifest", "terminal-job-record"],
                recoverable_locator=None,
                capture_tool={"name": "storage-test", "version": "v1"},
                source_bindings={"source_commit": "1" * 40},
                provisional_retain_not_before_utc=RETAIN_UNTIL,
            )
            artifact = fallback / "failed-run"
            self.assertEqual(result["receipt"]["reason_code"], "SEAL_FAILURE")
            self.assertEqual(
                verify_sealed_artifact(artifact)["raw_manifest_sha256"],
                result["raw_manifest_sha256"],
            )
            (artifact / "capture-failure.json").write_bytes(b"{}\n")
            (artifact / "capture-failure.json").chmod(0o600)
            with self.assertRaises(ArtifactIntegrityError):
                verify_sealed_artifact(artifact)

    def test_receipts_form_one_append_only_chain_and_reject_a_fork(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts_root = private_directory(root / "receipts")
            store = AppendOnlyReceiptStore(receipts_root)
            digest = "a" * 64
            first = store.append(
                kind="copy-verification",
                issuer_role_id="evidence-custodian",
                protected_artifact_id=ARTIFACT_ID,
                raw_manifest_sha256=digest,
                raw_manifest_size_bytes=100,
                result="passed",
                details={"copy_id": COPY_ID},
            )
            second = store.append(
                kind="retention",
                issuer_role_id="evidence-custodian",
                protected_artifact_id=ARTIFACT_ID,
                raw_manifest_sha256=digest,
                raw_manifest_size_bytes=100,
                result="active",
                details={"retain_not_before_utc": RETAIN_UNTIL},
            )

            self.assertIsNone(first["previous_receipt_id"])
            self.assertEqual(second["previous_receipt_id"], first["receipt_id"])
            self.assertEqual(
                [item["receipt_id"] for item in store.read_chain()],
                [first["receipt_id"], second["receipt_id"]],
            )
            for receipt_id in (first["receipt_id"], second["receipt_id"]):
                self.assertEqual(
                    stat.S_IMODE((receipts_root / f"{receipt_id}.json").stat().st_mode),
                    0o600,
                )
            with self.assertRaises(ReceiptChainError):
                store.append(
                    kind="renewal",
                    issuer_role_id="evidence-custodian",
                    protected_artifact_id=ARTIFACT_ID,
                    raw_manifest_sha256=digest,
                    raw_manifest_size_bytes=100,
                    result="active",
                    details={},
                    previous_receipt_id=first["receipt_id"],
                )

            second_path = receipts_root / f"{second['receipt_id']}.json"
            changed = storage.json.loads(second_path.read_text(encoding="utf-8"))
            changed["result"] = "changed"
            second_path.write_bytes(storage.canonical_json_bytes(changed))
            second_path.chmod(0o600)
            with self.assertRaisesRegex(ReceiptChainError, "cryptographic ID"):
                store.read_chain()

    def test_duplicate_receipt_cannot_overwrite_or_fork_the_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = AppendOnlyReceiptStore(private_directory(root / "receipts"))
            common = {
                "kind": "retention",
                "issuer_role_id": "evidence-custodian",
                "protected_artifact_id": ARTIFACT_ID,
                "raw_manifest_sha256": "a" * 64,
                "raw_manifest_size_bytes": 100,
                "result": "active",
                "details": {},
                "created_at_utc": "2026-08-08T12:00:00+00:00",
            }
            first = store.append(**common)
            with self.assertRaises(ReceiptChainError):
                store.append(**common, previous_receipt_id=None)
            with self.assertRaisesRegex(EvidenceStorageError, "content digest"):
                store.append(**common, receipt_id=RECEIPT_ONE)
            self.assertRegex(first["receipt_id"], r"^receipt_[0-9a-f]{32}$")


class RetrievalAndRetentionTests(unittest.TestCase):
    def test_retrieval_is_a_full_restore_into_a_fresh_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = private_directory(root / "source-vault")
            restore_root = private_directory(root / "restore-vault")
            source, source_result = sealed_artifact(source_root)
            destination = restore_root / "restored-artifact"

            details = retrieve_sealed_artifact(
                source,
                destination,
                source_copy_id=COPY_ID,
                source_failure_domain_id=DOMAIN_ID,
                expected_raw_manifest_sha256=source_result["raw_manifest_sha256"],
            )

            self.assertEqual(details["verification_result"], "passed")
            self.assertEqual(details["mismatch_count"], 0)
            self.assertGreaterEqual(details["duration_ns"], 0)
            self.assertEqual(details["restored_file_count"], 3)
            self.assertEqual(
                verify_copy_equality(source, destination)["verification_result"],
                "passed",
            )
            with self.assertRaises(FileExistsError):
                retrieve_sealed_artifact(
                    source,
                    destination,
                    source_copy_id=COPY_ID,
                    source_failure_domain_id=DOMAIN_ID,
                    expected_raw_manifest_sha256=source_result["raw_manifest_sha256"],
                )
            with self.assertRaises(FileExistsError):
                retrieve_sealed_artifact(
                    source,
                    source,
                    source_copy_id=COPY_ID,
                    source_failure_domain_id=DOMAIN_ID,
                    expected_raw_manifest_sha256=source_result["raw_manifest_sha256"],
                )

    def test_retrieval_mismatch_fails_closed_with_a_receiptable_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = private_directory(root / "source-vault")
            restore_root = private_directory(root / "restore-vault")
            source, _ = sealed_artifact(source_root)

            with self.assertRaises(RetrievalError) as raised:
                retrieve_sealed_artifact(
                    source,
                    restore_root / "restored-artifact",
                    source_copy_id=COPY_ID,
                    source_failure_domain_id=DOMAIN_ID,
                    expected_raw_manifest_sha256="0" * 64,
                )

            self.assertEqual(raised.exception.details["verification_result"], "failed")
            self.assertEqual(raised.exception.details["mismatch_count"], 1)
            self.assertEqual(
                raised.exception.details["observed_raw_manifest_sha256"],
                verify_sealed_artifact(source)["raw_manifest_sha256"],
            )

    def test_calendar_month_clamp_and_retention_gates(self) -> None:
        self.assertEqual(
            add_calendar_months_utc("2024-01-31T12:00:00+00:00", 1),
            "2024-02-29T12:00:00+00:00",
        )
        self.assertEqual(
            retention_deadline_utc("2024-02-29T12:00:00+00:00"),
            "2026-02-28T12:00:00+00:00",
        )
        now = datetime(2026, 8, 8, 12, tzinfo=timezone.utc)
        state = evaluate_retention_state(
            now_utc=now,
            retain_not_before_utc=now + timedelta(days=365),
            dependent_claims_active=True,
            verified_copy_count=2,
            failure_domain_count=2,
            off_host_copy_count=1,
            last_copy_verification_utc=now - timedelta(days=89),
            last_off_host_retrieval_utc=now - timedelta(days=179),
            copy_verification_passed=True,
            retrieval_passed=True,
        )
        self.assertTrue(state["claim_qualified"])
        self.assertFalse(state["claim_suspended"])
        self.assertFalse(state["deletion_allowed"])

        expired = evaluate_retention_state(
            now_utc=now,
            retain_not_before_utc=now - timedelta(seconds=1),
            dependent_claims_active=False,
            verified_copy_count=1,
            failure_domain_count=1,
            off_host_copy_count=0,
            last_copy_verification_utc=now - timedelta(days=91),
            last_off_host_retrieval_utc=now - timedelta(days=181),
            copy_verification_passed=False,
            retrieval_passed=False,
        )
        self.assertTrue(expired["claim_suspended"])
        self.assertTrue(expired["deletion_allowed"])
        self.assertIn("INSUFFICIENT_COPY_COUNT", expired["suspension_reasons"])
        self.assertIn("OFF_HOST_RETRIEVAL_NOT_CURRENT", expired["suspension_reasons"])


if __name__ == "__main__":
    unittest.main()
