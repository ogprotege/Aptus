from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from aptus.execution import JobSubmissionFailure

from tools.cuda_campaign.contracts import (
    EventLedgerWriter,
    canonical_jsonl_bytes,
    validate_event_ledger,
    validate_record,
)
from tools.cuda_campaign.harness import (
    CANCEL_REQUEST_SLA_NS,
    CancellationSLAError,
    CaptureHarness,
    CaptureHarnessError,
    ManagedActionSpec,
    SafetySignal,
    SelectedArtifact,
    SubmissionBlockedError,
    TelemetryCapture,
    verify_cancellation_milestones,
)
from tools.cuda_campaign.monitoring import (
    GIB,
    LinuxNvidiaHostProbe,
    ProbeFailure,
    construct_telemetry_sample,
    resolve_trusted_nvidia_smi,
)
from tools.cuda_campaign.phase4 import Phase4SourceFreezeVerification
from tools.cuda_campaign.qualification import QualifyingRunContext
from tools.cuda_campaign.sidecar import (
    BackgroundTelemetrySession,
    SidecarTelemetryCapture,
    TelemetrySnapshot,
)
from tools.cuda_campaign.storage import RawArtifactWriter, verify_sealed_artifact


SLOT_ID = "slot_" + "a" * 20
RUN_ID = "xrun_" + "b" * 32
JOB_ID = "job_" + "c" * 32
RETAIN_UNTIL = "2028-08-08T12:00:00+00:00"
WALL_TIME = "2026-08-08T12:00:00+00:00"


def private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def harness_at(root: Path, *, job_service: Any = None) -> CaptureHarness:
    return CaptureHarness(
        private_directory(root / "state"),
        attempt_slot_id=SLOT_ID,
        experiment_run_id=RUN_ID,
        provisional_retain_not_before_utc=RETAIN_UNTIL,
        job_service=job_service,
        source_bindings={"source_commit": "1" * 40},
    )


def fake_phase4_verification(
    root: Path,
    context: QualifyingRunContext,
    *,
    telemetry_configuration: dict[str, Any] | None = None,
) -> Phase4SourceFreezeVerification:
    baseline = dict(context.idle_baseline_binding)
    return Phase4SourceFreezeVerification(
        directory=root,
        source_freeze={"telemetry_configuration": dict(telemetry_configuration or {})},
        seal={},
        baseline_binding=baseline,
        source_freeze_sha256=baseline["phase4_source_freeze_sha256"],
        seal_sha256=baseline["phase4_source_freeze_seal_sha256"],
        samples_sha256=baseline["idle_baseline_samples_sha256"],
    )


def qualifying_harness_from_context(
    root: Path,
    context: QualifyingRunContext,
    verification: Phase4SourceFreezeVerification,
) -> CaptureHarness:
    planned = context.planned_slot_context
    activated = context.verified_activation
    if planned is None or activated is None:
        raise AssertionError("test qualifying context lacks activation authority")
    admitted = SimpleNamespace(admitted=True, decision={"reason_codes": []})
    with (
        patch(
            "tools.cuda_campaign.harness.collect_production_admission_observations",
            return_value=object(),
        ),
        patch(
            "tools.cuda_campaign.harness.evaluate_pre_slot_admission",
            return_value=admitted,
        ),
        patch("tools.cuda_campaign.harness.activate_admitted_slot"),
        patch(
            "tools.cuda_campaign.harness.verify_activated_slot",
            return_value=activated,
        ),
        patch(
            "tools.cuda_campaign.harness.verify_phase4_source_freeze_artifact",
            return_value=verification,
        ),
    ):
        return CaptureHarness.for_qualifying_campaign(
            Path(planned.run_proposal.fresh_state_root),
            Path(planned.run_proposal.fresh_state_root),
            planned_slot_context=planned,
            provisional_retain_not_before_utc=RETAIN_UNTIL,
            phase4_source_freeze_directory=root / "phase4",
            repository_root=root,
        )


def artifact_file(outcome: Any, relative_path: str) -> bytes:
    return (outcome.artifact_directory / relative_path).read_bytes()


def ledger_records(outcome: Any) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in artifact_file(outcome, "events/events.jsonl").splitlines()
    ]
    return validate_event_ledger(records)


def telemetry_sample(start_ns: int, stop_ns: int) -> dict[str, Any]:
    total = 8 * GIB
    free = 7 * GIB
    return construct_telemetry_sample(
        sequence=0,
        experiment_run_id=RUN_ID,
        scheduled_slot=0,
        scheduled_monotonic_ns=start_ns,
        observed_monotonic_ns=max(start_ns, stop_ns),
        wall_time_utc=WALL_TIME,
        probe_reading={
            "gpu": {
                "uuid": "GPU-protected-test-uuid",
                "memory_used": {"value": str(total - free), "unit": "B"},
                "memory_free": {"value": str(free), "unit": "B"},
                "memory_reserved": {"value": "0", "unit": "B"},
                "memory_total": {"value": str(total), "unit": "B"},
                "utilization_percent": 0.0,
                "temperature_c": 40.0,
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
                "aptus_lease_active": True,
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
        },
        collector={
            "healthy": True,
            "status_code": None,
            "probe_duration_ns": 1000,
        },
        watchdog={
            "healthy": True,
            "heartbeat_monotonic_ns": max(start_ns, stop_ns),
            "ownership_certain": True,
        },
    )


class HealthyTelemetrySession:
    def __init__(self, signal: SafetySignal | None = None) -> None:
        self.signal = signal
        self.start_count = 0
        self.stop_count = 0
        self.start_ns = 0

    def start(self, *, experiment_run_id: str, start_monotonic_ns: int) -> None:
        if experiment_run_id != RUN_ID:
            raise AssertionError("Telemetry received the wrong run ID.")
        self.start_count += 1
        self.start_ns = start_monotonic_ns

    def safety_signal(self) -> SafetySignal | None:
        if (
            self.signal is not None
            and self.signal.detected_monotonic_ns < self.start_ns
        ):
            return SafetySignal(self.signal.reason_code, time.monotonic_ns())
        return self.signal

    def stop(self, *, stop_monotonic_ns: int) -> TelemetryCapture:
        self.stop_count += 1
        return TelemetryCapture(
            samples=(telemetry_sample(self.start_ns, stop_monotonic_ns),),
            healthy=True,
        )


class StartFailingTelemetrySession(HealthyTelemetrySession):
    def start(self, *, experiment_run_id: str, start_monotonic_ns: int) -> None:
        raise RuntimeError("collector startup failed")


class StopFailingTelemetrySession(HealthyTelemetrySession):
    def stop(self, *, stop_monotonic_ns: int) -> TelemetryCapture:
        self.stop_count += 1
        raise RuntimeError("private collector shutdown failure")


class ExactFailureTelemetrySession(HealthyTelemetrySession):
    def __init__(self, failure_code: str) -> None:
        super().__init__()
        self.failure_code = failure_code

    def stop(self, *, stop_monotonic_ns: int) -> TelemetryCapture:
        self.stop_count += 1
        return TelemetryCapture(
            samples=(telemetry_sample(self.start_ns, stop_monotonic_ns),),
            healthy=False,
            failure_code=self.failure_code,
        )


class AgedSignalTelemetrySession(HealthyTelemetrySession):
    def __init__(self, reason_code: str) -> None:
        super().__init__()
        self.reason_code = reason_code
        self.returned_signal: SafetySignal | None = None

    def safety_signal(self) -> SafetySignal | None:
        if self.returned_signal is None:
            self.returned_signal = SafetySignal(
                self.reason_code, time.monotonic_ns() - 1_000
            )
        return self.returned_signal


class SequencedJobService:
    def __init__(self, root: Path, terminal_states: list[str]) -> None:
        self.root = root
        self.terminal_states = list(terminal_states)
        self.submissions: list[str] = []
        self._records: dict[str, dict[str, Any]] = {}
        self._gets: dict[str, int] = {}

    def submit(self, bundle_dir: Path, **kwargs: Any) -> dict[str, Any]:
        index = len(self.submissions)
        action = kwargs["action"]
        self.submissions.append(action)
        job_id = "job_" + f"{index + 1:032x}"
        log = self.root / f"{index}-{action}.log"
        log.write_bytes(f"complete {action} log\n".encode())
        record = {
            "id": job_id,
            "job_id": job_id,
            "state": "queued",
            "action": action,
            "log": str(log),
            "return_code": None,
            "owner_status": "owning-service",
            "process_pid": None,
        }
        self._records[job_id] = record
        self._gets[job_id] = 0
        return dict(record)

    def get(
        self, job_id: str, *, include_validation_report: bool = True
    ) -> dict[str, Any]:
        record = dict(self._records[job_id])
        count = self._gets[job_id]
        self._gets[job_id] = count + 1
        index = int(job_id[4:], 16) - 1
        if count == 0:
            record.update(
                state="running",
                owner_status="owning-service",
                process_pid=4000 + index,
                process_group_id=4000 + index,
                process_identity=f"linux-start-ticks:{7000 + index}",
            )
        else:
            state = self.terminal_states[index]
            record.update(
                state=state,
                owner_status="terminal",
                process_pid=4000 + index,
                return_code=0 if state == "completed" else 9,
            )
        self._records[job_id] = dict(record)
        return record

    def cancel(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Successful sequence test must not cancel.")


class QualificationClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 1_000_000_000
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += seconds * 1_000_000_000


def qualifying_sample(
    *, sequence: int, start_monotonic_ns: int, experiment_run_id: str
) -> dict[str, Any]:
    observed = start_monotonic_ns + sequence * 1_000_000_000
    return construct_telemetry_sample(
        sequence=sequence,
        experiment_run_id=experiment_run_id,
        scheduled_slot=sequence,
        scheduled_monotonic_ns=observed,
        observed_monotonic_ns=observed,
        wall_time_utc=WALL_TIME,
        probe_reading={
            "gpu": {
                "uuid": "GPU-protected-qualifying",
                "memory_used": {"value": str(GIB), "unit": "B"},
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
        },
        collector={
            "healthy": True,
            "status_code": None,
            "probe_duration_ns": 1,
        },
        watchdog={
            "healthy": True,
            "heartbeat_monotonic_ns": observed,
            "ownership_certain": True,
        },
    )


class QualifyingTelemetry:
    qualifying_profile = True

    def __init__(
        self,
        clock: QualificationClock,
        experiment_run_id: str,
        configuration: dict[str, Any],
    ) -> None:
        self.clock = clock
        self.experiment_run_id = experiment_run_id
        self.configuration = configuration
        self.start_ns: int | None = None

    def configuration_record(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.configuration))

    def start(self, *, experiment_run_id: str, start_monotonic_ns: int) -> None:
        if experiment_run_id != self.experiment_run_id:
            raise AssertionError("wrong qualifying telemetry run ID")
        self.start_ns = start_monotonic_ns

    def safety_signal(self) -> None:
        return None

    def _samples(self) -> tuple[dict[str, Any], ...]:
        if self.start_ns is None:
            raise AssertionError("qualifying telemetry was not started")
        last_slot = (self.clock.value - self.start_ns) // 1_000_000_000
        return tuple(
            qualifying_sample(
                sequence=slot,
                start_monotonic_ns=self.start_ns,
                experiment_run_id=self.experiment_run_id,
            )
            for slot in range(last_slot + 1)
        )

    def snapshot(self) -> TelemetrySnapshot:
        self.clock.advance(118)
        return TelemetrySnapshot(
            samples=self._samples(),
            configuration=self.configuration_record(),
            safety_events=(),
            failure_code=None,
        )

    def stop(self, *, stop_monotonic_ns: int) -> SidecarTelemetryCapture:
        if stop_monotonic_ns > self.clock.value:
            self.clock.value = stop_monotonic_ns
        return SidecarTelemetryCapture(
            samples=self._samples(),
            healthy=True,
            configuration=self.configuration_record(),
            safety_events=(),
        )


class QualifyingJobService:
    def __init__(
        self, root: Path, clock: QualificationClock, experiment_run_id: str
    ) -> None:
        self.root = root
        self.clock = clock
        self.experiment_run_id = experiment_run_id
        self.root.mkdir(mode=0o700)
        self._records: dict[str, dict[str, Any]] = {}
        self._gets: dict[str, int] = {}
        self.submissions: list[str] = []

    def _journal(
        self, job_id: str, action: str
    ) -> tuple[Path, str, list[dict[str, Any]]]:
        directory = self.root / ".campaign-events"
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / f"{job_id}.jsonl"
        events: list[tuple[str, str]] = []
        if action == "pilot":
            events = [
                ("pilot.phase-started", "pilot-phase-1"),
                ("pilot.phase-finished", "pilot-phase-1"),
                ("pilot.phase-started", "pilot-phase-2"),
                ("pilot.phase-finished", "pilot-phase-2"),
            ]
        elif action == "train":
            events = [
                ("training.started", "training"),
                ("export.started", "final-export"),
                ("export.finished", "final-export"),
                ("training.finished", "training"),
                ("verification.started", "parent-verification"),
                ("verification.finished", "parent-verification"),
            ]
        records = [
            {
                "schema_version": "aptus.cuda-campaign-runtime-boundary.v1",
                "experiment_run_id": self.experiment_run_id,
                "job_id": job_id,
                "monotonic_ns": self.clock(),
                "wall_time_utc": WALL_TIME,
                "event_type": event_type,
                "phase": phase,
                "action": action,
                "native_outcome": (
                    "passed" if event_type.endswith("finished") else None
                ),
                "reason_code": "NONE",
            }
            for event_type, phase in events
        ]
        path.write_bytes(canonical_jsonl_bytes(records))
        path.chmod(0o600)
        metadata = path.stat()
        return path, f"{metadata.st_dev}:{metadata.st_ino}", records

    def submit(self, _bundle_dir: Path, **kwargs: Any) -> dict[str, Any]:
        action = kwargs["action"]
        self.submissions.append(action)
        index = len(self.submissions)
        job_id = "job_" + f"{index:032x}"
        log = self.root / f"{index}-{action}.log"
        log.write_text(f"complete {action} log\n", encoding="utf-8")
        path: Path | None = None
        identity: str | None = None
        if action in {"pilot", "train"}:
            self.assert_campaign_kwargs(kwargs)
            path, identity, _records = self._journal(job_id, action)
        record = {
            "id": job_id,
            "job_id": job_id,
            "state": "queued",
            "action": action,
            "log": str(log),
            "return_code": None,
            "run_id": "run_" + job_id[4:] if action == "train" else None,
            "owner_status": "owning-service",
            "process_pid": None,
            "campaign_event_capture": action in {"pilot", "train"},
            "campaign_experiment_run_id": (
                self.experiment_run_id if action in {"pilot", "train"} else None
            ),
            "campaign_event_sink": str(path) if path is not None else None,
            "campaign_event_sink_identity": identity,
        }
        self._records[job_id] = record
        self._gets[job_id] = 0
        return dict(record)

    def assert_campaign_kwargs(self, kwargs: dict[str, Any]) -> None:
        if (
            kwargs.get("campaign_event_capture") is not True
            or kwargs.get("campaign_experiment_run_id") != self.experiment_run_id
        ):
            raise AssertionError("qualifying runtime sink was not requested")

    def get(
        self, job_id: str, *, include_validation_report: bool = True
    ) -> dict[str, Any]:
        del include_validation_report
        record = dict(self._records[job_id])
        count = self._gets[job_id]
        self._gets[job_id] = count + 1
        if count == 0:
            record.update(
                state="running",
                owner_status="owning-service",
                process_pid=5000 + len(self.submissions),
                process_group_id=5000 + len(self.submissions),
                process_identity=f"linux-start-ticks:{8000 + len(self.submissions)}",
            )
        else:
            record.update(state="completed", owner_status="terminal", return_code=0)
        self._records[job_id] = record
        return dict(record)

    def cancel(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f"qualifying job {job_id} must not be cancelled: {kwargs}")


class RuntimeFailureJobService:
    def __init__(
        self,
        root: Path,
        boundaries: tuple[tuple[str, str, str | None, str], ...],
    ) -> None:
        self.root = private_directory(root)
        self.boundaries = boundaries
        self.job_id = "job_" + "f" * 32
        self.record: dict[str, Any] | None = None

    def submit(self, _bundle_dir: Path, **kwargs: Any) -> dict[str, Any]:
        action = kwargs["action"]
        path = self.root / "runtime-failure.jsonl"
        base = time.monotonic_ns()
        records = [
            {
                "schema_version": "aptus.cuda-campaign-runtime-boundary.v1",
                "experiment_run_id": RUN_ID,
                "job_id": self.job_id,
                "monotonic_ns": base + index,
                "wall_time_utc": WALL_TIME,
                "event_type": event_type,
                "phase": phase,
                "action": action,
                "native_outcome": native_outcome,
                "reason_code": reason_code,
            }
            for index, (event_type, phase, native_outcome, reason_code) in enumerate(
                self.boundaries, 1
            )
        ]
        path.write_bytes(canonical_jsonl_bytes(records))
        path.chmod(0o600)
        metadata = path.stat()
        log = self.root / "runtime-failure.log"
        log.write_text("failed\n", encoding="utf-8")
        self.record = {
            "id": self.job_id,
            "job_id": self.job_id,
            "state": "queued",
            "action": action,
            "log": str(log),
            "return_code": None,
            "run_id": "run_" + "f" * 32,
            "owner_status": "owning-service",
            "campaign_event_capture": True,
            "campaign_experiment_run_id": RUN_ID,
            "campaign_event_sink": str(path),
            "campaign_event_sink_identity": f"{metadata.st_dev}:{metadata.st_ino}",
        }
        return dict(self.record)

    def get(
        self, job_id: str, *, include_validation_report: bool = True
    ) -> dict[str, Any]:
        del include_validation_report
        if job_id != self.job_id or self.record is None:
            raise AssertionError("wrong runtime-failure job")
        self.record.update(
            state="failed",
            owner_status="terminal",
            return_code=9,
        )
        return dict(self.record)

    def cancel(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f"failed job {job_id} must not be cancelled: {kwargs}")


class FakeOwningJobService:
    def __init__(
        self,
        log_path: Path,
        *,
        terminal_state: str = "completed",
        ownership_uncertain: bool = False,
        incomplete_cancellation: bool = False,
        cancel_failure: bool = False,
    ) -> None:
        self.log_path = log_path
        self.log_path.write_bytes(b"complete managed log\n")
        self.terminal_state = terminal_state
        self.ownership_uncertain = ownership_uncertain
        self.incomplete_cancellation = incomplete_cancellation
        self.cancel_failure = cancel_failure
        self.submissions = 0
        self.gets = 0
        self.cancel_calls: list[tuple[str, str | None, int | None]] = []
        self.cancelled_record: dict[str, Any] | None = None

    def _record(self, state: str, owner_status: str) -> dict[str, Any]:
        return {
            "id": JOB_ID,
            "job_id": JOB_ID,
            "state": state,
            "action": "preflight",
            "log": str(self.log_path),
            "return_code": 0 if state == "completed" else None,
            "owner_status": owner_status,
        }

    def submit(self, bundle_dir: Path, **kwargs: Any) -> dict[str, Any]:
        self.submissions += 1
        self.bundle_dir = bundle_dir
        self.submit_kwargs = kwargs
        return self._record("queued", "owning-service")

    def get(
        self, job_id: str, *, include_validation_report: bool = True
    ) -> dict[str, Any]:
        if job_id != JOB_ID:
            raise AssertionError("Harness polled a different job ID.")
        self.gets += 1
        if self.cancelled_record is not None:
            return dict(self.cancelled_record)
        if self.ownership_uncertain:
            return self._record("running", "external-service")
        if self.gets == 1:
            return self._record("running", "owning-service")
        return self._record(self.terminal_state, "terminal")

    def cancel(
        self,
        job_id: str,
        *,
        reason_code: str | None = None,
        trigger_detected_monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        self.cancel_calls.append((job_id, reason_code, trigger_detected_monotonic_ns))
        if job_id != JOB_ID or trigger_detected_monotonic_ns is None:
            raise AssertionError("Cancellation authority was not exact.")
        if self.cancel_failure:
            raise ValueError("ownership could not be proven")
        requested = trigger_detected_monotonic_ns + 100_000_000
        terminated = requested + 200_000_000
        reconciled = terminated + 100_000_000
        record = self._record("cancelled", "terminal")
        record.update(
            return_code=-15,
            cancel_reason_code=reason_code,
            cancel_trigger_detected_monotonic_ns=(trigger_detected_monotonic_ns),
            cancel_requested_at=WALL_TIME,
            cancel_requested_monotonic_ns=requested,
            process_group_terminated_at=WALL_TIME,
            process_group_terminated_monotonic_ns=terminated,
            lease_reconciled_at=WALL_TIME,
            lease_reconciled_monotonic_ns=reconciled,
        )
        if self.incomplete_cancellation:
            record.pop("lease_reconciled_monotonic_ns")
        self.cancelled_record = dict(record)
        return record


class PlainCommandCaptureTests(unittest.TestCase):
    def test_multi_mebibyte_stdout_stderr_streams_byte_exactly_without_spool_leftovers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")
            size = 2 * 1024 * 1024
            outcome = harness.run_command(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        f"sys.stdout.buffer.write(b'A'*{size}); "
                        "sys.stdout.buffer.flush(); "
                        f"sys.stderr.buffer.write(b'B'*{size}); "
                        "sys.stderr.buffer.flush()"
                    ),
                ],
                artifact_directory=vault / "large-output",
                working_directory=root,
                timeout_seconds=10,
            )

            self.assertTrue(outcome.sealed)
            self.assertEqual(
                artifact_file(outcome, "command/combined-output.bin"),
                b"A" * size + b"B" * size,
            )
            self.assertEqual(
                list(harness.state_root.glob(".*.command-output.spool")), []
            )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_command_spool_is_no_clobber_and_rejects_a_preexisting_symlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")
            fixed_id = "artifact_" + "e" * 32
            target = root / "outside.bin"
            target.write_bytes(b"outside\n")
            spool = harness.state_root / f".{fixed_id}.command-output.spool"
            spool.symlink_to(target)

            with patch(
                "tools.cuda_campaign.harness.new_opaque_id",
                return_value=fixed_id,
            ):
                outcome = harness.run_command(
                    [sys.executable, "-c", "print('must-not-run')"],
                    artifact_directory=vault / "spool-symlink",
                    working_directory=root,
                    timeout_seconds=5,
                )

            self.assertFalse(outcome.sealed)
            self.assertEqual(outcome.capture_reason_code, "STREAM_CAPTURE_FAILURE")
            self.assertEqual(target.read_bytes(), b"outside\n")
            self.assertTrue(spool.is_symlink())

    def test_telemetry_start_failure_seals_fallback_and_blocks_submissions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")
            artifact = vault / "telemetry-start-failure"

            outcome = harness.run_command(
                [sys.executable, "-c", "raise SystemExit(99)"],
                artifact_directory=artifact,
                working_directory=root,
                timeout_seconds=5,
                telemetry_session=StartFailingTelemetrySession(),
            )

            self.assertEqual(outcome.native_outcome, "guard-blocked")
            self.assertEqual(outcome.reason_code, "TELEMETRY_COLLECTOR_FAILURE")
            self.assertEqual(outcome.evidence_status, "capture-invalid")
            self.assertEqual(outcome.capture_reason_code, "TELEMETRY_COLLECTOR_FAILURE")
            self.assertTrue(outcome.submission_blocked)
            self.assertFalse(outcome.sealed)
            fallback = artifact.with_name(artifact.name + ".capture-failure")
            receipt = verify_sealed_artifact(fallback)
            self.assertEqual(
                receipt["manifest"]["identity_bindings"]["capture_status"], "failed"
            )

    def test_command_launch_failure_is_an_immutable_capture_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")
            artifact = vault / "launch-failure"

            with patch(
                "tools.cuda_campaign.harness.subprocess.Popen",
                side_effect=OSError("protected executable detail"),
            ):
                outcome = harness.run_command(
                    ["missing-command"],
                    artifact_directory=artifact,
                    working_directory=root,
                    timeout_seconds=5,
                )

            self.assertEqual(outcome.native_outcome, "unknown")
            self.assertEqual(outcome.reason_code, "UNKNOWN_TERMINAL_STATE")
            self.assertEqual(outcome.capture_reason_code, "STREAM_CAPTURE_FAILURE")
            self.assertFalse(outcome.submission_blocked)
            fallback = artifact.with_name(artifact.name + ".capture-failure")
            verify_sealed_artifact(fallback)
            receipt = json.loads((fallback / "capture-failure.json").read_bytes())
            self.assertEqual(receipt["reason_code"], "STREAM_CAPTURE_FAILURE")
            self.assertNotIn("protected executable detail", json.dumps(receipt))

    def test_termination_failure_closes_and_removes_command_spool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")
            with patch(
                "tools.cuda_campaign.harness._terminate_plain_process",
                side_effect=CaptureHarnessError("injected termination uncertainty"),
            ):
                outcome = harness.run_command(
                    [sys.executable, "-c", "print('finished')"],
                    artifact_directory=vault / "termination-failure",
                    working_directory=root,
                    timeout_seconds=5,
                )

            self.assertFalse(outcome.sealed)
            self.assertTrue(outcome.submission_blocked)
            self.assertEqual(outcome.capture_reason_code, "STREAM_CAPTURE_FAILURE")
            self.assertEqual(
                list(harness.state_root.glob(".*.command-output.spool")), []
            )

    def test_same_byte_inode_swap_at_spool_copy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")
            original_copy = RawArtifactWriter.copy_payload_from_descriptor
            swapped_inodes: list[tuple[int, int]] = []

            def swap_then_copy(
                writer: RawArtifactWriter,
                source_descriptor: int,
                source_path: Path,
                source_fingerprint: tuple[int, ...],
                relative_path: str,
                **kwargs: Any,
            ) -> dict[str, Any]:
                moved = source_path.with_name(source_path.name + ".moved")
                source_path.rename(moved)
                same_bytes = moved.read_bytes()
                source_path.write_bytes(same_bytes)
                source_path.chmod(0o600)
                swapped_inodes.append((moved.stat().st_ino, source_path.stat().st_ino))
                return original_copy(
                    writer,
                    source_descriptor,
                    source_path,
                    source_fingerprint,
                    relative_path,
                    **kwargs,
                )

            with patch.object(
                RawArtifactWriter,
                "copy_payload_from_descriptor",
                new=swap_then_copy,
            ):
                outcome = harness.run_command(
                    [sys.executable, "-c", "print('same bytes')"],
                    artifact_directory=vault / "inode-swap",
                    working_directory=root,
                    timeout_seconds=5,
                )

            self.assertFalse(outcome.sealed)
            self.assertEqual(outcome.capture_reason_code, "STREAM_CAPTURE_FAILURE")
            self.assertEqual(len(swapped_inodes), 1)
            self.assertNotEqual(*swapped_inodes[0])
            self.assertFalse(
                (outcome.artifact_directory / "command/combined-output.bin").exists()
            )

    def test_exact_argv_success_preserves_complete_combined_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")
            literal = "$HOME;not-a-shell"
            selected = root / "selected.json"
            selected.write_bytes(b'{"selected":true}\n')
            command = [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "sys.stdout.write('stdout:' + sys.argv[1] + '\\n'); "
                    "sys.stdout.flush(); "
                    "sys.stderr.write('stderr\\n'); sys.stderr.flush()"
                ),
                literal,
            ]

            outcome = harness.run_command(
                command,
                artifact_directory=vault / "success",
                working_directory=root,
                timeout_seconds=5,
                selected_artifacts=(
                    SelectedArtifact(
                        selected,
                        "selected/result.json",
                        "selected-artifact",
                        "application/json",
                    ),
                ),
            )

            self.assertTrue(outcome.sealed)
            self.assertEqual(outcome.native_outcome, "passed")
            self.assertEqual(outcome.reason_code, "NONE")
            self.assertEqual(outcome.exit_code, 0)
            self.assertEqual(
                artifact_file(outcome, "command/combined-output.bin"),
                b"stdout:$HOME;not-a-shell\nstderr\n",
            )
            record = json.loads(artifact_file(outcome, "command/record.json"))
            self.assertEqual(record["exact_argv"], command)
            self.assertEqual(
                artifact_file(outcome, "selected/result.json"),
                b'{"selected":true}\n',
            )
            self.assertLessEqual(
                record["started_monotonic_ns"], record["finished_monotonic_ns"]
            )
            events = ledger_records(outcome)
            self.assertEqual(events[0]["event_type"], "clock.mapping")
            self.assertEqual(events[-1]["event_type"], "seal.started")
            self.assertIn("command.started", [row["event_type"] for row in events])
            self.assertIn("command.finished", [row["event_type"] for row in events])
            verify_sealed_artifact(outcome.artifact_directory)
            for path in outcome.artifact_directory.rglob("*"):
                if path.is_file():
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_nonzero_is_native_failure_but_capture_is_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")

            outcome = harness.run_command(
                [sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"],
                artifact_directory=vault / "nonzero",
                working_directory=root,
                timeout_seconds=5,
            )

            self.assertTrue(outcome.sealed)
            self.assertEqual(outcome.exit_code, 7)
            self.assertEqual(outcome.native_outcome, "failed")
            self.assertEqual(outcome.reason_code, "PROCESS_EXIT_NONZERO")
            self.assertEqual(
                artifact_file(outcome, "command/combined-output.bin"), b"bad\n"
            )

    def test_timeout_retains_partial_output_and_marks_timed_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")

            outcome = harness.run_command(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys,time; "
                        "sys.stdout.buffer.write(b'before-timeout\\n'); "
                        "sys.stdout.buffer.flush(); time.sleep(30)"
                    ),
                ],
                artifact_directory=vault / "timeout",
                working_directory=root,
                timeout_seconds=0.1,
            )

            self.assertTrue(outcome.sealed)
            self.assertTrue(outcome.timed_out)
            self.assertEqual(outcome.native_outcome, "timed-out")
            self.assertEqual(outcome.reason_code, "EMERGENCY_DEADLINE_EXCEEDED")
            self.assertEqual(
                artifact_file(outcome, "command/combined-output.bin"),
                b"before-timeout\n",
            )
            self.assertEqual(
                list(harness.state_root.glob(".*.command-output.spool")), []
            )

    def test_no_clobber_and_immutable_capture_failure_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")
            artifact = vault / "seal-failure"
            original_seal = RawArtifactWriter.seal

            def fail_normal_seal(writer: RawArtifactWriter) -> dict[str, Any]:
                if writer.directory == artifact.resolve():
                    raise OSError("disk")
                return original_seal(writer)

            with patch.object(RawArtifactWriter, "seal", new=fail_normal_seal):
                outcome = harness.run_command(
                    [sys.executable, "-c", "print('captured')"],
                    artifact_directory=artifact,
                    working_directory=root,
                    timeout_seconds=5,
                )

            self.assertFalse(outcome.sealed)
            self.assertEqual(outcome.reason_code, "NONE")
            self.assertEqual(outcome.evidence_status, "capture-invalid")
            self.assertEqual(outcome.capture_reason_code, "SEAL_FAILURE")
            fallback_artifact = artifact.with_name(artifact.name + ".capture-failure")
            receipt_path = fallback_artifact / "capture-failure.json"
            self.assertEqual(outcome.capture_failure_receipt, receipt_path)
            receipt = validate_record(json.loads(receipt_path.read_bytes()))
            self.assertEqual(receipt["reason_code"], "SEAL_FAILURE")
            self.assertEqual(len(receipt["available_files"]), 3)
            verify_sealed_artifact(fallback_artifact)
            before = receipt_path.read_bytes()
            with self.assertRaises(FileExistsError):
                harness.run_command(
                    [sys.executable, "-c", "print('second')"],
                    artifact_directory=artifact,
                    working_directory=root,
                    timeout_seconds=5,
                )
            self.assertEqual(receipt_path.read_bytes(), before)

    def test_fallback_seal_failure_durably_blocks_later_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root)
            vault = private_directory(root / "vault")
            with (
                patch.object(RawArtifactWriter, "seal", side_effect=OSError("disk")),
                patch(
                    "tools.cuda_campaign.harness.write_sealed_capture_failure_artifact",
                    side_effect=OSError("fallback disk"),
                ),
                self.assertRaises(CaptureHarnessError),
            ):
                harness.run_command(
                    [sys.executable, "-c", "print('captured')"],
                    artifact_directory=vault / "double-seal-failure",
                    working_directory=root,
                    timeout_seconds=5,
                )
            self.assertTrue(harness.submissions_blocked)
            marker = json.loads(harness.submission_block_path.read_bytes())
            self.assertEqual(marker["reason_code"], "SEAL_FAILURE")


class ManagedJobCaptureTests(unittest.TestCase):
    def test_safety_signal_rejects_warning_and_capture_only_codes(self) -> None:
        for reason_code in (
            "THERMAL_WARNING_SUSTAINED",
            "TELEMETRY_QUALIFYING_GAP",
            "MISSING_REQUIRED_EVIDENCE",
        ):
            with self.subTest(reason_code=reason_code), self.assertRaises(ValueError):
                SafetySignal(reason_code, 1)

    def test_get_failure_cancels_exact_validated_job_and_verifies_milestones(
        self,
    ) -> None:
        class GetFailingService(FakeOwningJobService):
            def get(
                self, job_id: str, *, include_validation_report: bool = True
            ) -> dict[str, Any]:
                if job_id != JOB_ID:
                    raise AssertionError("Harness polled a different job ID.")
                raise RuntimeError("private polling failure")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = GetFailingService(root / "managed.log")
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "get-failure",
                supervision_timeout_seconds=5,
                telemetry_session=HealthyTelemetrySession(),
            )

            self.assertEqual(outcome.native_outcome, "cancelled")
            self.assertEqual(outcome.reason_code, "OWNERSHIP_UNCERTAIN")
            self.assertEqual(len(service.cancel_calls), 1)
            self.assertEqual(service.cancel_calls[0][0], JOB_ID)
            self.assertTrue(outcome.submission_blocked)

    def test_get_identity_mismatch_cancels_original_validated_job(self) -> None:
        class GetMismatchService(FakeOwningJobService):
            def get(
                self, job_id: str, *, include_validation_report: bool = True
            ) -> dict[str, Any]:
                if job_id != JOB_ID:
                    raise AssertionError("Harness polled a different job ID.")
                record = self._record("running", "owning-service")
                record["id"] = "job_" + "e" * 32
                record["job_id"] = record["id"]
                return record

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = GetMismatchService(root / "managed.log")
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "get-mismatch",
                supervision_timeout_seconds=5,
                telemetry_session=HealthyTelemetrySession(),
            )

            self.assertEqual(outcome.native_outcome, "cancelled")
            self.assertEqual(len(service.cancel_calls), 1)
            self.assertEqual(service.cancel_calls[0][0], JOB_ID)
            self.assertTrue(outcome.submission_blocked)

    def test_uncertain_owner_cancel_refusal_remains_unknown_and_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = FakeOwningJobService(
                root / "managed.log",
                ownership_uncertain=True,
                cancel_failure=True,
            )
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "uncertain-cancel-refused",
                supervision_timeout_seconds=5,
                poll_interval_seconds=0.001,
                allow_nonqualifying_without_telemetry_for_test=True,
            )

            self.assertEqual(outcome.native_outcome, "unknown")
            self.assertEqual(outcome.reason_code, "OWNERSHIP_UNCERTAIN")
            self.assertTrue(outcome.submission_blocked)
            self.assertEqual(len(service.cancel_calls), 1)

    def test_safety_cancellation_uses_owner_exact_id_and_persists_sla_events(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = FakeOwningJobService(root / "managed.log")
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")

            def safety(record: dict[str, Any]) -> SafetySignal | None:
                if record["state"] == "running":
                    return SafetySignal("THERMAL_STOP_IMMEDIATE", time.monotonic_ns())
                return None

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "cancelled",
                supervision_timeout_seconds=5,
                poll_interval_seconds=0.001,
                safety_check=safety,
                allow_nonqualifying_without_telemetry_for_test=True,
            )

            self.assertTrue(outcome.sealed)
            self.assertEqual(outcome.native_outcome, "cancelled")
            self.assertEqual(outcome.reason_code, "THERMAL_STOP_IMMEDIATE")
            self.assertFalse(outcome.submission_blocked)
            self.assertEqual(len(service.cancel_calls), 1)
            self.assertEqual(service.cancel_calls[0][0], JOB_ID)
            self.assertEqual(
                artifact_file(outcome, "job/full.log"), b"complete managed log\n"
            )
            terminal = json.loads(artifact_file(outcome, "job/terminal.json"))
            milestones = verify_cancellation_milestones(
                terminal,
                reason_code="THERMAL_STOP_IMMEDIATE",
                trigger_detected_monotonic_ns=(
                    service.cancel_calls[0][2]  # type: ignore[arg-type]
                ),
            )
            self.assertLessEqual(
                milestones.cancel_requested_monotonic_ns
                - milestones.trigger_detected_monotonic_ns,
                CANCEL_REQUEST_SLA_NS,
            )
            event_types = [row["event_type"] for row in ledger_records(outcome)]
            for expected in (
                "safety.triggered",
                "cancellation.requested",
                "process-group.terminated",
                "lease.reconciled",
            ):
                self.assertIn(expected, event_types)

    def test_ownership_uncertainty_requests_exact_cancel_and_blocks_later_submission(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = FakeOwningJobService(
                root / "managed.log", ownership_uncertain=True
            )
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "uncertain",
                supervision_timeout_seconds=5,
                poll_interval_seconds=0.001,
                allow_nonqualifying_without_telemetry_for_test=True,
            )

            self.assertTrue(outcome.sealed)
            self.assertEqual(outcome.native_outcome, "cancelled")
            self.assertEqual(outcome.reason_code, "OWNERSHIP_UNCERTAIN")
            self.assertTrue(outcome.submission_blocked)
            self.assertEqual(len(service.cancel_calls), 1)
            self.assertEqual(service.cancel_calls[0][0], JOB_ID)
            marker = harness.submission_block_path
            self.assertTrue(marker.is_file())
            self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
            with self.assertRaises(SubmissionBlockedError):
                harness.run_managed_job(
                    root,
                    artifact_directory=vault / "must-not-submit",
                    supervision_timeout_seconds=5,
                )
            self.assertEqual(service.submissions, 1)

    def test_watchdog_loss_with_missing_reconciliation_blocks_later_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = FakeOwningJobService(
                root / "managed.log", incomplete_cancellation=True
            )
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "watchdog-loss",
                supervision_timeout_seconds=5,
                poll_interval_seconds=0.001,
                safety_check=lambda _: SafetySignal(
                    "WATCHDOG_HEARTBEAT_LOST", time.monotonic_ns()
                ),
                allow_nonqualifying_without_telemetry_for_test=True,
            )

            self.assertTrue(outcome.sealed)
            self.assertEqual(outcome.native_outcome, "unknown")
            self.assertEqual(outcome.reason_code, "LEASE_RECONCILIATION_FAILURE")
            self.assertTrue(outcome.submission_blocked)
            with self.assertRaises(SubmissionBlockedError):
                harness.run_managed_job(
                    root,
                    artifact_directory=vault / "blocked",
                    supervision_timeout_seconds=5,
                )

    def test_late_cancellation_milestone_is_rejected(self) -> None:
        trigger = 1_000_000_000
        record = {
            "cancel_reason_code": "THERMAL_STOP_IMMEDIATE",
            "cancel_trigger_detected_monotonic_ns": trigger,
            "cancel_requested_monotonic_ns": (trigger + CANCEL_REQUEST_SLA_NS + 1),
            "process_group_terminated_monotonic_ns": (
                trigger + CANCEL_REQUEST_SLA_NS + 2
            ),
            "lease_reconciled_monotonic_ns": trigger + CANCEL_REQUEST_SLA_NS + 3,
        }
        with self.assertRaises(CancellationSLAError):
            verify_cancellation_milestones(
                record,
                reason_code="THERMAL_STOP_IMMEDIATE",
                trigger_detected_monotonic_ns=trigger,
            )

    def test_attached_telemetry_safety_signal_cannot_be_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = FakeOwningJobService(root / "managed.log")
            session = AgedSignalTelemetrySession("THERMAL_STOP_IMMEDIATE")
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "automatic-sidecar-stop",
                supervision_timeout_seconds=5,
                poll_interval_seconds=0.001,
                telemetry_session=session,
            )

            self.assertEqual(outcome.native_outcome, "guard-blocked")
            self.assertEqual(service.submissions, 0)
            self.assertEqual(len(service.cancel_calls), 0)
            safety_event = next(
                row
                for row in ledger_records(outcome)
                if row["event_type"] == "safety.triggered"
            )
            self.assertEqual(
                safety_event["monotonic_ns"],
                session.returned_signal.detected_monotonic_ns,  # type: ignore[union-attr]
            )
            self.assertEqual(session.start_count, 1)
            self.assertEqual(session.stop_count, 1)


class ManagedSequenceTests(unittest.TestCase):
    def test_runtime_failure_closure_controls_exact_terminal_reason(self) -> None:
        cases = {
            "pre-export-oom": (
                (
                    ("training.started", "training", None, "NONE"),
                    ("training.finished", "training", "failed", "CUDA_OOM"),
                ),
                "CUDA_OOM",
            ),
            "nested-export": (
                (
                    ("training.started", "training", None, "NONE"),
                    ("export.started", "final-export", None, "NONE"),
                    (
                        "export.finished",
                        "final-export",
                        "failed",
                        "EXPORT_VERIFICATION_FAILURE",
                    ),
                    (
                        "training.finished",
                        "training",
                        "failed",
                        "EXPORT_VERIFICATION_FAILURE",
                    ),
                ),
                "EXPORT_VERIFICATION_FAILURE",
            ),
            "parent-verification": (
                (
                    ("training.started", "training", None, "NONE"),
                    ("export.started", "final-export", None, "NONE"),
                    ("export.finished", "final-export", "passed", "NONE"),
                    ("training.finished", "training", "passed", "NONE"),
                    ("verification.started", "parent-verification", None, "NONE"),
                    (
                        "verification.finished",
                        "parent-verification",
                        "failed",
                        "EXPORT_VERIFICATION_FAILURE",
                    ),
                ),
                "EXPORT_VERIFICATION_FAILURE",
            ),
        }
        for name, (boundaries, expected_reason) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                service = RuntimeFailureJobService(root / "service", boundaries)
                harness = harness_at(root, job_service=service)
                ledger = harness._new_ledger()
                spec = ManagedActionSpec(
                    "train",
                    "train",
                    5,
                    submit_kwargs={
                        "campaign_event_capture": True,
                        "campaign_experiment_run_id": RUN_ID,
                    },
                )

                result = harness._supervise_managed_action(
                    service,
                    root,
                    spec,
                    ledger=ledger,
                    poll_interval_seconds=0.001,
                    safety_check=None,
                    telemetry_session=None,
                    pre_action_check=None,
                )

                self.assertEqual(result.native_outcome, "failed")
                self.assertEqual(result.reason_code, expected_reason)
                terminal_events = [
                    row
                    for row in ledger.records
                    if row["event_type"] in {"job.state-observed", "command.finished"}
                ]
                self.assertTrue(terminal_events)
                self.assertTrue(
                    all(
                        row["reason_code"] == expected_reason for row in terminal_events
                    )
                )

    def test_qualifying_telemetry_factory_rejects_untrusted_nvidia_binary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "fake-nvidia-smi"
            fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake.chmod(0o777)

            with self.assertRaisesRegex(ProbeFailure, "NVIDIA_SMI_UNTRUSTED"):
                resolve_trusted_nvidia_smi(str(fake))
            with self.assertRaisesRegex(ProbeFailure, "NVIDIA_SMI_UNAVAILABLE"):
                resolve_trusted_nvidia_smi(str(root / "missing-nvidia-smi"))

    def test_synthetic_qualifying_dependencies_cannot_be_protocol_valid(
        self,
    ) -> None:
        from tests.tools.test_cuda_campaign_qualification import qualifying_context

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            context = qualifying_context(
                bundle_path=str(bundle),
                state_root=str(root / "fresh-state"),
            )
            clock = QualificationClock()
            service = QualifyingJobService(
                root / "jobs", clock, context.experiment_run_id
            )
            with self.assertRaisesRegex(TypeError, "qualification_context"):
                CaptureHarness(
                    private_directory(root / "injected-state"),
                    attempt_slot_id=context.attempt_slot_id,
                    experiment_run_id=context.experiment_run_id,
                    provisional_retain_not_before_utc=RETAIN_UNTIL,
                    job_service=service,
                    qualification_context=context,
                    monotonic_ns=clock,
                    wall_time=lambda: WALL_TIME,
                    sleep=lambda _seconds: None,
                )

            forged_root = private_directory(root / "forged-state")
            borrowed = CaptureHarness.with_job_root(
                forged_root,
                forged_root,
                attempt_slot_id=context.attempt_slot_id,
                experiment_run_id=context.experiment_run_id,
                provisional_retain_not_before_utc=RETAIN_UNTIL,
            )
            authority = object()
            borrowed.qualification_context = context
            borrowed._qualifying_authority = authority
            borrowed._qualifying_job_service = borrowed.job_service  # type: ignore[assignment]
            self.assertFalse(borrowed._authorized_for_qualifying_factory(authority))
            with self.assertRaisesRegex(
                CaptureHarnessError, "qualifying campaign harness"
            ):
                borrowed.create_qualifying_telemetry_session(
                    filesystem_path=root,
                    unavailable_optional_sensors=(
                        "cpu_temperature",
                        "nvme_temperature",
                    ),
                )

            injected_probe = LinuxNvidiaHostProbe(
                filesystem_path=bundle,
                managed_pids=lambda: (),
                managed_process_groups=lambda: (),
                kernel_events=lambda: {
                    "xid_errors": [],
                    "reset_detected": False,
                    "device_lost": False,
                    "hardware_error": False,
                },
                lease_active=lambda: False,
                disk_growth_bytes=lambda: 0,
            )
            injected_session = BackgroundTelemetrySession.qualifying_production(
                probe=injected_probe,
                ownership_certain=lambda: True,
                emergency_deadline_seconds=context.emergency_deadline_seconds,
                remaining_disk_budget_bytes=context.remaining_disk_budget_bytes,
                initial_thermal_limits_available=False,
                provider_name="linux-nvidia-host-probe",
                provider_version="aptus-cuda-campaign-v1",
                support_bindings={
                    "cpu_temperature": "unsupported:reviewed-not-configured",
                    "gpu_thermal_limits": "unsupported:test-provider",
                    "hardware_events": "journal:test-provider",
                    "nvidia_smi_binary": "sha256:" + "1" * 64,
                    "nvme_temperature": "unsupported:reviewed-not-configured",
                    "xid_projection": "journal:test-provider",
                },
                ownership_binding="factory-owned-job-service-process-group-v1",
                disk_growth_binding="factory-owned-statvfs-baseline-v1",
            )
            injected_session._qualifying_profile = True
            injected_session._qualifying_authority = authority
            self.assertFalse(injected_session._authorized_for_harness(authority))
            with self.assertRaisesRegex(ValueError, "authority is unavailable"):
                borrowed._require_qualifying_runtime_authority(injected_session)

    def test_between_action_pre_submit_signal_blocks_without_second_submission(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SequencedJobService(root, ["completed", "completed"])
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")
            checks = 0

            def guard(_spec: ManagedActionSpec) -> SafetySignal | None:
                nonlocal checks
                checks += 1
                if checks == 2:
                    return SafetySignal("THERMAL_STOP_IMMEDIATE", time.monotonic_ns())
                return None

            outcome = harness.run_managed_sequence(
                root,
                actions=(
                    ManagedActionSpec("dependencies", "dependency", 5),
                    ManagedActionSpec("pilot-phase", "pilot", 5),
                ),
                artifact_directory=vault / "guarded-second-action",
                poll_interval_seconds=0.001,
                telemetry_session=HealthyTelemetrySession(),
                pre_action_check=guard,
            )

            self.assertEqual(service.submissions, ["dependency"])
            self.assertEqual(outcome.native_outcome, "guard-blocked")
            self.assertEqual(outcome.reason_code, "THERMAL_STOP_IMMEDIATE")
            safety = [
                row
                for row in ledger_records(outcome)
                if row["event_type"] == "safety.triggered"
            ]
            self.assertEqual(len(safety), 1)
            self.assertEqual(safety[0]["phase"], "pilot-phase")

    def test_final_source_rewalk_rejects_selected_artifact_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SequencedJobService(root, ["completed"])
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")
            selected = root / "selected.json"
            selected.write_bytes(b'{"value":"initial"}\n')
            from tools.cuda_campaign import harness as harness_module

            original_hash = harness_module._hash_stable_regular_source
            target_hashes = 0

            def mutate_before_final_rewalk(path: Path) -> str:
                nonlocal target_hashes
                if path == selected:
                    target_hashes += 1
                    if target_hashes == 2:
                        selected.write_bytes(b'{"value":"changed"}\n')
                return original_hash(path)

            with patch(
                "tools.cuda_campaign.harness._hash_stable_regular_source",
                side_effect=mutate_before_final_rewalk,
            ):
                outcome = harness.run_managed_job(
                    root,
                    artifact_directory=vault / "source-mutation",
                    supervision_timeout_seconds=5,
                    selected_artifacts=(
                        SelectedArtifact(
                            selected,
                            "selected/result.json",
                            "selected-artifact",
                            "application/json",
                        ),
                    ),
                    telemetry_session=HealthyTelemetrySession(),
                )

            self.assertFalse(outcome.sealed)
            self.assertEqual(outcome.capture_reason_code, "ARTIFACT_INTEGRITY_FAILURE")
            self.assertIsNotNone(outcome.capture_failure_receipt)
            verify_sealed_artifact(outcome.capture_failure_receipt.parent)  # type: ignore[union-attr]

    def test_malformed_submission_blocks_without_guessing_a_cancel_target(self) -> None:
        class MalformedService:
            def __init__(self) -> None:
                self.cancel_calls = 0

            def submit(self, _bundle_dir: Path, **_kwargs: Any) -> list[str]:
                return ["not", "a", "record"]

            def cancel(self, _job_id: str, **_kwargs: Any) -> dict[str, Any]:
                self.cancel_calls += 1
                raise AssertionError("Malformed submission must not be cancelled.")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = MalformedService()
            harness = harness_at(root, job_service=service)  # type: ignore[arg-type]
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "malformed",
                supervision_timeout_seconds=5,
                telemetry_session=HealthyTelemetrySession(),
            )

            self.assertEqual(outcome.native_outcome, "unknown")
            self.assertEqual(outcome.reason_code, "OWNERSHIP_UNCERTAIN")
            self.assertTrue(outcome.submission_blocked)
            self.assertEqual(service.cancel_calls, 0)

    def test_mismatched_submission_ids_block_without_guessing_cancel_target(
        self,
    ) -> None:
        class MismatchedService:
            def __init__(self) -> None:
                self.cancel_calls = 0

            def submit(self, _bundle_dir: Path, **_kwargs: Any) -> dict[str, Any]:
                return {
                    "id": "job_" + "1" * 32,
                    "job_id": "job_" + "2" * 32,
                    "state": "queued",
                }

            def cancel(self, _job_id: str, **_kwargs: Any) -> dict[str, Any]:
                self.cancel_calls += 1
                raise AssertionError("Ambiguous IDs must never be cancelled.")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = MismatchedService()
            harness = harness_at(root, job_service=service)  # type: ignore[arg-type]
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "mismatched",
                supervision_timeout_seconds=5,
                telemetry_session=HealthyTelemetrySession(),
            )

            self.assertEqual(outcome.native_outcome, "unknown")
            self.assertTrue(outcome.submission_blocked)
            self.assertEqual(service.cancel_calls, 0)

    def test_invalid_submission_id_blocks_without_guessing_cancel_target(self) -> None:
        class InvalidIdentityService:
            def __init__(self) -> None:
                self.cancel_calls = 0

            def submit(self, _bundle_dir: Path, **_kwargs: Any) -> dict[str, Any]:
                return {"id": "not-a-job", "job_id": "not-a-job", "state": "queued"}

            def cancel(self, _job_id: str, **_kwargs: Any) -> dict[str, Any]:
                self.cancel_calls += 1
                raise AssertionError("Invalid IDs must never be cancelled.")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = InvalidIdentityService()
            harness = harness_at(root, job_service=service)  # type: ignore[arg-type]
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "invalid-identity",
                supervision_timeout_seconds=5,
                telemetry_session=HealthyTelemetrySession(),
            )

            self.assertEqual(outcome.native_outcome, "unknown")
            self.assertTrue(outcome.submission_blocked)
            self.assertEqual(service.cancel_calls, 0)

    def test_unexpected_ledger_failure_cancels_known_job_and_seals_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = FakeOwningJobService(root / "managed.log")
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")
            artifact = vault / "ledger-failure"
            session = HealthyTelemetrySession()
            original_append = EventLedgerWriter.append

            def fail_state_event(self, **kwargs):
                if kwargs.get("event_type") == "job.state-observed":
                    raise RuntimeError("private ledger persistence failure")
                return original_append(self, **kwargs)

            with patch.object(EventLedgerWriter, "append", new=fail_state_event):
                outcome = harness.run_managed_job(
                    root,
                    artifact_directory=artifact,
                    supervision_timeout_seconds=5,
                    poll_interval_seconds=0.001,
                    telemetry_session=session,
                )

            self.assertFalse(outcome.sealed)
            self.assertEqual(outcome.capture_reason_code, "STREAM_CAPTURE_FAILURE")
            self.assertEqual(len(service.cancel_calls), 1)
            self.assertEqual(service.cancel_calls[0][0], JOB_ID)
            self.assertEqual(session.stop_count, 1)
            verify_sealed_artifact(
                artifact.with_name(artifact.name + ".capture-failure")
            )

    def test_telemetry_stop_failure_is_called_once_and_sealed_nonqualifying(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SequencedJobService(root, ["completed"])
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")
            session = StopFailingTelemetrySession()

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "stop-failure",
                supervision_timeout_seconds=5,
                poll_interval_seconds=0.001,
                telemetry_session=session,
            )

            self.assertEqual(session.stop_count, 1)
            self.assertTrue(outcome.sealed)
            self.assertEqual(outcome.evidence_status, "capture-invalid")
            self.assertEqual(outcome.capture_reason_code, "TELEMETRY_COLLECTOR_FAILURE")

    def test_exact_telemetry_failure_code_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SequencedJobService(root, ["completed"])
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")
            session = ExactFailureTelemetrySession("THERMAL_LIMIT_DISAPPEARED")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "exact-telemetry-failure",
                supervision_timeout_seconds=5,
                poll_interval_seconds=0.001,
                telemetry_session=session,
            )

            self.assertEqual(outcome.capture_reason_code, "THERMAL_LIMIT_DISAPPEARED")
            failed_event = next(
                row
                for row in ledger_records(outcome)
                if row["event_type"] == "telemetry.failed"
            )
            self.assertEqual(failed_event["reason_code"], "THERMAL_LIMIT_DISAPPEARED")

    def test_post_persist_submission_failure_captures_exact_terminal_record(
        self,
    ) -> None:
        class PostPersistFailureService:
            def __init__(self, root: Path) -> None:
                self.job_id = "job_" + "d" * 32
                self.log = root / "post-persist.log"
                self.log.write_text("submission failed\n", encoding="utf-8")

            def submit(self, _bundle_dir: Path, **_kwargs: Any) -> dict[str, Any]:
                terminal = {
                    "id": self.job_id,
                    "job_id": self.job_id,
                    "state": "failed",
                    "log": str(self.log),
                    "return_code": None,
                }
                raise JobSubmissionFailure(self.job_id, terminal, "WORKER_START_FAILED")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = PostPersistFailureService(root)
            harness = harness_at(root, job_service=service)  # type: ignore[arg-type]
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_job(
                root,
                artifact_directory=vault / "post-persist",
                supervision_timeout_seconds=5,
                telemetry_session=HealthyTelemetrySession(),
            )

            terminal = json.loads(
                artifact_file(outcome, "job/terminal.json").decode("utf-8")
            )
            self.assertEqual(terminal["job_id"], service.job_id)
            self.assertEqual(terminal["state"], "failed")
            self.assertEqual(outcome.native_outcome, "failed")
            self.assertFalse(outcome.submission_blocked)

    def test_owned_process_group_snapshot_is_identity_bound_and_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = harness_at(root, job_service=SequencedJobService(root, []))
            running = {
                "state": "running",
                "owner_status": "owning-service",
                "process_pid": 4321,
                "process_group_id": 4321,
                "process_identity": "linux-start-ticks:9876",
            }

            harness._update_managed_pid(JOB_ID, running)

            self.assertEqual(harness.managed_pids(), frozenset({4321}))
            groups = harness.managed_process_groups()
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].process_group_id, 4321)
            self.assertEqual(groups[0].leader_pid, 4321)
            self.assertEqual(groups[0].leader_identity, "linux-start-ticks:9876")

            harness._update_managed_pid(
                JOB_ID,
                {**running, "state": "completed", "owner_status": "terminal"},
            )

            self.assertEqual(harness.managed_pids(), frozenset())
            self.assertEqual(harness.managed_process_groups(), ())

    def test_two_action_success_uses_one_telemetry_ledger_and_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SequencedJobService(root, ["completed", "completed"])
            harness = harness_at(root, job_service=service)
            session = HealthyTelemetrySession()
            vault = private_directory(root / "vault")
            observed_pids: list[frozenset[int]] = []

            outcome = harness.run_managed_sequence(
                root,
                actions=(
                    ManagedActionSpec("dependencies", "dependency", 5),
                    ManagedActionSpec("pilot-phase", "pilot", 5),
                ),
                artifact_directory=vault / "two-actions",
                poll_interval_seconds=0.001,
                telemetry_session=session,
                safety_check=lambda _spec, _record: (
                    observed_pids.append(harness.managed_pids()) or None
                ),
            )

            self.assertTrue(outcome.sealed)
            self.assertEqual(outcome.native_outcome, "passed")
            self.assertEqual(outcome.evidence_status, "capture-invalid")
            self.assertEqual(outcome.capture_reason_code, "MISSING_REQUIRED_EVIDENCE")
            self.assertEqual(service.submissions, ["dependency", "pilot"])
            self.assertEqual(session.start_count, 1)
            self.assertEqual(session.stop_count, 1)
            self.assertTrue(any(values for values in observed_pids))
            self.assertEqual(harness.managed_pids(), frozenset())
            self.assertEqual(
                artifact_file(outcome, "actions/dependencies/full.log"),
                b"complete dependency log\n",
            )
            self.assertEqual(
                artifact_file(outcome, "actions/pilot-phase/full.log"),
                b"complete pilot log\n",
            )
            summary = json.loads(artifact_file(outcome, "sequence/summary.json"))
            self.assertEqual(len(summary["started_actions"]), 2)
            events = ledger_records(outcome)
            event_types = [row["event_type"] for row in events]
            self.assertEqual(event_types.count("telemetry.started"), 1)
            self.assertEqual(event_types.count("telemetry.stopped"), 1)
            self.assertEqual(event_types.count("command.started"), 2)
            self.assertEqual(event_types.count("command.finished"), 2)
            self.assertEqual(event_types.count("pilot.phase-started"), 0)
            self.assertEqual(event_types.count("pilot.phase-finished"), 0)
            bindings = outcome.seal_verification["manifest"][  # type: ignore[index]
                "required_role_bindings"
            ]
            self.assertEqual(len(bindings["job-log"]), 2)
            self.assertEqual(len(bindings["terminal-job-record"]), 2)

    def test_sequence_stops_on_failure_while_capture_remains_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SequencedJobService(root, ["failed", "completed"])
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")

            outcome = harness.run_managed_sequence(
                root,
                actions=(
                    ManagedActionSpec("preflight", "preflight", 5),
                    ManagedActionSpec(
                        "training",
                        "train",
                        5,
                        {"confirm_full_train": True},
                    ),
                ),
                artifact_directory=vault / "stop-on-failure",
                poll_interval_seconds=0.001,
                telemetry_session=HealthyTelemetrySession(),
            )

            self.assertEqual(service.submissions, ["preflight"])
            self.assertEqual(outcome.native_outcome, "failed")
            self.assertEqual(outcome.reason_code, "PROCESS_EXIT_NONZERO")
            self.assertEqual(outcome.evidence_status, "capture-invalid")
            self.assertEqual(outcome.capture_reason_code, "MISSING_REQUIRED_EVIDENCE")
            self.assertTrue(
                (
                    outcome.artifact_directory / "actions/preflight/terminal.json"
                ).is_file()
            )
            self.assertFalse((outcome.artifact_directory / "actions/training").exists())

    def test_missing_telemetry_requires_explicit_nonqualifying_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SequencedJobService(root, ["completed"])
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")
            spec = ManagedActionSpec("preflight", "preflight", 5)

            with self.assertRaisesRegex(ValueError, "requires telemetry"):
                harness.run_managed_sequence(
                    root,
                    actions=(spec,),
                    artifact_directory=vault / "must-reject",
                )
            self.assertEqual(service.submissions, [])

            outcome = harness.run_managed_sequence(
                root,
                actions=(spec,),
                artifact_directory=vault / "nonqualifying-test",
                poll_interval_seconds=0.001,
                allow_nonqualifying_without_telemetry_for_test=True,
            )
            self.assertEqual(outcome.native_outcome, "passed")
            self.assertEqual(outcome.evidence_status, "capture-invalid")
            self.assertEqual(outcome.capture_reason_code, "MISSING_REQUIRED_EVIDENCE")
            self.assertTrue(outcome.sealed)

    def test_telemetry_start_failure_seals_fallback_and_blocks_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SequencedJobService(root, ["completed"])
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")
            artifact = vault / "telemetry-start-failure"

            outcome = harness.run_managed_sequence(
                root,
                actions=(ManagedActionSpec("preflight", "preflight", 5),),
                artifact_directory=artifact,
                telemetry_session=StartFailingTelemetrySession(),
            )

            self.assertEqual(service.submissions, [])
            self.assertEqual(outcome.native_outcome, "guard-blocked")
            self.assertEqual(outcome.evidence_status, "capture-invalid")
            self.assertEqual(outcome.capture_reason_code, "TELEMETRY_COLLECTOR_FAILURE")
            self.assertTrue(outcome.submission_blocked)
            self.assertFalse(outcome.sealed)
            fallback = artifact.with_name(artifact.name + ".capture-failure")
            verify_sealed_artifact(fallback)
            with self.assertRaises(SubmissionBlockedError):
                harness.run_managed_sequence(
                    root,
                    actions=(ManagedActionSpec("retry", "preflight", 5),),
                    artifact_directory=vault / "blocked",
                    telemetry_session=HealthyTelemetrySession(),
                )

    def test_action_labels_and_paths_are_strict_and_unique(self) -> None:
        with self.assertRaises(ValueError):
            ManagedActionSpec("../escape", "preflight", 5)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = SequencedJobService(root, ["completed"])
            harness = harness_at(root, job_service=service)
            vault = private_directory(root / "vault")
            duplicate = ManagedActionSpec("same", "preflight", 5)
            with self.assertRaisesRegex(ValueError, "unique"):
                harness.run_managed_sequence(
                    root,
                    actions=(duplicate, duplicate),
                    artifact_directory=vault / "duplicate",
                    telemetry_session=HealthyTelemetrySession(),
                )
            self.assertEqual(service.submissions, [])


if __name__ == "__main__":
    unittest.main()
