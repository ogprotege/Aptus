"""Opt-in capture harness for the bounded CUDA evidence campaign.

This module is deliberately separate from Aptus's ordinary execution paths.  It
can capture a plain command or supervise an ordered action sequence submitted
through one owning :class:`aptus.execution.JobService` instance.  The harness
never signals an Aptus child PID directly.

Telemetry is injected behind :class:`TelemetrySession`.  That keeps the capture
and cancellation authority stable while the host-specific probe implementation
evolves independently.
"""

from __future__ import annotations

import math
import json
import hashlib
import os
import re
import signal
import stat
import subprocess
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from aptus.execution import (
    ActiveJobError,
    JobPrerequisiteError,
    JobService,
    JobSubmissionFailure,
)
from aptus.generation import verify_bundle_archive

from .admission import (
    ACTIVATION_FILE_NAMES,
    ACTIVATION_SEAL_NAME,
    AdmissionError,
    Phase4CurrentAuthority,
    PlannedSlotContext,
    VerifiedActivatedSlot,
    activate_admitted_slot,
    collect_production_admission_observations,
    evaluate_pre_slot_admission,
    verify_activated_slot,
)
from .contracts import (
    REASON_CODES,
    EventLedgerWriter,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_opaque_id,
    sha256_bytes,
    utc_now,
    validate_safe_relative_path,
)
from .monitoring import (
    ManagedProcessGroup,
    WindowValidation,
    summarize_telemetry,
    validate_cooldown,
    validate_telemetry_sample,
)
from .qualification import (
    QUALIFYING_ACTION_ORDER,
    REQUIRED_QUALIFYING_ARTIFACT_ROLES,
    REQUIRED_QUALIFYING_AUTHORITY_ROLES,
    QualificationError,
    QualifyingRunContext,
    build_segment_summaries,
    evaluate_passing_qualification,
    validate_qualifying_telemetry_configuration,
)
from .phase4 import (
    PHASE4_IDLE_SAMPLES_NAME,
    PHASE4_SOURCE_FREEZE_NAME,
    PHASE4_SOURCE_FREEZE_SEAL_NAME,
    Phase4SourceFreezeVerification,
    verify_phase4_source_freeze_artifact,
)
from .outcomes import OutcomeProfileError, validate_managed_sequence_outcome
from .runtime_events import (
    RuntimeBoundary,
    RuntimeBoundaryError,
    RuntimeBoundaryJournalReader,
)
from .storage import (
    RawArtifactWriter,
    ensure_private_directory,
    write_sealed_capture_failure_artifact,
)


_ATTEMPT_SLOT_ID = re.compile(r"^slot_[0-9a-f]{20}$")
_EXPERIMENT_RUN_ID = re.compile(r"^xrun_[0-9a-f]{32}$")
_JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
_ACTIVE_STATES = frozenset({"queued", "running", "cancelling"})
_MANAGED_ACTIONS = frozenset(
    {"dependency", "model-data", "preflight", "pilot", "train"}
)
_ACTION_LABEL = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_BLOCK_MARKER = "SUBMISSIONS_BLOCKED.json"

_ACTIVATION_PROVENANCE = (
    (
        "admission-decision.json",
        "activation/admission-decision.json",
        "activation-admission-decision",
        "application/json",
    ),
    (
        "admission-observations.json",
        "activation/admission-observations.json",
        "activation-admission-observations",
        "application/json",
    ),
    (
        "execution-configuration.json",
        "activation/execution-configuration.json",
        "activation-execution-configuration",
        "application/json",
    ),
    (
        "experiment-run-template.json",
        "activation/experiment-run-template.json",
        "activation-experiment-run-template",
        "application/json",
    ),
    (
        "started-identity-template.json",
        "activation/started-identity-template.json",
        "activation-started-identity-template",
        "application/json",
    ),
    (
        "activation-decision.json",
        "activation/activation-decision.json",
        "activation-decision",
        "application/json",
    ),
    (
        ACTIVATION_SEAL_NAME,
        "activation/ACTIVATED.json",
        "activation-seal",
        "application/json",
    ),
)
if tuple(item[0] for item in _ACTIVATION_PROVENANCE[:-1]) != ACTIVATION_FILE_NAMES:
    raise RuntimeError("activation provenance mapping differs from admission authority")

_LIVE_STOP_REASON_CODES = frozenset(
    {
        "CUDA_XID",
        "CUDA_OOM",
        "CUDA_DEVICE_RESET",
        "CUDA_DEVICE_LOST",
        "HARDWARE_ERROR",
        "NONFINITE_VALUE",
        "ARTIFACT_INTEGRITY_FAILURE",
        "CHECKPOINT_CONTINUATION_FAILURE",
        "EXPORT_VERIFICATION_FAILURE",
        "THERMAL_THROTTLE",
        "THERMAL_LIMIT_DISAPPEARED",
        "UNRELATED_GPU_ACTIVITY",
        "TELEMETRY_COLLECTOR_FAILURE",
        "WATCHDOG_HEARTBEAT_LOST",
        "OWNERSHIP_UNCERTAIN",
        "EMERGENCY_DEADLINE_EXCEEDED",
        "TELEMETRY_HARD_GAP",
        "THERMAL_STOP_IMMEDIATE",
        "THERMAL_STOP_SUSTAINED",
        "FREE_VRAM_FLOOR",
        "HOST_RAM_FLOOR",
        "DISK_FLOOR",
        "DISK_BUDGET_INSUFFICIENT",
        "SWAP_RATE_LIMIT",
    }
)

CANCEL_REQUEST_SLA_NS = 2_000_000_000
CANCEL_TERMINATION_SLA_NS = 10_000_000_000
CANCEL_RECONCILIATION_SLA_NS = 2_000_000_000
CANCEL_SEQUENCE_SLA_NS = 15_000_000_000
QUALIFYING_COOLDOWN_SAMPLES = 120
QUALIFYING_COOLDOWN_MAXIMUM_WAIT_SECONDS = 1800


class CaptureHarnessError(RuntimeError):
    """The opt-in harness could not preserve its fail-closed contract."""


class SubmissionBlockedError(CaptureHarnessError):
    """A durable uncertainty marker forbids another managed submission."""


class CancellationSLAError(CaptureHarnessError):
    """Persisted cancellation milestones are absent, unordered, or late."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "CANCELLATION_DEADLINE_EXCEEDED",
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class ManagedJobService(Protocol):
    """Narrow surface used from the exact owning ``JobService`` instance."""

    def submit(self, bundle_dir: Path, **kwargs: Any) -> dict[str, Any]: ...

    def get(
        self, job_id: str, *, include_validation_report: bool = True
    ) -> dict[str, Any]: ...

    def cancel(
        self,
        job_id: str,
        *,
        reason_code: str | None = None,
        trigger_detected_monotonic_ns: int | None = None,
    ) -> dict[str, Any]: ...


class TelemetrySession(Protocol):
    """Injected, background telemetry lifecycle.

    ``start`` must return only after its collector and watchdog are ready.
    ``stop`` must join the collector and return every captured sample.
    """

    def start(self, *, experiment_run_id: str, start_monotonic_ns: int) -> None: ...

    def safety_signal(self) -> "SafetySignal | None": ...

    def stop(self, *, stop_monotonic_ns: int) -> "TelemetryCapture": ...


class QualifyingTelemetrySession(TelemetrySession, Protocol):
    """Additional frozen sidecar evidence required for a qualifying run."""

    @property
    def qualifying_profile(self) -> bool: ...

    def configuration_record(self) -> dict[str, Any]: ...

    def snapshot(self) -> Any: ...


@dataclass(frozen=True)
class TelemetryCapture:
    samples: tuple[Mapping[str, Any], ...]
    healthy: bool
    failure_code: str | None = None


@dataclass(frozen=True)
class _StoppedTelemetry:
    payload: bytes | None
    samples: tuple[dict[str, Any], ...]
    healthy: bool | None
    failure_code: str | None
    stop_monotonic_ns: int | None
    configuration: Mapping[str, Any] | None = None
    safety_events: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class SafetySignal:
    reason_code: str
    detected_monotonic_ns: int

    def __post_init__(self) -> None:
        if self.reason_code not in _LIVE_STOP_REASON_CODES:
            raise ValueError("Safety signal reason_code is not a live stop reason.")
        if (
            isinstance(self.detected_monotonic_ns, bool)
            or not isinstance(self.detected_monotonic_ns, int)
            or self.detected_monotonic_ns < 0
        ):
            raise ValueError("Safety signal monotonic timestamp is invalid.")


@dataclass(frozen=True)
class SelectedArtifact:
    source: Path
    relative_path: str
    role: str
    media_type: str = "application/octet-stream"


@dataclass(frozen=True)
class ManagedActionSpec:
    """One exact action in an ordered, continuously monitored attempt."""

    label: str
    action: str
    supervision_timeout_seconds: float
    submit_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not _ACTION_LABEL.fullmatch(self.label):
            raise ValueError("Managed action label is not a safe unique path segment.")
        if self.action not in _MANAGED_ACTIONS:
            raise ValueError("Managed action is not supported by JobService.")
        timeout = _require_timeout(
            self.supervision_timeout_seconds,
            "Managed action supervision_timeout_seconds",
        )
        if not isinstance(self.submit_kwargs, Mapping):
            raise ValueError("Managed action submit_kwargs must be an object.")
        kwargs = dict(self.submit_kwargs)
        allowed = {
            "confirm_full_train",
            "resume_from",
            "expected_artifact_fingerprint",
            "campaign_event_capture",
            "campaign_experiment_run_id",
        }
        if set(kwargs) - allowed or any(not isinstance(key, str) for key in kwargs):
            raise ValueError("Managed action submit_kwargs contain unsupported fields.")
        if "confirm_full_train" in kwargs and not isinstance(
            kwargs["confirm_full_train"], bool
        ):
            raise ValueError("confirm_full_train must be boolean.")
        if "campaign_event_capture" in kwargs and not isinstance(
            kwargs["campaign_event_capture"], bool
        ):
            raise ValueError("campaign_event_capture must be boolean.")
        for name in ("resume_from", "expected_artifact_fingerprint"):
            if (
                name in kwargs
                and kwargs[name] is not None
                and not isinstance(kwargs[name], str)
            ):
                raise ValueError(f"{name} must be null or a string.")
        campaign_run_id = kwargs.get("campaign_experiment_run_id")
        if campaign_run_id is not None and (
            not isinstance(campaign_run_id, str)
            or _EXPERIMENT_RUN_ID.fullmatch(campaign_run_id) is None
        ):
            raise ValueError("campaign_experiment_run_id is invalid.")
        canonical_json_bytes(kwargs)
        object.__setattr__(self, "supervision_timeout_seconds", timeout)
        object.__setattr__(self, "submit_kwargs", MappingProxyType(kwargs))


@dataclass(frozen=True)
class CancellationMilestones:
    trigger_detected_monotonic_ns: int
    cancel_requested_monotonic_ns: int
    process_group_terminated_monotonic_ns: int
    lease_reconciled_monotonic_ns: int


@dataclass(frozen=True)
class CaptureOutcome:
    experiment_run_id: str
    attempt_slot_id: str
    native_outcome: str
    reason_code: str
    evidence_status: str
    capture_reason_code: str
    artifact_directory: Path
    sealed: bool
    seal_verification: Mapping[str, Any] | None
    capture_failure_receipt: Path | None
    exit_code: int | None
    timed_out: bool
    submission_blocked: bool
    telemetry_healthy: bool | None


@dataclass(frozen=True)
class _ManagedActionResult:
    spec: ManagedActionSpec
    job_id: str | None
    record: Mapping[str, Any]
    native_outcome: str
    reason_code: str
    exit_code: int | None
    timed_out: bool
    terminal: bool
    capture_reason_code: str = "NONE"
    runtime_boundaries: tuple[RuntimeBoundary, ...] = ()
    runtime_journal_bytes: bytes | None = None


@dataclass(frozen=True)
class _Payload:
    data: bytes | None
    source: Path | None
    relative_path: str
    role: str
    media_type: str
    entry_id: str | None = None
    source_descriptor: int | None = None
    source_fingerprint: tuple[int, ...] | None = None


def _require_identifier(value: str, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{label} is invalid.")
    return value


def _require_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("argv must be a non-empty sequence of strings.")
    exact = tuple(argv)
    if any(not isinstance(item, str) or not item or "\x00" in item for item in exact):
        raise ValueError("argv contains an invalid argument.")
    return exact


def _require_timeout(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be positive and finite.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite.")
    return result


def _exclusive_private_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(dict(value))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive OS failure.
                raise OSError("Submission-block marker write made no progress.")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _read_stable_regular_source(path: Path) -> bytes:
    """Read one selected source without following links or accepting a swap."""

    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError("Selected artifact source is unavailable.") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        raise ValueError("Selected artifact source is not a regular private file.")
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_identity != identity:
            raise ValueError("Selected artifact source changed during open.")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
        finished = os.fstat(descriptor)
        finished_identity = (
            finished.st_dev,
            finished.st_ino,
            finished.st_mode,
            finished.st_nlink,
            finished.st_size,
            finished.st_mtime_ns,
            finished.st_ctime_ns,
        )
        if finished_identity != identity:
            raise ValueError("Selected artifact source changed during read.")
    finally:
        os.close(descriptor)
    after = path.lstat()
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if after_identity != identity:
        raise ValueError("Selected artifact path changed during read.")
    return bytes(payload)


def _hash_stable_regular_source(path: Path) -> str:
    """Hash one selected source through the same no-follow identity checks."""

    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError("Selected artifact source is unavailable.") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        raise ValueError("Selected artifact source is not a regular private file.")
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        if (
            lambda item: (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_nlink,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
        )(os.fstat(descriptor)) != identity:
            raise ValueError("Selected artifact source changed during open.")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        finished = os.fstat(descriptor)
        if (
            finished.st_dev,
            finished.st_ino,
            finished.st_mode,
            finished.st_nlink,
            finished.st_size,
            finished.st_mtime_ns,
            finished.st_ctime_ns,
        ) != identity:
            raise ValueError("Selected artifact source changed during read.")
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != identity:
        raise ValueError("Selected artifact path changed during read.")
    return digest.hexdigest()


def _stable_regular_source_identity(path: Path) -> tuple[int, ...]:
    """Return the exact no-link path identity used for a later source rewalk."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError("Selected artifact source is unavailable.") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError("Selected artifact source is not a regular private file.")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def verify_cancellation_milestones(
    record: Mapping[str, Any],
    *,
    reason_code: str,
    trigger_detected_monotonic_ns: int,
) -> CancellationMilestones:
    """Verify the frozen 2s/10s/2s/15s persisted cancellation SLA."""

    if record.get("cancel_reason_code") != reason_code:
        raise CancellationSLAError("Persisted cancellation reason does not match.")
    if (
        record.get("cancel_trigger_detected_monotonic_ns")
        != trigger_detected_monotonic_ns
    ):
        raise CancellationSLAError("Persisted trigger timestamp does not match.")
    names = (
        "cancel_requested_monotonic_ns",
        "process_group_terminated_monotonic_ns",
        "lease_reconciled_monotonic_ns",
    )
    values: list[int] = []
    for name in names:
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CancellationSLAError(
                f"Persisted milestone is missing: {name}.",
                reason_code=(
                    "LEASE_RECONCILIATION_FAILURE"
                    if name == "lease_reconciled_monotonic_ns"
                    else "CANCELLATION_DEADLINE_EXCEEDED"
                ),
            )
        values.append(value)
    requested, terminated, reconciled = values
    if not trigger_detected_monotonic_ns <= requested <= terminated <= reconciled:
        raise CancellationSLAError("Persisted cancellation milestones are unordered.")
    if requested - trigger_detected_monotonic_ns > CANCEL_REQUEST_SLA_NS:
        raise CancellationSLAError("Cancellation request exceeded its two-second SLA.")
    if terminated - requested > CANCEL_TERMINATION_SLA_NS:
        raise CancellationSLAError("Process termination exceeded its ten-second SLA.")
    if reconciled - terminated > CANCEL_RECONCILIATION_SLA_NS:
        raise CancellationSLAError(
            "Lease reconciliation exceeded its two-second SLA.",
            reason_code="LEASE_RECONCILIATION_FAILURE",
        )
    if reconciled - trigger_detected_monotonic_ns > CANCEL_SEQUENCE_SLA_NS:
        raise CancellationSLAError("Cancellation sequence exceeded fifteen seconds.")
    return CancellationMilestones(
        trigger_detected_monotonic_ns=trigger_detected_monotonic_ns,
        cancel_requested_monotonic_ns=requested,
        process_group_terminated_monotonic_ns=terminated,
        lease_reconciled_monotonic_ns=reconciled,
    )


def _terminate_plain_process(process: subprocess.Popen[Any]) -> None:
    """Stop only the process group created by this harness and wait for it."""

    process_group_id = process.pid if os.name == "posix" else None
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if process_group_id is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:  # pragma: no cover - exercised on Windows CI only.
            process.kill()
        process.wait()
    if process.poll() is None:
        raise CaptureHarnessError("Plain command leader was not waitable.")
    if process_group_id is not None:
        deadline = time.monotonic() + 2.0
        while True:
            try:
                os.killpg(process_group_id, 0)
            except ProcessLookupError:
                break
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                raise CaptureHarnessError(
                    "Plain command process group survived forced termination."
                )
            time.sleep(0.01)


def _create_command_spool(state_root: Path) -> tuple[Path, int]:
    name = f".{new_opaque_id('artifact')}.command-output.spool"
    path = state_root / name
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
            or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600)
        ):
            raise CaptureHarnessError("Command output spool identity is unsafe.")
        return path, descriptor
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def _finalize_command_spool(path: Path, descriptor: int) -> tuple[int, ...]:
    """Fsync the output while retaining its creation-time descriptor."""

    os.fsync(descriptor)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
        or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600)
    ):
        raise CaptureHarnessError("Command output spool changed before capture.")
    fingerprint = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    try:
        path_metadata = path.lstat()
    except OSError as error:
        raise CaptureHarnessError(
            "Command output spool path disappeared before capture."
        ) from error
    path_fingerprint = (
        path_metadata.st_dev,
        path_metadata.st_ino,
        path_metadata.st_mode,
        path_metadata.st_nlink,
        path_metadata.st_uid,
        path_metadata.st_gid,
        path_metadata.st_size,
        path_metadata.st_mtime_ns,
        path_metadata.st_ctime_ns,
    )
    if path_fingerprint != fingerprint:
        raise CaptureHarnessError("Command output spool path was replaced.")
    return fingerprint


def _discard_command_spool(path: Path, descriptor: int) -> None:
    try:
        os.close(descriptor)
    finally:
        path.unlink(missing_ok=True)


class CaptureHarness:
    """Build one protected, sealed evidence artifact per opt-in invocation."""

    def __init__(
        self,
        state_root: Path,
        *,
        attempt_slot_id: str,
        experiment_run_id: str,
        provisional_retain_not_before_utc: str,
        job_service: ManagedJobService | JobService | None = None,
        capture_tool: Mapping[str, Any] | None = None,
        source_bindings: Mapping[str, Any] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_time: Callable[[], str] = utc_now,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.state_root = ensure_private_directory(state_root)
        self.attempt_slot_id = _require_identifier(
            attempt_slot_id, _ATTEMPT_SLOT_ID, "attempt_slot_id"
        )
        self.experiment_run_id = _require_identifier(
            experiment_run_id, _EXPERIMENT_RUN_ID, "experiment_run_id"
        )
        self.provisional_retain_not_before_utc = provisional_retain_not_before_utc
        self.job_service = job_service
        self.capture_tool = dict(
            capture_tool or {"name": "aptus-cuda-campaign-harness", "version": "v1"}
        )
        self.qualification_context: QualifyingRunContext | None = None
        self._qualifying_authority: object | None = None
        self._qualifying_job_service: JobService | None = None
        self._phase4_verification: Phase4SourceFreezeVerification | None = None
        self._phase4_repository_root: Path | None = None
        self._phase4_nvidia_smi_path: str | None = None
        self._phase4_gpu_index = 0
        self._planned_slot_context: PlannedSlotContext | None = None
        self._activation_authority: Phase4CurrentAuthority | None = None
        self._activation_directory: Path | None = None
        self._admission_filesystem_device: int | None = None
        self.source_bindings = dict(source_bindings or {})
        self._monotonic_ns = monotonic_ns
        self._wall_time = wall_time
        self._sleep = sleep
        self._managed_lock = threading.Lock()
        self._managed_pid_lock = threading.Lock()
        self._managed_pids_by_job: dict[str, int] = {}
        self._managed_process_groups_by_job: dict[str, ManagedProcessGroup] = {}
        self._known_active_job_ids: set[str] = set()

    @classmethod
    def for_qualifying_campaign(
        cls,
        state_root: Path,
        job_root: Path,
        *,
        planned_slot_context: PlannedSlotContext,
        provisional_retain_not_before_utc: str,
        phase4_source_freeze_directory: Path,
        repository_root: Path,
        runtime_environment: Mapping[str, str] | None = None,
        nvidia_smi_path: str | None = None,
        gpu_index: int = 0,
    ) -> "CaptureHarness":
        """Create the only harness eligible to evaluate qualifying evidence.

        The factory owns the concrete ``JobService``, production admission,
        sealed activation, activation re-verification, real clock boundaries,
        and a per-harness object-identity authority.  No execution or run ID is
        accepted from the caller.
        """

        if cls is not CaptureHarness:
            raise TypeError("Qualifying harness subclasses are not supported.")
        if type(planned_slot_context) is not PlannedSlotContext:
            raise TypeError("planned_slot_context must be a PlannedSlotContext.")
        declared_state = Path(
            planned_slot_context.run_proposal.fresh_state_root
        ).resolve()
        if (
            state_root.resolve() != declared_state
            or job_root.resolve() != declared_state
        ):
            raise ValueError(
                "Qualifying harness and JobService roots must equal fresh_state_root."
            )
        private_state = ensure_private_directory(state_root)
        try:
            bundle = Path(planned_slot_context.run_proposal.bundle_path).resolve(
                strict=True
            )
            repository = repository_root.resolve(strict=True)
        except OSError as error:
            raise AdmissionError(
                "qualifying admission paths are unavailable"
            ) from error
        if not bundle.is_dir() or not repository.is_dir():
            raise AdmissionError("qualifying admission paths must be directories")
        if (
            Path(planned_slot_context.run_proposal.working_directory).resolve()
            != bundle
            or Path(planned_slot_context.run_proposal.output_path).resolve()
            != bundle / "runs"
        ):
            raise AdmissionError(
                "qualifying run storage paths are not bound to the bundle"
            )
        admission_filesystem_device = bundle.stat().st_dev
        if private_state.stat().st_dev != admission_filesystem_device:
            raise AdmissionError(
                "qualifying state and output storage are on different filesystems"
            )
        service = JobService(private_state, runtime_environment=runtime_environment)
        activation_authority = Phase4CurrentAuthority(
            directory=phase4_source_freeze_directory,
            repository_root=repository,
            campaign=planned_slot_context.campaign,
            comparison_cohort=planned_slot_context.comparison_cohort,
            comparison_cell=planned_slot_context.comparison_cell,
            nvidia_smi_path=nvidia_smi_path,
            gpu_index=gpu_index,
        )
        observations = collect_production_admission_observations(
            planned_slot_context,
            authority=activation_authority,
            filesystem_path=bundle,
            job_service=service,
            gpu_index=gpu_index,
            nvidia_smi_path=nvidia_smi_path,
        )
        admission = evaluate_pre_slot_admission(
            planned_slot_context,
            observations,
            authority=activation_authority,
        )
        if not admission.admitted:
            raise AdmissionError(
                "qualifying slot was not admitted: "
                + ",".join(admission.decision["reason_codes"])
            )
        activation_directory = private_state / "qualifying-activation"
        activate_admitted_slot(
            admission,
            authority=activation_authority,
            destination=activation_directory,
        )
        verified_activation = verify_activated_slot(
            activation_directory,
            expected_context=planned_slot_context,
            authority=activation_authority,
        )
        if (
            type(verified_activation) is not VerifiedActivatedSlot
            or verified_activation.production_qualifying is not True
            or not verified_activation.authorized_for_qualifying_harness()
        ):
            raise AdmissionError("qualifying activation verification failed")
        qualification_context = QualifyingRunContext(
            planned_slot_context, verified_activation
        )
        verification = verify_phase4_source_freeze_artifact(
            phase4_source_freeze_directory,
            repository_root=repository,
            campaign=qualification_context.campaign,
            comparison_cohort=qualification_context.comparison_cohort,
            comparison_cell=qualification_context.comparison_cell,
            nvidia_smi_path=nvidia_smi_path,
            gpu_index=gpu_index,
        )
        if dict(verification.baseline_binding) != dict(
            qualification_context.idle_baseline_binding
        ):
            raise ValueError(
                "Qualifying context does not bind the verified Phase-4 baseline."
            )
        authority = object()
        harness = cls(
            private_state,
            attempt_slot_id=qualification_context.attempt_slot_id,
            experiment_run_id=qualification_context.experiment_run_id,
            provisional_retain_not_before_utc=provisional_retain_not_before_utc,
            job_service=service,
        )
        harness.capture_tool = {
            "name": "aptus-cuda-campaign-qualifying-harness",
            "version": "v1",
        }
        harness.qualification_context = qualification_context
        harness._qualifying_authority = authority
        harness._qualifying_job_service = service
        harness._phase4_verification = verification
        harness._phase4_repository_root = repository
        harness._phase4_nvidia_smi_path = nvidia_smi_path
        harness._phase4_gpu_index = gpu_index
        harness._planned_slot_context = planned_slot_context
        harness._activation_authority = activation_authority
        harness._activation_directory = activation_directory
        harness._admission_filesystem_device = admission_filesystem_device
        harness.source_bindings.update(qualification_context.source_bindings())
        harness.source_bindings.update(
            phase4_source_freeze_sha256=verification.source_freeze_sha256,
            phase4_source_freeze_seal_sha256=verification.seal_sha256,
            phase4_idle_baseline_samples_sha256=verification.samples_sha256,
        )
        return harness

    def create_qualifying_telemetry_session(
        self,
        *,
        filesystem_path: Path,
        gpu_index: int = 0,
        nvidia_smi_path: str | None = None,
        unavailable_optional_sensors: Sequence[str],
    ) -> TelemetrySession:
        """Construct the exact real-clock Linux/NVIDIA qualifying sidecar.

        The only caller declarations are the filesystem/GPU selection and an
        exact reviewed declaration that the optional CPU, NVMe, and reported
        GPU thermal-limit channels are unavailable. All safety-critical
        providers are concrete and harness-owned.
        """

        context = self.qualification_context
        authority = self._qualifying_authority
        service = self.job_service
        phase4 = self._phase4_verification
        if (
            context is None
            or authority is None
            or not _qualifying_harness_is_registered(self, authority)
            or type(service) is not JobService
            or service is not self._qualifying_job_service
            or phase4 is None
        ):
            raise CaptureHarnessError(
                "Only a qualifying campaign harness may request qualifying telemetry."
            )
        if isinstance(unavailable_optional_sensors, (str, bytes)):
            raise ValueError("Optional sensor declarations must be exact names.")
        declarations = tuple(unavailable_optional_sensors)
        from .sidecar import BackgroundTelemetrySession

        session = BackgroundTelemetrySession._qualifying_for_harness(
            harness=self,
            authority=authority,
            filesystem_path=filesystem_path,
            gpu_index=gpu_index,
            nvidia_smi_path=nvidia_smi_path,
            unavailable_optional_sensors=declarations,
        )
        if (
            session.configuration_record()
            != phase4.source_freeze["telemetry_configuration"]
        ):
            raise CaptureHarnessError(
                "Current telemetry configuration differs from the Phase-4 freeze."
            )
        return session

    def _qualifying_ownership_certain(self) -> bool:
        """Project only factory-owned service and durable uncertainty state."""

        with self._managed_pid_lock:
            groups = tuple(self._managed_process_groups_by_job.values())
            pids = frozenset(self._managed_pids_by_job.values())
        group_leaders = frozenset(group.leader_pid for group in groups)
        return bool(
            self.qualification_context is not None
            and self._qualifying_authority is not None
            and _qualifying_harness_is_registered(self, self._qualifying_authority)
            and type(self.job_service) is JobService
            and self.job_service is self._qualifying_job_service
            and not self.submissions_blocked
            and (os.name != "posix" or group_leaders == pids)
        )

    def _authorized_for_qualifying_factory(self, authority: object) -> bool:
        """Return closure-registry authority for the exact factory-built harness."""

        return _qualifying_harness_is_registered(self, authority)

    @classmethod
    def with_job_root(
        cls,
        state_root: Path,
        job_root: Path,
        *,
        attempt_slot_id: str,
        experiment_run_id: str,
        provisional_retain_not_before_utc: str,
        runtime_environment: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "CaptureHarness":
        """Construct the actual owning ``JobService`` used for submission."""

        service = JobService(job_root, runtime_environment=runtime_environment)
        return cls(
            state_root,
            attempt_slot_id=attempt_slot_id,
            experiment_run_id=experiment_run_id,
            provisional_retain_not_before_utc=provisional_retain_not_before_utc,
            job_service=service,
            **kwargs,
        )

    @property
    def submission_block_path(self) -> Path:
        return self.state_root / _BLOCK_MARKER

    @property
    def submissions_blocked(self) -> bool:
        path = self.submission_block_path
        return path.exists() or path.is_symlink()

    def managed_pids(self) -> frozenset[int]:
        """Return exact PIDs from currently owned, active JobService records."""

        with self._managed_pid_lock:
            return frozenset(self._managed_pids_by_job.values())

    def managed_process_groups(self) -> tuple[ManagedProcessGroup, ...]:
        """Return identity-bound groups from active records owned by this service.

        This is a read-only telemetry snapshot. Cancellation remains exclusively
        delegated to the exact owning ``JobService`` and exact Aptus job ID.
        """

        with self._managed_pid_lock:
            return tuple(
                sorted(
                    self._managed_process_groups_by_job.values(),
                    key=lambda group: group.process_group_id,
                )
            )

    def _update_managed_pid(self, job_id: str, record: Mapping[str, Any]) -> None:
        state = record.get("state")
        owner = record.get("owner_status")
        pid = record.get("process_pid")
        process_group_id = record.get("process_group_id")
        process_identity = record.get("process_identity")
        trusted = bool(
            state in _ACTIVE_STATES
            and owner == "owning-service"
            and isinstance(pid, int)
            and not isinstance(pid, bool)
            and pid > 0
        )
        with self._managed_pid_lock:
            if trusted:
                self._managed_pids_by_job[job_id] = pid
            else:
                self._managed_pids_by_job.pop(job_id, None)
            if (
                trusted
                and isinstance(process_group_id, int)
                and not isinstance(process_group_id, bool)
                and process_group_id > 0
                and isinstance(process_identity, str)
            ):
                try:
                    group = ManagedProcessGroup(
                        process_group_id=process_group_id,
                        leader_pid=pid,
                        leader_identity=process_identity,
                    )
                except ValueError:
                    self._managed_process_groups_by_job.pop(job_id, None)
                else:
                    self._managed_process_groups_by_job[job_id] = group
            else:
                self._managed_process_groups_by_job.pop(job_id, None)

    def _record_known_active_job(self, job_id: str) -> None:
        with self._managed_pid_lock:
            self._known_active_job_ids.add(job_id)

    def _clear_known_active_job(self, job_id: str) -> None:
        with self._managed_pid_lock:
            self._known_active_job_ids.discard(job_id)

    def _cancel_known_active_jobs(
        self, service: ManagedJobService | JobService
    ) -> bool:
        with self._managed_pid_lock:
            job_ids = tuple(sorted(self._known_active_job_ids))
        all_terminal = True
        for job_id in job_ids:
            detected_ns = self._monotonic_ns()
            try:
                cancelled = service.cancel(
                    job_id,
                    reason_code="OWNERSHIP_UNCERTAIN",
                    trigger_detected_monotonic_ns=detected_ns,
                )
                if (
                    not isinstance(cancelled, Mapping)
                    or cancelled.get("id") != job_id
                    or cancelled.get("job_id") != job_id
                    or cancelled.get("state") not in _TERMINAL_STATES
                ):
                    raise CancellationSLAError(
                        "Unexpected cleanup did not return the exact terminal job."
                    )
                verify_cancellation_milestones(
                    cancelled,
                    reason_code="OWNERSHIP_UNCERTAIN",
                    trigger_detected_monotonic_ns=detected_ns,
                )
            except Exception:
                all_terminal = False
            else:
                self._clear_known_active_job(job_id)
                self._update_managed_pid(job_id, {})
        return all_terminal

    def _require_submission_allowed(self) -> None:
        if self.submissions_blocked:
            raise SubmissionBlockedError(
                "Managed submissions are blocked by a durable uncertainty marker."
            )

    def _block_submissions(
        self,
        *,
        reason_code: str,
        job_id: str | None,
        detected_monotonic_ns: int,
    ) -> None:
        marker = {
            "marker_kind": "aptus-cuda-campaign-submission-block-v1",
            "reason_code": reason_code,
            "experiment_run_id": self.experiment_run_id,
            "attempt_slot_id": self.attempt_slot_id,
            "job_id": job_id,
            "detected_monotonic_ns": detected_monotonic_ns,
            "created_at_utc": self._wall_time(),
            "manual_review_required": True,
        }
        try:
            _exclusive_private_json(self.submission_block_path, marker)
        except FileExistsError:
            return

    def _new_ledger(self) -> EventLedgerWriter:
        ledger = EventLedgerWriter(self.experiment_run_id)
        ledger.append(
            monotonic_ns=self._monotonic_ns(),
            wall_time_utc=self._wall_time(),
            event_type="clock.mapping",
            observation_kind="observed",
        )
        ledger.append(
            monotonic_ns=self._monotonic_ns(),
            wall_time_utc=self._wall_time(),
            event_type="harness.started",
            subject_kind="experiment-run",
            subject_id=self.experiment_run_id,
        )
        return ledger

    def _finish_ledger(
        self,
        ledger: EventLedgerWriter,
        *,
        native_outcome: str,
        reason_code: str,
        exit_code: int | None,
    ) -> bytes:
        last_monotonic_ns = ledger.records[-1]["monotonic_ns"]
        finished_monotonic_ns = max(self._monotonic_ns(), last_monotonic_ns)
        ledger.append(
            monotonic_ns=finished_monotonic_ns,
            wall_time_utc=self._wall_time(),
            event_type="harness.finished",
            subject_kind="experiment-run",
            subject_id=self.experiment_run_id,
            exit_code=exit_code,
            native_outcome=native_outcome,
            reason_code=reason_code,
        )
        mapping_monotonic_ns = max(self._monotonic_ns(), finished_monotonic_ns)
        ledger.append(
            monotonic_ns=mapping_monotonic_ns,
            wall_time_utc=self._wall_time(),
            event_type="clock.mapping",
            observation_kind="observed",
        )
        seal_monotonic_ns = max(self._monotonic_ns(), mapping_monotonic_ns)
        ledger.append(
            monotonic_ns=seal_monotonic_ns,
            wall_time_utc=self._wall_time(),
            event_type="seal.started",
            subject_kind="experiment-run",
            subject_id=self.experiment_run_id,
            exit_code=exit_code,
            native_outcome=native_outcome,
            reason_code=reason_code,
        )
        return ledger.to_bytes()

    def _start_telemetry(
        self,
        ledger: EventLedgerWriter,
        session: TelemetrySession | None,
        started_ns: int,
    ) -> bool:
        if session is None:
            return False
        session.start(
            experiment_run_id=self.experiment_run_id,
            start_monotonic_ns=started_ns,
        )
        ledger.append(
            monotonic_ns=started_ns,
            wall_time_utc=self._wall_time(),
            event_type="telemetry.started",
            subject_kind="experiment-run",
            subject_id=self.experiment_run_id,
        )
        return True

    def _stop_telemetry(
        self,
        ledger: EventLedgerWriter,
        session: TelemetrySession | None,
        started: bool,
    ) -> tuple[bytes | None, bool | None, str | None]:
        result = self._stop_telemetry_detailed(ledger, session, started)
        return result.payload, result.healthy, result.failure_code

    def _stop_telemetry_detailed(
        self,
        ledger: EventLedgerWriter,
        session: TelemetrySession | None,
        started: bool,
    ) -> _StoppedTelemetry:
        if session is None or not started:
            return _StoppedTelemetry(None, (), None, None, None)
        stopped_ns = max(self._monotonic_ns(), ledger.records[-1]["monotonic_ns"])
        failure_code: str | None = None
        configuration: Mapping[str, Any] | None = None
        safety_events: tuple[Mapping[str, Any], ...] = ()
        try:
            settle_stop_boundary = getattr(session, "settle_stop_boundary", None)
            if callable(settle_stop_boundary):
                settle_stop_boundary()
                stopped_ns = max(
                    self._monotonic_ns(), ledger.records[-1]["monotonic_ns"]
                )
            capture = session.stop(stop_monotonic_ns=stopped_ns)
            samples = [validate_telemetry_sample(sample) for sample in capture.samples]
            payload = canonical_jsonl_bytes(samples)
            raw_configuration = getattr(capture, "configuration", None)
            if isinstance(raw_configuration, Mapping):
                configuration = dict(raw_configuration)
            raw_safety_events = getattr(capture, "safety_events", ())
            if isinstance(raw_safety_events, (tuple, list)) and all(
                isinstance(item, Mapping) for item in raw_safety_events
            ):
                safety_events = tuple(dict(item) for item in raw_safety_events)
            healthy = bool(samples and capture.healthy and capture.failure_code is None)
            if not healthy:
                failure_code = (
                    capture.failure_code
                    if capture.failure_code in REASON_CODES
                    and capture.failure_code != "NONE"
                    else "TELEMETRY_COLLECTOR_FAILURE"
                )
        except Exception as error:
            payload = None
            samples = []
            healthy = False
            observed_code = getattr(error, "code", None)
            failure_code = (
                observed_code
                if observed_code in REASON_CODES and observed_code != "NONE"
                else "TELEMETRY_COLLECTOR_FAILURE"
            )
        ledger.append(
            monotonic_ns=stopped_ns,
            wall_time_utc=self._wall_time(),
            event_type="telemetry.stopped" if healthy else "telemetry.failed",
            subject_kind="experiment-run",
            subject_id=self.experiment_run_id,
            reason_code="NONE" if healthy else failure_code,
        )
        return _StoppedTelemetry(
            payload,
            tuple(samples),
            healthy,
            failure_code,
            stopped_ns,
            configuration,
            safety_events,
        )

    def _seal_payloads(
        self,
        artifact_directory: Path,
        *,
        payloads: Sequence[_Payload],
        required_role_bindings: Mapping[str, str | Sequence[str]],
        native_outcome: str,
        reason_code: str,
        exit_code: int | None,
        timed_out: bool,
        telemetry_healthy: bool | None,
        evidence_status: str,
        capture_reason_code: str,
    ) -> CaptureOutcome:
        artifact_id = new_opaque_id("artifact")
        writer = RawArtifactWriter(
            artifact_directory,
            protected_artifact_id=artifact_id,
            record_kind="experiment-run",
            identity_bindings={
                "attempt_slot_id": self.attempt_slot_id,
                "experiment_run_id": self.experiment_run_id,
                "capture_kind": "command",
                "capture_status": "complete",
                "evidence_status": evidence_status,
                "capture_reason_code": capture_reason_code,
            },
            capture_tool=self.capture_tool,
            source_bindings=self.source_bindings,
            provisional_retain_not_before_utc=(self.provisional_retain_not_before_utc),
            required_role_bindings=required_role_bindings,
        )
        available: list[dict[str, Any]] = []
        failure_reason = "STREAM_CAPTURE_FAILURE"
        try:
            for payload in payloads:
                if payload.data is not None:
                    entry = writer.write_payload(
                        payload.data,
                        payload.relative_path,
                        role=payload.role,
                        media_type=payload.media_type,
                        entry_id=payload.entry_id,
                    )
                elif payload.source is not None:
                    if (
                        payload.source_descriptor is not None
                        and payload.source_fingerprint is not None
                    ):
                        entry = writer.copy_payload_from_descriptor(
                            payload.source_descriptor,
                            payload.source,
                            payload.source_fingerprint,
                            payload.relative_path,
                            role=payload.role,
                            media_type=payload.media_type,
                            entry_id=payload.entry_id,
                        )
                    else:
                        entry = writer.copy_payload(
                            payload.source,
                            payload.relative_path,
                            role=payload.role,
                            media_type=payload.media_type,
                            entry_id=payload.entry_id,
                        )
                else:  # pragma: no cover - internal construction invariant.
                    raise AssertionError("Capture payload has no source.")
                available.append(entry)
            failure_reason = "SEAL_FAILURE"
            verification = writer.seal()
        except Exception:
            return self._sealed_capture_failure_outcome(
                artifact_directory,
                protected_artifact_id=artifact_id,
                native_outcome=native_outcome,
                reason_code=reason_code,
                capture_reason_code=failure_reason,
                exit_code=exit_code,
                timed_out=timed_out,
                telemetry_healthy=telemetry_healthy,
                available_files=available,
                missing_fields=("SEALED.json", "raw-manifest.json"),
                recoverable_locator=str(writer.directory),
            )
        return CaptureOutcome(
            experiment_run_id=self.experiment_run_id,
            attempt_slot_id=self.attempt_slot_id,
            native_outcome=native_outcome,
            reason_code=reason_code,
            evidence_status=evidence_status,
            capture_reason_code=capture_reason_code,
            artifact_directory=writer.directory,
            sealed=True,
            seal_verification=verification,
            capture_failure_receipt=None,
            exit_code=exit_code,
            timed_out=timed_out,
            submission_blocked=self.submissions_blocked,
            telemetry_healthy=telemetry_healthy,
        )

    def _sealed_capture_failure_outcome(
        self,
        artifact_directory: Path,
        *,
        protected_artifact_id: str,
        native_outcome: str,
        reason_code: str,
        capture_reason_code: str,
        exit_code: int | None,
        timed_out: bool,
        telemetry_healthy: bool | None,
        available_files: Sequence[Mapping[str, Any]] = (),
        missing_fields: Sequence[str],
        recoverable_locator: str | None,
        block_submissions: bool = False,
    ) -> CaptureOutcome:
        """Seal the one permitted fallback or durably stop later submissions."""

        fallback_directory = artifact_directory.with_name(
            artifact_directory.name + ".capture-failure"
        )
        try:
            write_sealed_capture_failure_artifact(
                fallback_directory,
                protected_artifact_id=protected_artifact_id,
                attempt_slot_id=self.attempt_slot_id,
                experiment_run_id=self.experiment_run_id,
                reason_code=capture_reason_code,
                available_files=available_files,
                missing_fields=missing_fields,
                recoverable_locator=recoverable_locator,
                capture_tool=self.capture_tool,
                source_bindings=self.source_bindings,
                provisional_retain_not_before_utc=(
                    self.provisional_retain_not_before_utc
                ),
            )
            if block_submissions:
                self._block_submissions(
                    reason_code=capture_reason_code,
                    job_id=None,
                    detected_monotonic_ns=self._monotonic_ns(),
                )
        except Exception:
            self._block_submissions(
                reason_code="SEAL_FAILURE",
                job_id=None,
                detected_monotonic_ns=self._monotonic_ns(),
            )
            raise CaptureHarnessError(
                "Normal capture and its sealed fallback both failed; later submissions are blocked."
            ) from None
        return CaptureOutcome(
            experiment_run_id=self.experiment_run_id,
            attempt_slot_id=self.attempt_slot_id,
            native_outcome=native_outcome,
            reason_code=reason_code,
            evidence_status="capture-invalid",
            capture_reason_code=capture_reason_code,
            artifact_directory=artifact_directory,
            sealed=False,
            seal_verification=None,
            capture_failure_receipt=fallback_directory / "capture-failure.json",
            exit_code=exit_code,
            timed_out=timed_out,
            submission_blocked=self.submissions_blocked,
            telemetry_healthy=telemetry_healthy,
        )

    def run_command(
        self,
        argv: Sequence[str],
        *,
        artifact_directory: Path,
        working_directory: Path,
        timeout_seconds: float,
        selected_artifacts: Sequence[SelectedArtifact] = (),
        telemetry_session: TelemetrySession | None = None,
    ) -> CaptureOutcome:
        """Run exact argv without a shell and retain byte-complete combined output."""

        exact_argv = _require_argv(argv)
        timeout = _require_timeout(timeout_seconds, "timeout_seconds")
        cwd = working_directory.resolve(strict=True)
        if not cwd.is_dir():
            raise ValueError("working_directory is not a directory.")
        ledger = self._new_ledger()
        capture_started_ns = self._monotonic_ns()
        try:
            telemetry_started = self._start_telemetry(
                ledger, telemetry_session, capture_started_ns
            )
        except Exception:
            return self._sealed_capture_failure_outcome(
                artifact_directory,
                protected_artifact_id=new_opaque_id("artifact"),
                native_outcome="guard-blocked",
                reason_code="TELEMETRY_COLLECTOR_FAILURE",
                capture_reason_code="TELEMETRY_COLLECTOR_FAILURE",
                exit_code=None,
                timed_out=False,
                telemetry_healthy=False,
                missing_fields=(
                    "command/record.json",
                    "command/combined-output.bin",
                    "events/events.jsonl",
                    "telemetry/samples.jsonl",
                ),
                recoverable_locator=None,
                block_submissions=True,
            )
        started_ns = self._monotonic_ns()
        ledger.append(
            monotonic_ns=started_ns,
            wall_time_utc=self._wall_time(),
            event_type="command.started",
            action="command",
            subject_kind="process",
            subject_id=self.experiment_run_id,
        )
        process_options: dict[str, Any] = {}
        if os.name == "posix":
            process_options["start_new_session"] = True
        elif os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            spool_path, spool_descriptor = _create_command_spool(self.state_root)
        except Exception:
            _telemetry_payload, telemetry_healthy, _telemetry_failure = (
                self._stop_telemetry(ledger, telemetry_session, telemetry_started)
            )
            return self._sealed_capture_failure_outcome(
                artifact_directory,
                protected_artifact_id=new_opaque_id("artifact"),
                native_outcome="guard-blocked",
                reason_code="UNKNOWN_TERMINAL_STATE",
                capture_reason_code="STREAM_CAPTURE_FAILURE",
                exit_code=None,
                timed_out=False,
                telemetry_healthy=telemetry_healthy,
                missing_fields=(
                    "command/combined-output.bin",
                    "events/events.jsonl",
                ),
                recoverable_locator=None,
            )
        try:
            process = subprocess.Popen(
                exact_argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=spool_descriptor,
                stderr=subprocess.STDOUT,
                shell=False,
                **process_options,
            )
        except Exception:
            _discard_command_spool(spool_path, spool_descriptor)
            _telemetry_payload, telemetry_healthy, _telemetry_failure = (
                self._stop_telemetry(ledger, telemetry_session, telemetry_started)
            )
            return self._sealed_capture_failure_outcome(
                artifact_directory,
                protected_artifact_id=new_opaque_id("artifact"),
                native_outcome="unknown",
                reason_code="UNKNOWN_TERMINAL_STATE",
                capture_reason_code="STREAM_CAPTURE_FAILURE",
                exit_code=None,
                timed_out=False,
                telemetry_healthy=telemetry_healthy,
                missing_fields=(
                    "command/record.json",
                    "command/combined-output.bin",
                    "events/events.jsonl",
                ),
                recoverable_locator=None,
            )
        timed_out = False
        try:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            _terminate_plain_process(process)
        except Exception:
            termination_certain = True
            try:
                _terminate_plain_process(process)
            except Exception:
                termination_certain = False
            try:
                _finalize_command_spool(spool_path, spool_descriptor)
            except Exception:
                pass
            _discard_command_spool(spool_path, spool_descriptor)
            _telemetry_payload, telemetry_healthy, _telemetry_failure = (
                self._stop_telemetry(ledger, telemetry_session, telemetry_started)
            )
            return self._sealed_capture_failure_outcome(
                artifact_directory,
                protected_artifact_id=new_opaque_id("artifact"),
                native_outcome="unknown",
                reason_code="UNKNOWN_TERMINAL_STATE",
                capture_reason_code="STREAM_CAPTURE_FAILURE",
                exit_code=process.returncode,
                timed_out=False,
                telemetry_healthy=telemetry_healthy,
                missing_fields=(
                    "command/record.json",
                    "command/combined-output.bin",
                    "events/events.jsonl",
                ),
                recoverable_locator=None,
                block_submissions=not termination_certain,
            )
        try:
            spool_fingerprint = _finalize_command_spool(spool_path, spool_descriptor)
        except Exception:
            _discard_command_spool(spool_path, spool_descriptor)
            _telemetry_payload, telemetry_healthy, _telemetry_failure = (
                self._stop_telemetry(ledger, telemetry_session, telemetry_started)
            )
            return self._sealed_capture_failure_outcome(
                artifact_directory,
                protected_artifact_id=new_opaque_id("artifact"),
                native_outcome="unknown",
                reason_code="UNKNOWN_TERMINAL_STATE",
                capture_reason_code="STREAM_CAPTURE_FAILURE",
                exit_code=process.returncode,
                timed_out=timed_out,
                telemetry_healthy=telemetry_healthy,
                missing_fields=("command/combined-output.bin",),
                recoverable_locator=None,
            )
        finished_ns = self._monotonic_ns()
        exit_code = process.returncode
        if timed_out:
            native_outcome = "timed-out"
            reason_code = "EMERGENCY_DEADLINE_EXCEEDED"
        elif exit_code == 0:
            native_outcome = "passed"
            reason_code = "NONE"
        else:
            native_outcome = "failed"
            reason_code = "PROCESS_EXIT_NONZERO"
        ledger.append(
            monotonic_ns=finished_ns,
            wall_time_utc=self._wall_time(),
            event_type="command.finished",
            action="command",
            subject_kind="process",
            subject_id=self.experiment_run_id,
            exit_code=exit_code,
            native_outcome=native_outcome,
            reason_code=reason_code,
        )
        telemetry_payload, telemetry_healthy, telemetry_failure_code = (
            self._stop_telemetry(ledger, telemetry_session, telemetry_started)
        )
        command_record = {
            "record_kind": "aptus-cuda-campaign-command-capture-v1",
            "experiment_run_id": self.experiment_run_id,
            "attempt_slot_id": self.attempt_slot_id,
            "exact_argv": list(exact_argv),
            "working_directory": str(cwd),
            "started_monotonic_ns": started_ns,
            "finished_monotonic_ns": finished_ns,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "native_outcome": native_outcome,
            "reason_code": reason_code,
        }
        ledger_payload = self._finish_ledger(
            ledger,
            native_outcome=native_outcome,
            reason_code=reason_code,
            exit_code=exit_code,
        )
        payloads = [
            _Payload(
                canonical_json_bytes(command_record),
                None,
                "command/record.json",
                "command-record",
                "application/json",
                "entry_command-record",
            ),
            _Payload(
                None,
                spool_path,
                "command/combined-output.bin",
                "command-output",
                "application/octet-stream",
                "entry_command-output",
                spool_descriptor,
                spool_fingerprint,
            ),
            _Payload(
                ledger_payload,
                None,
                "events/events.jsonl",
                "event-ledger",
                "application/x-ndjson",
                "entry_event-ledger",
            ),
        ]
        required: dict[str, str | Sequence[str]] = {
            "command-record": "entry_command-record",
            "command-output": "entry_command-output",
            "event-ledger": "entry_event-ledger",
        }
        if telemetry_payload is not None:
            payloads.append(
                _Payload(
                    telemetry_payload,
                    None,
                    "telemetry/samples.jsonl",
                    "telemetry",
                    "application/x-ndjson",
                    "entry_telemetry",
                )
            )
            required["telemetry"] = "entry_telemetry"
        payloads.extend(
            _Payload(
                None,
                artifact.source,
                artifact.relative_path,
                artifact.role,
                artifact.media_type,
            )
            for artifact in selected_artifacts
        )
        try:
            return self._seal_payloads(
                artifact_directory,
                payloads=payloads,
                required_role_bindings=required,
                native_outcome=native_outcome,
                reason_code=reason_code,
                exit_code=exit_code,
                timed_out=timed_out,
                telemetry_healthy=telemetry_healthy,
                evidence_status=(
                    "capture-invalid"
                    if telemetry_healthy is False
                    else "protocol-valid"
                ),
                capture_reason_code=(
                    telemetry_failure_code if telemetry_healthy is False else "NONE"
                ),
            )
        finally:
            _discard_command_spool(spool_path, spool_descriptor)

    def _append_action_started(
        self, ledger: EventLedgerWriter, spec: ManagedActionSpec
    ) -> None:
        started_ns = self._monotonic_ns()
        ledger.append(
            monotonic_ns=started_ns,
            wall_time_utc=self._wall_time(),
            event_type="command.started",
            phase=spec.label,
            action=spec.action,
            subject_kind="managed-action",
            subject_id=spec.label,
        )

    def _append_action_finished(
        self,
        ledger: EventLedgerWriter,
        result: _ManagedActionResult,
    ) -> None:
        ledger.append(
            monotonic_ns=max(self._monotonic_ns(), ledger.records[-1]["monotonic_ns"]),
            wall_time_utc=self._wall_time(),
            event_type="command.finished",
            phase=result.spec.label,
            action=result.spec.action,
            subject_kind="managed-action",
            subject_id=result.spec.label,
            exit_code=result.exit_code,
            native_outcome=result.native_outcome,
            reason_code=result.reason_code,
        )

    def _runtime_reader(
        self,
        spec: ManagedActionSpec,
        job_id: str,
        record: Mapping[str, Any],
    ) -> RuntimeBoundaryJournalReader | None:
        if spec.action not in {"pilot", "train"}:
            return None
        capture_enabled = record.get("campaign_event_capture") is True
        if not capture_enabled:
            if self.qualification_context is not None:
                raise RuntimeBoundaryError(
                    "qualifying runtime boundary capture was not enabled"
                )
            return None
        if record.get("campaign_experiment_run_id") != self.experiment_run_id:
            raise RuntimeBoundaryError("runtime boundary run identity is misbound")
        path_value = record.get("campaign_event_sink")
        identity = record.get("campaign_event_sink_identity")
        if not isinstance(path_value, str) or not isinstance(identity, str):
            raise RuntimeBoundaryError("runtime boundary sink binding is incomplete")
        return RuntimeBoundaryJournalReader(
            Path(path_value),
            expected_file_identity=identity,
            experiment_run_id=self.experiment_run_id,
            job_id=job_id,
            action=spec.action,
        )

    @staticmethod
    def _drain_runtime_boundaries(
        ledger: EventLedgerWriter,
        reader: RuntimeBoundaryJournalReader | None,
    ) -> tuple[RuntimeBoundary, ...]:
        if reader is None:
            return ()
        observed = reader.drain()
        for boundary in observed:
            ledger.append(**boundary.ledger_fields())
        return observed

    def _append_job_state_observed(
        self,
        ledger: EventLedgerWriter,
        spec: ManagedActionSpec,
        job_id: str,
        record: Mapping[str, Any],
        observed_ns: int,
        *,
        terminal_reason_override: str | None = None,
    ) -> None:
        state = record.get("state")
        terminal_reason = (
            terminal_reason_override
            if state == "failed" and terminal_reason_override is not None
            else "PROCESS_EXIT_NONZERO"
            if state == "failed"
            else "NONE"
        )
        if state == "cancelled":
            observed_reason = record.get("cancel_reason_code")
            terminal_reason = (
                observed_reason
                if observed_reason in REASON_CODES and observed_reason != "NONE"
                else "UNKNOWN_TERMINAL_STATE"
            )
        ledger.append(
            monotonic_ns=observed_ns,
            wall_time_utc=self._wall_time(),
            event_type="job.state-observed",
            phase=spec.label,
            action=spec.action,
            subject_kind="aptus-job",
            subject_id=job_id,
            native_outcome=(
                "passed"
                if state == "completed"
                else "failed"
                if state == "failed"
                else "cancelled"
                if state == "cancelled"
                else None
            ),
            reason_code=terminal_reason,
        )

    @staticmethod
    def _trusted_runtime_failure_reason(
        spec: ManagedActionSpec,
        boundaries: Sequence[RuntimeBoundary],
    ) -> str | None:
        """Return the exact emitted failure closure, from inner to outer work."""

        allowed_finished = (
            {"pilot.phase-finished"}
            if spec.action == "pilot"
            else {"training.finished", "verification.finished"}
            if spec.action == "train"
            else set()
        )
        candidates = [
            boundary.reason_code
            for boundary in boundaries
            if boundary.action == spec.action
            and boundary.event_type in allowed_finished
            and boundary.native_outcome == "failed"
            and boundary.reason_code in REASON_CODES
            and boundary.reason_code != "NONE"
        ]
        return candidates[-1] if candidates else None

    def _pre_submit_signal(
        self,
        spec: ManagedActionSpec,
        *,
        telemetry_session: TelemetrySession | None,
        pre_action_check: Callable[[ManagedActionSpec], SafetySignal | None] | None,
    ) -> SafetySignal | None:
        detected_ns = self._monotonic_ns()
        if telemetry_session is not None:
            try:
                signal = telemetry_session.safety_signal()
            except Exception:
                return SafetySignal("TELEMETRY_COLLECTOR_FAILURE", detected_ns)
            if signal is not None:
                if type(signal) is not SafetySignal:
                    return SafetySignal("TELEMETRY_COLLECTOR_FAILURE", detected_ns)
                return signal
        if pre_action_check is None:
            return None
        try:
            signal = pre_action_check(spec)
        except Exception:
            return SafetySignal("OWNERSHIP_UNCERTAIN", detected_ns)
        if signal is not None and type(signal) is not SafetySignal:
            return SafetySignal("OWNERSHIP_UNCERTAIN", detected_ns)
        return signal

    def _guard_blocked_action(
        self,
        spec: ManagedActionSpec,
        *,
        ledger: EventLedgerWriter,
        signal: SafetySignal,
    ) -> _ManagedActionResult:
        event_ns = max(signal.detected_monotonic_ns, ledger.records[-1]["monotonic_ns"])
        if signal.reason_code == "OWNERSHIP_UNCERTAIN":
            self._block_submissions(
                reason_code=signal.reason_code,
                job_id=None,
                detected_monotonic_ns=signal.detected_monotonic_ns,
            )
        ledger.append(
            monotonic_ns=event_ns,
            wall_time_utc=self._wall_time(),
            event_type="safety.triggered",
            phase=spec.label,
            action=spec.action,
            subject_kind="managed-action",
            subject_id=spec.label,
            reason_code=signal.reason_code,
        )
        result = _ManagedActionResult(
            spec=spec,
            job_id=None,
            record={
                "record_kind": "aptus-cuda-campaign-pre-submit-guard-v1",
                "action_label": spec.label,
                "action": spec.action,
                "native_outcome": "guard-blocked",
                "reason_code": signal.reason_code,
            },
            native_outcome="guard-blocked",
            reason_code=signal.reason_code,
            exit_code=None,
            timed_out=False,
            terminal=True,
            capture_reason_code=(
                "MISSING_REQUIRED_EVIDENCE"
                if signal.detected_monotonic_ns < event_ns
                else "NONE"
            ),
        )
        self._append_action_finished(ledger, result)
        return result

    def _unknown_submission_result(
        self,
        spec: ManagedActionSpec,
        *,
        ledger: EventLedgerWriter,
        record_kind: str,
    ) -> _ManagedActionResult:
        detected_ns = self._monotonic_ns()
        self._block_submissions(
            reason_code="OWNERSHIP_UNCERTAIN",
            job_id=None,
            detected_monotonic_ns=detected_ns,
        )
        result = _ManagedActionResult(
            spec=spec,
            job_id=None,
            record={
                "record_kind": record_kind,
                "action_label": spec.label,
                "action": spec.action,
                "native_outcome": "unknown",
                "reason_code": "OWNERSHIP_UNCERTAIN",
            },
            native_outcome="unknown",
            reason_code="OWNERSHIP_UNCERTAIN",
            exit_code=None,
            timed_out=False,
            terminal=False,
            capture_reason_code="NONE",
        )
        self._append_action_finished(ledger, result)
        return result

    def _supervise_managed_action(
        self,
        service: ManagedJobService | JobService,
        bundle_dir: Path,
        spec: ManagedActionSpec,
        *,
        ledger: EventLedgerWriter,
        poll_interval_seconds: float,
        safety_check: (
            Callable[[ManagedActionSpec, Mapping[str, Any]], SafetySignal | None] | None
        ),
        telemetry_session: TelemetrySession | None,
        pre_action_check: Callable[[ManagedActionSpec], SafetySignal | None] | None,
    ) -> _ManagedActionResult:
        self._append_action_started(ledger, spec)
        pre_submit_signal = self._pre_submit_signal(
            spec,
            telemetry_session=telemetry_session,
            pre_action_check=pre_action_check,
        )
        if pre_submit_signal is not None:
            return self._guard_blocked_action(
                spec, ledger=ledger, signal=pre_submit_signal
            )
        action_started_ns = self._monotonic_ns()
        try:
            submitted = service.submit(
                bundle_dir, action=spec.action, **dict(spec.submit_kwargs)
            )
        except JobSubmissionFailure as error:
            terminal_record = error.terminal_record
            job_id = error.job_id
            if (
                not isinstance(terminal_record, Mapping)
                or _JOB_ID.fullmatch(job_id) is None
                or terminal_record.get("id") != job_id
                or terminal_record.get("job_id") != job_id
                or terminal_record.get("state") not in _TERMINAL_STATES
            ):
                return self._unknown_submission_result(
                    spec,
                    ledger=ledger,
                    record_kind="aptus-cuda-campaign-invalid-post-persist-failure-v1",
                )
            record = dict(terminal_record)
            result = _ManagedActionResult(
                spec=spec,
                job_id=job_id,
                record=record,
                native_outcome=(
                    "cancelled" if record["state"] == "cancelled" else "failed"
                ),
                reason_code=(
                    record.get("cancel_reason_code", "UNKNOWN_TERMINAL_STATE")
                    if record["state"] == "cancelled"
                    else "UNKNOWN_TERMINAL_STATE"
                ),
                exit_code=None,
                timed_out=False,
                terminal=True,
                capture_reason_code="MISSING_REQUIRED_EVIDENCE",
            )
            self._append_action_finished(ledger, result)
            return result
        except (ActiveJobError, JobPrerequisiteError, ValueError) as error:
            result = _ManagedActionResult(
                spec=spec,
                job_id=None,
                record={
                    "record_kind": "aptus-cuda-campaign-submission-refusal-v1",
                    "action_label": spec.label,
                    "action": spec.action,
                    "exception_type": type(error).__name__,
                    "native_outcome": "refused",
                    "reason_code": "APTUS_ADMISSION_REFUSAL",
                },
                native_outcome="refused",
                reason_code="APTUS_ADMISSION_REFUSAL",
                exit_code=None,
                timed_out=False,
                terminal=True,
            )
            self._append_action_finished(ledger, result)
            return result
        except Exception:
            return self._unknown_submission_result(
                spec,
                ledger=ledger,
                record_kind="aptus-cuda-campaign-ambiguous-submission-failure-v1",
            )

        if not isinstance(submitted, Mapping):
            return self._unknown_submission_result(
                spec,
                ledger=ledger,
                record_kind="aptus-cuda-campaign-malformed-submission-v1",
            )
        submitted_record = dict(submitted)
        submitted_id = submitted_record.get("id")
        submitted_job_id = submitted_record.get("job_id")
        if (
            not isinstance(submitted_id, str)
            or not isinstance(submitted_job_id, str)
            or _JOB_ID.fullmatch(submitted_id) is None
            or submitted_job_id != submitted_id
        ):
            return self._unknown_submission_result(
                spec,
                ledger=ledger,
                record_kind="aptus-cuda-campaign-ambiguous-submission-identity-v1",
            )
        job_id = submitted_id
        self._record_known_active_job(job_id)

        action_capture_reason = "NONE"
        runtime_reader: RuntimeBoundaryJournalReader | None = None
        try:
            runtime_reader = self._runtime_reader(spec, job_id, submitted_record)
            self._drain_runtime_boundaries(ledger, runtime_reader)
        except (RuntimeBoundaryError, TypeError, ValueError):
            runtime_reader = None
            action_capture_reason = "MISSING_REQUIRED_EVIDENCE"

        last_record: dict[str, Any] = submitted_record
        last_state: str | None = None
        native_outcome = "unknown"
        reason_code = "UNKNOWN_TERMINAL_STATE"
        timed_out = False
        terminal = False
        deadline_ns = action_started_ns + int(
            spec.supervision_timeout_seconds * 1_000_000_000
        )
        while True:
            polling_uncertain = False
            try:
                current = service.get(job_id, include_validation_report=False)
            except Exception:
                current = last_record
                polling_uncertain = True
            if not isinstance(current, Mapping):
                current = last_record
                polling_uncertain = True
            elif current.get("job_id") != job_id or current.get("id") != job_id:
                current = last_record
                polling_uncertain = True
            if polling_uncertain:
                self._update_managed_pid(job_id, {})
                self._block_submissions(
                    reason_code="OWNERSHIP_UNCERTAIN",
                    job_id=job_id,
                    detected_monotonic_ns=self._monotonic_ns(),
                )
            else:
                last_record = dict(current)
                self._update_managed_pid(job_id, current)
            observed_ns: int
            if runtime_reader is not None:
                try:
                    observed_boundaries, observed_ns = (
                        runtime_reader.drain_and_sample_monotonic(self._monotonic_ns)
                    )
                    for boundary in observed_boundaries:
                        ledger.append(**boundary.ledger_fields())
                except (RuntimeBoundaryError, TypeError, ValueError):
                    runtime_reader = None
                    action_capture_reason = "MISSING_REQUIRED_EVIDENCE"
                    observed_ns = self._monotonic_ns()
            else:
                observed_ns = self._monotonic_ns()
            state = current.get("state")
            state_changed = state != last_state
            if not polling_uncertain and state in _TERMINAL_STATES:
                if state == "completed":
                    native_outcome, reason_code = "passed", "NONE"
                elif state == "failed":
                    native_outcome = "failed"
                    reason_code = (
                        self._trusted_runtime_failure_reason(
                            spec,
                            runtime_reader.records
                            if runtime_reader is not None
                            else (),
                        )
                        or "PROCESS_EXIT_NONZERO"
                    )
                else:
                    cancel_reason = current.get("cancel_reason_code")
                    if cancel_reason not in REASON_CODES or cancel_reason == "NONE":
                        cancel_reason = "UNKNOWN_TERMINAL_STATE"
                    native_outcome, reason_code = "cancelled", cancel_reason
                if state_changed:
                    self._append_job_state_observed(
                        ledger,
                        spec,
                        job_id,
                        current,
                        observed_ns,
                        terminal_reason_override=(
                            reason_code if state == "failed" else None
                        ),
                    )
                    last_state = state if isinstance(state, str) else None
                terminal = True
                break
            ownership_uncertain = polling_uncertain or (
                state not in _ACTIVE_STATES
                or current.get("owner_status") != "owning-service"
            )
            stop_signal: SafetySignal | None = None
            if ownership_uncertain:
                self._block_submissions(
                    reason_code="OWNERSHIP_UNCERTAIN",
                    job_id=job_id,
                    detected_monotonic_ns=observed_ns,
                )
                # Ask the exact service used for submission to cancel the exact
                # job ID.  JobService performs the authoritative ownership and
                # process-group checks; a refusal remains unknown and blocked.
                stop_signal = SafetySignal("OWNERSHIP_UNCERTAIN", observed_ns)
            elif telemetry_session is not None:
                try:
                    stop_signal = telemetry_session.safety_signal()
                except Exception:
                    stop_signal = SafetySignal(
                        "TELEMETRY_COLLECTOR_FAILURE", observed_ns
                    )
            if stop_signal is None and safety_check is not None:
                try:
                    stop_signal = safety_check(spec, dict(current))
                except Exception:
                    stop_signal = SafetySignal("OWNERSHIP_UNCERTAIN", observed_ns)
            if stop_signal is None and observed_ns >= deadline_ns:
                stop_signal = SafetySignal("EMERGENCY_DEADLINE_EXCEEDED", observed_ns)
                timed_out = True
            if stop_signal is None:
                if state_changed:
                    self._append_job_state_observed(
                        ledger, spec, job_id, current, observed_ns
                    )
                    last_state = state if isinstance(state, str) else None
                self._sleep(poll_interval_seconds)
                continue
            ledger_floor = ledger.records[-1]["monotonic_ns"]
            signal_can_be_ledgered = stop_signal.detected_monotonic_ns >= ledger_floor
            if not signal_can_be_ledgered:
                action_capture_reason = "MISSING_REQUIRED_EVIDENCE"
                self._block_submissions(
                    reason_code="MISSING_REQUIRED_EVIDENCE",
                    job_id=job_id,
                    detected_monotonic_ns=stop_signal.detected_monotonic_ns,
                )
            elif stop_signal.detected_monotonic_ns <= observed_ns:
                ledger.append(
                    monotonic_ns=stop_signal.detected_monotonic_ns,
                    wall_time_utc=self._wall_time(),
                    event_type="safety.triggered",
                    phase=spec.label,
                    action=spec.action,
                    subject_kind="aptus-job",
                    subject_id=job_id,
                    reason_code=stop_signal.reason_code,
                )
            if state_changed:
                self._append_job_state_observed(
                    ledger, spec, job_id, current, observed_ns
                )
                last_state = state if isinstance(state, str) else None
            if (
                signal_can_be_ledgered
                and stop_signal.detected_monotonic_ns > observed_ns
            ):
                ledger.append(
                    monotonic_ns=stop_signal.detected_monotonic_ns,
                    wall_time_utc=self._wall_time(),
                    event_type="safety.triggered",
                    phase=spec.label,
                    action=spec.action,
                    subject_kind="aptus-job",
                    subject_id=job_id,
                    reason_code=stop_signal.reason_code,
                )
            try:
                cancelled = service.cancel(
                    job_id,
                    reason_code=stop_signal.reason_code,
                    trigger_detected_monotonic_ns=(stop_signal.detected_monotonic_ns),
                )
                if (
                    cancelled.get("job_id", cancelled.get("id")) != job_id
                    or cancelled.get("id", job_id) != job_id
                    or cancelled.get("state") not in _TERMINAL_STATES
                ):
                    raise CancellationSLAError(
                        "Cancellation did not return the exact terminal Aptus job."
                    )
                verify_cancellation_milestones(
                    cancelled,
                    reason_code=stop_signal.reason_code,
                    trigger_detected_monotonic_ns=(stop_signal.detected_monotonic_ns),
                )
            except CancellationSLAError as error:
                self._block_submissions(
                    reason_code=error.reason_code,
                    job_id=job_id,
                    detected_monotonic_ns=stop_signal.detected_monotonic_ns,
                )
                native_outcome = "unknown"
                reason_code = error.reason_code
                break
            except Exception:
                cancellation_failure_reason = (
                    "OWNERSHIP_UNCERTAIN"
                    if stop_signal.reason_code == "OWNERSHIP_UNCERTAIN"
                    else "CANCELLATION_DEADLINE_EXCEEDED"
                )
                self._block_submissions(
                    reason_code=cancellation_failure_reason,
                    job_id=job_id,
                    detected_monotonic_ns=stop_signal.detected_monotonic_ns,
                )
                native_outcome = "unknown"
                reason_code = cancellation_failure_reason
                break
            last_record = dict(cancelled)
            self._update_managed_pid(job_id, cancelled)
            terminal = True
            if signal_can_be_ledgered:
                for event_type, timestamp_name, wall_name in (
                    (
                        "cancellation.requested",
                        "cancel_requested_monotonic_ns",
                        "cancel_requested_at",
                    ),
                    (
                        "process-group.terminated",
                        "process_group_terminated_monotonic_ns",
                        "process_group_terminated_at",
                    ),
                    (
                        "lease.reconciled",
                        "lease_reconciled_monotonic_ns",
                        "lease_reconciled_at",
                    ),
                ):
                    ledger.append(
                        monotonic_ns=int(cancelled[timestamp_name]),
                        wall_time_utc=self._wall_time(),
                        event_type=event_type,
                        phase=spec.label,
                        action=spec.action,
                        subject_kind="aptus-job",
                        subject_id=job_id,
                        source_reported_at_utc=(
                            cancelled.get(wall_name)
                            if isinstance(cancelled.get(wall_name), str)
                            else None
                        ),
                        native_outcome="cancelled",
                        reason_code=stop_signal.reason_code,
                    )
            native_outcome = (
                "timed-out"
                if stop_signal.reason_code == "EMERGENCY_DEADLINE_EXCEEDED"
                else "cancelled"
            )
            reason_code = stop_signal.reason_code
            break

        exit_code = last_record.get("return_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, (int, type(None))):
            exit_code = None
        runtime_journal_bytes: bytes | None = None
        if runtime_reader is not None:
            try:
                self._drain_runtime_boundaries(ledger, runtime_reader)
                runtime_journal_bytes = runtime_reader.verified_bytes
            except (RuntimeBoundaryError, TypeError, ValueError):
                runtime_reader = None
                action_capture_reason = "MISSING_REQUIRED_EVIDENCE"
        result = _ManagedActionResult(
            spec=spec,
            job_id=job_id,
            record=last_record,
            native_outcome=native_outcome,
            reason_code=reason_code,
            exit_code=exit_code,
            timed_out=timed_out,
            terminal=terminal,
            capture_reason_code=action_capture_reason,
            runtime_boundaries=(
                runtime_reader.records if runtime_reader is not None else ()
            ),
            runtime_journal_bytes=runtime_journal_bytes,
        )
        self._update_managed_pid(job_id, last_record if terminal else {})
        if terminal:
            self._clear_known_active_job(job_id)
        self._append_action_finished(ledger, result)
        return result

    @staticmethod
    def _validate_action_sequence(
        actions: Sequence[ManagedActionSpec],
        selected_artifacts: Sequence[SelectedArtifact],
        *,
        legacy_single_action_paths: bool,
    ) -> tuple[ManagedActionSpec, ...]:
        if isinstance(actions, (str, bytes)) or not actions:
            raise ValueError("Managed action sequence must not be empty.")
        specs = tuple(actions)
        if any(type(spec) is not ManagedActionSpec for spec in specs):
            raise ValueError("Every managed action must be a ManagedActionSpec.")
        labels = [spec.label for spec in specs]
        if len(labels) != len(set(labels)):
            raise ValueError("Managed action labels and paths must be unique.")
        reserved = {
            "events/events.jsonl",
            "sequence/summary.json",
            "telemetry/samples.jsonl",
            "raw-manifest.json",
            "SEALED.json",
        }
        for spec in specs:
            prefix = "job" if legacy_single_action_paths else f"actions/{spec.label}"
            for name in ("terminal.json", "full.log", "submission.json"):
                path = f"{prefix}/{name}"
                if path in reserved:
                    raise ValueError("Managed action paths collide.")
                reserved.add(path)
        selected_paths: set[str] = set()
        core_roles = {
            "event-ledger",
            "sequence-summary",
            "telemetry",
            "job-log",
            "terminal-job-record",
            "last-observed-job-record",
            "action-submission-record",
            "capture-failure",
            "command-record",
            "command-output",
            "attempt-slot-record",
            "execution-configuration-record",
            "experiment-run-record",
            "idle-baseline-binding",
            "telemetry-configuration",
            "telemetry-summary",
            "cooldown-summary",
            "runtime-boundary-journal",
            *REQUIRED_QUALIFYING_AUTHORITY_ROLES,
        }
        for artifact in selected_artifacts:
            if type(artifact) is not SelectedArtifact:
                raise ValueError("Selected artifacts must use SelectedArtifact.")
            path = validate_safe_relative_path(artifact.relative_path)
            if path in reserved or path in selected_paths:
                raise ValueError(
                    "Selected artifact paths must be unique and unreserved."
                )
            if (
                not isinstance(artifact.role, str)
                or not artifact.role
                or not isinstance(artifact.media_type, str)
                or not artifact.media_type
                or artifact.role in core_roles
            ):
                raise ValueError("Selected artifact role or media type is reserved.")
            selected_paths.add(path)
        return specs

    def _validate_qualifying_sequence(
        self,
        bundle_dir: Path,
        artifact_directory: Path,
        specs: tuple[ManagedActionSpec, ...],
        selected_artifacts: Sequence[SelectedArtifact],
        telemetry_session: TelemetrySession | None,
        *,
        legacy_single_action_paths: bool,
        allow_nonqualifying_without_telemetry_for_test: bool,
    ) -> None:
        context = self.qualification_context
        if context is None:
            return
        phase4 = self._phase4_verification
        repository = self._phase4_repository_root
        planned = self._planned_slot_context
        activation_authority = self._activation_authority
        activation_directory = self._activation_directory
        if (
            phase4 is None
            or repository is None
            or type(planned) is not PlannedSlotContext
            or type(activation_authority) is not Phase4CurrentAuthority
            or activation_directory is None
        ):
            raise ValueError("Qualifying Phase-4 authority is unavailable.")
        repeated_activation = verify_activated_slot(
            activation_directory,
            expected_context=planned,
            authority=activation_authority,
        )
        refreshed_context = QualifyingRunContext(planned, repeated_activation)
        if (
            refreshed_context.source_bindings() != context.source_bindings()
            or dict(refreshed_context.execution_configuration)
            != dict(context.execution_configuration)
            or dict(refreshed_context.experiment_run_template)
            != dict(context.experiment_run_template)
        ):
            raise ValueError("Qualifying activation changed before execution.")
        repeated_phase4 = verify_phase4_source_freeze_artifact(
            phase4.directory,
            repository_root=repository,
            campaign=context.campaign,
            comparison_cohort=context.comparison_cohort,
            comparison_cell=context.comparison_cell,
            nvidia_smi_path=self._phase4_nvidia_smi_path,
            gpu_index=self._phase4_gpu_index,
        )
        if (
            repeated_phase4.source_freeze_sha256 != phase4.source_freeze_sha256
            or repeated_phase4.seal_sha256 != phase4.seal_sha256
            or repeated_phase4.samples_sha256 != phase4.samples_sha256
            or dict(repeated_phase4.baseline_binding)
            != dict(context.idle_baseline_binding)
        ):
            raise ValueError("Qualifying Phase-4 authority changed before execution.")
        if legacy_single_action_paths:
            raise ValueError(
                "A qualifying campaign run requires the full managed sequence."
            )
        try:
            bundle = bundle_dir.resolve(strict=True)
            artifact_parent = artifact_directory.parent.resolve(strict=True)
        except OSError:
            raise ValueError(
                "The qualifying bundle or artifact parent is unavailable."
            ) from None
        declared_bundle = Path(context.experiment_run_template["bundle_path"]).resolve(
            strict=True
        )
        declared_state = Path(
            context.experiment_run_template["fresh_state_root"]
        ).resolve(strict=True)
        service = self.job_service
        if (
            bundle != declared_bundle
            or self.state_root.resolve(strict=True) != declared_state
            or type(service) is not JobService
            or service.root.resolve(strict=True) != declared_state
            or Path(context.experiment_run_template["working_directory"]).resolve()
            != bundle
            or Path(context.experiment_run_template["output_path"]).resolve()
            != bundle / "runs"
            or self._admission_filesystem_device is None
            or bundle.stat().st_dev != self._admission_filesystem_device
            or artifact_parent.stat().st_dev != self._admission_filesystem_device
        ):
            raise ValueError("Qualifying runtime paths differ from the frozen run.")
        if tuple(spec.action for spec in specs) != QUALIFYING_ACTION_ORDER:
            raise ValueError(
                "A qualifying campaign run requires the frozen five-action order."
            )
        for spec in specs:
            kwargs = dict(spec.submit_kwargs)
            if spec.action in {"pilot", "train"}:
                if (
                    kwargs.get("campaign_event_capture") is not True
                    or kwargs.get("campaign_experiment_run_id")
                    != self.experiment_run_id
                ):
                    raise ValueError(
                        "Pilot and train require the exact qualifying runtime sink."
                    )
            elif {
                "campaign_event_capture",
                "campaign_experiment_run_id",
            } & set(kwargs):
                raise ValueError(
                    "Only pilot and train may enable runtime boundary capture."
                )
        if telemetry_session is None or allow_nonqualifying_without_telemetry_for_test:
            raise ValueError("A qualifying campaign run requires its frozen sidecar.")
        self._require_qualifying_runtime_authority(telemetry_session)
        role_counts: dict[str, int] = {}
        selected_by_role: dict[str, SelectedArtifact] = {}
        for artifact in selected_artifacts:
            role_counts[artifact.role] = role_counts.get(artifact.role, 0) + 1
            selected_by_role[artifact.role] = artifact
        dynamic_roles = {"training-metrics", "final-export-manifest"}
        caller_roles = REQUIRED_QUALIFYING_ARTIFACT_ROLES - dynamic_roles
        if any(role_counts.get(role) != 1 for role in caller_roles) or any(
            role_counts.get(role, 0) for role in dynamic_roles
        ):
            raise ValueError(
                "Qualifying selected artifacts must contain the five static roles only."
            )
        output_root = Path(context.experiment_run_template["output_path"])
        if output_root.exists():
            if (
                output_root.is_symlink()
                or not output_root.is_dir()
                or any(output_root.iterdir())
                or (os.name == "posix" and output_root.stat().st_mode & 0o077)
            ):
                raise ValueError("The qualifying output root is not fresh and private.")
        expected_sources = {
            "plan": bundle / "plan.json",
            "bundle-manifest": bundle / "bundle-manifest.json",
            "validation-report": bundle / "validation-report.json",
            "pilot-metrics": bundle / "pilot-output" / "metrics.json",
        }
        for role, expected in expected_sources.items():
            observed = Path(os.path.abspath(selected_by_role[role].source))
            if observed != Path(os.path.abspath(expected)):
                raise ValueError(
                    f"Qualifying artifact role {role} has the wrong source path."
                )
            if selected_by_role[role].media_type != "application/json":
                raise ValueError(
                    f"Qualifying artifact role {role} has the wrong media type."
                )
        archive = selected_by_role["bundle-archive"].source
        if selected_by_role["bundle-archive"].media_type != "application/zip" or (
            _hash_stable_regular_source(archive)
            != context.experiment_run_template["archive_sha256"]
        ):
            raise ValueError("The qualifying bundle archive digest is misbound.")
        try:
            archive_matches = verify_bundle_archive(bundle, archive)
        except (OSError, ValueError):
            raise ValueError(
                "The qualifying bundle archive could not be recomputed."
            ) from None
        if not archive_matches:
            raise ValueError(
                "The qualifying bundle archive is not the exact Aptus archive."
            )
        manifest = expected_sources["bundle-manifest"]
        try:
            plan_bytes = _read_stable_regular_source(expected_sources["plan"])
            manifest_bytes = _read_stable_regular_source(manifest)
            plan = json.loads(plan_bytes)
            manifest_record = json.loads(manifest_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("The qualifying bundle identity is unreadable.") from None
        execution = context.execution_configuration
        if (
            not isinstance(plan, dict)
            or not isinstance(manifest_record, dict)
            or canonical_json_bytes(plan) != plan_bytes
            or canonical_json_bytes(manifest_record) != manifest_bytes
            or plan.get("plan_id") != execution["plan_id"]
            or not isinstance(plan.get("recommended"), dict)
            or plan["recommended"].get("candidate_id") != execution["candidate_id"]
            or manifest_record.get("plan_id") != execution["plan_id"]
            or manifest_record.get("candidate_id") != execution["candidate_id"]
            or sha256_bytes(manifest_bytes) != execution["bundle_fingerprint"]
        ):
            raise ValueError("The qualifying bundle identity differs from its context.")
        if (
            sha256_bytes(manifest_bytes)
            != context.experiment_run_template["bundle_manifest_sha256"]
        ):
            raise ValueError(
                "The qualifying bundle manifest differs from its frozen run record."
            )
        if (
            _read_stable_regular_source(expected_sources["plan"]) != plan_bytes
            or _read_stable_regular_source(manifest) != manifest_bytes
        ):
            raise ValueError(
                "The qualifying bundle identity changed during archive verification."
            )
        configuration_provider = getattr(
            telemetry_session, "configuration_record", None
        )
        snapshot_provider = getattr(telemetry_session, "snapshot", None)
        if not callable(configuration_provider) or not callable(snapshot_provider):
            raise ValueError(
                "Qualifying telemetry evidence interfaces are unavailable."
            )
        configuration = configuration_provider()
        validate_qualifying_telemetry_configuration(configuration, context=context)
        if configuration != phase4.source_freeze["telemetry_configuration"]:
            raise ValueError("Qualifying telemetry configuration differs from Phase 4.")

    def _require_qualifying_runtime_authority(
        self, telemetry_session: TelemetrySession
    ) -> None:
        """Require the factory-owned service, clocks, and exact sidecar identity."""

        if self.qualification_context is None:
            return
        authority = self._qualifying_authority
        if (
            authority is None
            or not _qualifying_harness_is_registered(self, authority)
            or type(self.job_service) is not JobService
            or self.job_service is not self._qualifying_job_service
            or self._monotonic_ns is not time.monotonic_ns
            or self._wall_time is not utc_now
            or self._sleep is not time.sleep
        ):
            raise ValueError("Qualifying harness authority is unavailable or changed.")
        from .sidecar import BackgroundTelemetrySession

        if type(telemetry_session) is not BackgroundTelemetrySession or not (
            telemetry_session._authorized_for_harness(authority)
        ):
            raise ValueError(
                "Telemetry sidecar lacks this harness's production authority."
            )

    def _capture_qualifying_cooldown(
        self,
        ledger: EventLedgerWriter,
        session: TelemetrySession,
        context: QualifyingRunContext,
    ) -> WindowValidation:
        """Capture the exact post-run 120-sample window, bounded at 30 minutes."""

        started_ns = max(self._monotonic_ns(), ledger.records[-1]["monotonic_ns"])
        ledger.append(
            monotonic_ns=started_ns,
            wall_time_utc=self._wall_time(),
            event_type="cooldown.started",
            phase="cooldown",
            subject_kind="experiment-run",
            subject_id=self.experiment_run_id,
        )
        deadline_ns = started_ns + (
            QUALIFYING_COOLDOWN_MAXIMUM_WAIT_SECONDS * 1_000_000_000
        )
        result = WindowValidation(False, ("MISSING_REQUIRED_EVIDENCE",))
        while True:
            now_ns = self._monotonic_ns()
            try:
                signal = session.safety_signal()
                snapshot = session.snapshot()  # type: ignore[attr-defined]
                raw_samples = getattr(snapshot, "samples")
                samples = [validate_telemetry_sample(item) for item in raw_samples]
                snapshot_failure = getattr(snapshot, "failure_code", None)
                raw_events = getattr(snapshot, "safety_events", ())
                safety_events = (
                    tuple(raw_events)
                    if isinstance(raw_events, (tuple, list))
                    else (raw_events,)
                )
            except Exception:
                signal = SafetySignal("TELEMETRY_COLLECTOR_FAILURE", now_ns)
                samples = []
                snapshot_failure = "TELEMETRY_COLLECTOR_FAILURE"
                safety_events = ()
            if signal is not None:
                code = (
                    signal.reason_code
                    if type(signal) is SafetySignal
                    else "TELEMETRY_COLLECTOR_FAILURE"
                )
                result = WindowValidation(False, (code,))
                break
            if snapshot_failure is not None:
                code = (
                    snapshot_failure
                    if snapshot_failure in REASON_CODES and snapshot_failure != "NONE"
                    else "TELEMETRY_COLLECTOR_FAILURE"
                )
                result = WindowValidation(False, (code,))
                break
            if safety_events:
                first = safety_events[0]
                observed = (
                    first.get("reason_code") if isinstance(first, Mapping) else None
                )
                code = (
                    observed
                    if observed in REASON_CODES
                    else "MISSING_REQUIRED_EVIDENCE"
                )
                result = WindowValidation(False, (code,))
                break
            post_run_samples = [
                item for item in samples if item["observed_monotonic_ns"] >= started_ns
            ]
            if len(post_run_samples) >= QUALIFYING_COOLDOWN_SAMPLES:
                cooldown_samples = post_run_samples[-QUALIFYING_COOLDOWN_SAMPLES:]
                result = validate_cooldown(
                    cooldown_samples,
                    context.idle_baseline_summary,
                    required_samples=QUALIFYING_COOLDOWN_SAMPLES,
                )
                if result.valid:
                    now_ns = cooldown_samples[-1]["observed_monotonic_ns"]
                    break
            if now_ns >= deadline_ns:
                break
            self._sleep(0.25)
        reasons = result.reason_codes or ("MISSING_REQUIRED_EVIDENCE",)
        reason_code = "NONE" if result.valid else reasons[0]
        if reason_code not in REASON_CODES:
            reason_code = "MISSING_REQUIRED_EVIDENCE"
        ledger.append(
            monotonic_ns=max(now_ns, ledger.records[-1]["monotonic_ns"]),
            wall_time_utc=self._wall_time(),
            event_type="cooldown.finished",
            phase="cooldown",
            subject_kind="experiment-run",
            subject_id=self.experiment_run_id,
            native_outcome="passed" if result.valid else "failed",
            reason_code=reason_code,
        )
        return result

    def run_managed_job(
        self,
        bundle_dir: Path,
        *,
        artifact_directory: Path,
        action: str = "preflight",
        submit_kwargs: Mapping[str, Any] | None = None,
        supervision_timeout_seconds: float,
        poll_interval_seconds: float = 0.1,
        safety_check: Callable[[Mapping[str, Any]], SafetySignal | None] | None = None,
        pre_action_check: Callable[[ManagedActionSpec], SafetySignal | None]
        | None = None,
        selected_artifacts: Sequence[SelectedArtifact] = (),
        telemetry_session: TelemetrySession | None = None,
        allow_nonqualifying_without_telemetry_for_test: bool = False,
    ) -> CaptureOutcome:
        """Compatibility wrapper for one action in the sequence harness."""

        spec = ManagedActionSpec(
            label=action,
            action=action,
            supervision_timeout_seconds=supervision_timeout_seconds,
            submit_kwargs=dict(submit_kwargs or {}),
        )
        wrapped_safety = (
            None if safety_check is None else lambda _spec, record: safety_check(record)
        )
        return self._run_managed_sequence(
            bundle_dir,
            actions=(spec,),
            artifact_directory=artifact_directory,
            poll_interval_seconds=poll_interval_seconds,
            safety_check=wrapped_safety,
            pre_action_check=pre_action_check,
            selected_artifacts=selected_artifacts,
            telemetry_session=telemetry_session,
            allow_nonqualifying_without_telemetry_for_test=(
                allow_nonqualifying_without_telemetry_for_test
            ),
            legacy_single_action_paths=True,
        )

    def run_managed_sequence(
        self,
        bundle_dir: Path,
        *,
        actions: Sequence[ManagedActionSpec],
        artifact_directory: Path,
        poll_interval_seconds: float = 0.1,
        safety_check: (
            Callable[[ManagedActionSpec, Mapping[str, Any]], SafetySignal | None] | None
        ) = None,
        pre_action_check: Callable[[ManagedActionSpec], SafetySignal | None]
        | None = None,
        selected_artifacts: Sequence[SelectedArtifact] = (),
        telemetry_session: TelemetrySession | None = None,
        allow_nonqualifying_without_telemetry_for_test: bool = False,
    ) -> CaptureOutcome:
        """Capture an ordered action sequence under one telemetry/ledger/seal."""

        return self._run_managed_sequence(
            bundle_dir,
            actions=actions,
            artifact_directory=artifact_directory,
            poll_interval_seconds=poll_interval_seconds,
            safety_check=safety_check,
            pre_action_check=pre_action_check,
            selected_artifacts=selected_artifacts,
            telemetry_session=telemetry_session,
            allow_nonqualifying_without_telemetry_for_test=(
                allow_nonqualifying_without_telemetry_for_test
            ),
            legacy_single_action_paths=False,
        )

    def _run_managed_sequence(
        self,
        bundle_dir: Path,
        *,
        actions: Sequence[ManagedActionSpec],
        artifact_directory: Path,
        poll_interval_seconds: float,
        safety_check: (
            Callable[[ManagedActionSpec, Mapping[str, Any]], SafetySignal | None] | None
        ),
        pre_action_check: Callable[[ManagedActionSpec], SafetySignal | None] | None,
        selected_artifacts: Sequence[SelectedArtifact],
        telemetry_session: TelemetrySession | None,
        allow_nonqualifying_without_telemetry_for_test: bool,
        legacy_single_action_paths: bool,
    ) -> CaptureOutcome:
        poll_interval = _require_timeout(poll_interval_seconds, "poll_interval_seconds")
        if self.job_service is None:
            raise ValueError("Managed capture requires one owning JobService instance.")
        if not isinstance(allow_nonqualifying_without_telemetry_for_test, bool):
            raise ValueError("The test-only telemetry override must be boolean.")
        self._require_submission_allowed()
        if (
            telemetry_session is None
            and not allow_nonqualifying_without_telemetry_for_test
        ):
            raise ValueError(
                "Managed capture requires telemetry; the only override is explicitly nonqualifying and test-only."
            )
        specs = self._validate_action_sequence(
            actions,
            selected_artifacts,
            legacy_single_action_paths=legacy_single_action_paths,
        )
        self._validate_qualifying_sequence(
            bundle_dir,
            artifact_directory,
            specs,
            selected_artifacts,
            telemetry_session,
            legacy_single_action_paths=legacy_single_action_paths,
            allow_nonqualifying_without_telemetry_for_test=(
                allow_nonqualifying_without_telemetry_for_test
            ),
        )
        with self._managed_lock:
            self._require_submission_allowed()
            return self._run_managed_sequence_locked(
                bundle_dir,
                specs=specs,
                artifact_directory=artifact_directory,
                poll_interval_seconds=poll_interval,
                safety_check=safety_check,
                pre_action_check=pre_action_check,
                selected_artifacts=selected_artifacts,
                telemetry_session=telemetry_session,
                allow_nonqualifying_without_telemetry_for_test=(
                    allow_nonqualifying_without_telemetry_for_test
                ),
                legacy_single_action_paths=legacy_single_action_paths,
            )

    def _run_managed_sequence_locked(
        self,
        bundle_dir: Path,
        *,
        specs: tuple[ManagedActionSpec, ...],
        artifact_directory: Path,
        poll_interval_seconds: float,
        safety_check: (
            Callable[[ManagedActionSpec, Mapping[str, Any]], SafetySignal | None] | None
        ),
        pre_action_check: Callable[[ManagedActionSpec], SafetySignal | None] | None,
        selected_artifacts: Sequence[SelectedArtifact],
        telemetry_session: TelemetrySession | None,
        allow_nonqualifying_without_telemetry_for_test: bool,
        legacy_single_action_paths: bool,
    ) -> CaptureOutcome:
        service = self.job_service
        if service is None:  # pragma: no cover - guarded by public method.
            raise AssertionError("Owning service disappeared.")
        qualification_context = self.qualification_context
        identity_bindings: dict[str, Any] = {
            "attempt_slot_id": self.attempt_slot_id,
            "experiment_run_id": self.experiment_run_id,
            "capture_kind": (
                "managed-job" if legacy_single_action_paths else "managed-sequence"
            ),
            "capture_status": "complete",
        }
        if qualification_context is not None:
            identity_bindings.update(
                campaign_id=str(qualification_context.campaign["campaign_id"]),
                comparison_cohort_id=str(
                    qualification_context.comparison_cohort["comparison_cohort_id"]
                ),
                comparison_cell_id=str(
                    qualification_context.comparison_cell["comparison_cell_id"]
                ),
                execution_configuration_id=str(
                    qualification_context.execution_configuration[
                        "execution_configuration_id"
                    ]
                ),
            )
        artifact_id = new_opaque_id("artifact")
        writer = RawArtifactWriter(
            artifact_directory,
            protected_artifact_id=artifact_id,
            record_kind="experiment-run",
            identity_bindings=identity_bindings,
            capture_tool=self.capture_tool,
            source_bindings=self.source_bindings,
            provisional_retain_not_before_utc=(self.provisional_retain_not_before_utc),
            required_role_bindings={
                "event-ledger": "entry_event-ledger",
                "sequence-summary": "entry_sequence-summary",
            },
        )
        available: list[dict[str, Any]] = []
        bound_entry_ids: dict[str, list[str]] = {}
        source_snapshots: list[tuple[Path, tuple[int, ...], str]] = []
        capture_failure_reason: str | None = None

        def write_bytes(
            data: bytes,
            relative_path: str,
            role: str,
            media_type: str,
            entry_id: str,
        ) -> bool:
            nonlocal capture_failure_reason
            try:
                entry = writer.write_payload(
                    data,
                    relative_path,
                    role=role,
                    media_type=media_type,
                    entry_id=entry_id,
                )
                available.append(entry)
                bound_entry_ids.setdefault(role, []).append(entry["entry_id"])
            except Exception:
                capture_failure_reason = (
                    capture_failure_reason or "STREAM_CAPTURE_FAILURE"
                )
                return False
            return True

        def copy_file(
            source: Path,
            relative_path: str,
            role: str,
            media_type: str,
            entry_id: str | None = None,
        ) -> bool:
            nonlocal capture_failure_reason
            try:
                entry = writer.copy_payload(
                    source,
                    relative_path,
                    role=role,
                    media_type=media_type,
                    entry_id=entry_id,
                )
                source_identity = _stable_regular_source_identity(source)
                source_digest = _hash_stable_regular_source(source)
                if (
                    source_digest != entry["sha256"]
                    or source_identity[4] != entry["size_bytes"]
                    or _stable_regular_source_identity(source) != source_identity
                ):
                    raise ValueError("Selected source differs from its captured bytes.")
                source_snapshots.append((source, source_identity, source_digest))
                available.append(entry)
                bound_entry_ids.setdefault(role, []).append(entry["entry_id"])
            except Exception:
                capture_failure_reason = (
                    capture_failure_reason or "STREAM_CAPTURE_FAILURE"
                )
                return False
            return True

        ledger = self._new_ledger()
        telemetry_started = False
        telemetry_healthy: bool | None = None
        evidence_status = "capture-invalid"
        capture_reason_code = "MISSING_REQUIRED_EVIDENCE"
        native_outcome = "guard-blocked"
        reason_code = "TELEMETRY_COLLECTOR_FAILURE"
        exit_code: int | None = None
        timed_out = False
        results: list[_ManagedActionResult] = []
        capture_artifacts = tuple(
            artifact
            for artifact in selected_artifacts
            if artifact.role != "pilot-metrics"
        )
        telemetry_start_failed = False
        telemetry_start_ns: int | None = None
        telemetry_configuration: dict[str, Any] | None = None
        cooldown = WindowValidation(False, ("MISSING_REQUIRED_EVIDENCE",))

        def unexpected_lifecycle_outcome() -> CaptureOutcome:
            nonlocal telemetry_started, telemetry_healthy
            cancellation_certain = self._cancel_known_active_jobs(service)
            if telemetry_started:
                # Clear before stopping so even a ledger/write exception cannot
                # invoke the session's one-shot stop lifecycle twice.
                telemetry_started = False
                try:
                    _payload, telemetry_healthy, _failure_code = self._stop_telemetry(
                        ledger, telemetry_session, True
                    )
                except Exception:
                    telemetry_healthy = False
            return self._sealed_capture_failure_outcome(
                artifact_directory,
                protected_artifact_id=artifact_id,
                native_outcome="unknown",
                reason_code="OWNERSHIP_UNCERTAIN",
                capture_reason_code="STREAM_CAPTURE_FAILURE",
                exit_code=None,
                timed_out=False,
                telemetry_healthy=telemetry_healthy,
                available_files=available,
                missing_fields=(
                    "complete-managed-lifecycle",
                    "events/events.jsonl",
                    "SEALED.json",
                ),
                recoverable_locator=str(writer.directory),
                block_submissions=not cancellation_certain,
            )

        if telemetry_session is None:
            if allow_nonqualifying_without_telemetry_for_test:
                evidence_status = "capture-invalid"
                capture_reason_code = "MISSING_REQUIRED_EVIDENCE"
        else:
            telemetry_start_ns = self._monotonic_ns()
            if qualification_context is not None:
                try:
                    telemetry_configuration = dict(
                        telemetry_session.configuration_record()  # type: ignore[attr-defined]
                    )
                    validate_qualifying_telemetry_configuration(
                        telemetry_configuration,
                        context=qualification_context,
                    )
                except Exception:
                    telemetry_start_failed = True
                    capture_reason_code = "MISSING_REQUIRED_EVIDENCE"
                    capture_failure_reason = "MISSING_REQUIRED_EVIDENCE"
            try:
                if not telemetry_start_failed:
                    telemetry_started = self._start_telemetry(
                        ledger, telemetry_session, telemetry_start_ns
                    )
            except Exception:
                telemetry_start_failed = True
                evidence_status = "capture-invalid"
                capture_reason_code = "TELEMETRY_COLLECTOR_FAILURE"
                capture_failure_reason = "TELEMETRY_COLLECTOR_FAILURE"
                self._block_submissions(
                    reason_code="TELEMETRY_COLLECTOR_FAILURE",
                    job_id=None,
                    detected_monotonic_ns=telemetry_start_ns,
                )
                event_ns = max(self._monotonic_ns(), ledger.records[-1]["monotonic_ns"])
                ledger.append(
                    monotonic_ns=event_ns,
                    wall_time_utc=self._wall_time(),
                    event_type="telemetry.started",
                    subject_kind="experiment-run",
                    subject_id=self.experiment_run_id,
                )
                ledger.append(
                    monotonic_ns=max(self._monotonic_ns(), event_ns),
                    wall_time_utc=self._wall_time(),
                    event_type="telemetry.failed",
                    subject_kind="experiment-run",
                    subject_id=self.experiment_run_id,
                    reason_code="TELEMETRY_COLLECTOR_FAILURE",
                )

        if not telemetry_start_failed:
            for spec in specs:
                try:
                    result = self._supervise_managed_action(
                        service,
                        bundle_dir,
                        spec,
                        ledger=ledger,
                        poll_interval_seconds=poll_interval_seconds,
                        safety_check=safety_check,
                        telemetry_session=telemetry_session,
                        pre_action_check=pre_action_check,
                    )
                except Exception:
                    return unexpected_lifecycle_outcome()
                results.append(result)
                native_outcome = result.native_outcome
                reason_code = result.reason_code
                exit_code = result.exit_code
                timed_out = result.timed_out
                if result.capture_reason_code != "NONE":
                    evidence_status = "capture-invalid"
                    capture_reason_code = result.capture_reason_code
                prefix = (
                    "job" if legacy_single_action_paths else f"actions/{spec.label}"
                )
                if result.job_id is None:
                    captured = write_bytes(
                        canonical_json_bytes(dict(result.record)),
                        f"{prefix}/submission.json",
                        "action-submission-record",
                        "application/json",
                        f"entry_{spec.label}-submission",
                    )
                else:
                    record_role = (
                        "terminal-job-record"
                        if result.terminal
                        else "last-observed-job-record"
                    )
                    captured = write_bytes(
                        canonical_json_bytes(dict(result.record)),
                        f"{prefix}/terminal.json",
                        record_role,
                        "application/json",
                        f"entry_{spec.label}-job-record",
                    )
                    log_value = result.record.get("log")
                    log_source = (
                        Path(log_value)
                        if isinstance(log_value, str) and log_value
                        else self.state_root / ".missing-job-log"
                    )
                    captured = (
                        copy_file(
                            log_source,
                            f"{prefix}/full.log",
                            "job-log",
                            "text/plain",
                            f"entry_{spec.label}-job-log",
                        )
                        and captured
                    )
                    if result.runtime_journal_bytes is not None:
                        captured = (
                            write_bytes(
                                result.runtime_journal_bytes,
                                f"{prefix}/runtime-boundaries.jsonl",
                                "runtime-boundary-journal",
                                "application/x-ndjson",
                                f"entry_{spec.label}-runtime-boundaries",
                            )
                            and captured
                        )
                if not captured:
                    evidence_status = "capture-invalid"
                    capture_reason_code = "STREAM_CAPTURE_FAILURE"
                    break
                if not result.terminal and result.native_outcome != "unknown":
                    evidence_status = "capture-invalid"
                    capture_reason_code = "UNKNOWN_TERMINAL_STATE"
                    break
                if result.native_outcome != "passed":
                    break

        if (
            qualification_context is not None
            and telemetry_started
            and len(results) == len(specs)
            and all(result.native_outcome == "passed" for result in results)
        ):
            try:
                cooldown = self._capture_qualifying_cooldown(
                    ledger,
                    telemetry_session,  # type: ignore[arg-type]
                    qualification_context,
                )
            except Exception:
                cooldown = WindowValidation(False, ("TELEMETRY_COLLECTOR_FAILURE",))
            if not cooldown.valid:
                capture_reason_code = (
                    cooldown.reason_codes[0]
                    if cooldown.reason_codes
                    else "MISSING_REQUIRED_EVIDENCE"
                )

        telemetry_payload: bytes | None = None
        stopped_telemetry = _StoppedTelemetry(None, (), None, None, None)
        if telemetry_started:
            telemetry_started = False
            try:
                stopped_telemetry = self._stop_telemetry_detailed(
                    ledger, telemetry_session, True
                )
                telemetry_payload = stopped_telemetry.payload
                telemetry_healthy = stopped_telemetry.healthy
                telemetry_failure_code = stopped_telemetry.failure_code
            except Exception:
                return unexpected_lifecycle_outcome()
            if telemetry_healthy is not True:
                evidence_status = "capture-invalid"
                if capture_reason_code in {"NONE", "MISSING_REQUIRED_EVIDENCE"}:
                    capture_reason_code = (
                        telemetry_failure_code or "TELEMETRY_COLLECTOR_FAILURE"
                    )
            if telemetry_payload is not None:
                write_bytes(
                    telemetry_payload,
                    "telemetry/samples.jsonl",
                    "telemetry",
                    "application/x-ndjson",
                    "entry_telemetry",
                )

        if qualification_context is not None and results:
            pilot_result = next(
                (result for result in results if result.spec.action == "pilot"),
                None,
            )
            if pilot_result is not None and pilot_result.native_outcome == "passed":
                capture_artifacts += tuple(
                    artifact
                    for artifact in selected_artifacts
                    if artifact.role == "pilot-metrics"
                )
            train_result = next(
                (result for result in results if result.spec.action == "train"),
                None,
            )
            train_record = train_result.record if train_result is not None else {}
            train_output_value = train_record.get("run_output_dir")
            if (
                train_result is not None
                and train_result.native_outcome == "passed"
                and isinstance(train_output_value, str)
            ):
                train_output = Path(train_output_value)
                capture_artifacts += (
                    SelectedArtifact(
                        train_output / "metrics.json",
                        "selected/training-metrics.json",
                        "training-metrics",
                        "application/json",
                    ),
                    SelectedArtifact(
                        train_output / "final-export.json",
                        "selected/final-export-manifest.json",
                        "final-export-manifest",
                        "application/json",
                    ),
                )

        if qualification_context is not None:
            planned = self._planned_slot_context
            activation_directory = self._activation_directory
            if (
                type(planned) is not PlannedSlotContext
                or activation_directory is None
                or not write_bytes(
                    canonical_json_bytes(planned.record()),
                    "activation/planned-slot-context.json",
                    "planned-slot-context",
                    "application/json",
                    "entry_planned-slot-context",
                )
            ):
                capture_failure_reason = "MISSING_REQUIRED_EVIDENCE"
            else:
                capture_artifacts += tuple(
                    SelectedArtifact(
                        activation_directory / filename,
                        relative_path,
                        role,
                        media_type,
                    )
                    for filename, relative_path, role, media_type in (
                        _ACTIVATION_PROVENANCE
                    )
                )
            phase4 = self._phase4_verification
            if phase4 is None:
                capture_failure_reason = "MISSING_REQUIRED_EVIDENCE"
            else:
                capture_artifacts += (
                    SelectedArtifact(
                        phase4.directory / PHASE4_SOURCE_FREEZE_NAME,
                        "phase4/phase4-source-freeze.json",
                        "phase4-source-freeze",
                        "application/json",
                    ),
                    SelectedArtifact(
                        phase4.directory / PHASE4_SOURCE_FREEZE_SEAL_NAME,
                        "phase4/PHASE4-SEALED.json",
                        "phase4-source-freeze-seal",
                        "application/json",
                    ),
                    SelectedArtifact(
                        phase4.directory / PHASE4_IDLE_SAMPLES_NAME,
                        "phase4/idle-baseline-samples.jsonl",
                        "phase4-idle-baseline-samples",
                        "application/x-ndjson",
                    ),
                )

        for artifact in capture_artifacts:
            if not copy_file(
                artifact.source,
                validate_safe_relative_path(artifact.relative_path),
                artifact.role,
                artifact.media_type,
            ):
                evidence_status = "capture-invalid"
                capture_reason_code = "STREAM_CAPTURE_FAILURE"
                break

        qualification_decision = None
        telemetry_summary_payload: dict[str, Any] | None = None
        selected_role_digests = {
            str(entry["role"]): str(entry["sha256"])
            for entry in available
            if entry.get("role")
            in REQUIRED_QUALIFYING_ARTIFACT_ROLES | REQUIRED_QUALIFYING_AUTHORITY_ROLES
        }
        passing_candidate = len(results) == len(specs) and all(
            result.native_outcome == "passed" for result in results
        )
        if qualification_context is not None:
            activation_role_digests = {
                role: digest
                for role, digest in selected_role_digests.items()
                if role == "planned-slot-context" or role.startswith("activation-")
            }
            expected_activation_roles = {
                "planned-slot-context",
                *(item[2] for item in _ACTIVATION_PROVENANCE),
            }
            if set(activation_role_digests) == expected_activation_roles:
                writer.source_bindings["activation_provenance_sha256_by_role"] = dict(
                    sorted(activation_role_digests.items())
                )
            else:
                capture_failure_reason = (
                    capture_failure_reason or "MISSING_REQUIRED_EVIDENCE"
                )
            runtime_boundaries = [
                boundary for result in results for boundary in result.runtime_boundaries
            ]
            final_configuration = (
                dict(stopped_telemetry.configuration)
                if isinstance(stopped_telemetry.configuration, Mapping)
                else {}
            )
            qualification_failure_code: str | None = None
            try:
                if telemetry_session is None:
                    raise QualificationError(
                        "qualifying telemetry authority is unavailable"
                    )
                self._require_qualifying_runtime_authority(telemetry_session)
                if (
                    telemetry_start_ns is None
                    or stopped_telemetry.stop_monotonic_ns is None
                    or telemetry_configuration is None
                    or final_configuration != telemetry_configuration
                    or telemetry_healthy is not True
                ):
                    raise QualificationError(
                        "qualifying telemetry lifecycle is incomplete"
                    )
                if any(result.capture_reason_code != "NONE" for result in results):
                    raise QualificationError("qualifying action capture is incomplete")
                if passing_candidate:
                    train_record = results[-1].record
                    train_output = Path(
                        str(train_record.get("run_output_dir", ""))
                    ).resolve()
                    selected_by_role = {
                        artifact.role: artifact for artifact in capture_artifacts
                    }
                    if (
                        selected_by_role["training-metrics"].source.resolve()
                        != train_output / "metrics.json"
                        or selected_by_role["final-export-manifest"].source.resolve()
                        != train_output / "final-export.json"
                    ):
                        raise QualificationError(
                            "qualifying training artifacts are misbound"
                        )
                    qualification_decision = evaluate_passing_qualification(
                        context=qualification_context,
                        action_records=[dict(result.record) for result in results],
                        selected_artifact_roles={
                            artifact.role for artifact in capture_artifacts
                        },
                        runtime_boundaries=runtime_boundaries,
                        telemetry_samples=list(stopped_telemetry.samples),
                        telemetry_start_monotonic_ns=telemetry_start_ns,
                        telemetry_stop_monotonic_ns=(
                            stopped_telemetry.stop_monotonic_ns
                        ),
                        telemetry_configuration=final_configuration,
                        safety_events=[
                            dict(item) for item in stopped_telemetry.safety_events
                        ],
                        cooldown=cooldown,
                        ledger_records=list(ledger.records),
                    )
                    if not qualification_decision.valid:
                        qualification_failure_code = (
                            qualification_decision.reason_codes[0]
                            if qualification_decision.reason_codes
                            else "MISSING_REQUIRED_EVIDENCE"
                        )
                    else:
                        telemetry_summary_payload = {
                            "record_kind": ("aptus-cuda-campaign-telemetry-summary-v1"),
                            "experiment_run_id": self.experiment_run_id,
                            "telemetry": qualification_decision.telemetry_summary,
                            "segments": list(qualification_decision.segment_summaries),
                        }
                else:
                    telemetry_summary = summarize_telemetry(
                        list(stopped_telemetry.samples),
                        telemetry_start_ns,
                        stopped_telemetry.stop_monotonic_ns,
                    )
                    present_slots = {
                        sample["scheduled_slot"] for sample in stopped_telemetry.samples
                    }
                    telemetry_summary["missing_scheduled_slots"] = [
                        slot
                        for slot in range(telemetry_summary["expected_sample_count"])
                        if slot not in present_slots
                    ]
                    telemetry_summary_payload = {
                        "record_kind": "aptus-cuda-campaign-telemetry-summary-v1",
                        "experiment_run_id": self.experiment_run_id,
                        "telemetry": telemetry_summary,
                        "segments": build_segment_summaries(
                            list(stopped_telemetry.samples),
                            list(ledger.records),
                            allow_open_terminal_prefix=True,
                        ),
                    }
            except (KeyError, OSError, QualificationError, TypeError, ValueError):
                qualification_failure_code = (
                    qualification_failure_code or "MISSING_REQUIRED_EVIDENCE"
                )

            expected_selected_roles = REQUIRED_QUALIFYING_ARTIFACT_ROLES | (
                REQUIRED_QUALIFYING_AUTHORITY_ROLES
                - {
                    "campaign-record",
                    "comparison-cohort-record",
                    "comparison-cell-record",
                }
            )
            if not passing_candidate:
                expected_selected_roles -= {
                    "training-metrics",
                    "final-export-manifest",
                }
                if not any(
                    result.spec.action == "pilot" and result.native_outcome == "passed"
                    for result in results
                ):
                    expected_selected_roles.discard("pilot-metrics")
            if set(selected_role_digests) != expected_selected_roles:
                qualification_failure_code = "MISSING_REQUIRED_EVIDENCE"

            if (
                qualification_failure_code is None
                and capture_failure_reason is None
                and telemetry_summary_payload is not None
                and (not passing_candidate or qualification_decision is not None)
            ):
                evidence_status = "protocol-valid"
                capture_reason_code = "NONE"
            else:
                evidence_status = "capture-invalid"
                capture_reason_code = (
                    qualification_failure_code or "MISSING_REQUIRED_EVIDENCE"
                )

        command_boundaries = [
            row
            for row in ledger.records
            if row["event_type"] in {"command.started", "command.finished"}
        ]
        five_action_duration_ns = (
            command_boundaries[-1]["monotonic_ns"]
            - command_boundaries[0]["monotonic_ns"]
            if len(results) == 5 and len(command_boundaries) == 10
            else None
        )
        summary = {
            "record_kind": "aptus-cuda-campaign-managed-sequence-v1",
            "experiment_run_id": self.experiment_run_id,
            "attempt_slot_id": self.attempt_slot_id,
            "configured_actions": [
                {
                    "label": spec.label,
                    "action": spec.action,
                    "supervision_timeout_seconds": (spec.supervision_timeout_seconds),
                    "submit_kwargs": dict(spec.submit_kwargs),
                }
                for spec in specs
            ],
            "started_actions": [
                {
                    "label": result.spec.label,
                    "action": result.spec.action,
                    "job_id": result.job_id,
                    "native_outcome": result.native_outcome,
                    "reason_code": result.reason_code,
                    "terminal": result.terminal,
                    "capture_reason_code": result.capture_reason_code,
                }
                for result in results
            ],
            "native_outcome": native_outcome,
            "reason_code": reason_code,
            "evidence_status": evidence_status,
            "capture_reason_code": capture_reason_code,
            "telemetry_required": True,
            "telemetry_test_override": (allow_nonqualifying_without_telemetry_for_test),
            "telemetry_healthy": telemetry_healthy,
            "stopped_early": len(results) < len(specs)
            or any(result.native_outcome != "passed" for result in results),
            "five_action_duration_ns": five_action_duration_ns,
        }
        ledger_payload = self._finish_ledger(
            ledger,
            native_outcome=native_outcome,
            reason_code=reason_code,
            exit_code=exit_code,
        )
        if qualification_context is not None:
            try:
                outcome_profile = validate_managed_sequence_outcome(
                    summary, list(ledger.records)
                )
                publication_expected = (
                    passing_candidate
                    and qualification_decision is not None
                    and qualification_decision.valid
                    and capture_failure_reason is None
                    and evidence_status == "protocol-valid"
                )
                if outcome_profile.publication_eligible != publication_expected:
                    raise OutcomeProfileError(
                        "managed outcome eligibility differs from qualification"
                    )
            except (OutcomeProfileError, TypeError, ValueError):
                capture_failure_reason = (
                    capture_failure_reason or "MISSING_REQUIRED_EVIDENCE"
                )
                evidence_status = "capture-invalid"
                if capture_reason_code == "NONE":
                    capture_reason_code = "MISSING_REQUIRED_EVIDENCE"
                summary["evidence_status"] = evidence_status
                summary["capture_reason_code"] = capture_reason_code

        if (
            qualification_context is not None
            and results
            and capture_failure_reason is None
        ):
            action_records = [dict(result.record) for result in results]
            try:
                preliminary_slot, _preliminary_run = (
                    qualification_context.finalize_records(
                        action_records,
                        native_outcome=native_outcome,
                        evidence_status=evidence_status,
                        reason_code=reason_code,
                    )
                )
                qualifying_payloads = {
                    "campaign-record": canonical_json_bytes(
                        dict(qualification_context.campaign)
                    ),
                    "comparison-cohort-record": canonical_json_bytes(
                        dict(qualification_context.comparison_cohort)
                    ),
                    "comparison-cell-record": canonical_json_bytes(
                        dict(qualification_context.comparison_cell)
                    ),
                    "attempt-slot-record": canonical_json_bytes(preliminary_slot),
                    "execution-configuration-record": canonical_json_bytes(
                        dict(qualification_context.execution_configuration)
                    ),
                    "idle-baseline-binding": canonical_json_bytes(
                        dict(qualification_context.idle_baseline_binding)
                    ),
                    "telemetry-configuration": canonical_json_bytes(
                        dict(stopped_telemetry.configuration or {})
                    ),
                }
                if telemetry_summary_payload is not None:
                    qualifying_payloads["telemetry-summary"] = canonical_json_bytes(
                        telemetry_summary_payload
                    )
                if (
                    passing_candidate
                    and qualification_decision is not None
                    and qualification_decision.valid
                ):
                    qualifying_payloads["cooldown-summary"] = canonical_json_bytes(
                        {
                            "record_kind": ("aptus-cuda-campaign-cooldown-summary-v1"),
                            "experiment_run_id": self.experiment_run_id,
                            "valid": True,
                            "reason_codes": [],
                            "summary": qualification_decision.cooldown_summary,
                        }
                    )
                terminal_role_digests = {
                    role: sha256_bytes(payload)
                    for role, payload in qualifying_payloads.items()
                }
                terminal_role_digests.update(selected_role_digests)
                finalized_slot, finalized_run = qualification_context.finalize_records(
                    action_records,
                    native_outcome=native_outcome,
                    evidence_status=evidence_status,
                    reason_code=reason_code,
                    evidence_role_sha256=(
                        terminal_role_digests
                        if evidence_status == "protocol-valid"
                        else None
                    ),
                )
                qualifying_payloads["attempt-slot-record"] = canonical_json_bytes(
                    finalized_slot
                )
                qualifying_payloads["experiment-run-record"] = canonical_json_bytes(
                    finalized_run
                )
                all_role_digests = {
                    role: sha256_bytes(payload)
                    for role, payload in qualifying_payloads.items()
                }
                all_role_digests.update(selected_role_digests)
                writer.source_bindings["evidence_role_sha256"] = dict(
                    sorted(all_role_digests.items())
                )
                qualifying_paths = {
                    "campaign-record": "records/campaign.json",
                    "comparison-cohort-record": "records/comparison-cohort.json",
                    "comparison-cell-record": "records/comparison-cell.json",
                    "attempt-slot-record": "records/attempt-slot.json",
                    "execution-configuration-record": (
                        "records/execution-configuration.json"
                    ),
                    "experiment-run-record": "records/experiment-run.json",
                    "idle-baseline-binding": "records/idle-baseline-binding.json",
                    "telemetry-configuration": "telemetry/configuration.json",
                    "telemetry-summary": "telemetry/summary.json",
                    "cooldown-summary": "telemetry/cooldown-summary.json",
                }
                for role, payload in qualifying_payloads.items():
                    if not write_bytes(
                        payload,
                        qualifying_paths[role],
                        role,
                        "application/json",
                        f"entry_{role}",
                    ):
                        evidence_status = "capture-invalid"
                        capture_reason_code = "STREAM_CAPTURE_FAILURE"
                        break
            except (QualificationError, TypeError, ValueError):
                capture_failure_reason = "MISSING_REQUIRED_EVIDENCE"
                evidence_status = "capture-invalid"
                capture_reason_code = "MISSING_REQUIRED_EVIDENCE"
        summary["evidence_status"] = evidence_status
        summary["capture_reason_code"] = capture_reason_code
        write_bytes(
            canonical_json_bytes(summary),
            "sequence/summary.json",
            "sequence-summary",
            "application/json",
            "entry_sequence-summary",
        )
        write_bytes(
            ledger_payload,
            "events/events.jsonl",
            "event-ledger",
            "application/x-ndjson",
            "entry_event-ledger",
        )
        try:
            for source, identity, digest in source_snapshots:
                if (
                    _stable_regular_source_identity(source) != identity
                    or _hash_stable_regular_source(source) != digest
                    or _stable_regular_source_identity(source) != identity
                ):
                    raise ValueError(
                        "Selected source changed before the final inventory."
                    )
        except Exception:
            capture_failure_reason = "ARTIFACT_INTEGRITY_FAILURE"
            evidence_status = "capture-invalid"
            capture_reason_code = "ARTIFACT_INTEGRITY_FAILURE"

        if capture_failure_reason is None:
            writer.identity_bindings.update(
                evidence_status=evidence_status,
                capture_reason_code=capture_reason_code,
            )
            single_roles = {
                "event-ledger",
                "sequence-summary",
                "telemetry",
                "attempt-slot-record",
                "execution-configuration-record",
                "experiment-run-record",
                "idle-baseline-binding",
                "telemetry-configuration",
                "telemetry-summary",
                "cooldown-summary",
                *REQUIRED_QUALIFYING_ARTIFACT_ROLES,
                *REQUIRED_QUALIFYING_AUTHORITY_ROLES,
            }
            captured_roles = {
                "event-ledger",
                "sequence-summary",
                "telemetry",
                "job-log",
                "terminal-job-record",
                "last-observed-job-record",
                "action-submission-record",
                "runtime-boundary-journal",
                "attempt-slot-record",
                "execution-configuration-record",
                "experiment-run-record",
                "idle-baseline-binding",
                "telemetry-configuration",
                "telemetry-summary",
                "cooldown-summary",
                *REQUIRED_QUALIFYING_ARTIFACT_ROLES,
                *REQUIRED_QUALIFYING_AUTHORITY_ROLES,
            }
            writer.required_role_bindings = {
                role: identifiers[0] if role in single_roles else list(identifiers)
                for role, identifiers in sorted(bound_entry_ids.items())
                if role in captured_roles
            }
            try:
                verification = writer.seal()
            except Exception:
                capture_failure_reason = "SEAL_FAILURE"
            else:
                return CaptureOutcome(
                    experiment_run_id=self.experiment_run_id,
                    attempt_slot_id=self.attempt_slot_id,
                    native_outcome=native_outcome,
                    reason_code=reason_code,
                    evidence_status=evidence_status,
                    capture_reason_code=capture_reason_code,
                    artifact_directory=writer.directory,
                    sealed=True,
                    seal_verification=verification,
                    capture_failure_receipt=None,
                    exit_code=exit_code,
                    timed_out=timed_out,
                    submission_blocked=self.submissions_blocked,
                    telemetry_healthy=telemetry_healthy,
                )

        fallback_reason = capture_failure_reason or "STREAM_CAPTURE_FAILURE"
        fallback_directory = artifact_directory.with_name(
            artifact_directory.name + ".capture-failure"
        )
        try:
            write_sealed_capture_failure_artifact(
                fallback_directory,
                protected_artifact_id=artifact_id,
                attempt_slot_id=self.attempt_slot_id,
                experiment_run_id=self.experiment_run_id,
                reason_code=fallback_reason,
                available_files=available,
                missing_fields=["required-capture-payload", "SEALED.json"],
                recoverable_locator=str(writer.directory),
                capture_tool=self.capture_tool,
                source_bindings=self.source_bindings,
                provisional_retain_not_before_utc=(
                    self.provisional_retain_not_before_utc
                ),
            )
        except Exception:
            self._block_submissions(
                reason_code="SEAL_FAILURE",
                job_id=None,
                detected_monotonic_ns=self._monotonic_ns(),
            )
            raise CaptureHarnessError(
                "Normal capture and its sealed fallback both failed; later submissions are blocked."
            ) from None
        return CaptureOutcome(
            experiment_run_id=self.experiment_run_id,
            attempt_slot_id=self.attempt_slot_id,
            native_outcome=native_outcome,
            reason_code=reason_code,
            evidence_status="capture-invalid",
            capture_reason_code=fallback_reason,
            artifact_directory=writer.directory,
            sealed=False,
            seal_verification=None,
            capture_failure_receipt=fallback_directory / "capture-failure.json",
            exit_code=exit_code,
            timed_out=timed_out,
            submission_blocked=self.submissions_blocked,
            telemetry_healthy=telemetry_healthy,
        )


def _install_qualifying_factory_authority(
    factory: Callable[..., CaptureHarness],
) -> tuple[classmethod, Callable[[CaptureHarness, object | None], bool]]:
    """Keep the qualifying-construction registry inside an unexported closure."""

    registry: weakref.WeakKeyDictionary[CaptureHarness, tuple[object, ...]] = (
        weakref.WeakKeyDictionary()
    )
    lock = threading.Lock()

    def registered_factory(
        cls: type[CaptureHarness], *args: Any, **kwargs: Any
    ) -> CaptureHarness:
        harness = factory(cls, *args, **kwargs)
        snapshot = (
            harness._qualifying_authority,
            harness.job_service,
            harness.qualification_context,
            harness._phase4_verification,
            harness._phase4_repository_root,
            harness._planned_slot_context,
            harness._activation_authority,
            harness._activation_directory,
            harness._admission_filesystem_device,
        )
        with lock:
            registry[harness] = snapshot
        return harness

    def is_registered(harness: CaptureHarness, authority: object | None = None) -> bool:
        with lock:
            expected = registry.get(harness)
        observed = (
            harness._qualifying_authority,
            harness.job_service,
            harness.qualification_context,
            harness._phase4_verification,
            harness._phase4_repository_root,
            harness._planned_slot_context,
            harness._activation_authority,
            harness._activation_directory,
            harness._admission_filesystem_device,
        )
        return bool(
            expected is not None
            and all(
                observed[index] is expected[index] for index in range(len(expected) - 1)
            )
            and observed[-1] == expected[-1]
            and (authority is None or authority is expected[0])
            and type(harness) is CaptureHarness
            and type(harness.job_service) is JobService
            and harness.job_service is harness._qualifying_job_service
            and harness._monotonic_ns is time.monotonic_ns
            and harness._wall_time is utc_now
            and harness._sleep is time.sleep
        )

    return classmethod(registered_factory), is_registered


_original_qualifying_factory = CaptureHarness.__dict__[
    "for_qualifying_campaign"
].__func__
(
    CaptureHarness.for_qualifying_campaign,
    _qualifying_harness_is_registered,
) = _install_qualifying_factory_authority(_original_qualifying_factory)
del _install_qualifying_factory_authority, _original_qualifying_factory


__all__ = [
    "CANCEL_RECONCILIATION_SLA_NS",
    "CANCEL_REQUEST_SLA_NS",
    "CANCEL_SEQUENCE_SLA_NS",
    "CANCEL_TERMINATION_SLA_NS",
    "CancellationMilestones",
    "CancellationSLAError",
    "CaptureHarness",
    "CaptureHarnessError",
    "CaptureOutcome",
    "ManagedActionSpec",
    "SafetySignal",
    "SelectedArtifact",
    "SubmissionBlockedError",
    "TelemetryCapture",
    "TelemetrySession",
    "verify_cancellation_milestones",
]
