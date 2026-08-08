"""Strict reader for opt-in CUDA runtime boundary journals.

The bundle runtime and the owning ``JobService`` append these records.  The
capture harness only observes and projects them into the canonical event
ledger; it never fabricates a successful runtime boundary.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover - qualifying CUDA capture is Linux-only.
    fcntl = None

from .contracts import NATIVE_OUTCOMES, REASON_CODES, compact_canonical_json_bytes


RUNTIME_BOUNDARY_SCHEMA = "aptus.cuda-campaign-runtime-boundary.v1"
MAX_RUNTIME_JOURNAL_BYTES = 1024 * 1024
MAX_RUNTIME_BOUNDARIES = 256
_RUN_ID = re.compile(r"^xrun_[0-9a-f]{32}$")
_JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
_EVENT_TYPES = frozenset(
    {
        "pilot.phase-started",
        "pilot.phase-finished",
        "training.started",
        "training.finished",
        "export.started",
        "export.finished",
        "verification.started",
        "verification.finished",
    }
)
_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_run_id",
        "job_id",
        "monotonic_ns",
        "wall_time_utc",
        "event_type",
        "phase",
        "action",
        "native_outcome",
        "reason_code",
    }
)
_T = TypeVar("_T")


class RuntimeBoundaryError(ValueError):
    """The runtime event journal is missing, mutable, or semantically invalid."""


@dataclass(frozen=True)
class RuntimeBoundary:
    experiment_run_id: str
    job_id: str
    monotonic_ns: int
    wall_time_utc: str
    event_type: str
    phase: str
    action: str
    native_outcome: str | None
    reason_code: str

    def ledger_fields(self) -> dict[str, Any]:
        return {
            "monotonic_ns": self.monotonic_ns,
            "wall_time_utc": self.wall_time_utc,
            "event_type": self.event_type,
            "phase": self.phase,
            "action": self.action,
            "subject_kind": "aptus-job",
            "subject_id": self.job_id,
            "observation_kind": "emitted",
            "native_outcome": self.native_outcome,
            "reason_code": self.reason_code,
        }


def _validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise RuntimeBoundaryError("runtime boundary UTC timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise RuntimeBoundaryError(
            "runtime boundary UTC timestamp is invalid"
        ) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeBoundaryError("runtime boundary UTC timestamp lacks an offset")
    return value


def validate_runtime_boundary(
    value: Any,
    *,
    expected_run_id: str,
    expected_job_id: str,
    expected_action: str,
) -> RuntimeBoundary:
    if type(value) is not dict or set(value) != _FIELDS:
        raise RuntimeBoundaryError("runtime boundary has the wrong exact fields")
    if value["schema_version"] != RUNTIME_BOUNDARY_SCHEMA:
        raise RuntimeBoundaryError("runtime boundary schema is unsupported")
    if value["experiment_run_id"] != expected_run_id or value["job_id"] != (
        expected_job_id
    ):
        raise RuntimeBoundaryError("runtime boundary identity does not match its job")
    if value["action"] != expected_action:
        raise RuntimeBoundaryError("runtime boundary action does not match its job")
    monotonic_ns = value["monotonic_ns"]
    if (
        isinstance(monotonic_ns, bool)
        or not isinstance(monotonic_ns, int)
        or monotonic_ns < 0
    ):
        raise RuntimeBoundaryError("runtime boundary monotonic timestamp is invalid")
    event_type = value["event_type"]
    phase = value["phase"]
    if event_type not in _EVENT_TYPES or not isinstance(phase, str) or not phase:
        raise RuntimeBoundaryError("runtime boundary type or phase is invalid")
    expected_event_action = "pilot" if event_type.startswith("pilot.") else "train"
    if expected_action != expected_event_action:
        raise RuntimeBoundaryError("runtime boundary type is invalid for its action")
    allowed_phases = (
        {"pilot-phase-1", "pilot-phase-2"}
        if expected_action == "pilot"
        else {
            "training"
            if event_type.startswith("training.")
            else "final-export"
            if event_type.startswith("export.")
            else "parent-verification"
        }
    )
    if phase not in allowed_phases:
        raise RuntimeBoundaryError("runtime boundary phase is not frozen")
    native_outcome = value["native_outcome"]
    reason_code = value["reason_code"]
    if native_outcome is not None and native_outcome not in NATIVE_OUTCOMES:
        raise RuntimeBoundaryError("runtime boundary outcome is invalid")
    if reason_code not in REASON_CODES:
        raise RuntimeBoundaryError("runtime boundary reason is invalid")
    started = event_type.endswith("started")
    if started and (native_outcome is not None or reason_code != "NONE"):
        raise RuntimeBoundaryError("a started runtime boundary cannot be terminal")
    if not started and (
        native_outcome not in {"passed", "failed", "cancelled"}
        or (native_outcome == "passed") != (reason_code == "NONE")
    ):
        raise RuntimeBoundaryError("a finished runtime boundary is not terminal")
    return RuntimeBoundary(
        experiment_run_id=expected_run_id,
        job_id=expected_job_id,
        monotonic_ns=monotonic_ns,
        wall_time_utc=_validate_timestamp(value["wall_time_utc"]),
        event_type=event_type,
        phase=phase,
        action=expected_action,
        native_outcome=native_outcome,
        reason_code=reason_code,
    )


class RuntimeBoundaryJournalReader:
    """Read one identity-pinned, append-only private runtime event journal."""

    def __init__(
        self,
        path: Path,
        *,
        expected_file_identity: str,
        experiment_run_id: str,
        job_id: str,
        action: str,
    ) -> None:
        path = Path(path)
        if not path.is_absolute():
            raise RuntimeBoundaryError("runtime boundary path must be absolute")
        if _RUN_ID.fullmatch(experiment_run_id) is None:
            raise RuntimeBoundaryError("runtime boundary run ID is invalid")
        if _JOB_ID.fullmatch(job_id) is None:
            raise RuntimeBoundaryError("runtime boundary job ID is invalid")
        if action not in {"pilot", "train"}:
            raise RuntimeBoundaryError("runtime boundary action is unsupported")
        if re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", expected_file_identity) is None:
            raise RuntimeBoundaryError("runtime boundary file identity is invalid")
        self.path = path
        self.expected_file_identity = expected_file_identity
        self.experiment_run_id = experiment_run_id
        self.job_id = job_id
        self.action = action
        self._verified_bytes = b""
        self._records: tuple[RuntimeBoundary, ...] = ()

    @property
    def records(self) -> tuple[RuntimeBoundary, ...]:
        return self._records

    @property
    def verified_bytes(self) -> bytes:
        """Return the exact append-only prefix accepted by the last drain."""

        return bytes(self._verified_bytes)

    def _validated_metadata(self, descriptor: int) -> os.stat_result:
        metadata = os.fstat(descriptor)
        observed_identity = f"{metadata.st_dev}:{metadata.st_ino}"
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or observed_identity != self.expected_file_identity
            or metadata.st_size > MAX_RUNTIME_JOURNAL_BYTES
        ):
            raise RuntimeBoundaryError(
                "runtime boundary journal integrity check failed"
            )
        return metadata

    def _with_locked_data(
        self,
        lock_operation: int,
        consume: Callable[[bytes], _T],
    ) -> _T:
        """Pin, lock, read, and consume one stable journal snapshot."""

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except OSError:
            raise RuntimeBoundaryError(
                "runtime boundary journal is unavailable"
            ) from None
        try:
            self._validated_metadata(descriptor)
            if fcntl is None:
                raise RuntimeBoundaryError(
                    "runtime boundary journal locking is unavailable"
                )
            fcntl.flock(descriptor, lock_operation)
            try:
                locked_metadata = self._validated_metadata(descriptor)
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 64 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
                finished_metadata = self._validated_metadata(descriptor)
                if (
                    finished_metadata.st_size != locked_metadata.st_size
                    or len(data) != locked_metadata.st_size
                ):
                    raise RuntimeBoundaryError(
                        "runtime boundary journal changed during locked read"
                    )
                return consume(data)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _validated_snapshot(
        self, data: bytes
    ) -> tuple[tuple[RuntimeBoundary, ...], tuple[RuntimeBoundary, ...]]:
        """Parse one snapshot against the already accepted append-only prefix."""

        if not data.startswith(self._verified_bytes):
            raise RuntimeBoundaryError("runtime boundary journal is not append-only")
        if data and not data.endswith(b"\n"):
            raise RuntimeBoundaryError("runtime boundary journal has a partial record")
        lines = data.splitlines()
        if len(lines) > MAX_RUNTIME_BOUNDARIES:
            raise RuntimeBoundaryError("runtime boundary journal has too many records")
        records: list[RuntimeBoundary] = []
        prior_ns = -1
        for line in lines:
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise RuntimeBoundaryError("runtime boundary JSON is invalid") from None
            if compact_canonical_json_bytes(value) != line:
                raise RuntimeBoundaryError("runtime boundary JSON is not canonical")
            boundary = validate_runtime_boundary(
                value,
                expected_run_id=self.experiment_run_id,
                expected_job_id=self.job_id,
                expected_action=self.action,
            )
            if boundary.monotonic_ns < prior_ns:
                raise RuntimeBoundaryError("runtime boundary time moved backward")
            prior_ns = boundary.monotonic_ns
            records.append(boundary)
        previous_count = len(self._records)
        if tuple(records[:previous_count]) != self._records:
            raise RuntimeBoundaryError("runtime boundary records changed after capture")
        snapshot = tuple(records)
        return snapshot, snapshot[previous_count:]

    def _commit_snapshot(
        self,
        data: bytes,
        records: tuple[RuntimeBoundary, ...],
    ) -> None:
        self._verified_bytes = data
        self._records = records

    def drain(self) -> tuple[RuntimeBoundary, ...]:
        """Drain new records while holding the journal's shared reader lock."""

        def consume(data: bytes) -> tuple[RuntimeBoundary, ...]:
            records, new_records = self._validated_snapshot(data)
            self._commit_snapshot(data, records)
            return new_records

        lock_operation = fcntl.LOCK_SH if fcntl is not None else 0
        return self._with_locked_data(lock_operation, consume)

    def drain_and_sample_monotonic(
        self,
        monotonic_ns: Callable[[], int],
    ) -> tuple[tuple[RuntimeBoundary, ...], int]:
        """Drain and sample parent time under one exclusive journal lock.

        Runtime emitters acquire the same exclusive lock and timestamp only
        after acquisition.  Consequently, a child boundary appended after this
        method unlocks cannot precede ``sampled_ns`` in the shared monotonic
        clock domain.
        """

        if not callable(monotonic_ns):
            raise RuntimeBoundaryError("runtime boundary monotonic clock is invalid")

        def consume(
            data: bytes,
        ) -> tuple[tuple[RuntimeBoundary, ...], int]:
            records, new_records = self._validated_snapshot(data)
            try:
                sampled_ns = monotonic_ns()
            except Exception as error:
                raise RuntimeBoundaryError(
                    "runtime boundary monotonic clock failed"
                ) from error
            if (
                isinstance(sampled_ns, bool)
                or not isinstance(sampled_ns, int)
                or sampled_ns < 0
            ):
                raise RuntimeBoundaryError(
                    "runtime boundary monotonic clock returned an invalid value"
                )
            self._commit_snapshot(data, records)
            return new_records, sampled_ns

        lock_operation = fcntl.LOCK_EX if fcntl is not None else 0
        return self._with_locked_data(lock_operation, consume)


__all__ = [
    "MAX_RUNTIME_BOUNDARIES",
    "MAX_RUNTIME_JOURNAL_BYTES",
    "RUNTIME_BOUNDARY_SCHEMA",
    "RuntimeBoundary",
    "RuntimeBoundaryError",
    "RuntimeBoundaryJournalReader",
    "validate_runtime_boundary",
]
