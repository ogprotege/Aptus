"""Opt-in runtime boundary emission for the CUDA evidence campaign.

Ordinary bundles do not set the four campaign environment variables, so this
module is a no-op outside the separately managed evidence harness.  When the
campaign channel is enabled, losing or replacing its private journal is a hard
runtime error rather than silently producing incomplete evidence.
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
from datetime import datetime, timezone

try:
    import fcntl
except ImportError:  # pragma: no cover - CUDA bundles are executed on Linux.
    fcntl = None


_SINK_ENV = "APTUS_CUDA_CAMPAIGN_EVENT_SINK"
_SINK_IDENTITY_ENV = "APTUS_CUDA_CAMPAIGN_EVENT_SINK_IDENTITY"
_RUN_ID_ENV = "APTUS_CUDA_CAMPAIGN_EXPERIMENT_RUN_ID"
_JOB_ID_ENV = "APTUS_CUDA_CAMPAIGN_JOB_ID"
_RUN_ID = re.compile(r"^xrun_[0-9a-f]{32}$")
_JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
_SINK_IDENTITY = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")
_EVENT_TYPES = frozenset(
    {
        "pilot.phase-started",
        "pilot.phase-finished",
        "training.started",
        "training.finished",
        "export.started",
        "export.finished",
    }
)
_RUNTIME_REASON_CODES = frozenset(
    {
        "NONE",
        "ARTIFACT_INTEGRITY_FAILURE",
        "CHECKPOINT_CONTINUATION_FAILURE",
        "CUDA_OOM",
        "EXPORT_VERIFICATION_FAILURE",
        "NONFINITE_VALUE",
        "PROCESS_EXIT_NONZERO",
    }
)
_MAX_JOURNAL_BYTES = 1024 * 1024


def _campaign_binding() -> tuple[str, str, str, str] | None:
    values = (
        os.environ.get(_SINK_ENV),
        os.environ.get(_SINK_IDENTITY_ENV),
        os.environ.get(_RUN_ID_ENV),
        os.environ.get(_JOB_ID_ENV),
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise RuntimeError("CUDA campaign event binding is incomplete.")
    sink, sink_identity, run_id, job_id = values
    assert sink is not None
    assert sink_identity is not None
    assert run_id is not None
    assert job_id is not None
    if not os.path.isabs(sink) or "\x00" in sink:
        raise RuntimeError("CUDA campaign event sink path is invalid.")
    if _SINK_IDENTITY.fullmatch(sink_identity) is None:
        raise RuntimeError("CUDA campaign event sink identity is invalid.")
    if _RUN_ID.fullmatch(run_id) is None or _JOB_ID.fullmatch(job_id) is None:
        raise RuntimeError("CUDA campaign run or job identity is invalid.")
    return sink, sink_identity, run_id, job_id


def _rank_zero() -> bool:
    raw = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        rank = int(raw)
    except ValueError:
        raise RuntimeError("CUDA campaign runtime rank is invalid.") from None
    if rank < 0:
        raise RuntimeError("CUDA campaign runtime rank is invalid.")
    return rank == 0


def _require_sink_metadata(metadata: os.stat_result, *, expected_identity: str) -> None:
    observed_identity = f"{metadata.st_dev}:{metadata.st_ino}"
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.getuid()
        or observed_identity != expected_identity
        or metadata.st_size > _MAX_JOURNAL_BYTES
    ):
        raise RuntimeError("CUDA campaign event sink integrity check failed.")


def emit_boundary(
    event_type: str,
    *,
    phase: str,
    action: str,
    native_outcome: str | None = None,
    reason_code: str = "NONE",
) -> None:
    """Append one canonical, identity-bound runtime boundary when opted in."""

    binding = _campaign_binding()
    if binding is None or not _rank_zero():
        return
    if event_type not in _EVENT_TYPES:
        raise RuntimeError("CUDA campaign runtime event type is invalid.")
    expected_action = "pilot" if event_type.startswith("pilot.") else "train"
    expected_phases = (
        {"pilot-phase-1", "pilot-phase-2"}
        if expected_action == "pilot"
        else {"training"}
        if event_type.startswith("training.")
        else {"final-export"}
    )
    if action != expected_action or phase not in expected_phases:
        raise RuntimeError("CUDA campaign runtime phase or action is invalid.")
    if native_outcome is not None and native_outcome not in {
        "passed",
        "failed",
        "cancelled",
    }:
        raise RuntimeError("CUDA campaign runtime outcome is invalid.")
    if reason_code not in _RUNTIME_REASON_CODES:
        raise RuntimeError("CUDA campaign runtime reason is invalid.")
    started = event_type.endswith("started")
    if started and (native_outcome is not None or reason_code != "NONE"):
        raise RuntimeError("CUDA campaign started boundary cannot be terminal.")
    if not started and (
        native_outcome not in {"passed", "failed", "cancelled"}
        or (native_outcome == "passed") != (reason_code == "NONE")
    ):
        raise RuntimeError("CUDA campaign finished boundary is not terminal.")

    sink, expected_identity, run_id, job_id = binding
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(sink, flags)
    try:
        metadata = os.fstat(descriptor)
        _require_sink_metadata(metadata, expected_identity=expected_identity)
        if fcntl is None:
            raise RuntimeError("CUDA campaign event journal locking is unavailable.")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            metadata = os.fstat(descriptor)
            _require_sink_metadata(metadata, expected_identity=expected_identity)
            record = {
                "schema_version": "aptus.cuda-campaign-runtime-boundary.v1",
                "experiment_run_id": run_id,
                "job_id": job_id,
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "phase": phase,
                "action": action,
                "native_outcome": native_outcome,
                "reason_code": reason_code,
            }
            payload = (
                json.dumps(
                    record,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            if metadata.st_size + len(payload) > _MAX_JOURNAL_BYTES:
                raise RuntimeError("CUDA campaign event journal size limit exceeded.")
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise RuntimeError("CUDA campaign event append made no progress.")
                written += count
            os.fsync(descriptor)
            _require_sink_metadata(
                os.fstat(descriptor), expected_identity=expected_identity
            )
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


__all__ = ["emit_boundary"]
