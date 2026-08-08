from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import stat
import threading
import time
import uuid
import weakref
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping, Sequence

from aptus.generation import validate_bundle_archive_file

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses the in-process lock.
    fcntl = None

from .contracts import (
    ContractError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    compact_canonical_json_bytes,
    new_opaque_id,
    sha256_bytes,
    utc_now,
    validate_event_ledger,
    validate_record,
    validate_safe_relative_path,
)
from .admission import (
    ACTIVATION_FILE_NAMES,
    ACTIVATION_SEAL_NAME,
    AdmissionError,
    RetainedActivatedSlot,
    validate_retained_activated_slot,
)
from .monitoring import (
    TelemetryValidationError,
    summarize_telemetry,
    validate_cooldown,
    validate_telemetry_sample,
)
from .qualification import (
    QUALIFYING_ACTION_ORDER,
    REQUIRED_QUALIFYING_ARTIFACT_ROLES,
    REQUIRED_QUALIFYING_AUTHORITY_ROLES,
    QualificationError,
    build_segment_summaries,
    terminal_timing_summary,
    validate_idle_baseline_binding,
    validate_qualifying_terminal_timing,
    validate_qualifying_telemetry_configuration,
)
from .phase4 import (
    Phase4SourceFreezeError,
    validate_retained_phase4_source_freeze,
)
from .outcomes import OutcomeProfileError, validate_managed_sequence_outcome
from .runtime_events import RuntimeBoundaryError, validate_runtime_boundary


RAW_MANIFEST_SCHEMA = "aptus.experiment-raw-manifest.v1"
RAW_SEAL_SCHEMA = "aptus.experiment-raw-seal.v1"
CAPTURE_FAILURE_SCHEMA = "aptus.experiment-capture-failure.v1"
EVIDENCE_RECEIPT_SCHEMA = "aptus.experiment-evidence-receipt.v1"
RAW_MANIFEST_NAME = "raw-manifest.json"
SEAL_NAME = "SEALED.json"
RETENTION_POLICY_ID = "cuda-v02-public-claim-evidence-24m-v1"
COPY_VERIFICATION_CADENCE_DAYS = 90
OFF_HOST_RETRIEVAL_CADENCE_DAYS = 180
RETENTION_MONTHS = 24

_HEX_ID = re.compile(r"^[a-z][a-z0-9-]*_[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_LOCK = threading.RLock()
_EXPERIMENT_SEAL_LOCK = threading.RLock()
_AUTO_PREVIOUS = object()
_EXPERIMENT_SEAL_LOCK_NAME = ".experiment-run-seals.lock"
_ATTEMPT_SLOT_ID = re.compile(r"^slot_[0-9a-f]{20}$")
_EXPERIMENT_RUN_ID = re.compile(r"^xrun_[0-9a-f]{32}$")
_EXECUTION_CONFIGURATION_ID = re.compile(r"^exec_[0-9a-f]{20}$")
_JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
_CAPTURE_KINDS = frozenset({"command", "managed-job", "managed-sequence"})
_QUALIFYING_SINGLE_ROLES = (
    frozenset(
        {
            "attempt-slot-record",
            "execution-configuration-record",
            "experiment-run-record",
            "telemetry-configuration",
            "telemetry-summary",
            "cooldown-summary",
            "idle-baseline-binding",
        }
    )
    | REQUIRED_QUALIFYING_ARTIFACT_ROLES
    | REQUIRED_QUALIFYING_AUTHORITY_ROLES
)
_QUALIFYING_RUNTIME_JOURNAL_PATHS = {
    "pilot": "actions/pilot/runtime-boundaries.jsonl",
    "train": "actions/train/runtime-boundaries.jsonl",
}
_QUALIFYING_RUNTIME_SEQUENCE = {
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
_RUN_EMBEDDED_DIGEST_ROLES = _QUALIFYING_SINGLE_ROLES - {"experiment-run-record"}
_ACTIVATION_ROLE_FILES = {
    "activation-admission-decision": "admission-decision.json",
    "activation-admission-observations": "admission-observations.json",
    "activation-execution-configuration": "execution-configuration.json",
    "activation-experiment-run-template": "experiment-run-template.json",
    "activation-started-identity-template": "started-identity-template.json",
    "activation-decision": "activation-decision.json",
    "activation-seal": ACTIVATION_SEAL_NAME,
}
if tuple(_ACTIVATION_ROLE_FILES.values()) != (
    *ACTIVATION_FILE_NAMES,
    ACTIVATION_SEAL_NAME,
):
    raise RuntimeError("retained activation role map differs from admission authority")
_COMMAND_CORE_ROLES = frozenset({"command-record", "command-output", "event-ledger"})
_MANAGED_CORE_ROLES = frozenset(
    {
        "job-log",
        "terminal-job-record",
        "last-observed-job-record",
        "action-submission-record",
        "sequence-summary",
        "event-ledger",
    }
)


class EvidenceStorageError(ValueError):
    """The protected evidence store violated its filesystem or data contract."""


class ArtifactIntegrityError(EvidenceStorageError):
    """A sealed artifact no longer matches its immutable manifest and seal."""


class ReceiptChainError(EvidenceStorageError):
    """An append-only receipt chain is malformed, forked, or discontinuous."""


class RetrievalError(ArtifactIntegrityError):
    """A full restore did not reproduce a valid sealed artifact."""

    def __init__(self, message: str, *, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


class _ExperimentSealEpochError(EvidenceStorageError):
    """The pinned experiment-seal lock epoch changed before commit."""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _require_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX_ID.fullmatch(value):
        raise EvidenceStorageError(f"{label} is invalid.")
    return value


def _require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise EvidenceStorageError(f"{label} is not a lowercase SHA-256 digest.")
    return value


def _parse_utc(value: str | datetime, *, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise EvidenceStorageError(
                f"{label} is not an RFC 3339 timestamp."
            ) from error
    else:
        raise EvidenceStorageError(f"{label} is not an RFC 3339 timestamp.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceStorageError(f"{label} must include a UTC offset.")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def add_calendar_months_utc(
    value: str | datetime, months: int = RETENTION_MONTHS
) -> str:
    """Add calendar months in UTC, clamping only an invalid target-month day."""

    if not isinstance(months, int) or isinstance(months, bool) or months < 1:
        raise ValueError("months must be a positive integer.")
    current = _parse_utc(value, label="retention basis")
    month_index = current.year * 12 + current.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return _format_utc(current.replace(year=year, month=month, day=day))


def retention_deadline_utc(value: str | datetime) -> str:
    return add_calendar_months_utc(value, RETENTION_MONTHS)


def _require_owner(metadata: os.stat_result, path: Path) -> None:
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise PermissionError(f"Protected evidence is owned by another user: {path}")


def _require_mode(metadata: os.stat_result, expected: int, path: Path) -> None:
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != expected:
        raise PermissionError(f"Protected evidence mode must be {expected:04o}: {path}")


def _require_private_directory(path: Path) -> Path:
    if path.is_symlink():
        raise PermissionError(
            f"Protected evidence directories cannot be symlinks: {path}"
        )
    try:
        metadata = path.stat()
    except OSError as error:
        raise PermissionError(
            f"Protected evidence directory is unavailable: {path}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError(f"Protected evidence path is not a directory: {path}")
    _require_owner(metadata, path)
    _require_mode(metadata, 0o700, path)
    return path


def ensure_private_directory(path: Path) -> Path:
    """Create or validate a custodian-owned 0700 directory without accepting a symlink."""

    target = path.expanduser()
    if target.exists() or target.is_symlink():
        return _require_private_directory(target).resolve()
    parent = target.parent.resolve(strict=True)
    _require_private_directory(parent)
    resolved = parent / target.name
    os.mkdir(resolved, 0o700)
    resolved.chmod(0o700)
    _fsync_directory(parent)
    return _require_private_directory(resolved).resolve()


def _create_fresh_private_directory(path: Path) -> Path:
    target = path.expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"Protected evidence destination already exists: {target}"
        )
    parent = target.parent.resolve(strict=True)
    _require_private_directory(parent)
    resolved = parent / target.name
    os.mkdir(resolved, 0o700)
    resolved.chmod(0o700)
    _fsync_directory(parent)
    return _require_private_directory(resolved).resolve()


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":  # pragma: no cover - Windows has no O_DIRECTORY.
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_metadata(path: Path, *, mode: int = 0o600) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ArtifactIntegrityError(
            f"Required evidence file is unavailable: {path}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ArtifactIntegrityError(
            f"Evidence files must be regular, not symlinks: {path}"
        )
    if metadata.st_nlink != 1:
        raise ArtifactIntegrityError(f"Evidence files cannot be hardlinked: {path}")
    _require_owner(metadata, path)
    _require_mode(metadata, mode, path)
    return metadata


def _evidence_file_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the identity and mutation-sensitive metadata for one open file."""

    return (
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


def _validate_open_evidence_file(
    metadata: os.stat_result, path: Path, *, mode: int
) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ArtifactIntegrityError(
            f"Evidence files must be regular and non-hardlinked: {path}"
        )
    _require_owner(metadata, path)
    _require_mode(metadata, mode, path)


def _private_directory_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    """Return identity and mutation-sensitive metadata for one directory."""

    return (
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


def _validate_open_private_directory(metadata: os.stat_result, path: Path) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactIntegrityError(
            f"Protected evidence path is not a directory: {path}"
        )
    _require_owner(metadata, path)
    _require_mode(metadata, 0o700, path)


def _private_directory_lstat(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ArtifactIntegrityError(
            f"Protected evidence directory is unavailable: {path}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ArtifactIntegrityError(
            f"Protected evidence directories cannot be symlinks: {path}"
        )
    _validate_open_private_directory(metadata, path)
    return metadata


def _recheck_pinned_private_directory(
    path: Path, descriptor: int, expected_fingerprint: tuple[int, ...]
) -> None:
    opened = os.fstat(descriptor)
    _validate_open_private_directory(opened, path)
    if _private_directory_fingerprint(opened) != expected_fingerprint:
        raise ArtifactIntegrityError(
            f"Protected evidence directory changed while it was verified: {path}"
        )
    path_after = _private_directory_lstat(path)
    if _private_directory_fingerprint(path_after) != expected_fingerprint:
        raise ArtifactIntegrityError(
            f"Protected evidence directory path changed while it was verified: {path}"
        )


@contextmanager
def _pinned_private_directory(
    path: Path,
) -> Iterator[tuple[int, tuple[int, ...]]]:
    """Pin a private artifact root and recheck its exact inode on success."""

    before = _private_directory_lstat(path)
    before_fingerprint = _private_directory_fingerprint(before)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactIntegrityError(
            f"Protected evidence directory could not be opened safely: {path}"
        ) from error
    try:
        _recheck_pinned_private_directory(path, descriptor, before_fingerprint)
        yield descriptor, before_fingerprint
        _recheck_pinned_private_directory(path, descriptor, before_fingerprint)
    finally:
        os.close(descriptor)


@contextmanager
def _pinned_evidence_descriptor(path: Path, *, mode: int = 0o600) -> Iterator[int]:
    """Open one evidence file without following links and pin its identity.

    The pathname is checked before open, the descriptor is checked before and
    after use, and the pathname is checked again before success.  A same-user
    rename or replacement therefore cannot redirect a read between validation
    and consumption.
    """

    before = _regular_metadata(path, mode=mode)
    before_fingerprint = _evidence_file_fingerprint(before)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactIntegrityError(
            f"Required evidence file could not be opened safely: {path}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        _validate_open_evidence_file(opened, path, mode=mode)
        if _evidence_file_fingerprint(opened) != before_fingerprint:
            raise ArtifactIntegrityError(
                f"Evidence file identity changed while it was opened: {path}"
            )
        yield descriptor
        finished = os.fstat(descriptor)
        _validate_open_evidence_file(finished, path, mode=mode)
        if _evidence_file_fingerprint(finished) != before_fingerprint:
            raise ArtifactIntegrityError(
                f"Evidence file changed while it was read: {path}"
            )
        after = _regular_metadata(path, mode=mode)
        if _evidence_file_fingerprint(after) != before_fingerprint:
            raise ArtifactIntegrityError(
                f"Evidence file path changed while it was read: {path}"
            )
    finally:
        os.close(descriptor)


def _read_pinned_evidence_bytes(path: Path, *, mode: int = 0o600) -> bytes:
    payload = bytearray()
    with _pinned_evidence_descriptor(path, mode=mode) as descriptor:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
    return bytes(payload)


def _hash_pinned_evidence_file(path: Path, *, mode: int = 0o600) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with _pinned_evidence_descriptor(path, mode=mode) as descriptor:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            digest.update(block)
    return total, digest.hexdigest()


def _source_metadata(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise EvidenceStorageError(f"Capture source is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EvidenceStorageError(
            f"Capture source must be a regular non-symlink: {path}"
        )
    if metadata.st_nlink != 1:
        raise EvidenceStorageError(f"Capture source cannot be hardlinked: {path}")
    return metadata


def _source_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _destination(root: Path, relative_path: str) -> tuple[str, Path]:
    normalized = validate_safe_relative_path(relative_path)
    if normalized in {RAW_MANIFEST_NAME, SEAL_NAME}:
        raise EvidenceStorageError(f"Payload path is reserved: {normalized}")
    parts = PurePosixPath(normalized).parts
    return normalized, root.joinpath(*parts)


def _ensure_payload_parent(root: Path, relative_path: str) -> Path:
    parts = PurePosixPath(relative_path).parts[:-1]
    cursor = root
    for part in parts:
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            _require_private_directory(cursor)
            continue
        os.mkdir(cursor, 0o700)
        cursor.chmod(0o700)
        _fsync_directory(cursor.parent)
        _require_private_directory(cursor)
    return cursor


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - defensive OS failure path.
            raise OSError("Evidence write made no progress.")
        view = view[written:]


def _exclusive_descriptor(path: Path, mode: int = 0o600) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    os.fchmod(descriptor, mode)
    return descriptor


def _write_exclusive_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    descriptor: int | None = None
    created = False
    try:
        descriptor = _exclusive_descriptor(path, mode)
        created = True
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _regular_metadata(path, mode=mode)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            path.unlink(missing_ok=True)
            _fsync_directory(path.parent)
        raise


def _write_exclusive_control_bytes(
    path: Path, payload: bytes, *, mode: int = 0o600
) -> tuple[int, int]:
    """Write one seal control file and return its exact created inode."""

    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    try:
        descriptor = _exclusive_descriptor(path, mode)
        opened = os.fstat(descriptor)
        created_identity = (opened.st_dev, opened.st_ino)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        finished = os.fstat(descriptor)
        if (finished.st_dev, finished.st_ino) != created_identity:
            raise ArtifactIntegrityError(
                f"Created seal control-file identity changed: {path}"
            )
        os.close(descriptor)
        descriptor = None
        observed = _regular_metadata(path, mode=mode)
        if (observed.st_dev, observed.st_ino) != created_identity:
            raise ArtifactIntegrityError(
                f"Created seal control-file path was replaced: {path}"
            )
        _fsync_directory(path.parent)
        return created_identity
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created_identity is not None:
            _unlink_created_control_file(path, created_identity, missing_ok=True)
        raise


def _unlink_created_control_file(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    missing_ok: bool = False,
) -> None:
    """Remove only the exact control-file inode created by this sealer."""

    try:
        observed = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return
        raise ArtifactIntegrityError(
            f"Created seal control file disappeared before rollback: {path}"
        ) from None
    except OSError as error:
        raise ArtifactIntegrityError(
            f"Created seal control file is unavailable for rollback: {path}"
        ) from error
    if (
        stat.S_ISLNK(observed.st_mode)
        or (
            observed.st_dev,
            observed.st_ino,
        )
        != expected_identity
    ):
        raise ArtifactIntegrityError(
            f"Refusing to remove a replaced seal control file: {path}"
        )
    path.unlink()
    _fsync_directory(path.parent)


def _copy_file(source: Path, destination: Path) -> tuple[int, str]:
    source_metadata = _source_metadata(source)
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source, source_flags)
    destination_descriptor: int | None = None
    destination_created = False
    digest = hashlib.sha256()
    total = 0
    try:
        opened_metadata = os.fstat(source_descriptor)
        if _source_fingerprint(opened_metadata) != _source_fingerprint(source_metadata):
            raise EvidenceStorageError("Capture source changed while it was opened.")
        destination_descriptor = _exclusive_descriptor(destination)
        destination_created = True
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
            _write_all(destination_descriptor, block)
        finished_metadata = os.fstat(source_descriptor)
        if _source_fingerprint(finished_metadata) != _source_fingerprint(
            source_metadata
        ):
            raise EvidenceStorageError("Capture source changed while it was copied.")
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = None
        _regular_metadata(destination)
        _fsync_directory(destination.parent)
        return total, digest.hexdigest()
    except BaseException:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if destination_created:
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
        raise
    finally:
        os.close(source_descriptor)


def _copy_pinned_descriptor(
    source_descriptor: int,
    source_path: Path,
    destination: Path,
    expected_fingerprint: tuple[int, ...],
) -> tuple[int, str]:
    """Copy from the exact already-open source, rejecting any path replacement."""

    if isinstance(source_descriptor, bool) or not isinstance(source_descriptor, int):
        raise EvidenceStorageError("Pinned capture descriptor is invalid.")
    before = os.fstat(source_descriptor)
    if _evidence_file_fingerprint(before) != expected_fingerprint:
        raise EvidenceStorageError("Pinned capture descriptor identity changed.")
    _validate_open_evidence_file(before, source_path, mode=0o600)
    try:
        path_before = source_path.lstat()
    except OSError as error:
        raise EvidenceStorageError("Pinned capture path is unavailable.") from error
    if _evidence_file_fingerprint(path_before) != expected_fingerprint:
        raise EvidenceStorageError("Pinned capture path was replaced before copy.")

    destination_descriptor: int | None = None
    destination_created = False
    digest = hashlib.sha256()
    total = 0
    try:
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        destination_descriptor = _exclusive_descriptor(destination)
        destination_created = True
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
            _write_all(destination_descriptor, block)
        after = os.fstat(source_descriptor)
        try:
            path_after = source_path.lstat()
        except OSError as error:
            raise EvidenceStorageError(
                "Pinned capture path disappeared during copy."
            ) from error
        if (
            _evidence_file_fingerprint(after) != expected_fingerprint
            or _evidence_file_fingerprint(path_after) != expected_fingerprint
        ):
            raise EvidenceStorageError("Pinned capture source changed during copy.")
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = None
        _regular_metadata(destination)
        _fsync_directory(destination.parent)
        return total, digest.hexdigest()
    except BaseException:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if destination_created:
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
        raise


def _fsync_regular_file(path: Path) -> None:
    with _pinned_evidence_descriptor(path) as descriptor:
        os.fsync(descriptor)


def _walk_artifact(root: Path) -> tuple[list[Path], list[Path]]:
    _require_private_directory(root)
    directories: list[Path] = [root]
    files: list[Path] = []
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        _require_private_directory(current)
        for name in sorted(directory_names):
            directory = current / name
            if directory.is_symlink():
                raise ArtifactIntegrityError(
                    f"Sealed artifacts cannot contain directory symlinks: {directory}"
                )
            _require_private_directory(directory)
            directories.append(directory)
        for name in sorted(file_names):
            path = current / name
            _regular_metadata(path)
            files.append(path)
    return sorted(set(directories)), sorted(files)


def _assert_exact_artifact_inventory(
    root: Path, expected_files: Mapping[str, tuple[int, str]]
) -> None:
    """Rewalk and rehash the complete file/directory inventory fail closed."""

    directories, files = _walk_artifact(root)
    actual_file_names = {path.relative_to(root).as_posix() for path in files}
    if actual_file_names != set(expected_files):
        raise ArtifactIntegrityError(
            "Artifact file inventory changed during verification."
        )
    expected_directory_names = {"."}
    for name in expected_files:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            expected_directory_names.add(parent.as_posix())
            parent = parent.parent
    actual_directory_names = {
        "." if path == root else path.relative_to(root).as_posix()
        for path in directories
    }
    if actual_directory_names != expected_directory_names:
        raise ArtifactIntegrityError(
            "Artifact directory inventory changed during verification."
        )
    for name, expected in expected_files.items():
        observed = _hash_pinned_evidence_file(root.joinpath(*PurePosixPath(name).parts))
        if observed != expected:
            raise ArtifactIntegrityError(
                f"Artifact payload fingerprint changed during verification: {name}"
            )


def _load_canonical_record(path: Path, schema: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_pinned_evidence_bytes(path)
    try:
        value = json.loads(payload)
    except (RecursionError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"Evidence record is invalid JSON: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"Evidence record must be an object: {path.name}")
    try:
        validated = validate_record(value, expected_schema=schema)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"Evidence record is invalid: {path.name}"
        ) from error
    if payload != canonical_json_bytes(validated):
        raise ArtifactIntegrityError(f"Evidence record is not canonical: {path.name}")
    return validated, payload


def _normalize_required_role_bindings(
    bindings: Mapping[str, str | Sequence[str]], entries: Sequence[Mapping[str, Any]]
) -> dict[str, str | list[str]]:
    by_id = {entry.get("entry_id"): entry for entry in entries}
    normalized: dict[str, str | list[str]] = {}
    for role, raw_ids in sorted(bindings.items()):
        if not isinstance(role, str) or not role:
            raise EvidenceStorageError(
                "Required role binding names must be non-empty strings."
            )
        if isinstance(raw_ids, str):
            identifiers = [raw_ids]
        elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, bytes):
            identifiers = list(raw_ids)
        else:
            raise EvidenceStorageError(f"Required role binding {role!r} is invalid.")
        if not identifiers or any(not isinstance(item, str) for item in identifiers):
            raise EvidenceStorageError(f"Required role binding {role!r} is invalid.")
        if len(set(identifiers)) != len(identifiers):
            raise EvidenceStorageError(f"Required role binding {role!r} is duplicated.")
        for entry_id in identifiers:
            entry = by_id.get(entry_id)
            if entry is None or entry.get("role") != role:
                raise EvidenceStorageError(
                    f"Required role binding {role!r} does not identify a matching entry."
                )
        normalized[role] = identifiers[0] if isinstance(raw_ids, str) else identifiers
    return normalized


def _bound_entry_ids(
    bindings: Mapping[str, str | Sequence[str]], role: str
) -> list[str]:
    raw = bindings.get(role)
    if raw is None:
        return []
    return [raw] if isinstance(raw, str) else list(raw)


def _require_complete_role_binding(
    role: str,
    *,
    bindings: Mapping[str, str | Sequence[str]],
    entries: Sequence[Mapping[str, Any]],
    minimum: int,
    maximum: int | None = None,
) -> list[str]:
    actual = [str(entry["entry_id"]) for entry in entries if entry.get("role") == role]
    bound = _bound_entry_ids(bindings, role)
    if len(actual) < minimum or (maximum is not None and len(actual) > maximum):
        if maximum == minimum:
            qualifier = f"exactly {minimum}"
        elif maximum is not None:
            qualifier = f"between {minimum} and {maximum}"
        else:
            qualifier = f"at least {minimum}"
        raise EvidenceStorageError(
            f"Experiment-run capture requires {qualifier} {role!r} entries."
        )
    if len(actual) != len(set(actual)) or set(bound) != set(actual):
        raise EvidenceStorageError(
            f"Experiment-run role {role!r} must bind every matching entry exactly once."
        )
    return bound


def _require_absent_roles(
    roles: Iterable[str], entries: Sequence[Mapping[str, Any]], *, profile: str
) -> None:
    forbidden = set(roles)
    present = sorted(
        {str(entry.get("role")) for entry in entries if entry.get("role") in forbidden}
    )
    if present:
        raise EvidenceStorageError(
            f"Experiment-run {profile} capture contains contradictory roles: "
            + ", ".join(present)
        )


def _validate_capture_failure_payload(
    root: Path,
    *,
    protected_artifact_id: str,
    identity_bindings: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    required_role_bindings: Mapping[str, str | Sequence[str]],
) -> None:
    identifiers = _require_complete_role_binding(
        "capture-failure",
        bindings=required_role_bindings,
        entries=entries,
        minimum=1,
        maximum=1,
    )
    entry = next(item for item in entries if item.get("entry_id") == identifiers[0])
    if (
        entry.get("relative_path") != "capture-failure.json"
        or entry.get("media_type") != "application/json"
    ):
        raise EvidenceStorageError(
            "Capture-failure evidence must be canonical JSON at capture-failure.json."
        )
    receipt, _ = _load_canonical_record(
        root / "capture-failure.json", CAPTURE_FAILURE_SCHEMA
    )
    if (
        receipt.get("protected_artifact_id") != protected_artifact_id
        or receipt.get("attempt_slot_id") != identity_bindings.get("attempt_slot_id")
        or receipt.get("experiment_run_id")
        != identity_bindings.get("experiment_run_id")
        or receipt.get("reason_code") == "NONE"
    ):
        raise EvidenceStorageError(
            "Capture-failure evidence does not bind the exact failed experiment run."
        )


def _entry_path(root: Path, entry: Mapping[str, Any]) -> Path:
    return root.joinpath(*PurePosixPath(str(entry["relative_path"])).parts)


def _load_canonical_json_entry(
    root: Path,
    entry: Mapping[str, Any],
    *,
    require_json_media_type: bool = False,
) -> dict[str, Any]:
    if require_json_media_type and entry.get("media_type") != "application/json":
        raise EvidenceStorageError("A protocol JSON role has the wrong media type.")
    path = _entry_path(root, entry)
    _regular_metadata(path)
    try:
        payload = _read_pinned_evidence_bytes(path)
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceStorageError(
            "A core evidence JSON payload is invalid."
        ) from error
    if type(value) is not dict or payload != canonical_json_bytes(value):
        raise EvidenceStorageError("A core evidence JSON payload is not canonical.")
    return value


def _load_canonical_jsonl_entry(
    root: Path,
    entry: Mapping[str, Any],
    *,
    require_jsonl_media_type: bool = False,
) -> list[dict[str, Any]]:
    if require_jsonl_media_type and entry.get("media_type") != "application/x-ndjson":
        raise EvidenceStorageError("A protocol JSONL role has the wrong media type.")
    path = _entry_path(root, entry)
    _regular_metadata(path)
    try:
        payload = _read_pinned_evidence_bytes(path)
        records = [json.loads(line) for line in payload.splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceStorageError(
            "A core evidence JSONL payload is invalid."
        ) from error
    if any(type(record) is not dict for record in records):
        raise EvidenceStorageError("A core evidence JSONL row must be an object.")
    if payload != canonical_jsonl_bytes(records):
        raise EvidenceStorageError("A core evidence JSONL payload is not canonical.")
    return records


def _entry_with_id(
    entries: Sequence[Mapping[str, Any]], entry_id: str
) -> Mapping[str, Any]:
    try:
        return next(entry for entry in entries if entry.get("entry_id") == entry_id)
    except StopIteration as error:  # pragma: no cover - normalized bindings guard this.
        raise EvidenceStorageError("A required evidence entry is absent.") from error


def _parse_available_core_payloads(
    root: Path, entries: Sequence[Mapping[str, Any]]
) -> None:
    """Apply syntax and canonical-byte checks even to nonqualifying captures."""

    json_roles = {
        "action-submission-record",
        "attempt-slot-record",
        "command-record",
        "cooldown-summary",
        "execution-configuration-record",
        "experiment-run-record",
        "idle-baseline-binding",
        "last-observed-job-record",
        "sequence-summary",
        "telemetry-configuration",
        "telemetry-summary",
        "terminal-job-record",
        "campaign-record",
        "comparison-cohort-record",
        "comparison-cell-record",
        "phase4-source-freeze",
        "phase4-source-freeze-seal",
    } | (REQUIRED_QUALIFYING_ARTIFACT_ROLES - {"bundle-archive"})
    for entry in entries:
        role = entry.get("role")
        if role in json_roles:
            _load_canonical_json_entry(root, entry)
        elif role in {
            "event-ledger",
            "telemetry",
            "runtime-boundary-journal",
            "phase4-idle-baseline-samples",
        }:
            records = _load_canonical_jsonl_entry(root, entry)
            if role == "runtime-boundary-journal":
                continue
            for record in records:
                if "schema_version" not in record:
                    continue
                try:
                    validate_record(record)
                except (ContractError, TypeError, ValueError) as error:
                    raise EvidenceStorageError(
                        "A typed core evidence row violates its declared contract."
                    ) from error


def _require_exact_object_fields(
    value: object, expected: set[str], *, label: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise EvidenceStorageError(f"{label} fields are not exact.")
    return value


def _validate_protocol_sequence_summary(
    summary: Mapping[str, Any],
    *,
    attempt_slot_id: str,
    experiment_run_id: str,
) -> list[dict[str, Any]]:
    required = {
        "record_kind",
        "experiment_run_id",
        "attempt_slot_id",
        "configured_actions",
        "started_actions",
        "native_outcome",
        "reason_code",
        "evidence_status",
        "capture_reason_code",
        "telemetry_required",
        "telemetry_test_override",
        "telemetry_healthy",
        "stopped_early",
        "five_action_duration_ns",
    }
    value = _require_exact_object_fields(summary, required, label="Sequence summary")
    if (
        value["record_kind"] != "aptus-cuda-campaign-managed-sequence-v1"
        or value["experiment_run_id"] != experiment_run_id
        or value["attempt_slot_id"] != attempt_slot_id
        or value["evidence_status"] != "protocol-valid"
        or value["capture_reason_code"] != "NONE"
        or value["telemetry_required"] is not True
        or value["telemetry_test_override"] is not False
        or value["telemetry_healthy"] is not True
        or (
            value["five_action_duration_ns"] is not None
            and (
                isinstance(value["five_action_duration_ns"], bool)
                or not isinstance(value["five_action_duration_ns"], int)
                or value["five_action_duration_ns"] < 0
            )
        )
    ):
        raise EvidenceStorageError(
            "Sequence summary does not bind a qualifying experiment run."
        )
    configured = value["configured_actions"]
    started = value["started_actions"]
    if (
        type(configured) is not list
        or not configured
        or type(started) is not list
        or not started
        or len(started) > len(configured)
    ):
        raise EvidenceStorageError("Sequence action inventories are invalid.")
    configured_rows: list[dict[str, Any]] = []
    labels: set[str] = set()
    for raw in configured:
        row = _require_exact_object_fields(
            raw,
            {"label", "action", "supervision_timeout_seconds", "submit_kwargs"},
            label="Configured action",
        )
        label = row["label"]
        action = row["action"]
        timeout = row["supervision_timeout_seconds"]
        if (
            not isinstance(label, str)
            or not label
            or "/" in label
            or label in labels
            or not isinstance(action, str)
            or not action
            or isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            or type(row["submit_kwargs"]) is not dict
        ):
            raise EvidenceStorageError("Configured action identity is invalid.")
        try:
            validate_safe_relative_path(label)
        except (ContractError, ValueError) as error:
            raise EvidenceStorageError("Configured action label is unsafe.") from error
        labels.add(label)
        configured_rows.append(dict(row))
    started_rows: list[dict[str, Any]] = []
    job_ids: set[str] = set()
    for index, raw in enumerate(started):
        row = _require_exact_object_fields(
            raw,
            {
                "label",
                "action",
                "job_id",
                "native_outcome",
                "reason_code",
                "terminal",
                "capture_reason_code",
            },
            label="Started action",
        )
        configured_row = configured_rows[index]
        job_id = row["job_id"]
        if (
            row["label"] != configured_row["label"]
            or row["action"] != configured_row["action"]
            or type(row["terminal"]) is not bool
            or row["capture_reason_code"] != "NONE"
            or not isinstance(row["native_outcome"], str)
            or not isinstance(row["reason_code"], str)
            or (
                job_id is not None
                and (
                    not isinstance(job_id, str)
                    or _JOB_ID.fullmatch(job_id) is None
                    or job_id in job_ids
                )
            )
        ):
            raise EvidenceStorageError("Started action identity is invalid.")
        if job_id is None and row["native_outcome"] not in {
            "refused",
            "guard-blocked",
            "unknown",
        }:
            raise EvidenceStorageError(
                "A jobless action is not an exact pre-submit disposition."
            )
        if job_id is not None and row["native_outcome"] in {
            "refused",
            "guard-blocked",
        }:
            raise EvidenceStorageError(
                "A pre-submit disposition spuriously contains a job identity."
            )
        if row["terminal"] is not (row["native_outcome"] != "unknown"):
            raise EvidenceStorageError(
                "Started action terminal state contradicts its native outcome."
            )
        if job_id is not None:
            job_ids.add(job_id)
        started_rows.append(dict(row))
    expected_stopped_early = len(started_rows) < len(configured_rows) or any(
        row["native_outcome"] != "passed" for row in started_rows
    )
    if value["stopped_early"] is not expected_stopped_early:
        raise EvidenceStorageError("Sequence stopped_early is inconsistent.")
    final = started_rows[-1]
    if (
        value["native_outcome"] != final["native_outcome"]
        or value["reason_code"] != final["reason_code"]
    ):
        raise EvidenceStorageError("Sequence terminal outcome is inconsistent.")
    return started_rows


def _validate_protocol_event_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment_run_id: str,
    started_actions: Sequence[Mapping[str, Any]] | None,
    require_telemetry: bool,
) -> tuple[list[dict[str, Any]], int | None, int | None]:
    try:
        ledger = validate_event_ledger(records)
    except (ContractError, TypeError, ValueError) as error:
        raise EvidenceStorageError("The event ledger contract is invalid.") from error
    if any(row["experiment_run_id"] != experiment_run_id for row in ledger):
        raise EvidenceStorageError("The event ledger binds another experiment run.")
    starts = [row for row in ledger if row["event_type"] == "telemetry.started"]
    stops = [row for row in ledger if row["event_type"] == "telemetry.stopped"]
    failures = [row for row in ledger if row["event_type"] == "telemetry.failed"]
    if (
        failures
        or (require_telemetry and (len(starts) != 1 or len(stops) != 1))
        or (not require_telemetry and (starts or stops))
    ):
        raise EvidenceStorageError(
            "Protocol-valid evidence requires one successful telemetry window."
        )
    start_ns: int | None = None
    stop_ns: int | None = None
    if require_telemetry:
        for row in (starts[0], stops[0]):
            if (
                row["subject_kind"] != "experiment-run"
                or row["subject_id"] != experiment_run_id
            ):
                raise EvidenceStorageError("Telemetry events bind another subject.")
        start_ns = starts[0]["monotonic_ns"]
        stop_ns = stops[0]["monotonic_ns"]
        if stop_ns < start_ns:
            raise EvidenceStorageError("The telemetry window moved backward.")

    if started_actions is not None:
        expected_actions = {
            (row["label"], row["action"]): row for row in started_actions
        }
        for event_type in ("command.started", "command.finished"):
            observed: dict[tuple[str, str], int] = {}
            for event in ledger:
                if event["event_type"] != event_type:
                    continue
                key = (event["phase"], event["action"])
                if (
                    key not in expected_actions
                    or event["subject_kind"] != "managed-action"
                    or event["subject_id"] != event["phase"]
                ):
                    raise EvidenceStorageError(
                        "An action ledger boundary has a spoofed identity."
                    )
                observed[key] = observed.get(key, 0) + 1
            if set(observed) != set(expected_actions) or any(
                count != 1 for count in observed.values()
            ):
                raise EvidenceStorageError(
                    "The event ledger does not bind every started action exactly once."
                )

        expected_jobs = {
            row["job_id"]: (row["label"], row["action"], row["native_outcome"])
            for row in started_actions
            if row["job_id"] is not None
        }
        runtime_event_phases = {
            "pilot.phase-started": ("pilot", {"pilot-phase-1", "pilot-phase-2"}),
            "pilot.phase-finished": ("pilot", {"pilot-phase-1", "pilot-phase-2"}),
            "training.started": ("train", {"training"}),
            "training.finished": ("train", {"training"}),
            "export.started": ("train", {"final-export"}),
            "export.finished": ("train", {"final-export"}),
            "verification.started": ("train", {"parent-verification"}),
            "verification.finished": ("train", {"parent-verification"}),
        }
        action_job_event_types = {
            "safety.triggered",
            "cancellation.requested",
            "process-group.terminated",
            "lease.reconciled",
        }
        observed_jobs: dict[str, list[dict[str, Any]]] = {}
        for event in ledger:
            if event["subject_kind"] != "aptus-job":
                continue
            job_id = event["subject_id"]
            expected = expected_jobs.get(job_id)
            if expected is None or event["action"] != expected[1]:
                raise EvidenceStorageError(
                    "An event-ledger job subject is not a started action."
                )
            if event["event_type"] == "job.state-observed":
                if event["phase"] != expected[0]:
                    raise EvidenceStorageError(
                        "A job-state event does not use its action label."
                    )
                observed_jobs.setdefault(job_id, []).append(event)
                continue
            if event["event_type"] in action_job_event_types:
                if event["phase"] != expected[0]:
                    raise EvidenceStorageError(
                        "A managed-job event does not use its action label."
                    )
                continue
            runtime_contract = runtime_event_phases.get(event["event_type"])
            if (
                runtime_contract is None
                or expected[1] != runtime_contract[0]
                or event["phase"] not in runtime_contract[1]
                or event["observation_kind"] != "emitted"
            ):
                raise EvidenceStorageError(
                    "A runtime event violates its frozen action and phase binding."
                )
        if set(observed_jobs) != set(expected_jobs):
            raise EvidenceStorageError(
                "The event ledger lacks terminal evidence for a started job."
            )
        for job_id, observations in observed_jobs.items():
            expected_outcome = expected_jobs[job_id][2]
            observed_outcome = observations[-1]["native_outcome"]
            allowed_outcomes = (
                {expected_outcome}
                if expected_outcome in {"passed", "failed"}
                else {None, "cancelled"}
                if expected_outcome in {"cancelled", "timed-out"}
                else {None}
            )
            if observed_outcome not in allowed_outcomes:
                raise EvidenceStorageError(
                    "The event ledger terminal job outcome is inconsistent."
                )
    return ledger, start_ns, stop_ns


def _validate_protocol_telemetry(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment_run_id: str,
    start_monotonic_ns: int,
    stop_monotonic_ns: int,
) -> list[dict[str, Any]]:
    if not records:
        raise EvidenceStorageError("Protocol-valid telemetry must not be empty.")
    samples: list[dict[str, Any]] = []
    try:
        for record in records:
            samples.append(validate_telemetry_sample(record))
    except (ContractError, TelemetryValidationError, TypeError, ValueError) as error:
        raise EvidenceStorageError("A telemetry sample is invalid.") from error
    expected_count = ((stop_monotonic_ns - start_monotonic_ns) // 1_000_000_000) + 1
    prior_slot = -1
    for sequence, sample in enumerate(samples):
        slot = sample["scheduled_slot"]
        scheduled = sample["scheduled_monotonic_ns"]
        if (
            sample["sequence"] != sequence
            or sample["experiment_run_id"] != experiment_run_id
            or slot <= prior_slot
            or slot >= expected_count
            or scheduled != start_monotonic_ns + slot * 1_000_000_000
            or sample["observed_monotonic_ns"] > stop_monotonic_ns
        ):
            raise EvidenceStorageError(
                "Telemetry sequence or scheduled-window binding is invalid."
            )
        if prior_slot >= 0 and slot - prior_slot > 2:
            raise EvidenceStorageError(
                "Protocol-valid telemetry exceeds the maximum qualifying gap."
            )
        if (
            sample["collector"]["healthy"] is not True
            or sample["watchdog"]["healthy"] is not True
            or sample["watchdog"]["ownership_certain"] is not True
        ):
            raise EvidenceStorageError(
                "Protocol-valid telemetry reports an unhealthy collector or watchdog."
            )
        prior_slot = slot
    if samples[0]["scheduled_slot"] != 0 or samples[-1]["scheduled_slot"] != (
        expected_count - 1
    ):
        raise EvidenceStorageError("Telemetry omits a scheduled window boundary.")
    if len(samples) / expected_count < 0.99:
        raise EvidenceStorageError("Telemetry coverage is below the protocol minimum.")
    return samples


def _validate_qualifying_event_sequence(
    ledger: Sequence[Mapping[str, Any]],
    *,
    experiment_run_id: str,
    started_actions: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Require the exact successful action, runtime, and cooldown boundaries."""

    if tuple(row["action"] for row in started_actions) != QUALIFYING_ACTION_ORDER:
        raise EvidenceStorageError("The qualifying action sequence is not frozen.")
    if any(
        row["job_id"] is None
        or row["native_outcome"] != "passed"
        or row["reason_code"] != "NONE"
        for row in started_actions
    ):
        raise EvidenceStorageError("A qualifying action is not an exact pass.")

    by_action = {str(row["action"]): row for row in started_actions}
    command_rows = [
        row
        for row in ledger
        if row["event_type"] in {"command.started", "command.finished"}
    ]
    expected_command_types = tuple(
        event_type
        for _started in started_actions
        for event_type in ("command.started", "command.finished")
    )
    if tuple(row["event_type"] for row in command_rows) != expected_command_types:
        raise EvidenceStorageError(
            "The qualifying command boundary sequence is incomplete or reordered."
        )
    for index, started in enumerate(started_actions):
        started_event = command_rows[index * 2]
        finished_event = command_rows[index * 2 + 1]
        expected_identity = (
            started["label"],
            started["action"],
            "managed-action",
            started["label"],
        )
        if (
            (
                started_event["phase"],
                started_event["action"],
                started_event["subject_kind"],
                started_event["subject_id"],
            )
            != expected_identity
            or started_event["native_outcome"] is not None
            or started_event["reason_code"] != "NONE"
            or started_event["exit_code"] is not None
            or (
                finished_event["phase"],
                finished_event["action"],
                finished_event["subject_kind"],
                finished_event["subject_id"],
            )
            != expected_identity
            or finished_event["native_outcome"] != "passed"
            or finished_event["reason_code"] != "NONE"
            or finished_event["exit_code"] != 0
        ):
            raise EvidenceStorageError(
                "A qualifying command boundary has a spoofed identity or result."
            )
        job_rows = [
            row
            for row in ledger
            if row["subject_kind"] == "aptus-job"
            and row["subject_id"] == started["job_id"]
        ]
        if not job_rows or any(
            not (
                started_event["sequence"] < row["sequence"] < finished_event["sequence"]
            )
            for row in job_rows
        ):
            raise EvidenceStorageError(
                "A qualifying job boundary is outside its action command."
            )

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
    pilot_job_id = by_action["pilot"]["job_id"]
    train_job_id = by_action["train"]["job_id"]
    expected_runtime = (
        ("pilot.phase-started", "pilot-phase-1", "pilot", pilot_job_id),
        ("pilot.phase-finished", "pilot-phase-1", "pilot", pilot_job_id),
        ("pilot.phase-started", "pilot-phase-2", "pilot", pilot_job_id),
        ("pilot.phase-finished", "pilot-phase-2", "pilot", pilot_job_id),
        ("training.started", "training", "train", train_job_id),
        ("export.started", "final-export", "train", train_job_id),
        ("export.finished", "final-export", "train", train_job_id),
        ("training.finished", "training", "train", train_job_id),
        ("verification.started", "parent-verification", "train", train_job_id),
        ("verification.finished", "parent-verification", "train", train_job_id),
    )
    runtime_rows = [row for row in ledger if row["event_type"] in runtime_types]
    if (
        tuple(
            (row["event_type"], row["phase"], row["action"], row["subject_id"])
            for row in runtime_rows
        )
        != expected_runtime
    ):
        raise EvidenceStorageError(
            "The runtime boundary sequence is incomplete, reordered, or misbound."
        )
    for row in runtime_rows:
        is_started = row["event_type"].endswith("started")
        if (
            row["subject_kind"] != "aptus-job"
            or row["observation_kind"] != "emitted"
            or row["exit_code"] is not None
            or (
                is_started
                and (row["native_outcome"] is not None or row["reason_code"] != "NONE")
            )
            or (
                not is_started
                and (row["native_outcome"] != "passed" or row["reason_code"] != "NONE")
            )
        ):
            raise EvidenceStorageError("A runtime boundary is not an emitted pass.")

    terminal_job_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in ledger:
        if row["event_type"] != "job.state-observed":
            continue
        if row["observation_kind"] != "observed":
            raise EvidenceStorageError("A job-state boundary is not observed evidence.")
        terminal_job_rows.setdefault(str(row["subject_id"]), []).append(row)
    for started in started_actions:
        observations = terminal_job_rows.get(str(started["job_id"]), [])
        if not observations:
            raise EvidenceStorageError(
                "A qualifying job lacks its observed terminal state."
            )
        final = observations[-1]
        if final["native_outcome"] != "passed" or final["reason_code"] != "NONE":
            raise EvidenceStorageError("A qualifying job-state boundary is not a pass.")

    cooldown_rows = [
        row
        for row in ledger
        if row["event_type"] in {"cooldown.started", "cooldown.finished"}
    ]
    if [row["event_type"] for row in cooldown_rows] != [
        "cooldown.started",
        "cooldown.finished",
    ]:
        raise EvidenceStorageError(
            "The qualifying cooldown boundary pair is incomplete."
        )
    for index, row in enumerate(cooldown_rows):
        if (
            row["phase"] != "cooldown"
            or row["action"] is not None
            or row["subject_kind"] != "experiment-run"
            or row["subject_id"] != experiment_run_id
            or row["exit_code"] is not None
            or row["reason_code"] != "NONE"
            or row["native_outcome"] != (None if index == 0 else "passed")
        ):
            raise EvidenceStorageError("A cooldown boundary is not an exact pass.")

    allowed = {
        "clock.mapping",
        "harness.started",
        "telemetry.started",
        "telemetry.stopped",
        "command.started",
        "command.finished",
        "job.state-observed",
        "cooldown.started",
        "cooldown.finished",
        "harness.finished",
        "seal.started",
    } | runtime_types
    if any(row["event_type"] not in allowed for row in ledger):
        raise EvidenceStorageError(
            "A passing qualifying ledger contains a failure or safety event."
        )
    telemetry_start = next(
        row for row in ledger if row["event_type"] == "telemetry.started"
    )
    telemetry_stop = next(
        row for row in ledger if row["event_type"] == "telemetry.stopped"
    )
    if not (
        telemetry_start["sequence"]
        < command_rows[0]["sequence"]
        < command_rows[-1]["sequence"]
        < cooldown_rows[0]["sequence"]
        < cooldown_rows[1]["sequence"]
        < telemetry_stop["sequence"]
    ):
        raise EvidenceStorageError("The qualifying lifecycle boundaries are reordered.")
    for event_type in ("harness.finished", "seal.started"):
        row = next(item for item in ledger if item["event_type"] == event_type)
        if (
            row["native_outcome"] != "passed"
            or row["reason_code"] != "NONE"
            or row["subject_kind"] != "experiment-run"
            or row["subject_id"] != experiment_run_id
        ):
            raise EvidenceStorageError(
                "The qualifying harness disposition is not a pass."
            )
    return cooldown_rows[0]["monotonic_ns"], cooldown_rows[1]["monotonic_ns"]


def _cooldown_summary(
    samples: Sequence[Mapping[str, Any]],
    *,
    start_monotonic_ns: int,
    stop_monotonic_ns: int,
    idle_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    cooldown = [
        sample
        for sample in samples
        if start_monotonic_ns <= sample["observed_monotonic_ns"] <= stop_monotonic_ns
    ]
    if len(cooldown) != 120:
        raise EvidenceStorageError("Cooldown does not contain exactly 120 samples.")
    if (
        cooldown[0]["scheduled_monotonic_ns"] != start_monotonic_ns
        or cooldown[-1]["scheduled_monotonic_ns"] != stop_monotonic_ns
    ):
        raise EvidenceStorageError("Cooldown samples do not cover its exact boundary.")
    try:
        validation = validate_cooldown(cooldown, idle_baseline)
    except (TelemetryValidationError, TypeError, ValueError) as error:
        raise EvidenceStorageError(
            "Cooldown cannot be recomputed from its retained idle baseline."
        ) from error
    if not validation.valid:
        raise EvidenceStorageError(
            "Cooldown violates its retained baseline-relative thresholds."
        )
    return dict(validation.summary)


def _validate_protocol_managed_payloads(
    root: Path,
    *,
    capture_kind: str,
    entries: Sequence[Mapping[str, Any]],
    started_actions: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    terminal_entries = {
        str(entry["relative_path"]): entry
        for entry in entries
        if entry.get("role") == "terminal-job-record"
    }
    log_entries = {
        str(entry["relative_path"]): entry
        for entry in entries
        if entry.get("role") == "job-log"
    }
    submission_entries = {
        str(entry["relative_path"]): entry
        for entry in entries
        if entry.get("role") == "action-submission-record"
    }
    observed_entries = {
        str(entry["relative_path"]): entry
        for entry in entries
        if entry.get("role") == "last-observed-job-record"
    }
    expected_terminal_paths: set[str] = set()
    expected_observed_paths: set[str] = set()
    expected_log_paths: set[str] = set()
    expected_submission_paths: set[str] = set()
    job_ids: list[str] = []
    run_ids: list[str] = []
    terminal_jobs: list[dict[str, Any]] = []
    for started in started_actions:
        prefix = (
            "job" if capture_kind == "managed-job" else f"actions/{started['label']}"
        )
        if started["job_id"] is None:
            submission_path = f"{prefix}/submission.json"
            expected_submission_paths.add(submission_path)
            entry = submission_entries.get(submission_path)
            if entry is None or entry.get("media_type") != "application/json":
                raise EvidenceStorageError(
                    "An admission refusal lacks its exact submission record."
                )
            record = _load_canonical_json_entry(
                root, entry, require_json_media_type=True
            )
            common = {
                "record_kind",
                "action_label",
                "action",
                "native_outcome",
                "reason_code",
            }
            required = (
                common | {"exception_type"}
                if started["native_outcome"] == "refused"
                else common
            )
            allowed_record_kinds = {
                "refused": {"aptus-cuda-campaign-submission-refusal-v1"},
                "guard-blocked": {"aptus-cuda-campaign-pre-submit-guard-v1"},
                "unknown": {
                    "aptus-cuda-campaign-invalid-post-persist-failure-v1",
                    "aptus-cuda-campaign-ambiguous-submission-failure-v1",
                    "aptus-cuda-campaign-malformed-submission-v1",
                    "aptus-cuda-campaign-ambiguous-submission-identity-v1",
                },
            }
            if (
                set(record) != required
                or record["record_kind"]
                not in allowed_record_kinds[started["native_outcome"]]
                or record["action_label"] != started["label"]
                or record["action"] != started["action"]
                or record["native_outcome"] != started["native_outcome"]
                or record["reason_code"] != started["reason_code"]
                or (
                    "exception_type" in record
                    and (
                        not isinstance(record["exception_type"], str)
                        or not record["exception_type"]
                    )
                )
            ):
                raise EvidenceStorageError(
                    "A pre-submit disposition record is spoofed."
                )
            continue

        record_path = f"{prefix}/terminal.json"
        log_path = f"{prefix}/full.log"
        if started["terminal"]:
            expected_terminal_paths.add(record_path)
            record_entry = terminal_entries.get(record_path)
        else:
            expected_observed_paths.add(record_path)
            record_entry = observed_entries.get(record_path)
        expected_log_paths.add(log_path)
        log_entry = log_entries.get(log_path)
        if (
            record_entry is None
            or record_entry.get("media_type") != "application/json"
            or log_entry is None
            or log_entry.get("media_type") != "text/plain"
        ):
            raise EvidenceStorageError(
                "A started job lacks its exact terminal record and paired log."
            )
        record = _load_canonical_json_entry(
            root, record_entry, require_json_media_type=True
        )
        job_id = started["job_id"]
        state_outcomes = {
            "completed": "passed",
            "failed": "failed",
            "cancelled": "cancelled",
            "queued": "unknown",
            "running": "unknown",
            "cancelling": "unknown",
        }
        observed_native_outcome = state_outcomes.get(record.get("state"))
        expected_native_outcomes = (
            {"cancelled", "timed-out"}
            if observed_native_outcome == "cancelled"
            else {observed_native_outcome}
        )
        if (
            record.get("id") != job_id
            or record.get("job_id") != job_id
            or record.get("action") != started["action"]
            or started["native_outcome"] not in expected_native_outcomes
            or started["terminal"] != (record.get("state") in _TERMINAL_STATES)
        ):
            raise EvidenceStorageError("A retained job record identity is spoofed.")
        run_id = record.get("run_id")
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise EvidenceStorageError("A terminal Aptus run ID is invalid.")
        return_code = record.get("return_code")
        if isinstance(return_code, bool) or not isinstance(
            return_code, (int, type(None))
        ):
            raise EvidenceStorageError("A terminal Aptus return code is invalid.")
        if started["native_outcome"] == "passed" and return_code != 0:
            raise EvidenceStorageError("A passing terminal job lacks return code zero.")
        if started["native_outcome"] in {"cancelled", "timed-out"} and (
            record.get("cancel_reason_code") != started["reason_code"]
        ):
            raise EvidenceStorageError(
                "A cancelled terminal record has the wrong exact reason."
            )
        job_ids.append(job_id)
        if run_id is not None:
            run_ids.append(run_id)
        if started["terminal"]:
            terminal_jobs.append(
                {
                    "job_id": job_id,
                    "run_id": run_id,
                    "action": started["action"],
                    "state": record["state"],
                    "return_code": return_code,
                    "monotonic_clock_binding": record.get("monotonic_clock_binding"),
                    "queued_monotonic_ns": record.get("queued_monotonic_ns"),
                    "child_process_started_monotonic_ns": record.get(
                        "child_process_started_monotonic_ns"
                    ),
                    "child_process_finished_monotonic_ns": record.get(
                        "child_process_finished_monotonic_ns"
                    ),
                    "terminal_monotonic_ns": record.get("terminal_monotonic_ns"),
                }
            )
    if (
        set(terminal_entries) != expected_terminal_paths
        or set(observed_entries) != expected_observed_paths
        or set(log_entries) != expected_log_paths
        or set(submission_entries) != expected_submission_paths
    ):
        raise EvidenceStorageError(
            "Managed evidence contains an unpaired or mislabeled action payload."
        )
    if len(run_ids) != len(set(run_ids)):
        raise EvidenceStorageError("Aptus run IDs must be unique within one run.")
    return job_ids, run_ids, terminal_jobs


def _validate_qualifying_runtime_journals(
    root: Path,
    *,
    experiment_run_id: str,
    entries: Sequence[Mapping[str, Any]],
    required_role_bindings: Mapping[str, str | Sequence[str]],
    started_actions: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
) -> None:
    """Cross-check the two retained runtime journals against emitted ledger rows."""

    journal_ids = _require_complete_role_binding(
        "runtime-boundary-journal",
        bindings=required_role_bindings,
        entries=entries,
        minimum=2,
        maximum=2,
    )
    journal_entries = [_entry_with_id(entries, entry_id) for entry_id in journal_ids]
    by_path = {str(entry["relative_path"]): entry for entry in journal_entries}
    if set(by_path) != set(_QUALIFYING_RUNTIME_JOURNAL_PATHS.values()):
        raise EvidenceStorageError(
            "Qualifying runtime journals do not use the frozen pilot/train paths."
        )

    jobs_by_action = {
        str(row["action"]): row["job_id"]
        for row in started_actions
        if row["action"] in _QUALIFYING_RUNTIME_JOURNAL_PATHS
    }
    if set(jobs_by_action) != set(_QUALIFYING_RUNTIME_JOURNAL_PATHS) or any(
        not isinstance(job_id, str) for job_id in jobs_by_action.values()
    ):
        raise EvidenceStorageError(
            "Qualifying runtime journals lack exact pilot/train job identities."
        )

    boundaries = []
    for action in ("pilot", "train"):
        entry = by_path[_QUALIFYING_RUNTIME_JOURNAL_PATHS[action]]
        records = _load_canonical_jsonl_entry(
            root, entry, require_jsonl_media_type=True
        )
        expected_sequence = _QUALIFYING_RUNTIME_SEQUENCE[action]
        if len(records) != len(expected_sequence):
            raise EvidenceStorageError(
                "A qualifying runtime journal has an incomplete boundary sequence."
            )
        action_boundaries = []
        for record in records:
            try:
                boundary = validate_runtime_boundary(
                    record,
                    expected_run_id=experiment_run_id,
                    expected_job_id=str(jobs_by_action[action]),
                    expected_action=action,
                )
            except (RuntimeBoundaryError, TypeError, ValueError) as error:
                raise EvidenceStorageError(
                    "A retained runtime boundary violates its exact contract."
                ) from error
            action_boundaries.append(boundary)
        if (
            tuple(
                (boundary.event_type, boundary.phase) for boundary in action_boundaries
            )
            != expected_sequence
        ):
            raise EvidenceStorageError(
                "A qualifying runtime journal is reordered or has an extra boundary."
            )
        if any(
            later.monotonic_ns < earlier.monotonic_ns
            for earlier, later in zip(action_boundaries, action_boundaries[1:])
        ):
            raise EvidenceStorageError(
                "A qualifying runtime journal moves backward in monotonic time."
            )
        boundaries.extend(action_boundaries)

    runtime_types = {
        event_type
        for sequence in _QUALIFYING_RUNTIME_SEQUENCE.values()
        for event_type, _phase in sequence
    }
    ledger_rows = [row for row in ledger if row["event_type"] in runtime_types]
    if len(ledger_rows) != len(boundaries):  # pragma: no cover - sequence guard.
        raise EvidenceStorageError(
            "The emitted ledger and retained runtime journals differ in length."
        )
    for row, boundary in zip(ledger_rows, boundaries):
        expected_row = {
            "schema_version": "aptus.experiment-event.v1",
            "sequence": row["sequence"],
            "experiment_run_id": experiment_run_id,
            **boundary.ledger_fields(),
            "source_reported_at_utc": None,
            "exit_code": None,
        }
        if row != expected_row:
            raise EvidenceStorageError(
                "An emitted ledger boundary differs from its retained runtime journal."
            )


def _validate_qualifying_runtime_journal_prefix(
    root: Path,
    *,
    experiment_run_id: str,
    entries: Sequence[Mapping[str, Any]],
    required_role_bindings: Mapping[str, str | Sequence[str]],
    started_actions: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
) -> None:
    """Cross-check every applicable pilot/train journal through a non-pass stop."""

    jobs_by_action = {
        str(row["action"]): str(row["job_id"])
        for row in started_actions
        if row["action"] in _QUALIFYING_RUNTIME_JOURNAL_PATHS
        and row["job_id"] is not None
    }
    journal_ids = _require_complete_role_binding(
        "runtime-boundary-journal",
        bindings=required_role_bindings,
        entries=entries,
        minimum=len(jobs_by_action),
        maximum=len(jobs_by_action),
    )
    journal_entries = [_entry_with_id(entries, entry_id) for entry_id in journal_ids]
    by_path = {str(entry["relative_path"]): entry for entry in journal_entries}
    expected_paths = {
        _QUALIFYING_RUNTIME_JOURNAL_PATHS[action] for action in jobs_by_action
    }
    if set(by_path) != expected_paths:
        raise EvidenceStorageError(
            "A qualifying runtime journal crosses the started action prefix."
        )
    runtime_types = {
        event_type
        for sequence in _QUALIFYING_RUNTIME_SEQUENCE.values()
        for event_type, _phase in sequence
    }
    for action, job_id in jobs_by_action.items():
        entry = by_path[_QUALIFYING_RUNTIME_JOURNAL_PATHS[action]]
        records = _load_canonical_jsonl_entry(
            root, entry, require_jsonl_media_type=True
        )
        boundaries = []
        for record in records:
            try:
                boundaries.append(
                    validate_runtime_boundary(
                        record,
                        expected_run_id=experiment_run_id,
                        expected_job_id=job_id,
                        expected_action=action,
                    )
                )
            except (RuntimeBoundaryError, TypeError, ValueError) as error:
                raise EvidenceStorageError(
                    "A retained runtime-prefix boundary is invalid."
                ) from error
        if any(
            later.monotonic_ns < earlier.monotonic_ns
            for earlier, later in zip(boundaries, boundaries[1:])
        ):
            raise EvidenceStorageError(
                "A retained runtime-prefix journal moves backward."
            )
        ledger_rows = [
            row
            for row in ledger
            if row["event_type"] in runtime_types and row["action"] == action
        ]
        if len(ledger_rows) != len(boundaries):
            raise EvidenceStorageError(
                "A retained runtime-prefix journal differs from the ledger."
            )
        for row, boundary in zip(ledger_rows, boundaries):
            expected_row = {
                "schema_version": "aptus.experiment-event.v1",
                "sequence": row["sequence"],
                "experiment_run_id": experiment_run_id,
                **boundary.ledger_fields(),
                "source_reported_at_utc": None,
                "exit_code": None,
            }
            if row != expected_row:
                raise EvidenceStorageError(
                    "A runtime-prefix ledger row differs from its source journal."
                )


def _validate_protocol_experiment_run_record(
    root: Path,
    *,
    entry: Mapping[str, Any],
    attempt_slot_id: str,
    experiment_run_id: str,
    execution_configuration_id: str,
    aptus_job_ids: Sequence[str],
    aptus_run_ids: Sequence[str],
    terminal_jobs: Sequence[Mapping[str, Any]] | None = None,
    evidence_role_sha256: Mapping[str, str] | None = None,
    native_outcome: str = "passed",
    evidence_status: str = "protocol-valid",
    reason_code: str = "NONE",
) -> dict[str, Any]:
    raw = _load_canonical_json_entry(root, entry, require_json_media_type=True)
    try:
        record = validate_record(raw, "aptus.experiment-run.v1")
    except (ContractError, TypeError, ValueError) as error:
        raise EvidenceStorageError(
            "The canonical experiment-run record is invalid."
        ) from error
    if (
        record["attempt_slot_id"] != attempt_slot_id
        or record["experiment_run_id"] != experiment_run_id
        or record["execution_configuration_id"] != execution_configuration_id
        or record["aptus_job_ids"] != list(aptus_job_ids)
        or record["aptus_run_ids"] != list(aptus_run_ids)
    ):
        raise EvidenceStorageError(
            "The experiment-run record does not cross-bind exact run identities."
        )
    if terminal_jobs is not None:
        terminal = record["terminal_evidence"]
        try:
            timing = (
                terminal_timing_summary([dict(item) for item in terminal_jobs])
                if terminal_jobs
                else None
            )
        except (QualificationError, TypeError, ValueError) as error:
            raise EvidenceStorageError(
                "The experiment-run terminal timing is invalid."
            ) from error
        expected = {
            "native_outcome": native_outcome,
            "evidence_status": evidence_status,
            "reason_code": reason_code,
            "jobs": [dict(item) for item in terminal_jobs],
            "timing": timing,
            "evidence_role_sha256": dict(evidence_role_sha256 or {}),
        }
        if terminal != expected:
            raise EvidenceStorageError(
                "The experiment-run terminal evidence is not the exact retained set."
            )
    return record


def _validate_retained_activation_provenance(
    root: Path,
    *,
    identity_bindings: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    required_role_bindings: Mapping[str, str | Sequence[str]],
) -> RetainedActivatedSlot:
    """Require and deep-verify the planned context plus all seven activation files."""

    expected_paths = {
        "planned-slot-context": "activation/planned-slot-context.json",
        **{
            role: (
                "activation/ACTIVATED.json"
                if filename == ACTIVATION_SEAL_NAME
                else f"activation/{filename}"
            )
            for role, filename in _ACTIVATION_ROLE_FILES.items()
        },
    }
    role_entries: dict[str, Mapping[str, Any]] = {}
    for role, expected_path in expected_paths.items():
        entry_id = _require_complete_role_binding(
            role,
            bindings=required_role_bindings,
            entries=entries,
            minimum=1,
            maximum=1,
        )[0]
        entry = _entry_with_id(entries, entry_id)
        if (
            entry.get("relative_path") != expected_path
            or entry.get("media_type") != "application/json"
        ):
            raise EvidenceStorageError(
                "Retained activation provenance uses a noncanonical path or media type."
            )
        role_entries[role] = entry
    planned_record = _load_canonical_json_entry(
        root,
        role_entries["planned-slot-context"],
        require_json_media_type=True,
    )
    activation_payloads = {
        filename: _read_pinned_evidence_bytes(_entry_path(root, role_entries[role]))
        for role, filename in _ACTIVATION_ROLE_FILES.items()
    }
    try:
        retained = validate_retained_activated_slot(
            activation_payloads,
            planned_slot_context_record=planned_record,
        )
    except (AdmissionError, TypeError, ValueError) as error:
        raise EvidenceStorageError(
            "Retained activation provenance does not deep-verify."
        ) from error
    role_digests = {role: str(entry["sha256"]) for role, entry in role_entries.items()}
    if source_bindings.get("activation_provenance_sha256_by_role") != dict(
        sorted(role_digests.items())
    ):
        raise EvidenceStorageError(
            "Activation provenance is not exactly cross-bound by the raw manifest."
        )
    context = retained.planned_slot_context
    phase4_binding = dict(context.phase4_binding)
    expected_context_sources = {
        "campaign_sha256": sha256_bytes(canonical_json_bytes(dict(context.campaign))),
        "comparison_cohort_sha256": sha256_bytes(
            canonical_json_bytes(dict(context.comparison_cohort))
        ),
        "comparison_cell_sha256": sha256_bytes(
            canonical_json_bytes(dict(context.comparison_cell))
        ),
        "idle_baseline_binding_sha256": sha256_bytes(
            canonical_json_bytes(phase4_binding)
        ),
        "phase4_source_freeze_sha256": phase4_binding["phase4_source_freeze_sha256"],
        "phase4_source_freeze_seal_sha256": phase4_binding[
            "phase4_source_freeze_seal_sha256"
        ],
        "phase4_idle_baseline_samples_sha256": phase4_binding[
            "idle_baseline_samples_sha256"
        ],
    }
    if any(
        source_bindings.get(name) != digest
        for name, digest in expected_context_sources.items()
    ):
        raise EvidenceStorageError(
            "Activation provenance differs from its retained campaign or Phase-4 authority."
        )
    if (
        source_bindings.get("planned_slot_context_sha256") != context.sha256
        or source_bindings.get("planned_attempt_slot_sha256")
        != sha256_bytes(canonical_json_bytes(dict(context.planned_attempt_slot)))
        or source_bindings.get("admission_decision_sha256")
        != role_digests["activation-admission-decision"]
        or source_bindings.get("admission_observations_sha256")
        != role_digests["activation-admission-observations"]
        or source_bindings.get("activation_decision_sha256")
        != role_digests["activation-decision"]
        or source_bindings.get("started_identity_template_sha256")
        != role_digests["activation-started-identity-template"]
        or source_bindings.get("execution_configuration_sha256")
        != role_digests["activation-execution-configuration"]
        or source_bindings.get("experiment_run_template_sha256")
        != role_digests["activation-experiment-run-template"]
    ):
        raise EvidenceStorageError(
            "Activation provenance source and identity digests are inconsistent."
        )
    if (
        identity_bindings.get("campaign_id") != context.campaign["campaign_id"]
        or identity_bindings.get("comparison_cohort_id")
        != context.comparison_cohort["comparison_cohort_id"]
        or identity_bindings.get("comparison_cell_id")
        != context.comparison_cell["comparison_cell_id"]
        or identity_bindings.get("attempt_slot_id")
        != context.planned_attempt_slot["attempt_slot_id"]
        or identity_bindings.get("execution_configuration_id")
        != retained.execution_configuration["execution_configuration_id"]
        or identity_bindings.get("experiment_run_id")
        != retained.experiment_run_template["experiment_run_id"]
    ):
        raise EvidenceStorageError(
            "Activation provenance differs from the raw artifact identity chain."
        )
    return retained


def _validate_qualifying_payloads(
    root: Path,
    *,
    identity_bindings: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    required_role_bindings: Mapping[str, str | Sequence[str]],
    experiment_record: Mapping[str, Any],
    sequence_summary: Mapping[str, Any],
    terminal_jobs: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
    telemetry_start_monotonic_ns: int,
    telemetry_stop_monotonic_ns: int,
    cooldown_start_monotonic_ns: int,
    cooldown_stop_monotonic_ns: int,
) -> dict[str, str]:
    role_entries: dict[str, Mapping[str, Any]] = {}
    role_payloads: dict[str, dict[str, Any]] = {}
    for role in sorted(_QUALIFYING_SINGLE_ROLES):
        entry_id = _require_complete_role_binding(
            role,
            bindings=required_role_bindings,
            entries=entries,
            minimum=1,
            maximum=1,
        )[0]
        entry = _entry_with_id(entries, entry_id)
        role_entries[role] = entry
        if role not in {
            "bundle-archive",
            "phase4-idle-baseline-samples",
            "activation-admission-observations",
        }:
            role_payloads[role] = _load_canonical_json_entry(
                root, entry, require_json_media_type=True
            )

    archive_entry = role_entries["bundle-archive"]
    archive_path = root.joinpath(
        *PurePosixPath(str(archive_entry["relative_path"])).parts
    )
    try:
        archive_inventory = validate_bundle_archive_file(archive_path)
    except (OSError, ValueError):
        raise EvidenceStorageError(
            "The retained bundle archive is not exact deterministic Aptus output."
        ) from None
    if (
        archive_entry.get("media_type") != "application/zip"
        or not archive_entry.get("size_bytes")
        or not archive_inventory
    ):
        raise EvidenceStorageError("The retained bundle archive contract is invalid.")

    role_digests = {role: str(entry["sha256"]) for role, entry in role_entries.items()}
    if (
        archive_inventory.get("bundle-manifest.json", {}).get("sha256")
        != role_digests["bundle-manifest"]
        or archive_inventory.get("plan.json", {}).get("sha256") != role_digests["plan"]
    ):
        raise EvidenceStorageError(
            "The retained Aptus archive differs from its plan or bundle manifest."
        )
    bundle_manifest = role_payloads["bundle-manifest"]
    manifest_files = bundle_manifest.get("files")
    if not isinstance(manifest_files, list) or any(
        type(item) is not dict
        or set(item) != {"path", "sha256", "size_bytes"}
        or not isinstance(item["path"], str)
        or not item["path"]
        or not isinstance(item["sha256"], str)
        or _SHA256.fullmatch(item["sha256"]) is None
        or isinstance(item["size_bytes"], bool)
        or not isinstance(item["size_bytes"], int)
        or item["size_bytes"] < 0
        for item in manifest_files
    ):
        raise EvidenceStorageError("The retained bundle manifest inventory is invalid.")
    manifest_paths = [item["path"] for item in manifest_files]
    try:
        normalized_manifest_paths = [
            validate_safe_relative_path(path) for path in manifest_paths
        ]
    except (TypeError, ValueError) as error:
        raise EvidenceStorageError(
            "The retained bundle manifest path inventory is invalid."
        ) from error
    if (
        manifest_paths != sorted(manifest_paths)
        or manifest_paths != normalized_manifest_paths
        or len(manifest_paths) != len(set(manifest_paths))
        or "bundle-manifest.json" in manifest_paths
    ):
        raise EvidenceStorageError(
            "The retained bundle manifest paths are not canonical and unique."
        )
    expected_archive_inventory = {
        item["path"]: {
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in manifest_files
    }
    expected_archive_inventory["bundle-manifest.json"] = archive_inventory[
        "bundle-manifest.json"
    ]
    if archive_inventory != expected_archive_inventory:
        raise EvidenceStorageError(
            "The retained Aptus archive member inventory differs from its manifest."
        )
    manifest_digest_binding = source_bindings.get("evidence_role_sha256")
    if (
        type(manifest_digest_binding) is not dict
        or manifest_digest_binding != role_digests
    ):
        raise EvidenceStorageError(
            "The raw-manifest source bindings do not bind every qualifying role digest."
        )
    embedded_digests = experiment_record["terminal_evidence"].get(
        "evidence_role_sha256"
    )
    expected_embedded = {
        role: role_digests[role] for role in sorted(_RUN_EMBEDDED_DIGEST_ROLES)
    }
    if embedded_digests != expected_embedded:
        raise EvidenceStorageError(
            "The experiment-run record does not bind every non-self evidence digest."
        )

    try:
        campaign = validate_record(
            role_payloads["campaign-record"], "aptus.experiment-campaign.v1"
        )
        cohort = validate_record(
            role_payloads["comparison-cohort-record"],
            "aptus.experiment-comparison-cohort.v1",
        )
        cell = validate_record(
            role_payloads["comparison-cell-record"],
            "aptus.experiment-comparison-cell.v1",
        )
        attempt = validate_record(
            role_payloads["attempt-slot-record"], "aptus.experiment-attempt-slot.v1"
        )
        execution = validate_record(
            role_payloads["execution-configuration-record"],
            "aptus.experiment-execution-configuration.v1",
        )
        idle_baseline = validate_idle_baseline_binding(
            role_payloads["idle-baseline-binding"]
        )
        phase4_samples_path = _entry_path(
            root, role_entries["phase4-idle-baseline-samples"]
        )
        phase4 = validate_retained_phase4_source_freeze(
            source_freeze_bytes=_read_pinned_evidence_bytes(
                _entry_path(root, role_entries["phase4-source-freeze"])
            ),
            idle_samples_bytes=_read_pinned_evidence_bytes(phase4_samples_path),
            seal_bytes=_read_pinned_evidence_bytes(
                _entry_path(root, role_entries["phase4-source-freeze-seal"])
            ),
            campaign=campaign,
            comparison_cohort=cohort,
            comparison_cell=cell,
        )
    except (ContractError, Phase4SourceFreezeError, TypeError, ValueError) as error:
        raise EvidenceStorageError(
            "A qualifying identity record violates its canonical schema."
        ) from error
    if dict(phase4.baseline_binding) != idle_baseline:
        raise EvidenceStorageError(
            "The retained idle-baseline binding differs from its Phase-4 source."
        )
    try:
        for terminal_job in terminal_jobs:
            terminal_job.update(
                validate_qualifying_terminal_timing(
                    terminal_job,
                    expected_boot_sha256=idle_baseline["current_boot_id_sha256"],
                )
            )
        terminal_timing_summary([dict(item) for item in terminal_jobs])
    except (QualificationError, TypeError, ValueError) as error:
        raise EvidenceStorageError(
            "A qualifying terminal job has invalid same-boot monotonic timing."
        ) from error
    if (
        cohort["campaign_id"] != campaign["campaign_id"]
        or cell["campaign_id"] != campaign["campaign_id"]
        or cell["comparison_cell_id"] not in cohort["member_cell_ids"]
        or attempt["comparison_cohort_id"] != cohort["comparison_cohort_id"]
        or attempt["comparison_cell_id"] != cell["comparison_cell_id"]
        or execution["comparison_cell_id"] != cell["comparison_cell_id"]
    ):
        raise EvidenceStorageError(
            "The retained campaign, cohort, cell, slot, and execution chain is misbound."
        )
    if (
        identity_bindings.get("campaign_id") != campaign["campaign_id"]
        or identity_bindings.get("comparison_cohort_id")
        != cohort["comparison_cohort_id"]
        or identity_bindings.get("comparison_cell_id") != cell["comparison_cell_id"]
        or attempt["slot_status"] != "started"
        or attempt["attempt_slot_id"] != identity_bindings["attempt_slot_id"]
        or attempt["experiment_run_id"] != identity_bindings["experiment_run_id"]
        or attempt["execution_configuration_id"]
        != identity_bindings["execution_configuration_id"]
        or attempt["native_outcome"] != sequence_summary["native_outcome"]
        or attempt["evidence_status"] != "protocol-valid"
        or attempt["reason_code"] != sequence_summary["reason_code"]
        or execution["execution_configuration_id"]
        != identity_bindings["execution_configuration_id"]
        or execution["comparison_cell_id"] != attempt["comparison_cell_id"]
        or execution["training_seed"] != attempt["scheduled_seed"]
    ):
        raise EvidenceStorageError(
            "The attempt-slot and execution-configuration identities are misbound."
        )
    if any(
        experiment_record[field] != execution[field]
        for field in ("plan_id", "candidate_id", "bundle_fingerprint")
    ):
        raise EvidenceStorageError(
            "The execution configuration differs from the experiment run."
        )
    retained_plan = role_payloads["plan"]
    if (
        execution["bundle_fingerprint"] != role_digests["bundle-manifest"]
        or retained_plan.get("plan_id") != execution["plan_id"]
        or type(retained_plan.get("recommended")) is not dict
        or retained_plan["recommended"].get("candidate_id") != execution["candidate_id"]
        or bundle_manifest.get("plan_id") != execution["plan_id"]
        or bundle_manifest.get("candidate_id") != execution["candidate_id"]
        or bundle_manifest.get("plan_sha256") != role_digests["plan"]
    ):
        raise EvidenceStorageError(
            "The retained plan, manifest, and execution bundle identity differ."
        )
    if any(
        not role_entries[role].get("size_bytes")
        for role in REQUIRED_QUALIFYING_ARTIFACT_ROLES
    ):
        raise EvidenceStorageError(
            "A frozen qualifying artifact role has an empty canonical payload."
        )
    if experiment_record["bundle_manifest_sha256"] != role_digests["bundle-manifest"]:
        raise EvidenceStorageError(
            "The retained bundle manifest differs from the experiment-run digest."
        )
    if experiment_record["archive_sha256"] != role_digests["bundle-archive"]:
        raise EvidenceStorageError(
            "The retained bundle archive differs from the experiment-run digest."
        )
    run_order = experiment_record["run_order"]
    if (
        type(run_order) is not dict
        or run_order.get("block") != attempt["block"]
        or run_order.get("position") != attempt["order_position"]
    ):
        raise EvidenceStorageError("The experiment run order is not its attempt slot.")
    if (
        source_bindings.get("execution_configuration_sha256")
        != role_digests["execution-configuration-record"]
    ):
        raise EvidenceStorageError(
            "The predeclared execution-configuration digest is not retained."
        )
    expected_authority_sources = {
        "campaign_sha256": role_digests["campaign-record"],
        "comparison_cohort_sha256": role_digests["comparison-cohort-record"],
        "comparison_cell_sha256": role_digests["comparison-cell-record"],
        "phase4_source_freeze_sha256": role_digests["phase4-source-freeze"],
        "phase4_source_freeze_seal_sha256": role_digests["phase4-source-freeze-seal"],
        "phase4_idle_baseline_samples_sha256": role_digests[
            "phase4-idle-baseline-samples"
        ],
    }
    if any(
        source_bindings.get(name) != digest
        for name, digest in expected_authority_sources.items()
    ):
        raise EvidenceStorageError(
            "The retained campaign or Phase-4 authority digest is misbound."
        )
    baseline_digest = role_digests["idle-baseline-binding"]
    observed_host_state = experiment_record["observed_host_state"]
    if (
        source_bindings.get("idle_baseline_binding_sha256") != baseline_digest
        or type(observed_host_state) is not dict
        or observed_host_state.get("idle_baseline_sha256") != baseline_digest
    ):
        raise EvidenceStorageError(
            "The retained idle-baseline binding differs from its run provenance."
        )
    if experiment_record["terminal_evidence"].get("jobs") != [
        dict(item) for item in terminal_jobs
    ]:
        raise EvidenceStorageError(
            "The experiment-run job evidence differs from retained terminals."
        )

    behavior = execution["exact_behavior_values"]
    remaining_disk_budget = behavior.get("remaining_disk_budget_bytes")
    if (
        isinstance(remaining_disk_budget, bool)
        or not isinstance(remaining_disk_budget, int)
        or remaining_disk_budget < 0
    ):
        raise EvidenceStorageError(
            "The execution configuration lacks its remaining disk budget."
        )
    telemetry_context = SimpleNamespace(
        emergency_deadline_seconds=execution["emergency_deadline_seconds"],
        remaining_disk_budget_bytes=remaining_disk_budget,
    )
    try:
        telemetry_configuration = validate_qualifying_telemetry_configuration(
            role_payloads["telemetry-configuration"], context=telemetry_context
        )
    except (QualificationError, TypeError, ValueError) as error:
        raise EvidenceStorageError(
            "The retained telemetry configuration is not the frozen profile."
        ) from error
    if telemetry_configuration != phase4.source_freeze["telemetry_configuration"]:
        raise EvidenceStorageError(
            "The retained telemetry configuration differs from Phase 4."
        )

    telemetry_envelope = role_payloads["telemetry-summary"]
    if set(telemetry_envelope) != {
        "record_kind",
        "experiment_run_id",
        "telemetry",
        "segments",
    } or (
        telemetry_envelope["record_kind"] != "aptus-cuda-campaign-telemetry-summary-v1"
        or telemetry_envelope["experiment_run_id"]
        != identity_bindings["experiment_run_id"]
    ):
        raise EvidenceStorageError("The telemetry-summary envelope is misbound.")
    try:
        expected_telemetry = summarize_telemetry(
            telemetry,
            telemetry_start_monotonic_ns,
            telemetry_stop_monotonic_ns,
        )
        present_slots = {sample["scheduled_slot"] for sample in telemetry}
        expected_telemetry["missing_scheduled_slots"] = [
            slot
            for slot in range(expected_telemetry["expected_sample_count"])
            if slot not in present_slots
        ]
        expected_segments = build_segment_summaries(list(telemetry), list(ledger))
    except (
        QualificationError,
        TelemetryValidationError,
        TypeError,
        ValueError,
    ) as error:
        raise EvidenceStorageError(
            "The retained telemetry summary cannot be derived."
        ) from error
    if (
        telemetry_envelope["telemetry"] != expected_telemetry
        or telemetry_envelope["segments"] != expected_segments
    ):
        raise EvidenceStorageError(
            "The telemetry summary differs from the retained samples and ledger."
        )

    expected_cooldown = _cooldown_summary(
        telemetry,
        start_monotonic_ns=cooldown_start_monotonic_ns,
        stop_monotonic_ns=cooldown_stop_monotonic_ns,
        idle_baseline=idle_baseline["summary"],
    )
    cooldown_envelope = role_payloads["cooldown-summary"]
    if cooldown_envelope != {
        "record_kind": "aptus-cuda-campaign-cooldown-summary-v1",
        "experiment_run_id": identity_bindings["experiment_run_id"],
        "valid": True,
        "reason_codes": [],
        "summary": expected_cooldown,
    }:
        raise EvidenceStorageError(
            "The cooldown summary differs from its exact retained window."
        )
    return role_digests


def _validate_qualifying_prefix_payloads(
    root: Path,
    *,
    identity_bindings: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    required_role_bindings: Mapping[str, str | Sequence[str]],
    experiment_record: Mapping[str, Any],
    sequence_summary: Mapping[str, Any],
    started_actions: Sequence[Mapping[str, Any]],
    terminal_jobs: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
    telemetry_start_monotonic_ns: int,
    telemetry_stop_monotonic_ns: int,
) -> dict[str, str]:
    """Deep-verify the exact retained prefix for one protocol-valid non-pass."""

    expected_roles = {
        "attempt-slot-record",
        "execution-configuration-record",
        "experiment-run-record",
        "idle-baseline-binding",
        "telemetry-configuration",
        "telemetry-summary",
        "plan",
        "bundle-manifest",
        "validation-report",
        "bundle-archive",
    } | REQUIRED_QUALIFYING_AUTHORITY_ROLES
    if any(
        action["action"] == "pilot" and action["native_outcome"] == "passed"
        for action in started_actions
    ):
        expected_roles.add("pilot-metrics")
    forbidden = {
        "training-metrics",
        "final-export-manifest",
        "cooldown-summary",
    }
    present_single_roles = {
        str(entry["role"])
        for entry in entries
        if entry.get("role") in _QUALIFYING_SINGLE_ROLES
    }
    if present_single_roles != expected_roles or present_single_roles & forbidden:
        raise EvidenceStorageError(
            "The retained qualifying prefix crosses its exact stopping boundary."
        )

    role_entries: dict[str, Mapping[str, Any]] = {}
    role_payloads: dict[str, dict[str, Any]] = {}
    for role in sorted(expected_roles):
        entry_id = _require_complete_role_binding(
            role,
            bindings=required_role_bindings,
            entries=entries,
            minimum=1,
            maximum=1,
        )[0]
        entry = _entry_with_id(entries, entry_id)
        role_entries[role] = entry
        if role not in {
            "bundle-archive",
            "phase4-idle-baseline-samples",
            "activation-admission-observations",
        }:
            role_payloads[role] = _load_canonical_json_entry(
                root, entry, require_json_media_type=True
            )

    role_digests = {role: str(entry["sha256"]) for role, entry in role_entries.items()}
    if source_bindings.get("evidence_role_sha256") != role_digests:
        raise EvidenceStorageError(
            "The raw manifest does not bind the exact qualifying prefix digests."
        )
    expected_embedded = {
        role: digest
        for role, digest in role_digests.items()
        if role != "experiment-run-record"
    }
    if experiment_record["terminal_evidence"].get("evidence_role_sha256") != dict(
        sorted(expected_embedded.items())
    ):
        raise EvidenceStorageError(
            "The final run does not bind its exact non-pass evidence prefix."
        )

    archive_entry = role_entries["bundle-archive"]
    archive_path = _entry_path(root, archive_entry)
    try:
        archive_inventory = validate_bundle_archive_file(archive_path)
        campaign = validate_record(
            role_payloads["campaign-record"], "aptus.experiment-campaign.v1"
        )
        cohort = validate_record(
            role_payloads["comparison-cohort-record"],
            "aptus.experiment-comparison-cohort.v1",
        )
        cell = validate_record(
            role_payloads["comparison-cell-record"],
            "aptus.experiment-comparison-cell.v1",
        )
        attempt = validate_record(
            role_payloads["attempt-slot-record"],
            "aptus.experiment-attempt-slot.v1",
        )
        execution = validate_record(
            role_payloads["execution-configuration-record"],
            "aptus.experiment-execution-configuration.v1",
        )
        idle_baseline = validate_idle_baseline_binding(
            role_payloads["idle-baseline-binding"]
        )
        phase4 = validate_retained_phase4_source_freeze(
            source_freeze_bytes=_read_pinned_evidence_bytes(
                _entry_path(root, role_entries["phase4-source-freeze"])
            ),
            idle_samples_bytes=_read_pinned_evidence_bytes(
                _entry_path(root, role_entries["phase4-idle-baseline-samples"])
            ),
            seal_bytes=_read_pinned_evidence_bytes(
                _entry_path(root, role_entries["phase4-source-freeze-seal"])
            ),
            campaign=campaign,
            comparison_cohort=cohort,
            comparison_cell=cell,
        )
    except (
        ContractError,
        OSError,
        Phase4SourceFreezeError,
        TypeError,
        ValueError,
    ) as error:
        raise EvidenceStorageError(
            "A retained qualifying-prefix authority is invalid."
        ) from error
    if not archive_inventory or archive_entry.get("media_type") != "application/zip":
        raise EvidenceStorageError("The retained prefix archive is invalid.")
    manifest = role_payloads["bundle-manifest"]
    manifest_files = manifest.get("files")
    if type(manifest_files) is not list:
        raise EvidenceStorageError("The retained prefix manifest is invalid.")
    try:
        expected_archive_inventory = {
            validate_safe_relative_path(str(item["path"])): {
                "sha256": str(item["sha256"]),
                "size_bytes": int(item["size_bytes"]),
            }
            for item in manifest_files
            if type(item) is dict
        }
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceStorageError(
            "The retained prefix manifest inventory is invalid."
        ) from error
    if len(expected_archive_inventory) != len(manifest_files):
        raise EvidenceStorageError(
            "The retained prefix manifest paths are duplicated or invalid."
        )
    expected_archive_inventory["bundle-manifest.json"] = archive_inventory.get(
        "bundle-manifest.json", {}
    )
    if (
        archive_inventory != expected_archive_inventory
        or archive_inventory.get("plan.json", {}).get("sha256") != role_digests["plan"]
        or archive_inventory.get("bundle-manifest.json", {}).get("sha256")
        != role_digests["bundle-manifest"]
    ):
        raise EvidenceStorageError(
            "The retained prefix archive differs from its exact manifest."
        )

    if dict(phase4.baseline_binding) != idle_baseline:
        raise EvidenceStorageError(
            "The retained prefix baseline differs from its Phase-4 authority."
        )
    try:
        for terminal_job in terminal_jobs:
            terminal_job.update(
                validate_qualifying_terminal_timing(
                    terminal_job,
                    expected_boot_sha256=idle_baseline["current_boot_id_sha256"],
                )
            )
        if terminal_jobs:
            terminal_timing_summary([dict(item) for item in terminal_jobs])
    except (QualificationError, TypeError, ValueError) as error:
        raise EvidenceStorageError(
            "A non-pass terminal job has invalid same-boot timing."
        ) from error
    if (
        cohort["campaign_id"] != campaign["campaign_id"]
        or cell["campaign_id"] != campaign["campaign_id"]
        or cell["comparison_cell_id"] not in cohort["member_cell_ids"]
        or attempt["comparison_cohort_id"] != cohort["comparison_cohort_id"]
        or attempt["comparison_cell_id"] != cell["comparison_cell_id"]
        or execution["comparison_cell_id"] != cell["comparison_cell_id"]
        or attempt["slot_status"] != "started"
        or attempt["native_outcome"] != sequence_summary["native_outcome"]
        or attempt["evidence_status"] != "protocol-valid"
        or attempt["reason_code"] != sequence_summary["reason_code"]
        or attempt["experiment_run_id"] != identity_bindings["experiment_run_id"]
        or attempt["execution_configuration_id"]
        != identity_bindings["execution_configuration_id"]
        or execution["execution_configuration_id"]
        != identity_bindings["execution_configuration_id"]
        or execution["training_seed"] != attempt["scheduled_seed"]
    ):
        raise EvidenceStorageError(
            "The retained non-pass campaign identity chain is misbound."
        )
    if any(
        experiment_record[field] != execution[field]
        for field in ("plan_id", "candidate_id", "bundle_fingerprint")
    ):
        raise EvidenceStorageError(
            "The non-pass experiment run differs from its execution configuration."
        )
    plan = role_payloads["plan"]
    if (
        execution["bundle_fingerprint"] != role_digests["bundle-manifest"]
        or plan.get("plan_id") != execution["plan_id"]
        or type(plan.get("recommended")) is not dict
        or plan["recommended"].get("candidate_id") != execution["candidate_id"]
        or manifest.get("plan_id") != execution["plan_id"]
        or manifest.get("candidate_id") != execution["candidate_id"]
        or manifest.get("plan_sha256") != role_digests["plan"]
        or experiment_record["bundle_manifest_sha256"]
        != role_digests["bundle-manifest"]
        or experiment_record["archive_sha256"] != role_digests["bundle-archive"]
    ):
        raise EvidenceStorageError(
            "The retained non-pass plan, manifest, archive, and run differ."
        )
    expected_authority_sources = {
        "campaign_sha256": role_digests["campaign-record"],
        "comparison_cohort_sha256": role_digests["comparison-cohort-record"],
        "comparison_cell_sha256": role_digests["comparison-cell-record"],
        "phase4_source_freeze_sha256": role_digests["phase4-source-freeze"],
        "phase4_source_freeze_seal_sha256": role_digests["phase4-source-freeze-seal"],
        "phase4_idle_baseline_samples_sha256": role_digests[
            "phase4-idle-baseline-samples"
        ],
        "execution_configuration_sha256": role_digests[
            "execution-configuration-record"
        ],
        "idle_baseline_binding_sha256": role_digests["idle-baseline-binding"],
    }
    if any(
        source_bindings.get(name) != digest
        for name, digest in expected_authority_sources.items()
    ):
        raise EvidenceStorageError(
            "The retained non-pass authority source digests are misbound."
        )

    behavior = execution["exact_behavior_values"]
    remaining_disk_budget = behavior.get("remaining_disk_budget_bytes")
    if (
        isinstance(remaining_disk_budget, bool)
        or not isinstance(remaining_disk_budget, int)
        or remaining_disk_budget < 0
    ):
        raise EvidenceStorageError(
            "The non-pass execution lacks its remaining disk budget."
        )
    telemetry_context = SimpleNamespace(
        emergency_deadline_seconds=execution["emergency_deadline_seconds"],
        remaining_disk_budget_bytes=remaining_disk_budget,
    )
    try:
        telemetry_configuration = validate_qualifying_telemetry_configuration(
            role_payloads["telemetry-configuration"], context=telemetry_context
        )
        expected_telemetry = summarize_telemetry(
            telemetry,
            telemetry_start_monotonic_ns,
            telemetry_stop_monotonic_ns,
        )
        present_slots = {sample["scheduled_slot"] for sample in telemetry}
        expected_telemetry["missing_scheduled_slots"] = [
            slot
            for slot in range(expected_telemetry["expected_sample_count"])
            if slot not in present_slots
        ]
        expected_segments = build_segment_summaries(
            list(telemetry),
            list(ledger),
            allow_open_terminal_prefix=True,
        )
    except (
        QualificationError,
        TelemetryValidationError,
        TypeError,
        ValueError,
    ) as error:
        raise EvidenceStorageError(
            "The non-pass telemetry summary cannot be derived."
        ) from error
    telemetry_envelope = role_payloads["telemetry-summary"]
    if telemetry_configuration != phase4.source_freeze[
        "telemetry_configuration"
    ] or telemetry_envelope != {
        "record_kind": "aptus-cuda-campaign-telemetry-summary-v1",
        "experiment_run_id": identity_bindings["experiment_run_id"],
        "telemetry": expected_telemetry,
        "segments": expected_segments,
    }:
        raise EvidenceStorageError(
            "The retained non-pass telemetry authority is misbound."
        )
    if experiment_record["terminal_evidence"].get("jobs") != [
        dict(item) for item in terminal_jobs
    ]:
        raise EvidenceStorageError(
            "The non-pass final run differs from retained terminal jobs."
        )
    return role_digests


def _validate_protocol_command_record(
    command: Mapping[str, Any],
    *,
    attempt_slot_id: str,
    experiment_run_id: str,
    experiment_record: Mapping[str, Any] | None,
    ledger: Sequence[Mapping[str, Any]],
) -> None:
    required = {
        "record_kind",
        "experiment_run_id",
        "attempt_slot_id",
        "exact_argv",
        "working_directory",
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "exit_code",
        "timed_out",
        "native_outcome",
        "reason_code",
    }
    value = _require_exact_object_fields(command, required, label="Command record")
    starts = [row for row in ledger if row["event_type"] == "command.started"]
    finishes = [row for row in ledger if row["event_type"] == "command.finished"]
    if (
        len(starts) != 1
        or len(finishes) != 1
        or value["record_kind"] != "aptus-cuda-campaign-command-capture-v1"
        or value["experiment_run_id"] != experiment_run_id
        or value["attempt_slot_id"] != attempt_slot_id
        or (
            experiment_record is not None
            and value["exact_argv"] != experiment_record["exact_argv"]
        )
        or (
            experiment_record is not None
            and value["working_directory"] != experiment_record["working_directory"]
        )
        or value["started_monotonic_ns"] != starts[0]["monotonic_ns"]
        or value["finished_monotonic_ns"] != finishes[0]["monotonic_ns"]
        or value["native_outcome"] != finishes[0]["native_outcome"]
        or value["reason_code"] != finishes[0]["reason_code"]
        or value["exit_code"] != finishes[0]["exit_code"]
        or starts[0]["action"] != "command"
        or finishes[0]["action"] != "command"
        or starts[0]["subject_kind"] != "process"
        or finishes[0]["subject_kind"] != "process"
        or starts[0]["subject_id"] != experiment_run_id
        or finishes[0]["subject_id"] != experiment_run_id
    ):
        raise EvidenceStorageError("The command capture identity is inconsistent.")
    if (
        type(value["exact_argv"]) is not list
        or not value["exact_argv"]
        or any(
            not isinstance(item, str) or not item or "\x00" in item
            for item in value["exact_argv"]
        )
        or not isinstance(value["working_directory"], str)
        or not Path(value["working_directory"]).is_absolute()
        or value["started_monotonic_ns"] > value["finished_monotonic_ns"]
        or not isinstance(value["timed_out"], bool)
        or value["timed_out"] != (value["native_outcome"] == "timed-out")
    ):
        raise EvidenceStorageError("The command capture fields are invalid.")


def _validate_protocol_valid_payloads(
    root: Path,
    *,
    identity_bindings: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    required_role_bindings: Mapping[str, str | Sequence[str]],
    capture_kind: str,
) -> None:
    attempt_slot_id = str(identity_bindings["attempt_slot_id"])
    experiment_run_id = str(identity_bindings["experiment_run_id"])
    execution_configuration_id = identity_bindings.get("execution_configuration_id")
    qualifying_sequence = capture_kind == "managed-sequence"
    if (
        identity_bindings.get("capture_kind") != capture_kind
        or identity_bindings.get("capture_status", "complete") != "complete"
        or identity_bindings.get("capture_reason_code") != "NONE"
    ):
        raise EvidenceStorageError("Protocol-valid identity bindings are inconsistent.")
    if qualifying_sequence and (
        not isinstance(execution_configuration_id, str)
        or _EXECUTION_CONFIGURATION_ID.fullmatch(execution_configuration_id) is None
    ):
        raise EvidenceStorageError(
            "Protocol-valid identity bindings lack an exact execution configuration."
        )
    if execution_configuration_id is not None and (
        not isinstance(execution_configuration_id, str)
        or _EXECUTION_CONFIGURATION_ID.fullmatch(execution_configuration_id) is None
    ):
        raise EvidenceStorageError("An optional execution configuration ID is invalid.")
    ledger_id = _require_complete_role_binding(
        "event-ledger",
        bindings=required_role_bindings,
        entries=entries,
        minimum=1,
        maximum=1,
    )[0]
    telemetry_ids = _require_complete_role_binding(
        "telemetry",
        bindings=required_role_bindings,
        entries=entries,
        minimum=1 if qualifying_sequence else 0,
        maximum=1,
    )
    run_record_ids = _require_complete_role_binding(
        "experiment-run-record",
        bindings=required_role_bindings,
        entries=entries,
        minimum=1 if qualifying_sequence else 0,
        maximum=1,
    )
    ledger_entry = _entry_with_id(entries, ledger_id)
    telemetry_entry = (
        _entry_with_id(entries, telemetry_ids[0]) if telemetry_ids else None
    )
    run_record_entry = (
        _entry_with_id(entries, run_record_ids[0]) if run_record_ids else None
    )
    ledger_records = _load_canonical_jsonl_entry(
        root, ledger_entry, require_jsonl_media_type=True
    )

    started_actions: list[dict[str, Any]] | None = None
    aptus_job_ids: list[str] = []
    aptus_run_ids: list[str] = []
    terminal_jobs: list[dict[str, Any]] = []
    sequence_summary: dict[str, Any] | None = None
    if capture_kind in {"managed-job", "managed-sequence"}:
        summary_id = _require_complete_role_binding(
            "sequence-summary",
            bindings=required_role_bindings,
            entries=entries,
            minimum=1,
            maximum=1,
        )[0]
        summary_entry = _entry_with_id(entries, summary_id)
        sequence_summary = _load_canonical_json_entry(
            root, summary_entry, require_json_media_type=True
        )
        started_actions = _validate_protocol_sequence_summary(
            sequence_summary,
            attempt_slot_id=attempt_slot_id,
            experiment_run_id=experiment_run_id,
        )
        aptus_job_ids, aptus_run_ids, terminal_jobs = (
            _validate_protocol_managed_payloads(
                root,
                capture_kind=capture_kind,
                entries=entries,
                started_actions=started_actions,
            )
        )

    ledger, telemetry_start_ns, telemetry_stop_ns = _validate_protocol_event_ledger(
        ledger_records,
        experiment_run_id=experiment_run_id,
        started_actions=started_actions,
        require_telemetry=telemetry_entry is not None,
    )
    outcome_profile = None
    if sequence_summary is not None:
        try:
            outcome_profile = validate_managed_sequence_outcome(
                sequence_summary, ledger
            )
        except (OutcomeProfileError, TypeError, ValueError) as error:
            raise EvidenceStorageError(
                "The retained managed outcome profile is invalid."
            ) from error
    if sequence_summary is not None:
        command_boundaries = [
            row
            for row in ledger
            if row["event_type"] in {"command.started", "command.finished"}
        ]
        expected_five_action_duration = (
            command_boundaries[-1]["monotonic_ns"]
            - command_boundaries[0]["monotonic_ns"]
            if started_actions is not None
            and len(started_actions) == 5
            and len(command_boundaries) == 10
            else None
        )
        if sequence_summary["five_action_duration_ns"] != expected_five_action_duration:
            raise EvidenceStorageError(
                "The five-action duration differs from command ledger boundaries."
            )
    telemetry: list[dict[str, Any]] = []
    if telemetry_entry is not None:
        if telemetry_start_ns is None or telemetry_stop_ns is None:  # pragma: no cover
            raise EvidenceStorageError("A telemetry payload lacks its ledger window.")
        telemetry_records = _load_canonical_jsonl_entry(
            root, telemetry_entry, require_jsonl_media_type=True
        )
        telemetry = _validate_protocol_telemetry(
            telemetry_records,
            experiment_run_id=experiment_run_id,
            start_monotonic_ns=telemetry_start_ns,
            stop_monotonic_ns=telemetry_stop_ns,
        )
    experiment_record: dict[str, Any] | None = None
    if run_record_entry is not None:
        if not isinstance(execution_configuration_id, str):
            raise EvidenceStorageError(
                "An experiment-run record lacks its execution configuration ID."
            )
        experiment_record = _validate_protocol_experiment_run_record(
            root,
            entry=run_record_entry,
            attempt_slot_id=attempt_slot_id,
            experiment_run_id=experiment_run_id,
            execution_configuration_id=execution_configuration_id,
            aptus_job_ids=aptus_job_ids,
            aptus_run_ids=aptus_run_ids,
        )
    retained_activation: RetainedActivatedSlot | None = None
    if qualifying_sequence and "campaign_id" in identity_bindings:
        retained_activation = _validate_retained_activation_provenance(
            root,
            identity_bindings=identity_bindings,
            source_bindings=source_bindings,
            entries=entries,
            required_role_bindings=required_role_bindings,
        )
        if experiment_record is not None:
            template = dict(retained_activation.experiment_run_template)
            mutable_terminal_fields = {
                "exact_argv",
                "aptus_job_ids",
                "aptus_run_ids",
                "terminal_evidence",
            }
            if any(
                experiment_record[field] != value
                for field, value in template.items()
                if field not in mutable_terminal_fields
            ):
                raise EvidenceStorageError(
                    "Final experiment run differs from its activated template."
                )
    if (
        qualifying_sequence
        and outcome_profile is not None
        and outcome_profile.publication_eligible
    ):
        if started_actions is None or sequence_summary is None:  # pragma: no cover
            raise EvidenceStorageError("A qualifying sequence summary is absent.")
        if (
            experiment_record is None
            or run_record_entry is None
            or telemetry_start_ns is None
            or telemetry_stop_ns is None
        ):  # pragma: no cover - role requirements guard this.
            raise EvidenceStorageError("A qualifying semantic role is absent.")
        cooldown_start_ns, cooldown_stop_ns = _validate_qualifying_event_sequence(
            ledger,
            experiment_run_id=experiment_run_id,
            started_actions=started_actions,
        )
        _validate_qualifying_runtime_journals(
            root,
            experiment_run_id=experiment_run_id,
            entries=entries,
            required_role_bindings=required_role_bindings,
            started_actions=started_actions,
            ledger=ledger,
        )
        role_digests = _validate_qualifying_payloads(
            root,
            identity_bindings=identity_bindings,
            source_bindings=source_bindings,
            entries=entries,
            required_role_bindings=required_role_bindings,
            experiment_record=experiment_record,
            sequence_summary=sequence_summary,
            terminal_jobs=terminal_jobs,
            ledger=ledger,
            telemetry=telemetry,
            telemetry_start_monotonic_ns=telemetry_start_ns,
            telemetry_stop_monotonic_ns=telemetry_stop_ns,
            cooldown_start_monotonic_ns=cooldown_start_ns,
            cooldown_stop_monotonic_ns=cooldown_stop_ns,
        )
        _validate_protocol_experiment_run_record(
            root,
            entry=run_record_entry,
            attempt_slot_id=attempt_slot_id,
            experiment_run_id=experiment_run_id,
            execution_configuration_id=execution_configuration_id,
            aptus_job_ids=aptus_job_ids,
            aptus_run_ids=aptus_run_ids,
            terminal_jobs=terminal_jobs,
            evidence_role_sha256={
                role: role_digests[role] for role in sorted(_RUN_EMBEDDED_DIGEST_ROLES)
            },
            native_outcome=sequence_summary["native_outcome"],
            evidence_status=sequence_summary["evidence_status"],
            reason_code=sequence_summary["reason_code"],
        )
    elif qualifying_sequence and outcome_profile is not None:
        if (
            started_actions is None
            or sequence_summary is None
            or experiment_record is None
            or run_record_entry is None
            or telemetry_start_ns is None
            or telemetry_stop_ns is None
            or retained_activation is None
        ):  # pragma: no cover - qualifying role guards.
            raise EvidenceStorageError(
                "A qualifying non-pass prefix lacks its semantic authorities."
            )
        _validate_qualifying_runtime_journal_prefix(
            root,
            experiment_run_id=experiment_run_id,
            entries=entries,
            required_role_bindings=required_role_bindings,
            started_actions=started_actions,
            ledger=ledger,
        )
        role_digests = _validate_qualifying_prefix_payloads(
            root,
            identity_bindings=identity_bindings,
            source_bindings=source_bindings,
            entries=entries,
            required_role_bindings=required_role_bindings,
            experiment_record=experiment_record,
            sequence_summary=sequence_summary,
            started_actions=started_actions,
            terminal_jobs=terminal_jobs,
            ledger=ledger,
            telemetry=telemetry,
            telemetry_start_monotonic_ns=telemetry_start_ns,
            telemetry_stop_monotonic_ns=telemetry_stop_ns,
        )
        _validate_protocol_experiment_run_record(
            root,
            entry=run_record_entry,
            attempt_slot_id=attempt_slot_id,
            experiment_run_id=experiment_run_id,
            execution_configuration_id=execution_configuration_id,
            aptus_job_ids=aptus_job_ids,
            aptus_run_ids=aptus_run_ids,
            terminal_jobs=terminal_jobs,
            evidence_role_sha256={
                role: digest
                for role, digest in role_digests.items()
                if role != "experiment-run-record"
            },
            native_outcome=sequence_summary["native_outcome"],
            evidence_status=sequence_summary["evidence_status"],
            reason_code=sequence_summary["reason_code"],
        )
    if capture_kind == "command":
        command_id = _require_complete_role_binding(
            "command-record",
            bindings=required_role_bindings,
            entries=entries,
            minimum=1,
            maximum=1,
        )[0]
        command = _load_canonical_json_entry(
            root,
            _entry_with_id(entries, command_id),
            require_json_media_type=True,
        )
        _validate_protocol_command_record(
            command,
            attempt_slot_id=attempt_slot_id,
            experiment_run_id=experiment_run_id,
            experiment_record=experiment_record,
            ledger=ledger,
        )


def _validate_experiment_run_completeness(
    root: Path,
    *,
    protected_artifact_id: str,
    identity_bindings: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    required_role_bindings: Mapping[str, str | Sequence[str]],
) -> None:
    """Reject an experiment-run seal unless its core capture is self-complete."""

    attempt_slot_id = identity_bindings.get("attempt_slot_id")
    experiment_run_id = identity_bindings.get("experiment_run_id")
    if not isinstance(attempt_slot_id, str) or not _ATTEMPT_SLOT_ID.fullmatch(
        attempt_slot_id
    ):
        raise EvidenceStorageError(
            "Experiment-run identity requires an exact attempt_slot_id."
        )
    if not isinstance(experiment_run_id, str) or not _EXPERIMENT_RUN_ID.fullmatch(
        experiment_run_id
    ):
        raise EvidenceStorageError(
            "Experiment-run identity requires an exact experiment_run_id."
        )

    capture_status = identity_bindings.get("capture_status", "complete")
    if capture_status not in {"complete", "failed"}:
        raise EvidenceStorageError("Experiment-run capture_status is invalid.")
    capture_kind = identity_bindings.get("capture_kind")
    if capture_kind is not None and capture_kind not in _CAPTURE_KINDS:
        raise EvidenceStorageError("Experiment-run capture_kind is invalid.")
    evidence_status = identity_bindings.get("evidence_status")
    if evidence_status not in {None, "protocol-valid", "capture-invalid"}:
        raise EvidenceStorageError("Experiment-run evidence_status is invalid.")
    if evidence_status == "protocol-valid" and capture_status != "complete":
        raise EvidenceStorageError(
            "Protocol-valid experiment evidence cannot be a failed capture."
        )

    if capture_status == "failed":
        _require_absent_roles(
            _COMMAND_CORE_ROLES | _MANAGED_CORE_ROLES,
            entries,
            profile="failed",
        )
        _validate_capture_failure_payload(
            root,
            protected_artifact_id=protected_artifact_id,
            identity_bindings=identity_bindings,
            entries=entries,
            required_role_bindings=required_role_bindings,
        )
        return

    _parse_available_core_payloads(root, entries)
    _require_absent_roles({"capture-failure"}, entries, profile="successful")
    present_roles = {str(entry.get("role")) for entry in entries}
    has_command = bool(present_roles & (_COMMAND_CORE_ROLES - {"event-ledger"}))
    has_managed = bool(present_roles & (_MANAGED_CORE_ROLES - {"event-ledger"}))
    if has_command == has_managed:
        raise EvidenceStorageError(
            "Experiment-run evidence does not identify one complete capture profile."
        )
    managed_job_count = sum(entry.get("role") == "job-log" for entry in entries)
    inferred_kind = (
        "command"
        if has_command
        else "managed-sequence"
        if managed_job_count > 1
        else "managed-job"
    )
    if capture_kind is not None:
        if capture_kind == "command" and not has_command:
            raise EvidenceStorageError(
                "Experiment-run capture_kind contradicts its evidence roles."
            )
        if capture_kind != "command" and not has_managed:
            raise EvidenceStorageError(
                "Experiment-run capture_kind contradicts its evidence roles."
            )
        inferred_kind = capture_kind

    _require_complete_role_binding(
        "event-ledger",
        bindings=required_role_bindings,
        entries=entries,
        minimum=1,
        maximum=1,
    )
    telemetry_count = sum(entry.get("role") == "telemetry" for entry in entries)
    if telemetry_count:
        _require_complete_role_binding(
            "telemetry",
            bindings=required_role_bindings,
            entries=entries,
            minimum=1,
            maximum=1,
        )

    if inferred_kind == "command":
        _require_absent_roles(
            _MANAGED_CORE_ROLES - {"event-ledger"},
            entries,
            profile="command",
        )
        for role in ("command-record", "command-output"):
            _require_complete_role_binding(
                role,
                bindings=required_role_bindings,
                entries=entries,
                minimum=1,
                maximum=1,
            )
        if evidence_status == "protocol-valid":
            _validate_protocol_valid_payloads(
                root,
                identity_bindings=identity_bindings,
                source_bindings=source_bindings,
                entries=entries,
                required_role_bindings=required_role_bindings,
                capture_kind=inferred_kind,
            )
        return

    _require_absent_roles(
        _COMMAND_CORE_ROLES - {"event-ledger"}, entries, profile="managed"
    )
    _require_complete_role_binding(
        "sequence-summary",
        bindings=required_role_bindings,
        entries=entries,
        minimum=1,
        maximum=1,
    )
    job_log_ids = _require_complete_role_binding(
        "job-log",
        bindings=required_role_bindings,
        entries=entries,
        minimum=0,
        maximum=1 if inferred_kind == "managed-job" else None,
    )
    terminal_ids = _require_complete_role_binding(
        "terminal-job-record",
        bindings=required_role_bindings,
        entries=entries,
        minimum=0,
        maximum=1 if inferred_kind == "managed-job" else None,
    )
    observed_ids = _require_complete_role_binding(
        "last-observed-job-record",
        bindings=required_role_bindings,
        entries=entries,
        minimum=0,
        maximum=1 if inferred_kind == "managed-job" else None,
    )
    submission_ids = _require_complete_role_binding(
        "action-submission-record",
        bindings=required_role_bindings,
        entries=entries,
        minimum=0,
        maximum=1 if inferred_kind == "managed-job" else None,
    )
    if not job_log_ids and not submission_ids:
        raise EvidenceStorageError(
            "Managed experiment-run capture requires retained evidence for at least "
            "one attempted action."
        )
    if inferred_kind == "managed-job" and len(job_log_ids) + len(submission_ids) != 1:
        raise EvidenceStorageError(
            "Managed-job capture requires exactly one submitted job or admission "
            "record."
        )
    if len(terminal_ids) + len(observed_ids) != len(job_log_ids):
        raise EvidenceStorageError(
            "Managed experiment-run capture requires one terminal or last-observed "
            "job record for every complete job log."
        )
    if evidence_status == "protocol-valid":
        _validate_protocol_valid_payloads(
            root,
            identity_bindings=identity_bindings,
            source_bindings=source_bindings,
            entries=entries,
            required_role_bindings=required_role_bindings,
            capture_kind=inferred_kind,
        )


_ExperimentSealLockEpoch = tuple[
    Path,
    int,
    tuple[int, int, int, int, int],
    int,
    tuple[int, int, int, int, int, int],
    int,
    list[tuple[Path, tuple[int, int]]],
]


def _experiment_root_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _experiment_lock_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
    )


def _validate_experiment_root(metadata: os.stat_result, path: Path) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise EvidenceStorageError("The experiment vault must remain a directory.")
    _require_owner(metadata, path)
    _require_mode(metadata, 0o700, path)


def _validate_experiment_lock(metadata: os.stat_result, path: Path) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise EvidenceStorageError(
            "The experiment-seal lock must be a regular non-hardlinked file."
        )
    _require_owner(metadata, path)
    _require_mode(metadata, 0o600, path)


def _require_experiment_lock_epoch(epoch: _ExperimentSealLockEpoch) -> None:
    (
        root,
        root_descriptor,
        root_identity,
        lock_descriptor,
        lock_identity,
        owner_thread,
        _created_controls,
    ) = epoch
    if threading.get_ident() != owner_thread:
        raise _ExperimentSealEpochError(
            "The experiment-seal lock cannot cross its owning thread."
        )
    lock_path = root / _EXPERIMENT_SEAL_LOCK_NAME
    try:
        opened_root = os.fstat(root_descriptor)
        path_root = root.lstat()
        opened_lock = os.fstat(lock_descriptor)
        path_lock = os.stat(
            _EXPERIMENT_SEAL_LOCK_NAME,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(path_root.st_mode):
            raise EvidenceStorageError("The experiment vault cannot become a symlink.")
        _validate_experiment_root(opened_root, root)
        _validate_experiment_root(path_root, root)
        _validate_experiment_lock(opened_lock, lock_path)
        _validate_experiment_lock(path_lock, lock_path)
    except (OSError, PermissionError, EvidenceStorageError) as error:
        raise _ExperimentSealEpochError(
            "The experiment-seal lock epoch is no longer available."
        ) from error
    if (
        _experiment_root_identity(opened_root) != root_identity
        or _experiment_root_identity(path_root) != root_identity
        or _experiment_lock_identity(opened_lock) != lock_identity
        or _experiment_lock_identity(path_lock) != lock_identity
    ):
        raise _ExperimentSealEpochError(
            "The experiment vault or seal-lock identity changed during sealing."
        )


def _rollback_experiment_epoch_controls(epoch: _ExperimentSealLockEpoch) -> None:
    """Remove only control-file inodes created under one invalid lock epoch."""

    rollback_errors: list[BaseException] = []
    for path, identity in reversed(epoch[6]):
        try:
            _unlink_created_control_file(path, identity, missing_ok=True)
        except BaseException as error:
            rollback_errors.append(error)
    if rollback_errors:
        raise ArtifactIntegrityError(
            "An invalid experiment-seal epoch could not be rolled back safely."
        ) from rollback_errors[0]


@contextmanager
def _locked_experiment_root(root: Path) -> Iterator[_ExperimentSealLockEpoch]:
    """Serialize seals under one pinned vault and lock-file epoch."""

    with _EXPERIMENT_SEAL_LOCK:
        protected_root = _require_private_directory(root).resolve()
        root_before = protected_root.lstat()
        _validate_experiment_root(root_before, protected_root)
        root_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        root_descriptor = os.open(protected_root, root_flags)
        lock_descriptor: int | None = None
        root_locked = False
        lock_locked = False
        try:
            try:
                opened_root = os.fstat(root_descriptor)
                _validate_experiment_root(opened_root, protected_root)
                root_identity = _experiment_root_identity(root_before)
                if _experiment_root_identity(opened_root) != root_identity:
                    raise EvidenceStorageError(
                        "The experiment vault changed while its seal lock was acquired."
                    )
                if fcntl is not None:
                    fcntl.flock(root_descriptor, fcntl.LOCK_EX)
                    root_locked = True
                lock_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
                lock_flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    lock_descriptor = os.open(
                        _EXPERIMENT_SEAL_LOCK_NAME,
                        lock_flags,
                        0o600,
                        dir_fd=root_descriptor,
                    )
                    os.fchmod(lock_descriptor, 0o600)
                    os.fsync(root_descriptor)
                except FileExistsError:
                    lock_descriptor = os.open(
                        _EXPERIMENT_SEAL_LOCK_NAME,
                        os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=root_descriptor,
                    )
                lock_metadata = os.fstat(lock_descriptor)
                _validate_experiment_lock(
                    lock_metadata,
                    protected_root / _EXPERIMENT_SEAL_LOCK_NAME,
                )
                lock_identity = _experiment_lock_identity(lock_metadata)
                if fcntl is not None:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
                    lock_locked = True
                epoch: _ExperimentSealLockEpoch = (
                    protected_root,
                    root_descriptor,
                    root_identity,
                    lock_descriptor,
                    lock_identity,
                    threading.get_ident(),
                    [],
                )
                _require_experiment_lock_epoch(epoch)
            except OSError as error:
                raise EvidenceStorageError(
                    "The experiment-seal lock could not be secured."
                ) from error

            try:
                yield epoch
            except BaseException as body_error:
                try:
                    _require_experiment_lock_epoch(epoch)
                except _ExperimentSealEpochError as epoch_error:
                    _rollback_experiment_epoch_controls(epoch)
                    raise epoch_error from body_error
                raise
            else:
                try:
                    _require_experiment_lock_epoch(epoch)
                except _ExperimentSealEpochError:
                    _rollback_experiment_epoch_controls(epoch)
                    raise
        finally:
            if lock_descriptor is not None:
                if lock_locked and fcntl is not None:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            if root_locked and fcntl is not None:
                fcntl.flock(root_descriptor, fcntl.LOCK_UN)
            os.close(root_descriptor)


def _assert_experiment_run_is_unsealed_elsewhere(
    root: Path, current: Path, experiment_run_id: str
) -> None:
    for candidate in sorted(root.iterdir()):
        if candidate == current or candidate.name == _EXPERIMENT_SEAL_LOCK_NAME:
            continue
        try:
            candidate_metadata = candidate.lstat()
        except OSError as error:
            raise ArtifactIntegrityError(
                "The experiment vault changed during duplicate-seal verification."
            ) from error
        if stat.S_ISLNK(candidate_metadata.st_mode):
            raise ArtifactIntegrityError(
                "The experiment vault cannot contain sibling symlinks while sealing."
            )
        if not stat.S_ISDIR(candidate_metadata.st_mode):
            continue
        seal_path = candidate / SEAL_NAME
        try:
            seal_path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ArtifactIntegrityError(
                "A sibling experiment completion marker is unavailable."
            ) from error
        _require_private_directory(candidate)
        _regular_metadata(seal_path)
        manifest, _ = _load_canonical_record(
            candidate / RAW_MANIFEST_NAME, RAW_MANIFEST_SCHEMA
        )
        if manifest.get("record_kind") != "experiment-run":
            continue
        identity = manifest.get("identity_bindings")
        if not isinstance(identity, dict):
            raise ArtifactIntegrityError(
                "A sealed experiment sibling has invalid identity bindings."
            )
        if identity.get("experiment_run_id") == experiment_run_id:
            raise EvidenceStorageError(
                "The experiment_run_id already has a sealed artifact in this vault."
            )


class RawArtifactWriter:
    """Build one fresh protected artifact and seal it exactly once."""

    def __init__(
        self,
        directory: Path,
        *,
        protected_artifact_id: str | None = None,
        record_kind: str,
        identity_bindings: Mapping[str, Any],
        capture_tool: Mapping[str, Any],
        source_bindings: Mapping[str, Any],
        retention_policy_id: str = RETENTION_POLICY_ID,
        provisional_retain_not_before_utc: str,
        required_role_bindings: Mapping[str, str | Sequence[str]] | None = None,
    ) -> None:
        if record_kind not in {"experiment-run", "legacy-recovery"}:
            raise EvidenceStorageError("Raw artifact record_kind is invalid.")
        artifact_id = protected_artifact_id or new_opaque_id("artifact")
        _require_identifier(artifact_id, label="protected_artifact_id")
        identity = dict(identity_bindings)
        tool = dict(capture_tool)
        sources = dict(source_bindings)
        if not isinstance(retention_policy_id, str) or not retention_policy_id:
            raise EvidenceStorageError("retention_policy_id is required.")
        _parse_utc(
            provisional_retain_not_before_utc,
            label="provisional_retain_not_before_utc",
        )
        self.directory = _create_fresh_private_directory(directory)
        self.protected_artifact_id = artifact_id
        self.record_kind = record_kind
        self.identity_bindings = identity
        self.capture_tool = tool
        self.source_bindings = sources
        self.retention_policy_id = retention_policy_id
        self.provisional_retain_not_before_utc = provisional_retain_not_before_utc
        self.required_role_bindings = dict(required_role_bindings or {})
        self._entries: dict[str, dict[str, Any]] = {}
        self._entry_ids: set[str] = set()
        self._sealed = False

    def _require_open(self) -> None:
        if self._sealed or (self.directory / SEAL_NAME).exists():
            raise FileExistsError("The protected artifact is already sealed.")
        if (self.directory / RAW_MANIFEST_NAME).exists():
            raise FileExistsError(
                "The protected artifact already has a raw manifest and cannot be resealed."
            )

    def _entry(
        self,
        *,
        relative_path: str,
        role: str,
        media_type: str,
        size_bytes: int,
        digest: str,
        entry_id: str | None,
        captured_at_utc: str | None,
    ) -> dict[str, Any]:
        if not isinstance(role, str) or not role:
            raise EvidenceStorageError("Payload role must be a non-empty string.")
        if not isinstance(media_type, str) or not media_type:
            raise EvidenceStorageError("Payload media_type must be a non-empty string.")
        identifier = entry_id or _new_id("entry")
        if not isinstance(identifier, str) or not identifier:
            raise EvidenceStorageError("Payload entry_id must be a non-empty string.")
        if identifier in self._entry_ids:
            raise EvidenceStorageError(f"Duplicate payload entry_id: {identifier}")
        captured = captured_at_utc or utc_now()
        _parse_utc(captured, label="captured_at_utc")
        entry = {
            "entry_id": identifier,
            "role": role,
            "relative_path": relative_path,
            "media_type": media_type,
            "size_bytes": size_bytes,
            "sha256": _require_sha256(digest, label="payload digest"),
            "captured_at_utc": captured,
        }
        self._entry_ids.add(identifier)
        self._entries[relative_path] = entry
        return dict(entry)

    def copy_payload(
        self,
        source: Path,
        relative_path: str,
        *,
        role: str,
        media_type: str = "application/octet-stream",
        entry_id: str | None = None,
        captured_at_utc: str | None = None,
    ) -> dict[str, Any]:
        self._require_open()
        normalized, destination = _destination(self.directory, relative_path)
        if normalized in self._entries:
            raise FileExistsError(f"Payload path is already captured: {normalized}")
        _ensure_payload_parent(self.directory, normalized)
        size, digest = _copy_file(source, destination)
        try:
            return self._entry(
                relative_path=normalized,
                role=role,
                media_type=media_type,
                size_bytes=size,
                digest=digest,
                entry_id=entry_id,
                captured_at_utc=captured_at_utc,
            )
        except BaseException:
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
            raise

    def copy_payload_from_descriptor(
        self,
        source_descriptor: int,
        source_path: Path,
        source_fingerprint: tuple[int, ...],
        relative_path: str,
        *,
        role: str,
        media_type: str = "application/octet-stream",
        entry_id: str | None = None,
        captured_at_utc: str | None = None,
    ) -> dict[str, Any]:
        """Capture an already-open file without reopening its pathname."""

        self._require_open()
        normalized, destination = _destination(self.directory, relative_path)
        if normalized in self._entries:
            raise FileExistsError(f"Payload path is already captured: {normalized}")
        _ensure_payload_parent(self.directory, normalized)
        size, digest = _copy_pinned_descriptor(
            source_descriptor,
            source_path,
            destination,
            source_fingerprint,
        )
        try:
            return self._entry(
                relative_path=normalized,
                role=role,
                media_type=media_type,
                size_bytes=size,
                digest=digest,
                entry_id=entry_id,
                captured_at_utc=captured_at_utc,
            )
        except BaseException:
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
            raise

    def write_payload(
        self,
        payload: bytes,
        relative_path: str,
        *,
        role: str,
        media_type: str = "application/octet-stream",
        entry_id: str | None = None,
        captured_at_utc: str | None = None,
    ) -> dict[str, Any]:
        self._require_open()
        if not isinstance(payload, bytes):
            raise TypeError("Protected payloads must be bytes.")
        normalized, destination = _destination(self.directory, relative_path)
        if normalized in self._entries:
            raise FileExistsError(f"Payload path is already captured: {normalized}")
        _ensure_payload_parent(self.directory, normalized)
        _write_exclusive_bytes(destination, payload)
        try:
            return self._entry(
                relative_path=normalized,
                role=role,
                media_type=media_type,
                size_bytes=len(payload),
                digest=sha256_bytes(payload),
                entry_id=entry_id,
                captured_at_utc=captured_at_utc,
            )
        except BaseException:
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
            raise

    def seal(self) -> dict[str, Any]:
        self._require_open()
        _, actual_files = _walk_artifact(self.directory)
        actual_relative = {
            path.relative_to(self.directory).as_posix() for path in actual_files
        }
        expected_relative = set(self._entries)
        if actual_relative != expected_relative:
            raise ArtifactIntegrityError(
                "Raw artifact payload set differs from the registered capture inventory."
            )
        entries = [self._entries[path] for path in sorted(self._entries)]
        payload_fingerprints = {
            str(entry["relative_path"]): (
                int(entry["size_bytes"]),
                str(entry["sha256"]),
            )
            for entry in entries
        }
        for entry in entries:
            path = self.directory.joinpath(*PurePosixPath(entry["relative_path"]).parts)
            size, digest = _hash_pinned_evidence_file(path)
            if size != entry["size_bytes"] or digest != entry["sha256"]:
                raise ArtifactIntegrityError(
                    f"Payload changed before sealing: {entry['relative_path']}"
                )
            _fsync_regular_file(path)

        required_role_bindings = _normalize_required_role_bindings(
            self.required_role_bindings, entries
        )
        if self.record_kind == "experiment-run":
            _validate_experiment_run_completeness(
                self.directory,
                protected_artifact_id=self.protected_artifact_id,
                identity_bindings=self.identity_bindings,
                source_bindings=self.source_bindings,
                entries=entries,
                required_role_bindings=required_role_bindings,
            )
        manifest = {
            "schema_version": RAW_MANIFEST_SCHEMA,
            "protected_artifact_id": self.protected_artifact_id,
            "record_kind": self.record_kind,
            "identity_bindings": self.identity_bindings,
            "capture_tool": self.capture_tool,
            "source_bindings": self.source_bindings,
            "retention_policy_id": self.retention_policy_id,
            "provisional_retain_not_before_utc": (
                self.provisional_retain_not_before_utc
            ),
            "files": entries,
            "file_count": len(entries),
            "total_bytes": sum(entry["size_bytes"] for entry in entries),
            "required_role_bindings": required_role_bindings,
            "completion_marker": SEAL_NAME,
        }
        manifest = validate_record(manifest, expected_schema=RAW_MANIFEST_SCHEMA)
        manifest_bytes = canonical_json_bytes(manifest)
        guard = (
            _locked_experiment_root(self.directory.parent)
            if self.record_kind == "experiment-run"
            else nullcontext()
        )
        manifest_path = self.directory / RAW_MANIFEST_NAME
        seal_path = self.directory / SEAL_NAME
        manifest_identity: tuple[int, int] | None = None
        seal_identity: tuple[int, int] | None = None
        try:
            with guard as experiment_epoch:
                self._require_open()
                if self.record_kind == "experiment-run":
                    _assert_experiment_run_is_unsealed_elsewhere(
                        self.directory.parent,
                        self.directory,
                        str(self.identity_bindings["experiment_run_id"]),
                    )
                _assert_exact_artifact_inventory(self.directory, payload_fingerprints)
                manifest_identity = _write_exclusive_control_bytes(
                    manifest_path, manifest_bytes
                )
                if experiment_epoch is not None:
                    experiment_epoch[6].append((manifest_path, manifest_identity))
                _fsync_directory(self.directory)

                _assert_exact_artifact_inventory(
                    self.directory,
                    {
                        **payload_fingerprints,
                        RAW_MANIFEST_NAME: (
                            len(manifest_bytes),
                            sha256_bytes(manifest_bytes),
                        ),
                    },
                )

                seal = {
                    "schema_version": RAW_SEAL_SCHEMA,
                    "protected_artifact_id": self.protected_artifact_id,
                    "raw_manifest_sha256": sha256_bytes(manifest_bytes),
                    "raw_manifest_size_bytes": len(manifest_bytes),
                    "sealed_at_utc": utc_now(),
                }
                seal = validate_record(seal, expected_schema=RAW_SEAL_SCHEMA)
                seal_identity = _write_exclusive_control_bytes(
                    seal_path,
                    canonical_json_bytes(seal),
                )
                if experiment_epoch is not None:
                    experiment_epoch[6].append((seal_path, seal_identity))
                _fsync_directory(self.directory)
                self._sealed = True
                result = verify_sealed_artifact(self.directory)
        except _ExperimentSealEpochError:
            rollback_errors: list[BaseException] = []
            for path, identity in (
                (seal_path, seal_identity),
                (manifest_path, manifest_identity),
            ):
                if identity is None:
                    continue
                try:
                    _unlink_created_control_file(path, identity, missing_ok=True)
                except BaseException as error:
                    rollback_errors.append(error)
            self._sealed = False
            if rollback_errors:
                raise ArtifactIntegrityError(
                    "An invalid experiment-seal epoch could not be rolled back safely."
                ) from rollback_errors[0]
            raise
        return result


def _manifest_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_entries = manifest.get("files")
    if not isinstance(raw_entries, list):
        raise ArtifactIntegrityError("Raw manifest files must be a list.")
    entries: list[dict[str, Any]] = []
    paths: list[str] = []
    identifiers: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ArtifactIntegrityError("Raw manifest file entries must be objects.")
        required = {
            "entry_id",
            "role",
            "relative_path",
            "media_type",
            "size_bytes",
            "sha256",
            "captured_at_utc",
        }
        if set(raw_entry) != required:
            raise ArtifactIntegrityError("Raw manifest file entry fields are invalid.")
        relative = validate_safe_relative_path(raw_entry["relative_path"])
        if relative in {RAW_MANIFEST_NAME, SEAL_NAME}:
            raise ArtifactIntegrityError(
                "Raw manifest recursively inventories its metadata."
            )
        entry_id = raw_entry["entry_id"]
        if not isinstance(entry_id, str) or not entry_id or entry_id in identifiers:
            raise ArtifactIntegrityError("Raw manifest entry IDs must be unique.")
        identifiers.add(entry_id)
        if not isinstance(raw_entry["role"], str) or not raw_entry["role"]:
            raise ArtifactIntegrityError(
                "Raw manifest roles must be non-empty strings."
            )
        if not isinstance(raw_entry["media_type"], str) or not raw_entry["media_type"]:
            raise ArtifactIntegrityError(
                "Raw manifest media types must be non-empty strings."
            )
        size = raw_entry["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArtifactIntegrityError(
                "Raw manifest sizes must be non-negative bytes."
            )
        _require_sha256(raw_entry["sha256"], label="raw manifest file digest")
        _parse_utc(raw_entry["captured_at_utc"], label="captured_at_utc")
        paths.append(relative)
        entries.append(dict(raw_entry))
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ArtifactIntegrityError(
            "Raw manifest paths must be sorted, normalized, and unique."
        )
    return entries


def verify_sealed_artifact(directory: Path) -> dict[str, Any]:
    """Deeply verify one sealed artifact, including exact file set and permissions."""

    if directory.is_symlink():
        raise ArtifactIntegrityError("A sealed artifact root cannot be a symlink.")
    root = directory.resolve(strict=True)
    _, actual_files = _walk_artifact(root)
    manifest, manifest_bytes = _load_canonical_record(
        root / RAW_MANIFEST_NAME, RAW_MANIFEST_SCHEMA
    )
    seal, seal_bytes = _load_canonical_record(root / SEAL_NAME, RAW_SEAL_SCHEMA)
    entries = _manifest_entries(manifest)
    entry_paths = {entry["relative_path"] for entry in entries}
    expected_files = entry_paths | {RAW_MANIFEST_NAME, SEAL_NAME}
    actual_relative = {path.relative_to(root).as_posix() for path in actual_files}
    if actual_relative != expected_files:
        raise ArtifactIntegrityError(
            "Sealed artifact contains a missing or unexpected file."
        )
    if manifest.get("file_count") != len(entries) or manifest.get("total_bytes") != sum(
        entry["size_bytes"] for entry in entries
    ):
        raise ArtifactIntegrityError("Raw manifest totals are inconsistent.")
    if manifest.get("completion_marker") != SEAL_NAME:
        raise ArtifactIntegrityError("Raw manifest has the wrong completion marker.")
    required_role_bindings = _normalize_required_role_bindings(
        manifest.get("required_role_bindings", {}), entries
    )
    if manifest.get("record_kind") == "experiment-run":
        identity_bindings = manifest.get("identity_bindings")
        if not isinstance(identity_bindings, dict):
            raise ArtifactIntegrityError(
                "Experiment-run identity bindings must be an object."
            )
        try:
            _validate_experiment_run_completeness(
                root,
                protected_artifact_id=str(manifest.get("protected_artifact_id")),
                identity_bindings=identity_bindings,
                source_bindings=manifest["source_bindings"],
                entries=entries,
                required_role_bindings=required_role_bindings,
            )
        except EvidenceStorageError as error:
            raise ArtifactIntegrityError(
                "Sealed experiment-run evidence is semantically incomplete."
            ) from error
    for entry in entries:
        path = root.joinpath(*PurePosixPath(entry["relative_path"]).parts)
        size, digest = _hash_pinned_evidence_file(path)
        if size != entry["size_bytes"] or digest != entry["sha256"]:
            raise ArtifactIntegrityError(
                f"Sealed payload does not match its manifest: {entry['relative_path']}"
            )
    manifest_digest = sha256_bytes(manifest_bytes)
    if (
        seal.get("protected_artifact_id") != manifest.get("protected_artifact_id")
        or seal.get("raw_manifest_sha256") != manifest_digest
        or seal.get("raw_manifest_size_bytes") != len(manifest_bytes)
    ):
        raise ArtifactIntegrityError("Raw seal does not bind the exact raw manifest.")
    _parse_utc(seal.get("sealed_at_utc"), label="sealed_at_utc")
    _assert_exact_artifact_inventory(
        root,
        {
            **{
                str(entry["relative_path"]): (
                    int(entry["size_bytes"]),
                    str(entry["sha256"]),
                )
                for entry in entries
            },
            RAW_MANIFEST_NAME: (len(manifest_bytes), manifest_digest),
            SEAL_NAME: (len(seal_bytes), sha256_bytes(seal_bytes)),
        },
    )
    return {
        "protected_artifact_id": manifest["protected_artifact_id"],
        "raw_manifest_sha256": manifest_digest,
        "raw_manifest_size_bytes": len(manifest_bytes),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest": manifest,
        "seal": seal,
    }


def _all_file_fingerprints(root: Path) -> dict[str, tuple[int, str]]:
    _, files = _walk_artifact(root)
    return {
        path.relative_to(root).as_posix(): _hash_pinned_evidence_file(path)
        for path in files
    }


def _descriptor_relative_file_fingerprint(
    parent_descriptor: int,
    name: str,
    display_path: Path,
    before: os.stat_result,
) -> tuple[int, str]:
    """Hash one directory entry while pinning its descriptor and pathname."""

    _validate_open_evidence_file(before, display_path, mode=0o600)
    expected_fingerprint = _evidence_file_fingerprint(before)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ArtifactIntegrityError(
            f"Evidence file could not be opened relative to its pinned root: {display_path}"
        ) from error
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        _validate_open_evidence_file(opened, display_path, mode=0o600)
        if _evidence_file_fingerprint(opened) != expected_fingerprint:
            raise ArtifactIntegrityError(
                f"Evidence file identity changed while it was opened: {display_path}"
            )
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            digest.update(block)
        finished = os.fstat(descriptor)
        _validate_open_evidence_file(finished, display_path, mode=0o600)
        if _evidence_file_fingerprint(finished) != expected_fingerprint:
            raise ArtifactIntegrityError(
                f"Evidence file changed while it was hashed: {display_path}"
            )
    finally:
        os.close(descriptor)
    try:
        path_after = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ArtifactIntegrityError(
            f"Evidence file path disappeared while it was hashed: {display_path}"
        ) from error
    _validate_open_evidence_file(path_after, display_path, mode=0o600)
    if _evidence_file_fingerprint(path_after) != expected_fingerprint:
        raise ArtifactIntegrityError(
            f"Evidence file path changed while it was hashed: {display_path}"
        )
    return total, digest.hexdigest()


def _descriptor_relative_artifact_fingerprints(
    descriptor: int,
    display_root: Path,
    expected_root_fingerprint: tuple[int, ...],
) -> dict[str, tuple[int, str]]:
    """Inventory and hash an artifact entirely below one pinned root descriptor."""

    fingerprints: dict[str, tuple[int, str]] = {}

    def walk_directory(
        current_descriptor: int,
        display_directory: Path,
        relative_directory: PurePosixPath,
        expected_directory_fingerprint: tuple[int, ...],
    ) -> None:
        opened = os.fstat(current_descriptor)
        _validate_open_private_directory(opened, display_directory)
        if _private_directory_fingerprint(opened) != expected_directory_fingerprint:
            raise ArtifactIntegrityError(
                "Protected evidence directory identity changed during inventory: "
                f"{display_directory}"
            )
        try:
            names_before = sorted(os.listdir(current_descriptor))
        except OSError as error:
            raise ArtifactIntegrityError(
                f"Protected evidence directory could not be inventoried: {display_directory}"
            ) from error
        for name in names_before:
            display_path = display_directory / name
            relative_path = relative_directory / name
            try:
                entry_metadata = os.stat(
                    name,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ArtifactIntegrityError(
                    f"Evidence entry disappeared during inventory: {display_path}"
                ) from error
            if stat.S_ISDIR(entry_metadata.st_mode):
                _validate_open_private_directory(entry_metadata, display_path)
                expected_child_fingerprint = _private_directory_fingerprint(
                    entry_metadata
                )
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    child_descriptor = os.open(
                        name,
                        flags,
                        dir_fd=current_descriptor,
                    )
                except OSError as error:
                    raise ArtifactIntegrityError(
                        "Protected evidence directory could not be opened relative "
                        f"to its pinned root: {display_path}"
                    ) from error
                try:
                    child_opened = os.fstat(child_descriptor)
                    _validate_open_private_directory(child_opened, display_path)
                    if (
                        _private_directory_fingerprint(child_opened)
                        != expected_child_fingerprint
                    ):
                        raise ArtifactIntegrityError(
                            "Protected evidence directory identity changed while it "
                            f"was opened: {display_path}"
                        )
                    walk_directory(
                        child_descriptor,
                        display_path,
                        relative_path,
                        expected_child_fingerprint,
                    )
                finally:
                    os.close(child_descriptor)
                try:
                    child_after = os.stat(
                        name,
                        dir_fd=current_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise ArtifactIntegrityError(
                        "Protected evidence directory disappeared during inventory: "
                        f"{display_path}"
                    ) from error
                _validate_open_private_directory(child_after, display_path)
                if (
                    _private_directory_fingerprint(child_after)
                    != expected_child_fingerprint
                ):
                    raise ArtifactIntegrityError(
                        "Protected evidence directory path changed during inventory: "
                        f"{display_path}"
                    )
            elif stat.S_ISREG(entry_metadata.st_mode):
                fingerprints[relative_path.as_posix()] = (
                    _descriptor_relative_file_fingerprint(
                        current_descriptor,
                        name,
                        display_path,
                        entry_metadata,
                    )
                )
            else:
                raise ArtifactIntegrityError(
                    f"Evidence artifacts cannot contain links or special files: {display_path}"
                )
        try:
            names_after = sorted(os.listdir(current_descriptor))
        except OSError as error:
            raise ArtifactIntegrityError(
                f"Protected evidence directory could not be reinventoried: {display_directory}"
            ) from error
        if names_after != names_before:
            raise ArtifactIntegrityError(
                f"Protected evidence inventory changed while it was hashed: {display_directory}"
            )
        finished = os.fstat(current_descriptor)
        _validate_open_private_directory(finished, display_directory)
        if _private_directory_fingerprint(finished) != expected_directory_fingerprint:
            raise ArtifactIntegrityError(
                f"Protected evidence directory changed during inventory: {display_directory}"
            )

    walk_directory(
        descriptor,
        display_root,
        PurePosixPath(),
        expected_root_fingerprint,
    )
    return fingerprints


def _require_verified_descriptor_inventory(
    result: Mapping[str, Any],
    fingerprints: Mapping[str, tuple[int, str]],
    root: Path,
) -> None:
    expected = {
        str(entry["relative_path"]): (
            int(entry["size_bytes"]),
            str(entry["sha256"]),
        )
        for entry in result["manifest"]["files"]
    }
    expected[RAW_MANIFEST_NAME] = (
        int(result["raw_manifest_size_bytes"]),
        str(result["raw_manifest_sha256"]),
    )
    seal_bytes = canonical_json_bytes(result["seal"])
    expected[SEAL_NAME] = (len(seal_bytes), sha256_bytes(seal_bytes))
    if dict(fingerprints) != expected:
        raise ArtifactIntegrityError(
            f"Pinned protected evidence does not match semantic verification: {root}"
        )


def verify_copy_equality(first: Path, second: Path) -> dict[str, Any]:
    """Require two independently stored artifacts to contain byte-identical files."""

    first_root = first.expanduser()
    second_root = second.expanduser()
    with _pinned_private_directory(first_root) as (
        first_descriptor,
        first_root_fingerprint,
    ):
        with _pinned_private_directory(second_root) as (
            second_descriptor,
            second_root_fingerprint,
        ):
            if first_root_fingerprint[:2] == second_root_fingerprint[:2]:
                raise ArtifactIntegrityError(
                    "Protected evidence copies must have distinct root identities."
                )

            def recheck_roots() -> None:
                _recheck_pinned_private_directory(
                    first_root,
                    first_descriptor,
                    first_root_fingerprint,
                )
                _recheck_pinned_private_directory(
                    second_root,
                    second_descriptor,
                    second_root_fingerprint,
                )

            recheck_roots()
            first_result = verify_sealed_artifact(first_root)
            recheck_roots()
            second_result = verify_sealed_artifact(second_root)
            recheck_roots()
            first_fingerprints = _descriptor_relative_artifact_fingerprints(
                first_descriptor,
                first_root,
                first_root_fingerprint,
            )
            recheck_roots()
            second_fingerprints = _descriptor_relative_artifact_fingerprints(
                second_descriptor,
                second_root,
                second_root_fingerprint,
            )
            recheck_roots()
            _require_verified_descriptor_inventory(
                first_result,
                first_fingerprints,
                first_root,
            )
            _require_verified_descriptor_inventory(
                second_result,
                second_fingerprints,
                second_root,
            )
            if (
                first_result["protected_artifact_id"]
                != second_result["protected_artifact_id"]
                or first_fingerprints != second_fingerprints
            ):
                raise ArtifactIntegrityError(
                    "Protected evidence copies are not byte-identical."
                )
            return {
                "verification_result": "passed",
                "protected_artifact_id": first_result["protected_artifact_id"],
                "raw_manifest_sha256": first_result["raw_manifest_sha256"],
                "raw_manifest_size_bytes": first_result["raw_manifest_size_bytes"],
                "file_count": first_result["file_count"],
                "total_bytes": first_result["total_bytes"],
                "stored_file_count": len(first_fingerprints),
                "stored_total_bytes": sum(
                    size for size, _ in first_fingerprints.values()
                ),
            }


def _copy_exact_regular_file(source: Path, destination: Path) -> None:
    expected_size, expected_digest = _hash_pinned_evidence_file(source)
    observed_size, observed_digest = _copy_file(source, destination)
    if observed_size != expected_size or observed_digest != expected_digest:
        raise ArtifactIntegrityError(f"Evidence copy changed in transit: {source.name}")


def copy_sealed_artifact(source: Path, destination: Path) -> dict[str, Any]:
    """Copy a sealed artifact into a fresh directory and verify exact equality."""

    source_result = verify_sealed_artifact(source)
    source_root = source.resolve(strict=True)
    destination_parent = destination.expanduser().parent.resolve(strict=True)
    destination_candidate = destination_parent / destination.name
    if (
        source_root == destination_candidate
        or source_root in destination_candidate.parents
    ):
        raise EvidenceStorageError(
            "A protected evidence copy cannot be created inside its sealed source."
        )
    destination_root = _create_fresh_private_directory(destination)
    manifest = source_result["manifest"]
    ordered = [entry["relative_path"] for entry in manifest["files"]]
    ordered.append(RAW_MANIFEST_NAME)
    for relative in ordered:
        if relative == RAW_MANIFEST_NAME:
            destination_path = destination_root / relative
        else:
            _, destination_path = _destination(destination_root, relative)
            _ensure_payload_parent(destination_root, relative)
        source_path = source_root.joinpath(*PurePosixPath(relative).parts)
        _copy_exact_regular_file(source_path, destination_path)
    _fsync_directory(destination_root)
    _copy_exact_regular_file(source_root / SEAL_NAME, destination_root / SEAL_NAME)
    for directory in reversed(_walk_artifact(destination_root)[0]):
        _fsync_directory(directory)
    return verify_copy_equality(source_root, destination_root)


def _safe_observed_manifest_digest(source: Path) -> str | None:
    manifest = source / RAW_MANIFEST_NAME
    try:
        _, digest = _hash_pinned_evidence_file(manifest)
        return digest
    except (OSError, EvidenceStorageError):
        return None


def retrieve_sealed_artifact(
    source: Path,
    destination: Path,
    *,
    source_copy_id: str,
    source_failure_domain_id: str,
    expected_raw_manifest_sha256: str,
    destination_restore_id: str | None = None,
) -> dict[str, Any]:
    """Perform a full off-host-style restore into a fresh destination and verify it."""

    _require_identifier(source_copy_id, label="source_copy_id")
    _require_identifier(source_failure_domain_id, label="source_failure_domain_id")
    restore_id = destination_restore_id or _new_id("restore")
    _require_identifier(restore_id, label="destination_restore_id")
    expected = _require_sha256(
        expected_raw_manifest_sha256, label="expected_raw_manifest_sha256"
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Retrieval destination must be fresh: {destination}")
    started_at = utc_now()
    started_ns = time.monotonic_ns()
    details: dict[str, Any] = {
        "source_copy_id": source_copy_id,
        "source_failure_domain_id": source_failure_domain_id,
        "destination_restore_id": restore_id,
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "duration_ns": None,
        "restored_file_count": 0,
        "restored_total_bytes": 0,
        "expected_raw_manifest_sha256": expected,
        "observed_raw_manifest_sha256": None,
        "mismatch_count": 1,
        "verification_result": "failed",
    }
    try:
        source_result = verify_sealed_artifact(source)
        if source_result["raw_manifest_sha256"] != expected:
            details["observed_raw_manifest_sha256"] = source_result[
                "raw_manifest_sha256"
            ]
            raise ArtifactIntegrityError(
                "Retrieval source does not match the expected raw manifest."
            )
        copy_sealed_artifact(source, destination)
        restored = verify_sealed_artifact(destination)
        restored_files = _all_file_fingerprints(destination.resolve(strict=True))
        details.update(
            restored_file_count=len(restored_files),
            restored_total_bytes=sum(size for size, _ in restored_files.values()),
            observed_raw_manifest_sha256=restored["raw_manifest_sha256"],
            mismatch_count=(0 if restored["raw_manifest_sha256"] == expected else 1),
            verification_result=(
                "passed" if restored["raw_manifest_sha256"] == expected else "failed"
            ),
        )
        if details["mismatch_count"]:
            raise ArtifactIntegrityError(
                "Retrieved artifact does not match the expected raw manifest."
            )
    except (OSError, EvidenceStorageError) as error:
        if details["observed_raw_manifest_sha256"] is None:
            details["observed_raw_manifest_sha256"] = _safe_observed_manifest_digest(
                source
            )
        details["finished_at_utc"] = utc_now()
        details["duration_ns"] = time.monotonic_ns() - started_ns
        raise RetrievalError(
            "Full protected-evidence retrieval failed.", details=details
        ) from error
    details["finished_at_utc"] = utc_now()
    details["duration_ns"] = time.monotonic_ns() - started_ns
    return details


def _normalize_available_files(
    available_files: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    paths: set[str] = set()
    entry_ids: set[str] = set()
    for raw in available_files:
        if not isinstance(raw, Mapping):
            raise EvidenceStorageError(
                "Capture-failure inventory entries must be objects."
            )
        raw_relative = raw.get("relative_path")
        if not isinstance(raw_relative, str):
            raise EvidenceStorageError(
                "Capture-failure inventory relative_path is invalid."
            )
        relative = validate_safe_relative_path(raw_relative)
        if relative in paths:
            raise EvidenceStorageError(
                "Capture-failure inventory paths must be unique."
            )
        paths.add(relative)
        required = {
            "entry_id",
            "role",
            "relative_path",
            "media_type",
            "size_bytes",
            "sha256",
            "captured_at_utc",
        }
        if set(raw) != required:
            raise EvidenceStorageError(
                "Capture-failure inventory entry fields are invalid."
            )
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise EvidenceStorageError("Capture-failure inventory size is invalid.")
        _require_sha256(digest, label="capture-failure inventory digest")
        if (
            not isinstance(raw["entry_id"], str)
            or not raw["entry_id"]
            or raw["entry_id"] in entry_ids
        ):
            raise EvidenceStorageError("Capture-failure inventory entry_id is invalid.")
        entry_ids.add(raw["entry_id"])
        if not isinstance(raw["role"], str) or not raw["role"]:
            raise EvidenceStorageError("Capture-failure inventory role is invalid.")
        if not isinstance(raw["media_type"], str) or not raw["media_type"]:
            raise EvidenceStorageError(
                "Capture-failure inventory media_type is invalid."
            )
        _parse_utc(raw["captured_at_utc"], label="captured_at_utc")
        normalized.append(
            {
                "entry_id": raw["entry_id"],
                "role": raw["role"],
                "relative_path": relative,
                "media_type": raw["media_type"],
                "size_bytes": size,
                "sha256": digest,
                "captured_at_utc": raw["captured_at_utc"],
            }
        )
    return sorted(normalized, key=lambda item: item["relative_path"])


def write_capture_failure_receipt(
    path: Path,
    *,
    protected_artifact_id: str,
    attempt_slot_id: str,
    experiment_run_id: str,
    reason_code: str,
    available_files: Iterable[Mapping[str, Any]],
    missing_fields: Iterable[str],
    recoverable_locator: str | None,
) -> dict[str, Any]:
    """Persist one no-clobber fallback receipt before it is sealed by its caller."""

    parent = ensure_private_directory(path.parent)
    target = parent / path.name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Capture-failure receipt already exists: {target}")
    inventory = _normalize_available_files(available_files)
    raw_missing = list(missing_fields)
    if any(not isinstance(item, str) or not item for item in raw_missing):
        raise EvidenceStorageError("Capture-failure missing fields are invalid.")
    missing = sorted(set(raw_missing))
    if not isinstance(reason_code, str) or not reason_code:
        raise EvidenceStorageError("Capture-failure reason_code is required.")
    if recoverable_locator is not None and (
        not isinstance(recoverable_locator, str) or not recoverable_locator
    ):
        raise EvidenceStorageError("recoverable_locator must be null or non-empty.")
    receipt = {
        "schema_version": CAPTURE_FAILURE_SCHEMA,
        "protected_artifact_id": protected_artifact_id,
        "created_at_utc": utc_now(),
        "attempt_slot_id": attempt_slot_id,
        "experiment_run_id": experiment_run_id,
        "reason_code": reason_code,
        "available_files": inventory,
        "missing_fields": missing,
        "recoverable_locator": recoverable_locator,
    }
    receipt = validate_record(receipt, expected_schema=CAPTURE_FAILURE_SCHEMA)
    _write_exclusive_bytes(target, canonical_json_bytes(receipt))
    return receipt


def write_sealed_capture_failure_artifact(
    directory: Path,
    *,
    protected_artifact_id: str,
    attempt_slot_id: str,
    experiment_run_id: str,
    reason_code: str,
    available_files: Iterable[Mapping[str, Any]],
    missing_fields: Iterable[str],
    recoverable_locator: str | None,
    capture_tool: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    provisional_retain_not_before_utc: str,
) -> dict[str, Any]:
    """Create and seal the immutable fallback required when normal capture fails."""

    inventory = _normalize_available_files(available_files)
    raw_missing = list(missing_fields)
    if any(not isinstance(item, str) or not item for item in raw_missing):
        raise EvidenceStorageError("Capture-failure missing fields are invalid.")
    missing = sorted(set(raw_missing))
    receipt = {
        "schema_version": CAPTURE_FAILURE_SCHEMA,
        "protected_artifact_id": protected_artifact_id,
        "created_at_utc": utc_now(),
        "attempt_slot_id": attempt_slot_id,
        "experiment_run_id": experiment_run_id,
        "reason_code": reason_code,
        "available_files": inventory,
        "missing_fields": missing,
        "recoverable_locator": recoverable_locator,
    }
    receipt = validate_record(receipt, expected_schema=CAPTURE_FAILURE_SCHEMA)
    entry_id = _new_id("entry")
    writer = RawArtifactWriter(
        directory,
        protected_artifact_id=protected_artifact_id,
        record_kind="experiment-run",
        identity_bindings={
            "attempt_slot_id": attempt_slot_id,
            "experiment_run_id": experiment_run_id,
            "capture_status": "failed",
        },
        capture_tool=capture_tool,
        source_bindings=source_bindings,
        provisional_retain_not_before_utc=provisional_retain_not_before_utc,
        required_role_bindings={"capture-failure": entry_id},
    )
    writer.write_payload(
        canonical_json_bytes(receipt),
        "capture-failure.json",
        role="capture-failure",
        media_type="application/json",
        entry_id=entry_id,
        captured_at_utc=receipt["created_at_utc"],
    )
    sealed = writer.seal()
    return {"receipt": receipt, **sealed}


_ReceiptLockEpoch = tuple[
    Path,
    int,
    tuple[int, int, int, int, int],
    int,
    tuple[int, int, int, int, int, int],
    int,
    object,
]


def _receipt_directory_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _receipt_lock_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
    )


def _validate_receipt_directory(metadata: os.stat_result, path: Path) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ReceiptChainError("The receipt store must remain a directory.")
    _require_owner(metadata, path)
    _require_mode(metadata, 0o700, path)


def _validate_receipt_lock(metadata: os.stat_result, path: Path) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ReceiptChainError(
            "The receipt-store lock must be a regular non-hardlinked file."
        )
    _require_owner(metadata, path)
    _require_mode(metadata, 0o600, path)


def _require_receipt_lock_epoch(epoch: _ReceiptLockEpoch) -> None:
    (
        root,
        root_descriptor,
        root_identity,
        lock_descriptor,
        lock_identity,
        owner_thread,
        _,
    ) = epoch
    if threading.get_ident() != owner_thread:
        raise ReceiptChainError(
            "The receipt transaction cannot cross its lock-owning thread."
        )
    try:
        opened_root = os.fstat(root_descriptor)
        path_root = root.lstat()
        opened_lock = os.fstat(lock_descriptor)
        path_lock = os.stat(
            ".receipts.lock",
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ReceiptChainError(
            "The locked receipt-store identity is no longer available."
        ) from error
    if stat.S_ISLNK(path_root.st_mode):
        raise ReceiptChainError("The receipt-store root cannot become a symlink.")
    _validate_receipt_directory(opened_root, root)
    _validate_receipt_directory(path_root, root)
    _validate_receipt_lock(opened_lock, root / ".receipts.lock")
    _validate_receipt_lock(path_lock, root / ".receipts.lock")
    if (
        _receipt_directory_identity(opened_root) != root_identity
        or _receipt_directory_identity(path_root) != root_identity
        or _receipt_lock_identity(opened_lock) != lock_identity
        or _receipt_lock_identity(path_lock) != lock_identity
    ):
        raise ReceiptChainError(
            "The receipt-store root or lock identity changed during the transaction."
        )


@contextmanager
def _locked_receipt_store(root: Path) -> Iterator[_ReceiptLockEpoch]:
    with _RECEIPT_LOCK:
        root = _require_private_directory(root).resolve()
        root_before = root.lstat()
        if stat.S_ISLNK(root_before.st_mode):
            raise ReceiptChainError("The receipt-store root cannot be a symlink.")
        _validate_receipt_directory(root_before, root)
        root_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        root_descriptor = os.open(root, root_flags)
        lock_descriptor: int | None = None
        locked = False
        try:
            opened_root = os.fstat(root_descriptor)
            _validate_receipt_directory(opened_root, root)
            root_identity = _receipt_directory_identity(root_before)
            if _receipt_directory_identity(opened_root) != root_identity:
                raise ReceiptChainError(
                    "The receipt-store root changed while its lock was acquired."
                )
            lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            lock_descriptor = os.open(
                ".receipts.lock",
                lock_flags,
                0o600,
                dir_fd=root_descriptor,
            )
            os.fchmod(lock_descriptor, 0o600)
            lock_metadata = os.fstat(lock_descriptor)
            _validate_receipt_lock(lock_metadata, root / ".receipts.lock")
            lock_identity = _receipt_lock_identity(lock_metadata)
            if fcntl is not None:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            locked = True
            epoch: _ReceiptLockEpoch = (
                root,
                root_descriptor,
                root_identity,
                lock_descriptor,
                lock_identity,
                threading.get_ident(),
                object(),
            )
            _require_receipt_lock_epoch(epoch)
            yield epoch
            _require_receipt_lock_epoch(epoch)
        finally:
            if lock_descriptor is not None:
                if locked and fcntl is not None:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            os.close(root_descriptor)


class _LockedReceiptTransaction:
    """One receipt-chain view held under the store's cross-process lock."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "Receipt transactions can only be created by "
            "AppendOnlyReceiptStore.transaction()."
        )

    def read_chain(self) -> list[dict[str, Any]]:
        store, _epoch = _require_authentic_receipt_transaction(self)
        return AppendOnlyReceiptStore._read_chain_locked(store, self)

    def append(self, **kwargs: Any) -> dict[str, Any]:
        store, _epoch = _require_authentic_receipt_transaction(self)
        return AppendOnlyReceiptStore._append_locked(store, self, **kwargs)


class AppendOnlyReceiptStore:
    """Persist a single no-fork, append-only chain of evidence receipts."""

    def __init__(self, root: Path) -> None:
        self.root = ensure_private_directory(root)

    def _path(self, receipt_id: str) -> Path:
        _require_identifier(receipt_id, label="receipt_id")
        return self.root / f"{receipt_id}.json"

    def _records(
        self, transaction: _LockedReceiptTransaction
    ) -> dict[str, dict[str, Any]]:
        _store, epoch = _require_authentic_receipt_transaction(
            transaction, expected_store=self
        )
        _require_receipt_lock_epoch(epoch)
        root_descriptor = epoch[1]
        records: dict[str, dict[str, Any]] = {}
        try:
            names = sorted(os.listdir(root_descriptor))
        except OSError as error:
            raise ReceiptChainError(
                "The locked receipt store could not be enumerated."
            ) from error
        for name in names:
            path = self.root / name
            if name == ".receipts.lock":
                lock_metadata = os.stat(
                    name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                _validate_receipt_lock(lock_metadata, path)
                if _receipt_lock_identity(lock_metadata) != epoch[4]:
                    raise ReceiptChainError(
                        "The receipt-store lock identity changed during a read."
                    )
                continue
            if Path(name).name != name or Path(name).suffix != ".json":
                raise ReceiptChainError(f"Unexpected receipt-store entry: {name}")
            try:
                before = os.stat(
                    name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ReceiptChainError(
                    f"Receipt-store entry is unavailable: {name}"
                ) from error
            try:
                _validate_open_evidence_file(before, path, mode=0o600)
            except EvidenceStorageError as error:
                raise ReceiptChainError(
                    f"Receipt-store entry is unsafe: {name}"
                ) from error
            fingerprint = _evidence_file_fingerprint(before)
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_descriptor,
                )
                opened = os.fstat(descriptor)
                _validate_open_evidence_file(opened, path, mode=0o600)
                if _evidence_file_fingerprint(opened) != fingerprint:
                    raise ReceiptChainError(
                        f"Receipt identity changed while opening: {name}"
                    )
                payload = bytearray()
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    payload.extend(block)
                finished = os.fstat(descriptor)
                _validate_open_evidence_file(finished, path, mode=0o600)
                if _evidence_file_fingerprint(finished) != fingerprint:
                    raise ReceiptChainError(f"Receipt changed while reading: {name}")
            except OSError as error:
                raise ReceiptChainError(f"Receipt could not be read: {name}") from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            after = os.stat(
                name,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            if _evidence_file_fingerprint(after) != fingerprint:
                raise ReceiptChainError(f"Receipt path changed while reading: {name}")
            try:
                decoded = json.loads(payload)
                value = validate_record(
                    decoded, expected_schema=EVIDENCE_RECEIPT_SCHEMA
                )
            except (RecursionError, TypeError, ValueError) as error:
                raise ReceiptChainError(f"Receipt record is invalid: {name}") from error
            if canonical_json_bytes(value) != bytes(payload):
                raise ReceiptChainError(f"Receipt record is not canonical: {name}")
            receipt_id = value.get("receipt_id")
            if name != f"{receipt_id}.json" or receipt_id in records:
                raise ReceiptChainError("Receipt filename or ID is inconsistent.")
            if receipt_id != AppendOnlyReceiptStore._content_id(value):
                raise ReceiptChainError(
                    "Receipt content no longer matches its cryptographic ID."
                )
            records[receipt_id] = value
        _require_receipt_lock_epoch(epoch)
        return records

    @staticmethod
    def _content_id(record: Mapping[str, Any]) -> str:
        identity = {key: value for key, value in record.items() if key != "receipt_id"}
        return "receipt_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]

    @staticmethod
    def _ordered(records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        roots = [
            receipt_id
            for receipt_id, value in records.items()
            if value.get("previous_receipt_id") is None
        ]
        if len(roots) != 1:
            raise ReceiptChainError("Receipt chain must contain exactly one root.")
        children: dict[str, list[str]] = {}
        for receipt_id, value in records.items():
            previous = value.get("previous_receipt_id")
            if previous is None:
                continue
            if previous not in records:
                raise ReceiptChainError(
                    "Receipt chain references a missing predecessor."
                )
            children.setdefault(previous, []).append(receipt_id)
        if any(len(items) != 1 for items in children.values()):
            raise ReceiptChainError("Receipt chain cannot fork.")
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        current: str | None = roots[0]
        while current is not None:
            if current in seen:
                raise ReceiptChainError("Receipt chain contains a cycle.")
            seen.add(current)
            ordered.append(dict(records[current]))
            next_items = children.get(current, [])
            current = next_items[0] if next_items else None
        if seen != set(records):
            raise ReceiptChainError("Receipt chain is disconnected.")
        return ordered

    def _read_chain_locked(
        self, transaction: _LockedReceiptTransaction
    ) -> list[dict[str, Any]]:
        _require_authentic_receipt_transaction(transaction, expected_store=self)
        return AppendOnlyReceiptStore._ordered(
            AppendOnlyReceiptStore._records(self, transaction)
        )

    def transaction(self) -> Iterator[_LockedReceiptTransaction]:
        raise RuntimeError("Receipt transaction authority was not installed.")

    def read_chain(self) -> list[dict[str, Any]]:
        with AppendOnlyReceiptStore.transaction(self) as transaction:
            return transaction.read_chain()

    def append(
        self,
        *,
        kind: str,
        issuer_role_id: str,
        protected_artifact_id: str,
        raw_manifest_sha256: str,
        raw_manifest_size_bytes: int,
        result: str,
        details: Mapping[str, Any],
        receipt_id: str | None = None,
        previous_receipt_id: str | None | object = _AUTO_PREVIOUS,
        created_at_utc: str | None = None,
    ) -> dict[str, Any]:
        with AppendOnlyReceiptStore.transaction(self) as transaction:
            return transaction.append(
                kind=kind,
                issuer_role_id=issuer_role_id,
                protected_artifact_id=protected_artifact_id,
                raw_manifest_sha256=raw_manifest_sha256,
                raw_manifest_size_bytes=raw_manifest_size_bytes,
                result=result,
                details=details,
                receipt_id=receipt_id,
                previous_receipt_id=previous_receipt_id,
                created_at_utc=created_at_utc,
            )

    def _append_locked(
        self,
        transaction: _LockedReceiptTransaction,
        *,
        kind: str,
        issuer_role_id: str,
        protected_artifact_id: str,
        raw_manifest_sha256: str,
        raw_manifest_size_bytes: int,
        result: str,
        details: Mapping[str, Any],
        receipt_id: str | None = None,
        previous_receipt_id: str | None | object = _AUTO_PREVIOUS,
        created_at_utc: str | None = None,
    ) -> dict[str, Any]:
        _store, epoch = _require_authentic_receipt_transaction(
            transaction, expected_store=self
        )
        chain = AppendOnlyReceiptStore._read_chain_locked(self, transaction)
        tail = chain[-1]["receipt_id"] if chain else None
        previous = (
            tail if previous_receipt_id is _AUTO_PREVIOUS else previous_receipt_id
        )
        if previous != tail:
            raise ReceiptChainError(
                "New evidence receipt must extend the current chain tail."
            )
        if (
            not isinstance(raw_manifest_size_bytes, int)
            or isinstance(raw_manifest_size_bytes, bool)
            or raw_manifest_size_bytes < 1
        ):
            raise EvidenceStorageError("raw_manifest_size_bytes must be positive.")
        created = created_at_utc or utc_now()
        _parse_utc(created, label="created_at_utc")
        receipt_without_id = {
            "schema_version": EVIDENCE_RECEIPT_SCHEMA,
            "kind": kind,
            "created_at_utc": created,
            "issuer_role_id": issuer_role_id,
            "protected_artifact_id": protected_artifact_id,
            "raw_manifest_sha256": _require_sha256(
                raw_manifest_sha256, label="raw_manifest_sha256"
            ),
            "raw_manifest_size_bytes": raw_manifest_size_bytes,
            "previous_receipt_id": previous,
            "result": result,
            "details": dict(details),
        }
        identifier = AppendOnlyReceiptStore._content_id(receipt_without_id)
        if receipt_id is not None and receipt_id != identifier:
            raise EvidenceStorageError(
                "Explicit receipt_id does not match the receipt content digest."
            )
        _require_identifier(identifier, label="receipt_id")
        target = AppendOnlyReceiptStore._path(self, identifier)
        receipt = {"receipt_id": identifier, **receipt_without_id}
        receipt = validate_record(receipt, expected_schema=EVIDENCE_RECEIPT_SCHEMA)
        payload = canonical_json_bytes(receipt)
        root_descriptor = epoch[1]
        descriptor: int | None = None
        created = False
        created_identity: tuple[int, int] | None = None
        try:
            _require_receipt_lock_epoch(epoch)
            descriptor = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_descriptor,
            )
            created = True
            os.fchmod(descriptor, 0o600)
            opened = os.fstat(descriptor)
            created_identity = (opened.st_dev, opened.st_ino)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            _validate_open_evidence_file(metadata, target, mode=0o600)
            os.close(descriptor)
            descriptor = None
            path_metadata = os.stat(
                target.name,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            if _evidence_file_fingerprint(path_metadata) != _evidence_file_fingerprint(
                metadata
            ):
                raise ReceiptChainError(
                    "The appended receipt path changed before it was committed."
                )
            os.fsync(root_descriptor)
            _require_receipt_lock_epoch(epoch)
            final_chain = AppendOnlyReceiptStore._read_chain_locked(self, transaction)
            if final_chain[-1] != receipt:
                raise ReceiptChainError(
                    "The appended receipt is not the exact locked chain tail."
                )
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            if created and created_identity is not None:
                try:
                    failed_path = os.stat(
                        target.name,
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                except OSError:
                    pass
                else:
                    if (failed_path.st_dev, failed_path.st_ino) == created_identity:
                        os.unlink(target.name, dir_fd=root_descriptor)
                        os.fsync(root_descriptor)
            raise
        return receipt


def _install_receipt_transaction_authority():
    registrations: weakref.WeakKeyDictionary[
        _LockedReceiptTransaction,
        tuple[AppendOnlyReceiptStore, _ReceiptLockEpoch],
    ] = weakref.WeakKeyDictionary()
    active_roots: dict[tuple[int, int], object] = {}

    @contextmanager
    def transaction(
        store: AppendOnlyReceiptStore,
    ) -> Iterator[_LockedReceiptTransaction]:
        """Mint one revocable capability for an exact store and lock epoch."""

        if type(store) is not AppendOnlyReceiptStore:
            raise TypeError("Receipt transactions require an exact receipt store.")
        with _RECEIPT_LOCK:
            current_root = store.root.lstat()
            _validate_receipt_directory(current_root, store.root)
            current_root_key = (current_root.st_dev, current_root.st_ino)
            if current_root_key in active_roots:
                raise ReceiptChainError(
                    "The receipt store already has an active lock epoch."
                )
            with _locked_receipt_store(store.root) as epoch:
                root_key = (epoch[2][0], epoch[2][1])
                if root_key in active_roots:
                    raise ReceiptChainError(
                        "The receipt store already has an active lock epoch."
                    )
                transaction_value = object.__new__(_LockedReceiptTransaction)
                token = epoch[6]
                active_roots[root_key] = token
                registrations[transaction_value] = (store, epoch)
                try:
                    yield transaction_value
                    require(transaction_value, expected_store=store)
                finally:
                    registrations.pop(transaction_value, None)
                    if active_roots.get(root_key) is token:
                        del active_roots[root_key]

    def require(
        value: object,
        *,
        expected_store: AppendOnlyReceiptStore | None = None,
    ) -> tuple[AppendOnlyReceiptStore, _ReceiptLockEpoch]:
        if type(value) is not _LockedReceiptTransaction:
            raise ReceiptChainError(
                "Receipt transaction authority is missing or has the wrong type."
            )
        registration = registrations.get(value)
        if registration is None:
            raise ReceiptChainError(
                "Receipt transaction is inactive, forged, or already consumed."
            )
        store, epoch = registration
        root_key = (epoch[2][0], epoch[2][1])
        if (
            (expected_store is not None and store is not expected_store)
            or type(store) is not AppendOnlyReceiptStore
            or store.root != epoch[0]
            or active_roots.get(root_key) is not epoch[6]
        ):
            raise ReceiptChainError(
                "Receipt transaction does not authorize this exact store and epoch."
            )
        _require_receipt_lock_epoch(epoch)
        return store, epoch

    AppendOnlyReceiptStore.transaction = transaction  # type: ignore[method-assign]
    return require


_require_authentic_receipt_transaction = _install_receipt_transaction_authority()
del _install_receipt_transaction_authority


def evaluate_retention_state(
    *,
    now_utc: str | datetime,
    retain_not_before_utc: str | datetime,
    dependent_claims_active: bool,
    verified_copy_count: int,
    failure_domain_count: int,
    off_host_copy_count: int,
    last_copy_verification_utc: str | datetime | None,
    last_off_host_retrieval_utc: str | datetime | None,
    copy_verification_passed: bool,
    retrieval_passed: bool,
) -> dict[str, Any]:
    """Evaluate claim, cadence, renewal, and deletion gates without deleting data."""

    for name, value in {
        "verified_copy_count": verified_copy_count,
        "failure_domain_count": failure_domain_count,
        "off_host_copy_count": off_host_copy_count,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    for name, value in {
        "dependent_claims_active": dependent_claims_active,
        "copy_verification_passed": copy_verification_passed,
        "retrieval_passed": retrieval_passed,
    }.items():
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean.")
    now = _parse_utc(now_utc, label="now_utc")
    retain_until = _parse_utc(retain_not_before_utc, label="retain_not_before_utc")
    copy_at = (
        _parse_utc(last_copy_verification_utc, label="last_copy_verification_utc")
        if last_copy_verification_utc is not None
        else None
    )
    retrieval_at = (
        _parse_utc(last_off_host_retrieval_utc, label="last_off_host_retrieval_utc")
        if last_off_host_retrieval_utc is not None
        else None
    )
    copy_due = (
        copy_at + timedelta(days=COPY_VERIFICATION_CADENCE_DAYS)
        if copy_at is not None
        else None
    )
    retrieval_due = (
        retrieval_at + timedelta(days=OFF_HOST_RETRIEVAL_CADENCE_DAYS)
        if retrieval_at is not None
        else None
    )
    redundancy_current = (
        verified_copy_count >= 2
        and failure_domain_count >= 2
        and off_host_copy_count >= 1
    )
    copy_current = bool(
        copy_verification_passed and copy_due is not None and now <= copy_due
    )
    retrieval_current = bool(
        retrieval_passed and retrieval_due is not None and now <= retrieval_due
    )
    retention_current = now < retain_until
    reasons: list[str] = []
    if verified_copy_count < 2:
        reasons.append("INSUFFICIENT_COPY_COUNT")
    if failure_domain_count < 2:
        reasons.append("INSUFFICIENT_FAILURE_DOMAIN_COUNT")
    if off_host_copy_count < 1:
        reasons.append("MISSING_OFF_HOST_COPY")
    if not copy_current:
        reasons.append("COPY_VERIFICATION_NOT_CURRENT")
    if not retrieval_current:
        reasons.append("OFF_HOST_RETRIEVAL_NOT_CURRENT")
    if not retention_current:
        reasons.append("RETENTION_RECEIPT_EXPIRED")
    claim_qualified = bool(
        redundancy_current and copy_current and retrieval_current and retention_current
    )
    return {
        "policy_id": RETENTION_POLICY_ID,
        "evaluated_at_utc": _format_utc(now),
        "retain_not_before_utc": _format_utc(retain_until),
        "copy_verification_due_at_utc": (
            _format_utc(copy_due) if copy_due is not None else None
        ),
        "off_host_retrieval_due_at_utc": (
            _format_utc(retrieval_due) if retrieval_due is not None else None
        ),
        "renewal_due": now >= retain_until - timedelta(days=90),
        "redundancy_current": redundancy_current,
        "copy_verification_current": copy_current,
        "off_host_retrieval_current": retrieval_current,
        "retention_current": retention_current,
        "claim_qualified": claim_qualified,
        "claim_suspended": not claim_qualified,
        "suspension_reasons": reasons,
        "deletion_allowed": now >= retain_until and not dependent_claims_active,
    }
