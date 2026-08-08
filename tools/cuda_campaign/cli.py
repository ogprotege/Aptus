"""Privacy-bounded operator CLI for Phase 2 CUDA campaign tooling.

This CLI is an opt-in wrapper around the capture, storage, probe, sanitizer,
and independent-review APIs.  It never invokes a shell, never prints protected
paths or raw exception text, and never installs, updates, cleans, or otherwise
mutates the Ubuntu software environment.

The production telemetry-session adapter is a Python composition API rather
than an operator-CLI shortcut.  ``capture-command`` therefore fails closed in
its default mode.  Only callers that explicitly select a ``nonqualifying`` or
``setup`` role and pass ``--without-telemetry`` may capture a command without
the sidecar.  The recovery-publication commands accept sealed provenance only;
the legacy dictionary-input projection helper is deliberately not exposed here.
Copy and retrieval outcomes are journaled before receipt append so persistence
can resume without repeating a no-clobber data operation.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import (
    ContractError,
    PROCEDURAL_ROLE_ID_RE,
    canonical_json_bytes,
    compact_canonical_json_bytes,
    sha256_bytes,
    utc_now,
    validate_record,
    validate_safe_relative_path,
)
from .harness import CaptureHarness, CaptureHarnessError
from .eligibility import (
    FinalizedSanitizerBinding,
    PublicationCandidateBinding,
    evaluate_publication_eligibility,
    seal_publication_candidate,
    verify_publication_candidate,
)
from .monitoring import LinuxNvidiaHostProbe, ProbeFailure, exact_bytes
from .publication import PublicationError, publish_candidate
from .sanitizer import (
    SanitizationError,
    finalize_projection_stage,
    project_verified_recovery_supplement,
    seal_projection_review,
    verify_finalized_projection,
    verify_projection_stage,
    write_projection_stage,
)
from .storage import (
    AppendOnlyReceiptStore,
    ArtifactIntegrityError,
    EVIDENCE_RECEIPT_SCHEMA,
    EvidenceStorageError,
    RetrievalError,
    copy_sealed_artifact,
    retrieve_sealed_artifact,
    verify_copy_equality,
    verify_sealed_artifact,
)

_ID_PATTERNS = {
    "attempt_slot_id": re.compile(r"^slot_[0-9a-f]{20}$"),
    "experiment_run_id": re.compile(r"^xrun_[0-9a-f]{32}$"),
    "protected_artifact_id": re.compile(r"^artifact_[0-9a-f]{32}$"),
    "copy_id": re.compile(r"^copy_[0-9a-f]{32}$"),
    "failure_domain_id": re.compile(r"^domain_[0-9a-f]{32}$"),
    "host_id": re.compile(r"^host_[0-9a-f]{32}$"),
    "destination_restore_id": re.compile(r"^restore_[0-9a-f]{32}$"),
    "review_id": re.compile(r"^review_[0-9a-f]{32}$"),
    "journal_id": re.compile(r"^journal_[0-9a-f]{32}$"),
    "operation_id": re.compile(r"^operation_[0-9a-f]{32}$"),
    "receipt_id": re.compile(r"^receipt_[0-9a-f]{32}$"),
    "campaign_id": re.compile(r"^campaign_[0-9a-f]{20}$"),
    "destination_id": re.compile(r"^destination_[0-9a-f]{32}$"),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLAIM_KEY = re.compile(r"^[a-z][a-z0-9-]{2,127}$")
_OPERATION_INTENT_FORMAT = "aptus.cuda-campaign-operation-intent.v1"
_OPERATION_OUTCOME_FORMAT = "aptus.cuda-campaign-operation-outcome.v1"
_OPERATION_COMPLETION_FORMAT = "aptus.cuda-campaign-operation-completion.v1"
_OPERATION_PATH_KEY_NAME = ".operation-path-key"


class OperatorCliError(RuntimeError):
    """One stable, public-safe CLI failure."""

    def __init__(
        self,
        error_code: str,
        exit_code: int,
        *,
        safe_fields: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.exit_code = exit_code
        self.safe_fields = dict(safe_fields or {})


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise OperatorCliError("INVALID_ARGUMENT", 2)


def _emit(stream: Any, value: Mapping[str, Any]) -> None:
    # Summaries are constructed locally from an explicit field allowlist.  No
    # object returned by a protected API is printed wholesale.
    stream.write(
        json.dumps(
            dict(value),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def _success(command: str, **fields: Any) -> int:
    _emit(sys.stdout, {"command": command, "ok": True, **fields})
    return 0


def _failure(error_code: str, exit_code: int, **safe_fields: Any) -> int:
    _emit(sys.stderr, {"error_code": error_code, "ok": False, **safe_fields})
    return exit_code


def _require_id(value: str, kind: str) -> str:
    pattern = _ID_PATTERNS[kind]
    if pattern.fullmatch(value) is None:
        raise OperatorCliError("INVALID_IDENTIFIER", 2)
    return value


def _require_digest(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise OperatorCliError("INVALID_DIGEST", 2)
    return value


def _require_claim_key(value: str) -> str:
    if _CLAIM_KEY.fullmatch(value) is None:
        raise OperatorCliError("INVALID_CLAIM_KEY", 2)
    return value


def _private_directory(path: Path) -> Path:
    if path.is_symlink():
        raise OperatorCliError("PROTECTED_PATH_INVALID", 4)
    try:
        metadata = path.stat()
    except OSError:
        raise OperatorCliError("PROTECTED_PATH_INVALID", 4) from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise OperatorCliError("PROTECTED_PATH_INVALID", 4)
    if os.name == "posix" and metadata.st_mode & 0o777 != 0o700:
        raise OperatorCliError("PROTECTED_PATH_INVALID", 4)
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise OperatorCliError("PROTECTED_PATH_INVALID", 4)
    try:
        return path.resolve(strict=True)
    except OSError:
        raise OperatorCliError("PROTECTED_PATH_INVALID", 4) from None


def _fresh_private_child(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise OperatorCliError("DESTINATION_NOT_FRESH", 4)
    parent = _private_directory(path.parent)
    if not path.name or path.name in {".", ".."}:
        raise OperatorCliError("PROTECTED_PATH_INVALID", 4)
    return parent / path.name


def _private_child_candidate(path: Path) -> Path:
    parent = _private_directory(path.parent)
    if not path.name or path.name in {".", ".."}:
        raise OperatorCliError("PROTECTED_PATH_INVALID", 4)
    return parent / path.name


def _match_artifact_id(result: Mapping[str, Any], expected: str) -> None:
    if result.get("protected_artifact_id") != expected:
        raise OperatorCliError("ARTIFACT_ID_MISMATCH", 5)


def _require_role_id(value: str) -> str:
    if PROCEDURAL_ROLE_ID_RE.fullmatch(value) is None:
        raise OperatorCliError("INVALID_ROLE_ID", 2)
    return value


def _require_disjoint_paths(*paths: Path) -> None:
    """Reject equal, nested, or ancestor paths before any mutation begins."""

    resolved = [path.resolve(strict=False) for path in paths]
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise OperatorCliError("STORAGE_BOUNDARY_COLLISION", 4)


def _receipt_store(
    args: argparse.Namespace, *artifacts: Path
) -> AppendOnlyReceiptStore:
    root = _private_directory(Path(args.receipt_store))
    _require_disjoint_paths(root, *artifacts)
    return AppendOnlyReceiptStore(root)


def _receipt_journal_root(args: argparse.Namespace, *protected_paths: Path) -> Path:
    root = _private_directory(Path(args.receipt_journal))
    _require_disjoint_paths(root, *protected_paths)
    return root


def _private_file_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
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


def _validate_private_file_metadata(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600)
        or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
    ):
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)


def _read_pinned_private_bytes(path: Path) -> bytes:
    try:
        before = path.lstat()
        _validate_private_file_metadata(before)
        before_fingerprint = _private_file_fingerprint(before)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except (OSError, OperatorCliError):
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
    payload = bytearray()
    completed = False
    try:
        opened = os.fstat(descriptor)
        _validate_private_file_metadata(opened)
        if _private_file_fingerprint(opened) != before_fingerprint:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
        finished = os.fstat(descriptor)
        _validate_private_file_metadata(finished)
        if _private_file_fingerprint(finished) != before_fingerprint:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        completed = True
    except (OSError, OperatorCliError):
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
    finally:
        os.close(descriptor)
    if completed:
        try:
            after = path.lstat()
            _validate_private_file_metadata(after)
        except (OSError, OperatorCliError):
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
        if _private_file_fingerprint(after) != before_fingerprint:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    return bytes(payload)


def _write_private_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        _validate_private_file_metadata(opened)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("private journal write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        finished = os.fstat(descriptor)
        _validate_private_file_metadata(finished)
        expected_fingerprint = _private_file_fingerprint(finished)
    finally:
        os.close(descriptor)
    actual = path.lstat()
    _validate_private_file_metadata(actual)
    if _private_file_fingerprint(actual) != expected_fingerprint:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    directory = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _require_timestamp(value: object) -> str:
    if not isinstance(value, str):
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    return value


def _parsed_timestamp(value: object) -> datetime:
    exact = _require_timestamp(value)
    return datetime.fromisoformat(exact.replace("Z", "+00:00"))


def _validate_operation_intent(value: object) -> dict[str, Any]:
    fields = {
        "format_version",
        "operation_id",
        "operation_kind",
        "created_at_utc",
        "issuer_role_id",
        "protected_artifact_id",
        "raw_manifest_sha256",
        "raw_manifest_size_bytes",
        "expected_raw_manifest_sha256",
        "source_copy_id",
        "source_failure_domain_id",
        "destination_copy_id",
        "destination_failure_domain_id",
        "destination_restore_id",
        "off_experiment_host",
        "source_path_binding",
        "destination_path_binding",
        "receipt_store_path_binding",
        "receipt_tail_id",
        "receipt_chain_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    try:
        if value["format_version"] != _OPERATION_INTENT_FORMAT:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        _require_id(value["operation_id"], "operation_id")
        _require_timestamp(value["created_at_utc"])
        _require_role_id(value["issuer_role_id"])
        _require_id(value["protected_artifact_id"], "protected_artifact_id")
        _require_digest(value["raw_manifest_sha256"])
        _require_digest(value["expected_raw_manifest_sha256"])
        for field in (
            "source_path_binding",
            "destination_path_binding",
            "receipt_store_path_binding",
            "receipt_chain_sha256",
        ):
            _require_digest(value[field])
        size = value["raw_manifest_size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        tail = value["receipt_tail_id"]
        if tail is not None:
            _require_id(tail, "receipt_id")
        kind = value["operation_kind"]
        if kind == "copy-seal":
            if any(
                value[field] is not None
                for field in (
                    "source_copy_id",
                    "source_failure_domain_id",
                    "destination_restore_id",
                )
            ):
                raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
            _require_id(value["destination_copy_id"], "copy_id")
            _require_id(value["destination_failure_domain_id"], "failure_domain_id")
            if type(value["off_experiment_host"]) is not bool:
                raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
            if value["expected_raw_manifest_sha256"] != value["raw_manifest_sha256"]:
                raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        elif kind == "retrieve":
            _require_id(value["source_copy_id"], "copy_id")
            _require_id(value["source_failure_domain_id"], "failure_domain_id")
            _require_id(value["destination_restore_id"], "destination_restore_id")
            if any(
                value[field] is not None
                for field in (
                    "destination_copy_id",
                    "destination_failure_domain_id",
                    "off_experiment_host",
                )
            ):
                raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        else:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    except (KeyError, TypeError, OperatorCliError):
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
    return dict(value)


def _receipt_projection(
    intent: Mapping[str, Any], outcome: Mapping[str, Any], previous: str | None
) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_RECEIPT_SCHEMA,
        "kind": outcome["receipt_kind"],
        "created_at_utc": outcome["receipt_created_at_utc"],
        "issuer_role_id": intent["issuer_role_id"],
        "protected_artifact_id": intent["protected_artifact_id"],
        "raw_manifest_sha256": intent["raw_manifest_sha256"],
        "raw_manifest_size_bytes": intent["raw_manifest_size_bytes"],
        "previous_receipt_id": previous,
        "result": outcome["result"],
        "details": outcome["details"],
    }


def _validate_operation_outcome(
    value: object, *, intent: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "format_version",
        "operation_id",
        "intent_sha256",
        "recorded_at_utc",
        "receipt_kind",
        "receipt_created_at_utc",
        "result",
        "details",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    try:
        if (
            value["format_version"] != _OPERATION_OUTCOME_FORMAT
            or value["operation_id"] != intent["operation_id"]
            or value["intent_sha256"]
            != sha256_bytes(canonical_json_bytes(dict(intent)))
        ):
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        intent_created_at = _parsed_timestamp(intent["created_at_utc"])
        recorded_at = _parsed_timestamp(value["recorded_at_utc"])
        receipt_created_at = _parsed_timestamp(value["receipt_created_at_utc"])
        if not intent_created_at <= receipt_created_at <= recorded_at:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        expected_kind = (
            "copy-verification"
            if intent["operation_kind"] == "copy-seal"
            else "retrieval"
        )
        if value["receipt_kind"] != expected_kind:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        if value["result"] not in {"passed", "failed"}:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        receipt_without_id = _receipt_projection(intent, value, None)
        receipt_id = (
            "receipt_"
            + sha256_bytes(compact_canonical_json_bytes(receipt_without_id))[:32]
        )
        validate_record(
            {"receipt_id": receipt_id, **receipt_without_id},
            expected_schema=EVIDENCE_RECEIPT_SCHEMA,
        )
        if value["receipt_kind"] == "copy-verification":
            details = value["details"]
            if (
                not isinstance(details, dict)
                or set(details)
                != {
                    "copy_id",
                    "failure_domain_id",
                    "off_experiment_host",
                    "verification_result",
                }
                or details["copy_id"] != intent["destination_copy_id"]
                or details["failure_domain_id"]
                != intent["destination_failure_domain_id"]
                or details["off_experiment_host"] is not intent["off_experiment_host"]
                or details["verification_result"] != value["result"]
            ):
                raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        else:
            details = value["details"]
            if not isinstance(details, dict):
                raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
            started_at = _parsed_timestamp(details.get("started_at_utc"))
            finished_at = _parsed_timestamp(details.get("finished_at_utc"))
            duration_ns = details.get("duration_ns")
            if (
                details.get("source_copy_id") != intent["source_copy_id"]
                or details.get("source_failure_domain_id")
                != intent["source_failure_domain_id"]
                or details.get("destination_restore_id")
                != intent["destination_restore_id"]
                or details.get("expected_raw_manifest_sha256")
                != intent["expected_raw_manifest_sha256"]
                or not intent_created_at <= started_at <= finished_at
                or finished_at != receipt_created_at
                or not isinstance(duration_ns, int)
                or isinstance(duration_ns, bool)
                or duration_ns < 1
                or (
                    value["result"] == "passed"
                    and intent["expected_raw_manifest_sha256"]
                    != intent["raw_manifest_sha256"]
                )
            ):
                raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    except (ContractError, KeyError, TypeError, OperatorCliError, ValueError):
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
    return dict(value)


def _validate_operation_completion(
    value: object,
    *,
    intent: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "format_version",
        "operation_id",
        "intent_sha256",
        "outcome_sha256",
        "receipt_id",
        "previous_receipt_id",
        "completed_at_utc",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    try:
        if (
            value["format_version"] != _OPERATION_COMPLETION_FORMAT
            or value["operation_id"] != intent["operation_id"]
            or value["intent_sha256"]
            != sha256_bytes(canonical_json_bytes(dict(intent)))
            or value["outcome_sha256"]
            != sha256_bytes(canonical_json_bytes(dict(outcome)))
        ):
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
        _require_id(value["receipt_id"], "receipt_id")
        if value["previous_receipt_id"] is not None:
            _require_id(value["previous_receipt_id"], "receipt_id")
        _require_timestamp(value["completed_at_utc"])
        if value["completed_at_utc"] != outcome["recorded_at_utc"]:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    except (KeyError, TypeError, OperatorCliError):
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
    return dict(value)


def _operation_record_path(root: Path, operation_id: str, part: str) -> Path:
    identifier = _require_id(operation_id, "operation_id")
    if part not in {"intent", "outcome", "completion"}:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    return root / f"{identifier}.{part}.json"


def _load_receipt_journal(
    root: Path,
    operation_id: str,
    part: str = "intent",
    *,
    intent: Mapping[str, Any] | None = None,
    outcome: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an operation journal part through one identity-pinned descriptor."""

    path = _operation_record_path(root, operation_id, part)
    payload = _read_pinned_private_bytes(path)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
    if canonical_json_bytes(value) != payload:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    if part == "intent":
        return _validate_operation_intent(value)
    if intent is None:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    if part == "outcome":
        return _validate_operation_outcome(value, intent=intent)
    if outcome is None:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    return _validate_operation_completion(value, intent=intent, outcome=outcome)


def _write_operation_record(
    root: Path,
    value: Mapping[str, Any],
    part: str,
    *,
    intent: Mapping[str, Any] | None = None,
    outcome: Mapping[str, Any] | None = None,
) -> bool:
    operation_id = str(value["operation_id"])
    path = _operation_record_path(root, operation_id, part)
    payload = canonical_json_bytes(dict(value))
    try:
        _write_private_bytes(path, payload)
    except FileExistsError:
        existing = _load_receipt_journal(
            root,
            operation_id,
            part,
            intent=intent,
            outcome=outcome,
        )
        if existing != value:
            raise OperatorCliError("OPERATION_INTENT_MISMATCH", 5) from None
        return False
    return True


def _load_or_create_path_key(root: Path) -> bytes:
    path = root / _OPERATION_PATH_KEY_NAME
    try:
        path.lstat()
    except FileNotFoundError:
        payload = secrets.token_bytes(32)
        try:
            _write_private_bytes(path, payload)
        except FileExistsError:
            payload = _read_pinned_private_bytes(path)
    except OSError:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
    else:
        payload = _read_pinned_private_bytes(path)
    if len(payload) != 32:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    return payload


def _path_binding(root: Path, path: Path) -> str:
    key = _load_or_create_path_key(root)
    resolved = str(path.resolve(strict=False)).encode("utf-8")
    return hmac.new(key, resolved, "sha256").hexdigest()


def _receipt_chain_digest(chain: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(
        compact_canonical_json_bytes(
            {"receipt_ids": [receipt["receipt_id"] for receipt in chain]}
        )
    )


def _build_operation_intent(
    *,
    operation_id: str,
    operation_kind: str,
    issuer_role_id: str,
    protected_artifact_id: str,
    raw_manifest_sha256: str,
    raw_manifest_size_bytes: int,
    expected_raw_manifest_sha256: str,
    source_copy_id: str | None,
    source_failure_domain_id: str | None,
    destination_copy_id: str | None,
    destination_failure_domain_id: str | None,
    destination_restore_id: str | None,
    off_experiment_host: bool | None,
    source: Path,
    destination: Path,
    receipt_store: AppendOnlyReceiptStore,
    receipt_journal: Path,
    chain: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    intent = {
        "format_version": _OPERATION_INTENT_FORMAT,
        "operation_id": operation_id,
        "operation_kind": operation_kind,
        "created_at_utc": utc_now(),
        "issuer_role_id": issuer_role_id,
        "protected_artifact_id": protected_artifact_id,
        "raw_manifest_sha256": raw_manifest_sha256,
        "raw_manifest_size_bytes": raw_manifest_size_bytes,
        "expected_raw_manifest_sha256": expected_raw_manifest_sha256,
        "source_copy_id": source_copy_id,
        "source_failure_domain_id": source_failure_domain_id,
        "destination_copy_id": destination_copy_id,
        "destination_failure_domain_id": destination_failure_domain_id,
        "destination_restore_id": destination_restore_id,
        "off_experiment_host": off_experiment_host,
        "source_path_binding": _path_binding(receipt_journal, source),
        "destination_path_binding": _path_binding(receipt_journal, destination),
        "receipt_store_path_binding": _path_binding(
            receipt_journal, receipt_store.root
        ),
        "receipt_tail_id": chain[-1]["receipt_id"] if chain else None,
        "receipt_chain_sha256": _receipt_chain_digest(chain),
    }
    return _validate_operation_intent(intent)


def _write_operation_outcome(
    root: Path, intent: Mapping[str, Any], outcome: Mapping[str, Any]
) -> bool:
    validated = _validate_operation_outcome(outcome, intent=intent)
    return _write_operation_record(root, validated, "outcome", intent=intent)


def _matching_outcome_receipts(
    chain: Sequence[Mapping[str, Any]],
    intent: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = _receipt_projection(intent, outcome, intent["receipt_tail_id"])
    matches: list[dict[str, Any]] = []
    for receipt in chain:
        if all(receipt.get(key) == item for key, item in expected.items()):
            matches.append(dict(receipt))
    return matches


def _pinned_chain_prefix(
    chain: Sequence[Mapping[str, Any]], intent: Mapping[str, Any]
) -> list[Mapping[str, Any]] | None:
    tail = intent["receipt_tail_id"]
    if tail is None:
        prefix: list[Mapping[str, Any]] = []
    else:
        indexes = [
            index
            for index, receipt in enumerate(chain)
            if receipt.get("receipt_id") == tail
        ]
        if len(indexes) != 1:
            return None
        prefix = list(chain[: indexes[0] + 1])
    if _receipt_chain_digest(prefix) != intent["receipt_chain_sha256"]:
        return None
    return prefix


def _complete_operation(
    root: Path,
    transaction: Any,
    intent: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    chain = transaction.read_chain()
    prefix = _pinned_chain_prefix(chain, intent)
    if prefix is None:
        raise OperatorCliError("RECEIPT_CHAIN_MOVED", 5)
    matches = _matching_outcome_receipts(chain, intent, outcome)
    if len(matches) > 1:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5)
    if matches:
        receipt = matches[0]
    else:
        current_tail = chain[-1]["receipt_id"] if chain else None
        if current_tail != intent["receipt_tail_id"] or len(prefix) != len(chain):
            raise OperatorCliError("RECEIPT_CHAIN_MOVED", 5)
        receipt = transaction.append(
            kind=outcome["receipt_kind"],
            issuer_role_id=intent["issuer_role_id"],
            protected_artifact_id=intent["protected_artifact_id"],
            raw_manifest_sha256=intent["raw_manifest_sha256"],
            raw_manifest_size_bytes=intent["raw_manifest_size_bytes"],
            result=outcome["result"],
            details=outcome["details"],
            previous_receipt_id=intent["receipt_tail_id"],
            created_at_utc=outcome["receipt_created_at_utc"],
        )
    completion = _validate_operation_completion(
        {
            "format_version": _OPERATION_COMPLETION_FORMAT,
            "operation_id": intent["operation_id"],
            "intent_sha256": sha256_bytes(canonical_json_bytes(dict(intent))),
            "outcome_sha256": sha256_bytes(canonical_json_bytes(dict(outcome))),
            "receipt_id": receipt["receipt_id"],
            "previous_receipt_id": receipt["previous_receipt_id"],
            "completed_at_utc": outcome["recorded_at_utc"],
        },
        intent=intent,
        outcome=outcome,
    )
    _write_operation_record(
        root,
        completion,
        "completion",
        intent=intent,
        outcome=outcome,
    )
    return receipt


def _has_off_host_copy_attestation(
    chain: Sequence[Mapping[str, Any]],
    *,
    protected_artifact_id: str,
    raw_manifest_sha256: str,
    raw_manifest_size_bytes: int,
    copy_id: str,
    failure_domain_id: str,
) -> bool:
    relevant = [
        receipt
        for receipt in chain
        if receipt.get("kind") == "copy-verification"
        and receipt.get("protected_artifact_id") == protected_artifact_id
        and receipt.get("raw_manifest_sha256") == raw_manifest_sha256
        and receipt.get("raw_manifest_size_bytes") == raw_manifest_size_bytes
        and isinstance(receipt.get("details"), Mapping)
        and receipt["details"].get("copy_id") == copy_id
        and receipt["details"].get("failure_domain_id") == failure_domain_id
    ]
    if not relevant:
        return False
    latest = relevant[-1]
    return latest.get("result") == "passed" and latest.get("details") == {
        "copy_id": copy_id,
        "failure_domain_id": failure_domain_id,
        "off_experiment_host": True,
        "verification_result": "passed",
    }


def _build_operation_outcome(
    intent: Mapping[str, Any], *, result: str, details: Mapping[str, Any]
) -> dict[str, Any]:
    recorded_at = utc_now()
    receipt_created_at = (
        details.get("finished_at_utc")
        if intent["operation_kind"] == "retrieve"
        else recorded_at
    )
    outcome = {
        "format_version": _OPERATION_OUTCOME_FORMAT,
        "operation_id": intent["operation_id"],
        "intent_sha256": sha256_bytes(canonical_json_bytes(dict(intent))),
        "recorded_at_utc": recorded_at,
        "receipt_kind": (
            "copy-verification"
            if intent["operation_kind"] == "copy-seal"
            else "retrieval"
        ),
        "receipt_created_at_utc": receipt_created_at,
        "result": result,
        "details": dict(details),
    }
    return _validate_operation_outcome(outcome, intent=intent)


def _load_existing_intent(root: Path, operation_id: str) -> dict[str, Any] | None:
    path = _operation_record_path(root, operation_id, "intent")
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
    return _load_receipt_journal(root, operation_id)


def _require_intent_values(
    intent: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if any(intent.get(key) != value for key, value in expected.items()):
        raise OperatorCliError("OPERATION_INTENT_MISMATCH", 5)


def _verify_intent_bindings(
    intent: Mapping[str, Any],
    *,
    source: Path,
    destination: Path,
    receipt_store: AppendOnlyReceiptStore,
    receipt_journal: Path,
    issuer_role_id: str,
) -> dict[str, Any]:
    if intent["issuer_role_id"] != issuer_role_id:
        raise OperatorCliError("RECEIPT_JOURNAL_ROLE_MISMATCH", 5)
    expected_bindings = {
        "source_path_binding": _path_binding(receipt_journal, source),
        "destination_path_binding": _path_binding(receipt_journal, destination),
        "receipt_store_path_binding": _path_binding(
            receipt_journal, receipt_store.root
        ),
    }
    if any(intent[key] != value for key, value in expected_bindings.items()):
        raise OperatorCliError("OPERATION_INTENT_MISMATCH", 5)
    source_result = verify_sealed_artifact(source)
    if (
        source_result.get("protected_artifact_id") != intent["protected_artifact_id"]
        or source_result.get("raw_manifest_sha256") != intent["raw_manifest_sha256"]
        or source_result.get("raw_manifest_size_bytes")
        != intent["raw_manifest_size_bytes"]
    ):
        raise OperatorCliError("OPERATION_INTENT_MISMATCH", 5)
    return source_result


def _reconcile_completed_destination(
    intent: Mapping[str, Any], *, source: Path, destination: Path
) -> dict[str, Any]:
    if intent["operation_kind"] != "copy-seal":
        # A completed restore without its durable outcome has lost the exact
        # monotonic timing result.  Byte equality cannot recreate measured
        # duration, so the operation stays unresolved and a new independently
        # timed retrieval must use a fresh operation ID and destination.
        raise OperatorCliError(
            "OPERATION_RECONCILIATION_REQUIRED",
            5,
            safe_fields={"operation_id": intent["operation_id"]},
        )
    try:
        destination_root = _private_directory(destination)
        verification = verify_copy_equality(source, destination_root)
    except (OSError, OperatorCliError, EvidenceStorageError):
        raise OperatorCliError(
            "OPERATION_RECONCILIATION_REQUIRED",
            5,
            safe_fields={"operation_id": intent["operation_id"]},
        ) from None
    if (
        verification.get("protected_artifact_id") != intent["protected_artifact_id"]
        or verification.get("raw_manifest_sha256") != intent["raw_manifest_sha256"]
        or verification.get("raw_manifest_size_bytes")
        != intent["raw_manifest_size_bytes"]
    ):
        raise OperatorCliError(
            "OPERATION_RECONCILIATION_REQUIRED",
            5,
            safe_fields={"operation_id": intent["operation_id"]},
        )
    details = {
        "copy_id": intent["destination_copy_id"],
        "failure_domain_id": intent["destination_failure_domain_id"],
        "off_experiment_host": intent["off_experiment_host"],
        "verification_result": "passed",
    }
    return _build_operation_outcome(intent, result="passed", details=details)


def _revalidate_passing_outcome_destination(
    intent: Mapping[str, Any],
    outcome: Mapping[str, Any],
    *,
    source: Path,
    destination: Path,
) -> None:
    """Require current destination bytes before a pending pass is receipted."""

    if outcome["result"] != "passed":
        return
    try:
        verification = verify_copy_equality(source, destination)
    except (OSError, EvidenceStorageError, FileExistsError):
        raise OperatorCliError(
            "OPERATION_RECONCILIATION_REQUIRED",
            5,
            safe_fields={"operation_id": intent["operation_id"]},
        ) from None
    common_matches = (
        verification.get("verification_result") == "passed"
        and verification.get("protected_artifact_id") == intent["protected_artifact_id"]
        and verification.get("raw_manifest_sha256") == intent["raw_manifest_sha256"]
        and verification.get("raw_manifest_size_bytes")
        == intent["raw_manifest_size_bytes"]
    )
    retrieval_matches = True
    if intent["operation_kind"] == "retrieve":
        details = outcome["details"]
        retrieval_matches = (
            details["observed_raw_manifest_sha256"]
            == verification.get("raw_manifest_sha256")
            and details["restored_file_count"] == verification.get("stored_file_count")
            and details["restored_total_bytes"]
            == verification.get("stored_total_bytes")
        )
    if not common_matches or not retrieval_matches:
        raise OperatorCliError(
            "OPERATION_RECONCILIATION_REQUIRED",
            5,
            safe_fields={"operation_id": intent["operation_id"]},
        )


def _quantity_bytes(value: object) -> int:
    if not isinstance(value, Mapping):
        raise OperatorCliError("PROBE_RESULT_INVALID", 3)
    try:
        return exact_bytes(value["value"], value["unit"])
    except (KeyError, TypeError, ValueError, ProbeFailure):
        raise OperatorCliError("PROBE_RESULT_INVALID", 3) from None


def _optional_sensor_summary(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "invalid", "value": None}
    status = value.get("status")
    reading = value.get("value")
    return {
        "status": status if isinstance(status, str) else "invalid",
        "value": reading if type(reading) in (int, float) else None,
    }


def _probe_summary(host_id: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    try:
        gpu = snapshot["gpu"]
        host = snapshot["host"]
        if not isinstance(gpu, Mapping) or not isinstance(host, Mapping):
            raise KeyError
        summary = {
            "host_id": host_id,
            "qualification_status": "nonqualifying-read-only-probe",
            "hardware_event_projection": "not-configured",
            "gpu": {
                "memory_used_bytes": _quantity_bytes(gpu["memory_used"]),
                "memory_free_bytes": _quantity_bytes(gpu["memory_free"]),
                "memory_total_bytes": _quantity_bytes(gpu["memory_total"]),
                "utilization_percent": gpu["utilization_percent"],
                "temperature_c": gpu["temperature_c"],
                "power_draw_w": gpu["power_draw_w"],
                "power_limit_w": gpu["power_limit_w"],
                "graphics_clock_mhz": gpu["graphics_clock_mhz"],
                "memory_clock_mhz": gpu["memory_clock_mhz"],
                "performance_state": gpu["performance_state"],
                "throttle_active": bool(gpu["throttle_reasons"]),
                "compute_process_count": len(gpu["compute_processes"]),
            },
            "host": {
                "mem_available_bytes": host["mem_available_bytes"],
                "swap_used_bytes": host["swap_used_bytes"],
                "swap_read_bytes": host["swap_read_bytes"],
                "swap_write_bytes": host["swap_write_bytes"],
                "load_1m": host["load_1m"],
                "filesystem_free_bytes": host["filesystem_free_bytes"],
                "managed_process_rss_bytes": host["managed_process_rss_bytes"],
                "managed_process_cpu_seconds": host["managed_process_cpu_seconds"],
                "managed_process_read_bytes": host["managed_process_read_bytes"],
                "managed_process_write_bytes": host["managed_process_write_bytes"],
                "disk_growth_bytes": host["disk_growth_bytes"],
                "aptus_lease_active": host["aptus_lease_active"],
                "cpu_temperature": _optional_sensor_summary(host["cpu_temperature"]),
                "nvme_temperature": _optional_sensor_summary(host["nvme_temperature"]),
            },
        }
    except (KeyError, TypeError):
        raise OperatorCliError("PROBE_RESULT_INVALID", 3) from None
    return summary


def _cmd_probe(args: argparse.Namespace) -> int:
    host_id = _require_id(args.host_id, "host_id")
    probe = LinuxNvidiaHostProbe(
        filesystem_path=Path(args.filesystem_path),
        managed_pids=lambda: (),
        xid_errors=lambda: (),
        hardware_events=lambda: {
            "reset_detected": False,
            "device_lost": False,
            "hardware_error": False,
        },
        lease_active=lambda: False,
        disk_growth_bytes=lambda: 0,
        gpu_index=args.gpu_index,
        nvidia_smi_path=args.nvidia_smi_path,
    )
    snapshot = probe()
    return _success("probe", **_probe_summary(host_id, snapshot))


def _cmd_verify_seal(args: argparse.Namespace) -> int:
    artifact_id = _require_id(args.protected_artifact_id, "protected_artifact_id")
    artifact = _private_directory(Path(args.artifact))
    result = verify_sealed_artifact(artifact)
    _match_artifact_id(result, artifact_id)
    return _success(
        "verify-seal",
        protected_artifact_id=artifact_id,
        raw_manifest_sha256=result["raw_manifest_sha256"],
        raw_manifest_size_bytes=result["raw_manifest_size_bytes"],
        file_count=result["file_count"],
        total_bytes=result["total_bytes"],
        verification_result="passed",
    )


def _cmd_copy_seal(args: argparse.Namespace) -> int:
    operation_id = _require_id(args.operation_id, "operation_id")
    artifact_id = _require_id(args.protected_artifact_id, "protected_artifact_id")
    copy_id = _require_id(args.copy_id, "copy_id")
    failure_domain_id = _require_id(args.failure_domain_id, "failure_domain_id")
    issuer_role_id = _require_role_id(args.issuer_role_id)
    source = _private_directory(Path(args.source))
    source_result = verify_sealed_artifact(source)
    _match_artifact_id(source_result, artifact_id)
    destination = _private_child_candidate(Path(args.destination))
    receipt_store = _receipt_store(args, source, destination)
    receipt_journal = _receipt_journal_root(
        args, source, destination, receipt_store.root
    )
    with receipt_store.transaction() as transaction:
        existing = _load_existing_intent(receipt_journal, operation_id)
        if existing is not None:
            _verify_intent_bindings(
                existing,
                source=source,
                destination=destination,
                receipt_store=receipt_store,
                receipt_journal=receipt_journal,
                issuer_role_id=issuer_role_id,
            )
            _require_intent_values(
                existing,
                {
                    "operation_kind": "copy-seal",
                    "protected_artifact_id": artifact_id,
                    "destination_copy_id": copy_id,
                    "destination_failure_domain_id": failure_domain_id,
                    "off_experiment_host": args.off_experiment_host,
                },
            )
            raise OperatorCliError(
                "OPERATION_RESUME_REQUIRED",
                5,
                safe_fields={"operation_id": operation_id},
            )
        destination = _fresh_private_child(destination)
        chain = transaction.read_chain()
        intent = _build_operation_intent(
            operation_id=operation_id,
            operation_kind="copy-seal",
            issuer_role_id=issuer_role_id,
            protected_artifact_id=artifact_id,
            raw_manifest_sha256=source_result["raw_manifest_sha256"],
            raw_manifest_size_bytes=source_result["raw_manifest_size_bytes"],
            expected_raw_manifest_sha256=source_result["raw_manifest_sha256"],
            source_copy_id=None,
            source_failure_domain_id=None,
            destination_copy_id=copy_id,
            destination_failure_domain_id=failure_domain_id,
            destination_restore_id=None,
            off_experiment_host=args.off_experiment_host,
            source=source,
            destination=destination,
            receipt_store=receipt_store,
            receipt_journal=receipt_journal,
            chain=chain,
        )
        _write_operation_record(receipt_journal, intent, "intent")
        try:
            result = copy_sealed_artifact(source, destination)
        except (OSError, EvidenceStorageError):
            raise OperatorCliError(
                "OPERATION_RECONCILIATION_REQUIRED",
                5,
                safe_fields={"operation_id": operation_id},
            ) from None
        if (
            result.get("verification_result") != "passed"
            or result.get("protected_artifact_id") != artifact_id
            or result.get("raw_manifest_sha256") != intent["raw_manifest_sha256"]
            or result.get("raw_manifest_size_bytes")
            != intent["raw_manifest_size_bytes"]
        ):
            raise OperatorCliError(
                "OPERATION_RECONCILIATION_REQUIRED",
                5,
                safe_fields={"operation_id": operation_id},
            )
        details = {
            "copy_id": copy_id,
            "failure_domain_id": failure_domain_id,
            "off_experiment_host": args.off_experiment_host,
            "verification_result": "passed",
        }
        outcome = _build_operation_outcome(intent, result="passed", details=details)
        try:
            _write_operation_outcome(receipt_journal, intent, outcome)
        except OSError:
            raise OperatorCliError(
                "OPERATION_OUTCOME_PENDING",
                5,
                safe_fields={"operation_id": operation_id},
            ) from None
        try:
            receipt = _complete_operation(receipt_journal, transaction, intent, outcome)
        except (OSError, EvidenceStorageError, FileExistsError):
            raise OperatorCliError(
                "RECEIPT_APPEND_PENDING",
                5,
                safe_fields={"operation_id": operation_id},
            ) from None
    return _success(
        "copy-seal",
        operation_id=operation_id,
        protected_artifact_id=artifact_id,
        copy_id=copy_id,
        failure_domain_id=failure_domain_id,
        off_experiment_host=args.off_experiment_host,
        receipt_id=receipt["receipt_id"],
        raw_manifest_sha256=result["raw_manifest_sha256"],
        raw_manifest_size_bytes=result["raw_manifest_size_bytes"],
        file_count=result["file_count"],
        total_bytes=result["total_bytes"],
        verification_result="passed",
    )


def _cmd_retrieve(args: argparse.Namespace) -> int:
    operation_id = _require_id(args.operation_id, "operation_id")
    artifact_id = _require_id(args.protected_artifact_id, "protected_artifact_id")
    copy_id = _require_id(args.source_copy_id, "copy_id")
    failure_domain_id = _require_id(args.source_failure_domain_id, "failure_domain_id")
    restore_id = _require_id(args.destination_restore_id, "destination_restore_id")
    expected_digest = _require_digest(args.expected_raw_manifest_sha256)
    issuer_role_id = _require_role_id(args.issuer_role_id)
    source = _private_directory(Path(args.source))
    source_result = verify_sealed_artifact(source)
    _match_artifact_id(source_result, artifact_id)
    destination = _private_child_candidate(Path(args.destination))
    receipt_store = _receipt_store(args, source, destination)
    receipt_journal = _receipt_journal_root(
        args, source, destination, receipt_store.root
    )
    retrieval_error: RetrievalError | None = None
    with receipt_store.transaction() as transaction:
        existing = _load_existing_intent(receipt_journal, operation_id)
        if existing is not None:
            _verify_intent_bindings(
                existing,
                source=source,
                destination=destination,
                receipt_store=receipt_store,
                receipt_journal=receipt_journal,
                issuer_role_id=issuer_role_id,
            )
            _require_intent_values(
                existing,
                {
                    "operation_kind": "retrieve",
                    "protected_artifact_id": artifact_id,
                    "expected_raw_manifest_sha256": expected_digest,
                    "source_copy_id": copy_id,
                    "source_failure_domain_id": failure_domain_id,
                    "destination_restore_id": restore_id,
                },
            )
            raise OperatorCliError(
                "OPERATION_RESUME_REQUIRED",
                5,
                safe_fields={"operation_id": operation_id},
            )
        destination = _fresh_private_child(destination)
        chain = transaction.read_chain()
        if not _has_off_host_copy_attestation(
            chain,
            protected_artifact_id=artifact_id,
            raw_manifest_sha256=source_result["raw_manifest_sha256"],
            raw_manifest_size_bytes=source_result["raw_manifest_size_bytes"],
            copy_id=copy_id,
            failure_domain_id=failure_domain_id,
        ):
            raise OperatorCliError("OFF_HOST_COPY_ATTESTATION_REQUIRED", 5)
        intent = _build_operation_intent(
            operation_id=operation_id,
            operation_kind="retrieve",
            issuer_role_id=issuer_role_id,
            protected_artifact_id=artifact_id,
            raw_manifest_sha256=source_result["raw_manifest_sha256"],
            raw_manifest_size_bytes=source_result["raw_manifest_size_bytes"],
            expected_raw_manifest_sha256=expected_digest,
            source_copy_id=copy_id,
            source_failure_domain_id=failure_domain_id,
            destination_copy_id=None,
            destination_failure_domain_id=None,
            destination_restore_id=restore_id,
            off_experiment_host=None,
            source=source,
            destination=destination,
            receipt_store=receipt_store,
            receipt_journal=receipt_journal,
            chain=chain,
        )
        _write_operation_record(receipt_journal, intent, "intent")
        try:
            result = retrieve_sealed_artifact(
                source,
                destination,
                source_copy_id=copy_id,
                source_failure_domain_id=failure_domain_id,
                expected_raw_manifest_sha256=expected_digest,
                destination_restore_id=restore_id,
            )
            outcome = _build_operation_outcome(intent, result="passed", details=result)
        except RetrievalError as error:
            retrieval_error = error
            result = error.details
            outcome = _build_operation_outcome(intent, result="failed", details=result)
        try:
            _write_operation_outcome(receipt_journal, intent, outcome)
        except OSError:
            raise OperatorCliError(
                "OPERATION_OUTCOME_PENDING",
                5,
                safe_fields={"operation_id": operation_id},
            ) from None
        try:
            receipt = _complete_operation(receipt_journal, transaction, intent, outcome)
        except (OSError, EvidenceStorageError, FileExistsError):
            raise OperatorCliError(
                "RECEIPT_APPEND_PENDING",
                5,
                safe_fields={"operation_id": operation_id},
            ) from None
    if retrieval_error is not None:
        raise retrieval_error
    return _success(
        "retrieve",
        operation_id=operation_id,
        protected_artifact_id=artifact_id,
        source_copy_id=copy_id,
        source_failure_domain_id=failure_domain_id,
        destination_restore_id=restore_id,
        receipt_id=receipt["receipt_id"],
        restored_file_count=result["restored_file_count"],
        restored_total_bytes=result["restored_total_bytes"],
        raw_manifest_sha256=result["observed_raw_manifest_sha256"],
        duration_ns=result["duration_ns"],
        verification_result=result["verification_result"],
    )


def _cmd_resume_operation(args: argparse.Namespace) -> int:
    operation_id = _require_id(args.operation_id, "operation_id")
    issuer_role_id = _require_role_id(args.issuer_role_id)
    source = _private_directory(Path(args.source))
    destination = _private_child_candidate(Path(args.destination))
    receipt_store = _receipt_store(args, source, destination)
    receipt_journal = _receipt_journal_root(
        args, source, destination, receipt_store.root
    )
    intent = _load_receipt_journal(receipt_journal, operation_id)
    with receipt_store.transaction() as transaction:
        _verify_intent_bindings(
            intent,
            source=source,
            destination=destination,
            receipt_store=receipt_store,
            receipt_journal=receipt_journal,
            issuer_role_id=issuer_role_id,
        )
        outcome_path = _operation_record_path(receipt_journal, operation_id, "outcome")
        try:
            outcome_path.lstat()
        except FileNotFoundError:
            outcome = _reconcile_completed_destination(
                intent, source=source, destination=destination
            )
            try:
                _write_operation_outcome(receipt_journal, intent, outcome)
            except OSError:
                raise OperatorCliError(
                    "OPERATION_OUTCOME_PENDING",
                    5,
                    safe_fields={"operation_id": operation_id},
                ) from None
        except OSError:
            raise OperatorCliError("RECEIPT_JOURNAL_INVALID", 5) from None
        else:
            outcome = _load_receipt_journal(
                receipt_journal, operation_id, "outcome", intent=intent
            )
        current_chain = transaction.read_chain()
        matching_receipts = _matching_outcome_receipts(current_chain, intent, outcome)
        if not matching_receipts:
            prefix = _pinned_chain_prefix(current_chain, intent)
            if prefix is None or len(prefix) != len(current_chain):
                raise OperatorCliError("RECEIPT_CHAIN_MOVED", 5)
            _revalidate_passing_outcome_destination(
                intent,
                outcome,
                source=source,
                destination=destination,
            )
        try:
            receipt = _complete_operation(receipt_journal, transaction, intent, outcome)
        except (OSError, EvidenceStorageError, FileExistsError):
            raise OperatorCliError(
                "RECEIPT_APPEND_PENDING",
                5,
                safe_fields={"operation_id": operation_id},
            ) from None
    return _success(
        "resume-operation",
        operation_id=operation_id,
        receipt_id=receipt["receipt_id"],
        receipt_kind=receipt["kind"],
        receipt_result=receipt["result"],
        protected_artifact_id=receipt["protected_artifact_id"],
        raw_manifest_sha256=receipt["raw_manifest_sha256"],
    )


def _command_argv(value: Sequence[str]) -> list[str]:
    exact = list(value)
    if exact and exact[0] == "--":
        exact.pop(0)
    if not exact or any(not item or "\x00" in item for item in exact):
        raise OperatorCliError("INVALID_COMMAND", 2)
    return exact


def _cmd_capture_command(args: argparse.Namespace) -> int:
    attempt_slot_id = _require_id(args.attempt_slot_id, "attempt_slot_id")
    experiment_run_id = _require_id(args.experiment_run_id, "experiment_run_id")
    if not args.without_telemetry:
        raise OperatorCliError("TELEMETRY_SIDECAR_UNAVAILABLE", 7)
    if args.mode not in {"nonqualifying", "setup"}:
        raise OperatorCliError("TELEMETRY_REQUIRED_FOR_QUALIFYING_CAPTURE", 7)
    state_root = _private_directory(Path(args.state_root))
    working_directory = Path(args.working_directory).resolve(strict=True)
    if not working_directory.is_dir():
        raise OperatorCliError("WORKING_DIRECTORY_INVALID", 4)
    artifact_directory = _fresh_private_child(Path(args.artifact_directory))
    harness = CaptureHarness(
        state_root,
        attempt_slot_id=attempt_slot_id,
        experiment_run_id=experiment_run_id,
        provisional_retain_not_before_utc=args.retain_not_before_utc,
        capture_tool={
            "name": "aptus-cuda-campaign-operator-cli",
            "version": "v1",
            "capture_role": args.mode,
            "telemetry": "explicitly-disabled",
        },
        source_bindings={"qualification_status": f"{args.mode}-only"},
    )
    outcome = harness.run_command(
        _command_argv(args.command_argv),
        artifact_directory=artifact_directory,
        working_directory=working_directory,
        timeout_seconds=args.timeout_seconds,
        telemetry_session=None,
    )
    summary = {
        "command": "capture-command",
        "ok": outcome.native_outcome == "passed" and outcome.sealed,
        "qualification_status": f"{args.mode}-without-telemetry",
        "attempt_slot_id": outcome.attempt_slot_id,
        "experiment_run_id": outcome.experiment_run_id,
        "native_outcome": outcome.native_outcome,
        "reason_code": outcome.reason_code,
        "sealed": outcome.sealed,
        "exit_code": outcome.exit_code,
        "timed_out": outcome.timed_out,
        "submission_blocked": outcome.submission_blocked,
        "telemetry_healthy": outcome.telemetry_healthy,
    }
    if outcome.seal_verification is not None:
        summary["protected_artifact_id"] = outcome.seal_verification.get(
            "protected_artifact_id"
        )
        summary["raw_manifest_sha256"] = outcome.seal_verification.get(
            "raw_manifest_sha256"
        )
    _emit(sys.stdout, summary)
    return 0 if summary["ok"] else 10


def _cmd_sanitize_recovery_stage(args: argparse.Namespace) -> int:
    recovery_artifact = _private_directory(Path(args.recovery_artifact))
    control_artifact = _private_directory(Path(args.control_artifact))
    stage_parent = _private_directory(Path(args.stage_parent))
    try:
        stage_name = validate_safe_relative_path(args.stage_name)
    except ContractError:
        raise OperatorCliError("PROTECTED_PATH_INVALID", 4) from None
    if "/" in stage_name:
        raise OperatorCliError("PROTECTED_PATH_INVALID", 4)
    stage = _fresh_private_child(stage_parent / stage_name)
    _require_disjoint_paths(stage, recovery_artifact, control_artifact)
    projection = project_verified_recovery_supplement(
        recovery_artifact=recovery_artifact,
        control_artifact=control_artifact,
    )
    digests = write_projection_stage(stage, projection)
    return _success(
        "sanitize-recovery-stage",
        stage_status="protected-review-stage-only",
        file_count=len(digests) + 1,
        claim_boundary_sha256=digests["claim-boundary.json"],
        recovery_supplement_sha256=digests["recovery-supplement.json"],
        sanitization_map_sha256=digests["sanitization-map.json"],
    )


def _cmd_review_recovery_stage(args: argparse.Namespace) -> int:
    stage = _private_directory(Path(args.stage))
    recovery_artifact = _private_directory(Path(args.recovery_artifact))
    control_artifact = _private_directory(Path(args.control_artifact))
    producer_role_id = _require_role_id(args.producer_role_id)
    reviewer_role_id = _require_role_id(args.reviewer_role_id)
    review = verify_projection_stage(
        stage,
        recovery_artifact=recovery_artifact,
        control_artifact=control_artifact,
        producer_role_id=producer_role_id,
        reviewer_role_id=reviewer_role_id,
    )
    return _success(
        "review-recovery-stage",
        review_id=review["review_id"],
        producer_role_id=review["producer_role_id"],
        reviewer_role_id=review["reviewer_role_id"],
        reviewed_at_utc=review["reviewed_at_utc"],
        result=review["result"],
        reason_code=review["reason_code"],
        checks=review["checks"],
    )


def _load_private_canonical_object(path: Path) -> dict[str, Any]:
    payload = _read_pinned_private_bytes(path)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise OperatorCliError("EVIDENCE_INPUT_INVALID", 5) from None
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise OperatorCliError("EVIDENCE_INPUT_INVALID", 5)
    return value


def _sanitizer_binding(args: argparse.Namespace) -> FinalizedSanitizerBinding:
    return FinalizedSanitizerBinding(
        projection_stage=_private_directory(Path(args.stage)),
        finalized_candidate_output=_private_directory(Path(args.finalized_candidate)),
        review_artifact=_private_directory(Path(args.review_artifact)),
        recovery_artifact=_private_directory(Path(args.recovery_artifact)),
        control_artifact=_private_directory(Path(args.control_artifact)),
        producer_role_id=_require_role_id(args.producer_role_id),
        reviewer_role_id=_require_role_id(args.reviewer_role_id),
        finalizer_role_id=_require_role_id(args.finalizer_role_id),
    )


def _external_evidence(
    args: argparse.Namespace, attestation: Mapping[str, Any]
) -> dict[str, Path]:
    fields = {
        "off_host_storage_evidence": Path(args.off_host_storage_evidence),
        "encryption_in_transit_evidence": Path(args.encryption_in_transit_evidence),
        "encryption_at_rest_evidence": Path(args.encryption_at_rest_evidence),
        "key_custody_evidence": Path(args.key_custody_evidence),
        "recovery_procedure_evidence": Path(args.recovery_procedure_evidence),
    }
    try:
        return {
            str(attestation[field]["reference_id"]): path
            for field, path in fields.items()
        }
    except (KeyError, TypeError):
        raise OperatorCliError("EVIDENCE_INPUT_INVALID", 5) from None


def _publication_evidence(
    args: argparse.Namespace,
) -> tuple[
    Path,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Path],
    FinalizedSanitizerBinding,
]:
    artifact = _private_directory(Path(args.artifact))
    receipt_store = AppendOnlyReceiptStore(_private_directory(Path(args.receipt_store)))
    attestation = _load_private_canonical_object(
        Path(args.external_recovery_attestation)
    )
    evidence = _external_evidence(args, attestation)
    sanitizer = _sanitizer_binding(args)
    return artifact, receipt_store.read_chain(), attestation, evidence, sanitizer


def _cmd_seal_projection_review(args: argparse.Namespace) -> int:
    stage = _private_directory(Path(args.stage))
    recovery_artifact = _private_directory(Path(args.recovery_artifact))
    control_artifact = _private_directory(Path(args.control_artifact))
    output = _fresh_private_child(Path(args.review_artifact_output))
    _require_disjoint_paths(output, stage, recovery_artifact, control_artifact)
    result = seal_projection_review(
        stage,
        output,
        recovery_artifact=recovery_artifact,
        control_artifact=control_artifact,
        producer_role_id=_require_role_id(args.producer_role_id),
        reviewer_role_id=_require_role_id(args.reviewer_role_id),
        review_id=_require_id(args.review_id, "review_id"),
        reviewed_at_utc=_require_timestamp(args.reviewed_at_utc),
    )
    review = result["review"]
    sealed = result["sealed_review_artifact"]
    return _success(
        "seal-projection-review",
        review_id=review["review_id"],
        producer_role_id=review["producer_role_id"],
        reviewer_role_id=review["reviewer_role_id"],
        reviewed_at_utc=review["reviewed_at_utc"],
        result=review["result"],
        reason_code=review["reason_code"],
        review_artifact_id=sealed["protected_artifact_id"],
        review_raw_manifest_sha256=sealed["raw_manifest_sha256"],
    )


def _cmd_finalize_publication_candidate(args: argparse.Namespace) -> int:
    stage = _private_directory(Path(args.stage))
    review_artifact = _private_directory(Path(args.review_artifact))
    recovery_artifact = _private_directory(Path(args.recovery_artifact))
    control_artifact = _private_directory(Path(args.control_artifact))
    output = _fresh_private_child(Path(args.finalized_candidate_output))
    _require_disjoint_paths(
        output, stage, review_artifact, recovery_artifact, control_artifact
    )
    result = finalize_projection_stage(
        stage,
        output,
        review_artifact,
        recovery_artifact=recovery_artifact,
        control_artifact=control_artifact,
        producer_role_id=_require_role_id(args.producer_role_id),
        reviewer_role_id=_require_role_id(args.reviewer_role_id),
        finalizer_role_id=_require_role_id(args.finalizer_role_id),
        finalized_at_utc=_require_timestamp(args.finalized_at_utc),
    )
    finalization = result["finalization"]
    return _success(
        "finalize-publication-candidate",
        publication_status="nonpublished-candidate",
        review_id=finalization["review_id"],
        finalization_id=finalization["finalization_id"],
        finalizer_role_id=finalization["finalizer_role_id"],
        finalized_at_utc=finalization["finalized_at_utc"],
        file_count=len(result["final_candidate_digests"]) + 1,
    )


def _cmd_verify_finalized_candidate(args: argparse.Namespace) -> int:
    binding = _sanitizer_binding(args)
    result = verify_finalized_projection(
        binding.projection_stage,
        binding.finalized_candidate_output,
        binding.review_artifact,
        recovery_artifact=binding.recovery_artifact,
        control_artifact=binding.control_artifact,
        producer_role_id=binding.producer_role_id,
        reviewer_role_id=binding.reviewer_role_id,
        finalizer_role_id=binding.finalizer_role_id,
    )
    return _success(
        "verify-finalized-candidate",
        publication_status="verified-nonpublished-candidate",
        review_id=result["review_id"],
        finalization_id=result["finalization_id"],
        producer_role_id=result["producer_role_id"],
        reviewer_role_id=result["reviewer_role_id"],
        finalizer_role_id=result["finalizer_role_id"],
        result=result["result"],
        reason_code=result["reason_code"],
    )


def _publication_candidate_binding(
    args: argparse.Namespace,
) -> PublicationCandidateBinding:
    return PublicationCandidateBinding(
        artifact=_private_directory(Path(args.publication_candidate_artifact)),
        campaign_id=_require_id(args.campaign_id, "campaign_id"),
        claim_key=_require_claim_key(args.claim_key),
        candidate_producer_role_id=_require_role_id(args.candidate_producer_role_id),
    )


def _cmd_seal_publication_candidate(args: argparse.Namespace) -> int:
    artifact, receipts, attestation, evidence, sanitizer = _publication_evidence(args)
    output = _fresh_private_child(Path(args.publication_candidate_output))
    _require_disjoint_paths(
        output,
        artifact,
        sanitizer.projection_stage,
        sanitizer.finalized_candidate_output,
        sanitizer.review_artifact,
        sanitizer.recovery_artifact,
        sanitizer.control_artifact,
    )
    result = seal_publication_candidate(
        output,
        campaign_id=_require_id(args.campaign_id, "campaign_id"),
        claim_key=_require_claim_key(args.claim_key),
        candidate_producer_role_id=_require_role_id(args.candidate_producer_role_id),
        created_at_utc=_require_timestamp(args.created_at_utc),
        artifact=artifact,
        receipts=receipts,
        external_recovery_attestation=attestation,
        external_evidence=evidence,
        sanitizer=sanitizer,
    )
    candidate = result["publication_candidate"]
    sealed = result["sealed_candidate_artifact"]
    return _success(
        "seal-publication-candidate",
        publication_status="sealed-nonpublished-candidate",
        candidate_id=candidate["candidate_id"],
        campaign_id=candidate["campaign_id"],
        claim_key=candidate["claim_key"],
        candidate_artifact_id=sealed["protected_artifact_id"],
        candidate_raw_manifest_sha256=sealed["raw_manifest_sha256"],
    )


def _cmd_verify_publication_candidate(args: argparse.Namespace) -> int:
    artifact, receipts, attestation, evidence, sanitizer = _publication_evidence(args)
    candidate = verify_publication_candidate(
        _publication_candidate_binding(args),
        artifact=artifact,
        receipts=receipts,
        external_recovery_attestation=attestation,
        external_evidence=evidence,
        sanitizer=sanitizer,
        now_utc=_require_timestamp(args.now_utc),
    )
    return _success(
        "verify-publication-candidate",
        publication_status="verified-nonpublished-candidate",
        candidate_id=candidate["candidate_id"],
        campaign_id=candidate["campaign_id"],
        claim_key=candidate["claim_key"],
    )


def _cmd_evaluate_publication(args: argparse.Namespace) -> int:
    artifact, receipts, attestation, evidence, sanitizer = _publication_evidence(args)
    result = evaluate_publication_eligibility(
        artifact=artifact,
        expected_protected_artifact_id=_require_id(
            args.expected_protected_artifact_id, "protected_artifact_id"
        ),
        expected_raw_manifest_sha256=_require_digest(args.expected_raw_manifest_sha256),
        expected_raw_manifest_size_bytes=args.expected_raw_manifest_size_bytes,
        receipts=receipts,
        external_recovery_attestation=attestation,
        external_evidence=evidence,
        now_utc=_require_timestamp(args.now_utc),
        sanitizer=sanitizer,
        publication_candidate=_publication_candidate_binding(args),
    )
    _emit(
        sys.stdout,
        {
            "command": "evaluate-publication",
            "ok": result.eligible,
            **result.as_dict(),
        },
    )
    return 0 if result.eligible else 11


def _cmd_publish_candidate(args: argparse.Namespace) -> int:
    artifact = _private_directory(Path(args.artifact))
    sanitizer = _sanitizer_binding(args)
    publication_candidate = _publication_candidate_binding(args)
    destination = _private_child_candidate(Path(args.destination))
    decision_artifact = _private_child_candidate(Path(args.decision_artifact_output))
    receipt_store = _receipt_store(
        args,
        artifact,
        sanitizer.projection_stage,
        sanitizer.finalized_candidate_output,
        sanitizer.review_artifact,
        sanitizer.recovery_artifact,
        sanitizer.control_artifact,
        publication_candidate.artifact,
        destination,
        decision_artifact,
    )
    _require_disjoint_paths(
        destination,
        decision_artifact,
        artifact,
        sanitizer.projection_stage,
        sanitizer.finalized_candidate_output,
        sanitizer.review_artifact,
        sanitizer.recovery_artifact,
        sanitizer.control_artifact,
        publication_candidate.artifact,
    )
    attestation = _load_private_canonical_object(
        Path(args.external_recovery_attestation)
    )
    evidence = _external_evidence(args, attestation)
    result = publish_candidate(
        destination,
        decision_artifact,
        destination_id=_require_id(args.destination_id, "destination_id"),
        evaluator_role_id=_require_role_id(args.evaluator_role_id),
        tool_source_sha256=_require_digest(args.tool_source_sha256),
        artifact=artifact,
        expected_protected_artifact_id=_require_id(
            args.expected_protected_artifact_id, "protected_artifact_id"
        ),
        expected_raw_manifest_sha256=_require_digest(args.expected_raw_manifest_sha256),
        expected_raw_manifest_size_bytes=args.expected_raw_manifest_size_bytes,
        receipt_store=receipt_store,
        external_recovery_attestation=attestation,
        external_evidence=evidence,
        sanitizer=sanitizer,
        publication_candidate=publication_candidate,
    )
    return _success(
        "publish-candidate",
        publication_status=result["publication_status"],
        destination_id=result["destination_id"],
        candidate_id=result["candidate_id"],
        decision_id=result["decision_id"],
        decision_artifact_id=result["decision_artifact_id"],
        decision_raw_manifest_sha256=result["decision_raw_manifest_sha256"],
        file_count=result["file_count"],
    )


def _add_sanitizer_binding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", required=True)
    parser.add_argument("--finalized-candidate", required=True)
    parser.add_argument("--review-artifact", required=True)
    parser.add_argument("--recovery-artifact", required=True)
    parser.add_argument("--control-artifact", required=True)
    parser.add_argument("--producer-role-id", required=True)
    parser.add_argument("--reviewer-role-id", required=True)
    parser.add_argument("--finalizer-role-id", required=True)


def _add_publication_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--receipt-store", required=True)
    parser.add_argument("--external-recovery-attestation", required=True)
    parser.add_argument("--off-host-storage-evidence", required=True)
    parser.add_argument("--encryption-in-transit-evidence", required=True)
    parser.add_argument("--encryption-at-rest-evidence", required=True)
    parser.add_argument("--key-custody-evidence", required=True)
    parser.add_argument("--recovery-procedure-evidence", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--claim-key", required=True)
    parser.add_argument("--candidate-producer-role-id", required=True)
    _add_sanitizer_binding_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="python -m tools.cuda_campaign")
    subparsers = parser.add_subparsers(dest="operator_command", required=True)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--host-id", required=True)
    probe.add_argument("--filesystem-path", required=True)
    probe.add_argument("--gpu-index", type=int, default=0)
    probe.add_argument("--nvidia-smi-path")
    probe.set_defaults(handler=_cmd_probe)

    verify = subparsers.add_parser("verify-seal")
    verify.add_argument("--artifact", required=True)
    verify.add_argument("--protected-artifact-id", required=True)
    verify.set_defaults(handler=_cmd_verify_seal)

    copy = subparsers.add_parser("copy-seal")
    copy.add_argument("--source", required=True)
    copy.add_argument("--destination", required=True)
    copy.add_argument("--operation-id", required=True)
    copy.add_argument("--protected-artifact-id", required=True)
    copy.add_argument("--copy-id", required=True)
    copy.add_argument("--failure-domain-id", required=True)
    copy.add_argument("--receipt-store", required=True)
    copy.add_argument("--receipt-journal", required=True)
    copy.add_argument("--issuer-role-id", required=True)
    copy_location = copy.add_mutually_exclusive_group(required=True)
    copy_location.add_argument(
        "--off-experiment-host", action="store_true", dest="off_experiment_host"
    )
    copy_location.add_argument(
        "--on-experiment-host", action="store_false", dest="off_experiment_host"
    )
    copy.set_defaults(handler=_cmd_copy_seal)

    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--source", required=True)
    retrieve.add_argument("--destination", required=True)
    retrieve.add_argument("--operation-id", required=True)
    retrieve.add_argument("--protected-artifact-id", required=True)
    retrieve.add_argument("--source-copy-id", required=True)
    retrieve.add_argument("--source-failure-domain-id", required=True)
    retrieve.add_argument("--expected-raw-manifest-sha256", required=True)
    retrieve.add_argument("--destination-restore-id", required=True)
    retrieve.add_argument("--receipt-store", required=True)
    retrieve.add_argument("--receipt-journal", required=True)
    retrieve.add_argument("--issuer-role-id", required=True)
    retrieve.set_defaults(handler=_cmd_retrieve)

    resume_operation = subparsers.add_parser(
        "resume-operation", aliases=["resume-receipt"]
    )
    resume_operation.add_argument("--source", required=True)
    resume_operation.add_argument("--destination", required=True)
    resume_operation.add_argument("--receipt-store", required=True)
    resume_operation.add_argument("--receipt-journal", required=True)
    resume_operation.add_argument("--operation-id", required=True)
    resume_operation.add_argument("--issuer-role-id", required=True)
    resume_operation.set_defaults(handler=_cmd_resume_operation)

    capture = subparsers.add_parser("capture-command")
    capture.add_argument("--state-root", required=True)
    capture.add_argument("--artifact-directory", required=True)
    capture.add_argument("--working-directory", required=True)
    capture.add_argument("--attempt-slot-id", required=True)
    capture.add_argument("--experiment-run-id", required=True)
    capture.add_argument("--retain-not-before-utc", required=True)
    capture.add_argument("--timeout-seconds", type=float, required=True)
    capture.add_argument(
        "--mode", choices=("qualifying", "nonqualifying", "setup"), required=True
    )
    capture.add_argument("--without-telemetry", action="store_true")
    capture.add_argument("command_argv", nargs=argparse.REMAINDER)
    capture.set_defaults(handler=_cmd_capture_command)

    sanitize = subparsers.add_parser("sanitize-recovery-stage")
    sanitize.add_argument("--recovery-artifact", required=True)
    sanitize.add_argument("--control-artifact", required=True)
    sanitize.add_argument("--stage-parent", required=True)
    sanitize.add_argument("--stage-name", required=True)
    sanitize.set_defaults(handler=_cmd_sanitize_recovery_stage)

    review = subparsers.add_parser("review-recovery-stage")
    review.add_argument("--stage", required=True)
    review.add_argument("--recovery-artifact", required=True)
    review.add_argument("--control-artifact", required=True)
    review.add_argument("--producer-role-id", required=True)
    review.add_argument("--reviewer-role-id", required=True)
    review.set_defaults(handler=_cmd_review_recovery_stage)

    sealed_review = subparsers.add_parser("seal-projection-review")
    sealed_review.add_argument("--stage", required=True)
    sealed_review.add_argument("--review-artifact-output", required=True)
    sealed_review.add_argument("--recovery-artifact", required=True)
    sealed_review.add_argument("--control-artifact", required=True)
    sealed_review.add_argument("--producer-role-id", required=True)
    sealed_review.add_argument("--reviewer-role-id", required=True)
    sealed_review.add_argument("--review-id", required=True)
    sealed_review.add_argument("--reviewed-at-utc", required=True)
    sealed_review.set_defaults(handler=_cmd_seal_projection_review)

    finalize_candidate = subparsers.add_parser("finalize-publication-candidate")
    finalize_candidate.add_argument("--stage", required=True)
    finalize_candidate.add_argument("--finalized-candidate-output", required=True)
    finalize_candidate.add_argument("--review-artifact", required=True)
    finalize_candidate.add_argument("--recovery-artifact", required=True)
    finalize_candidate.add_argument("--control-artifact", required=True)
    finalize_candidate.add_argument("--producer-role-id", required=True)
    finalize_candidate.add_argument("--reviewer-role-id", required=True)
    finalize_candidate.add_argument("--finalizer-role-id", required=True)
    finalize_candidate.add_argument("--finalized-at-utc", required=True)
    finalize_candidate.set_defaults(handler=_cmd_finalize_publication_candidate)

    verify_finalized = subparsers.add_parser("verify-finalized-candidate")
    _add_sanitizer_binding_arguments(verify_finalized)
    verify_finalized.set_defaults(handler=_cmd_verify_finalized_candidate)

    seal_candidate = subparsers.add_parser("seal-publication-candidate")
    _add_publication_evidence_arguments(seal_candidate)
    seal_candidate.add_argument("--publication-candidate-output", required=True)
    seal_candidate.add_argument("--created-at-utc", required=True)
    seal_candidate.set_defaults(handler=_cmd_seal_publication_candidate)

    verify_candidate = subparsers.add_parser("verify-publication-candidate")
    _add_publication_evidence_arguments(verify_candidate)
    verify_candidate.add_argument("--publication-candidate-artifact", required=True)
    verify_candidate.add_argument("--now-utc", required=True)
    verify_candidate.set_defaults(handler=_cmd_verify_publication_candidate)

    evaluate = subparsers.add_parser("evaluate-publication")
    _add_publication_evidence_arguments(evaluate)
    evaluate.add_argument("--publication-candidate-artifact", required=True)
    evaluate.add_argument("--expected-protected-artifact-id", required=True)
    evaluate.add_argument("--expected-raw-manifest-sha256", required=True)
    evaluate.add_argument("--expected-raw-manifest-size-bytes", type=int, required=True)
    evaluate.add_argument("--now-utc", required=True)
    evaluate.set_defaults(handler=_cmd_evaluate_publication)

    publish = subparsers.add_parser("publish-candidate")
    _add_publication_evidence_arguments(publish)
    publish.add_argument("--publication-candidate-artifact", required=True)
    publish.add_argument("--expected-protected-artifact-id", required=True)
    publish.add_argument("--expected-raw-manifest-sha256", required=True)
    publish.add_argument("--expected-raw-manifest-size-bytes", type=int, required=True)
    publish.add_argument("--destination", required=True)
    publish.add_argument("--destination-id", required=True)
    publish.add_argument("--decision-artifact-output", required=True)
    publish.add_argument("--evaluator-role-id", required=True)
    publish.add_argument("--tool-source-sha256", required=True)
    publish.set_defaults(handler=_cmd_publish_candidate)

    return parser


_Handler = Callable[[argparse.Namespace], int]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded operator command and return a stable process status."""

    try:
        args = build_parser().parse_args(argv)
        handler: _Handler = args.handler
        return handler(args)
    except OperatorCliError as error:
        return _failure(error.error_code, error.exit_code, **error.safe_fields)
    except ProbeFailure as error:
        return _failure(error.code, 3)
    except RetrievalError:
        return _failure("RETRIEVAL_FAILURE", 5)
    except ArtifactIntegrityError:
        return _failure("ARTIFACT_INTEGRITY_FAILURE", 5)
    except EvidenceStorageError:
        return _failure("STORAGE_OPERATION_FAILED", 5)
    except SanitizationError:
        return _failure("SANITIZATION_FAILURE", 6)
    except PublicationError:
        return _failure("PUBLICATION_REFUSED", 12)
    except CaptureHarnessError as error:
        reason_code = getattr(error, "reason_code", "CAPTURE_FAILURE")
        return _failure(reason_code, 7)
    except ContractError:
        return _failure("CONTRACT_VALIDATION_FAILED", 4)
    except PermissionError:
        return _failure("PROTECTED_PATH_INVALID", 4)
    except FileExistsError:
        return _failure("DESTINATION_NOT_FRESH", 4)
    except (OSError, ValueError, TypeError, KeyError):
        return _failure("OPERATION_FAILED", 70)
    except Exception:  # pragma: no cover - last-resort privacy boundary.
        return _failure("INTERNAL_ERROR", 70)


__all__ = ["OperatorCliError", "build_parser", "main"]
