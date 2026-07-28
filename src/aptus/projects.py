from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows only.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX only.
    msvcrt = None

from .local_store import (
    atomic_write_json,
    private_directory,
    quarantine_file,
    read_json_object,
    utc_now,
)
from .plan_contract import sha256_file, validate_bundle_manifest


PROJECT_SCHEMA_VERSION = "aptus.project.v1"
PROJECT_REVISION_SCHEMA_VERSION = "aptus.project-revision.v1"
CURRENT_PROJECT_SCHEMA_VERSION = "aptus.current-project.v1"
LEGACY_IMPORT_SCHEMA_VERSION = "aptus.legacy-project-import.v1"
REVISION_TRANSACTION_SCHEMA_VERSION = "aptus.project-revision-transaction.v1"

_PROJECT_ID = re.compile(r"^project_[0-9a-f]{32}$")
_REVISION_ID = re.compile(r"^revision_[0-9a-f]{32}$")
_PLAN_ID = re.compile(r"^plan_[0-9a-f]{20}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROJECT_REPOSITORY_THREAD_LOCK = threading.RLock()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _project_name(value: str) -> str:
    name = " ".join(value.strip().split())
    if not name or len(name) > 120 or any(ord(character) < 32 for character in name):
        raise ValueError("Project names must contain 1 to 120 printable characters.")
    return name


def _summary(revision: Mapping[str, Any]) -> dict[str, Any]:
    bundle = revision.get("bundle")
    validation = revision.get("validation")
    return {
        "revision_id": revision["revision_id"],
        "ordinal": revision["ordinal"],
        "created_at": revision["created_at"],
        "reason": revision["reason"],
        "plan_id": revision.get("plan_id"),
        "selected_candidate_id": revision.get("selected_candidate_id"),
        "bundle_dir": bundle.get("bundle_dir") if isinstance(bundle, dict) else None,
        "validation_state": (
            validation.get("state") if isinstance(validation, dict) else None
        ),
        "job_count": len(revision.get("job_ids", [])),
    }


def _durable_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    durable = dict(value)
    report = durable.get("report")
    if isinstance(report, dict):
        report = dict(report)
        report.pop("authorization_current", None)
        report.pop("authorization_error", None)
        report.pop("prelaunch_capacity_check", None)
        durable["report"] = report
    durable.pop("authorization_current", None)
    durable.pop("authorization_error", None)
    durable.pop("prelaunch_capacity_check", None)
    durable["training_authorization_current"] = False
    return durable


def _durable_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    durable = dict(value)
    if not durable:
        return durable
    bundle_dir = durable.get("bundle_dir")
    fingerprint = durable.get("artifact_fingerprint")
    if not isinstance(bundle_dir, str) or not bundle_dir.strip():
        raise ValueError("Compiled project revisions require a bundle directory.")
    if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
        raise ValueError(
            "Compiled project revisions require the bundle manifest SHA-256 fingerprint."
        )
    archive_path = durable.get("archive_path")
    if archive_path:
        archive_sha256 = durable.get("archive_sha256")
        archive_size_bytes = durable.get("archive_size_bytes")
        if not isinstance(archive_path, str):
            raise ValueError("Compiled project archive paths must be strings.")
        if not isinstance(archive_sha256, str) or not _SHA256.fullmatch(archive_sha256):
            raise ValueError(
                "Compiled project revisions require the archive SHA-256 fingerprint."
            )
        if (
            not isinstance(archive_size_bytes, int)
            or isinstance(archive_size_bytes, bool)
            or archive_size_bytes < 0
        ):
            raise ValueError(
                "Compiled project revisions require the archive size in bytes."
            )
    return durable


def _plan_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only compiler-introduced path portability from a plan snapshot."""

    normalized = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    dataset = normalized.get("dataset")
    if isinstance(dataset, dict):
        dataset.pop("source_path", None)
        dataset.pop("bundle_path", None)
        provenance = dataset.get("provenance")
        if isinstance(provenance, dict):
            provenance.pop("source", None)
    return normalized


class ProjectRepository:
    """Secure local repository for named projects and immutable revisions."""

    def __init__(self, state_root: Path) -> None:
        self.state_root = private_directory(state_root)
        self.root = private_directory(self.state_root / "projects")
        self.quarantine_root = self.state_root / "quarantine" / "projects"
        self.current_path = self.state_root / "current-project.json"
        self.import_receipt_path = self.state_root / "legacy-project-import.json"
        self.lock_path = self.state_root / ".projects.lock"
        self._lock = _PROJECT_REPOSITORY_THREAD_LOCK
        self._lock_depth = 0
        self._lock_file: BinaryIO | None = None

    @staticmethod
    def _acquire_file_lock(lock_file: BinaryIO) -> None:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            return
        if msvcrt is not None:  # pragma: no cover - Windows only.
            lock_file.seek(0)
            if not lock_file.read(1):
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            return
        raise RuntimeError("Cross-process project repository locking is unavailable.")

    @staticmethod
    def _release_file_lock(lock_file: BinaryIO) -> None:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:  # pragma: no cover - Windows only.
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    def _open_lock_file(self) -> BinaryIO:
        if self.lock_path.is_symlink():
            raise PermissionError(
                f"Aptus project repository locks cannot be symlinks: {self.lock_path}"
            )
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            return os.fdopen(descriptor, "a+b", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise

    @contextmanager
    def _repository_lock(self) -> Iterator[None]:
        """Serialize repository reads, recovery, and writes across processes."""

        with self._lock:
            if self._lock_depth == 0:
                lock_file = self._open_lock_file()
                try:
                    self._acquire_file_lock(lock_file)
                except BaseException:
                    lock_file.close()
                    raise
                self._lock_file = lock_file
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
                if self._lock_depth == 0:
                    lock_file = self._lock_file
                    self._lock_file = None
                    if lock_file is not None:
                        try:
                            self._release_file_lock(lock_file)
                        finally:
                            lock_file.close()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name != "posix":  # pragma: no cover - Windows only.
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        atomic_write_json(path, value, mode=0o600)
        self._fsync_directory(path.parent)

    def _durable_unlink(self, path: Path) -> None:
        path.unlink(missing_ok=True)
        self._fsync_directory(path.parent)

    @staticmethod
    def _require_project_id(project_id: str) -> str:
        if not _PROJECT_ID.fullmatch(project_id):
            raise KeyError(project_id)
        return project_id

    @staticmethod
    def _require_revision_id(revision_id: str) -> str:
        if not _REVISION_ID.fullmatch(revision_id):
            raise KeyError(revision_id)
        return revision_id

    def _project_root(self, project_id: str) -> Path:
        path = self.root / self._require_project_id(project_id)
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise PermissionError(
                f"Aptus project roots must be real directories: {path}"
            )
        return path

    def _manifest_path(self, project_id: str) -> Path:
        return self._project_root(project_id) / "project.json"

    def _revision_path(self, project_id: str, revision_id: str) -> Path:
        revisions = self._project_root(project_id) / "revisions"
        if revisions.is_symlink() or (revisions.exists() and not revisions.is_dir()):
            raise PermissionError(
                f"Aptus project revision roots must be real directories: {revisions}"
            )
        return revisions / f"{self._require_revision_id(revision_id)}.json"

    def _revision_transaction_path(self, project_id: str) -> Path:
        return self._project_root(project_id) / "revision-transaction.json"

    def _quarantine_project_file(self, path: Path, *, reason: str) -> None:
        if path.exists() or path.is_symlink():
            quarantine_file(path, self.quarantine_root, reason=reason)

    def _record_rejected_orphans(
        self,
        project_id: str,
        manifest: dict[str, Any],
        revision_ids: Iterable[str],
    ) -> None:
        rejected = [
            revision_id
            for revision_id in revision_ids
            if _REVISION_ID.fullmatch(revision_id)
            and revision_id not in manifest["revision_ids"]
        ]
        if not rejected:
            return
        previous = manifest.get("rejected_orphan_revision_ids", [])
        manifest["rejected_orphan_revision_ids"] = list(
            dict.fromkeys([*previous, *rejected])
        )
        manifest["recovered_at"] = utc_now()
        self._write_json(self._manifest_path(project_id), manifest)

    def _recover_indexed_revisions(
        self, project_id: str, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        safe_revision_ids: list[str] = []
        rejected_revision_ids: list[str] = []
        for revision_id in manifest["revision_ids"]:
            revision_path = self._revision_path(project_id, revision_id)
            try:
                revision = self._read_revision(project_id, revision_id)
                if revision.get("ordinal") != len(safe_revision_ids) + 1:
                    raise ValueError("The revision ordinal is inconsistent.")
                parent_id = revision.get("parent_revision_id")
                if parent_id is not None and parent_id not in safe_revision_ids:
                    raise ValueError(
                        "The revision parent is missing from the safe project chain."
                    )
            except (OSError, ValueError) as error:
                rejected_revision_ids.append(revision_id)
                self._quarantine_project_file(revision_path, reason=str(error))
                continue
            safe_revision_ids.append(revision_id)
        if not rejected_revision_ids:
            return manifest
        manifest["revision_ids"] = safe_revision_ids
        manifest["latest_revision_id"] = (
            safe_revision_ids[-1] if safe_revision_ids else None
        )
        manifest["updated_at"] = utc_now()
        manifest["recovered_at"] = manifest["updated_at"]
        previous = manifest.get("recovered_corrupt_revision_ids", [])
        manifest["recovered_corrupt_revision_ids"] = list(
            dict.fromkeys([*previous, *rejected_revision_ids])
        )
        self._write_json(self._manifest_path(project_id), manifest)
        return manifest

    def _recover_pending_revision(
        self, project_id: str, manifest: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        transaction_path = self._revision_transaction_path(project_id)
        if not transaction_path.exists() and not transaction_path.is_symlink():
            return manifest, False
        transaction: dict[str, Any] | None = None
        revision_id: Any = None
        try:
            transaction = read_json_object(
                transaction_path, "Aptus project revision transaction"
            )
            revision_id = transaction.get("revision_id")
            if (
                transaction.get("schema_version") != REVISION_TRANSACTION_SCHEMA_VERSION
                or transaction.get("project_id") != project_id
                or not isinstance(revision_id, str)
                or not _REVISION_ID.fullmatch(revision_id)
                or not isinstance(transaction.get("previous_revision_count"), int)
                or isinstance(transaction.get("previous_revision_count"), bool)
            ):
                raise ValueError("The pending project revision transaction is invalid.")
        except (OSError, ValueError) as error:
            if (
                isinstance(transaction, dict)
                and transaction.get("schema_version")
                == REVISION_TRANSACTION_SCHEMA_VERSION
                and transaction.get("project_id") == project_id
                and isinstance(revision_id, str)
                and _REVISION_ID.fullmatch(revision_id)
                and revision_id not in manifest["revision_ids"]
            ):
                rejected_ids = [revision_id]
            else:
                revisions_root = self._project_root(project_id) / "revisions"
                rejected_ids = [
                    path.stem
                    for path in sorted(revisions_root.glob("revision_*.json"))
                    if _REVISION_ID.fullmatch(path.stem)
                    and path.stem not in manifest["revision_ids"]
                ]
            self._record_rejected_orphans(project_id, manifest, rejected_ids)
            for rejected_id in rejected_ids:
                self._quarantine_project_file(
                    self._revision_path(project_id, rejected_id), reason=str(error)
                )
            self._quarantine_project_file(transaction_path, reason=str(error))
            return manifest, False

        revision_path = self._revision_path(project_id, revision_id)
        if not revision_path.exists() and not revision_path.is_symlink():
            self._durable_unlink(transaction_path)
            return manifest, False
        try:
            revision = self._read_revision(project_id, revision_id)
        except (OSError, ValueError) as error:
            self._quarantine_project_file(revision_path, reason=str(error))
            self._quarantine_project_file(transaction_path, reason=str(error))
            return manifest, False

        transaction_matches = (
            transaction.get("content_sha256") == revision.get("content_sha256")
            and transaction.get("base_revision_id")
            == revision.get("parent_revision_id")
            and transaction.get("ordinal") == revision.get("ordinal")
        )
        if not transaction_matches:
            if revision_id not in manifest["revision_ids"]:
                self._record_rejected_orphans(project_id, manifest, [revision_id])
                self._quarantine_project_file(
                    revision_path,
                    reason="The pending transaction does not identify its revision.",
                )
            self._quarantine_project_file(
                transaction_path,
                reason="The pending transaction does not identify its revision.",
            )
            return manifest, False

        revision_ids = manifest["revision_ids"]
        if revision_id not in revision_ids:
            previous_count = transaction["previous_revision_count"]
            previous_latest = transaction.get("previous_latest_revision_id")
            parent_id = revision.get("parent_revision_id")
            can_append = (
                len(revision_ids) == previous_count
                and manifest.get("latest_revision_id") == previous_latest
                and revision.get("ordinal") == previous_count + 1
                and (
                    (parent_id is None and not revision_ids)
                    or parent_id in revision_ids
                )
            )
            if not can_append:
                self._record_rejected_orphans(project_id, manifest, [revision_id])
                self._quarantine_project_file(
                    revision_path,
                    reason="The pending revision cannot extend the current manifest.",
                )
                self._quarantine_project_file(
                    transaction_path,
                    reason="The pending revision cannot extend the current manifest.",
                )
                return manifest, False
            revision_ids.append(revision_id)
            manifest["latest_revision_id"] = revision_id
            manifest["updated_at"] = revision["created_at"]
            recovered = manifest.get("recovered_transaction_revision_ids", [])
            manifest["recovered_transaction_revision_ids"] = list(
                dict.fromkeys([*recovered, revision_id])
            )
            self._write_json(self._manifest_path(project_id), manifest)

        latest = manifest.get("latest_revision_id")
        if isinstance(latest, str):
            self._repair_current_pointer(
                project_id,
                latest,
                force=True,
                selection_intent_at=(
                    str(transaction.get("prepared_at"))
                    if isinstance(transaction.get("prepared_at"), str)
                    else None
                ),
            )
        self._durable_unlink(transaction_path)
        return manifest, True

    def _recover_orphan_revisions(
        self, project_id: str, manifest: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        revisions_root = self._project_root(project_id) / "revisions"
        if not revisions_root.is_dir() or revisions_root.is_symlink():
            return manifest, False
        indexed = set(manifest["revision_ids"])
        rejected = {
            revision_id
            for revision_id in manifest.get("rejected_orphan_revision_ids", [])
            if isinstance(revision_id, str) and _REVISION_ID.fullmatch(revision_id)
        }
        orphans: dict[str, dict[str, Any]] = {}
        for path in sorted(revisions_root.glob("revision_*.json")):
            revision_id = path.stem
            if revision_id in indexed or not _REVISION_ID.fullmatch(revision_id):
                continue
            if revision_id in rejected:
                self._quarantine_project_file(
                    path,
                    reason=(
                        "The orphan revision belongs to a previously rejected "
                        "ambiguous recovery set."
                    ),
                )
                continue
            try:
                orphans[revision_id] = self._read_revision(project_id, revision_id)
            except (OSError, ValueError) as error:
                self._quarantine_project_file(path, reason=str(error))

        adopted: list[str] = []
        newly_rejected: list[str] = []
        while orphans:
            revision_ids = manifest["revision_ids"]
            expected_ordinal = len(revision_ids) + 1
            candidates = [
                revision_id
                for revision_id, revision in orphans.items()
                if revision.get("ordinal") == expected_ordinal
                and (
                    (revision.get("parent_revision_id") is None and not revision_ids)
                    or revision.get("parent_revision_id") in revision_ids
                )
            ]
            if len(candidates) != 1:
                reason = (
                    "The orphan revision cannot extend the manifest deterministically."
                )
                newly_rejected = sorted(orphans)
                self._record_rejected_orphans(project_id, manifest, newly_rejected)
                if adopted:
                    previous = manifest.get("recovered_orphan_revision_ids", [])
                    manifest["recovered_orphan_revision_ids"] = list(
                        dict.fromkeys([*previous, *adopted])
                    )
                    self._write_json(self._manifest_path(project_id), manifest)
                for revision_id in newly_rejected:
                    self._quarantine_project_file(
                        self._revision_path(project_id, revision_id), reason=reason
                    )
                break
            revision_id = candidates[0]
            revision = orphans.pop(revision_id)
            manifest["revision_ids"].append(revision_id)
            manifest["latest_revision_id"] = revision_id
            manifest["updated_at"] = revision["created_at"]
            adopted.append(revision_id)

        if not adopted:
            return manifest, False
        if not newly_rejected:
            previous = manifest.get("recovered_orphan_revision_ids", [])
            manifest["recovered_orphan_revision_ids"] = list(
                dict.fromkeys([*previous, *adopted])
            )
            self._write_json(self._manifest_path(project_id), manifest)
        return manifest, True

    def _read_manifest(self, project_id: str) -> dict[str, Any]:
        path = self._manifest_path(project_id)
        try:
            value = read_json_object(path, "Aptus project manifest")
            if value.get("schema_version") != PROJECT_SCHEMA_VERSION:
                raise ValueError("The project manifest schema is unsupported.")
            if value.get("project_id") != project_id:
                raise ValueError("The project ID does not match its directory.")
            _project_name(str(value.get("name", "")))
            revision_ids = value.get("revision_ids")
            if not isinstance(revision_ids, list) or any(
                not isinstance(item, str) or not _REVISION_ID.fullmatch(item)
                for item in revision_ids
            ):
                raise ValueError("The project revision index is invalid.")
            if len(revision_ids) != len(set(revision_ids)):
                raise ValueError("The project revision index contains duplicates.")
            latest = value.get("latest_revision_id")
            if latest is not None and (not revision_ids or latest != revision_ids[-1]):
                raise ValueError("The latest project revision is not the index tail.")
        except (OSError, ValueError) as error:
            if path.exists() or path.is_symlink():
                destination = quarantine_file(
                    path, self.quarantine_root, reason=str(error)
                )
                raise ValueError(
                    f"Project {project_id} is unavailable. Its manifest was preserved at {destination}: {error}"
                ) from error
            raise KeyError(project_id) from error

        value = self._recover_indexed_revisions(project_id, value)
        value, transaction_recovered = self._recover_pending_revision(project_id, value)
        value, orphan_recovered = self._recover_orphan_revisions(project_id, value)
        latest = value.get("latest_revision_id")
        if isinstance(latest, str):
            self._repair_current_pointer(
                project_id,
                latest,
                force=transaction_recovered or orphan_recovered,
            )
        return value

    def _read_revision(self, project_id: str, revision_id: str) -> dict[str, Any]:
        path = self._revision_path(project_id, revision_id)
        value = read_json_object(path, "Aptus project revision")
        if value.get("schema_version") != PROJECT_REVISION_SCHEMA_VERSION:
            raise ValueError("The project revision schema is unsupported.")
        if value.get("project_id") != project_id:
            raise ValueError("The revision project ID does not match its directory.")
        if value.get("revision_id") != revision_id:
            raise ValueError("The revision ID does not match its filename.")
        parent_id = value.get("parent_revision_id")
        if parent_id is not None and (
            not isinstance(parent_id, str) or not _REVISION_ID.fullmatch(parent_id)
        ):
            raise ValueError("The revision parent ID is invalid.")
        ordinal = value.get("ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal <= 0:
            raise ValueError("The revision ordinal is invalid.")
        if value.get("content_sha256") != _digest(
            {key: item for key, item in value.items() if key != "content_sha256"}
        ):
            raise ValueError("The immutable project revision digest does not match.")
        return value

    def _repair_current_pointer(
        self,
        project_id: str,
        revision_id: str,
        *,
        force: bool,
        selection_intent_at: str | None = None,
    ) -> None:
        pointer: dict[str, Any] | None = None
        try:
            pointer = read_json_object(
                self.current_path, "Aptus current-project pointer"
            )
            if pointer.get("schema_version") != CURRENT_PROJECT_SCHEMA_VERSION:
                raise ValueError("The current-project pointer schema is unsupported.")
        except ValueError as error:
            if not force:
                return
            self._quarantine_project_file(self.current_path, reason=str(error))
        if pointer is not None and pointer.get("project_id") != project_id:
            pointer_selected_at = pointer.get("selected_at")
            if (
                not force
                or not isinstance(pointer_selected_at, str)
                or selection_intent_at is None
                or selection_intent_at <= pointer_selected_at
            ):
                return
        if (
            pointer is not None
            and pointer.get("project_id") == project_id
            and pointer.get("revision_id") == revision_id
        ):
            return
        self._write_current(project_id, revision_id)

    def _write_current(self, project_id: str, revision_id: str) -> None:
        self._write_json(
            self.current_path,
            {
                "schema_version": CURRENT_PROJECT_SCHEMA_VERSION,
                "project_id": project_id,
                "revision_id": revision_id,
                "selected_at": utc_now(),
            },
        )

    def create(self, name: str) -> dict[str, Any]:
        with self._repository_lock():
            project_id = "project_" + uuid.uuid4().hex
            root = private_directory(self.root / project_id)
            private_directory(root / "revisions")
            now = utc_now()
            manifest = {
                "schema_version": PROJECT_SCHEMA_VERSION,
                "project_id": project_id,
                "name": _project_name(name),
                "created_at": now,
                "updated_at": now,
                "latest_revision_id": None,
                "revision_ids": [],
            }
            self._write_json(root / "project.json", manifest)
            return manifest

    def create_revision(
        self,
        project_id: str,
        *,
        reason: str,
        facts: Mapping[str, Any] | None = None,
        plan_id: str | None = None,
        plan_snapshot: Mapping[str, Any] | None = None,
        selected_candidate_id: str | None = None,
        bundle: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
        job_ids: Iterable[str] | None = None,
        base_revision_id: str | None = None,
        expected_latest_revision_id: str | None = None,
    ) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("Project revision reason is required.")
        with self._repository_lock():
            manifest = self._read_manifest(project_id)
            latest_revision_id = manifest.get("latest_revision_id")
            if (
                expected_latest_revision_id is not None
                and expected_latest_revision_id != latest_revision_id
            ):
                raise ValueError(
                    "The project changed after it was loaded. Refresh its history before saving another revision."
                )
            parent_id = (
                base_revision_id if base_revision_id is not None else latest_revision_id
            )
            base: dict[str, Any] = {}
            if parent_id is not None:
                if parent_id not in manifest["revision_ids"]:
                    raise ValueError("The base revision is not part of this project.")
                base = self._read_revision(project_id, parent_id)
            revision_id = "revision_" + uuid.uuid4().hex
            inherited_jobs = list(base.get("job_ids", []))
            if job_ids is not None:
                inherited_jobs = list(dict.fromkeys([*inherited_jobs, *job_ids]))
            revision: dict[str, Any] = {
                "schema_version": PROJECT_REVISION_SCHEMA_VERSION,
                "revision_id": revision_id,
                "project_id": project_id,
                "parent_revision_id": parent_id,
                "ordinal": len(manifest["revision_ids"]) + 1,
                "created_at": utc_now(),
                "reason": reason.strip(),
                "facts": dict(facts) if facts is not None else base.get("facts"),
                "plan_id": plan_id if plan_id is not None else base.get("plan_id"),
                "plan_snapshot": (
                    dict(plan_snapshot)
                    if plan_snapshot is not None
                    else base.get("plan_snapshot")
                ),
                "selected_candidate_id": (
                    selected_candidate_id
                    if selected_candidate_id is not None
                    else base.get("selected_candidate_id")
                ),
                "bundle": _durable_bundle(
                    bundle if bundle is not None else (base.get("bundle") or {})
                ),
                "validation": (
                    _durable_validation(validation)
                    if validation is not None
                    else base.get("validation")
                ),
                "job_ids": inherited_jobs,
                "training_authorization": {
                    "current": False,
                    "reason": "Training authorization is never durable. Submit a new confirmed train action against current evidence.",
                },
            }
            revision["content_sha256"] = _digest(revision)
            path = self._revision_path(project_id, revision_id)
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Project revision already exists: {path}")
            transaction_path = self._revision_transaction_path(project_id)
            if transaction_path.exists() or transaction_path.is_symlink():
                raise RuntimeError(
                    "A previous project revision transaction was not recovered."
                )
            self._write_json(
                transaction_path,
                {
                    "schema_version": REVISION_TRANSACTION_SCHEMA_VERSION,
                    "project_id": project_id,
                    "revision_id": revision_id,
                    "base_revision_id": parent_id,
                    "previous_latest_revision_id": latest_revision_id,
                    "previous_revision_count": len(manifest["revision_ids"]),
                    "ordinal": revision["ordinal"],
                    "content_sha256": revision["content_sha256"],
                    "prepared_at": utc_now(),
                },
            )
            self._write_json(path, revision)
            manifest["revision_ids"].append(revision_id)
            manifest["latest_revision_id"] = revision_id
            manifest["updated_at"] = revision["created_at"]
            self._write_json(self._manifest_path(project_id), manifest)
            self._write_current(project_id, revision_id)
            self._durable_unlink(transaction_path)
            return revision

    def list(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        with self._repository_lock():
            for path in sorted(self.root.glob("project_*/project.json")):
                project_id = path.parent.name
                try:
                    manifest = self._read_manifest(project_id)
                    latest_id = manifest.get("latest_revision_id")
                    latest = (
                        self._read_revision(project_id, latest_id)
                        if isinstance(latest_id, str)
                        else None
                    )
                except (KeyError, OSError, ValueError):
                    continue
                item = {
                    **manifest,
                    "revision_count": len(manifest["revision_ids"]),
                    "latest": _summary(latest) if latest is not None else None,
                }
                item.pop("revision_ids", None)
                result.append(item)
        return sorted(result, key=lambda item: item["updated_at"], reverse=True)

    def get(self, project_id: str) -> dict[str, Any]:
        with self._repository_lock():
            manifest = self._read_manifest(project_id)
            latest_id = manifest.get("latest_revision_id")
            latest = (
                self._read_revision(project_id, latest_id)
                if isinstance(latest_id, str)
                else None
            )
            return {
                **manifest,
                "revision_count": len(manifest["revision_ids"]),
                "latest_revision": latest,
            }

    def history(self, project_id: str) -> list[dict[str, Any]]:
        with self._repository_lock():
            manifest = self._read_manifest(project_id)
            revisions = [
                self._read_revision(project_id, revision_id)
                for revision_id in manifest["revision_ids"]
            ]
        return [_summary(item) for item in reversed(revisions)]

    def revision(self, project_id: str, revision_id: str) -> dict[str, Any]:
        with self._repository_lock():
            manifest = self._read_manifest(project_id)
            if revision_id not in manifest["revision_ids"]:
                raise KeyError(revision_id)
            return self._read_revision(project_id, revision_id)

    def validate_revision_artifacts(
        self, project_id: str, revision_id: str
    ) -> dict[str, Any]:
        """Validate the immutable plan and bundle binding for one revision."""

        with self._repository_lock():
            manifest = self._read_manifest(project_id)
            if revision_id not in manifest["revision_ids"]:
                raise KeyError(revision_id)
            revision = self._read_revision(project_id, revision_id)
            self._validate_recovery_artifacts(revision, verify_archive=False)
            return revision

    def current(self) -> dict[str, Any] | None:
        with self._repository_lock():
            try:
                pointer = read_json_object(
                    self.current_path, "Aptus current-project pointer"
                )
                if pointer.get("schema_version") != CURRENT_PROJECT_SCHEMA_VERSION:
                    raise ValueError(
                        "The current-project pointer schema is unsupported."
                    )
                project = self.get(str(pointer.get("project_id", "")))
                if pointer.get("revision_id") != project.get("latest_revision_id"):
                    latest = project.get("latest_revision_id")
                    if not isinstance(latest, str):
                        raise ValueError("The current-project pointer is stale.")
                    self._write_current(str(project["project_id"]), latest)
                return project
            except (KeyError, OSError, ValueError) as error:
                self._quarantine_project_file(self.current_path, reason=str(error))

            candidates: list[dict[str, Any]] = []
            for path in sorted(self.root.glob("project_*/project.json")):
                try:
                    manifest = self._read_manifest(path.parent.name)
                except (KeyError, OSError, ValueError):
                    continue
                if isinstance(manifest.get("latest_revision_id"), str):
                    candidates.append(manifest)
            if not candidates:
                return None
            manifest = max(
                candidates,
                key=lambda item: (str(item.get("updated_at", "")), item["project_id"]),
            )
            self._write_current(
                str(manifest["project_id"]), str(manifest["latest_revision_id"])
            )
            return self.get(str(manifest["project_id"]))

    def record_plan(
        self,
        *,
        project_id: str | None,
        project_name: str,
        facts: Mapping[str, Any],
        plan: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        with self._repository_lock():
            manifest = (
                self.create(project_name)
                if project_id is None
                else self._read_manifest(project_id)
            )
            plan_id = str(plan.get("plan_id", ""))
            recommended = plan.get("recommended")
            selected = (
                recommended.get("candidate_id")
                if isinstance(recommended, dict)
                else None
            )
            revision = self.create_revision(
                manifest["project_id"],
                reason="plan-created",
                facts=facts,
                plan_id=plan_id,
                plan_snapshot=plan,
                selected_candidate_id=selected,
                bundle={},
                validation={},
                job_ids=[],
            )
            return str(manifest["project_id"]), revision

    def _validate_recovery_artifacts(
        self, source: Mapping[str, Any], *, verify_archive: bool = True
    ) -> None:
        plan_id = source.get("plan_id")
        plan_snapshot = source.get("plan_snapshot")
        if plan_id is None:
            if plan_snapshot is not None:
                raise ValueError(
                    "The selected revision cannot be recovered because its plan identity is incomplete."
                )
        else:
            if not isinstance(plan_id, str) or not _PLAN_ID.fullmatch(plan_id):
                raise ValueError(
                    "The selected revision cannot be recovered because its plan ID is invalid."
                )
            if not isinstance(plan_snapshot, Mapping):
                raise ValueError(
                    "The selected revision cannot be recovered because its plan snapshot is missing."
                )
            if plan_snapshot.get("plan_id") != plan_id:
                raise ValueError(
                    "The selected revision cannot be recovered because its plan snapshot identifies a different plan."
                )
            plan_path = self.state_root / "plans" / f"{plan_id}.json"
            try:
                saved_plan = read_json_object(plan_path, "Aptus saved plan")
            except ValueError as error:
                raise ValueError(
                    "The selected revision cannot be recovered because its saved plan is unavailable."
                ) from error
            if saved_plan != dict(plan_snapshot):
                raise ValueError(
                    "The selected revision cannot be recovered because its saved plan no longer matches the immutable snapshot."
                )

        bundle = source.get("bundle")
        if bundle is None or bundle == {}:
            return
        if not isinstance(bundle, Mapping) or not isinstance(
            bundle.get("bundle_dir"), str
        ):
            raise ValueError(
                "The selected revision cannot be recovered because its bundle identity is invalid."
            )
        if not isinstance(plan_id, str) or not isinstance(plan_snapshot, Mapping):
            raise ValueError(
                "The selected revision cannot be recovered because its bundle is not bound to a plan snapshot."
            )

        bundle_reference = Path(bundle["bundle_dir"]).expanduser()
        if bundle_reference.is_symlink():
            raise ValueError(
                "The selected revision cannot be recovered because its bundle root is a symlink."
            )
        try:
            bundle_path = bundle_reference.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                "The selected revision cannot be recovered because its bundle is missing."
            ) from error
        if not bundle_path.is_dir():
            raise ValueError(
                "The selected revision cannot be recovered because its bundle is missing."
            )

        try:
            bundle_plan = read_json_object(
                bundle_path / "plan.json", "Aptus bundle plan"
            )
            manifest = read_json_object(
                bundle_path / "bundle-manifest.json", "Aptus bundle manifest"
            )
            manifest_errors = validate_bundle_manifest(bundle_path)
        except (OSError, ValueError) as error:
            raise ValueError(
                "The selected revision cannot be recovered because its bundle identity is unreadable."
            ) from error
        if manifest_errors:
            raise ValueError(
                "The selected revision cannot be recovered because its bundle manifest or file digests changed: "
                + " ".join(manifest_errors)
            )
        if (
            bundle_plan.get("plan_id") != plan_id
            or manifest.get("plan_id") != plan_id
            or _plan_identity(bundle_plan) != _plan_identity(plan_snapshot)
        ):
            raise ValueError(
                "The selected revision cannot be recovered because its bundle identifies a different plan."
            )

        candidate_id = source.get("selected_candidate_id")
        recommended = bundle_plan.get("recommended")
        bundle_candidate_id = (
            recommended.get("candidate_id")
            if isinstance(recommended, Mapping)
            else None
        )
        if candidate_id is not None and (
            manifest.get("candidate_id") != candidate_id
            or bundle_candidate_id != candidate_id
        ):
            raise ValueError(
                "The selected revision cannot be recovered because its bundle identifies a different candidate."
            )

        expected_manifest_digest = bundle.get("artifact_fingerprint")
        if not isinstance(expected_manifest_digest, str) or not _SHA256.fullmatch(
            expected_manifest_digest
        ):
            raise ValueError(
                "The selected revision cannot be recovered because its bundle fingerprint is missing."
            )
        if (
            sha256_file(bundle_path / "bundle-manifest.json")
            != expected_manifest_digest
        ):
            raise ValueError(
                "The selected revision cannot be recovered because its bundle manifest no longer matches the recorded artifact fingerprint."
            )
        validation = source.get("validation")
        report = validation.get("report") if isinstance(validation, Mapping) else None
        report_fingerprint = (
            report.get("artifact_fingerprint") if isinstance(report, Mapping) else None
        )
        if (
            report_fingerprint is not None
            and report_fingerprint != expected_manifest_digest
        ):
            raise ValueError(
                "The selected revision cannot be recovered because its validation report identifies a different artifact."
            )

        archive_path_value = bundle.get("archive_path")
        if verify_archive and archive_path_value:
            archive_sha256 = bundle.get("archive_sha256")
            archive_size_bytes = bundle.get("archive_size_bytes")
            if (
                not isinstance(archive_path_value, str)
                or not isinstance(archive_sha256, str)
                or not _SHA256.fullmatch(archive_sha256)
                or not isinstance(archive_size_bytes, int)
                or isinstance(archive_size_bytes, bool)
                or archive_size_bytes < 0
            ):
                raise ValueError(
                    "The selected revision cannot be recovered because its archive identity is incomplete."
                )
            archive_reference = Path(archive_path_value).expanduser()
            if archive_reference.is_symlink():
                raise ValueError(
                    "The selected revision cannot be recovered because its archive is a symlink."
                )
            try:
                archive_path = archive_reference.resolve(strict=True)
            except OSError as error:
                raise ValueError(
                    "The selected revision cannot be recovered because its archive is missing."
                ) from error
            if (
                not archive_path.is_file()
                or archive_path.stat().st_size != archive_size_bytes
                or sha256_file(archive_path) != archive_sha256
            ):
                raise ValueError(
                    "The selected revision cannot be recovered because its archive no longer matches the recorded identity."
                )

    def recover(self, project_id: str, revision_id: str) -> dict[str, Any]:
        with self._repository_lock():
            source = self.revision(project_id, revision_id)
            self._validate_recovery_artifacts(source)
            return self.create_revision(
                project_id,
                reason=f"recovered-from:{revision_id}",
                facts=source.get("facts"),
                plan_id=source.get("plan_id"),
                plan_snapshot=source.get("plan_snapshot"),
                selected_candidate_id=source.get("selected_candidate_id"),
                bundle=source.get("bundle"),
                validation=source.get("validation"),
                job_ids=source.get("job_ids", []),
                base_revision_id=revision_id,
            )

    def import_legacy(
        self,
        *,
        plans_dir: Path,
        current_bundle_path: Path,
        jobs: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        """Import the pre-project latest workspace once without deleting old state."""

        with self._repository_lock():
            if self.import_receipt_path.is_file() or self.list():
                return None
            bundle_reference: dict[str, Any] | None = None
            if current_bundle_path.is_file() and not current_bundle_path.is_symlink():
                try:
                    bundle_reference = read_json_object(
                        current_bundle_path, "Legacy current-bundle reference"
                    )
                except ValueError:
                    bundle_reference = None
            plan_payload: dict[str, Any] | None = None
            if isinstance(bundle_reference, dict):
                bundle_value = bundle_reference.get("bundle_dir")
                if isinstance(bundle_value, str):
                    plan_path = Path(bundle_value).resolve() / "plan.json"
                    if plan_path.is_file() and not plan_path.is_symlink():
                        try:
                            plan_payload = read_json_object(
                                plan_path, "Legacy bundle plan"
                            )
                        except ValueError:
                            plan_payload = None
            if plan_payload is None:
                candidates = sorted(
                    (
                        path
                        for path in plans_dir.glob("plan_*.json")
                        if path.is_file() and not path.is_symlink()
                    ),
                    key=lambda path: path.stat().st_mtime_ns,
                    reverse=True,
                )
                if candidates:
                    try:
                        plan_payload = read_json_object(
                            candidates[0], "Legacy saved plan"
                        )
                    except ValueError:
                        plan_payload = None
            if plan_payload is None:
                return None
            plan_id = str(plan_payload.get("plan_id", ""))
            name = f"Recovered {plan_id}" if plan_id else "Recovered Aptus workspace"
            project = self.create(name)
            recommended = plan_payload.get("recommended")
            selected = (
                recommended.get("candidate_id")
                if isinstance(recommended, dict)
                else None
            )
            facts = {
                key: plan_payload.get(key)
                for key in ("model", "dataset", "hardware", "target")
            }
            bundle: dict[str, Any] = {}
            if isinstance(bundle_reference, dict) and isinstance(
                bundle_reference.get("bundle_dir"), str
            ):
                bundle_path = Path(bundle_reference["bundle_dir"]).resolve()
                manifest_path = bundle_path / "bundle-manifest.json"
                if (
                    bundle_path.is_dir()
                    and not bundle_path.is_symlink()
                    and manifest_path.is_file()
                    and not manifest_path.is_symlink()
                    and not validate_bundle_manifest(bundle_path)
                ):
                    bundle = {
                        "bundle_dir": str(bundle_path),
                        "artifact_fingerprint": sha256_file(manifest_path),
                    }
                    archive_value = bundle_reference.get("archive_path")
                    if isinstance(archive_value, str) and archive_value:
                        archive_reference = Path(archive_value)
                        archive_path = archive_reference.resolve()
                        if (
                            not archive_reference.is_symlink()
                            and archive_path.is_file()
                        ):
                            bundle.update(
                                archive_path=str(archive_path),
                                archive_sha256=sha256_file(archive_path),
                                archive_size_bytes=archive_path.stat().st_size,
                            )
            matching_jobs = [
                str(item["id"])
                for item in jobs
                if isinstance(item.get("id"), str)
                and (not bundle or item.get("bundle_dir") == bundle.get("bundle_dir"))
            ]
            revision = self.create_revision(
                project["project_id"],
                reason="legacy-workspace-imported",
                facts=facts,
                plan_id=plan_id or None,
                plan_snapshot=plan_payload,
                selected_candidate_id=selected,
                bundle=bundle,
                validation={},
                job_ids=matching_jobs,
            )
            self._write_json(
                self.import_receipt_path,
                {
                    "schema_version": LEGACY_IMPORT_SCHEMA_VERSION,
                    "project_id": project["project_id"],
                    "revision_id": revision["revision_id"],
                    "imported_at": utc_now(),
                    "source_preserved": True,
                },
            )
            return revision
