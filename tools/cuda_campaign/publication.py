"""Fail-closed publication commit for exact CUDA campaign candidates.

This module never treats an older decision as authorization.  One invocation
reverifies the sealed candidate and every live eligibility input, records a
content-addressed operational decision, stages only reviewed allowlisted bytes,
repeats verification, and performs an atomic no-replace directory transition.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    PROCEDURAL_ROLE_ID_RE,
    canonical_json_bytes,
    compact_canonical_json_bytes,
    sha256_bytes,
    utc_now,
)
from .eligibility import (
    FinalizedSanitizerBinding,
    PUBLICATION_CANDIDATE_SCHEMA,
    PublicationCandidateBinding,
    PublicationEligibilityResult,
    _private_directory_bindings,
    _regular_file_bytes,
    evaluate_publication_eligibility,
    verify_publication_candidate,
)
from .storage import (
    AppendOnlyReceiptStore,
    RawArtifactWriter,
    _LockedReceiptTransaction,
    verify_sealed_artifact,
)


PUBLICATION_DECISION_FORMAT = "aptus.cuda-campaign-publication-decision.v1"
PUBLICATION_DECISION_BINDING_FORMAT = (
    "aptus.cuda-campaign-publication-decision-binding.v1"
)
PUBLICATION_CAPTURE_TOOL = {
    "name": "aptus-cuda-campaign-publisher",
    "version": "v1",
}
FINALIZED_CANDIDATE_ALLOWLIST = frozenset(
    {
        "SHA256SUMS",
        "claim-boundary.json",
        "finalization.json",
        "independent-review.json",
        "recovery-supplement.json",
        "review-bindings.json",
        "sanitization-map.json",
    }
)
PUBLIC_METADATA_FILES = frozenset(
    {
        "publication-candidate.json",
        "publication-decision-binding.json",
        "publication-decision.json",
    }
)
PUBLIC_CHECKSUM_NAME = "PUBLICATION-SHA256SUMS"

_DESTINATION_ID = re.compile(r"^destination_[0-9a-f]{32}$")
_ARTIFACT_ID = re.compile(r"^artifact_[0-9a-f]{32}$")
_CANDIDATE_ID = re.compile(r"^candidate_[0-9a-f]{32}$")
_CAMPAIGN_ID = re.compile(r"^campaign_[0-9a-f]{20}$")
_CLAIM_KEY = re.compile(r"^[a-z][a-z0-9-]{2,127}$")
_RECEIPT_ID = re.compile(r"^receipt_[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLICATION_CANDIDATE_FIELDS = {
    "schema_version",
    "candidate_id",
    "campaign_id",
    "claim_key",
    "candidate_producer_role_id",
    "created_at_utc",
    "primary_artifact",
    "receipt_chain",
    "external_recovery_attestation",
    "external_evidence",
    "sanitizer",
}
_PUBLICATION_DECISION_FIELDS = {
    "format_version",
    "decision_id",
    "evaluated_at_utc",
    "eligible",
    "reason_codes",
    "destination_id",
    "evaluator",
    "primary_artifact",
    "candidate",
    "receipt_chain",
    "external_recovery_attestation_sha256",
}
_DECISION_CANDIDATE_FIELDS = {
    "candidate_id",
    "campaign_id",
    "claim_key",
    "protected_artifact_id",
    "raw_manifest_sha256",
    "raw_manifest_size_bytes",
    "sealed_candidate_file_inventory",
    "finalized_candidate_file_inventory",
}
_INVENTORY_FIELDS = {"relative_path", "size_bytes", "sha256"}


class PublicationError(ValueError):
    """An exact publication authorization or commit boundary failed."""


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise PublicationError(f"{label} is invalid.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PublicationError(f"{label} is invalid.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PublicationError(f"{label} is invalid.")
    return parsed.astimezone(timezone.utc)


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PublicationError(f"{label} is invalid.")
    return value


def _require_positive_size(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PublicationError(f"{label} is invalid.")
    return value


def _validate_inventory(
    value: object, *, label: str, expected_names: frozenset[str] | None = None
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PublicationError(f"{label} is invalid.")
    normalized: list[dict[str, Any]] = []
    prior_name: str | None = None
    for item in value:
        if not isinstance(item, dict) or set(item) != _INVENTORY_FIELDS:
            raise PublicationError(f"{label} is invalid.")
        name = item["relative_path"]
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or name in {".", ".."}
            or (prior_name is not None and name <= prior_name)
        ):
            raise PublicationError(f"{label} is invalid.")
        _require_positive_size(item["size_bytes"], label=label)
        _require_digest(item["sha256"], label=label)
        normalized.append(dict(item))
        prior_name = name
    if (
        expected_names is not None
        and {item["relative_path"] for item in normalized} != expected_names
    ):
        raise PublicationError(f"{label} is invalid.")
    return normalized


def _candidate_content_id(record: Mapping[str, Any]) -> str:
    identity = {key: value for key, value in record.items() if key != "candidate_id"}
    return "candidate_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]


def _candidate_artifact_id(record: Mapping[str, Any]) -> str:
    return (
        "artifact_"
        + sha256_bytes(
            b"publication-candidate-artifact\0" + canonical_json_bytes(record)
        )[:32]
    )


def _validate_published_candidate(value: object) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != _PUBLICATION_CANDIDATE_FIELDS
        or value.get("schema_version") != PUBLICATION_CANDIDATE_SCHEMA
        or not isinstance(value.get("candidate_id"), str)
        or _CANDIDATE_ID.fullmatch(value["candidate_id"]) is None
        or value["candidate_id"] != _candidate_content_id(value)
        or not isinstance(value.get("campaign_id"), str)
        or _CAMPAIGN_ID.fullmatch(value["campaign_id"]) is None
        or not isinstance(value.get("claim_key"), str)
        or _CLAIM_KEY.fullmatch(value["claim_key"]) is None
        or not isinstance(value.get("candidate_producer_role_id"), str)
        or PROCEDURAL_ROLE_ID_RE.fullmatch(value["candidate_producer_role_id"]) is None
    ):
        raise PublicationError("Published publication candidate is invalid.")
    _parse_utc(value["created_at_utc"], label="Publication candidate timestamp")
    return dict(value)


def _decision_content_id(record: Mapping[str, Any]) -> str:
    identity = {key: value for key, value in record.items() if key != "decision_id"}
    return "decision_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]


def _decision_artifact_ids(record: Mapping[str, Any]) -> tuple[str, str]:
    seed = canonical_json_bytes(record)
    return (
        "artifact_" + sha256_bytes(b"publication-decision\0" + seed)[:32],
        "entry_" + sha256_bytes(b"publication-decision-entry\0" + seed)[:32],
    )


def _chain_binding(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact = [dict(receipt) for receipt in receipts]
    if not exact:
        raise PublicationError("Publication requires a nonempty receipt chain.")
    return {
        "ordered_receipt_ids": [receipt["receipt_id"] for receipt in exact],
        "head_receipt_id": exact[-1]["receipt_id"],
        "canonical_sha256": sha256_bytes(compact_canonical_json_bytes(exact)),
    }


def _require_fresh_path(path: Path, *, label: str) -> Path:
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise PublicationError(f"{label} cannot be checked safely.") from error
    else:
        raise FileExistsError(f"{label} already exists.")
    parent = path.parent
    try:
        metadata = parent.lstat()
    except OSError as error:
        raise PublicationError(f"{label} parent is unavailable.") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o700)
        or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
    ):
        raise PublicationError(f"{label} parent is not a private directory.")
    if not path.name or path.name in {".", ".."}:
        raise PublicationError(f"{label} name is invalid.")
    return path


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("publication staging write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename one directory without replacing an existing target."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if os.uname().sysname == "Darwin":
        renamex = getattr(libc, "renamex_np", None)
        if renamex is None:
            raise PublicationError("Atomic no-replace publication is unavailable.")
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        result = renamex(source_bytes, destination_bytes, 0x00000004)
    elif os.uname().sysname == "Linux":
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise PublicationError("Atomic no-replace publication is unavailable.")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source_bytes, -100, destination_bytes, 1)
    else:
        raise PublicationError("Atomic no-replace publication is unavailable.")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError("Publication destination already exists.")
        raise OSError(error_number, "atomic no-replace publication failed")


def _eligibility_and_candidate(
    *,
    artifact: Path,
    expected_protected_artifact_id: str,
    expected_raw_manifest_sha256: str,
    expected_raw_manifest_size_bytes: int,
    receipts: Sequence[Mapping[str, Any]],
    external_recovery_attestation: Mapping[str, Any],
    external_evidence: Mapping[str, Path],
    evaluated_at_utc: str,
    sanitizer: FinalizedSanitizerBinding,
    publication_candidate: PublicationCandidateBinding,
) -> tuple[PublicationEligibilityResult, dict[str, Any]]:
    candidate = verify_publication_candidate(
        publication_candidate,
        artifact=artifact,
        receipts=receipts,
        external_recovery_attestation=external_recovery_attestation,
        external_evidence=external_evidence,
        sanitizer=sanitizer,
        now_utc=evaluated_at_utc,
    )
    eligibility = evaluate_publication_eligibility(
        artifact=artifact,
        expected_protected_artifact_id=expected_protected_artifact_id,
        expected_raw_manifest_sha256=expected_raw_manifest_sha256,
        expected_raw_manifest_size_bytes=expected_raw_manifest_size_bytes,
        receipts=receipts,
        external_recovery_attestation=external_recovery_attestation,
        external_evidence=external_evidence,
        now_utc=evaluated_at_utc,
        sanitizer=sanitizer,
        publication_candidate=publication_candidate,
    )
    if not eligibility.eligible or eligibility.reason_codes:
        raise PublicationError("Live publication eligibility did not pass.")
    return eligibility, candidate


def _seal_decision_artifact(
    output: Path,
    *,
    decision: Mapping[str, Any],
    source_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_id, entry_id = _decision_artifact_ids(decision)
    writer = RawArtifactWriter(
        output,
        protected_artifact_id=artifact_id,
        record_kind="legacy-recovery",
        identity_bindings={
            "purpose": "publication-operational-decision",
            "decision_id": decision["decision_id"],
            "destination_id": decision["destination_id"],
            "evaluator_role_id": decision["evaluator"]["role_id"],
        },
        capture_tool=dict(PUBLICATION_CAPTURE_TOOL),
        source_bindings={
            "candidate_raw_manifest_sha256": decision["candidate"][
                "raw_manifest_sha256"
            ],
            "finalized_candidate_inventory_sha256": sha256_bytes(
                compact_canonical_json_bytes(
                    decision["candidate"]["finalized_candidate_file_inventory"]
                )
            ),
            "receipt_chain_sha256": decision["receipt_chain"]["canonical_sha256"],
            "tool_source_sha256": decision["evaluator"]["capture_tool"][
                "source_sha256"
            ],
            "primary_raw_manifest_sha256": decision["primary_artifact"][
                "raw_manifest_sha256"
            ],
            "destination_id": decision["destination_id"],
        },
        provisional_retain_not_before_utc=source_artifact["manifest"][
            "provisional_retain_not_before_utc"
        ],
        required_role_bindings={"publication-decision": entry_id},
    )
    writer.write_payload(
        canonical_json_bytes(dict(decision)),
        "publication-decision.json",
        role="publication-decision",
        media_type="application/json",
        entry_id=entry_id,
        captured_at_utc=decision["evaluated_at_utc"],
    )
    return writer.seal()


def _snapshot_public_payloads(
    *,
    finalized_candidate: Path,
    finalized_inventory: Sequence[Mapping[str, Any]],
    publication_candidate_artifact: Path,
    candidate_inventory: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> dict[str, bytes]:
    """Read the exact reviewed bytes before the final authorization pass."""

    payloads: dict[str, bytes] = {}
    observed_finalized: list[dict[str, Any]] = []
    for name in sorted(FINALIZED_CANDIDATE_ALLOWLIST):
        payload, digest, _identity = _regular_file_bytes(finalized_candidate / name)
        payloads[name] = payload
        observed_finalized.append(
            {
                "relative_path": name,
                "size_bytes": len(payload),
                "sha256": digest,
            }
        )
    if observed_finalized != list(finalized_inventory):
        raise PublicationError("Finalized candidate changed while snapshotting.")
    candidate_payload, candidate_digest, _identity = _regular_file_bytes(
        publication_candidate_artifact / "publication-candidate.json"
    )
    if candidate_payload != canonical_json_bytes(dict(candidate)):
        raise PublicationError("Publication candidate changed while snapshotting.")
    candidate_entry = next(
        (
            item
            for item in candidate_inventory
            if item.get("relative_path") == "publication-candidate.json"
        ),
        None,
    )
    if candidate_entry != {
        "relative_path": "publication-candidate.json",
        "size_bytes": len(candidate_payload),
        "sha256": candidate_digest,
    }:
        raise PublicationError("Publication candidate seal changed while snapshotting.")
    payloads["publication-candidate.json"] = candidate_payload
    return payloads


def _stage_public_bytes(
    stage: Path,
    *,
    public_payloads: Mapping[str, bytes],
    decision: Mapping[str, Any],
    decision_seal: Mapping[str, Any],
) -> None:
    expected_payloads = FINALIZED_CANDIDATE_ALLOWLIST | {"publication-candidate.json"}
    if set(public_payloads) != expected_payloads or any(
        not isinstance(payload, bytes) for payload in public_payloads.values()
    ):
        raise PublicationError("Publication snapshot inventory is invalid.")
    os.mkdir(stage, 0o700)
    stage.chmod(0o700)
    payloads = dict(public_payloads)
    decision_payload = canonical_json_bytes(dict(decision))
    payloads["publication-decision.json"] = decision_payload
    decision_binding = {
        "format_version": PUBLICATION_DECISION_BINDING_FORMAT,
        "decision_id": decision["decision_id"],
        "decision_artifact_id": decision_seal["protected_artifact_id"],
        "decision_raw_manifest_sha256": decision_seal["raw_manifest_sha256"],
        "decision_raw_manifest_size_bytes": decision_seal["raw_manifest_size_bytes"],
    }
    payloads["publication-decision-binding.json"] = canonical_json_bytes(
        decision_binding
    )
    for name in sorted(payloads):
        _write_exclusive(stage / name, payloads[name])
    checksum = "".join(
        f"{sha256_bytes(payloads[name])}  {name}\n" for name in sorted(payloads)
    ).encode("utf-8")
    _write_exclusive(stage / PUBLIC_CHECKSUM_NAME, checksum)
    _fsync_directory(stage)


def _pin_private_directory(path: Path, *, label: str) -> tuple[int, tuple[int, int]]:
    """Open one private directory and bind its path to the open inode."""

    try:
        before = path.lstat()
    except OSError as error:
        raise PublicationError(f"{label} is unavailable.") from error
    if (
        not stat.S_ISDIR(before.st_mode)
        or (os.name == "posix" and stat.S_IMODE(before.st_mode) != 0o700)
        or (hasattr(os, "getuid") and before.st_uid != os.getuid())
    ):
        raise PublicationError(f"{label} is not a private directory.")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PublicationError(f"{label} cannot be pinned safely.") from error
    try:
        opened = os.fstat(descriptor)
        after = path.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise PublicationError(f"{label} changed while pinning.")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, (opened.st_dev, opened.st_ino)


def _rollback_failed_publication(
    destination: Path,
    stage: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Atomically remove a failed commit from the public destination."""

    try:
        committed = destination.lstat()
    except OSError as error:
        raise PublicationError("Failed publication destination disappeared.") from error
    if (committed.st_dev, committed.st_ino) != expected_identity:
        raise PublicationError("Failed publication destination identity changed.")
    rejected: Path | None = None
    rollback_error: BaseException | None = None
    for _attempt in range(8):
        candidate = stage.parent / f"{stage.name}-rejected-{os.urandom(16).hex()}"
        try:
            _atomic_no_replace(destination, candidate)
        except FileExistsError:
            continue
        except BaseException as error:
            rollback_error = error
            break
        rejected = candidate
        try:
            _fsync_directory(destination.parent)
        except BaseException as error:
            rollback_error = error
        break
    if rejected is None:
        raise PublicationError(
            "Failed publication could not be removed from the public destination."
        ) from rollback_error
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise PublicationError("Publication rollback could not be verified.") from error
    else:
        raise PublicationError("Publication rollback left a public destination.")
    try:
        rejected_metadata = rejected.lstat()
    except OSError as error:
        raise PublicationError(
            "Publication rollback evidence is unavailable."
        ) from error
    if (rejected_metadata.st_dev, rejected_metadata.st_ino) != expected_identity:
        raise PublicationError("Publication rollback evidence identity changed.")
    if rollback_error is not None:
        raise PublicationError(
            "Publication rollback reached verified absence but was not durable."
        ) from rollback_error


def _atomic_boundary_identity(path: Path) -> tuple[int, int] | None:
    """Return one path identity without treating absence as an error."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise PublicationError(
            "Atomic publication outcome could not be reconciled."
        ) from error
    return metadata.st_dev, metadata.st_ino


def _verify_decision_artifact(
    path: Path, *, decision: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the protected authorization anchor for one public decision."""

    sealed = verify_sealed_artifact(path)
    manifest = sealed["manifest"]
    artifact_id, entry_id = _decision_artifact_ids(decision)
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 1:
        raise PublicationError("Publication decision artifact inventory is invalid.")
    entry = entries[0]
    decision_payload = canonical_json_bytes(dict(decision))
    expected_identity = {
        "purpose": "publication-operational-decision",
        "decision_id": decision["decision_id"],
        "destination_id": decision["destination_id"],
        "evaluator_role_id": decision["evaluator"]["role_id"],
    }
    expected_sources = {
        "candidate_raw_manifest_sha256": decision["candidate"]["raw_manifest_sha256"],
        "finalized_candidate_inventory_sha256": sha256_bytes(
            compact_canonical_json_bytes(
                decision["candidate"]["finalized_candidate_file_inventory"]
            )
        ),
        "receipt_chain_sha256": decision["receipt_chain"]["canonical_sha256"],
        "tool_source_sha256": decision["evaluator"]["capture_tool"]["source_sha256"],
        "primary_raw_manifest_sha256": decision["primary_artifact"][
            "raw_manifest_sha256"
        ],
        "destination_id": decision["destination_id"],
    }
    artifact_payload, payload_digest, _identity = _regular_file_bytes(
        path / "publication-decision.json"
    )
    if (
        artifact_payload != decision_payload
        or payload_digest != entry.get("sha256")
        or len(artifact_payload) != entry.get("size_bytes")
        or sealed["protected_artifact_id"] != artifact_id
        or manifest.get("record_kind") != "legacy-recovery"
        or entry.get("entry_id") != entry_id
        or entry.get("role") != "publication-decision"
        or entry.get("relative_path") != "publication-decision.json"
        or entry.get("media_type") != "application/json"
        or entry.get("captured_at_utc") != decision["evaluated_at_utc"]
        or manifest.get("identity_bindings") != expected_identity
        or manifest.get("capture_tool") != PUBLICATION_CAPTURE_TOOL
        or manifest.get("source_bindings") != expected_sources
        or manifest.get("required_role_bindings") != {"publication-decision": entry_id}
    ):
        raise PublicationError("Publication decision artifact is not exact.")
    return sealed


def verify_published_output(path: Path, *, decision_artifact: Path) -> dict[str, Any]:
    """Verify a public packet against its protected sealed authorization."""

    inventory = _private_directory_bindings(path, label="published output")
    expected_names = (
        FINALIZED_CANDIDATE_ALLOWLIST | PUBLIC_METADATA_FILES | {PUBLIC_CHECKSUM_NAME}
    )
    if {item["relative_path"] for item in inventory} != expected_names:
        raise PublicationError("Published output has an unexpected file inventory.")
    payloads: dict[str, bytes] = {}
    for name in sorted(expected_names - {PUBLIC_CHECKSUM_NAME}):
        payload, _digest, _identity = _regular_file_bytes(path / name)
        payloads[name] = payload
    expected_checksum = "".join(
        f"{sha256_bytes(payloads[name])}  {name}\n" for name in sorted(payloads)
    ).encode("utf-8")
    checksum_payload, _digest, _identity = _regular_file_bytes(
        path / PUBLIC_CHECKSUM_NAME
    )
    if checksum_payload != expected_checksum:
        raise PublicationError("Published output checksums are not exact.")
    try:
        candidate = _validate_published_candidate(
            json.loads(payloads["publication-candidate.json"])
        )
        decision = json.loads(payloads["publication-decision.json"])
        decision_binding = json.loads(payloads["publication-decision-binding.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationError("Published metadata is unreadable.") from error
    for value, name in (
        (candidate, "publication-candidate.json"),
        (decision, "publication-decision.json"),
        (decision_binding, "publication-decision-binding.json"),
    ):
        if not isinstance(value, dict) or canonical_json_bytes(value) != payloads[name]:
            raise PublicationError("Published metadata is not canonical.")
    if not isinstance(decision, dict) or set(decision) != _PUBLICATION_DECISION_FIELDS:
        raise PublicationError("Published decision cross-binding is invalid.")
    decision_candidate = decision.get("candidate")
    evaluator = decision.get("evaluator")
    capture_tool = (
        evaluator.get("capture_tool") if isinstance(evaluator, dict) else None
    )
    primary = decision.get("primary_artifact")
    receipt_chain = decision.get("receipt_chain")
    if (
        decision.get("format_version") != PUBLICATION_DECISION_FORMAT
        or decision.get("decision_id") != _decision_content_id(decision)
        or decision.get("eligible") is not True
        or decision.get("reason_codes") != []
        or not isinstance(decision.get("destination_id"), str)
        or _DESTINATION_ID.fullmatch(decision["destination_id"]) is None
        or not isinstance(evaluator, dict)
        or set(evaluator) != {"role_id", "capture_tool"}
        or not isinstance(evaluator.get("role_id"), str)
        or PROCEDURAL_ROLE_ID_RE.fullmatch(evaluator["role_id"]) is None
        or not isinstance(capture_tool, dict)
        or set(capture_tool) != {"name", "version", "source_sha256"}
        or capture_tool.get("name") != PUBLICATION_CAPTURE_TOOL["name"]
        or capture_tool.get("version") != PUBLICATION_CAPTURE_TOOL["version"]
        or not isinstance(decision_candidate, dict)
        or set(decision_candidate) != _DECISION_CANDIDATE_FIELDS
        or not isinstance(primary, dict)
        or set(primary)
        != {
            "protected_artifact_id",
            "raw_manifest_sha256",
            "raw_manifest_size_bytes",
        }
        or not isinstance(receipt_chain, dict)
        or set(receipt_chain)
        != {"ordered_receipt_ids", "head_receipt_id", "canonical_sha256"}
        or not isinstance(decision_binding, dict)
        or set(decision_binding)
        != {
            "format_version",
            "decision_id",
            "decision_artifact_id",
            "decision_raw_manifest_sha256",
            "decision_raw_manifest_size_bytes",
        }
        or decision_binding.get("format_version") != PUBLICATION_DECISION_BINDING_FORMAT
        or decision_binding.get("decision_id") != decision.get("decision_id")
    ):
        raise PublicationError("Published decision cross-binding is invalid.")
    _parse_utc(decision["evaluated_at_utc"], label="Publication decision timestamp")
    for binding, label in (
        (primary, "Primary artifact"),
        (decision_candidate, "Candidate artifact"),
    ):
        artifact_id = binding.get("protected_artifact_id")
        if (
            not isinstance(artifact_id, str)
            or _ARTIFACT_ID.fullmatch(artifact_id) is None
        ):
            raise PublicationError(f"{label} binding is invalid.")
        _require_digest(binding.get("raw_manifest_sha256"), label=label)
        _require_positive_size(binding.get("raw_manifest_size_bytes"), label=label)
    _require_digest(capture_tool["source_sha256"], label="Publication tool digest")
    _require_digest(
        decision.get("external_recovery_attestation_sha256"),
        label="External recovery attestation digest",
    )
    ordered_receipts = receipt_chain["ordered_receipt_ids"]
    if (
        not isinstance(ordered_receipts, list)
        or not ordered_receipts
        or any(
            not isinstance(receipt_id, str) or _RECEIPT_ID.fullmatch(receipt_id) is None
            for receipt_id in ordered_receipts
        )
        or len(set(ordered_receipts)) != len(ordered_receipts)
        or receipt_chain["head_receipt_id"] != ordered_receipts[-1]
    ):
        raise PublicationError("Published receipt-chain binding is invalid.")
    _require_digest(receipt_chain["canonical_sha256"], label="Receipt chain digest")
    if (
        decision_candidate["candidate_id"] != candidate["candidate_id"]
        or decision_candidate["campaign_id"] != candidate["campaign_id"]
        or decision_candidate["claim_key"] != candidate["claim_key"]
        or decision_candidate["protected_artifact_id"]
        != _candidate_artifact_id(candidate)
    ):
        raise PublicationError("Published candidate cross-binding is invalid.")
    candidate_primary = candidate.get("primary_artifact")
    candidate_receipts = candidate.get("receipt_chain")
    candidate_attestation = candidate.get("external_recovery_attestation")
    candidate_sanitizer = candidate.get("sanitizer")
    if (
        not isinstance(candidate_primary, dict)
        or {
            key: candidate_primary.get(key)
            for key in (
                "protected_artifact_id",
                "raw_manifest_sha256",
                "raw_manifest_size_bytes",
            )
        }
        != primary
        or candidate_receipts != receipt_chain
        or not isinstance(candidate_attestation, dict)
        or candidate_attestation.get("canonical_sha256")
        != decision["external_recovery_attestation_sha256"]
        or not isinstance(candidate_sanitizer, dict)
    ):
        raise PublicationError("Published candidate provenance binding is invalid.")
    sealed_inventory = _validate_inventory(
        decision_candidate["sealed_candidate_file_inventory"],
        label="Sealed candidate inventory",
        expected_names=frozenset(
            {"publication-candidate.json", "raw-manifest.json", "SEALED.json"}
        ),
    )
    candidate_entry = next(
        item
        for item in sealed_inventory
        if item["relative_path"] == "publication-candidate.json"
    )
    if candidate_entry["size_bytes"] != len(
        payloads["publication-candidate.json"]
    ) or candidate_entry["sha256"] != sha256_bytes(
        payloads["publication-candidate.json"]
    ):
        raise PublicationError("Published candidate payload is not sealed exactly.")
    finalized_inventory = _validate_inventory(
        decision_candidate["finalized_candidate_file_inventory"],
        label="Finalized candidate inventory",
        expected_names=FINALIZED_CANDIDATE_ALLOWLIST,
    )
    if candidate_sanitizer.get("finalized_candidate_files") != finalized_inventory:
        raise PublicationError("Published candidate finalization binding is invalid.")
    actual_finalized = [
        item
        for item in inventory
        if item["relative_path"] in FINALIZED_CANDIDATE_ALLOWLIST
    ]
    if actual_finalized != finalized_inventory:
        raise PublicationError("Published reviewed bytes differ from the decision.")
    sealed_decision = _verify_decision_artifact(decision_artifact, decision=decision)
    expected_decision_binding = {
        "format_version": PUBLICATION_DECISION_BINDING_FORMAT,
        "decision_id": decision["decision_id"],
        "decision_artifact_id": sealed_decision["protected_artifact_id"],
        "decision_raw_manifest_sha256": sealed_decision["raw_manifest_sha256"],
        "decision_raw_manifest_size_bytes": sealed_decision["raw_manifest_size_bytes"],
    }
    if decision_binding != expected_decision_binding:
        raise PublicationError("Published decision artifact binding is invalid.")
    return {
        "decision_id": decision["decision_id"],
        "candidate_id": candidate["candidate_id"],
        "destination_id": decision["destination_id"],
        "file_count": len(inventory),
    }


def _publish_candidate_locked(
    destination: Path,
    decision_artifact_output: Path,
    *,
    destination_id: str,
    evaluator_role_id: str,
    tool_source_sha256: str,
    artifact: Path,
    expected_protected_artifact_id: str,
    expected_raw_manifest_sha256: str,
    expected_raw_manifest_size_bytes: int,
    receipt_store: AppendOnlyReceiptStore,
    receipt_transaction: _LockedReceiptTransaction,
    external_recovery_attestation: Mapping[str, Any],
    external_evidence: Mapping[str, Path],
    sanitizer: FinalizedSanitizerBinding,
    publication_candidate: PublicationCandidateBinding,
    commit_state: dict[str, Any],
) -> dict[str, Any]:
    """Publish under one authentic locked receipt transaction and two live passes."""

    if _DESTINATION_ID.fullmatch(destination_id) is None:
        raise PublicationError("Publication destination ID is invalid.")
    if PROCEDURAL_ROLE_ID_RE.fullmatch(evaluator_role_id) is None:
        raise PublicationError("Publication evaluator role ID is invalid.")
    if _SHA256.fullmatch(tool_source_sha256) is None:
        raise PublicationError("Publication tool source digest is invalid.")
    destination = _require_fresh_path(destination, label="Publication destination")
    decision_artifact_output = _require_fresh_path(
        decision_artifact_output, label="Publication decision artifact"
    )
    initial_now = utc_now()
    try:
        live_receipts = AppendOnlyReceiptStore._read_chain_locked(
            receipt_store, receipt_transaction
        )
    except (TypeError, ValueError) as error:
        raise PublicationError(
            "Publication requires a live locked receipt transaction."
        ) from error
    initial_eligibility, candidate = _eligibility_and_candidate(
        artifact=artifact,
        expected_protected_artifact_id=expected_protected_artifact_id,
        expected_raw_manifest_sha256=expected_raw_manifest_sha256,
        expected_raw_manifest_size_bytes=expected_raw_manifest_size_bytes,
        receipts=live_receipts,
        external_recovery_attestation=external_recovery_attestation,
        external_evidence=external_evidence,
        evaluated_at_utc=initial_now,
        sanitizer=sanitizer,
        publication_candidate=publication_candidate,
    )
    primary = verify_sealed_artifact(artifact)
    candidate_seal = verify_sealed_artifact(publication_candidate.artifact)
    candidate_inventory = _private_directory_bindings(
        publication_candidate.artifact, label="sealed publication candidate"
    )
    finalized_inventory = _private_directory_bindings(
        sanitizer.finalized_candidate_output,
        label="finalized publication candidate",
    )
    if {
        item["relative_path"] for item in finalized_inventory
    } != FINALIZED_CANDIDATE_ALLOWLIST or finalized_inventory != candidate["sanitizer"][
        "finalized_candidate_files"
    ]:
        raise PublicationError("Finalized candidate inventory is not exact.")
    receipt_binding = _chain_binding(live_receipts)
    if receipt_binding != candidate["receipt_chain"]:
        raise PublicationError("Live receipt chain differs from the sealed candidate.")
    public_payloads = _snapshot_public_payloads(
        finalized_candidate=sanitizer.finalized_candidate_output,
        finalized_inventory=finalized_inventory,
        publication_candidate_artifact=publication_candidate.artifact,
        candidate_inventory=candidate_inventory,
        candidate=candidate,
    )
    try:
        final_receipts = AppendOnlyReceiptStore._read_chain_locked(
            receipt_store, receipt_transaction
        )
    except (TypeError, ValueError) as error:
        raise PublicationError(
            "Publication receipt transaction became unavailable."
        ) from error
    if _chain_binding(final_receipts) != receipt_binding:
        raise PublicationError("Publication receipt chain moved before commit.")
    final_now = utc_now()
    if _parse_utc(final_now, label="Final publication time") < _parse_utc(
        initial_now, label="Initial publication time"
    ):
        raise PublicationError("Publication time moved backward before commit.")
    final_eligibility, final_candidate = _eligibility_and_candidate(
        artifact=artifact,
        expected_protected_artifact_id=expected_protected_artifact_id,
        expected_raw_manifest_sha256=expected_raw_manifest_sha256,
        expected_raw_manifest_size_bytes=expected_raw_manifest_size_bytes,
        receipts=final_receipts,
        external_recovery_attestation=external_recovery_attestation,
        external_evidence=external_evidence,
        evaluated_at_utc=final_now,
        sanitizer=sanitizer,
        publication_candidate=publication_candidate,
    )
    final_primary = verify_sealed_artifact(artifact)
    final_candidate_seal = verify_sealed_artifact(publication_candidate.artifact)
    if (
        (
            final_eligibility.eligible,
            final_eligibility.reason_codes,
            final_eligibility.protected_artifact_id,
            final_eligibility.raw_manifest_sha256,
        )
        != (
            initial_eligibility.eligible,
            initial_eligibility.reason_codes,
            initial_eligibility.protected_artifact_id,
            initial_eligibility.raw_manifest_sha256,
        )
        or final_candidate != candidate
        or any(
            final_primary[key] != primary[key]
            for key in (
                "protected_artifact_id",
                "raw_manifest_sha256",
                "raw_manifest_size_bytes",
            )
        )
        or any(
            final_candidate_seal[key] != candidate_seal[key]
            for key in (
                "protected_artifact_id",
                "raw_manifest_sha256",
                "raw_manifest_size_bytes",
            )
        )
        or _private_directory_bindings(
            publication_candidate.artifact, label="sealed publication candidate"
        )
        != candidate_inventory
        or _private_directory_bindings(
            sanitizer.finalized_candidate_output,
            label="finalized publication candidate",
        )
        != finalized_inventory
        or _chain_binding(final_receipts) != receipt_binding
    ):
        raise PublicationError("Publication inputs changed before commit.")

    # No eligible authorization artifact exists until the final live pass above
    # has succeeded.  The public payloads were snapshotted before the adversarial
    # boundary and were cross-checked against the unchanged final inventories.
    decision_without_id = {
        "format_version": PUBLICATION_DECISION_FORMAT,
        "evaluated_at_utc": final_eligibility.evaluated_at_utc,
        "eligible": True,
        "reason_codes": [],
        "destination_id": destination_id,
        "evaluator": {
            "role_id": evaluator_role_id,
            "capture_tool": {
                **PUBLICATION_CAPTURE_TOOL,
                "source_sha256": tool_source_sha256,
            },
        },
        "primary_artifact": {
            "protected_artifact_id": primary["protected_artifact_id"],
            "raw_manifest_sha256": primary["raw_manifest_sha256"],
            "raw_manifest_size_bytes": primary["raw_manifest_size_bytes"],
        },
        "candidate": {
            "candidate_id": candidate["candidate_id"],
            "campaign_id": candidate["campaign_id"],
            "claim_key": candidate["claim_key"],
            "protected_artifact_id": candidate_seal["protected_artifact_id"],
            "raw_manifest_sha256": candidate_seal["raw_manifest_sha256"],
            "raw_manifest_size_bytes": candidate_seal["raw_manifest_size_bytes"],
            "sealed_candidate_file_inventory": candidate_inventory,
            "finalized_candidate_file_inventory": finalized_inventory,
        },
        "receipt_chain": receipt_binding,
        "external_recovery_attestation_sha256": candidate[
            "external_recovery_attestation"
        ]["canonical_sha256"],
    }
    decision = {
        **decision_without_id,
        "decision_id": _decision_content_id(decision_without_id),
    }
    decision_seal = _seal_decision_artifact(
        decision_artifact_output,
        decision=decision,
        source_artifact=primary,
    )
    stage = destination.parent / f".publication-stage-{decision['decision_id']}"
    _require_fresh_path(stage, label="Publication staging directory")
    _stage_public_bytes(
        stage,
        public_payloads=public_payloads,
        decision=decision,
        decision_seal=decision_seal,
    )
    sealed_decision = verify_sealed_artifact(decision_artifact_output)
    if (
        sealed_decision["protected_artifact_id"]
        != decision_seal["protected_artifact_id"]
        or sealed_decision["raw_manifest_sha256"]
        != decision_seal["raw_manifest_sha256"]
        or sealed_decision["raw_manifest_size_bytes"]
        != decision_seal["raw_manifest_size_bytes"]
    ):
        raise PublicationError("Publication decision artifact changed before commit.")
    stage_descriptor, stage_identity = _pin_private_directory(
        stage, label="Publication staging directory"
    )
    try:
        verify_published_output(stage, decision_artifact=decision_artifact_output)
        pre_commit_receipts = AppendOnlyReceiptStore._read_chain_locked(
            receipt_store, receipt_transaction
        )
        if _chain_binding(pre_commit_receipts) != receipt_binding:
            raise PublicationError(
                "Publication receipt chain moved before the commit boundary."
            )
        _require_fresh_path(destination, label="Publication destination")
        renamed = False
        try:
            try:
                _atomic_no_replace(stage, destination)
            except BaseException as atomic_error:
                observed_stage_identity = _atomic_boundary_identity(stage)
                observed_destination_identity = _atomic_boundary_identity(destination)
                if observed_destination_identity == stage_identity:
                    renamed = True
                    commit_state.update(
                        {
                            "destination": destination,
                            "stage": stage,
                            "identity": stage_identity,
                        }
                    )
                elif observed_stage_identity != stage_identity:
                    if observed_destination_identity is None:
                        raise PublicationError(
                            "Atomic publication outcome is invalid; public absence "
                            "was verified."
                        ) from atomic_error
                    raise PublicationError(
                        "Atomic publication outcome is ambiguous; an unrelated "
                        "destination was preserved."
                    ) from atomic_error
                raise
            renamed = True
            committed_identity = stage_identity
            commit_state.update(
                {
                    "destination": destination,
                    "stage": stage,
                    "identity": committed_identity,
                }
            )
            committed = destination.lstat()
            committed_identity = (committed.st_dev, committed.st_ino)
            commit_state["identity"] = committed_identity
            if committed_identity != stage_identity:
                raise PublicationError(
                    "Committed publication is not the verified staging directory."
                )
            _fsync_directory(destination.parent)
            committed = destination.lstat()
            committed_identity = (committed.st_dev, committed.st_ino)
            commit_state["identity"] = committed_identity
            if committed_identity != stage_identity:
                raise PublicationError(
                    "Committed publication is not the verified staging directory."
                )
            verified_output = verify_published_output(
                destination, decision_artifact=decision_artifact_output
            )
            post_commit_receipts = AppendOnlyReceiptStore._read_chain_locked(
                receipt_store, receipt_transaction
            )
            if _chain_binding(post_commit_receipts) != receipt_binding:
                raise PublicationError(
                    "Publication receipt chain moved across the commit boundary."
                )
        except BaseException as error:
            if renamed:
                try:
                    observed = destination.lstat()
                except FileNotFoundError:
                    pass
                else:
                    committed_identity = (observed.st_dev, observed.st_ino)
                    commit_state["identity"] = committed_identity
                    _rollback_failed_publication(
                        destination,
                        stage,
                        expected_identity=committed_identity,
                    )
                raise PublicationError(
                    "Committed publication failed verification and was rolled back."
                ) from error
            raise
    finally:
        os.close(stage_descriptor)
    return {
        **verified_output,
        "decision_artifact_id": decision_seal["protected_artifact_id"],
        "decision_raw_manifest_sha256": decision_seal["raw_manifest_sha256"],
        "publication_status": "published",
    }


def publish_candidate(
    destination: Path,
    decision_artifact_output: Path,
    *,
    destination_id: str,
    evaluator_role_id: str,
    tool_source_sha256: str,
    artifact: Path,
    expected_protected_artifact_id: str,
    expected_raw_manifest_sha256: str,
    expected_raw_manifest_size_bytes: int,
    receipt_store: AppendOnlyReceiptStore,
    external_recovery_attestation: Mapping[str, Any],
    external_evidence: Mapping[str, Path],
    sanitizer: FinalizedSanitizerBinding,
    publication_candidate: PublicationCandidateBinding,
) -> dict[str, Any]:
    """Own the exact locked receipt epoch through commit and final verification."""

    if type(receipt_store) is not AppendOnlyReceiptStore:
        raise PublicationError("Publication requires an exact receipt store.")
    commit_state: dict[str, Any] = {}
    completed_commit = False
    try:
        with AppendOnlyReceiptStore.transaction(receipt_store) as transaction:
            result = _publish_candidate_locked(
                destination,
                decision_artifact_output,
                destination_id=destination_id,
                evaluator_role_id=evaluator_role_id,
                tool_source_sha256=tool_source_sha256,
                artifact=artifact,
                expected_protected_artifact_id=expected_protected_artifact_id,
                expected_raw_manifest_sha256=expected_raw_manifest_sha256,
                expected_raw_manifest_size_bytes=expected_raw_manifest_size_bytes,
                receipt_store=receipt_store,
                receipt_transaction=transaction,
                external_recovery_attestation=external_recovery_attestation,
                external_evidence=external_evidence,
                sanitizer=sanitizer,
                publication_candidate=publication_candidate,
                commit_state=commit_state,
            )
            completed_commit = True
    except BaseException as error:
        if commit_state:
            committed_destination = commit_state["destination"]
            rollback_stage = commit_state["stage"]
            expected_identity = commit_state["identity"]
            try:
                observed = committed_destination.lstat()
            except FileNotFoundError:
                pass
            except OSError as rollback_error:
                raise PublicationError(
                    "Receipt authority failed and publication absence is unverifiable."
                ) from rollback_error
            else:
                if (observed.st_dev, observed.st_ino) != expected_identity:
                    raise PublicationError(
                        "Receipt authority failed after the public destination changed."
                    ) from error
                _rollback_failed_publication(
                    committed_destination,
                    rollback_stage,
                    expected_identity=expected_identity,
                )
            if completed_commit:
                raise PublicationError(
                    "Receipt authority failed after commit; publication was rolled back."
                ) from error
            if isinstance(error, PublicationError):
                raise
            raise PublicationError(
                "Publication failed after commit; public bytes were rolled back."
            ) from error
        raise
    return result


__all__ = [
    "FINALIZED_CANDIDATE_ALLOWLIST",
    "PublicationError",
    "publish_candidate",
    "verify_published_output",
]
