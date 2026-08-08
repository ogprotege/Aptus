"""Constructive public projections for the bounded CUDA evidence campaign.

This module never redacts or recursively copies a protected object.  Callers
must provide the exact protected recovery-input schema below, and every public
field is rebuilt from an explicit allowlist.  Defense-in-depth scans run after
construction and cannot make an otherwise unsafe projection acceptable.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
)
from .storage import (
    EVIDENCE_RECEIPT_SCHEMA,
    RawArtifactWriter,
    verify_sealed_artifact,
)


RECOVERY_INPUT_SCHEMA = "aptus.experiment-recovery-input.v1"
RECOVERY_SUPPLEMENT_SCHEMA = "aptus.experiment-recovery-supplement.v1"
SANITIZATION_MAP_SCHEMA = "aptus.experiment-sanitization-map.v1"
PUBLICATION_REVIEW_SCHEMA = "aptus.experiment-publication-review.v1"
PUBLICATION_FINALIZATION_SCHEMA = "aptus.experiment-publication-finalization.v1"
CLAIM_BOUNDARY_SCHEMA = "aptus.experiment-claim-boundary.v1"
EXPECTED_DIGEST_SCHEMA = "aptus.raw-artifact-digests.v1"
EXPECTED_DIGEST_SHA256 = (
    "db6c4845846dcc1bdd2cdb54992210d31b4eba489a514197f33a127ccb37da7a"
)
ORIGINAL_PACKET_REPOSITORY_PATH = (
    "docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance"
)
RETENTION_POLICY_ID = "cuda-v02-public-claim-evidence-24m-v1"

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_ID = re.compile(r"^(?:artifact|copy|domain|host)_[0-9a-f]{32}$")
_STABLE_ID = re.compile(r"^[a-z][a-z0-9-]*(?:_[a-z0-9-]+)*$")
_ROLE_ID = PROCEDURAL_ROLE_ID_RE
_FINALIZATION_ID = re.compile(r"^finalization_[0-9a-f]{32}$")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_GPU_UUID = re.compile(r"\bGPU-[0-9A-Fa-f-]{16,}\b")
_WINDOWS_PATH = re.compile(r"(?:\b[A-Za-z]:\\|\\\\[^\\\s]+\\[^\\\s]+)")
_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9._-])/(?:Users|home|root|private|tmp|var|etc|opt|srv|mnt|media|run/user)/[^\s\"']+"
)
_GENERIC_UUID = re.compile(
    r"(?i)(?<![0-9a-f])(?:[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})(?![0-9a-f])"
)
_SERIAL_STRUCTURE = re.compile(
    r"(?i)\b(?:serial(?:[_ -]?number)?|s/n)\s*[:=]\s*[A-Z0-9][A-Z0-9._-]{3,63}\b"
)
_HOSTNAME_STRUCTURE = re.compile(
    r"(?i)\b(?:host(?:name)?|computer[_ -]?name)\s*[:=]\s*"
    r"[A-Z0-9][A-Z0-9.-]{1,252}\b"
)
_CREDENTIAL = re.compile(
    r"(?i)(?:bearer\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"(?:token|password|passwd|secret|api[_-]?key|access[_-]?key)\s*[:=]\s*[^\s,;]{4,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)
_ENV_ASSIGNMENT = re.compile(
    r"(?m)(?:^|\s)(?:HOME|USER|USERNAME|HOSTNAME|PATH|HF_TOKEN|AWS_[A-Z0-9_]+)="
)
_PROTECTED_KEY_PARTS = {
    "argv",
    "exact_argv",
    "working_directory",
    "vault_path",
    "state_path",
    "bundle_path",
    "output_path",
    "cache_path",
    "username",
    "hostname",
    "ip_address",
    "gpu_uuid",
    "serial_number",
    "pid",
    "process_group_id",
    "lease_token",
    "environment",
    "environment_value",
    "environment_values",
    "raw_log",
    "job_state",
    "raw_exception",
    "traceback",
    "source_data",
    "model_bytes",
    "model_weights",
    "checkpoint_bytes",
    "checkpoint_path",
    "adapter_bytes",
    "adapter_path",
    "identity_bindings",
    "source_bindings",
}
_MAX_EXPECTED_MANIFEST_BYTES = 1024 * 1024
_MAX_CANONICAL_JSON_BYTES = 4 * 1024 * 1024
_MAX_CHECKSUM_BYTES = 1024 * 1024

_REASON_TEXT = {
    "NONE": "No failure reason applies.",
    "RECOVERED_MATCH": "Recovered bytes match the expected SHA-256.",
    "RECOVERED_MISMATCH": "Recovered bytes do not match the expected SHA-256.",
    "NOT_FOUND_AFTER_BOUNDED_SEARCH": (
        "The expected artifact was not found within the recorded bounded search."
    ),
    "ORIGINAL_TRANSCRIPT_NOT_FOUND": (
        "The original byte-exact Python test transcript was not found; it was not reconstructed."
    ),
    "COPY_VERIFICATION_FAILURE": "A required protected copy did not verify.",
    "RETRIEVAL_FAILURE": "The required full protected retrieval did not verify.",
    "SANITIZATION_FAILURE": "The constructive public projection did not pass review.",
}

_DISPOSITIONS = {
    "recovered-matching": "RECOVERED_MATCH",
    "recovered-mismatched": "RECOVERED_MISMATCH",
    "not-found": "NOT_FOUND_AFTER_BOUNDED_SEARCH",
}


class SanitizationError(ValueError):
    """Raised when a public projection cannot be proven safe and traceable."""


@dataclass(frozen=True)
class Projection:
    supplement: dict[str, Any]
    sanitization_map: dict[str, Any]
    claim_boundary: dict[str, Any]


@dataclass(frozen=True)
class VerifiedRecoveryContext:
    """Recovery inputs proven by two independently sealed raw manifests.

    The recovery artifact owns recovered bytes.  The control artifact owns the
    frozen expected-digest manifest, the exact recovery-input record, and the
    canonical typed receipts.  A production projection is accepted only from
    this context; a caller-supplied dictionary can never satisfy review.
    """

    source: dict[str, Any]
    expected: dict[str, Any]
    expected_rows: tuple[tuple[str, str, str], ...]
    recovery_directory: Path
    control_directory: Path
    recovery_verification: dict[str, Any]
    control_verification: dict[str, Any]
    recovery_entries: dict[str, dict[str, Any]]
    control_entries: dict[str, dict[str, Any]]
    input_entry_id: str
    expected_entry_id: str
    receipt_entry_ids: dict[str, str]
    json_sources: dict[tuple[str, str], object]
    forbidden_literals: tuple[str, ...]


def stable_reason(reason_code: str) -> str:
    """Return a fixed, bounded public explanation without accepting raw text."""

    if reason_code not in _REASON_TEXT:
        raise SanitizationError(f"Reason code {reason_code!r} has no public template.")
    reason = _REASON_TEXT[reason_code]
    if len(reason) > 240:
        raise AssertionError("A stable public reason exceeds 240 code points.")
    return reason


def _require_exact_object(
    value: object, *, required: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or isinstance(value, bool):
        raise SanitizationError(f"{label} must be a JSON object.")
    fields = set(value)
    if fields != required:
        missing = sorted(required - fields)
        unknown = sorted(fields - required)
        raise SanitizationError(
            f"{label} fields are not exact; missing={missing}, unknown={unknown}."
        )
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SanitizationError(f"{label} must be a non-empty string.")
    return value


def _require_digest(value: object, label: str) -> str:
    text = _require_string(value, label)
    if _DIGEST.fullmatch(text) is None:
        raise SanitizationError(f"{label} must be a lowercase SHA-256 digest.")
    return text


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SanitizationError(f"{label} must be a non-negative integer.")
    return value


def _require_positive_int(value: object, label: str) -> int:
    result = _require_nonnegative_int(value, label)
    if result == 0:
        raise SanitizationError(f"{label} must be positive.")
    return result


def _require_utc_timestamp(value: object, label: str) -> str:
    text = _require_string(value, label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise SanitizationError(f"{label} must be an RFC 3339 timestamp.") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise SanitizationError(f"{label} must be normalized UTC.")
    return text


def _read_pinned_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int | None,
    require_private: bool = True,
) -> tuple[bytes, os.stat_result]:
    """Read one exact inode through O_NOFOLLOW and reject path/inode races."""

    try:
        before = path.lstat()
    except OSError as error:
        raise SanitizationError(f"{label} is unavailable.") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (
            require_private
            and os.name == "posix"
            and stat.S_IMODE(before.st_mode) != 0o600
        )
        or (hasattr(os, "getuid") and before.st_uid != os.getuid())
    ):
        raise SanitizationError(f"{label} must be a regular unique owned file.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SanitizationError(f"{label} cannot be opened safely.") from error
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (
                require_private
                and os.name == "posix"
                and stat.S_IMODE(opened.st_mode) != 0o600
            )
            or (hasattr(os, "getuid") and opened.st_uid != os.getuid())
        ):
            raise SanitizationError(f"{label} changed while opening.")
        if max_bytes is not None and opened.st_size > max_bytes:
            raise SanitizationError(f"{label} exceeds its size limit.")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            if max_bytes is not None and len(payload) + len(block) > max_bytes:
                raise SanitizationError(f"{label} exceeds its size limit.")
            payload.extend(block)
        finished = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(finished, field) != getattr(opened, field)
            for field in stable_fields
        ):
            raise SanitizationError(f"{label} changed while reading.")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as error:
        raise SanitizationError(f"{label} path changed after reading.") from error
    if any(
        getattr(after, field) != getattr(finished, field) for field in stable_fields
    ):
        raise SanitizationError(f"{label} path changed after reading.")
    return bytes(payload), finished


def _require_opaque_id(value: object, label: str) -> str:
    text = _require_string(value, label)
    if _OPAQUE_ID.fullmatch(text) is None:
        raise SanitizationError(f"{label} must be a random opaque campaign ID.")
    return text


def _json_pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _flatten_expected_digests(value: object) -> list[tuple[str, str, str]]:
    root = _require_exact_object(
        value,
        required={
            "schema_version",
            "hash_algorithm",
            "source_and_compilation",
            "qualifying_runtime",
            "qualifying_job_records",
            "nonqualifying_rehearsal",
            "retention",
        },
        label="expected digest manifest",
    )
    if root["schema_version"] != EXPECTED_DIGEST_SCHEMA:
        raise SanitizationError("Expected digest manifest schema is not frozen v1.")
    if root["hash_algorithm"] != "sha256":
        raise SanitizationError("Expected digest manifest must use SHA-256.")

    rows: list[tuple[str, str, str]] = []

    def add(item_id: str, pointer: str, digest: object) -> None:
        rows.append((item_id, pointer, _require_digest(digest, pointer)))

    for section_name in ("source_and_compilation", "qualifying_runtime"):
        section = root[section_name]
        if not isinstance(section, dict):
            raise SanitizationError(f"{section_name} must be an object.")
        for item_id, digest in section.items():
            add(
                f"{section_name}.{item_id}",
                f"/{section_name}/{_json_pointer_escape(item_id)}",
                digest,
            )

    jobs = root["qualifying_job_records"]
    if not isinstance(jobs, list):
        raise SanitizationError("qualifying_job_records must be an array.")
    for index, job in enumerate(jobs):
        row = _require_exact_object(
            job,
            required={"action", "job_id", "record_sha256", "log_sha256"},
            label=f"qualifying job {index}",
        )
        action = _require_string(row["action"], f"qualifying job {index} action")
        add(
            f"qualifying_job_records.{index}.{action}.record",
            f"/qualifying_job_records/{index}/record_sha256",
            row["record_sha256"],
        )
        add(
            f"qualifying_job_records.{index}.{action}.log",
            f"/qualifying_job_records/{index}/log_sha256",
            row["log_sha256"],
        )

    rehearsal = _require_exact_object(
        root["nonqualifying_rehearsal"],
        required={"validation_report_sha256", "job_records", "full_train_job_record"},
        label="nonqualifying rehearsal",
    )
    add(
        "nonqualifying_rehearsal.validation_report",
        "/nonqualifying_rehearsal/validation_report_sha256",
        rehearsal["validation_report_sha256"],
    )
    rehearsal_jobs = rehearsal["job_records"]
    if not isinstance(rehearsal_jobs, list):
        raise SanitizationError("nonqualifying rehearsal job_records must be an array.")
    for index, job in enumerate(rehearsal_jobs):
        row = _require_exact_object(
            job,
            required={"action", "job_id", "record_sha256", "log_sha256"},
            label=f"rehearsal job {index}",
        )
        action = _require_string(row["action"], f"rehearsal job {index} action")
        add(
            f"nonqualifying_rehearsal.job_records.{index}.{action}.record",
            f"/nonqualifying_rehearsal/job_records/{index}/record_sha256",
            row["record_sha256"],
        )
        add(
            f"nonqualifying_rehearsal.job_records.{index}.{action}.log",
            f"/nonqualifying_rehearsal/job_records/{index}/log_sha256",
            row["log_sha256"],
        )
    if rehearsal["full_train_job_record"] is not None:
        raise SanitizationError(
            "The frozen expected manifest requires a null rehearsal train record."
        )
    if len(rows) != 40 or len({row[2] for row in rows}) != 39:
        raise SanitizationError(
            "Expected digest manifest must contain 40 logical and 39 unique digests."
        )
    return rows


def load_expected_digest_manifest(
    path: Path,
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    """Load and bind the exact tracked August 6 expected-digest manifest."""

    payload, _metadata = _read_pinned_regular_file(
        path,
        label="Expected digest manifest",
        max_bytes=_MAX_EXPECTED_MANIFEST_BYTES,
        require_private=False,
    )
    if sha256_bytes(payload) != EXPECTED_DIGEST_SHA256:
        raise SanitizationError(
            "Expected digest manifest does not match the frozen SHA-256."
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SanitizationError("Expected digest manifest is unreadable.") from error
    rows = _flatten_expected_digests(value)
    return value, rows


def _manifest_entries_by_id(verification: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = verification["manifest"]["files"]
    return {entry["entry_id"]: dict(entry) for entry in entries}


def _entries_with_role(
    entries: dict[str, dict[str, Any]], role: str
) -> list[dict[str, Any]]:
    return [entry for entry in entries.values() if entry["role"] == role]


def _read_verified_payload(
    directory: Path, entry: dict[str, Any], *, canonical_json: bool
) -> tuple[bytes, object | None]:
    path = directory.joinpath(*entry["relative_path"].split("/"))
    payload, _metadata = _read_pinned_regular_file(
        path,
        label="Sealed control payload",
        max_bytes=_MAX_CANONICAL_JSON_BYTES if canonical_json else None,
    )
    if len(payload) != entry["size_bytes"] or sha256_bytes(payload) != entry["sha256"]:
        raise SanitizationError("A sealed payload changed after manifest verification.")
    if not canonical_json:
        return payload, None
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SanitizationError(
            "A sealed control JSON payload is unreadable."
        ) from error
    if canonical_json_bytes(value) != payload:
        raise SanitizationError("A sealed control JSON payload is not canonical.")
    return payload, value


def _receipt_content_id(receipt: dict[str, Any]) -> str:
    identity = {key: value for key, value in receipt.items() if key != "receipt_id"}
    return "receipt_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]


def _safe_receipt_from_typed(
    receipt: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    kind = receipt["kind"]
    details = receipt["details"]
    if kind == "copy-verification":
        required = {
            "copy_id",
            "failure_domain_id",
            "off_experiment_host",
            "verification_result",
        }
        if (
            set(details) != required
            or type(details["off_experiment_host"]) is not bool
            or details["verification_result"] != "passed"
        ):
            raise SanitizationError("Copy receipt details are not exact and passing.")
        copy_id = details["copy_id"]
        domain_id = details["failure_domain_id"]
    elif kind == "retrieval":
        # The strict receipt contract validates the complete retrieval detail shape.
        copy_id = details["source_copy_id"]
        domain_id = details["source_failure_domain_id"]
    elif kind == "retention":
        required = {
            "retention_policy_id",
            "retain_not_before_utc",
            "verification_result",
        }
        if (
            set(details) != required
            or details["retention_policy_id"] != RETENTION_POLICY_ID
            or details["verification_result"] != "passed"
        ):
            raise SanitizationError(
                "Retention receipt details are not exact and active."
            )
        _require_utc_timestamp(
            details["retain_not_before_utc"], "retention retain-not-before timestamp"
        )
        copy_id = None
        domain_id = None
    else:
        raise SanitizationError(
            "A recovery control artifact contains an invalid receipt kind."
        )
    return {
        "receipt_id": receipt["receipt_id"],
        "kind": kind,
        "created_at_utc": receipt["created_at_utc"],
        "protected_artifact_id": receipt["protected_artifact_id"],
        # These are the public-binding bytes of the canonical typed receipt,
        # not a duplicate of the protected recovery-manifest binding carried
        # inside that receipt.
        "sha256": entry["sha256"],
        "size_bytes": entry["size_bytes"],
        "result": receipt["result"],
        "copy_id": copy_id,
        "failure_domain_id": domain_id,
    }


def _is_public_identifier(value: str) -> bool:
    return bool(
        _OPAQUE_ID.fullmatch(value)
        or _DIGEST.fullmatch(value)
        or re.fullmatch(r"campaign_[0-9a-f]{20}", value)
        or _STABLE_ID.fullmatch(value)
    )


def _derive_forbidden_literals(
    source: dict[str, Any], *additional_protected_values: object
) -> tuple[str, ...]:
    values = set(source["forbidden_private_literals"])

    def visit(value: object, *, protected: bool = False) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, protected=protected or key.lower() in _PROTECTED_KEY_PARTS)
        elif isinstance(value, list):
            for item in value:
                visit(item, protected=protected)
        elif protected and isinstance(value, str) and len(value) >= 4:
            if not _is_public_identifier(value):
                values.add(value)

    visit(source)
    for value in additional_protected_values:
        visit(value)
    return tuple(sorted(values, key=lambda item: (item.casefold(), item)))


def load_verified_recovery_context(
    *, recovery_artifact: Path, control_artifact: Path
) -> VerifiedRecoveryContext:
    """Load the only provenance form accepted by production review.

    ``recovery_artifact`` contains the recovered bytes. ``control_artifact``
    contains exactly one recovery input, the exact frozen digest manifest, and
    the complete typed receipt chain used by that input.
    """

    try:
        recovery = verify_sealed_artifact(recovery_artifact)
        control = verify_sealed_artifact(control_artifact)
    except (OSError, ValueError) as error:
        raise SanitizationError(
            "Recovery provenance is not a valid sealed artifact pair."
        ) from error
    if recovery["manifest"]["record_kind"] != "legacy-recovery":
        raise SanitizationError(
            "Recovered bytes must be held by a legacy-recovery artifact."
        )
    if control["manifest"]["record_kind"] != "legacy-recovery":
        raise SanitizationError(
            "Recovery controls must be held by a legacy-recovery artifact."
        )
    if (
        recovery["protected_artifact_id"] == control["protected_artifact_id"]
        or recovery["raw_manifest_sha256"] == control["raw_manifest_sha256"]
    ):
        raise SanitizationError("Recovery and control artifacts must be distinct.")

    recovery_entries = _manifest_entries_by_id(recovery)
    control_entries = _manifest_entries_by_id(control)
    input_entries = _entries_with_role(control_entries, "recovery-input")
    expected_entries = _entries_with_role(control_entries, "expected-digest-manifest")
    receipt_entries = _entries_with_role(control_entries, "evidence-receipt")
    if len(input_entries) != 1 or len(expected_entries) != 1:
        raise SanitizationError(
            "The control artifact requires exactly one input and one expected manifest."
        )
    if len(receipt_entries) < 4:
        raise SanitizationError(
            "The control artifact does not contain the required receipts."
        )
    if len(control_entries) != 2 + len(receipt_entries):
        raise SanitizationError(
            "The control artifact contains an extraneous payload role."
        )

    input_entry = input_entries[0]
    expected_entry = expected_entries[0]
    receipt_entry_ids_in_manifest = [entry["entry_id"] for entry in receipt_entries]
    if control["manifest"]["required_role_bindings"] != {
        "evidence-receipt": receipt_entry_ids_in_manifest,
        "expected-digest-manifest": expected_entry["entry_id"],
        "recovery-input": input_entry["entry_id"],
    }:
        raise SanitizationError(
            "The control artifact required-role inventory is not exact."
        )
    if (
        input_entry["relative_path"] != "recovery-input.json"
        or input_entry["media_type"] != "application/json"
        or expected_entry["relative_path"] != "expected-digest-manifest.json"
        or expected_entry["media_type"] != "application/json"
        or any(entry["media_type"] != "application/json" for entry in receipt_entries)
    ):
        raise SanitizationError(
            "The control artifact has an invalid role path or media type."
        )
    _input_bytes, raw_source = _read_verified_payload(
        control_artifact, input_entry, canonical_json=True
    )
    source = _validate_recovery_input(raw_source)
    expected_path = control_artifact.joinpath(
        *expected_entry["relative_path"].split("/")
    )
    if expected_entry["sha256"] != EXPECTED_DIGEST_SHA256:
        raise SanitizationError(
            "The sealed expected manifest does not match the frozen digest."
        )
    expected, expected_rows = load_expected_digest_manifest(expected_path)

    manifest_binding = source["recovery_raw_manifest"]
    if (
        manifest_binding["protected_artifact_id"] != recovery["protected_artifact_id"]
        or manifest_binding["sha256"] != recovery["raw_manifest_sha256"]
        or manifest_binding["size_bytes"] != recovery["raw_manifest_size_bytes"]
        or manifest_binding["entry_id"] != input_entry["entry_id"]
    ):
        raise SanitizationError(
            "Recovery input does not bind both exact sealed manifests."
        )

    sealed_recovery_entries = recovery["manifest"]["files"]
    sealed_recovery_ids = [entry["entry_id"] for entry in sealed_recovery_entries]
    source_recovery_ids = [
        recovered["entry_id"] for recovered in source["recovered_entries"]
    ]
    if any(entry["role"] != "recovered-artifact" for entry in sealed_recovery_entries):
        raise SanitizationError(
            "The sealed recovery artifact contains a non-recovery payload role."
        )
    expected_recovery_role_bindings: dict[str, str | list[str]] = {}
    if sealed_recovery_ids:
        expected_recovery_role_bindings["recovered-artifact"] = sealed_recovery_ids
    if recovery["manifest"]["required_role_bindings"] != (
        expected_recovery_role_bindings
    ):
        raise SanitizationError(
            "The sealed recovery required-role inventory is not exact."
        )
    if set(source_recovery_ids) != set(sealed_recovery_ids):
        raise SanitizationError(
            "Recovered input and sealed recovery payload inventories differ."
        )

    for recovered in source["recovered_entries"]:
        raw_entry = recovery_entries.get(recovered["entry_id"])
        if raw_entry is None:
            raise SanitizationError(
                "A recovered input entry is absent from the sealed recovery manifest."
            )
        if raw_entry["role"] != "recovered-artifact":
            raise SanitizationError(
                "A recovered input entry has the wrong manifest role."
            )
        if (
            recovered["sha256"] != raw_entry["sha256"]
            or recovered["size_bytes"] != raw_entry["size_bytes"]
        ):
            raise SanitizationError(
                "A recovered input entry disagrees with sealed bytes."
            )

    typed_receipts: dict[str, tuple[dict[str, Any], str]] = {}
    json_sources: dict[tuple[str, str], object] = {
        (control["raw_manifest_sha256"], input_entry["entry_id"]): source,
        (control["raw_manifest_sha256"], expected_entry["entry_id"]): expected,
    }
    for entry in receipt_entries:
        _payload, raw_receipt = _read_verified_payload(
            control_artifact, entry, canonical_json=True
        )
        try:
            receipt = validate_record(
                raw_receipt, expected_schema=EVIDENCE_RECEIPT_SCHEMA
            )
        except ContractError as error:
            raise SanitizationError(
                "A control receipt violates its typed contract."
            ) from error
        if receipt["receipt_id"] != _receipt_content_id(receipt):
            raise SanitizationError("A control receipt ID is not content-addressed.")
        if _ROLE_ID.fullmatch(receipt["issuer_role_id"]) is None:
            raise SanitizationError("A control receipt issuer is not a stable role ID.")
        if receipt["receipt_id"] in typed_receipts:
            raise SanitizationError("The control artifact repeats a receipt ID.")
        if (
            receipt["protected_artifact_id"] != recovery["protected_artifact_id"]
            or receipt["raw_manifest_sha256"] != recovery["raw_manifest_sha256"]
            or receipt["raw_manifest_size_bytes"] != recovery["raw_manifest_size_bytes"]
        ):
            raise SanitizationError(
                "A control receipt does not bind the exact recovery artifact."
            )
        typed_receipts[receipt["receipt_id"]] = (receipt, entry["entry_id"])
        json_sources[(control["raw_manifest_sha256"], entry["entry_id"])] = receipt

    safe_rows = list(source["copy_verification_receipts"]) + [
        source["retrieval_receipt"],
        source["retention_receipt"],
    ]
    expected_chain: list[str] = []
    receipt_entry_ids: dict[str, str] = {}
    for safe in safe_rows:
        receipt_id = safe["receipt_id"]
        try:
            typed, entry_id = typed_receipts[receipt_id]
        except KeyError as error:
            raise SanitizationError(
                "A projected receipt is absent from sealed controls."
            ) from error
        typed_entry = control_entries[entry_id]
        if _safe_receipt_from_typed(typed, typed_entry) != safe:
            raise SanitizationError(
                "A projected receipt differs from its sealed typed receipt."
            )
        expected_chain.append(receipt_id)
        receipt_entry_ids[receipt_id] = entry_id
    if set(typed_receipts) != set(expected_chain):
        raise SanitizationError("The control artifact has an extraneous typed receipt.")
    for index, receipt_id in enumerate(expected_chain):
        typed, entry_id = typed_receipts[receipt_id]
        entry = control_entries[entry_id]
        expected_path = f"receipts/{index:02d}-{typed['kind']}.json"
        if entry["relative_path"] != expected_path:
            raise SanitizationError(
                "The sealed receipt payload inventory is not canonically ordered."
            )
    previous: str | None = None
    for receipt_id in expected_chain:
        typed = typed_receipts[receipt_id][0]
        if typed["previous_receipt_id"] != previous:
            raise SanitizationError(
                "The sealed receipt chain is incomplete or reordered."
            )
        previous = receipt_id
    typed_copies = [
        typed_receipts[row["receipt_id"]][0]
        for row in source["copy_verification_receipts"]
    ]
    off_host_copy_bindings = {
        (receipt["details"]["copy_id"], receipt["details"]["failure_domain_id"])
        for receipt in typed_copies
        if receipt["details"]["off_experiment_host"] is True
    }
    typed_retrieval = typed_receipts[source["retrieval_receipt"]["receipt_id"]][0]
    retrieval_source = (
        typed_retrieval["details"]["source_copy_id"],
        typed_retrieval["details"]["source_failure_domain_id"],
    )
    if not off_host_copy_bindings or retrieval_source not in off_host_copy_bindings:
        raise SanitizationError(
            "The passing retrieval must originate from a verified off-host copy."
        )
    retrieval_details = typed_retrieval["details"]
    expected_restored_count = recovery["file_count"] + 2
    expected_restored_bytes = (
        recovery["total_bytes"]
        + recovery["raw_manifest_size_bytes"]
        + len(canonical_json_bytes(recovery["seal"]))
    )
    if (
        retrieval_details["expected_raw_manifest_sha256"]
        != recovery["raw_manifest_sha256"]
        or retrieval_details["observed_raw_manifest_sha256"]
        != recovery["raw_manifest_sha256"]
        or retrieval_details["restored_file_count"] != expected_restored_count
        or retrieval_details["restored_total_bytes"] != expected_restored_bytes
    ):
        raise SanitizationError(
            "The passing retrieval does not account for the complete sealed recovery artifact."
        )

    if control["manifest"]["source_bindings"] != {
        "recovery_artifact_id": recovery["protected_artifact_id"]
    }:
        raise SanitizationError(
            "The recovery control artifact source binding is not exact."
        )

    # Reverify after payload reads so a concurrent mutation cannot be accepted.
    try:
        if (
            verify_sealed_artifact(recovery_artifact)["raw_manifest_sha256"]
            != recovery["raw_manifest_sha256"]
            or verify_sealed_artifact(control_artifact)["raw_manifest_sha256"]
            != control["raw_manifest_sha256"]
        ):
            raise SanitizationError(
                "A provenance artifact changed while it was loaded."
            )
    except (OSError, ValueError) as error:
        if isinstance(error, SanitizationError):
            raise
        raise SanitizationError(
            "A provenance artifact changed while it was loaded."
        ) from error

    return VerifiedRecoveryContext(
        source=source,
        expected=expected,
        expected_rows=tuple(expected_rows),
        recovery_directory=recovery_artifact.resolve(strict=True),
        control_directory=control_artifact.resolve(strict=True),
        recovery_verification=recovery,
        control_verification=control,
        recovery_entries=recovery_entries,
        control_entries=control_entries,
        input_entry_id=input_entry["entry_id"],
        expected_entry_id=expected_entry["entry_id"],
        receipt_entry_ids=receipt_entry_ids,
        json_sources=json_sources,
        forbidden_literals=_derive_forbidden_literals(
            source,
            recovery["manifest"],
            control["manifest"],
            [item[0] for item in typed_receipts.values()],
        ),
    )


def _validate_safe_receipt_binding(value: object, label: str) -> dict[str, Any]:
    row = _require_exact_object(
        value,
        required={
            "receipt_id",
            "kind",
            "created_at_utc",
            "protected_artifact_id",
            "sha256",
            "size_bytes",
            "result",
            "copy_id",
            "failure_domain_id",
        },
        label=label,
    )
    _require_string(row["receipt_id"], f"{label}.receipt_id")
    _require_string(row["kind"], f"{label}.kind")
    _require_utc_timestamp(row["created_at_utc"], f"{label}.created_at_utc")
    _require_opaque_id(row["protected_artifact_id"], f"{label}.protected_artifact_id")
    _require_digest(row["sha256"], f"{label}.sha256")
    _require_positive_int(row["size_bytes"], f"{label}.size_bytes")
    if row["result"] not in {"passed", "failed", "active"}:
        raise SanitizationError(f"{label}.result is invalid.")
    if row["copy_id"] is not None:
        _require_opaque_id(row["copy_id"], f"{label}.copy_id")
    if row["failure_domain_id"] is not None:
        _require_opaque_id(row["failure_domain_id"], f"{label}.failure_domain_id")
    return dict(row)


def _validate_recovery_input(value: object) -> dict[str, Any]:
    root = _require_exact_object(
        value,
        required={
            "schema_version",
            "producer_role_id",
            "campaign_id",
            "original_packet",
            "recovery_raw_manifest",
            "recovered_entries",
            "copy_verification_receipts",
            "retrieval_receipt",
            "retention_receipt",
            "additional_search_results",
            "forbidden_private_literals",
        },
        label="protected recovery input",
    )
    if root["schema_version"] != RECOVERY_INPUT_SCHEMA:
        raise SanitizationError("Protected recovery input has the wrong schema.")
    producer = _require_string(root["producer_role_id"], "producer_role_id")
    if _ROLE_ID.fullmatch(producer) is None:
        raise SanitizationError(
            "producer_role_id must be a stable non-personal role ID."
        )
    campaign_id = _require_string(root["campaign_id"], "campaign_id")
    if re.fullmatch(r"campaign_[0-9a-f]{20}", campaign_id) is None:
        raise SanitizationError("campaign_id must be a canonical campaign identifier.")
    original_packet = _require_string(root["original_packet"], "original_packet")
    if original_packet != ORIGINAL_PACKET_REPOSITORY_PATH:
        raise SanitizationError(
            "original_packet must equal the frozen Phase 0 repository path."
        )

    manifest = _require_exact_object(
        root["recovery_raw_manifest"],
        required={
            "protected_artifact_id",
            "sha256",
            "size_bytes",
            "entry_id",
        },
        label="recovery raw manifest",
    )
    _require_opaque_id(manifest["protected_artifact_id"], "protected_artifact_id")
    _require_digest(manifest["sha256"], "recovery raw manifest sha256")
    _require_positive_int(manifest["size_bytes"], "recovery raw manifest size")
    _require_string(manifest["entry_id"], "recovery raw manifest entry_id")

    entries = root["recovered_entries"]
    if not isinstance(entries, list):
        raise SanitizationError("recovered_entries must be an array.")
    seen_entry_ids: set[str] = set()
    for index, value_entry in enumerate(entries):
        entry = _require_exact_object(
            value_entry,
            required={"entry_id", "sha256", "size_bytes", "logical_source_pointers"},
            label=f"recovered entry {index}",
        )
        entry_id = _require_string(entry["entry_id"], f"recovered entry {index} ID")
        if entry_id in seen_entry_ids:
            raise SanitizationError("recovered_entries contains a duplicate entry ID.")
        seen_entry_ids.add(entry_id)
        _require_digest(entry["sha256"], f"recovered entry {index} SHA-256")
        _require_nonnegative_int(entry["size_bytes"], f"recovered entry {index} size")
        pointers = entry["logical_source_pointers"]
        if not isinstance(pointers, list) or not pointers:
            raise SanitizationError(
                "Each recovered entry must bind source JSON pointers."
            )
        if any(
            not isinstance(pointer, str) or not pointer.startswith("/")
            for pointer in pointers
        ):
            raise SanitizationError("Recovered entry source pointers are invalid.")
        if len(pointers) != len(set(pointers)):
            raise SanitizationError(
                "Recovered entry source pointers contain duplicates."
            )

    receipts = root["copy_verification_receipts"]
    if not isinstance(receipts, list) or len(receipts) < 2:
        raise SanitizationError("At least two copy-verification receipts are required.")
    validated_receipts = [
        _validate_safe_receipt_binding(item, f"copy receipt {index}")
        for index, item in enumerate(receipts)
    ]
    receipt_ids: set[str] = set()
    copy_bindings: set[tuple[str, str]] = set()
    for item in validated_receipts:
        if item["receipt_id"] in receipt_ids:
            raise SanitizationError("Copy receipts contain a duplicate receipt ID.")
        receipt_ids.add(item["receipt_id"])
        if item["kind"] != "copy-verification" or item["result"] != "passed":
            raise SanitizationError(
                "Every published copy receipt must be a passing copy verification."
            )
        if item["protected_artifact_id"] != manifest["protected_artifact_id"]:
            raise SanitizationError(
                "Copy receipt does not bind the recovery protected artifact."
            )
        if item["copy_id"] is None or item["failure_domain_id"] is None:
            raise SanitizationError(
                "Copy verification requires a copy and non-null failure-domain ID."
            )
        copy_binding = (item["copy_id"], item["failure_domain_id"])
        if copy_binding in copy_bindings:
            raise SanitizationError("Copy receipts contain a duplicate copy binding.")
        copy_bindings.add(copy_binding)
    domains = {
        item["failure_domain_id"]
        for item in validated_receipts
        if item["result"] == "passed"
    }
    if len(domains) < 2:
        raise SanitizationError(
            "Two passing receipts in distinct failure domains are required."
        )
    retrieval = _validate_safe_receipt_binding(
        root["retrieval_receipt"], "retrieval receipt"
    )
    if (
        retrieval["kind"] != "retrieval"
        or retrieval["result"] != "passed"
        or retrieval["protected_artifact_id"] != manifest["protected_artifact_id"]
        or (retrieval["copy_id"], retrieval["failure_domain_id"]) not in copy_bindings
    ):
        raise SanitizationError(
            "Retrieval receipt must pass and bind one verified recovery copy."
        )
    retention = _validate_safe_receipt_binding(
        root["retention_receipt"], "retention receipt"
    )
    if (
        retention["kind"] != "retention"
        or retention["result"] != "active"
        or retention["protected_artifact_id"] != manifest["protected_artifact_id"]
        or retention["copy_id"] is not None
        or retention["failure_domain_id"] is not None
    ):
        raise SanitizationError(
            "Retention receipt must be active and bind the recovery protected artifact."
        )
    if retrieval["receipt_id"] in receipt_ids or retention["receipt_id"] in receipt_ids:
        raise SanitizationError("Recovery receipt IDs must be unique.")
    if retrieval["receipt_id"] == retention["receipt_id"]:
        raise SanitizationError("Recovery receipt IDs must be unique.")

    searches = root["additional_search_results"]
    if not isinstance(searches, list):
        raise SanitizationError("additional_search_results must be an array.")
    for index, search in enumerate(searches):
        row = _require_exact_object(
            search,
            required={"item_id", "disposition", "reason_code", "search_scope_codes"},
            label=f"additional search {index}",
        )
        _require_string(row["item_id"], f"additional search {index} item ID")
        if row["disposition"] != "not-found":
            raise SanitizationError("The transcript search may publish only not-found.")
        if row["reason_code"] != "ORIGINAL_TRANSCRIPT_NOT_FOUND":
            raise SanitizationError(
                "The transcript search must use its frozen reason code."
            )
        scopes = row["search_scope_codes"]
        if (
            not isinstance(scopes, list)
            or not scopes
            or any(
                not isinstance(scope, str) or _STABLE_ID.fullmatch(scope) is None
                for scope in scopes
            )
        ):
            raise SanitizationError("Search scope codes must be bounded stable IDs.")

    literals = root["forbidden_private_literals"]
    if not isinstance(literals, list) or any(
        not isinstance(item, str) or not item for item in literals
    ):
        raise SanitizationError("forbidden_private_literals must contain strings.")
    return root


def _trace(
    *,
    public_file: str,
    public_pointer: str,
    raw_manifest_sha256: str,
    entry_id: str,
    source_pointer: str,
    transform: str,
    evidence_class: str,
) -> dict[str, Any]:
    return {
        "public_file": public_file,
        "public_json_pointer": public_pointer,
        "source_raw_manifest_sha256": raw_manifest_sha256,
        "source_artifact_entry_id": entry_id,
        "source_json_pointer": source_pointer,
        "transform": transform,
        "evidence_class": evidence_class,
    }


def _leaf_values(value: object, pointer: str = "") -> Iterable[tuple[str, object]]:
    if isinstance(value, dict):
        if not value:
            yield pointer, value
            return
        for key, item in value.items():
            yield from _leaf_values(item, f"{pointer}/{_json_pointer_escape(str(key))}")
        return
    if isinstance(value, list):
        if not value:
            yield pointer, value
            return
        for index, item in enumerate(value):
            yield from _leaf_values(item, f"{pointer}/{index}")
        return
    yield pointer, value


def _project_recovery_supplement(
    *,
    source: dict[str, Any],
    expected_rows: Sequence[tuple[str, str, str]],
    context: VerifiedRecoveryContext | None,
) -> Projection:
    manifest = source["recovery_raw_manifest"]
    manifest_digest = manifest["sha256"]
    if context is None:
        # Deliberately nonpublishable placeholders for unit-level construction.
        control_digest = manifest_digest
        input_entry_id = manifest["entry_id"]
        expected_entry_id = manifest["entry_id"]
    else:
        control_digest = context.control_verification["raw_manifest_sha256"]
        input_entry_id = context.input_entry_id
        expected_entry_id = context.expected_entry_id

    input_source = (control_digest, input_entry_id)
    expected_source = (control_digest, expected_entry_id)
    entries_by_pointer: dict[str, dict[str, Any]] = {}
    for entry in source["recovered_entries"]:
        for pointer in entry["logical_source_pointers"]:
            if pointer in entries_by_pointer:
                raise SanitizationError(
                    "A logical source pointer maps to multiple recovered entries."
                )
            entries_by_pointer[pointer] = entry
    expected_pointers = {pointer for _logical, pointer, _digest in expected_rows}
    unexpected_pointers = set(entries_by_pointer) - expected_pointers
    if unexpected_pointers:
        raise SanitizationError(
            "Recovered entries contain pointers outside the frozen digest manifest."
        )

    items: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    traced_targets: set[tuple[str, str]] = set()

    def bind(
        public_file: str,
        public_pointer: str,
        source_ref: tuple[str, str],
        source_pointer: str,
        transform: str,
        evidence_class: str,
    ) -> None:
        target = (public_file, public_pointer)
        if target in traced_targets:
            raise AssertionError(f"Duplicate sanitization trace target: {target!r}")
        traces.append(
            _trace(
                public_file=public_file,
                public_pointer=public_pointer,
                raw_manifest_sha256=source_ref[0],
                entry_id=source_ref[1],
                source_pointer=source_pointer,
                transform=transform,
                evidence_class=evidence_class,
            )
        )
        traced_targets.add(target)

    counts = {
        "logical_digest_count": len(expected_rows),
        "recovered_matching": 0,
        "recovered_mismatched": 0,
        "not_found": 0,
    }
    for index, (logical_id, pointer, expected_sha256) in enumerate(expected_rows):
        entry = entries_by_pointer.get(pointer)
        if entry is None:
            disposition = "not-found"
            recovered_entry_id = None
            recovered_sha256 = None
            recovered_size = None
        else:
            disposition = (
                "recovered-matching"
                if entry["sha256"] == expected_sha256
                else "recovered-mismatched"
            )
            recovered_entry_id = entry["entry_id"]
            recovered_sha256 = entry["sha256"]
            recovered_size = entry["size_bytes"]
        reason_code = _DISPOSITIONS[disposition]
        counts[disposition.replace("-", "_")] += 1
        public_item = {
            "logical_item_id": logical_id,
            "source_json_pointer": pointer,
            "expected_sha256": expected_sha256,
            "disposition": disposition,
            "recovered_artifact_entry_id": recovered_entry_id,
            "recovered_sha256": recovered_sha256,
            "recovered_size_bytes": recovered_size,
            "reason_code": reason_code,
        }
        items.append(public_item)
        expected_fields = {
            "logical_item_id": ("constant", "declared"),
            "source_json_pointer": ("constant", "declared"),
            "expected_sha256": ("copy", "declared"),
        }
        for field_name, (transform, evidence_class) in expected_fields.items():
            bind(
                "recovery-supplement.json",
                f"/items/{index}/{field_name}",
                expected_source,
                pointer,
                transform,
                evidence_class,
            )
        if entry is None:
            observed_source = input_source
            observed_pointer = "/recovered_entries"
        else:
            observed_source = (manifest_digest, entry["entry_id"])
            observed_pointer = ""
        for field_name, transform, evidence_class in (
            ("disposition", "digest", "inferred"),
            ("recovered_artifact_entry_id", "opaque-id", "measured"),
            ("recovered_sha256", "digest", "measured"),
            ("recovered_size_bytes", "aggregate", "measured"),
            ("reason_code", "constant", "inferred"),
        ):
            if entry is None and field_name.startswith("recovered_"):
                transform = "constant"
            bind(
                "recovery-supplement.json",
                f"/items/{index}/{field_name}",
                observed_source,
                observed_pointer,
                transform,
                evidence_class,
            )

    if counts["logical_digest_count"] != (
        counts["recovered_matching"]
        + counts["recovered_mismatched"]
        + counts["not_found"]
    ):
        raise AssertionError("Recovery summary counts do not reconcile.")

    claim_boundary = {
        "schema_version": CLAIM_BOUNDARY_SCHEMA,
        "campaign_id": source["campaign_id"],
        "claim_key": "august-6-protected-raw-recovery-integrity",
        "exact_scope": {
            "original_packet": source["original_packet"],
            "recovery_raw_manifest_sha256": manifest_digest,
        },
        "allowed_claim_types": ["raw-recovery-integrity"],
        "forbidden_claims": [
            "retroactive timing or telemetry",
            "performance or resource measurement",
            "repeatability",
            "another CUDA method, model, host, or environment",
            "recreation of an absent Python test transcript",
            "closure of the Ubuntu source-test gate without the original transcript",
            "release readiness",
        ],
        "qualification_dependencies": [
            "sealed recovery raw manifest",
            "two verified copies in distinct failure domains",
            "verified full off-host retrieval",
            "active retention receipt",
            "constructive sanitization",
            "independent review",
        ],
        "statement": (
            "Recovery integrity only for protected artifacts bound to the original August 6 packet."
        ),
    }
    validate_record(claim_boundary)

    copy_receipts = [
        _validate_safe_receipt_binding(item, f"copy receipt {index}")
        for index, item in enumerate(source["copy_verification_receipts"])
    ]
    retrieval_receipt = _validate_safe_receipt_binding(
        source["retrieval_receipt"], "retrieval receipt"
    )
    retention_receipt = _validate_safe_receipt_binding(
        source["retention_receipt"], "retention receipt"
    )
    supplement = {
        "schema_version": RECOVERY_SUPPLEMENT_SCHEMA,
        "original_packet": {"repository_path": source["original_packet"]},
        "expected_digest_manifest": {
            "schema_version": EXPECTED_DIGEST_SCHEMA,
            "sha256": EXPECTED_DIGEST_SHA256,
            "logical_digest_count": 40,
            "unique_digest_count": 39,
        },
        "recovery_raw_manifest": {
            "protected_artifact_id": manifest["protected_artifact_id"],
            "sha256": manifest_digest,
            "size_bytes": manifest["size_bytes"],
        },
        "copy_verification_receipts": copy_receipts,
        "retrieval_receipt": retrieval_receipt,
        "retention_policy": {
            "policy_id": RETENTION_POLICY_ID,
            "minimum_calendar_months": 24,
            "claim_withdrawal_required_before_early_deletion": True,
        },
        "retention_receipt": retention_receipt,
        "sanitization_map": {
            "schema_version": SANITIZATION_MAP_SCHEMA,
            "status": "constructed-separately",
        },
        "independent_review": {
            "schema_version": PUBLICATION_REVIEW_SCHEMA,
            "status": "pending",
        },
        "claim_boundary": claim_boundary,
        "summary_counts": counts,
        "items": items,
        "additional_search_items": [
            {
                "item_id": row["item_id"],
                "disposition": row["disposition"],
                "reason_code": row["reason_code"],
                "search_scope_codes": list(row["search_scope_codes"]),
            }
            for row in source["additional_search_results"]
        ],
    }

    def receipt_source(
        index: int, field: str, receipt: dict[str, Any]
    ) -> tuple[tuple[str, str], str]:
        if context is None:
            ref = input_source
            return ref, f"/copy_verification_receipts/{index}/{field}"
        ref = (control_digest, context.receipt_entry_ids[receipt["receipt_id"]])
        direct = {
            "receipt_id": "/receipt_id",
            "kind": "/kind",
            "created_at_utc": "/created_at_utc",
            "protected_artifact_id": "/protected_artifact_id",
            "result": "/result",
        }
        if field in direct:
            return ref, direct[field]
        if field in {"sha256", "size_bytes"}:
            return ref, ""
        if receipt["kind"] == "copy-verification":
            return ref, f"/details/{field}"
        if receipt["kind"] == "retrieval":
            detail = {
                "copy_id": "source_copy_id",
                "failure_domain_id": "source_failure_domain_id",
            }[field]
            return ref, f"/details/{detail}"
        return ref, "/details"

    def claim_source(pointer: str) -> tuple[tuple[str, str], str, str, str]:
        if pointer == "/campaign_id":
            return input_source, "/campaign_id", "copy", "declared"
        if pointer == "/exact_scope/original_packet":
            return input_source, "/original_packet", "copy", "declared"
        if pointer == "/exact_scope/recovery_raw_manifest_sha256":
            return input_source, "/recovery_raw_manifest/sha256", "copy", "measured"
        if pointer == "/qualification_dependencies/2" and context is not None:
            for receipt in copy_receipts:
                entry_id = context.receipt_entry_ids[receipt["receipt_id"]]
                typed = context.json_sources[(control_digest, entry_id)]
                if typed["details"]["off_experiment_host"] is True:
                    return (
                        (control_digest, entry_id),
                        "/details/off_experiment_host",
                        "constant",
                        "measured",
                    )
        return expected_source, "/schema_version", "constant", "declared"

    for pointer, _leaf in _leaf_values(supplement):
        target = ("recovery-supplement.json", pointer)
        if target in traced_targets:
            continue
        transform = "copy"
        evidence_class = "declared"
        if pointer == "/schema_version":
            source_ref, source_pointer = input_source, "/schema_version"
            transform = "constant"
        elif pointer == "/original_packet/repository_path":
            source_ref, source_pointer = input_source, "/original_packet"
        elif pointer.startswith("/expected_digest_manifest/"):
            source_ref = expected_source
            field = pointer.rsplit("/", 1)[-1]
            source_pointer = "/schema_version" if field == "schema_version" else ""
            transform = "digest" if field == "sha256" else "aggregate"
        elif pointer.startswith("/recovery_raw_manifest/"):
            source_ref = input_source
            field = pointer.rsplit("/", 1)[-1]
            source_pointer = f"/recovery_raw_manifest/{field}"
            evidence_class = "measured"
        elif pointer.startswith("/copy_verification_receipts/"):
            parts = pointer.split("/")
            receipt_index = int(parts[2])
            field = parts[3]
            receipt = copy_receipts[receipt_index]
            source_ref, source_pointer = receipt_source(receipt_index, field, receipt)
            evidence_class = "measured"
            if field == "sha256":
                transform = "digest"
            elif field == "size_bytes":
                transform = "aggregate"
        elif pointer.startswith("/retrieval_receipt/"):
            field = pointer.rsplit("/", 1)[-1]
            if context is None:
                source_ref, source_pointer = input_source, f"/retrieval_receipt/{field}"
            else:
                source_ref, source_pointer = receipt_source(0, field, retrieval_receipt)
            evidence_class = "measured"
            if field == "sha256":
                transform = "digest"
            elif field == "size_bytes":
                transform = "aggregate"
        elif pointer.startswith("/retention_receipt/"):
            field = pointer.rsplit("/", 1)[-1]
            if context is None:
                source_ref, source_pointer = input_source, f"/retention_receipt/{field}"
            else:
                source_ref, source_pointer = receipt_source(0, field, retention_receipt)
            evidence_class = "measured"
            if field == "sha256":
                transform = "digest"
            elif field == "size_bytes":
                transform = "aggregate"
            elif field in {"copy_id", "failure_domain_id"}:
                transform = "constant"
        elif pointer.startswith("/retention_policy/"):
            if context is None:
                source_ref, source_pointer = input_source, "/retention_receipt"
            else:
                source_ref = (
                    control_digest,
                    context.receipt_entry_ids[retention_receipt["receipt_id"]],
                )
                source_pointer = "/details"
            transform = "constant"
        elif pointer.startswith("/claim_boundary/"):
            source_ref, source_pointer, transform, evidence_class = claim_source(
                pointer.removeprefix("/claim_boundary")
            )
        elif pointer.startswith("/summary_counts/"):
            source_ref, source_pointer = input_source, "/recovered_entries"
            transform = "count"
            evidence_class = "inferred"
        elif pointer.startswith("/additional_search_items/"):
            parts = pointer.split("/")
            source_ref = input_source
            source_pointer = (
                f"/additional_search_results/{parts[2]}/{'/'.join(parts[3:])}"
            )
        else:
            source_ref, source_pointer = input_source, "/schema_version"
            transform = "constant"
            evidence_class = "inferred"
        bind(
            "recovery-supplement.json",
            pointer,
            source_ref,
            source_pointer,
            transform,
            evidence_class,
        )
    for pointer, _leaf in _leaf_values(claim_boundary):
        source_ref, source_pointer, transform, evidence_class = claim_source(pointer)
        bind(
            "claim-boundary.json",
            pointer,
            source_ref,
            source_pointer,
            transform,
            evidence_class,
        )
    sanitization_map = {"schema_version": SANITIZATION_MAP_SCHEMA, "entries": []}
    sanitization_map["entries"] = sorted(
        traces,
        key=lambda row: (row["public_file"], row["public_json_pointer"]),
    )
    validate_record(supplement)
    validate_record(sanitization_map)
    private_literals = (
        context.forbidden_literals
        if context is not None
        else tuple(source["forbidden_private_literals"])
    )
    scan_public_value(supplement, forbidden_literals=private_literals)
    scan_public_value(sanitization_map, forbidden_literals=private_literals)
    scan_public_value(claim_boundary, forbidden_literals=private_literals)
    return Projection(
        supplement=supplement,
        sanitization_map=sanitization_map,
        claim_boundary=claim_boundary,
    )


def project_recovery_supplement(
    *, expected_digest_manifest_path: Path, protected_input: object
) -> Projection:
    """Construct a test-only projection from an in-memory dictionary.

    This compatibility helper is intentionally nonpublishable.  Production
    review has no dictionary input and independently reloads both sealed
    artifacts.  Use :func:`project_verified_recovery_supplement` for a stage
    intended to pass review.
    """

    _expected, expected_rows = load_expected_digest_manifest(
        expected_digest_manifest_path
    )
    source = _validate_recovery_input(protected_input)
    return _project_recovery_supplement(
        source=source, expected_rows=expected_rows, context=None
    )


def project_verified_recovery_supplement(
    *, recovery_artifact: Path, control_artifact: Path
) -> Projection:
    """Construct a publishable projection from independently sealed inputs."""

    context = load_verified_recovery_context(
        recovery_artifact=recovery_artifact, control_artifact=control_artifact
    )
    return _project_recovery_supplement(
        source=context.source,
        expected_rows=context.expected_rows,
        context=context,
    )


def _walk(value: object, pointer: str = "") -> Iterable[tuple[str, object]]:
    yield pointer or "/", value
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{pointer}/{_json_pointer_escape(str(key))}"
            yield from _walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{pointer}/{index}")


def _looks_like_ip(value: str) -> bool:
    candidates = re.findall(
        r"(?<![0-9A-Fa-f:.])[0-9A-Fa-f:.]{3,}(?![0-9A-Fa-f:.])", value
    )
    for candidate in candidates:
        try:
            ipaddress.ip_address(candidate.strip(".:") or candidate)
        except ValueError:
            continue
        return True
    return False


def scan_public_value(value: object, *, forbidden_literals: Sequence[str] = ()) -> None:
    """Reject defense-in-depth privacy indicators without exposing matched text."""

    for pointer, item in _walk(value):
        if isinstance(item, float) and not math.isfinite(item):
            raise SanitizationError(f"Non-finite public number at {pointer}.")
        if isinstance(item, dict):
            for key in item:
                normalized = str(key).lower()
                if normalized in _PROTECTED_KEY_PARTS:
                    raise SanitizationError(f"Protected public field at {pointer}.")
        if not isinstance(item, str):
            continue
        if any(
            literal and literal.casefold() in item.casefold()
            for literal in forbidden_literals
        ):
            raise SanitizationError(f"Private literal detected at {pointer}.")
        if (
            _EMAIL.search(item)
            or _GPU_UUID.search(item)
            or _WINDOWS_PATH.search(item)
            or _POSIX_PATH.search(item)
            or _GENERIC_UUID.search(item)
            or _SERIAL_STRUCTURE.search(item)
            or _HOSTNAME_STRUCTURE.search(item)
            or _CREDENTIAL.search(item)
            or _ENV_ASSIGNMENT.search(item)
            or _looks_like_ip(item)
        ):
            raise SanitizationError(f"Private-value pattern detected at {pointer}.")


def _write_exclusive(path: Path, data: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def write_projection_stage(
    output: Path,
    projection: Projection,
    *,
    forbidden_literals: Sequence[str] = (),
) -> dict[str, str]:
    """Write a private no-clobber review stage; it is not a published packet."""

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Projection stage already exists: {output}")
    output.mkdir(mode=0o700, parents=False)
    files = {
        "claim-boundary.json": projection.claim_boundary,
        "recovery-supplement.json": projection.supplement,
        "sanitization-map.json": projection.sanitization_map,
    }
    digests: dict[str, str] = {}
    try:
        for name, record in files.items():
            scan_public_value(record, forbidden_literals=forbidden_literals)
            data = canonical_json_bytes(record)
            _write_exclusive(output / name, data)
            digests[name] = sha256_bytes(data)
        checksum_text = "".join(
            f"{digest}  {name}\n" for name, digest in sorted(digests.items())
        ).encode("utf-8")
        _write_exclusive(output / "SHA256SUMS", checksum_text)
        directory_fd = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # A failed stage is deliberately retained for protected diagnosis.  The
        # caller must choose a new output path; this function never overwrites.
        raise
    return digests


def verify_projection_stage(
    output: Path,
    *,
    recovery_artifact: Path,
    control_artifact: Path,
    producer_role_id: str,
    reviewer_role_id: str,
    forbidden_literals: Sequence[str] = (),
) -> dict[str, Any]:
    """Independently reload sealed provenance and verify exact staged bytes."""

    context = load_verified_recovery_context(
        recovery_artifact=recovery_artifact, control_artifact=control_artifact
    )
    return _verify_projection_stage(
        output,
        context=context,
        producer_role_id=producer_role_id,
        reviewer_role_id=reviewer_role_id,
        forbidden_literals=forbidden_literals,
    )


def _json_pointer_value(value: object, pointer: str) -> object:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise SanitizationError("A sanitization trace has an invalid source pointer.")
    cursor = value
    for encoded in pointer[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(cursor, dict) and token in cursor:
            cursor = cursor[token]
        elif isinstance(cursor, list) and token.isdigit() and int(token) < len(cursor):
            cursor = cursor[int(token)]
        else:
            raise SanitizationError(
                "A sanitization trace points to absent source data."
            )
    return cursor


def _verify_trace_sources(
    sanitization_map: dict[str, Any],
    public_records: dict[str, dict[str, Any]],
    context: VerifiedRecoveryContext,
) -> None:
    manifests = {
        context.recovery_verification["raw_manifest_sha256"]: context.recovery_entries,
        context.control_verification["raw_manifest_sha256"]: context.control_entries,
    }
    for trace in sanitization_map["entries"]:
        digest = trace["source_raw_manifest_sha256"]
        entry_id = trace["source_artifact_entry_id"]
        entries = manifests.get(digest)
        if entries is None or entry_id not in entries:
            raise SanitizationError(
                "A sanitization trace does not identify an entry in a verified manifest."
            )
        source_pointer = trace["source_json_pointer"]
        json_source = context.json_sources.get((digest, entry_id))
        source_value: object | None = None
        if json_source is None:
            if source_pointer != "":
                raise SanitizationError(
                    "A binary recovery entry may only be traced as a deterministic whole-file derivation."
                )
        else:
            source_value = _json_pointer_value(json_source, source_pointer)
        public_value = _json_pointer_value(
            public_records[trace["public_file"]], trace["public_json_pointer"]
        )
        transform = trace["transform"]
        if transform == "copy" and public_value != source_value:
            raise SanitizationError("A copy trace does not preserve its source value.")
        if transform == "opaque-id" and public_value != entry_id:
            raise SanitizationError(
                "An opaque-ID trace does not preserve its entry ID."
            )
        if (
            transform == "digest"
            and isinstance(public_value, str)
            and _DIGEST.fullmatch(public_value)
            and public_value != entries[entry_id]["sha256"]
        ):
            raise SanitizationError("A digest trace does not match its source entry.")
        if (
            transform == "aggregate"
            and json_source is None
            and isinstance(public_value, int)
            and public_value != entries[entry_id]["size_bytes"]
        ):
            raise SanitizationError("A size trace does not match its source entry.")


def _stage_metadata(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SanitizationError(f"{label} is unavailable.") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise SanitizationError(f"{label} must be a regular non-hardlinked file.")
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600:
        raise SanitizationError(f"{label} mode must be 0600.")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise SanitizationError(f"{label} has the wrong owner.")
    return metadata


def _verify_projection_stage(
    output: Path,
    *,
    context: VerifiedRecoveryContext,
    producer_role_id: str,
    reviewer_role_id: str,
    forbidden_literals: Sequence[str] = (),
    review_id: str | None = None,
    reviewed_at_utc: str | None = None,
) -> dict[str, Any]:

    if producer_role_id == reviewer_role_id:
        raise SanitizationError("Independent reviewer must differ from the producer.")
    for role in (producer_role_id, reviewer_role_id):
        if _ROLE_ID.fullmatch(role) is None:
            raise SanitizationError(
                "Review role IDs must be bounded non-personal identifiers."
            )
    source = context.source
    if source["producer_role_id"] != producer_role_id:
        raise SanitizationError(
            "The review producer role does not match the protected projection input."
        )
    expected_projection = _project_recovery_supplement(
        source=source,
        expected_rows=context.expected_rows,
        context=context,
    )
    expected_records = {
        "claim-boundary.json": expected_projection.claim_boundary,
        "recovery-supplement.json": expected_projection.supplement,
        "sanitization-map.json": expected_projection.sanitization_map,
    }
    if output.is_symlink() or not output.is_dir():
        raise SanitizationError("Projection stage must be a regular directory.")
    directory_metadata = output.stat()
    if os.name == "posix" and directory_metadata.st_mode & 0o777 != 0o700:
        raise SanitizationError("Projection stage directory mode must be 0700.")
    if hasattr(os, "getuid") and directory_metadata.st_uid != os.getuid():
        raise SanitizationError("Projection stage is owned by another user.")
    expected_names = {
        "claim-boundary.json",
        "recovery-supplement.json",
        "sanitization-map.json",
        "SHA256SUMS",
    }
    children = list(output.iterdir())
    observed_names = {path.name for path in children}
    if len(children) != len(expected_names) or observed_names != expected_names:
        raise SanitizationError("Projection stage has an unexpected file set.")
    for child in children:
        _stage_metadata(child, label=f"Public stage file {child.name}")

    checksum_path = output / "SHA256SUMS"
    checksum_bytes, _checksum_metadata = _read_pinned_regular_file(
        checksum_path,
        label="Projection stage SHA256SUMS",
        max_bytes=_MAX_CHECKSUM_BYTES,
    )
    try:
        checksum_text = checksum_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SanitizationError("SHA256SUMS is not valid UTF-8.") from error
    expected_lines: list[str] = []
    loaded: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names - {"SHA256SUMS"}):
        path = output / name
        payload, _payload_metadata = _read_pinned_regular_file(
            path,
            label=f"Projection stage file {name}",
            max_bytes=_MAX_CANONICAL_JSON_BYTES,
        )
        digest = sha256_bytes(payload)
        expected_lines.append(f"{digest}  {name}")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SanitizationError(
                f"Public stage file {name} is unreadable."
            ) from error
        private_literals = context.forbidden_literals + tuple(forbidden_literals)
        scan_public_value(value, forbidden_literals=private_literals)
        try:
            validate_record(value)
        except ContractError as error:
            raise SanitizationError(
                f"Public stage file {name} violates its strict schema."
            ) from error
        if canonical_json_bytes(value) != payload:
            raise SanitizationError(f"Public stage file {name} is not canonical.")
        if value != expected_records[name]:
            raise SanitizationError(
                f"Public stage file {name} does not match the recomputed projection."
            )
        loaded[name] = value
    expected_checksum_bytes = "".join(f"{line}\n" for line in expected_lines).encode()
    if checksum_bytes != expected_checksum_bytes or checksum_text.count("\n") != len(
        expected_lines
    ):
        raise SanitizationError(
            "SHA256SUMS is incomplete, unsorted, duplicated, or stale."
        )
    expected_trace_targets = {
        ("recovery-supplement.json", pointer)
        for pointer, _value in _leaf_values(loaded["recovery-supplement.json"])
    }
    expected_trace_targets.update(
        ("claim-boundary.json", pointer)
        for pointer, _value in _leaf_values(loaded["claim-boundary.json"])
    )
    observed_trace_targets = {
        (entry["public_file"], entry["public_json_pointer"])
        for entry in loaded["sanitization-map.json"]["entries"]
    }
    if expected_trace_targets != observed_trace_targets:
        raise SanitizationError(
            "Raw-to-public traceability is incomplete or extraneous."
        )
    _verify_trace_sources(loaded["sanitization-map.json"], loaded, context)
    checks = {
        "strict-public-schema": True,
        "complete-raw-to-public-traceability": True,
        "private-value-absence": True,
        "numeric-recomputation": True,
        "claim-boundary-correctness": True,
        "complete-sorted-unique-sha256sums": True,
    }
    review = {
        "schema_version": PUBLICATION_REVIEW_SCHEMA,
        "review_id": review_id or "review_" + secrets.token_hex(16),
        "producer_role_id": producer_role_id,
        "reviewer_role_id": reviewer_role_id,
        "reviewed_at_utc": reviewed_at_utc or utc_now(),
        "checks": checks,
        "result": "passed",
        "reason_code": "NONE",
    }
    return validate_record(review)


def _stage_bindings(output: Path) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for name in sorted(path.name for path in output.iterdir()):
        path = output / name
        payload, metadata = _read_pinned_regular_file(
            path,
            label=f"Finalized stage file {name}",
            max_bytes=(
                _MAX_CHECKSUM_BYTES
                if name == "SHA256SUMS"
                else _MAX_CANONICAL_JSON_BYTES
            ),
        )
        bindings.append(
            {
                "relative_path": name,
                "size_bytes": metadata.st_size,
                "sha256": sha256_bytes(payload),
            }
        )
    return bindings


def _review_artifact_metadata(
    review: dict[str, Any], bindings: dict[str, Any]
) -> tuple[str, str, str]:
    seed = canonical_json_bytes({"review": review, "bindings": bindings})
    return (
        "artifact_" + sha256_bytes(b"artifact\0" + seed)[:32],
        "entry_" + sha256_bytes(b"review\0" + seed)[:32],
        "entry_" + sha256_bytes(b"bindings\0" + seed)[:32],
    )


def _utc_datetime(value: object, *, label: str) -> datetime:
    return datetime.fromisoformat(_require_utc_timestamp(value, label))


def _review_bindings(
    output: Path, context: VerifiedRecoveryContext, review: dict[str, Any]
) -> dict[str, Any]:
    return {
        "format_version": "aptus.experiment-publication-review-bindings.v1",
        "review_id": review["review_id"],
        "stage_files": _stage_bindings(output),
        "recovery_raw_manifest_sha256": context.recovery_verification[
            "raw_manifest_sha256"
        ],
        "control_raw_manifest_sha256": context.control_verification[
            "raw_manifest_sha256"
        ],
    }


def _verify_review_chronology(
    review: Mapping[str, Any], context: VerifiedRecoveryContext
) -> None:
    reviewed_at = _utc_datetime(review.get("reviewed_at_utc"), label="reviewed_at_utc")
    for label, verification in (
        ("recovery artifact", context.recovery_verification),
        ("control artifact", context.control_verification),
    ):
        sealed_at = _utc_datetime(
            verification["seal"]["sealed_at_utc"], label=f"{label} sealed_at_utc"
        )
        if reviewed_at < sealed_at:
            raise SanitizationError(
                "Independent review cannot predate its sealed provenance."
            )


def seal_projection_review(
    output: Path,
    review_artifact_output: Path,
    *,
    recovery_artifact: Path,
    control_artifact: Path,
    producer_role_id: str,
    reviewer_role_id: str,
    review_id: str,
    reviewed_at_utc: str,
) -> dict[str, Any]:
    """Create a durable review artifact in a distinct pre-finalization step."""

    context = load_verified_recovery_context(
        recovery_artifact=recovery_artifact, control_artifact=control_artifact
    )
    review = _verify_projection_stage(
        output,
        context=context,
        producer_role_id=producer_role_id,
        reviewer_role_id=reviewer_role_id,
        review_id=review_id,
        reviewed_at_utc=reviewed_at_utc,
    )
    _verify_review_chronology(review, context)
    bindings = _review_bindings(output, context, review)
    review_artifact_id, review_entry_id, bindings_entry_id = _review_artifact_metadata(
        review, bindings
    )
    writer = RawArtifactWriter(
        review_artifact_output,
        protected_artifact_id=review_artifact_id,
        record_kind="legacy-recovery",
        identity_bindings={
            "purpose": "independent-publication-review",
            "review_id": review["review_id"],
        },
        capture_tool={"name": "aptus-cuda-campaign-sanitizer", "version": "v1"},
        source_bindings={
            "recovery_raw_manifest_sha256": bindings["recovery_raw_manifest_sha256"],
            "control_raw_manifest_sha256": bindings["control_raw_manifest_sha256"],
        },
        provisional_retain_not_before_utc=context.recovery_verification["manifest"][
            "provisional_retain_not_before_utc"
        ],
        required_role_bindings={
            "independent-review": review_entry_id,
            "review-bindings": bindings_entry_id,
        },
    )
    writer.write_payload(
        canonical_json_bytes(review),
        "independent-review.json",
        role="independent-review",
        media_type="application/json",
        entry_id=review_entry_id,
        captured_at_utc=reviewed_at_utc,
    )
    writer.write_payload(
        canonical_json_bytes(bindings),
        "review-bindings.json",
        role="review-bindings",
        media_type="application/json",
        entry_id=bindings_entry_id,
        captured_at_utc=reviewed_at_utc,
    )
    return {
        "review": review,
        "sealed_review_artifact": writer.seal(),
    }


def _load_sealed_projection_review(
    output: Path,
    review_artifact: Path,
    *,
    context: VerifiedRecoveryContext,
    producer_role_id: str,
    reviewer_role_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        sealed_review = verify_sealed_artifact(review_artifact)
    except (OSError, ValueError) as error:
        raise SanitizationError("The prior review artifact is not sealed.") from error
    if sealed_review["manifest"]["record_kind"] != "legacy-recovery":
        raise SanitizationError("The prior review artifact has the wrong kind.")
    entries = _manifest_entries_by_id(sealed_review)
    review_entries = _entries_with_role(entries, "independent-review")
    binding_entries = _entries_with_role(entries, "review-bindings")
    if len(review_entries) != 1 or len(binding_entries) != 1 or len(entries) != 2:
        raise SanitizationError("The prior review artifact has an invalid inventory.")
    _review_bytes, raw_review = _read_verified_payload(
        review_artifact, review_entries[0], canonical_json=True
    )
    _binding_bytes, raw_bindings = _read_verified_payload(
        review_artifact, binding_entries[0], canonical_json=True
    )
    try:
        review = validate_record(raw_review, expected_schema=PUBLICATION_REVIEW_SCHEMA)
    except ContractError as error:
        raise SanitizationError(
            "The prior review payload violates its schema."
        ) from error
    bindings = _require_exact_object(
        raw_bindings,
        required={
            "format_version",
            "review_id",
            "stage_files",
            "recovery_raw_manifest_sha256",
            "control_raw_manifest_sha256",
        },
        label="prior review bindings",
    )
    expected_bindings = _review_bindings(output, context, review)
    if bindings != expected_bindings:
        raise SanitizationError(
            "Prior review bindings do not match exact candidate bytes."
        )
    review_artifact_id, review_entry_id, bindings_entry_id = _review_artifact_metadata(
        review, expected_bindings
    )
    manifest = sealed_review["manifest"]
    if (
        sealed_review["protected_artifact_id"] != review_artifact_id
        or manifest["identity_bindings"]
        != {
            "purpose": "independent-publication-review",
            "review_id": review["review_id"],
        }
        or manifest["capture_tool"]
        != {"name": "aptus-cuda-campaign-sanitizer", "version": "v1"}
        or manifest["source_bindings"]
        != {
            "recovery_raw_manifest_sha256": expected_bindings[
                "recovery_raw_manifest_sha256"
            ],
            "control_raw_manifest_sha256": expected_bindings[
                "control_raw_manifest_sha256"
            ],
        }
        or manifest["provisional_retain_not_before_utc"]
        != context.recovery_verification["manifest"][
            "provisional_retain_not_before_utc"
        ]
        or manifest["required_role_bindings"]
        != {
            "independent-review": review_entry_id,
            "review-bindings": bindings_entry_id,
        }
    ):
        raise SanitizationError("The prior review manifest metadata is not exact.")
    expected_review = _verify_projection_stage(
        output,
        context=context,
        producer_role_id=producer_role_id,
        reviewer_role_id=reviewer_role_id,
        review_id=review["review_id"],
        reviewed_at_utc=review["reviewed_at_utc"],
    )
    if review != expected_review:
        raise SanitizationError(
            "The sealed prior review differs from independent recomputation."
        )
    _verify_review_chronology(review, context)
    try:
        if (
            verify_sealed_artifact(review_artifact)["raw_manifest_sha256"]
            != sealed_review["raw_manifest_sha256"]
        ):
            raise SanitizationError("The prior review artifact changed during use.")
    except (OSError, ValueError) as error:
        if isinstance(error, SanitizationError):
            raise
        raise SanitizationError(
            "The prior review artifact changed during use."
        ) from error
    return review, expected_bindings, sealed_review


def _finalization_content_id(record: Mapping[str, Any]) -> str:
    identity = {key: value for key, value in record.items() if key != "finalization_id"}
    return "finalization_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]


def _finalization_record(
    review: Mapping[str, Any],
    sealed_review: Mapping[str, Any],
    *,
    producer_role_id: str,
    reviewer_role_id: str,
    finalizer_role_id: str,
    finalized_at_utc: str,
) -> dict[str, Any]:
    if _ROLE_ID.fullmatch(finalizer_role_id) is None or finalizer_role_id in {
        producer_role_id,
        reviewer_role_id,
    }:
        raise SanitizationError(
            "Finalizer role must be bounded and distinct from producer and reviewer."
        )
    finalized_at = _utc_datetime(finalized_at_utc, label="finalized_at_utc")
    reviewed_at = _utc_datetime(review["reviewed_at_utc"], label="reviewed_at_utc")
    if finalized_at < reviewed_at:
        raise SanitizationError("Finalization cannot predate its durable prior review.")
    without_id = {
        "format_version": PUBLICATION_FINALIZATION_SCHEMA,
        "review_id": review["review_id"],
        "finalizer_role_id": finalizer_role_id,
        "finalized_at_utc": finalized_at_utc,
        "review_artifact_id": sealed_review["protected_artifact_id"],
        "review_raw_manifest_sha256": sealed_review["raw_manifest_sha256"],
        "review_raw_manifest_size_bytes": sealed_review["raw_manifest_size_bytes"],
    }
    return {**without_id, "finalization_id": _finalization_content_id(without_id)}


def _write_final_candidate_packet(
    output: Path,
    *,
    projection: Projection,
    review: dict[str, Any],
    bindings: dict[str, Any],
    finalization: dict[str, Any],
    forbidden_literals: Sequence[str],
) -> dict[str, str]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Final candidate packet already exists: {output}")
    output.mkdir(mode=0o700, parents=False)
    records = {
        "claim-boundary.json": projection.claim_boundary,
        "finalization.json": finalization,
        "independent-review.json": review,
        "recovery-supplement.json": projection.supplement,
        "review-bindings.json": bindings,
        "sanitization-map.json": projection.sanitization_map,
    }
    digests: dict[str, str] = {}
    for name, record in records.items():
        scan_public_value(record, forbidden_literals=forbidden_literals)
        payload = canonical_json_bytes(record)
        _write_exclusive(output / name, payload)
        digests[name] = sha256_bytes(payload)
    checksum = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(digests.items())
    ).encode("utf-8")
    _write_exclusive(output / "SHA256SUMS", checksum)
    descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digests


def _verify_final_candidate_packet(
    output: Path,
    *,
    projection: Projection,
    review: dict[str, Any],
    bindings: dict[str, Any],
    finalization: dict[str, Any],
    forbidden_literals: Sequence[str],
) -> None:
    expected_records = {
        "claim-boundary.json": projection.claim_boundary,
        "finalization.json": finalization,
        "independent-review.json": review,
        "recovery-supplement.json": projection.supplement,
        "review-bindings.json": bindings,
        "sanitization-map.json": projection.sanitization_map,
    }
    expected_names = set(expected_records) | {"SHA256SUMS"}
    if output.is_symlink() or not output.is_dir():
        raise SanitizationError("Final candidate packet must be a regular directory.")
    metadata = output.stat()
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o700:
        raise SanitizationError("Final candidate packet directory mode must be 0700.")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise SanitizationError("Final candidate packet is owned by another user.")
    children = list(output.iterdir())
    if (
        len(children) != len(expected_names)
        or {item.name for item in children} != expected_names
    ):
        raise SanitizationError("Final candidate packet has an unexpected file set.")
    for child in children:
        _stage_metadata(child, label=f"Final candidate file {child.name}")
    checksum_lines: list[str] = []
    for name in sorted(expected_records):
        payload, _metadata = _read_pinned_regular_file(
            output / name,
            label=f"Final candidate file {name}",
            max_bytes=_MAX_CANONICAL_JSON_BYTES,
        )
        if payload != canonical_json_bytes(expected_records[name]):
            raise SanitizationError(
                f"Final candidate file {name} is not the reviewed record."
            )
        scan_public_value(expected_records[name], forbidden_literals=forbidden_literals)
        checksum_lines.append(f"{sha256_bytes(payload)}  {name}\n")
    checksum_payload, _metadata = _read_pinned_regular_file(
        output / "SHA256SUMS",
        label="Final candidate SHA256SUMS",
        max_bytes=_MAX_CHECKSUM_BYTES,
    )
    if checksum_payload != "".join(checksum_lines).encode("utf-8"):
        raise SanitizationError("Final candidate SHA256SUMS is incomplete or stale.")


def finalize_projection_stage(
    output: Path,
    finalized_candidate_output: Path,
    review_artifact: Path,
    *,
    recovery_artifact: Path,
    control_artifact: Path,
    producer_role_id: str,
    reviewer_role_id: str,
    finalizer_role_id: str,
    finalized_at_utc: str,
) -> dict[str, Any]:
    """Consume a durable prior review to create a nonpublished final candidate."""

    context = load_verified_recovery_context(
        recovery_artifact=recovery_artifact, control_artifact=control_artifact
    )
    review, bindings, sealed_review = _load_sealed_projection_review(
        output,
        review_artifact,
        context=context,
        producer_role_id=producer_role_id,
        reviewer_role_id=reviewer_role_id,
    )
    finalization = _finalization_record(
        review,
        sealed_review,
        producer_role_id=producer_role_id,
        reviewer_role_id=reviewer_role_id,
        finalizer_role_id=finalizer_role_id,
        finalized_at_utc=finalized_at_utc,
    )
    candidate_digests = _write_final_candidate_packet(
        finalized_candidate_output,
        projection=_project_recovery_supplement(
            source=context.source,
            expected_rows=context.expected_rows,
            context=context,
        ),
        review=review,
        bindings=bindings,
        finalization=finalization,
        forbidden_literals=context.forbidden_literals,
    )
    return {
        "review": review,
        "finalization": finalization,
        "sealed_review_artifact": sealed_review,
        "final_candidate_digests": candidate_digests,
    }


def verify_finalized_projection(
    output: Path,
    finalized_candidate_output: Path,
    review_artifact: Path,
    *,
    recovery_artifact: Path,
    control_artifact: Path,
    producer_role_id: str,
    reviewer_role_id: str,
    finalizer_role_id: str,
) -> dict[str, Any]:
    """Verify a nonpublished candidate and its consumed durable prior review."""

    context = load_verified_recovery_context(
        recovery_artifact=recovery_artifact, control_artifact=control_artifact
    )
    review, expected_bindings, sealed_review = _load_sealed_projection_review(
        output,
        review_artifact,
        context=context,
        producer_role_id=producer_role_id,
        reviewer_role_id=reviewer_role_id,
    )
    finalization_path = finalized_candidate_output / "finalization.json"
    finalization_payload, _finalization_metadata = _read_pinned_regular_file(
        finalization_path,
        label="Final candidate finalization",
        max_bytes=_MAX_CANONICAL_JSON_BYTES,
    )
    try:
        raw_finalization = json.loads(finalization_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SanitizationError(
            "Final candidate finalization is unreadable."
        ) from error
    required_finalization_fields = {
        "format_version",
        "finalization_id",
        "review_id",
        "finalizer_role_id",
        "finalized_at_utc",
        "review_artifact_id",
        "review_raw_manifest_sha256",
        "review_raw_manifest_size_bytes",
    }
    finalization = _require_exact_object(
        raw_finalization,
        required=required_finalization_fields,
        label="final candidate finalization",
    )
    if (
        finalization.get("format_version") != PUBLICATION_FINALIZATION_SCHEMA
        or not isinstance(finalization.get("finalization_id"), str)
        or _FINALIZATION_ID.fullmatch(finalization["finalization_id"]) is None
        or finalization.get("finalizer_role_id") != finalizer_role_id
    ):
        raise SanitizationError("Final candidate finalization identity is invalid.")
    expected_finalization = _finalization_record(
        review,
        sealed_review,
        producer_role_id=producer_role_id,
        reviewer_role_id=reviewer_role_id,
        finalizer_role_id=finalizer_role_id,
        finalized_at_utc=finalization["finalized_at_utc"],
    )
    if finalization != expected_finalization:
        raise SanitizationError("Final candidate finalization is not exact.")
    projection = _project_recovery_supplement(
        source=context.source,
        expected_rows=context.expected_rows,
        context=context,
    )
    _verify_final_candidate_packet(
        finalized_candidate_output,
        projection=projection,
        review=review,
        bindings=expected_bindings,
        finalization=finalization,
        forbidden_literals=context.forbidden_literals,
    )
    return {
        **review,
        "finalization_id": finalization["finalization_id"],
        "finalizer_role_id": finalization["finalizer_role_id"],
        "finalized_at_utc": finalization["finalized_at_utc"],
    }
