"""Read-only, fail-closed publication eligibility for CUDA evidence.

The evaluator deliberately stays separate from capture, storage, sanitization,
and operator commands: it accepts already-materialized evidence, independently
verifies every binding it relies on, and never writes or repairs evidence.  A
separate explicit helper seals a nonpublished publication candidate before that
read-only decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import ctypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    ContractError,
    PROCEDURAL_ROLE_ID_RE,
    canonical_json_bytes,
    compact_canonical_json_bytes,
    sha256_bytes,
    validate_safe_relative_path,
    validate_record,
)
from .sanitizer import RECOVERY_SUPPLEMENT_SCHEMA, verify_finalized_projection
from .storage import (
    COPY_VERIFICATION_CADENCE_DAYS,
    OFF_HOST_RETRIEVAL_CADENCE_DAYS,
    RETENTION_POLICY_ID,
    RawArtifactWriter,
    verify_sealed_artifact,
)


EXTERNAL_RECOVERY_ATTESTATION_SCHEMA = (
    "aptus.cuda-campaign-external-recovery-attestation.v1"
)
EXTERNAL_EVIDENCE_SCHEMA = "aptus.cuda-campaign-external-evidence.v1"
PUBLICATION_CANDIDATE_SCHEMA = "aptus.cuda-campaign-publication-candidate.v1"
PUBLICATION_INELIGIBILITY_REASON_CODES = (
    "INPUT_INVALID",
    "ARTIFACT_VERIFICATION_FAILED",
    "ARTIFACT_ID_MISMATCH",
    "ARTIFACT_MANIFEST_DIGEST_MISMATCH",
    "ARTIFACT_MANIFEST_SIZE_MISMATCH",
    "CAPTURE_KIND_NOT_PUBLICATION_QUALIFYING",
    "NATIVE_OUTCOME_NOT_PASSED",
    "RECEIPT_CHAIN_INVALID",
    "RECEIPT_ARTIFACT_BINDING_MISMATCH",
    "VERIFIED_COPY_COUNT_INSUFFICIENT",
    "FAILURE_DOMAIN_COUNT_INSUFFICIENT",
    "COPY_VERIFICATION_NOT_CURRENT",
    "EXTERNAL_RECOVERY_ATTESTATION_INVALID",
    "EXTERNAL_RECOVERY_ATTESTATION_UNBOUND",
    "EXTERNAL_RECOVERY_EVIDENCE_INVALID",
    "OFF_HOST_RETRIEVAL_NOT_CURRENT",
    "RETENTION_RECEIPT_INVALID",
    "RETENTION_NOT_CURRENT",
    "RETENTION_RENEWAL_NOT_CURRENT",
    "CLAIM_STATE_INVALID",
    "CLAIM_SUSPENDED",
    "CLAIM_WITHDRAWN",
    "SANITIZER_FINALIZATION_INVALID",
    "INDEPENDENT_REVIEW_NOT_PASSED",
    "PUBLICATION_CANDIDATE_INVALID",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID = re.compile(r"^artifact_[0-9a-f]{32}$")
_COPY_ID = re.compile(r"^copy_[0-9a-f]{32}$")
_DOMAIN_ID = re.compile(r"^domain_[0-9a-f]{32}$")
_RECEIPT_ID = re.compile(r"^receipt_[0-9a-f]{32}$")
_ATTESTATION_ID = re.compile(r"^attest_[0-9a-f]{32}$")
_CANDIDATE_ID = re.compile(r"^candidate_[0-9a-f]{32}$")
_CAMPAIGN_ID = re.compile(r"^campaign_[0-9a-f]{20}$")
_CLAIM_KEY = re.compile(r"^[a-z][a-z0-9-]{2,127}$")
_OPAQUE_ID = re.compile(r"^[a-z][a-z0-9-]*_[0-9a-f]{32}$")
_ROLE_ID = PROCEDURAL_ROLE_ID_RE
_COPY_DETAIL_FIELDS = {
    "copy_id",
    "failure_domain_id",
    "off_experiment_host",
    "verification_result",
}
_RETRIEVAL_DETAIL_FIELDS = {
    "source_copy_id",
    "source_failure_domain_id",
    "destination_restore_id",
    "started_at_utc",
    "finished_at_utc",
    "duration_ns",
    "restored_file_count",
    "restored_total_bytes",
    "expected_raw_manifest_sha256",
    "observed_raw_manifest_sha256",
    "mismatch_count",
    "verification_result",
}
_RETENTION_DETAIL_FIELDS = {
    "retention_policy_id",
    "retain_not_before_utc",
    "verification_result",
}
_EVIDENCE_REFERENCE_FIELDS = {"reference_id", "sha256"}
_ATTESTATION_FIELDS = {
    "schema_version",
    "attestation_id",
    "attester_role_id",
    "evidence_custodian_role_id",
    "attested_at_utc",
    "protected_artifact_id",
    "raw_manifest_sha256",
    "raw_manifest_size_bytes",
    "copy_id",
    "failure_domain_id",
    "copy_verification_receipt_id",
    "retrieval_receipt_id",
    "off_host_storage_evidence",
    "encryption_in_transit_evidence",
    "encryption_at_rest_evidence",
    "key_custodian_role_id",
    "key_custody_evidence",
    "recovery_procedure_id",
    "recovery_procedure_evidence",
}
_EXTERNAL_EVIDENCE_COMMON_FIELDS = {
    "schema_version",
    "evidence_kind",
    "reference_id",
    "created_at_utc",
    "issuer_role_id",
    "protected_artifact_id",
    "raw_manifest_sha256",
    "raw_manifest_size_bytes",
    "copy_id",
    "failure_domain_id",
    "verification_result",
}
_EXTERNAL_EVIDENCE_KINDS = {
    "off_host_storage_evidence": (
        "off-host-storage",
        {
            "copy_verification_receipt_id",
            "storage_control_id",
            "off_experiment_host",
        },
    ),
    "encryption_in_transit_evidence": (
        "encryption-in-transit",
        {
            "copy_verification_receipt_id",
            "transport_control_id",
            "transport_security",
        },
    ),
    "encryption_at_rest_evidence": (
        "encryption-at-rest",
        {
            "copy_verification_receipt_id",
            "encryption_control_id",
            "encryption_state",
        },
    ),
    "key_custody_evidence": (
        "key-custody",
        {"key_custodian_role_id", "key_control_id", "custody_state"},
    ),
    "recovery_procedure_evidence": (
        "recovery-procedure",
        {"retrieval_receipt_id", "recovery_procedure_id", "procedure_state"},
    ),
}
_MAX_EXTERNAL_EVIDENCE_BYTES = 64 * 1024
_MAX_PRIVATE_JSON_BYTES = 4 * 1024 * 1024
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
_RENEWAL_LEAD_DAYS = 90


@dataclass(frozen=True)
class FinalizedSanitizerBinding:
    """Paths and independent roles needed to reverify a finalized projection."""

    projection_stage: Path
    finalized_candidate_output: Path
    review_artifact: Path
    recovery_artifact: Path
    control_artifact: Path
    producer_role_id: str
    reviewer_role_id: str
    finalizer_role_id: str


@dataclass(frozen=True)
class PublicationCandidateBinding:
    """Expected sealed candidate identity and claim boundary."""

    artifact: Path
    campaign_id: str
    claim_key: str
    candidate_producer_role_id: str


@dataclass(frozen=True)
class PublicationEligibilityResult:
    """Deterministic publication decision with stable ordered reason codes."""

    eligible: bool
    reason_codes: tuple[str, ...]
    evaluated_at_utc: str | None
    protected_artifact_id: str | None
    raw_manifest_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reason_codes": list(self.reason_codes),
            "evaluated_at_utc": self.evaluated_at_utc,
            "protected_artifact_id": self.protected_artifact_id,
            "raw_manifest_sha256": self.raw_manifest_sha256,
        }


class _InvalidEvidence(ValueError):
    pass


def _parse_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise _InvalidEvidence("invalid timestamp") from error
    else:
        raise _InvalidEvidence("invalid timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _InvalidEvidence("timestamp lacks an offset")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _content_receipt_id(record: Mapping[str, Any]) -> str:
    identity = {key: value for key, value in record.items() if key != "receipt_id"}
    return "receipt_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]


def _validate_receipt_chain(
    receipts: Sequence[Mapping[str, Any]], *, now: datetime
) -> list[dict[str, Any]]:
    if isinstance(receipts, (str, bytes)) or not receipts:
        raise _InvalidEvidence("receipt chain is empty")
    records: dict[str, dict[str, Any]] = {}
    for raw in receipts:
        if not isinstance(raw, Mapping):
            raise _InvalidEvidence("receipt is not an object")
        try:
            record = validate_record(raw, "aptus.experiment-evidence-receipt.v1")
        except (ContractError, TypeError, ValueError) as error:
            raise _InvalidEvidence("receipt violates its contract") from error
        receipt_id = record["receipt_id"]
        if (
            not isinstance(receipt_id, str)
            or _RECEIPT_ID.fullmatch(receipt_id) is None
            or receipt_id != _content_receipt_id(record)
            or receipt_id in records
        ):
            raise _InvalidEvidence("receipt content ID is invalid")
        records[receipt_id] = record

    roots = [
        receipt_id
        for receipt_id, record in records.items()
        if record["previous_receipt_id"] is None
    ]
    if len(roots) != 1:
        raise _InvalidEvidence("receipt chain does not have one root")
    children: dict[str, list[str]] = {}
    for receipt_id, record in records.items():
        previous = record["previous_receipt_id"]
        if previous is None:
            continue
        if previous not in records:
            raise _InvalidEvidence("receipt predecessor is missing")
        children.setdefault(previous, []).append(receipt_id)
    if any(len(items) != 1 for items in children.values()):
        raise _InvalidEvidence("receipt chain forks")

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: str | None = roots[0]
    prior_created: datetime | None = None
    while current is not None:
        if current in seen:
            raise _InvalidEvidence("receipt chain cycles")
        seen.add(current)
        record = records[current]
        created = _parse_utc(record["created_at_utc"])
        if created > now or (prior_created is not None and created < prior_created):
            raise _InvalidEvidence("receipt time is impossible or out of order")
        prior_created = created
        ordered.append(record)
        next_items = children.get(current, [])
        current = next_items[0] if next_items else None
    if seen != set(records):
        raise _InvalidEvidence("receipt chain is disconnected")
    return ordered


def _receipt_binds_artifact(
    receipt: Mapping[str, Any], verified_artifact: Mapping[str, Any]
) -> bool:
    return (
        receipt["protected_artifact_id"] == verified_artifact["protected_artifact_id"]
        and receipt["raw_manifest_sha256"] == verified_artifact["raw_manifest_sha256"]
        and receipt["raw_manifest_size_bytes"]
        == verified_artifact["raw_manifest_size_bytes"]
    )


def _copy_receipt_binding(
    receipt: Mapping[str, Any], *, now: datetime
) -> dict[str, Any] | None:
    details = receipt.get("details")
    if (
        receipt.get("kind") != "copy-verification"
        or receipt.get("result") != "passed"
        or not isinstance(details, Mapping)
        or set(details) != _COPY_DETAIL_FIELDS
        or not isinstance(details.get("copy_id"), str)
        or _COPY_ID.fullmatch(details["copy_id"]) is None
        or not isinstance(details.get("failure_domain_id"), str)
        or _DOMAIN_ID.fullmatch(details["failure_domain_id"]) is None
        or type(details.get("off_experiment_host")) is not bool
        or details.get("verification_result") != "passed"
    ):
        return None
    created = _parse_utc(receipt["created_at_utc"])
    return {
        "receipt_id": receipt["receipt_id"],
        "copy_id": details["copy_id"],
        "failure_domain_id": details["failure_domain_id"],
        "off_experiment_host": details["off_experiment_host"],
        "issuer_role_id": receipt["issuer_role_id"],
        "created_at": created,
        "current": now <= created + timedelta(days=COPY_VERIFICATION_CADENCE_DAYS),
    }


def _latest_copy_bindings(
    receipts: Sequence[Mapping[str, Any]], *, now: datetime
) -> list[dict[str, Any]]:
    """Return only the latest passing state for each stable copy binding."""

    copy_domains: dict[str, str] = {}
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for receipt in receipts:
        if receipt.get("kind") != "copy-verification":
            continue
        details = receipt.get("details")
        if (
            not isinstance(details, Mapping)
            or set(details) != _COPY_DETAIL_FIELDS
            or not isinstance(details.get("copy_id"), str)
            or _COPY_ID.fullmatch(details["copy_id"]) is None
            or not isinstance(details.get("failure_domain_id"), str)
            or _DOMAIN_ID.fullmatch(details["failure_domain_id"]) is None
            or type(details.get("off_experiment_host")) is not bool
            or receipt.get("result") not in {"passed", "failed"}
            or details.get("verification_result") != receipt.get("result")
        ):
            raise _InvalidEvidence("copy verification binding is invalid")
        copy_id = details["copy_id"]
        failure_domain_id = details["failure_domain_id"]
        prior_domain = copy_domains.setdefault(copy_id, failure_domain_id)
        if prior_domain != failure_domain_id:
            raise _InvalidEvidence("copy ID changes failure domain")
        latest[(copy_id, failure_domain_id)] = receipt

    result: list[dict[str, Any]] = []
    for receipt in latest.values():
        binding = _copy_receipt_binding(receipt, now=now)
        if binding is not None:
            result.append(binding)
    return result


def _has_two_distinct_copy_bindings(bindings: Sequence[Mapping[str, Any]]) -> bool:
    for index, first in enumerate(bindings):
        for second in bindings[index + 1 :]:
            if (
                first["copy_id"] != second["copy_id"]
                and first["failure_domain_id"] != second["failure_domain_id"]
            ):
                return True
    return False


def _current_retrievals(
    receipts: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    valid_copies: Sequence[Mapping[str, Any]],
    verified_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    valid_copy_pairs = {
        (item["copy_id"], item["failure_domain_id"])
        for item in valid_copies
        if item["current"]
    }
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for receipt in receipts:
        if receipt.get("kind") != "retrieval":
            continue
        details = receipt.get("details")
        if (
            not isinstance(details, Mapping)
            or set(details) != _RETRIEVAL_DETAIL_FIELDS
            or not isinstance(details.get("source_copy_id"), str)
            or _COPY_ID.fullmatch(details["source_copy_id"]) is None
            or not isinstance(details.get("source_failure_domain_id"), str)
            or _DOMAIN_ID.fullmatch(details["source_failure_domain_id"]) is None
            or not isinstance(details.get("destination_restore_id"), str)
            or _OPAQUE_ID.fullmatch(details["destination_restore_id"]) is None
            or receipt.get("result") not in {"passed", "failed"}
            or details.get("verification_result") != receipt.get("result")
            or any(
                isinstance(details.get(field), bool)
                or not isinstance(details.get(field), int)
                or details[field] < 0
                for field in (
                    "duration_ns",
                    "restored_file_count",
                    "restored_total_bytes",
                    "mismatch_count",
                )
            )
            or not isinstance(details.get("expected_raw_manifest_sha256"), str)
            or _SHA256.fullmatch(details["expected_raw_manifest_sha256"]) is None
            or details["expected_raw_manifest_sha256"]
            != verified_artifact["raw_manifest_sha256"]
            or (
                details.get("observed_raw_manifest_sha256") is not None
                and (
                    not isinstance(details["observed_raw_manifest_sha256"], str)
                    or _SHA256.fullmatch(details["observed_raw_manifest_sha256"])
                    is None
                )
            )
            or (
                receipt.get("result") == "passed"
                and (
                    details.get("observed_raw_manifest_sha256")
                    != details["expected_raw_manifest_sha256"]
                    or details["observed_raw_manifest_sha256"]
                    != verified_artifact["raw_manifest_sha256"]
                    or details["restored_file_count"]
                    != verified_artifact["file_count"] + 2
                    or details["restored_total_bytes"]
                    != (
                        verified_artifact["total_bytes"]
                        + verified_artifact["raw_manifest_size_bytes"]
                        + len(canonical_json_bytes(verified_artifact["seal"]))
                    )
                    or details["mismatch_count"] != 0
                )
            )
            or (receipt.get("result") == "failed" and details["mismatch_count"] < 1)
        ):
            raise _InvalidEvidence("retrieval binding is invalid")
        _parse_utc(details["started_at_utc"])
        _parse_utc(details["finished_at_utc"])
        source_pair = (
            details["source_copy_id"],
            details["source_failure_domain_id"],
        )
        latest[source_pair] = receipt

    result: list[dict[str, Any]] = []
    for source_pair, receipt in latest.items():
        if receipt.get("result") != "passed":
            continue
        details = receipt["details"]
        started = _parse_utc(details["started_at_utc"])
        finished = _parse_utc(details["finished_at_utc"])
        created = _parse_utc(receipt["created_at_utc"])
        if (
            source_pair not in valid_copy_pairs
            or not (started <= finished <= created <= now)
            or now > finished + timedelta(days=OFF_HOST_RETRIEVAL_CADENCE_DAYS)
        ):
            continue
        result.append(
            {
                "receipt_id": receipt["receipt_id"],
                "copy_id": source_pair[0],
                "failure_domain_id": source_pair[1],
                "finished_at": finished,
                "created_at": created,
                "issuer_role_id": receipt["issuer_role_id"],
            }
        )
    return sorted(result, key=lambda item: item["finished_at"])


def _claim_state(
    receipts: Sequence[Mapping[str, Any]],
) -> tuple[str, datetime | None, datetime | None, set[str]]:
    """Evaluate append-only claim control receipts without optimistic defaults."""

    state = "active"
    last_suspension_at: datetime | None = None
    last_restoration_at: datetime | None = None
    reasons: set[str] = set()
    for receipt in receipts:
        kind = receipt["kind"]
        expected_result = {
            "claim-suspension": "suspended",
            "claim-restoration": "restored",
            "claim-withdrawal": "withdrawn",
        }.get(kind)
        if expected_result is not None and receipt.get("result") != expected_result:
            reasons.add("CLAIM_STATE_INVALID")
            continue
        if kind == "claim-withdrawal":
            state = "withdrawn"
            reasons.discard("CLAIM_SUSPENDED")
            reasons.add("CLAIM_WITHDRAWN")
            continue
        if state == "withdrawn" and kind in {
            "claim-suspension",
            "claim-restoration",
        }:
            reasons.add("CLAIM_STATE_INVALID")
            continue
        if kind == "claim-suspension":
            if state == "suspended":
                reasons.add("CLAIM_STATE_INVALID")
                continue
            state = "suspended"
            last_suspension_at = _parse_utc(receipt["created_at_utc"])
            last_restoration_at = None
            reasons.add("CLAIM_SUSPENDED")
        elif kind == "claim-restoration":
            if state != "suspended":
                reasons.add("CLAIM_STATE_INVALID")
                continue
            state = "restored"
            last_restoration_at = _parse_utc(receipt["created_at_utc"])
            reasons.discard("CLAIM_SUSPENDED")
    return state, last_suspension_at, last_restoration_at, reasons


def _retention_state(
    receipts: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    provisional_deadline: datetime,
) -> tuple[str | None, set[str]]:
    relevant = [
        receipt for receipt in receipts if receipt["kind"] in {"retention", "renewal"}
    ]
    if not relevant:
        return None, {"RETENTION_RECEIPT_INVALID"}
    latest = relevant[-1]
    details = latest.get("details")
    if (
        latest.get("result") != "active"
        or not isinstance(details, Mapping)
        or set(details) != _RETENTION_DETAIL_FIELDS
        or details.get("retention_policy_id") != RETENTION_POLICY_ID
        or details.get("verification_result") != "passed"
    ):
        return None, {"RETENTION_RECEIPT_INVALID"}
    try:
        deadline = _parse_utc(details["retain_not_before_utc"])
    except _InvalidEvidence:
        return None, {"RETENTION_RECEIPT_INVALID"}
    if deadline < provisional_deadline:
        return latest["receipt_id"], {"RETENTION_RECEIPT_INVALID"}
    if deadline <= now:
        return latest["receipt_id"], {"RETENTION_NOT_CURRENT"}
    if deadline <= now + timedelta(days=_RENEWAL_LEAD_DAYS):
        return latest["receipt_id"], {"RETENTION_RENEWAL_NOT_CURRENT"}
    return latest["receipt_id"], set()


def _evidence_reference(value: object) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == _EVIDENCE_REFERENCE_FIELDS
        and isinstance(value.get("reference_id"), str)
        and _OPAQUE_ID.fullmatch(value["reference_id"]) is not None
        and isinstance(value.get("sha256"), str)
        and _SHA256.fullmatch(value["sha256"]) is not None
    )


def _validate_external_attestation(
    raw: Mapping[str, Any], *, now: datetime
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _ATTESTATION_FIELDS:
        raise _InvalidEvidence("external attestation fields are not exact")
    record = dict(raw)
    string_patterns = {
        "attestation_id": _ATTESTATION_ID,
        "attester_role_id": _ROLE_ID,
        "evidence_custodian_role_id": _ROLE_ID,
        "protected_artifact_id": _ARTIFACT_ID,
        "raw_manifest_sha256": _SHA256,
        "copy_id": _COPY_ID,
        "failure_domain_id": _DOMAIN_ID,
        "copy_verification_receipt_id": _RECEIPT_ID,
        "retrieval_receipt_id": _RECEIPT_ID,
        "key_custodian_role_id": _ROLE_ID,
        "recovery_procedure_id": _OPAQUE_ID,
    }
    if record["schema_version"] != EXTERNAL_RECOVERY_ATTESTATION_SCHEMA:
        raise _InvalidEvidence("external attestation schema is unknown")
    if any(
        not isinstance(record.get(field), str)
        or pattern.fullmatch(record[field]) is None
        for field, pattern in string_patterns.items()
    ):
        raise _InvalidEvidence("external attestation identifier is invalid")
    if (
        not isinstance(record["raw_manifest_size_bytes"], int)
        or isinstance(record["raw_manifest_size_bytes"], bool)
        or record["raw_manifest_size_bytes"] < 1
    ):
        raise _InvalidEvidence("external attestation size is invalid")
    evidence_fields = (
        "off_host_storage_evidence",
        "encryption_in_transit_evidence",
        "encryption_at_rest_evidence",
        "key_custody_evidence",
        "recovery_procedure_evidence",
    )
    if not all(_evidence_reference(record[field]) for field in evidence_fields):
        raise _InvalidEvidence("external attestation lacks evidence references")
    if record["attester_role_id"] in {
        record["evidence_custodian_role_id"],
        record["key_custodian_role_id"],
    }:
        raise _InvalidEvidence("external attester is not independent")
    attested_at = _parse_utc(record["attested_at_utc"])
    if attested_at > now:
        raise _InvalidEvidence("external attestation is future-dated")
    identity = {key: value for key, value in record.items() if key != "attestation_id"}
    expected_id = "attest_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]
    if record["attestation_id"] != expected_id:
        raise _InvalidEvidence("external attestation content ID is invalid")
    record["attested_at"] = attested_at
    return record


def _attestation_is_bound(
    attestation: Mapping[str, Any],
    *,
    verified_artifact: Mapping[str, Any],
    copies: Sequence[Mapping[str, Any]],
    retrievals: Sequence[Mapping[str, Any]],
) -> bool:
    copy = next(
        (
            item
            for item in copies
            if item["receipt_id"] == attestation["copy_verification_receipt_id"]
        ),
        None,
    )
    retrieval = next(
        (
            item
            for item in retrievals
            if item["receipt_id"] == attestation["retrieval_receipt_id"]
        ),
        None,
    )
    if copy is None or retrieval is None:
        return False
    return bool(
        copy["current"]
        and copy["off_experiment_host"] is True
        and attestation["protected_artifact_id"]
        == verified_artifact["protected_artifact_id"]
        and attestation["raw_manifest_sha256"]
        == verified_artifact["raw_manifest_sha256"]
        and attestation["raw_manifest_size_bytes"]
        == verified_artifact["raw_manifest_size_bytes"]
        and attestation["copy_id"] == copy["copy_id"] == retrieval["copy_id"]
        and attestation["failure_domain_id"]
        == copy["failure_domain_id"]
        == retrieval["failure_domain_id"]
        and attestation["evidence_custodian_role_id"]
        == copy["issuer_role_id"]
        == retrieval["issuer_role_id"]
        and attestation["attested_at"] >= copy["created_at"]
        and attestation["attested_at"] >= retrieval["created_at"]
    )


def _metadata_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
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


def _private_regular_file(metadata: os.stat_result) -> bool:
    return bool(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and (os.name != "posix" or stat.S_IMODE(metadata.st_mode) == 0o600)
        and (not hasattr(os, "getuid") or metadata.st_uid == os.getuid())
    )


def _private_directory_metadata(metadata: os.stat_result) -> bool:
    return bool(
        stat.S_ISDIR(metadata.st_mode)
        and (os.name != "posix" or stat.S_IMODE(metadata.st_mode) == 0o700)
        and (not hasattr(os, "getuid") or metadata.st_uid == os.getuid())
    )


def _linux_directory_mutation_watch(path: Path) -> int | None:
    """Watch one Linux directory for transient entry replacement."""

    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        initialize = libc.inotify_init1
        initialize.argtypes = [ctypes.c_int]
        initialize.restype = ctypes.c_int
        add_watch = libc.inotify_add_watch
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        descriptor = initialize(
            getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        if descriptor < 0:
            return None
        mutation_mask = (
            0x00000004
            | 0x00000040
            | 0x00000080
            | 0x00000100
            | 0x00000200
            | 0x00000400
            | 0x00000800
        )
        if add_watch(descriptor, os.fsencode(path), mutation_mask) < 0:
            os.close(descriptor)
            return None
        return descriptor
    except (AttributeError, OSError):
        return None


def _linux_watch_saw_mutation(descriptor: int | None) -> bool:
    if descriptor is None:
        return False
    try:
        return bool(os.read(descriptor, 64 * 1024))
    except BlockingIOError:
        return False
    except OSError:
        return True


def _regular_file_bytes(
    path: object,
    *,
    max_bytes: int = _MAX_PRIVATE_JSON_BYTES,
) -> tuple[bytes, str, tuple[int, int]]:
    if not isinstance(path, Path):
        raise _InvalidEvidence("external evidence path is not a Path")
    try:
        before = path.lstat()
    except OSError as error:
        raise _InvalidEvidence("external evidence is unavailable") from error
    if not _private_regular_file(before):
        raise _InvalidEvidence("external evidence is not a regular private file")
    before_fingerprint = _metadata_fingerprint(before)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise _InvalidEvidence("external evidence cannot be opened safely") from error
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        if (
            not _private_regular_file(opened)
            or _metadata_fingerprint(opened) != before_fingerprint
        ):
            raise _InvalidEvidence("external evidence changed while opening")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            if len(payload) + len(block) > max_bytes:
                raise _InvalidEvidence("private JSON evidence exceeds its size limit")
            payload.extend(block)
            digest.update(block)
        finished = os.fstat(descriptor)
        if (
            not _private_regular_file(finished)
            or _metadata_fingerprint(finished) != before_fingerprint
        ):
            raise _InvalidEvidence("external evidence changed while hashing")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as error:
        raise _InvalidEvidence("external evidence changed after hashing") from error
    if (
        not _private_regular_file(after)
        or _metadata_fingerprint(after) != before_fingerprint
    ):
        raise _InvalidEvidence("external evidence path changed while hashing")
    return bytes(payload), digest.hexdigest(), (opened.st_dev, opened.st_ino)


def _content_evidence_reference_id(record: Mapping[str, Any]) -> str:
    identity = {key: value for key, value in record.items() if key != "reference_id"}
    return "evidence_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]


def _validate_typed_external_evidence(
    raw: object,
    *,
    field: str,
    attestation: Mapping[str, Any],
    verified_artifact: Mapping[str, Any],
    copy: Mapping[str, Any],
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_kind, specific_fields = _EXTERNAL_EVIDENCE_KINDS[field]
    if not isinstance(raw, Mapping) or set(raw) != (
        _EXTERNAL_EVIDENCE_COMMON_FIELDS | specific_fields
    ):
        raise _InvalidEvidence("external evidence fields are not exact")
    record = dict(raw)
    if (
        record["schema_version"] != EXTERNAL_EVIDENCE_SCHEMA
        or record["evidence_kind"] != evidence_kind
        or not isinstance(record["reference_id"], str)
        or _OPAQUE_ID.fullmatch(record["reference_id"]) is None
        or record["reference_id"] != _content_evidence_reference_id(record)
        or not isinstance(record["issuer_role_id"], str)
        or _ROLE_ID.fullmatch(record["issuer_role_id"]) is None
        or record["verification_result"] != "passed"
        or not (
            record["protected_artifact_id"]
            == verified_artifact["protected_artifact_id"]
            == attestation["protected_artifact_id"]
        )
        or not (
            record["raw_manifest_sha256"]
            == verified_artifact["raw_manifest_sha256"]
            == attestation["raw_manifest_sha256"]
        )
        or not (
            record["raw_manifest_size_bytes"]
            == verified_artifact["raw_manifest_size_bytes"]
            == attestation["raw_manifest_size_bytes"]
        )
        or not (record["copy_id"] == copy["copy_id"] == retrieval["copy_id"])
        or not (
            record["failure_domain_id"]
            == copy["failure_domain_id"]
            == retrieval["failure_domain_id"]
        )
    ):
        raise _InvalidEvidence("external evidence common binding is invalid")
    created_at = _parse_utc(record["created_at_utc"])
    if not (copy["created_at"] <= created_at <= attestation["attested_at"]):
        raise _InvalidEvidence("external evidence chronology is invalid")
    if field == "off_host_storage_evidence":
        valid = (
            record["issuer_role_id"] == attestation["evidence_custodian_role_id"]
            and record["copy_verification_receipt_id"] == copy["receipt_id"]
            and isinstance(record["storage_control_id"], str)
            and _OPAQUE_ID.fullmatch(record["storage_control_id"]) is not None
            and record["off_experiment_host"] is True
        )
    elif field == "encryption_in_transit_evidence":
        valid = (
            record["issuer_role_id"] == attestation["evidence_custodian_role_id"]
            and record["copy_verification_receipt_id"] == copy["receipt_id"]
            and isinstance(record["transport_control_id"], str)
            and _OPAQUE_ID.fullmatch(record["transport_control_id"]) is not None
            and record["transport_security"] == "encrypted-authenticated-channel"
        )
    elif field == "encryption_at_rest_evidence":
        valid = (
            record["issuer_role_id"] == attestation["evidence_custodian_role_id"]
            and record["copy_verification_receipt_id"] == copy["receipt_id"]
            and isinstance(record["encryption_control_id"], str)
            and _OPAQUE_ID.fullmatch(record["encryption_control_id"]) is not None
            and record["encryption_state"] == "encrypted-at-rest"
        )
    elif field == "key_custody_evidence":
        valid = (
            record["issuer_role_id"] == attestation["key_custodian_role_id"]
            and record["key_custodian_role_id"] == attestation["key_custodian_role_id"]
            and isinstance(record["key_control_id"], str)
            and _OPAQUE_ID.fullmatch(record["key_control_id"]) is not None
            and record["custody_state"] == "assigned"
        )
    else:
        valid = (
            record["issuer_role_id"] == attestation["evidence_custodian_role_id"]
            and record["retrieval_receipt_id"] == retrieval["receipt_id"]
            and record["recovery_procedure_id"] == attestation["recovery_procedure_id"]
            and record["procedure_state"] == "verified-by-full-retrieval"
            and created_at >= retrieval["created_at"]
        )
    if not valid:
        raise _InvalidEvidence("external evidence kind binding is invalid")
    return record


def _verify_external_evidence(
    attestation: Mapping[str, Any],
    evidence: Mapping[str, Path],
    *,
    verified_artifact: Mapping[str, Any],
    copies: Sequence[Mapping[str, Any]],
    retrievals: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(evidence, Mapping):
        raise _InvalidEvidence("external evidence index is not a mapping")
    fields = tuple(_EXTERNAL_EVIDENCE_KINDS)
    references = [attestation[field] for field in fields]
    reference_ids = [reference["reference_id"] for reference in references]
    if len(set(reference_ids)) != len(reference_ids) or set(evidence) != set(
        reference_ids
    ):
        raise _InvalidEvidence("external evidence index is incomplete or ambiguous")
    copy = next(
        (
            item
            for item in copies
            if item["receipt_id"] == attestation["copy_verification_receipt_id"]
        ),
        None,
    )
    retrieval = next(
        (
            item
            for item in retrievals
            if item["receipt_id"] == attestation["retrieval_receipt_id"]
        ),
        None,
    )
    if copy is None or retrieval is None:
        raise _InvalidEvidence("external evidence source binding is absent")
    file_identities: set[tuple[int, int]] = set()
    validated: dict[str, dict[str, Any]] = {}
    for field, reference in zip(fields, references, strict=True):
        payload, observed_digest, file_identity = _regular_file_bytes(
            evidence[reference["reference_id"]],
            max_bytes=_MAX_EXTERNAL_EVIDENCE_BYTES,
        )
        if observed_digest != reference["sha256"]:
            raise _InvalidEvidence("external evidence digest does not match")
        if file_identity in file_identities:
            raise _InvalidEvidence("external evidence files are not distinct")
        file_identities.add(file_identity)
        try:
            raw = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _InvalidEvidence("external evidence is not JSON") from error
        if canonical_json_bytes(raw) != payload:
            raise _InvalidEvidence("external evidence is not canonical JSON")
        record = _validate_typed_external_evidence(
            raw,
            field=field,
            attestation=attestation,
            verified_artifact=verified_artifact,
            copy=copy,
            retrieval=retrieval,
        )
        if record["reference_id"] != reference["reference_id"]:
            raise _InvalidEvidence("external evidence reference ID does not match")
        validated[field] = record
    return validated


def _sealed_binding(
    verification: Mapping[str, Any], *, include_retention: bool = False
) -> dict[str, Any]:
    binding = {
        "protected_artifact_id": verification["protected_artifact_id"],
        "raw_manifest_sha256": verification["raw_manifest_sha256"],
        "raw_manifest_size_bytes": verification["raw_manifest_size_bytes"],
    }
    if include_retention:
        binding["provisional_retain_not_before_utc"] = verification["manifest"][
            "provisional_retain_not_before_utc"
        ]
    return binding


def _publication_native_outcome_passed(
    artifact: Path, verification: Mapping[str, Any]
) -> bool:
    """Require the retained managed-sequence outcome itself to be a pass."""

    try:
        entries = verification["manifest"]["files"]
        summaries = [
            entry for entry in entries if entry.get("role") == "sequence-summary"
        ]
        if len(summaries) != 1:
            return False
        entry = summaries[0]
        relative = validate_safe_relative_path(entry["relative_path"])
        payload, digest, _identity = _regular_file_bytes(
            artifact.joinpath(*relative.split("/"))
        )
        if len(payload) != entry["size_bytes"] or digest != entry["sha256"]:
            return False
        summary = json.loads(payload)
        if not isinstance(summary, dict) or canonical_json_bytes(summary) != payload:
            return False
        started = summary.get("started_actions")
        return bool(
            summary.get("record_kind") == "aptus-cuda-campaign-managed-sequence-v1"
            and summary.get("native_outcome") == "passed"
            and summary.get("reason_code") == "NONE"
            and summary.get("evidence_status") == "protocol-valid"
            and summary.get("capture_reason_code") == "NONE"
            and summary.get("stopped_early") is False
            and isinstance(started, list)
            and started
            and all(
                isinstance(row, dict)
                and row.get("native_outcome") == "passed"
                and row.get("reason_code") == "NONE"
                and row.get("capture_reason_code") == "NONE"
                and row.get("terminal") is True
                for row in started
            )
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _regular_file_bytes_at(
    directory_descriptor: int,
    name: str,
    *,
    label: str,
    max_bytes: int = _MAX_PRIVATE_JSON_BYTES,
) -> tuple[bytes, str, tuple[int, int]]:
    """Read one child relative to an already-pinned private directory."""

    if not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise _InvalidEvidence(f"{label} has an unsafe child name")
    try:
        before = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise _InvalidEvidence(f"{label} child is unavailable") from error
    if not _private_regular_file(before):
        raise _InvalidEvidence(f"{label} child is not a regular private file")
    before_fingerprint = _metadata_fingerprint(before)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise _InvalidEvidence(f"{label} child cannot be opened safely") from error
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        if (
            not _private_regular_file(opened)
            or _metadata_fingerprint(opened) != before_fingerprint
        ):
            raise _InvalidEvidence(f"{label} child changed while opening")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            if len(payload) + len(block) > max_bytes:
                raise _InvalidEvidence(f"{label} child exceeds its size limit")
            payload.extend(block)
            digest.update(block)
        finished = os.fstat(descriptor)
        if (
            not _private_regular_file(finished)
            or _metadata_fingerprint(finished) != before_fingerprint
        ):
            raise _InvalidEvidence(f"{label} child changed while hashing")
    finally:
        os.close(descriptor)
    try:
        after = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise _InvalidEvidence(f"{label} child changed after hashing") from error
    if (
        not _private_regular_file(after)
        or _metadata_fingerprint(after) != before_fingerprint
    ):
        raise _InvalidEvidence(f"{label} child changed while hashing")
    return bytes(payload), digest.hexdigest(), (opened.st_dev, opened.st_ino)


def _private_directory_snapshot(
    path: object, *, label: str
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    if not isinstance(path, Path):
        raise _InvalidEvidence(f"{label} is not a Path")
    try:
        parent_metadata = path.parent.lstat()
        metadata = path.lstat()
    except OSError as error:
        raise _InvalidEvidence(f"{label} is unavailable") from error
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise _InvalidEvidence(f"{label} parent is not a directory")
    parent_fingerprint = _metadata_fingerprint(parent_metadata)
    if not _private_directory_metadata(metadata):
        raise _InvalidEvidence(f"{label} is not a private directory")
    directory_fingerprint = _metadata_fingerprint(metadata)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_watch = _linux_directory_mutation_watch(path.parent)
    try:
        directory_descriptor = os.open(path, flags)
    except OSError as error:
        if parent_watch is not None:
            os.close(parent_watch)
        raise _InvalidEvidence(f"{label} cannot be opened safely") from error
    parent_mutated = False
    try:
        opened_directory = os.fstat(directory_descriptor)
        if (
            not _private_directory_metadata(opened_directory)
            or _metadata_fingerprint(opened_directory) != directory_fingerprint
        ):
            raise _InvalidEvidence(f"{label} changed while opening")
        try:
            initial_names = sorted(os.listdir(directory_descriptor))
        except OSError as error:
            raise _InvalidEvidence(f"{label} cannot be inventoried") from error
        if not initial_names:
            raise _InvalidEvidence(f"{label} is empty")
        if len(initial_names) != len(set(initial_names)):
            raise _InvalidEvidence(f"{label} inventory is ambiguous")
        bindings: list[dict[str, Any]] = []
        payloads: dict[str, bytes] = {}
        for name in initial_names:
            payload, digest, _identity = _regular_file_bytes_at(
                directory_descriptor,
                name,
                label=label,
            )
            payloads[name] = payload
            bindings.append(
                {
                    "relative_path": name,
                    "size_bytes": len(payload),
                    "sha256": digest,
                }
            )
        try:
            final_names = sorted(os.listdir(directory_descriptor))
        except OSError as error:
            raise _InvalidEvidence(f"{label} cannot be reinventoried") from error
        if final_names != initial_names:
            raise _InvalidEvidence(f"{label} inventory changed while hashing")
        finished_directory = os.fstat(directory_descriptor)
        if (
            not _private_directory_metadata(finished_directory)
            or _metadata_fingerprint(finished_directory) != directory_fingerprint
        ):
            raise _InvalidEvidence(f"{label} changed while hashing")
        parent_mutated = _linux_watch_saw_mutation(parent_watch)
    finally:
        os.close(directory_descriptor)
        if parent_watch is not None:
            os.close(parent_watch)
    try:
        after_parent = path.parent.lstat()
        after = path.lstat()
    except OSError as error:
        raise _InvalidEvidence(f"{label} changed after hashing") from error
    if (
        parent_mutated
        or not stat.S_ISDIR(after_parent.st_mode)
        or _metadata_fingerprint(after_parent) != parent_fingerprint
    ):
        raise _InvalidEvidence(f"{label} changed while hashing")
    if (
        not _private_directory_metadata(after)
        or _metadata_fingerprint(after) != directory_fingerprint
    ):
        raise _InvalidEvidence(f"{label} path changed while hashing")
    return bindings, payloads


def _private_directory_bindings(path: object, *, label: str) -> list[dict[str, Any]]:
    bindings, _payloads = _private_directory_snapshot(path, label=label)
    return bindings


def _load_canonical_record_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _InvalidEvidence(f"{label} is unreadable") from error
    if canonical_json_bytes(raw) != payload or not isinstance(raw, dict):
        raise _InvalidEvidence(f"{label} is not canonical")
    return raw


def _live_receipt_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Project one typed live receipt exactly as the sanitizer does."""

    details = receipt.get("details")
    if not isinstance(details, Mapping):
        raise _InvalidEvidence("live receipt details are invalid")
    kind = receipt.get("kind")
    if kind == "copy-verification":
        copy_id = details.get("copy_id")
        failure_domain_id = details.get("failure_domain_id")
    elif kind == "retrieval":
        copy_id = details.get("source_copy_id")
        failure_domain_id = details.get("source_failure_domain_id")
    elif kind == "retention":
        copy_id = None
        failure_domain_id = None
    else:
        raise _InvalidEvidence("finalized receipt prefix has an invalid kind")
    payload = canonical_json_bytes(receipt)
    return {
        "receipt_id": receipt.get("receipt_id"),
        "kind": kind,
        "created_at_utc": receipt.get("created_at_utc"),
        "protected_artifact_id": receipt.get("protected_artifact_id"),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "result": receipt.get("result"),
        "copy_id": copy_id,
        "failure_domain_id": failure_domain_id,
    }


def _bind_finalized_receipt_prefix(
    finalized_payloads: Mapping[str, bytes],
    live_chain: Sequence[Mapping[str, Any]],
) -> None:
    """Bind finalized sanitizer provenance to an exact live-chain prefix."""

    try:
        payload = finalized_payloads["recovery-supplement.json"]
    except KeyError as error:
        raise _InvalidEvidence("finalized recovery supplement is absent") from error
    raw = _load_canonical_record_bytes(payload, label="finalized recovery supplement")
    try:
        copy_receipts = raw["copy_verification_receipts"]
        retrieval_receipt = raw["retrieval_receipt"]
        retention_receipt = raw["retention_receipt"]
    except KeyError as error:
        raise _InvalidEvidence(
            "finalized recovery supplement lacks its receipt projection"
        ) from error
    if (
        raw.get("schema_version") != RECOVERY_SUPPLEMENT_SCHEMA
        or not isinstance(copy_receipts, list)
        or len(copy_receipts) < 2
        or not all(isinstance(item, dict) for item in copy_receipts)
        or not isinstance(retrieval_receipt, dict)
        or not isinstance(retention_receipt, dict)
    ):
        raise _InvalidEvidence("finalized receipt projection is invalid")
    finalized_projection = [
        *copy_receipts,
        retrieval_receipt,
        retention_receipt,
    ]
    if len(live_chain) < len(finalized_projection):
        raise _InvalidEvidence("live receipt chain omits finalized provenance")
    live_projection = [
        _live_receipt_projection(receipt)
        for receipt in live_chain[: len(finalized_projection)]
    ]
    if finalized_projection != live_projection:
        raise _InvalidEvidence(
            "live receipt chain is not the finalized provenance prefix"
        )


def _candidate_content_id(record: Mapping[str, Any]) -> str:
    identity = {key: value for key, value in record.items() if key != "candidate_id"}
    return "candidate_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]


def _candidate_artifact_metadata(record: Mapping[str, Any]) -> tuple[str, str]:
    seed = canonical_json_bytes(record)
    return (
        "artifact_" + sha256_bytes(b"publication-candidate-artifact\0" + seed)[:32],
        "entry_" + sha256_bytes(b"publication-candidate-entry\0" + seed)[:32],
    )


def _publication_candidate_record(
    *,
    campaign_id: str,
    claim_key: str,
    candidate_producer_role_id: str,
    created_at: datetime,
    artifact: Path,
    receipts: Sequence[Mapping[str, Any]],
    external_recovery_attestation: Mapping[str, Any],
    external_evidence: Mapping[str, Path],
    sanitizer: FinalizedSanitizerBinding,
    verified_review: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(campaign_id, str)
        or _CAMPAIGN_ID.fullmatch(campaign_id) is None
        or not isinstance(claim_key, str)
        or _CLAIM_KEY.fullmatch(claim_key) is None
        or not isinstance(candidate_producer_role_id, str)
        or _ROLE_ID.fullmatch(candidate_producer_role_id) is None
    ):
        raise _InvalidEvidence("publication candidate identity is invalid")
    primary = verify_sealed_artifact(artifact)
    recovery = verify_sealed_artifact(sanitizer.recovery_artifact)
    control = verify_sealed_artifact(sanitizer.control_artifact)
    sealed_review = verify_sealed_artifact(sanitizer.review_artifact)
    primary_binding = _sealed_binding(primary, include_retention=True)
    recovery_binding = _sealed_binding(recovery)
    if {
        key: primary_binding[key]
        for key in (
            "protected_artifact_id",
            "raw_manifest_sha256",
            "raw_manifest_size_bytes",
        )
    } != recovery_binding:
        raise _InvalidEvidence(
            "publication artifact is not the sanitizer recovery artifact"
        )

    chain = _validate_receipt_chain(receipts, now=created_at)
    if not all(_receipt_binds_artifact(receipt, primary) for receipt in chain):
        raise _InvalidEvidence("candidate receipt chain binds another artifact")
    copies = _latest_copy_bindings(chain, now=created_at)
    retrievals = _current_retrievals(
        chain,
        now=created_at,
        valid_copies=copies,
        verified_artifact=primary,
    )
    attestation = _validate_external_attestation(
        external_recovery_attestation, now=created_at
    )
    if not _attestation_is_bound(
        attestation,
        verified_artifact=primary,
        copies=copies,
        retrievals=retrievals,
    ):
        raise _InvalidEvidence("candidate external attestation is unbound")
    typed_evidence = _verify_external_evidence(
        attestation,
        external_evidence,
        verified_artifact=primary,
        copies=copies,
        retrievals=retrievals,
    )

    review = verify_finalized_projection(
        sanitizer.projection_stage,
        sanitizer.finalized_candidate_output,
        sanitizer.review_artifact,
        recovery_artifact=sanitizer.recovery_artifact,
        control_artifact=sanitizer.control_artifact,
        producer_role_id=sanitizer.producer_role_id,
        reviewer_role_id=sanitizer.reviewer_role_id,
        finalizer_role_id=sanitizer.finalizer_role_id,
    )
    if verified_review is not None and dict(verified_review) != review:
        raise _InvalidEvidence("candidate sanitizer review changed during evaluation")
    if (
        review.get("result") != "passed"
        or review.get("reason_code") != "NONE"
        or review.get("producer_role_id") != sanitizer.producer_role_id
        or review.get("reviewer_role_id") != sanitizer.reviewer_role_id
        or review.get("finalizer_role_id") != sanitizer.finalizer_role_id
        or sanitizer.producer_role_id == sanitizer.reviewer_role_id
        or sanitizer.finalizer_role_id
        in {sanitizer.producer_role_id, sanitizer.reviewer_role_id}
    ):
        raise _InvalidEvidence("candidate sanitizer review is not independent")
    reviewed_at = _parse_utc(review.get("reviewed_at_utc"))
    if reviewed_at > created_at:
        raise _InvalidEvidence("publication candidate predates its review")
    finalized_at = _parse_utc(review.get("finalized_at_utc"))
    if finalized_at < reviewed_at or finalized_at > created_at:
        raise _InvalidEvidence("publication candidate predates its finalization")

    stage_files, _stage_payloads = _private_directory_snapshot(
        sanitizer.projection_stage, label="projection stage"
    )
    finalized_files, finalized_payloads = _private_directory_snapshot(
        sanitizer.finalized_candidate_output, label="finalized publication candidate"
    )
    _bind_finalized_receipt_prefix(finalized_payloads, chain)
    try:
        claim_boundary_payload = finalized_payloads["claim-boundary.json"]
    except KeyError as error:
        raise _InvalidEvidence("public claim boundary is absent") from error
    claim_boundary = _load_canonical_record_bytes(
        claim_boundary_payload,
        label="public claim boundary",
    )
    if (
        claim_boundary.get("campaign_id") != campaign_id
        or claim_boundary.get("claim_key") != claim_key
    ):
        raise _InvalidEvidence("candidate claim boundary identity does not match")

    reverified_review = verify_finalized_projection(
        sanitizer.projection_stage,
        sanitizer.finalized_candidate_output,
        sanitizer.review_artifact,
        recovery_artifact=sanitizer.recovery_artifact,
        control_artifact=sanitizer.control_artifact,
        producer_role_id=sanitizer.producer_role_id,
        reviewer_role_id=sanitizer.reviewer_role_id,
        finalizer_role_id=sanitizer.finalizer_role_id,
    )
    reverified_stage_files = _private_directory_bindings(
        sanitizer.projection_stage, label="projection stage"
    )
    reverified_finalized_files = _private_directory_bindings(
        sanitizer.finalized_candidate_output, label="finalized publication candidate"
    )
    if (
        reverified_review != review
        or reverified_stage_files != stage_files
        or reverified_finalized_files != finalized_files
    ):
        raise _InvalidEvidence("sanitizer provenance changed while binding candidate")

    record_without_id = {
        "schema_version": PUBLICATION_CANDIDATE_SCHEMA,
        "campaign_id": campaign_id,
        "claim_key": claim_key,
        "candidate_producer_role_id": candidate_producer_role_id,
        "created_at_utc": _format_utc(created_at),
        "primary_artifact": primary_binding,
        "receipt_chain": {
            "ordered_receipt_ids": [receipt["receipt_id"] for receipt in chain],
            "head_receipt_id": chain[-1]["receipt_id"],
            "canonical_sha256": sha256_bytes(compact_canonical_json_bytes(chain)),
        },
        "external_recovery_attestation": {
            "attestation_id": attestation["attestation_id"],
            "canonical_sha256": sha256_bytes(
                compact_canonical_json_bytes(external_recovery_attestation)
            ),
        },
        "external_evidence": [
            {
                "attestation_field": field,
                "evidence_kind": typed_evidence[field]["evidence_kind"],
                "reference_id": external_recovery_attestation[field]["reference_id"],
                "sha256": external_recovery_attestation[field]["sha256"],
            }
            for field in _EXTERNAL_EVIDENCE_KINDS
        ],
        "sanitizer": {
            "producer_role_id": sanitizer.producer_role_id,
            "reviewer_role_id": sanitizer.reviewer_role_id,
            "finalizer_role_id": sanitizer.finalizer_role_id,
            "review_id": review["review_id"],
            "reviewed_at_utc": review["reviewed_at_utc"],
            "finalization_id": review["finalization_id"],
            "finalized_at_utc": review["finalized_at_utc"],
            "recovery_artifact": recovery_binding,
            "control_artifact": _sealed_binding(control),
            "review_artifact": _sealed_binding(sealed_review),
            "projection_stage_files": stage_files,
            "finalized_candidate_files": finalized_files,
        },
    }
    record = {
        **record_without_id,
        "candidate_id": _candidate_content_id(record_without_id),
    }
    return record, primary


def seal_publication_candidate(
    output: Path,
    *,
    campaign_id: str,
    claim_key: str,
    candidate_producer_role_id: str,
    created_at_utc: str | datetime,
    artifact: Path,
    receipts: Sequence[Mapping[str, Any]],
    external_recovery_attestation: Mapping[str, Any],
    external_evidence: Mapping[str, Path],
    sanitizer: FinalizedSanitizerBinding,
) -> dict[str, Any]:
    """Seal an exact, nonpublished candidate spanning every publication input."""

    created_at = _parse_utc(created_at_utc)
    record, primary = _publication_candidate_record(
        campaign_id=campaign_id,
        claim_key=claim_key,
        candidate_producer_role_id=candidate_producer_role_id,
        created_at=created_at,
        artifact=artifact,
        receipts=receipts,
        external_recovery_attestation=external_recovery_attestation,
        external_evidence=external_evidence,
        sanitizer=sanitizer,
    )
    candidate_payload = canonical_json_bytes(record)
    if len(candidate_payload) > _MAX_PRIVATE_JSON_BYTES:
        raise _InvalidEvidence("publication candidate exceeds its size limit")
    candidate_artifact_id, entry_id = _candidate_artifact_metadata(record)
    writer = RawArtifactWriter(
        output,
        protected_artifact_id=candidate_artifact_id,
        record_kind="legacy-recovery",
        identity_bindings={
            "purpose": "nonpublished-publication-candidate",
            "candidate_id": record["candidate_id"],
            "campaign_id": campaign_id,
            "claim_key": claim_key,
            "candidate_producer_role_id": candidate_producer_role_id,
        },
        capture_tool={"name": "aptus-cuda-campaign-eligibility", "version": "v1"},
        source_bindings={
            "primary_raw_manifest_sha256": record["primary_artifact"][
                "raw_manifest_sha256"
            ],
            "recovery_raw_manifest_sha256": record["sanitizer"]["recovery_artifact"][
                "raw_manifest_sha256"
            ],
            "control_raw_manifest_sha256": record["sanitizer"]["control_artifact"][
                "raw_manifest_sha256"
            ],
            "review_raw_manifest_sha256": record["sanitizer"]["review_artifact"][
                "raw_manifest_sha256"
            ],
            "receipt_chain_sha256": record["receipt_chain"]["canonical_sha256"],
            "external_attestation_sha256": record["external_recovery_attestation"][
                "canonical_sha256"
            ],
        },
        provisional_retain_not_before_utc=primary["manifest"][
            "provisional_retain_not_before_utc"
        ],
        required_role_bindings={"publication-candidate": entry_id},
    )
    writer.write_payload(
        candidate_payload,
        "publication-candidate.json",
        role="publication-candidate",
        media_type="application/json",
        entry_id=entry_id,
        captured_at_utc=_format_utc(created_at),
    )
    return {
        "publication_candidate": record,
        "sealed_candidate_artifact": writer.seal(),
    }


def _load_publication_candidate(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sealed = verify_sealed_artifact(path)
    manifest = sealed["manifest"]
    entries = manifest["files"]
    if (
        manifest["record_kind"] != "legacy-recovery"
        or len(entries) != 1
        or entries[0].get("role") != "publication-candidate"
        or entries[0].get("relative_path") != "publication-candidate.json"
        or entries[0].get("media_type") != "application/json"
    ):
        raise _InvalidEvidence("sealed publication candidate inventory is invalid")
    payload_path = path / "publication-candidate.json"
    payload, digest, _identity = _regular_file_bytes(payload_path)
    entry = entries[0]
    if len(payload) != entry["size_bytes"] or digest != entry["sha256"]:
        raise _InvalidEvidence("publication candidate payload changed")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _InvalidEvidence("publication candidate is unreadable") from error
    if (
        canonical_json_bytes(raw) != payload
        or not isinstance(raw, dict)
        or set(raw) != _PUBLICATION_CANDIDATE_FIELDS
        or raw.get("schema_version") != PUBLICATION_CANDIDATE_SCHEMA
        or not isinstance(raw.get("candidate_id"), str)
        or _CANDIDATE_ID.fullmatch(raw["candidate_id"]) is None
        or raw["candidate_id"] != _candidate_content_id(raw)
    ):
        raise _InvalidEvidence("publication candidate record is invalid")
    candidate_artifact_id, entry_id = _candidate_artifact_metadata(raw)
    expected_identity = {
        "purpose": "nonpublished-publication-candidate",
        "candidate_id": raw["candidate_id"],
        "campaign_id": raw["campaign_id"],
        "claim_key": raw["claim_key"],
        "candidate_producer_role_id": raw["candidate_producer_role_id"],
    }
    expected_sources = {
        "primary_raw_manifest_sha256": raw["primary_artifact"]["raw_manifest_sha256"],
        "recovery_raw_manifest_sha256": raw["sanitizer"]["recovery_artifact"][
            "raw_manifest_sha256"
        ],
        "control_raw_manifest_sha256": raw["sanitizer"]["control_artifact"][
            "raw_manifest_sha256"
        ],
        "review_raw_manifest_sha256": raw["sanitizer"]["review_artifact"][
            "raw_manifest_sha256"
        ],
        "receipt_chain_sha256": raw["receipt_chain"]["canonical_sha256"],
        "external_attestation_sha256": raw["external_recovery_attestation"][
            "canonical_sha256"
        ],
    }
    if (
        sealed["protected_artifact_id"] != candidate_artifact_id
        or entry["entry_id"] != entry_id
        or entry["captured_at_utc"] != raw["created_at_utc"]
        or manifest["identity_bindings"] != expected_identity
        or manifest["source_bindings"] != expected_sources
        or manifest["capture_tool"]
        != {"name": "aptus-cuda-campaign-eligibility", "version": "v1"}
        or manifest["required_role_bindings"] != {"publication-candidate": entry_id}
        or manifest["provisional_retain_not_before_utc"]
        != raw["primary_artifact"]["provisional_retain_not_before_utc"]
    ):
        raise _InvalidEvidence("publication candidate manifest metadata is invalid")
    return raw, sealed


def _verify_publication_candidate(
    binding: PublicationCandidateBinding,
    *,
    artifact: Path,
    receipts: Sequence[Mapping[str, Any]],
    external_recovery_attestation: Mapping[str, Any],
    external_evidence: Mapping[str, Path],
    sanitizer: FinalizedSanitizerBinding,
    now: datetime,
    verified_review: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(binding, PublicationCandidateBinding):
        raise _InvalidEvidence("publication candidate binding is invalid")
    record, _sealed = _load_publication_candidate(binding.artifact)
    created_at = _parse_utc(record["created_at_utc"])
    if created_at > now:
        raise _InvalidEvidence("publication candidate is future-dated")
    expected, _primary = _publication_candidate_record(
        campaign_id=binding.campaign_id,
        claim_key=binding.claim_key,
        candidate_producer_role_id=binding.candidate_producer_role_id,
        created_at=created_at,
        artifact=artifact,
        receipts=receipts,
        external_recovery_attestation=external_recovery_attestation,
        external_evidence=external_evidence,
        sanitizer=sanitizer,
        verified_review=verified_review,
    )
    if record != expected:
        raise _InvalidEvidence("publication candidate does not bind exact live inputs")
    return record


def verify_publication_candidate(
    binding: PublicationCandidateBinding,
    *,
    artifact: Path,
    receipts: Sequence[Mapping[str, Any]],
    external_recovery_attestation: Mapping[str, Any],
    external_evidence: Mapping[str, Path],
    sanitizer: FinalizedSanitizerBinding,
    now_utc: str | datetime,
) -> dict[str, Any]:
    """Independently verify a sealed candidate and all exact cross-bindings."""

    return _verify_publication_candidate(
        binding,
        artifact=artifact,
        receipts=receipts,
        external_recovery_attestation=external_recovery_attestation,
        external_evidence=external_evidence,
        sanitizer=sanitizer,
        now=_parse_utc(now_utc),
        verified_review=None,
    )


def _ordered_reasons(reasons: set[str]) -> tuple[str, ...]:
    unknown = reasons - set(PUBLICATION_INELIGIBILITY_REASON_CODES)
    if unknown:  # pragma: no cover - programmer error, not evidence input.
        raise AssertionError(f"Unknown publication reason codes: {sorted(unknown)!r}")
    return tuple(
        code for code in PUBLICATION_INELIGIBILITY_REASON_CODES if code in reasons
    )


def evaluate_publication_eligibility(
    *,
    artifact: Path,
    expected_protected_artifact_id: str,
    expected_raw_manifest_sha256: str,
    expected_raw_manifest_size_bytes: int,
    receipts: Sequence[Mapping[str, Any]],
    external_recovery_attestation: Mapping[str, Any],
    external_evidence: Mapping[str, Path],
    now_utc: str | datetime,
    sanitizer: FinalizedSanitizerBinding,
    publication_candidate: PublicationCandidateBinding,
) -> PublicationEligibilityResult:
    """Reverify every publication gate without mutating external state.

    Malformed or missing evidence is represented by stable ineligibility reason
    codes rather than optimistic defaults or caller-supplied pass booleans.
    """

    reasons: set[str] = set()
    try:
        now = _parse_utc(now_utc)
    except _InvalidEvidence:
        now = None
        reasons.add("INPUT_INVALID")
    if (
        not isinstance(expected_protected_artifact_id, str)
        or _ARTIFACT_ID.fullmatch(expected_protected_artifact_id) is None
        or not isinstance(expected_raw_manifest_sha256, str)
        or _SHA256.fullmatch(expected_raw_manifest_sha256) is None
        or not isinstance(expected_raw_manifest_size_bytes, int)
        or isinstance(expected_raw_manifest_size_bytes, bool)
        or expected_raw_manifest_size_bytes < 1
    ):
        reasons.add("INPUT_INVALID")

    verified: dict[str, Any] | None = None
    try:
        verified = verify_sealed_artifact(artifact)
    except (AttributeError, OSError, TypeError, ValueError):
        reasons.add("ARTIFACT_VERIFICATION_FAILED")
    if verified is not None:
        if verified["protected_artifact_id"] != expected_protected_artifact_id:
            reasons.add("ARTIFACT_ID_MISMATCH")
        if verified["raw_manifest_sha256"] != expected_raw_manifest_sha256:
            reasons.add("ARTIFACT_MANIFEST_DIGEST_MISMATCH")
        if verified["raw_manifest_size_bytes"] != expected_raw_manifest_size_bytes:
            reasons.add("ARTIFACT_MANIFEST_SIZE_MISMATCH")
        manifest = verified.get("manifest")
        if isinstance(manifest, Mapping) and manifest.get("record_kind") == (
            "experiment-run"
        ):
            identity = manifest.get("identity_bindings")
            capture_identity_qualifying = not (
                not isinstance(identity, Mapping)
                or identity.get("capture_kind") != "managed-sequence"
                or identity.get("evidence_status") != "protocol-valid"
            )
            if not capture_identity_qualifying:
                reasons.add("CAPTURE_KIND_NOT_PUBLICATION_QUALIFYING")
            elif not _publication_native_outcome_passed(artifact, verified):
                reasons.add("NATIVE_OUTCOME_NOT_PASSED")

    chain: list[dict[str, Any]] | None = None
    claim_state = "invalid"
    last_suspension_at: datetime | None = None
    last_restoration_at: datetime | None = None
    if now is not None:
        try:
            chain = _validate_receipt_chain(receipts, now=now)
            (
                claim_state,
                last_suspension_at,
                last_restoration_at,
                claim_reasons,
            ) = _claim_state(chain)
            reasons.update(claim_reasons)
        except (ContractError, KeyError, TypeError, ValueError):
            reasons.add("RECEIPT_CHAIN_INVALID")

    copies: list[dict[str, Any]] = []
    retrievals: list[dict[str, Any]] = []
    if chain is not None and verified is not None and now is not None:
        if not all(_receipt_binds_artifact(item, verified) for item in chain):
            reasons.add("RECEIPT_ARTIFACT_BINDING_MISMATCH")
        else:
            try:
                copies = _latest_copy_bindings(chain, now=now)
            except (KeyError, TypeError, ValueError):
                copies = []
                reasons.add("RECEIPT_CHAIN_INVALID")
                chain = None
            if chain is not None:
                if len({item["copy_id"] for item in copies}) < 2:
                    reasons.add("VERIFIED_COPY_COUNT_INSUFFICIENT")
                if len({item["failure_domain_id"] for item in copies}) < 2:
                    reasons.add("FAILURE_DOMAIN_COUNT_INSUFFICIENT")
                current_copies = [item for item in copies if item["current"]]
                if not _has_two_distinct_copy_bindings(current_copies):
                    reasons.add("COPY_VERIFICATION_NOT_CURRENT")
                try:
                    retrievals = _current_retrievals(
                        chain,
                        now=now,
                        valid_copies=copies,
                        verified_artifact=verified,
                    )
                except (KeyError, TypeError, ValueError):
                    retrievals = []
                    reasons.add("RECEIPT_CHAIN_INVALID")
                    chain = None
            if chain is not None:
                if not retrievals:
                    reasons.add("OFF_HOST_RETRIEVAL_NOT_CURRENT")
                if claim_state == "restored":
                    if (
                        last_suspension_at is None
                        or last_restoration_at is None
                        or not _has_two_distinct_copy_bindings(
                            [
                                item
                                for item in current_copies
                                if last_suspension_at
                                <= item["created_at"]
                                <= last_restoration_at
                            ]
                        )
                        or not any(
                            last_suspension_at
                            <= item["created_at"]
                            <= last_restoration_at
                            for item in retrievals
                        )
                    ):
                        reasons.add("CLAIM_STATE_INVALID")
                try:
                    provisional = _parse_utc(
                        verified["manifest"]["provisional_retain_not_before_utc"]
                    )
                    _retention_id, retention_reasons = _retention_state(
                        chain,
                        now=now,
                        provisional_deadline=provisional,
                    )
                    reasons.update(retention_reasons)
                except (KeyError, TypeError, ValueError):
                    reasons.add("RETENTION_RECEIPT_INVALID")

    attestation: dict[str, Any] | None = None
    if now is not None:
        try:
            attestation = _validate_external_attestation(
                external_recovery_attestation, now=now
            )
        except (KeyError, TypeError, ValueError):
            reasons.add("EXTERNAL_RECOVERY_ATTESTATION_INVALID")
    attestation_bound = bool(
        attestation is not None
        and verified is not None
        and chain is not None
        and _attestation_is_bound(
            attestation,
            verified_artifact=verified,
            copies=copies,
            retrievals=retrievals,
        )
    )
    if (
        attestation is not None
        and verified is not None
        and chain is not None
        and not attestation_bound
    ):
        reasons.add("EXTERNAL_RECOVERY_ATTESTATION_UNBOUND")
    if attestation_bound and verified is not None:
        try:
            _verify_external_evidence(
                attestation,
                external_evidence,
                verified_artifact=verified,
                copies=copies,
                retrievals=retrievals,
            )
        except (KeyError, OSError, TypeError, ValueError):
            reasons.add("EXTERNAL_RECOVERY_EVIDENCE_INVALID")

    candidate_review: Mapping[str, Any] | None = None
    try:
        review = verify_finalized_projection(
            sanitizer.projection_stage,
            sanitizer.finalized_candidate_output,
            sanitizer.review_artifact,
            recovery_artifact=sanitizer.recovery_artifact,
            control_artifact=sanitizer.control_artifact,
            producer_role_id=sanitizer.producer_role_id,
            reviewer_role_id=sanitizer.reviewer_role_id,
            finalizer_role_id=sanitizer.finalizer_role_id,
        )
    except Exception:  # noqa: BLE001 - any verifier failure must fail closed.
        reasons.add("SANITIZER_FINALIZATION_INVALID")
    else:
        if (
            not isinstance(review, Mapping)
            or review.get("result") != "passed"
            or review.get("reason_code") != "NONE"
            or review.get("producer_role_id") != sanitizer.producer_role_id
            or review.get("reviewer_role_id") != sanitizer.reviewer_role_id
            or review.get("finalizer_role_id") != sanitizer.finalizer_role_id
            or sanitizer.producer_role_id == sanitizer.reviewer_role_id
            or sanitizer.finalizer_role_id
            in {sanitizer.producer_role_id, sanitizer.reviewer_role_id}
        ):
            reasons.add("INDEPENDENT_REVIEW_NOT_PASSED")
        else:
            try:
                reviewed_at = _parse_utc(review["reviewed_at_utc"])
                finalized_at = _parse_utc(review["finalized_at_utc"])
            except (KeyError, TypeError, ValueError):
                reasons.add("INDEPENDENT_REVIEW_NOT_PASSED")
            else:
                if now is None or reviewed_at > now:
                    reasons.add("INDEPENDENT_REVIEW_NOT_PASSED")
                if now is None or not (reviewed_at <= finalized_at <= now):
                    reasons.add("SANITIZER_FINALIZATION_INVALID")
                if (
                    now is not None
                    and reviewed_at <= now
                    and reviewed_at <= finalized_at <= now
                ):
                    candidate_review = review
                if (
                    claim_state == "restored"
                    and last_restoration_at is not None
                    and reviewed_at < last_restoration_at
                ):
                    reasons.add("CLAIM_STATE_INVALID")

    if candidate_review is not None and now is not None:
        try:
            _verify_publication_candidate(
                publication_candidate,
                artifact=artifact,
                receipts=receipts,
                external_recovery_attestation=external_recovery_attestation,
                external_evidence=external_evidence,
                sanitizer=sanitizer,
                now=now,
                verified_review=candidate_review,
            )
        except Exception:  # noqa: BLE001 - candidate failures must fail closed.
            reasons.add("PUBLICATION_CANDIDATE_INVALID")

    ordered = _ordered_reasons(reasons)
    return PublicationEligibilityResult(
        eligible=not ordered,
        reason_codes=ordered,
        evaluated_at_utc=_format_utc(now) if now is not None else None,
        protected_artifact_id=(
            verified["protected_artifact_id"] if verified is not None else None
        ),
        raw_manifest_sha256=(
            verified["raw_manifest_sha256"] if verified is not None else None
        ),
    )
