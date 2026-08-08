"""Strict canonical contracts for the bounded CUDA evidence campaign.

The frozen Phase 1 companion names sixteen record schemas but intentionally
does not provide JSON Schema documents for them.  Phase 2 therefore makes the
following conservative shape decisions explicit:

* every top-level field listed in :data:`SCHEMA_FIELDS` is required;
* fields that may be absent semantically are present with ``null`` instead;
* unknown top-level fields and implicit type coercion are always rejected;
* nested policy/binding objects remain opaque JSON objects unless Phase 1
  freezes their inner shape, while all nested values are still checked for
  JSON safety and finite numbers;
* telemetry owns fixed common envelope fields here, and the monitoring module
  owns the deeper GPU, host, collector, and watchdog channel shapes;
* capture-failure file inventories reuse the raw-manifest entry contract; and
* no aggregate or setup-record schema is invented because Phase 1 did not name
  one.  Such records need a reviewed protocol amendment or must be nested in a
  named record by their owning implementation.

These choices are deliberately fail closed.  This module uses only the Python
standard library so that protected evidence can be validated without importing
or installing Aptus.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_SCHEMA_VERSION = "aptus.cuda-campaign-protocol.v1"
PROCEDURAL_ROLE_ID_RE = re.compile(r"^(?=.{3,64}$)[a-z][a-z0-9-]*(?:_[a-z0-9-]+)*$")

SCHEMA_VERSIONS = {
    "attempt_slot": "aptus.experiment-attempt-slot.v1",
    "campaign": "aptus.experiment-campaign.v1",
    "capture_failure": "aptus.experiment-capture-failure.v1",
    "claim_boundary": "aptus.experiment-claim-boundary.v1",
    "comparison_cell": "aptus.experiment-comparison-cell.v1",
    "comparison_cohort": "aptus.experiment-comparison-cohort.v1",
    "event_ledger_row": "aptus.experiment-event.v1",
    "execution_configuration": "aptus.experiment-execution-configuration.v1",
    "experiment_run": "aptus.experiment-run.v1",
    "independent_review": "aptus.experiment-publication-review.v1",
    "raw_manifest": "aptus.experiment-raw-manifest.v1",
    "raw_seal": "aptus.experiment-raw-seal.v1",
    "receipt": "aptus.experiment-evidence-receipt.v1",
    "recovery_supplement": "aptus.experiment-recovery-supplement.v1",
    "sanitization_map": "aptus.experiment-sanitization-map.v1",
    "telemetry_sample": "aptus.experiment-telemetry-sample.v1",
}

SLOT_STATUSES = ("started", "planned-not-started")
NATIVE_OUTCOMES = (
    "passed",
    "refused",
    "failed",
    "cancelled",
    "timed-out",
    "guard-blocked",
    "unknown",
)
EVIDENCE_STATUSES = ("protocol-valid", "capture-invalid", "not-started")
OBSERVATION_KINDS = ("emitted", "observed", "derived")
EVENT_TYPES = (
    "clock.mapping",
    "harness.started",
    "telemetry.started",
    "telemetry.stopped",
    "telemetry.failed",
    "command.started",
    "command.finished",
    "job.state-observed",
    "pilot.phase-started",
    "pilot.phase-finished",
    "training.started",
    "training.finished",
    "export.started",
    "export.finished",
    "verification.started",
    "verification.finished",
    "safety.triggered",
    "cancellation.requested",
    "process-group.terminated",
    "lease.reconciled",
    "cooldown.started",
    "cooldown.finished",
    "harness.finished",
    "seal.started",
)
RECEIPT_KINDS = (
    "copy-verification",
    "retrieval",
    "retention",
    "renewal",
    "claim-suspension",
    "claim-restoration",
    "claim-withdrawal",
)
RECOVERY_DISPOSITIONS = (
    "recovered-matching",
    "recovered-mismatched",
    "not-found",
)
TRACEABILITY_TRANSFORMS = (
    "copy",
    "count",
    "digest",
    "opaque-id",
    "aggregate",
    "constant",
)
EVIDENCE_CLASSES = ("declared", "inferred", "estimated", "measured")
REVIEW_CHECKS = (
    "strict-public-schema",
    "complete-raw-to-public-traceability",
    "private-value-absence",
    "numeric-recomputation",
    "claim-boundary-correctness",
    "complete-sorted-unique-sha256sums",
)
REASON_CODES = (
    "NONE",
    "PRIOR_STOP_RULE",
    "CONDITIONING_ATTEMPT_NOT_QUALIFYING",
    "APTUS_ADMISSION_REFUSAL",
    "POLICY_REPLAN_REQUIRED",
    "PROCESS_EXIT_NONZERO",
    "CUDA_OOM",
    "CUDA_XID",
    "CUDA_DEVICE_RESET",
    "CUDA_DEVICE_LOST",
    "HARDWARE_ERROR",
    "NONFINITE_VALUE",
    "ARTIFACT_INTEGRITY_FAILURE",
    "CHECKPOINT_CONTINUATION_FAILURE",
    "EXPORT_VERIFICATION_FAILURE",
    "THERMAL_WARNING_SUSTAINED",
    "THERMAL_STOP_SUSTAINED",
    "THERMAL_STOP_IMMEDIATE",
    "THERMAL_THROTTLE",
    "THERMAL_LIMIT_DISAPPEARED",
    "FREE_VRAM_WARNING",
    "FREE_VRAM_FLOOR",
    "HOST_RAM_WARNING",
    "HOST_RAM_FLOOR",
    "DISK_WARNING",
    "DISK_FLOOR",
    "DISK_BUDGET_INSUFFICIENT",
    "SWAP_RATE_WARNING",
    "SWAP_RATE_LIMIT",
    "UNRELATED_GPU_ACTIVITY",
    "TELEMETRY_QUALIFYING_GAP",
    "TELEMETRY_HARD_GAP",
    "TELEMETRY_COLLECTOR_FAILURE",
    "WATCHDOG_HEARTBEAT_WARNING",
    "WATCHDOG_HEARTBEAT_LOST",
    "OWNERSHIP_UNCERTAIN",
    "CANCELLATION_DEADLINE_EXCEEDED",
    "LEASE_RECONCILIATION_FAILURE",
    "EMERGENCY_DEADLINE_EXCEEDED",
    "STREAM_CAPTURE_FAILURE",
    "MISSING_REQUIRED_EVIDENCE",
    "SEAL_FAILURE",
    "SANITIZATION_FAILURE",
    "COPY_VERIFICATION_FAILURE",
    "RETRIEVAL_FAILURE",
    "UNKNOWN_TERMINAL_STATE",
    "RECOVERED_MATCH",
    "RECOVERED_MISMATCH",
    "NOT_FOUND_AFTER_BOUNDED_SEARCH",
    "ORIGINAL_TRANSCRIPT_NOT_FOUND",
)

RAW_FILE_ENTRY_FIELDS = (
    "entry_id",
    "role",
    "relative_path",
    "media_type",
    "size_bytes",
    "sha256",
    "captured_at_utc",
)
TRACEABILITY_ENTRY_FIELDS = (
    "public_file",
    "public_json_pointer",
    "source_raw_manifest_sha256",
    "source_artifact_entry_id",
    "source_json_pointer",
    "transform",
    "evidence_class",
)
RECOVERY_ITEM_FIELDS = (
    "logical_item_id",
    "source_json_pointer",
    "expected_sha256",
    "disposition",
    "recovered_artifact_entry_id",
    "recovered_sha256",
    "recovered_size_bytes",
    "reason_code",
)
ADDITIONAL_SEARCH_ITEM_FIELDS = (
    "item_id",
    "disposition",
    "reason_code",
    "search_scope_codes",
)


class ContractError(ValueError):
    """Raised when a campaign record violates its frozen contract."""


Check = Callable[[Any, str], None]


@dataclass(frozen=True)
class RecordSchema:
    """One exact top-level record shape."""

    schema_version: str
    fields: Mapping[str, Check]


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?\+00:00$"
)
_JSON_POINTER_RE = re.compile(r"^(?:/(?:[^~/]|~0|~1)*)*$")
_GENERIC_ID_RE = re.compile(r"^[a-z][a-z0-9-]*_[A-Za-z0-9._:-]+$")


def _fail(path: str, message: str) -> None:
    raise ContractError(f"{path}: {message}")


def _validate_json_value(value: Any, path: str = "$") -> None:
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(path, "non-finite numbers are forbidden")
        return
    if type(value) is str:
        if "\x00" in value:
            _fail(path, "NUL characters are forbidden")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ContractError(f"{path}: invalid UTF-8 string") from error
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(path, "object keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    _fail(path, f"unsupported JSON type {type(value).__name__}")


def _string(value: Any, path: str) -> None:
    if type(value) is not str or not value:
        _fail(path, "must be a nonempty string")


def _bounded_string(maximum: int) -> Check:
    def check(value: Any, path: str) -> None:
        _string(value, path)
        if len(value) > maximum:
            _fail(path, f"must contain at most {maximum} Unicode code points")

    return check


def _nullable(check: Check) -> Check:
    def nullable_check(value: Any, path: str) -> None:
        if value is not None:
            check(value, path)

    return nullable_check


def _integer(value: Any, path: str) -> None:
    if type(value) is not int:
        _fail(path, "must be an integer (booleans are not integers)")


def _nonnegative_integer(value: Any, path: str) -> None:
    _integer(value, path)
    if value < 0:
        _fail(path, "must be nonnegative")


def _positive_integer(value: Any, path: str) -> None:
    _integer(value, path)
    if value <= 0:
        _fail(path, "must be positive")


def _positive_number(value: Any, path: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        _fail(path, "must be a positive finite number")


def _boolean(value: Any, path: str) -> None:
    if type(value) is not bool:
        _fail(path, "must be a boolean")


def _object(value: Any, path: str) -> None:
    if type(value) is not dict:
        _fail(path, "must be an object")


def _list(value: Any, path: str) -> None:
    if type(value) is not list:
        _fail(path, "must be an array")


def _string_list(value: Any, path: str) -> None:
    _list(value, path)
    for index, item in enumerate(value):
        _string(item, f"{path}[{index}]")


def _unique_string_list(value: Any, path: str) -> None:
    _string_list(value, path)
    if len(value) != len(set(value)):
        _fail(path, "must not contain duplicate values")


def _enum(values: Sequence[str]) -> Check:
    allowed = frozenset(values)

    def check(value: Any, path: str) -> None:
        if type(value) is not str or value not in allowed:
            _fail(path, f"must be one of {sorted(allowed)!r}")

    return check


def _digest(value: Any, path: str) -> None:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        _fail(path, "must be a lowercase 64-character SHA-256 digest")


def _timestamp(value: Any, path: str) -> None:
    if type(value) is not str or _RFC3339_UTC_RE.fullmatch(value) is None:
        _fail(path, "must be an RFC 3339 UTC timestamp normalized to +00:00")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ContractError(f"{path}: invalid timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        _fail(path, "must use UTC")


def _json_pointer(value: Any, path: str) -> None:
    if type(value) is not str or _JSON_POINTER_RE.fullmatch(value) is None:
        _fail(path, "must be a valid JSON Pointer")


def validate_safe_relative_path(path: str) -> str:
    """Validate and return one normalized relative POSIX path."""

    if type(path) is not str or not path:
        raise ContractError("path: must be a nonempty string")
    if "\x00" in path or "\\" in path:
        raise ContractError("path: NUL and backslash are forbidden")
    if path.startswith("/") or path.endswith("/"):
        raise ContractError("path: must be relative and have no trailing slash")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ContractError("path: empty, dot, and parent segments are forbidden")
    normalized = str(PurePosixPath(path))
    if normalized != path or PurePosixPath(path).is_absolute():
        raise ContractError("path: must be a normalized relative POSIX path")
    return path


def _relative_path(value: Any, path: str) -> None:
    if type(value) is not str:
        _fail(path, "must be a string")
    try:
        validate_safe_relative_path(value)
    except ContractError as error:
        raise ContractError(f"{path}: {error}") from error


def _identifier(prefix: str, suffix_length: int) -> Check:
    pattern = re.compile(rf"^{re.escape(prefix)}[0-9a-f]{{{suffix_length}}}$")

    def check(value: Any, path: str) -> None:
        if type(value) is not str or pattern.fullmatch(value) is None:
            _fail(
                path,
                f"must be {prefix!r} plus {suffix_length} lowercase hex characters",
            )

    return check


def _generic_identifier(value: Any, path: str) -> None:
    if type(value) is not str or _GENERIC_ID_RE.fullmatch(value) is None:
        _fail(path, "must be a typed, nonempty identifier")


def _exact_object(value: Any, fields: Sequence[str], path: str) -> None:
    _object(value, path)
    expected = set(fields)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        _fail(path, f"wrong fields; missing={missing!r}, unknown={unknown!r}")


def _raw_file_entries(value: Any, path: str) -> None:
    _list(value, path)
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    prior_path: str | None = None
    for index, entry in enumerate(value):
        entry_path = f"{path}[{index}]"
        _exact_object(entry, RAW_FILE_ENTRY_FIELDS, entry_path)
        _generic_identifier(entry["entry_id"], f"{entry_path}.entry_id")
        _string(entry["role"], f"{entry_path}.role")
        _relative_path(entry["relative_path"], f"{entry_path}.relative_path")
        _string(entry["media_type"], f"{entry_path}.media_type")
        _nonnegative_integer(entry["size_bytes"], f"{entry_path}.size_bytes")
        _digest(entry["sha256"], f"{entry_path}.sha256")
        _timestamp(entry["captured_at_utc"], f"{entry_path}.captured_at_utc")
        if entry["entry_id"] in seen_ids:
            _fail(entry_path, "duplicate entry_id")
        if entry["relative_path"] in seen_paths:
            _fail(entry_path, "duplicate relative_path")
        if prior_path is not None and entry["relative_path"] <= prior_path:
            _fail(path, "entries must be sorted by unique relative_path")
        seen_ids.add(entry["entry_id"])
        seen_paths.add(entry["relative_path"])
        prior_path = entry["relative_path"]


def _traceability_entries(value: Any, path: str) -> None:
    _list(value, path)
    seen_targets: set[tuple[str, str]] = set()
    for index, entry in enumerate(value):
        entry_path = f"{path}[{index}]"
        _exact_object(entry, TRACEABILITY_ENTRY_FIELDS, entry_path)
        _relative_path(entry["public_file"], f"{entry_path}.public_file")
        _json_pointer(entry["public_json_pointer"], f"{entry_path}.public_json_pointer")
        _digest(
            entry["source_raw_manifest_sha256"],
            f"{entry_path}.source_raw_manifest_sha256",
        )
        _generic_identifier(
            entry["source_artifact_entry_id"],
            f"{entry_path}.source_artifact_entry_id",
        )
        _json_pointer(entry["source_json_pointer"], f"{entry_path}.source_json_pointer")
        _enum(TRACEABILITY_TRANSFORMS)(entry["transform"], f"{entry_path}.transform")
        _enum(EVIDENCE_CLASSES)(entry["evidence_class"], f"{entry_path}.evidence_class")
        target = (entry["public_file"], entry["public_json_pointer"])
        if target in seen_targets:
            _fail(entry_path, "duplicate public file and JSON Pointer target")
        seen_targets.add(target)


def _recovery_items(value: Any, path: str) -> None:
    _list(value, path)
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        _exact_object(item, RECOVERY_ITEM_FIELDS, item_path)
        _string(item["logical_item_id"], f"{item_path}.logical_item_id")
        _json_pointer(item["source_json_pointer"], f"{item_path}.source_json_pointer")
        _digest(item["expected_sha256"], f"{item_path}.expected_sha256")
        _enum(RECOVERY_DISPOSITIONS)(item["disposition"], f"{item_path}.disposition")
        _nullable(_generic_identifier)(
            item["recovered_artifact_entry_id"],
            f"{item_path}.recovered_artifact_entry_id",
        )
        _nullable(_digest)(item["recovered_sha256"], f"{item_path}.recovered_sha256")
        _nullable(_nonnegative_integer)(
            item["recovered_size_bytes"],
            f"{item_path}.recovered_size_bytes",
        )
        _enum(REASON_CODES)(item["reason_code"], f"{item_path}.reason_code")
        item_id = item["logical_item_id"]
        if item_id in seen:
            _fail(item_path, "duplicate logical_item_id")
        seen.add(item_id)
        recovered = item["disposition"] != "not-found"
        recovered_values = (
            item["recovered_artifact_entry_id"],
            item["recovered_sha256"],
            item["recovered_size_bytes"],
        )
        if recovered and any(entry is None for entry in recovered_values):
            _fail(item_path, "recovered dispositions require every recovered field")
        if not recovered and any(entry is not None for entry in recovered_values):
            _fail(item_path, "not-found requires null recovered fields")


def _additional_search_items(value: Any, path: str) -> None:
    _list(value, path)
    if len(value) != 1:
        _fail(path, "must contain exactly the frozen Python transcript search item")
    item = value[0]
    item_path = f"{path}[0]"
    _exact_object(item, ADDITIONAL_SEARCH_ITEM_FIELDS, item_path)
    if item["item_id"] != "python-test-transcript":
        _fail(f"{item_path}.item_id", "must be python-test-transcript")
    if item["disposition"] != "not-found":
        _fail(f"{item_path}.disposition", "must be not-found")
    if item["reason_code"] != "ORIGINAL_TRANSCRIPT_NOT_FOUND":
        _fail(
            f"{item_path}.reason_code",
            "must be ORIGINAL_TRANSCRIPT_NOT_FOUND",
        )
    _unique_string_list(item["search_scope_codes"], f"{item_path}.search_scope_codes")
    if not item["search_scope_codes"]:
        _fail(f"{item_path}.search_scope_codes", "must not be empty")


def _review_checks(value: Any, path: str) -> None:
    _object(value, path)
    if set(value) != set(REVIEW_CHECKS):
        _fail(path, f"must contain exactly the frozen review checks {REVIEW_CHECKS!r}")
    for key, result in value.items():
        _boolean(result, f"{path}.{key}")


_CAMPAIGN_FIELDS = (
    "schema_version",
    "campaign_id",
    "protocol_schema_version",
    "program_key",
    "phase_sequence",
    "host_class",
    "allowed_methods",
    "allowed_placement",
    "allowed_world_size",
)
_COHORT_FIELDS = (
    "schema_version",
    "comparison_cohort_id",
    "campaign_id",
    "question",
    "held_controls",
    "varied_dimensions",
    "member_cell_ids",
    "attempt_counts",
    "seed_schedule",
    "block_schedule",
    "stopping_rule",
    "promotion_rule",
    "no_replacement_rule",
    "aggregate_rule",
)
_CELL_FIELDS = (
    "schema_version",
    "comparison_cell_id",
    "campaign_id",
    "source_binding",
    "host_binding",
    "environment_binding",
    "model_binding",
    "dataset_and_split_binding",
    "method",
    "precision",
    "quantization",
    "placement",
    "world_size",
    "sequence_length",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "training_budget",
    "checkpoint_rule",
    "adapter_targets",
    "seed_policy",
    "cache_policy",
    "cooldown_policy",
    "safety_policy",
    "capture_policy",
    "retention_policy_id",
)
_ATTEMPT_SLOT_FIELDS = (
    "schema_version",
    "attempt_slot_id",
    "comparison_cohort_id",
    "comparison_cell_id",
    "block",
    "ordinal",
    "role",
    "order_position",
    "scheduled_seed",
    "slot_status",
    "execution_configuration_id",
    "experiment_run_id",
    "native_outcome",
    "evidence_status",
    "reason_code",
)
_EXECUTION_CONFIGURATION_FIELDS = (
    "schema_version",
    "execution_configuration_id",
    "comparison_cell_id",
    "exact_behavior_values",
    "split_seed",
    "training_seed",
    "data_order_seed",
    "plan_id",
    "candidate_id",
    "bundle_fingerprint",
    "emergency_deadline_seconds",
)
_EXPERIMENT_RUN_FIELDS = (
    "schema_version",
    "experiment_run_id",
    "attempt_slot_id",
    "execution_configuration_id",
    "exact_argv",
    "working_directory",
    "fresh_state_root",
    "bundle_path",
    "output_path",
    "run_order",
    "observed_host_state",
    "plan_id",
    "candidate_id",
    "bundle_fingerprint",
    "bundle_manifest_sha256",
    "archive_sha256",
    "aptus_job_ids",
    "aptus_run_ids",
    "terminal_evidence",
)
_CLAIM_BOUNDARY_FIELDS = (
    "schema_version",
    "campaign_id",
    "claim_key",
    "exact_scope",
    "allowed_claim_types",
    "forbidden_claims",
    "qualification_dependencies",
    "statement",
)
_EVENT_FIELDS = (
    "schema_version",
    "sequence",
    "experiment_run_id",
    "monotonic_ns",
    "wall_time_utc",
    "event_type",
    "phase",
    "action",
    "subject_kind",
    "subject_id",
    "observation_kind",
    "source_reported_at_utc",
    "exit_code",
    "native_outcome",
    "reason_code",
)
_TELEMETRY_FIELDS = (
    "schema_version",
    "sequence",
    "experiment_run_id",
    "scheduled_slot",
    "scheduled_monotonic_ns",
    "observed_monotonic_ns",
    "wall_time_utc",
    "sample_interval_seconds",
    "gpu",
    "host",
    "collector",
    "watchdog",
)
_RAW_MANIFEST_FIELDS = (
    "schema_version",
    "protected_artifact_id",
    "record_kind",
    "identity_bindings",
    "capture_tool",
    "source_bindings",
    "retention_policy_id",
    "provisional_retain_not_before_utc",
    "files",
    "file_count",
    "total_bytes",
    "required_role_bindings",
    "completion_marker",
)
_RAW_SEAL_FIELDS = (
    "schema_version",
    "protected_artifact_id",
    "raw_manifest_sha256",
    "raw_manifest_size_bytes",
    "sealed_at_utc",
)
_CAPTURE_FAILURE_FIELDS = (
    "schema_version",
    "protected_artifact_id",
    "attempt_slot_id",
    "experiment_run_id",
    "created_at_utc",
    "reason_code",
    "available_files",
    "missing_fields",
    "recoverable_locator",
)
_RECEIPT_FIELDS = (
    "schema_version",
    "receipt_id",
    "kind",
    "created_at_utc",
    "issuer_role_id",
    "protected_artifact_id",
    "raw_manifest_sha256",
    "raw_manifest_size_bytes",
    "previous_receipt_id",
    "result",
    "details",
)
_SANITIZATION_MAP_FIELDS = ("schema_version", "entries")
_RECOVERY_SUPPLEMENT_FIELDS = (
    "schema_version",
    "original_packet",
    "expected_digest_manifest",
    "recovery_raw_manifest",
    "copy_verification_receipts",
    "retrieval_receipt",
    "retention_policy",
    "retention_receipt",
    "sanitization_map",
    "independent_review",
    "claim_boundary",
    "summary_counts",
    "items",
    "additional_search_items",
)
_INDEPENDENT_REVIEW_FIELDS = (
    "schema_version",
    "review_id",
    "producer_role_id",
    "reviewer_role_id",
    "reviewed_at_utc",
    "checks",
    "result",
    "reason_code",
)

SCHEMA_FIELDS = {
    SCHEMA_VERSIONS["campaign"]: _CAMPAIGN_FIELDS,
    SCHEMA_VERSIONS["comparison_cohort"]: _COHORT_FIELDS,
    SCHEMA_VERSIONS["comparison_cell"]: _CELL_FIELDS,
    SCHEMA_VERSIONS["attempt_slot"]: _ATTEMPT_SLOT_FIELDS,
    SCHEMA_VERSIONS["execution_configuration"]: _EXECUTION_CONFIGURATION_FIELDS,
    SCHEMA_VERSIONS["experiment_run"]: _EXPERIMENT_RUN_FIELDS,
    SCHEMA_VERSIONS["claim_boundary"]: _CLAIM_BOUNDARY_FIELDS,
    SCHEMA_VERSIONS["event_ledger_row"]: _EVENT_FIELDS,
    SCHEMA_VERSIONS["telemetry_sample"]: _TELEMETRY_FIELDS,
    SCHEMA_VERSIONS["raw_manifest"]: _RAW_MANIFEST_FIELDS,
    SCHEMA_VERSIONS["raw_seal"]: _RAW_SEAL_FIELDS,
    SCHEMA_VERSIONS["capture_failure"]: _CAPTURE_FAILURE_FIELDS,
    SCHEMA_VERSIONS["receipt"]: _RECEIPT_FIELDS,
    SCHEMA_VERSIONS["sanitization_map"]: _SANITIZATION_MAP_FIELDS,
    SCHEMA_VERSIONS["recovery_supplement"]: _RECOVERY_SUPPLEMENT_FIELDS,
    SCHEMA_VERSIONS["independent_review"]: _INDEPENDENT_REVIEW_FIELDS,
}

_SCHEMAS = {
    SCHEMA_VERSIONS["campaign"]: RecordSchema(
        SCHEMA_VERSIONS["campaign"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["campaign"],)),
            "campaign_id": _identifier("campaign_", 20),
            "protocol_schema_version": _enum((PROTOCOL_SCHEMA_VERSION,)),
            "program_key": _string,
            "phase_sequence": _list,
            "host_class": _string,
            "allowed_methods": _unique_string_list,
            "allowed_placement": _string,
            "allowed_world_size": _positive_integer,
        },
    ),
    SCHEMA_VERSIONS["comparison_cohort"]: RecordSchema(
        SCHEMA_VERSIONS["comparison_cohort"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["comparison_cohort"],)),
            "comparison_cohort_id": _identifier("cohort_", 20),
            "campaign_id": _identifier("campaign_", 20),
            "question": _string,
            "held_controls": _object,
            "varied_dimensions": _unique_string_list,
            "member_cell_ids": _unique_string_list,
            "attempt_counts": _object,
            "seed_schedule": _object,
            "block_schedule": _list,
            "stopping_rule": _object,
            "promotion_rule": _object,
            "no_replacement_rule": _boolean,
            "aggregate_rule": _object,
        },
    ),
    SCHEMA_VERSIONS["comparison_cell"]: RecordSchema(
        SCHEMA_VERSIONS["comparison_cell"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["comparison_cell"],)),
            "comparison_cell_id": _identifier("cell_", 20),
            "campaign_id": _identifier("campaign_", 20),
            "source_binding": _object,
            "host_binding": _object,
            "environment_binding": _object,
            "model_binding": _object,
            "dataset_and_split_binding": _object,
            "method": _string,
            "precision": _string,
            "quantization": _nullable(_string),
            "placement": _string,
            "world_size": _positive_integer,
            "sequence_length": _positive_integer,
            "micro_batch_size": _positive_integer,
            "gradient_accumulation_steps": _positive_integer,
            "effective_batch_size": _positive_integer,
            "training_budget": _object,
            "checkpoint_rule": _object,
            "adapter_targets": _unique_string_list,
            "seed_policy": _object,
            "cache_policy": _object,
            "cooldown_policy": _object,
            "safety_policy": _object,
            "capture_policy": _object,
            "retention_policy_id": _string,
        },
    ),
    SCHEMA_VERSIONS["attempt_slot"]: RecordSchema(
        SCHEMA_VERSIONS["attempt_slot"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["attempt_slot"],)),
            "attempt_slot_id": _identifier("slot_", 20),
            "comparison_cohort_id": _identifier("cohort_", 20),
            "comparison_cell_id": _identifier("cell_", 20),
            "block": _nonnegative_integer,
            "ordinal": _positive_integer,
            "role": _string,
            "order_position": _nonnegative_integer,
            "scheduled_seed": _nonnegative_integer,
            "slot_status": _enum(SLOT_STATUSES),
            "execution_configuration_id": _nullable(_identifier("exec_", 20)),
            "experiment_run_id": _nullable(_identifier("xrun_", 32)),
            "native_outcome": _nullable(_enum(NATIVE_OUTCOMES)),
            "evidence_status": _enum(EVIDENCE_STATUSES),
            "reason_code": _enum(REASON_CODES),
        },
    ),
    SCHEMA_VERSIONS["execution_configuration"]: RecordSchema(
        SCHEMA_VERSIONS["execution_configuration"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["execution_configuration"],)),
            "execution_configuration_id": _identifier("exec_", 20),
            "comparison_cell_id": _identifier("cell_", 20),
            "exact_behavior_values": _object,
            "split_seed": _nonnegative_integer,
            "training_seed": _nonnegative_integer,
            "data_order_seed": _nonnegative_integer,
            "plan_id": _generic_identifier,
            "candidate_id": _generic_identifier,
            "bundle_fingerprint": _digest,
            "emergency_deadline_seconds": _positive_number,
        },
    ),
    SCHEMA_VERSIONS["experiment_run"]: RecordSchema(
        SCHEMA_VERSIONS["experiment_run"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["experiment_run"],)),
            "experiment_run_id": _identifier("xrun_", 32),
            "attempt_slot_id": _identifier("slot_", 20),
            "execution_configuration_id": _identifier("exec_", 20),
            "exact_argv": _string_list,
            "working_directory": _string,
            "fresh_state_root": _string,
            "bundle_path": _string,
            "output_path": _string,
            "run_order": _object,
            "observed_host_state": _object,
            "plan_id": _generic_identifier,
            "candidate_id": _generic_identifier,
            "bundle_fingerprint": _digest,
            "bundle_manifest_sha256": _digest,
            "archive_sha256": _digest,
            "aptus_job_ids": _unique_string_list,
            "aptus_run_ids": _unique_string_list,
            "terminal_evidence": _object,
        },
    ),
    SCHEMA_VERSIONS["claim_boundary"]: RecordSchema(
        SCHEMA_VERSIONS["claim_boundary"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["claim_boundary"],)),
            "campaign_id": _identifier("campaign_", 20),
            "claim_key": _string,
            "exact_scope": _object,
            "allowed_claim_types": _unique_string_list,
            "forbidden_claims": _unique_string_list,
            "qualification_dependencies": _unique_string_list,
            "statement": _bounded_string(240),
        },
    ),
    SCHEMA_VERSIONS["event_ledger_row"]: RecordSchema(
        SCHEMA_VERSIONS["event_ledger_row"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["event_ledger_row"],)),
            "sequence": _nonnegative_integer,
            "experiment_run_id": _identifier("xrun_", 32),
            "monotonic_ns": _nonnegative_integer,
            "wall_time_utc": _timestamp,
            "event_type": _enum(EVENT_TYPES),
            "phase": _nullable(_string),
            "action": _nullable(_string),
            "subject_kind": _nullable(_string),
            "subject_id": _nullable(_string),
            "observation_kind": _enum(OBSERVATION_KINDS),
            "source_reported_at_utc": _nullable(_timestamp),
            "exit_code": _nullable(_integer),
            "native_outcome": _nullable(_enum(NATIVE_OUTCOMES)),
            "reason_code": _enum(REASON_CODES),
        },
    ),
    SCHEMA_VERSIONS["telemetry_sample"]: RecordSchema(
        SCHEMA_VERSIONS["telemetry_sample"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["telemetry_sample"],)),
            "sequence": _nonnegative_integer,
            "experiment_run_id": _identifier("xrun_", 32),
            "scheduled_slot": _nonnegative_integer,
            "scheduled_monotonic_ns": _nonnegative_integer,
            "observed_monotonic_ns": _nonnegative_integer,
            "wall_time_utc": _timestamp,
            "sample_interval_seconds": _positive_number,
            "gpu": _object,
            "host": _object,
            "collector": _object,
            "watchdog": _object,
        },
    ),
    SCHEMA_VERSIONS["raw_manifest"]: RecordSchema(
        SCHEMA_VERSIONS["raw_manifest"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["raw_manifest"],)),
            "protected_artifact_id": _identifier("artifact_", 32),
            "record_kind": _enum(("experiment-run", "legacy-recovery")),
            "identity_bindings": _object,
            "capture_tool": _object,
            "source_bindings": _object,
            "retention_policy_id": _string,
            "provisional_retain_not_before_utc": _timestamp,
            "files": _raw_file_entries,
            "file_count": _nonnegative_integer,
            "total_bytes": _nonnegative_integer,
            "required_role_bindings": _object,
            "completion_marker": _enum(("SEALED.json",)),
        },
    ),
    SCHEMA_VERSIONS["raw_seal"]: RecordSchema(
        SCHEMA_VERSIONS["raw_seal"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["raw_seal"],)),
            "protected_artifact_id": _identifier("artifact_", 32),
            "raw_manifest_sha256": _digest,
            "raw_manifest_size_bytes": _positive_integer,
            "sealed_at_utc": _timestamp,
        },
    ),
    SCHEMA_VERSIONS["capture_failure"]: RecordSchema(
        SCHEMA_VERSIONS["capture_failure"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["capture_failure"],)),
            "protected_artifact_id": _identifier("artifact_", 32),
            "attempt_slot_id": _identifier("slot_", 20),
            "experiment_run_id": _identifier("xrun_", 32),
            "created_at_utc": _timestamp,
            "reason_code": _enum(REASON_CODES),
            "available_files": _raw_file_entries,
            "missing_fields": _unique_string_list,
            "recoverable_locator": _nullable(_string),
        },
    ),
    SCHEMA_VERSIONS["receipt"]: RecordSchema(
        SCHEMA_VERSIONS["receipt"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["receipt"],)),
            "receipt_id": _generic_identifier,
            "kind": _enum(RECEIPT_KINDS),
            "created_at_utc": _timestamp,
            "issuer_role_id": _string,
            "protected_artifact_id": _identifier("artifact_", 32),
            "raw_manifest_sha256": _digest,
            "raw_manifest_size_bytes": _positive_integer,
            "previous_receipt_id": _nullable(_generic_identifier),
            "result": _string,
            "details": _object,
        },
    ),
    SCHEMA_VERSIONS["sanitization_map"]: RecordSchema(
        SCHEMA_VERSIONS["sanitization_map"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["sanitization_map"],)),
            "entries": _traceability_entries,
        },
    ),
    SCHEMA_VERSIONS["recovery_supplement"]: RecordSchema(
        SCHEMA_VERSIONS["recovery_supplement"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["recovery_supplement"],)),
            "original_packet": _object,
            "expected_digest_manifest": _object,
            "recovery_raw_manifest": _object,
            "copy_verification_receipts": _list,
            "retrieval_receipt": _object,
            "retention_policy": _object,
            "retention_receipt": _object,
            "sanitization_map": _object,
            "independent_review": _object,
            "claim_boundary": _object,
            "summary_counts": _object,
            "items": _recovery_items,
            "additional_search_items": _additional_search_items,
        },
    ),
    SCHEMA_VERSIONS["independent_review"]: RecordSchema(
        SCHEMA_VERSIONS["independent_review"],
        {
            "schema_version": _enum((SCHEMA_VERSIONS["independent_review"],)),
            "review_id": _generic_identifier,
            "producer_role_id": _string,
            "reviewer_role_id": _string,
            "reviewed_at_utc": _timestamp,
            "checks": _review_checks,
            "result": _enum(("passed", "failed")),
            "reason_code": _enum(REASON_CODES),
        },
    ),
}

IDENTITY_FIELDS = {
    SCHEMA_VERSIONS["campaign"]: (
        "schema_version",
        "protocol_schema_version",
        "program_key",
        "phase_sequence",
        "host_class",
        "allowed_methods",
        "allowed_placement",
        "allowed_world_size",
    ),
    SCHEMA_VERSIONS["comparison_cohort"]: (
        "schema_version",
        "campaign_id",
        "question",
        "held_controls",
        "varied_dimensions",
        "member_cell_ids",
        "attempt_counts",
        "seed_schedule",
        "block_schedule",
        "stopping_rule",
        "promotion_rule",
        "no_replacement_rule",
        "aggregate_rule",
    ),
    SCHEMA_VERSIONS["comparison_cell"]: (
        "schema_version",
        "campaign_id",
        "source_binding",
        "host_binding",
        "environment_binding",
        "model_binding",
        "dataset_and_split_binding",
        "method",
        "precision",
        "quantization",
        "placement",
        "world_size",
        "sequence_length",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "training_budget",
        "checkpoint_rule",
        "adapter_targets",
        "seed_policy",
        "cache_policy",
        "cooldown_policy",
        "safety_policy",
        "capture_policy",
        "retention_policy_id",
    ),
    SCHEMA_VERSIONS["attempt_slot"]: (
        "schema_version",
        "comparison_cohort_id",
        "comparison_cell_id",
        "block",
        "ordinal",
        "role",
        "order_position",
        "scheduled_seed",
    ),
    SCHEMA_VERSIONS["execution_configuration"]: (
        "schema_version",
        "comparison_cell_id",
        "exact_behavior_values",
        "split_seed",
        "training_seed",
        "data_order_seed",
        "plan_id",
        "candidate_id",
        "bundle_fingerprint",
    ),
}
IDENTITY_ID_FIELDS = {
    SCHEMA_VERSIONS["campaign"]: ("campaign_id", "campaign_"),
    SCHEMA_VERSIONS["comparison_cohort"]: ("comparison_cohort_id", "cohort_"),
    SCHEMA_VERSIONS["comparison_cell"]: ("comparison_cell_id", "cell_"),
    SCHEMA_VERSIONS["attempt_slot"]: ("attempt_slot_id", "slot_"),
    SCHEMA_VERSIONS["execution_configuration"]: (
        "execution_configuration_id",
        "exec_",
    ),
}


def canonical_json_bytes(value: Any, *, trailing_newline: bool = True) -> bytes:
    """Return frozen pretty canonical JSON bytes after JSON-safety validation."""

    _validate_json_value(value)
    rendered = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    if trailing_newline:
        rendered += "\n"
    return rendered.encode("utf-8")


def compact_canonical_json_bytes(value: Any) -> bytes:
    """Return compact canonical bytes used for content-addressed identities."""

    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    """Render one compact canonical JSON object plus LF per input record."""

    output = bytearray()
    for record in records:
        if type(record) is not dict:
            raise ContractError("JSON Lines rows must be objects")
        output.extend(compact_canonical_json_bytes(record))
        output.extend(b"\n")
    return bytes(output)


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 of exact bytes."""

    if type(data) is not bytes:
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 of one regular file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    """Return a normalized RFC 3339 UTC timestamp."""

    return datetime.now(UTC).isoformat()


def deterministic_id(prefix: str, identity: Mapping[str, Any]) -> str:
    """Create a frozen 20-hex content-addressed identifier."""

    if prefix not in {"campaign_", "cohort_", "cell_", "slot_", "exec_"}:
        raise ContractError(f"unsupported deterministic identifier prefix: {prefix!r}")
    if type(identity) is not dict:
        raise ContractError("identity must be an object")
    if type(identity.get("schema_version")) is not str:
        raise ContractError("identity must contain schema_version")
    suffix = sha256_bytes(compact_canonical_json_bytes(identity))[:20]
    return f"{prefix}{suffix}"


_OPAQUE_PREFIXES = {
    "xrun": "xrun_",
    "xrun_": "xrun_",
    "artifact": "artifact_",
    "artifact_": "artifact_",
    "copy": "copy_",
    "copy_": "copy_",
    "host": "host_",
    "host_": "host_",
    "domain": "domain_",
    "domain_": "domain_",
}


def new_opaque_id(kind_or_prefix: str) -> str:
    """Create a cryptographically random opaque identifier."""

    try:
        prefix = _OPAQUE_PREFIXES[kind_or_prefix]
    except (KeyError, TypeError) as error:
        raise ContractError(
            f"unsupported opaque identifier kind: {kind_or_prefix!r}"
        ) from error
    return f"{prefix}{secrets.token_hex(16)}"


def validate_record(
    record: Mapping[str, Any],
    expected_schema: str | None = None,
) -> dict[str, Any]:
    """Validate one strict record and return a detached JSON-safe copy."""

    if type(record) is not dict:
        raise ContractError("record: top level must be an object")
    _validate_json_value(record)
    schema_version = record.get("schema_version")
    if type(schema_version) is not str:
        raise ContractError("record.schema_version: must be a string")
    if expected_schema is not None and schema_version != expected_schema:
        raise ContractError(
            f"record.schema_version: expected {expected_schema!r}, got {schema_version!r}"
        )
    try:
        schema = _SCHEMAS[schema_version]
    except KeyError as error:
        raise ContractError(
            f"unsupported schema_version: {schema_version!r}"
        ) from error
    expected_fields = set(schema.fields)
    actual_fields = set(record)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        raise ContractError(
            f"record: wrong top-level fields; missing={missing!r}, unknown={unknown!r}"
        )
    for name, check in schema.fields.items():
        check(record[name], f"record.{name}")
    _validate_record_semantics(record, schema_version)
    return json.loads(compact_canonical_json_bytes(record))


def _validate_record_semantics(record: Mapping[str, Any], schema_version: str) -> None:
    identity_fields = IDENTITY_FIELDS.get(schema_version)
    if identity_fields is not None:
        id_field, prefix = IDENTITY_ID_FIELDS[schema_version]
        identity = {name: record[name] for name in identity_fields}
        expected_id = deterministic_id(prefix, identity)
        if record[id_field] != expected_id:
            _fail(
                f"record.{id_field}", f"does not match canonical identity {expected_id}"
            )

    if schema_version == SCHEMA_VERSIONS["campaign"]:
        if record["phase_sequence"] != list(range(11)):
            _fail(
                "record.phase_sequence",
                "must be the frozen phase sequence 0 through 10",
            )
    elif schema_version == SCHEMA_VERSIONS["comparison_cohort"]:
        for index, cell_id in enumerate(record["member_cell_ids"]):
            _identifier("cell_", 20)(cell_id, f"record.member_cell_ids[{index}]")
    elif schema_version == SCHEMA_VERSIONS["comparison_cell"]:
        if record["effective_batch_size"] != (
            record["micro_batch_size"] * record["gradient_accumulation_steps"]
        ):
            _fail(
                "record.effective_batch_size",
                "must equal micro_batch_size * gradient_accumulation_steps",
            )
    elif schema_version == SCHEMA_VERSIONS["attempt_slot"]:
        _validate_attempt_slot_semantics(record)
    elif schema_version == SCHEMA_VERSIONS["execution_configuration"]:
        deadline = record["exact_behavior_values"].get("emergency_deadline_seconds")
        if deadline != record["emergency_deadline_seconds"]:
            _fail(
                "record.exact_behavior_values.emergency_deadline_seconds",
                "must equal the top-level emergency deadline so identity changes with it",
            )
    elif schema_version == SCHEMA_VERSIONS["telemetry_sample"]:
        if record["sample_interval_seconds"] != 1:
            _fail(
                "record.sample_interval_seconds", "must equal the frozen 1 Hz interval"
            )
        if record["observed_monotonic_ns"] < record["scheduled_monotonic_ns"]:
            _fail(
                "record.observed_monotonic_ns",
                "must not precede the scheduled timestamp",
            )
    elif schema_version == SCHEMA_VERSIONS["raw_manifest"]:
        if record["file_count"] != len(record["files"]):
            _fail("record.file_count", "must equal len(files)")
        expected_total = sum(entry["size_bytes"] for entry in record["files"])
        if record["total_bytes"] != expected_total:
            _fail("record.total_bytes", "must equal the sum of file sizes")
        forbidden = {"raw-manifest.json", "SEALED.json"}
        if any(entry["relative_path"] in forbidden for entry in record["files"]):
            _fail("record.files", "manifest and seal must be excluded from inventory")
    elif schema_version == SCHEMA_VERSIONS["receipt"]:
        _validate_receipt_semantics(record)
    elif schema_version == SCHEMA_VERSIONS["recovery_supplement"]:
        _validate_recovery_supplement_semantics(record)
    elif schema_version == SCHEMA_VERSIONS["independent_review"]:
        if record["producer_role_id"] == record["reviewer_role_id"]:
            _fail("record.reviewer_role_id", "must differ from producer_role_id")
        passed = all(record["checks"].values())
        if (record["result"] == "passed") != passed:
            _fail("record.result", "must reflect the complete frozen review checks")
        if passed and record["reason_code"] != "NONE":
            _fail("record.reason_code", "must be NONE when every review check passes")
        if not passed and record["reason_code"] == "NONE":
            _fail("record.reason_code", "must identify why a review check failed")


def _validate_attempt_slot_semantics(record: Mapping[str, Any]) -> None:
    if record["slot_status"] == "planned-not-started":
        if record["execution_configuration_id"] is not None:
            _fail(
                "record.execution_configuration_id",
                "must be null for an unstarted slot",
            )
        if record["experiment_run_id"] is not None:
            _fail("record.experiment_run_id", "must be null for an unstarted slot")
        if record["native_outcome"] is not None:
            _fail("record.native_outcome", "must be null for an unstarted slot")
        if record["evidence_status"] != "not-started":
            _fail("record.evidence_status", "must be not-started for an unstarted slot")
    else:
        for field in (
            "execution_configuration_id",
            "experiment_run_id",
            "native_outcome",
        ):
            if record[field] is None:
                _fail(f"record.{field}", "must be present for a started terminal slot")
        if record["evidence_status"] == "not-started":
            _fail("record.evidence_status", "cannot be not-started for a started slot")


_RETRIEVAL_DETAIL_FIELDS = (
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
)


def _validate_receipt_semantics(record: Mapping[str, Any]) -> None:
    if record["kind"] != "retrieval":
        return
    details = record["details"]
    _exact_object(details, _RETRIEVAL_DETAIL_FIELDS, "record.details")
    _identifier("copy_", 32)(details["source_copy_id"], "record.details.source_copy_id")
    _identifier("domain_", 32)(
        details["source_failure_domain_id"],
        "record.details.source_failure_domain_id",
    )
    _generic_identifier(
        details["destination_restore_id"],
        "record.details.destination_restore_id",
    )
    _timestamp(details["started_at_utc"], "record.details.started_at_utc")
    _timestamp(details["finished_at_utc"], "record.details.finished_at_utc")
    for field in (
        "duration_ns",
        "restored_file_count",
        "restored_total_bytes",
        "mismatch_count",
    ):
        _nonnegative_integer(details[field], f"record.details.{field}")
    _digest(
        details["expected_raw_manifest_sha256"],
        "record.details.expected_raw_manifest_sha256",
    )
    _nullable(_digest)(
        details["observed_raw_manifest_sha256"],
        "record.details.observed_raw_manifest_sha256",
    )
    _string(details["verification_result"], "record.details.verification_result")
    passed = record["result"] == "passed"
    if passed != (details["verification_result"] == "passed"):
        _fail(
            "record.details.verification_result",
            "must agree with the receipt result",
        )
    if passed:
        if (
            details["mismatch_count"] != 0
            or details["observed_raw_manifest_sha256"]
            != details["expected_raw_manifest_sha256"]
        ):
            _fail(
                "record.details",
                "a passing retrieval must have zero mismatches and matching digests",
            )
    elif details["mismatch_count"] < 1:
        _fail("record.details.mismatch_count", "a failed retrieval requires a mismatch")


def _validate_recovery_supplement_semantics(record: Mapping[str, Any]) -> None:
    counts = record["summary_counts"]
    required_counts = {
        "logical_digest_count",
        "recovered_matching",
        "recovered_mismatched",
        "not_found",
    }
    if set(counts) != required_counts:
        _fail(
            "record.summary_counts", f"must contain exactly {sorted(required_counts)!r}"
        )
    for name in required_counts:
        _nonnegative_integer(counts[name], f"record.summary_counts.{name}")
    if counts["logical_digest_count"] != len(record["items"]):
        _fail("record.summary_counts.logical_digest_count", "must equal len(items)")
    if counts["logical_digest_count"] != (
        counts["recovered_matching"]
        + counts["recovered_mismatched"]
        + counts["not_found"]
    ):
        _fail("record.summary_counts", "disposition counts must sum to logical count")
    observed = {
        "recovered_matching": sum(
            item["disposition"] == "recovered-matching" for item in record["items"]
        ),
        "recovered_mismatched": sum(
            item["disposition"] == "recovered-mismatched" for item in record["items"]
        ),
        "not_found": sum(
            item["disposition"] == "not-found" for item in record["items"]
        ),
    }
    for name, value in observed.items():
        if counts[name] != value:
            _fail(f"record.summary_counts.{name}", "does not match item dispositions")
    expected_reasons = {
        "recovered-matching": "RECOVERED_MATCH",
        "recovered-mismatched": "RECOVERED_MISMATCH",
        "not-found": "NOT_FOUND_AFTER_BOUNDED_SEARCH",
    }
    for index, item in enumerate(record["items"]):
        disposition = item["disposition"]
        if item["reason_code"] != expected_reasons[disposition]:
            _fail(
                f"record.items[{index}].reason_code",
                "must match the recovery disposition",
            )
        if disposition == "recovered-matching" and (
            item["recovered_sha256"] != item["expected_sha256"]
        ):
            _fail(
                f"record.items[{index}].recovered_sha256",
                "must equal expected_sha256 for recovered-matching",
            )
        if disposition == "recovered-mismatched" and (
            item["recovered_sha256"] == item["expected_sha256"]
        ):
            _fail(
                f"record.items[{index}].recovered_sha256",
                "must differ from expected_sha256 for recovered-mismatched",
            )


def record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen identity object for one deterministic record."""

    validated = validate_record(record)
    schema_version = validated["schema_version"]
    try:
        fields = IDENTITY_FIELDS[schema_version]
    except KeyError as error:
        raise ContractError(
            f"schema has no deterministic identity: {schema_version}"
        ) from error
    return {name: validated[name] for name in fields}


def validate_event_ledger(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate a complete canonical event ledger."""

    validated = [
        validate_record(record, SCHEMA_VERSIONS["event_ledger_row"])
        for record in records
    ]
    if not validated:
        raise ContractError("event ledger must not be empty")
    run_id = validated[0]["experiment_run_id"]
    prior_monotonic = -1
    for index, record in enumerate(validated):
        if record["sequence"] != index:
            raise ContractError(
                f"event ledger sequence must be contiguous: expected {index}, "
                f"got {record['sequence']}"
            )
        if record["experiment_run_id"] != run_id:
            raise ContractError("event ledger mixes experiment_run_id values")
        if record["monotonic_ns"] < prior_monotonic:
            raise ContractError("event ledger monotonic time moved backward")
        prior_monotonic = record["monotonic_ns"]
    if validated[0]["event_type"] != "clock.mapping":
        raise ContractError("event ledger must begin with clock.mapping")
    mapping_indices = [
        index
        for index, record in enumerate(validated)
        if record["event_type"] == "clock.mapping"
    ]
    if mapping_indices != [0, len(validated) - 2]:
        raise ContractError(
            "event ledger requires exactly an initial and penultimate final "
            "clock.mapping row"
        )
    if validated[-1]["event_type"] != "seal.started":
        raise ContractError("event ledger must end with seal.started")
    event_sequence = [record["event_type"] for record in validated]
    event_types = set(event_sequence)
    if event_sequence.count("harness.started") != 1 or event_sequence[1] != (
        "harness.started"
    ):
        raise ContractError(
            "event ledger requires exactly one harness.started after the initial mapping"
        )
    if event_sequence.count("harness.finished") != 1 or event_sequence[-3] != (
        "harness.finished"
    ):
        raise ContractError(
            "event ledger requires exactly one harness.finished before the final mapping"
        )
    if "telemetry.started" in event_types and not (
        {"telemetry.stopped", "telemetry.failed"} & event_types
    ):
        raise ContractError(
            "telemetry.started requires telemetry.stopped or telemetry.failed"
        )
    paired_boundaries = (
        ("command.started", ("command.finished",)),
        ("pilot.phase-started", ("pilot.phase-finished",)),
        ("training.started", ("training.finished",)),
        ("export.started", ("export.finished",)),
        ("verification.started", ("verification.finished",)),
        ("cooldown.started", ("cooldown.finished",)),
        ("telemetry.started", ("telemetry.stopped", "telemetry.failed")),
    )

    def boundary_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            record["phase"],
            record["action"],
            record["subject_kind"],
            record["subject_id"],
        )

    unmatched_runtime_starts: list[dict[str, Any]] = []
    runtime_started_types = {
        "pilot.phase-started",
        "training.started",
        "export.started",
        "verification.started",
    }
    for started, finished_options in paired_boundaries:
        active: dict[tuple[Any, ...], int] = {}
        for index, record in enumerate(validated):
            event_type = record["event_type"]
            key = boundary_key(record)
            if event_type == started:
                if key in active:
                    raise ContractError(
                        f"event ledger has overlapping {started} boundaries"
                    )
                active[key] = index
            elif event_type in finished_options:
                start_index = active.pop(key, None)
                if start_index is None or start_index >= index:
                    raise ContractError(
                        f"event ledger has an unmatched {event_type} boundary"
                    )
        if active:
            if started not in runtime_started_types:
                raise ContractError(
                    f"event ledger has an incomplete {started} boundary"
                )
            unmatched_runtime_starts.extend(
                validated[index] for index in active.values()
            )

    if unmatched_runtime_starts:
        # A killed child cannot truthfully emit a finish boundary.  Permit that
        # narrow case only when the complete terminal ledger proves an exact
        # managed non-pass prefix (and, for cancellation, the full safety
        # chain).  The lazy import avoids a contracts/outcomes import cycle.
        from .outcomes import (
            OutcomeProfileError,
            validate_unmatched_runtime_terminal_prefix,
        )

        try:
            validate_unmatched_runtime_terminal_prefix(
                validated,
                unmatched_runtime_starts,
            )
        except OutcomeProfileError as error:
            raise ContractError(
                "event ledger has an unauthorized incomplete runtime boundary"
            ) from error

    cancellation_types = {
        "cancellation.requested",
        "process-group.terminated",
        "lease.reconciled",
    }
    cancellation_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in validated:
        if record["event_type"] not in cancellation_types:
            continue
        key = (
            record["action"],
            record["subject_kind"],
            record["subject_id"],
            record["reason_code"],
        )
        cancellation_groups.setdefault(key, []).append(record)
    for key, chain in cancellation_groups.items():
        observed_types = [record["event_type"] for record in chain]
        expected_types = [
            "cancellation.requested",
            "process-group.terminated",
            "lease.reconciled",
        ]
        if observed_types != expected_types:
            raise ContractError(
                "event ledger cancellation milestones must form one complete "
                "request, termination, and lease-reconciliation chain"
            )
        action, subject_kind, subject_id, reason_code = key
        trigger_indices = [
            index
            for index, record in enumerate(validated)
            if record["event_type"] == "safety.triggered"
            and record["action"] == action
            and record["subject_kind"] == subject_kind
            and record["subject_id"] == subject_id
            and record["reason_code"] == reason_code
            and record["sequence"] < chain[0]["sequence"]
        ]
        if not trigger_indices:
            raise ContractError(
                "event ledger cancellation chain lacks its preceding exact safety trigger"
            )
    return validated


class EventLedgerWriter:
    """In-memory no-clobber writer for one complete experiment event ledger."""

    def __init__(self, experiment_run_id: str) -> None:
        _identifier("xrun_", 32)(experiment_run_id, "experiment_run_id")
        self._experiment_run_id = experiment_run_id
        self._records: list[dict[str, Any]] = []
        self._sealed = False

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            json.loads(compact_canonical_json_bytes(row)) for row in self._records
        )

    def append(
        self,
        *,
        monotonic_ns: int,
        wall_time_utc: str,
        event_type: str,
        phase: str | None = None,
        action: str | None = None,
        subject_kind: str | None = None,
        subject_id: str | None = None,
        observation_kind: str = "observed",
        source_reported_at_utc: str | None = None,
        exit_code: int | None = None,
        native_outcome: str | None = None,
        reason_code: str = "NONE",
    ) -> dict[str, Any]:
        if self._sealed:
            raise ContractError("cannot append after seal.started")
        record = {
            "schema_version": SCHEMA_VERSIONS["event_ledger_row"],
            "sequence": len(self._records),
            "experiment_run_id": self._experiment_run_id,
            "monotonic_ns": monotonic_ns,
            "wall_time_utc": wall_time_utc,
            "event_type": event_type,
            "phase": phase,
            "action": action,
            "subject_kind": subject_kind,
            "subject_id": subject_id,
            "observation_kind": observation_kind,
            "source_reported_at_utc": source_reported_at_utc,
            "exit_code": exit_code,
            "native_outcome": native_outcome,
            "reason_code": reason_code,
        }
        validated = validate_record(record, SCHEMA_VERSIONS["event_ledger_row"])
        if (
            self._records
            and validated["monotonic_ns"] < self._records[-1]["monotonic_ns"]
        ):
            raise ContractError("event ledger monotonic time moved backward")
        if not self._records and event_type != "clock.mapping":
            raise ContractError("first event must be clock.mapping")
        self._records.append(validated)
        if event_type == "seal.started":
            self._sealed = True
        return json.loads(compact_canonical_json_bytes(validated))

    def to_bytes(self) -> bytes:
        validated = validate_event_ledger(self._records)
        return canonical_jsonl_bytes(validated)

    def write(self, path: Path) -> str:
        """Write a complete ledger with exclusive creation and return its digest."""

        payload = self.to_bytes()
        target = Path(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        if os.name == "posix":
            directory_descriptor = os.open(
                target.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        return sha256_bytes(payload)


__all__ = [
    "ContractError",
    "EVIDENCE_CLASSES",
    "EVIDENCE_STATUSES",
    "EVENT_TYPES",
    "EventLedgerWriter",
    "IDENTITY_FIELDS",
    "NATIVE_OUTCOMES",
    "OBSERVATION_KINDS",
    "PROTOCOL_SCHEMA_VERSION",
    "RAW_FILE_ENTRY_FIELDS",
    "REASON_CODES",
    "RECEIPT_KINDS",
    "RECOVERY_DISPOSITIONS",
    "REVIEW_CHECKS",
    "SCHEMA_FIELDS",
    "SCHEMA_VERSIONS",
    "SLOT_STATUSES",
    "TRACEABILITY_ENTRY_FIELDS",
    "TRACEABILITY_TRANSFORMS",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "compact_canonical_json_bytes",
    "deterministic_id",
    "new_opaque_id",
    "record_identity",
    "sha256_bytes",
    "sha256_file",
    "utc_now",
    "validate_event_ledger",
    "validate_record",
    "validate_safe_relative_path",
]
