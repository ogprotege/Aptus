"""Opt-in tooling for the bounded CUDA evidence campaign."""

from .contracts import (
    ContractError,
    EventLedgerWriter,
    PROCEDURAL_ROLE_ID_RE,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    compact_canonical_json_bytes,
    deterministic_id,
    new_opaque_id,
    sha256_bytes,
    sha256_file,
    utc_now,
    validate_event_ledger,
    validate_record,
    validate_safe_relative_path,
)

__all__ = [
    "ContractError",
    "EventLedgerWriter",
    "PROCEDURAL_ROLE_ID_RE",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "compact_canonical_json_bytes",
    "deterministic_id",
    "new_opaque_id",
    "sha256_bytes",
    "sha256_file",
    "utc_now",
    "validate_event_ledger",
    "validate_record",
    "validate_safe_relative_path",
]
