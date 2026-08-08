from __future__ import annotations

import json
import math
import os
import re
import stat
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tools.cuda_campaign.contracts import (
    ContractError,
    EVIDENCE_CLASSES,
    EVIDENCE_STATUSES,
    EVENT_TYPES,
    EventLedgerWriter,
    NATIVE_OUTCOMES,
    OBSERVATION_KINDS,
    RECEIPT_KINDS,
    RECOVERY_DISPOSITIONS,
    REASON_CODES,
    REVIEW_CHECKS,
    SCHEMA_VERSIONS,
    SLOT_STATUSES,
    TRACEABILITY_TRANSFORMS,
    canonical_json_bytes,
    compact_canonical_json_bytes,
    deterministic_id,
    new_opaque_id,
    record_identity,
    sha256_bytes,
    sha256_file,
    utc_now,
    validate_event_ledger,
    validate_record,
    validate_safe_relative_path,
)

TIMESTAMP = "2026-08-08T12:34:56+00:00"
DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
XRUN_ID = "xrun_" + "1" * 32
ARTIFACT_ID = "artifact_" + "2" * 32
COPY_ID = "copy_" + "3" * 32
DOMAIN_ID = "domain_" + "4" * 32


def with_identity(
    record: dict[str, object],
    *,
    id_field: str,
    prefix: str,
    identity_fields: tuple[str, ...],
) -> dict[str, object]:
    identity = {name: record[name] for name in identity_fields}
    record[id_field] = deterministic_id(prefix, identity)
    return record


def sample_campaign() -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["campaign"],
        "protocol_schema_version": "aptus.cuda-campaign-protocol.v1",
        "program_key": "rtx-3050-local",
        "phase_sequence": list(range(11)),
        "host_class": "single-rtx-3050-8gib",
        "allowed_methods": ["lora", "qlora"],
        "allowed_placement": "single",
        "allowed_world_size": 1,
    }
    return with_identity(
        record,
        id_field="campaign_id",
        prefix="campaign_",
        identity_fields=(
            "schema_version",
            "protocol_schema_version",
            "program_key",
            "phase_sequence",
            "host_class",
            "allowed_methods",
            "allowed_placement",
            "allowed_world_size",
        ),
    )


def sample_cohort(campaign_id: str) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["comparison_cohort"],
        "campaign_id": campaign_id,
        "question": "Does the anchor repeat?",
        "held_controls": {"placement": "single"},
        "varied_dimensions": ["training_seed"],
        "member_cell_ids": ["cell_" + "5" * 20],
        "attempt_counts": {"anchor": 5},
        "seed_schedule": {"training": [101]},
        "block_schedule": [{"block": 1}],
        "stopping_rule": {"rule": "no-replacement"},
        "promotion_rule": {"required": 5},
        "no_replacement_rule": True,
        "aggregate_rule": {"median": "type-7"},
    }
    return with_identity(
        record,
        id_field="comparison_cohort_id",
        prefix="cohort_",
        identity_fields=(
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
    )


def sample_cell(campaign_id: str) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["comparison_cell"],
        "campaign_id": campaign_id,
        "source_binding": {"commit": "f" * 40},
        "host_binding": {"host_id": "host_" + "6" * 32},
        "environment_binding": {"python": "3.12"},
        "model_binding": {"revision": "e" * 40},
        "dataset_and_split_binding": {"sha256": DIGEST},
        "method": "lora",
        "precision": "bf16",
        "quantization": None,
        "placement": "single",
        "world_size": 1,
        "sequence_length": 256,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 8,
        "training_budget": {"optimizer_steps": 128},
        "checkpoint_rule": {"cadence": 64},
        "adapter_targets": ["q_proj", "v_proj"],
        "seed_policy": {"split_seed": 424242},
        "cache_policy": {"model": "warm"},
        "cooldown_policy": {"samples": 120},
        "safety_policy": {"temperature_c": 84},
        "capture_policy": {"sample_interval_seconds": 1},
        "retention_policy_id": "cuda-v02-public-claim-evidence-24m-v1",
    }
    return with_identity(
        record,
        id_field="comparison_cell_id",
        prefix="cell_",
        identity_fields=tuple(key for key in record if key != "comparison_cell_id"),
    )


def sample_records() -> dict[str, dict[str, object]]:
    campaign = sample_campaign()
    campaign_id = str(campaign["campaign_id"])
    cohort = sample_cohort(campaign_id)
    cell = sample_cell(campaign_id)
    cell_id = str(cell["comparison_cell_id"])

    slot: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["attempt_slot"],
        "comparison_cohort_id": cohort["comparison_cohort_id"],
        "comparison_cell_id": cell_id,
        "block": 0,
        "ordinal": 1,
        "role": "anchor",
        "order_position": 0,
        "scheduled_seed": 101,
        "slot_status": "planned-not-started",
        "execution_configuration_id": None,
        "experiment_run_id": None,
        "native_outcome": None,
        "evidence_status": "not-started",
        "reason_code": "PRIOR_STOP_RULE",
    }
    with_identity(
        slot,
        id_field="attempt_slot_id",
        prefix="slot_",
        identity_fields=(
            "schema_version",
            "comparison_cohort_id",
            "comparison_cell_id",
            "block",
            "ordinal",
            "role",
            "order_position",
            "scheduled_seed",
        ),
    )

    behavior = {"emergency_deadline_seconds": 30, "command": ["true"]}
    execution: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["execution_configuration"],
        "comparison_cell_id": cell_id,
        "exact_behavior_values": behavior,
        "split_seed": 424242,
        "training_seed": 101,
        "data_order_seed": 1000101,
        "plan_id": "plan_example",
        "candidate_id": "candidate_example",
        "bundle_fingerprint": DIGEST,
        "emergency_deadline_seconds": 30,
    }
    with_identity(
        execution,
        id_field="execution_configuration_id",
        prefix="exec_",
        identity_fields=(
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
    )

    file_entry = {
        "entry_id": "entry_output",
        "role": "command-output",
        "relative_path": "logs/command.log",
        "media_type": "text/plain",
        "size_bytes": 12,
        "sha256": DIGEST,
        "captured_at_utc": TIMESTAMP,
    }
    retrieval_details = {
        "source_copy_id": COPY_ID,
        "source_failure_domain_id": DOMAIN_ID,
        "destination_restore_id": "restore_example",
        "started_at_utc": TIMESTAMP,
        "finished_at_utc": TIMESTAMP,
        "duration_ns": 1,
        "restored_file_count": 1,
        "restored_total_bytes": 12,
        "expected_raw_manifest_sha256": DIGEST,
        "observed_raw_manifest_sha256": DIGEST,
        "mismatch_count": 0,
        "verification_result": "passed",
    }
    records = {
        "campaign": campaign,
        "comparison_cohort": cohort,
        "comparison_cell": cell,
        "attempt_slot": slot,
        "execution_configuration": execution,
        "experiment_run": {
            "schema_version": SCHEMA_VERSIONS["experiment_run"],
            "experiment_run_id": XRUN_ID,
            "attempt_slot_id": slot["attempt_slot_id"],
            "execution_configuration_id": execution["execution_configuration_id"],
            "exact_argv": ["true"],
            "working_directory": "/protected/work",
            "fresh_state_root": "/protected/state",
            "bundle_path": "/protected/bundle",
            "output_path": "/protected/output",
            "run_order": {"block": 0, "slot": 0},
            "observed_host_state": {"idle": True},
            "plan_id": "plan_example",
            "candidate_id": "candidate_example",
            "bundle_fingerprint": DIGEST,
            "bundle_manifest_sha256": DIGEST,
            "archive_sha256": OTHER_DIGEST,
            "aptus_job_ids": ["job_example"],
            "aptus_run_ids": ["run_example"],
            "terminal_evidence": {"native_outcome": "passed"},
        },
        "claim_boundary": {
            "schema_version": SCHEMA_VERSIONS["claim_boundary"],
            "campaign_id": campaign_id,
            "claim_key": "raw-recovery-integrity",
            "exact_scope": {"host": "opaque"},
            "allowed_claim_types": ["raw-recovery-integrity"],
            "forbidden_claims": ["release-readiness"],
            "qualification_dependencies": ["independent-review"],
            "statement": "Exact protected recovery integrity only.",
        },
        "event_ledger_row": {
            "schema_version": SCHEMA_VERSIONS["event_ledger_row"],
            "sequence": 0,
            "experiment_run_id": XRUN_ID,
            "monotonic_ns": 1,
            "wall_time_utc": TIMESTAMP,
            "event_type": "clock.mapping",
            "phase": "capture",
            "action": "map-clock",
            "subject_kind": "harness",
            "subject_id": XRUN_ID,
            "observation_kind": "emitted",
            "source_reported_at_utc": None,
            "exit_code": None,
            "native_outcome": None,
            "reason_code": "NONE",
        },
        "telemetry_sample": {
            "schema_version": SCHEMA_VERSIONS["telemetry_sample"],
            "sequence": 0,
            "experiment_run_id": XRUN_ID,
            "scheduled_slot": 0,
            "scheduled_monotonic_ns": 1,
            "observed_monotonic_ns": 1,
            "wall_time_utc": TIMESTAMP,
            "sample_interval_seconds": 1,
            "gpu": {"status": "supported"},
            "host": {"status": "supported"},
            "collector": {"healthy": True},
            "watchdog": {"healthy": True},
        },
        "raw_manifest": {
            "schema_version": SCHEMA_VERSIONS["raw_manifest"],
            "protected_artifact_id": ARTIFACT_ID,
            "record_kind": "experiment-run",
            "identity_bindings": {"experiment_run_id": XRUN_ID},
            "capture_tool": {"version": "test"},
            "source_bindings": {"commit": "f" * 40},
            "retention_policy_id": "cuda-v02-public-claim-evidence-24m-v1",
            "provisional_retain_not_before_utc": TIMESTAMP,
            "files": [file_entry],
            "file_count": 1,
            "total_bytes": 12,
            "required_role_bindings": {"command-output": "entry_output"},
            "completion_marker": "SEALED.json",
        },
        "raw_seal": {
            "schema_version": SCHEMA_VERSIONS["raw_seal"],
            "protected_artifact_id": ARTIFACT_ID,
            "raw_manifest_sha256": DIGEST,
            "raw_manifest_size_bytes": 100,
            "sealed_at_utc": TIMESTAMP,
        },
        "capture_failure": {
            "schema_version": SCHEMA_VERSIONS["capture_failure"],
            "protected_artifact_id": ARTIFACT_ID,
            "attempt_slot_id": slot["attempt_slot_id"],
            "experiment_run_id": XRUN_ID,
            "created_at_utc": TIMESTAMP,
            "reason_code": "STREAM_CAPTURE_FAILURE",
            "available_files": [file_entry],
            "missing_fields": ["terminal_evidence"],
            "recoverable_locator": "artifact-locator-opaque",
        },
        "receipt": {
            "schema_version": SCHEMA_VERSIONS["receipt"],
            "receipt_id": "receipt_example",
            "kind": "retrieval",
            "created_at_utc": TIMESTAMP,
            "issuer_role_id": "operator-role",
            "protected_artifact_id": ARTIFACT_ID,
            "raw_manifest_sha256": DIGEST,
            "raw_manifest_size_bytes": 100,
            "previous_receipt_id": None,
            "result": "passed",
            "details": retrieval_details,
        },
        "sanitization_map": {
            "schema_version": SCHEMA_VERSIONS["sanitization_map"],
            "entries": [
                {
                    "public_file": "supplement.json",
                    "public_json_pointer": "/summary_counts/not_found",
                    "source_raw_manifest_sha256": DIGEST,
                    "source_artifact_entry_id": "entry_inventory",
                    "source_json_pointer": "/items/0/disposition",
                    "transform": "count",
                    "evidence_class": "inferred",
                }
            ],
        },
        "recovery_supplement": {
            "schema_version": SCHEMA_VERSIONS["recovery_supplement"],
            "original_packet": {"date": "2026-08-06"},
            "expected_digest_manifest": {"sha256": DIGEST},
            "recovery_raw_manifest": {"sha256": DIGEST},
            "copy_verification_receipts": [{"sha256": DIGEST}],
            "retrieval_receipt": {"sha256": DIGEST},
            "retention_policy": {"id": "cuda-v02-public-claim-evidence-24m-v1"},
            "retention_receipt": {"sha256": DIGEST},
            "sanitization_map": {"sha256": DIGEST},
            "independent_review": {"result": "passed"},
            "claim_boundary": {"claim": "raw-recovery-integrity"},
            "summary_counts": {
                "logical_digest_count": 1,
                "recovered_matching": 0,
                "recovered_mismatched": 0,
                "not_found": 1,
            },
            "items": [
                {
                    "logical_item_id": "python-test-transcript",
                    "source_json_pointer": "/python-test-transcript",
                    "expected_sha256": DIGEST,
                    "disposition": "not-found",
                    "recovered_artifact_entry_id": None,
                    "recovered_sha256": None,
                    "recovered_size_bytes": None,
                    "reason_code": "NOT_FOUND_AFTER_BOUNDED_SEARCH",
                }
            ],
            "additional_search_items": [
                {
                    "item_id": "python-test-transcript",
                    "disposition": "not-found",
                    "reason_code": "ORIGINAL_TRANSCRIPT_NOT_FOUND",
                    "search_scope_codes": ["BOUNDED_PHASE0_SEARCH_COMPLETE"],
                }
            ],
        },
        "independent_review": {
            "schema_version": SCHEMA_VERSIONS["independent_review"],
            "review_id": "review_example",
            "producer_role_id": "producer-role",
            "reviewer_role_id": "reviewer-role",
            "reviewed_at_utc": TIMESTAMP,
            "checks": dict.fromkeys(REVIEW_CHECKS, True),
            "result": "passed",
            "reason_code": "NONE",
        },
    }
    return records


class CudaCampaignContractTests(unittest.TestCase):
    def test_closed_vocabularies_match_the_frozen_phase1_companion(self) -> None:
        protocol_path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "reference"
            / "cuda-campaign-protocol.v1.json"
        )
        protocol = json.loads(protocol_path.read_bytes())

        self.assertEqual(protocol["record_schema_family"], SCHEMA_VERSIONS)
        self.assertEqual(protocol["state_axes"]["slot_status"], list(SLOT_STATUSES))
        self.assertEqual(
            protocol["state_axes"]["native_outcome"], list(NATIVE_OUTCOMES)
        )
        self.assertEqual(
            protocol["state_axes"]["evidence_status"], list(EVIDENCE_STATUSES)
        )
        self.assertEqual(
            protocol["event_ledger_contract"]["event_types"], list(EVENT_TYPES)
        )
        self.assertEqual(
            protocol["event_ledger_contract"]["observation_kinds"],
            list(OBSERVATION_KINDS),
        )
        self.assertEqual(protocol["receipt_contract"]["kinds"], list(RECEIPT_KINDS))
        self.assertEqual(
            protocol["recovery_supplement_contract"]["item_dispositions"],
            list(RECOVERY_DISPOSITIONS),
        )
        self.assertEqual(
            protocol["safety_contract"]["reason_codes"], list(REASON_CODES)
        )
        self.assertEqual(
            protocol["storage_contract"]["raw_manifest"]["record_kind_values"],
            ["experiment-run", "legacy-recovery"],
        )
        traceability = protocol["sanitizer_contract"]["traceability_map"]
        self.assertEqual(traceability["transforms"], list(TRACEABILITY_TRANSFORMS))
        self.assertEqual(traceability["evidence_classes"], list(EVIDENCE_CLASSES))
        self.assertEqual(
            [
                check.lower().replace(" ", "-")
                for check in protocol["sanitizer_contract"]["review"]["required_checks"]
            ],
            list(REVIEW_CHECKS),
        )

    def test_all_sixteen_frozen_schema_envelopes_validate(self) -> None:
        records = sample_records()
        self.assertEqual(set(records), set(SCHEMA_VERSIONS))
        for name, record in records.items():
            with self.subTest(name=name):
                self.assertEqual(
                    validate_record(record, SCHEMA_VERSIONS[name]),
                    record,
                )

    def test_validator_rejects_unknown_missing_coerced_and_nonfinite_values(
        self,
    ) -> None:
        record = sample_records()["telemetry_sample"]
        for mutation in (
            lambda value: value.update(unknown=True),
            lambda value: value.pop("host"),
            lambda value: value.update(sequence=True),
            lambda value: value["gpu"].update(value=math.inf),
        ):
            changed = deepcopy(record)
            mutation(changed)
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                validate_record(changed)
        with self.assertRaises(ContractError):
            validate_record([])  # type: ignore[arg-type]

    def test_canonical_json_and_content_identity_are_deterministic(self) -> None:
        value = {"z": "é", "a": [1, 2]}
        self.assertEqual(
            canonical_json_bytes(value),
            b'{\n  "a": [\n    1,\n    2\n  ],\n  "z": "\xc3\xa9"\n}\n',
        )
        self.assertEqual(
            compact_canonical_json_bytes(value), b'{"a":[1,2],"z":"\xc3\xa9"}'
        )
        identity = {"schema_version": "example.v1", "value": 1}
        first = deterministic_id("cell_", identity)
        self.assertEqual(
            first, deterministic_id("cell_", dict(reversed(identity.items())))
        )
        self.assertNotEqual(
            first,
            deterministic_id("cell_", {"schema_version": "example.v1", "value": 2}),
        )
        self.assertRegex(first, r"^cell_[0-9a-f]{20}$")

    def test_record_identity_matches_validated_record(self) -> None:
        record = sample_campaign()
        identity = record_identity(record)
        self.assertEqual(
            record["campaign_id"],
            deterministic_id("campaign_", identity),
        )
        changed = deepcopy(record)
        changed["host_class"] = "different"
        with self.assertRaises(ContractError):
            validate_record(changed)

    def test_opaque_ids_are_typed_random_and_not_derived_from_private_values(
        self,
    ) -> None:
        first = new_opaque_id("host")
        second = new_opaque_id("host_")
        self.assertRegex(first, r"^host_[0-9a-f]{32}$")
        self.assertRegex(second, r"^host_[0-9a-f]{32}$")
        self.assertNotEqual(first, second)
        with self.assertRaises(ContractError):
            new_opaque_id("receipt")

    def test_safe_relative_posix_path_is_fail_closed(self) -> None:
        self.assertEqual(
            validate_safe_relative_path("logs/command.log"), "logs/command.log"
        )
        for path in (
            "",
            "/absolute",
            "../escape",
            "a/../escape",
            "a//b",
            "a/./b",
            "a\\b",
            "trailing/",
        ):
            with self.subTest(path=path), self.assertRaises(ContractError):
                validate_safe_relative_path(path)

    def test_event_ledger_writer_enforces_complete_ordered_ledger(self) -> None:
        writer = EventLedgerWriter(XRUN_ID)
        event_types = (
            "clock.mapping",
            "harness.started",
            "telemetry.started",
            "telemetry.stopped",
            "harness.finished",
            "clock.mapping",
            "seal.started",
        )
        for monotonic_ns, event_type in enumerate(event_types, start=1):
            writer.append(
                monotonic_ns=monotonic_ns,
                wall_time_utc=TIMESTAMP,
                event_type=event_type,
                phase="capture",
                action="capture",
                subject_kind="harness",
                subject_id=XRUN_ID,
                observation_kind="emitted",
            )
        payload = writer.to_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        self.assertEqual(validate_event_ledger(rows), rows)
        self.assertEqual([row["sequence"] for row in rows], list(range(len(rows))))
        self.assertEqual(rows[-1]["event_type"], "seal.started")
        with self.assertRaises(ContractError):
            writer.append(
                monotonic_ns=99,
                wall_time_utc=TIMESTAMP,
                event_type="harness.finished",
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            digest = writer.write(path)
            self.assertEqual(digest, sha256_file(path))
            self.assertEqual(digest, sha256_bytes(payload))
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                writer.write(path)

    def test_event_ledger_rejects_sequence_time_mapping_and_final_event_defects(
        self,
    ) -> None:
        writer = EventLedgerWriter(XRUN_ID)
        for monotonic_ns, event_type in enumerate(
            (
                "clock.mapping",
                "harness.started",
                "harness.finished",
                "clock.mapping",
                "seal.started",
            ),
            start=1,
        ):
            writer.append(
                monotonic_ns=monotonic_ns,
                wall_time_utc=TIMESTAMP,
                event_type=event_type,
            )
        rows = list(writer.records)
        mutations = (
            lambda value: value[1].update(sequence=3),
            lambda value: value[1].update(monotonic_ns=0),
            lambda value: value.pop(3),
            lambda value: value[-1].update(event_type="harness.finished"),
            lambda value: value.insert(2, deepcopy(value[0])),
        )
        for mutation in mutations:
            changed = deepcopy(rows)
            mutation(changed)
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                validate_event_ledger(changed)

    def test_event_ledger_rejects_mismatched_boundaries_and_cancel_chains(
        self,
    ) -> None:
        def complete_rows(middle: list[dict[str, object]]) -> list[dict[str, object]]:
            writer = EventLedgerWriter(XRUN_ID)
            events = [
                {"event_type": "clock.mapping"},
                {"event_type": "harness.started"},
                *middle,
                {
                    "event_type": "harness.finished",
                    "native_outcome": "cancelled",
                    "reason_code": "WATCHDOG_HEARTBEAT_LOST",
                },
                {"event_type": "clock.mapping"},
                {
                    "event_type": "seal.started",
                    "native_outcome": "cancelled",
                    "reason_code": "WATCHDOG_HEARTBEAT_LOST",
                },
            ]
            for monotonic_ns, values in enumerate(events, start=1):
                writer.append(
                    monotonic_ns=monotonic_ns,
                    wall_time_utc=TIMESTAMP,
                    action=values.get("action"),
                    subject_kind=values.get("subject_kind"),
                    subject_id=values.get("subject_id"),
                    native_outcome=values.get("native_outcome"),
                    reason_code=str(values.get("reason_code", "NONE")),
                    event_type=str(values["event_type"]),
                )
            return list(writer.records)

        mismatched = complete_rows(
            [
                {
                    "event_type": "command.started",
                    "action": "setup-one",
                    "subject_kind": "process",
                    "subject_id": "process_one",
                },
                {
                    "event_type": "command.finished",
                    "action": "setup-two",
                    "subject_kind": "process",
                    "subject_id": "process_two",
                },
            ]
        )
        with self.assertRaisesRegex(ContractError, "unmatched|incomplete"):
            validate_event_ledger(mismatched)

        exact_chain = complete_rows(
            [
                {
                    "event_type": "safety.triggered",
                    "action": "train",
                    "subject_kind": "aptus-job",
                    "subject_id": "job_exact",
                    "reason_code": "WATCHDOG_HEARTBEAT_LOST",
                },
                {
                    "event_type": "cancellation.requested",
                    "action": "train",
                    "subject_kind": "aptus-job",
                    "subject_id": "job_exact",
                    "reason_code": "WATCHDOG_HEARTBEAT_LOST",
                },
                {
                    "event_type": "process-group.terminated",
                    "action": "train",
                    "subject_kind": "aptus-job",
                    "subject_id": "job_exact",
                    "reason_code": "WATCHDOG_HEARTBEAT_LOST",
                },
                {
                    "event_type": "lease.reconciled",
                    "action": "train",
                    "subject_kind": "aptus-job",
                    "subject_id": "job_exact",
                    "reason_code": "WATCHDOG_HEARTBEAT_LOST",
                },
            ]
        )
        self.assertEqual(validate_event_ledger(exact_chain), exact_chain)
        missing_termination = [
            row
            for row in exact_chain
            if row["event_type"] != "process-group.terminated"
        ]
        for sequence, row in enumerate(missing_termination):
            row["sequence"] = sequence
        with self.assertRaisesRegex(ContractError, "cancellation milestones"):
            validate_event_ledger(missing_termination)

    def test_raw_manifest_and_recovery_summary_invariants_fail_closed(self) -> None:
        records = sample_records()
        manifest = deepcopy(records["raw_manifest"])
        manifest["total_bytes"] = 13
        with self.assertRaises(ContractError):
            validate_record(manifest)
        supplement = deepcopy(records["recovery_supplement"])
        supplement["summary_counts"]["not_found"] = 0
        with self.assertRaises(ContractError):
            validate_record(supplement)

    def test_failed_retrieval_receipt_may_record_an_unreadable_manifest(self) -> None:
        receipt = deepcopy(sample_records()["receipt"])
        receipt["result"] = "failed"
        receipt["details"]["verification_result"] = "failed"
        receipt["details"]["observed_raw_manifest_sha256"] = None
        receipt["details"]["mismatch_count"] = 1
        self.assertEqual(validate_record(receipt), receipt)

    def test_timestamp_output_is_normalized_utc(self) -> None:
        self.assertRegex(utc_now(), re.compile(r"\+00:00$"))


if __name__ == "__main__":
    unittest.main()
