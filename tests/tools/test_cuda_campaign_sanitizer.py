from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tools.cuda_campaign.contracts import (
    canonical_json_bytes,
    compact_canonical_json_bytes,
    sha256_bytes,
)
from tools.cuda_campaign.sanitizer import (
    EXPECTED_DIGEST_SHA256,
    RECOVERY_INPUT_SCHEMA,
    SanitizationError,
    finalize_projection_stage,
    load_expected_digest_manifest,
    load_verified_recovery_context,
    project_recovery_supplement,
    project_verified_recovery_supplement,
    scan_public_value,
    seal_projection_review,
    stable_reason,
    verify_finalized_projection,
    verify_projection_stage,
    write_projection_stage,
)
from tools.cuda_campaign.storage import RawArtifactWriter, verify_sealed_artifact


REPOSITORY = Path(__file__).resolve().parents[2]
EXPECTED = (
    REPOSITORY
    / "docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance"
    / "raw-artifact-digests.json"
)


def _after_seals(*artifacts: Path, seconds: int = 1) -> str:
    latest = max(
        datetime.fromisoformat(
            verify_sealed_artifact(path)["seal"]["sealed_at_utc"]
        ).astimezone(timezone.utc)
        for path in artifacts
    )
    return (latest + timedelta(seconds=seconds)).isoformat()


def _after_timestamp(value: str, *, seconds: int = 1) -> str:
    return (
        datetime.fromisoformat(value).astimezone(timezone.utc)
        + timedelta(seconds=seconds)
    ).isoformat()


def _receipt(
    kind: str,
    marker: str,
    *,
    copy_id: str | None,
    domain_id: str | None,
    result: str = "passed",
) -> dict[str, object]:
    return {
        "receipt_id": f"receipt_{marker}",
        "kind": kind,
        "created_at_utc": "2026-08-08T12:00:00+00:00",
        "protected_artifact_id": "artifact_" + "a" * 32,
        "sha256": marker * 64,
        "size_bytes": 100,
        "result": result,
        "copy_id": copy_id,
        "failure_domain_id": domain_id,
    }


def _protected_input(*, omit_last: bool = False, mismatch_first: bool = False) -> dict:
    _manifest, expected_rows = load_expected_digest_manifest(EXPECTED)
    grouped: dict[str, list[str]] = {}
    for _logical_id, pointer, digest in expected_rows:
        grouped.setdefault(digest, []).append(pointer)
    entries = []
    for index, (digest, pointers) in enumerate(grouped.items()):
        if omit_last and index == len(grouped) - 1:
            continue
        observed = "f" * 64 if mismatch_first and index == 0 else digest
        entries.append(
            {
                "entry_id": f"entry_{index:03d}",
                "sha256": observed,
                "size_bytes": 1000 + index,
                "logical_source_pointers": pointers,
            }
        )
    return {
        "schema_version": RECOVERY_INPUT_SCHEMA,
        "producer_role_id": "phase2-packet-producer",
        "campaign_id": "campaign_" + "c" * 20,
        "original_packet": (
            "docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance"
        ),
        "recovery_raw_manifest": {
            "protected_artifact_id": "artifact_" + "a" * 32,
            "sha256": "b" * 64,
            "size_bytes": 12345,
            "entry_id": "entry_raw_manifest",
        },
        "recovered_entries": entries,
        "copy_verification_receipts": [
            _receipt(
                "copy-verification",
                "c",
                copy_id="copy_" + "1" * 32,
                domain_id="domain_" + "1" * 32,
            ),
            _receipt(
                "copy-verification",
                "d",
                copy_id="copy_" + "2" * 32,
                domain_id="domain_" + "2" * 32,
            ),
        ],
        "retrieval_receipt": _receipt(
            "retrieval",
            "e",
            copy_id="copy_" + "2" * 32,
            domain_id="domain_" + "2" * 32,
        ),
        "retention_receipt": _receipt(
            "retention", "f", copy_id=None, domain_id=None, result="active"
        ),
        "additional_search_results": [
            {
                "item_id": "python-test-transcript",
                "disposition": "not-found",
                "reason_code": "ORIGINAL_TRANSCRIPT_NOT_FOUND",
                "search_scope_codes": [
                    "source-host-boundary",
                    "verified-copy-one",
                    "verified-copy-two",
                ],
            }
        ],
        "forbidden_private_literals": [
            "example-private-user",
            "example-private-host",
            "10.22.33.44",
        ],
    }


def _typed_receipt(
    *,
    kind: str,
    recovery: dict,
    previous_receipt_id: str | None,
    result: str,
    details: dict,
) -> dict:
    without_id = {
        "schema_version": "aptus.experiment-evidence-receipt.v1",
        "kind": kind,
        "created_at_utc": "2026-08-08T12:00:00+00:00",
        "issuer_role_id": "phase2-evidence-custodian",
        "protected_artifact_id": recovery["protected_artifact_id"],
        "raw_manifest_sha256": recovery["raw_manifest_sha256"],
        "raw_manifest_size_bytes": recovery["raw_manifest_size_bytes"],
        "previous_receipt_id": previous_receipt_id,
        "result": result,
        "details": details,
    }
    receipt_id = (
        "receipt_" + sha256_bytes(compact_canonical_json_bytes(without_id))[:32]
    )
    return {"receipt_id": receipt_id, **without_id}


def _safe_receipt(receipt: dict) -> dict:
    details = receipt["details"]
    receipt_bytes = canonical_json_bytes(receipt)
    if receipt["kind"] == "copy-verification":
        copy_id = details["copy_id"]
        domain_id = details["failure_domain_id"]
    elif receipt["kind"] == "retrieval":
        copy_id = details["source_copy_id"]
        domain_id = details["source_failure_domain_id"]
    else:
        copy_id = None
        domain_id = None
    return {
        "receipt_id": receipt["receipt_id"],
        "kind": receipt["kind"],
        "created_at_utc": receipt["created_at_utc"],
        "protected_artifact_id": receipt["protected_artifact_id"],
        "sha256": sha256_bytes(receipt_bytes),
        "size_bytes": len(receipt_bytes),
        "result": receipt["result"],
        "copy_id": copy_id,
        "failure_domain_id": domain_id,
    }


def _sealed_context(
    root: Path,
    *,
    input_receipt_mismatch: bool = False,
    recovered_binding_mismatch: bool = False,
    extra_recovery_payload: bool = False,
    unsealed_recovered_input: bool = False,
    wrong_recovery_payload_role: bool = False,
    missing_recovery_role_binding: bool = False,
    receipt_id_mismatch: bool = False,
    expected_manifest_mismatch: bool = False,
    extra_control_payload: bool = False,
    extra_control_receipt: bool = False,
    extra_public_receipt: bool = False,
    noncanonical_receipt_path: bool = False,
    retrieval_inventory_mismatch: bool = False,
    control_source_mismatch: bool = False,
    no_off_host_copy: bool = False,
) -> tuple[Path, Path, dict]:
    recovery_path = root / "recovery"
    recovered_entry_ids = ["entry_recovered_000"]
    if extra_recovery_payload:
        recovered_entry_ids.append("entry_recovered_001")
    recovery_role = (
        "unexpected-recovery-role"
        if wrong_recovery_payload_role
        else "recovered-artifact"
    )
    recovery_role_bindings = (
        {} if missing_recovery_role_binding else {recovery_role: recovered_entry_ids}
    )
    recovery_writer = RawArtifactWriter(
        recovery_path,
        protected_artifact_id="artifact_" + "a" * 32,
        record_kind="legacy-recovery",
        identity_bindings={"purpose": "august-6-recovery"},
        capture_tool={"name": "sanitizer-test", "version": "v1"},
        source_bindings={"packet": "august-6"},
        provisional_retain_not_before_utc="2028-08-08T12:00:00+00:00",
        required_role_bindings=recovery_role_bindings,
    )
    recovered_entry = recovery_writer.write_payload(
        b"one recovered artifact with deliberately nonmatching bytes\n",
        "recovered/item.bin",
        role=recovery_role,
        entry_id="entry_recovered_000",
    )
    if extra_recovery_payload:
        recovery_writer.write_payload(
            b"a second sealed payload omitted from the recovery input\n",
            "recovered/second.bin",
            role=recovery_role,
            entry_id="entry_recovered_001",
        )
    recovery = recovery_writer.seal()

    copy_one = _typed_receipt(
        kind="copy-verification",
        recovery=recovery,
        previous_receipt_id=None,
        result="passed",
        details={
            "copy_id": "copy_" + "1" * 32,
            "failure_domain_id": "domain_" + "1" * 32,
            "off_experiment_host": False,
            "verification_result": "passed",
        },
    )
    if receipt_id_mismatch:
        copy_one["receipt_id"] = "receipt_" + "9" * 32
    copy_two = _typed_receipt(
        kind="copy-verification",
        recovery=recovery,
        previous_receipt_id=copy_one["receipt_id"],
        result="passed",
        details={
            "copy_id": "copy_" + "2" * 32,
            "failure_domain_id": "domain_" + "2" * 32,
            "off_experiment_host": not no_off_host_copy,
            "verification_result": "passed",
        },
    )
    retrieval = _typed_receipt(
        kind="retrieval",
        recovery=recovery,
        previous_receipt_id=copy_two["receipt_id"],
        result="passed",
        details={
            "source_copy_id": "copy_" + "2" * 32,
            "source_failure_domain_id": "domain_" + "2" * 32,
            "destination_restore_id": "restore_" + "3" * 32,
            "started_at_utc": "2026-08-08T11:59:59+00:00",
            "finished_at_utc": "2026-08-08T12:00:00+00:00",
            "duration_ns": 1_000_000_000,
            "restored_file_count": recovery["file_count"] + 2,
            "restored_total_bytes": recovery["total_bytes"]
            + recovery["raw_manifest_size_bytes"]
            + len(canonical_json_bytes(recovery["seal"])),
            "expected_raw_manifest_sha256": recovery["raw_manifest_sha256"],
            "observed_raw_manifest_sha256": recovery["raw_manifest_sha256"],
            "mismatch_count": 0,
            "verification_result": "passed",
        },
    )
    if retrieval_inventory_mismatch:
        retrieval["details"]["restored_file_count"] += 1
        without_id = {
            key: value for key, value in retrieval.items() if key != "receipt_id"
        }
        retrieval["receipt_id"] = (
            "receipt_" + sha256_bytes(compact_canonical_json_bytes(without_id))[:32]
        )
    retention = _typed_receipt(
        kind="retention",
        recovery=recovery,
        previous_receipt_id=retrieval["receipt_id"],
        result="active",
        details={
            "retention_policy_id": "cuda-v02-public-claim-evidence-24m-v1",
            "retain_not_before_utc": "2028-08-08T12:00:00+00:00",
            "verification_result": "passed",
        },
    )
    control_receipts = [copy_one, copy_two, retrieval, retention]
    if extra_control_receipt:
        control_receipts.append(
            _typed_receipt(
                kind="retention",
                recovery=recovery,
                previous_receipt_id=retention["receipt_id"],
                result="active",
                details={
                    "retention_policy_id": "cuda-v02-public-claim-evidence-24m-v1",
                    "retain_not_before_utc": "2028-08-08T12:00:00+00:00",
                    "verification_result": "passed",
                },
            )
        )
    _expected, rows = load_expected_digest_manifest(EXPECTED)
    recovered_digest = recovered_entry["sha256"]
    if recovered_binding_mismatch:
        recovered_digest = "0" * 64
    source = {
        "schema_version": RECOVERY_INPUT_SCHEMA,
        "producer_role_id": "phase2-packet-producer",
        "campaign_id": "campaign_" + "c" * 20,
        "original_packet": (
            "docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance"
        ),
        "recovery_raw_manifest": {
            "protected_artifact_id": recovery["protected_artifact_id"],
            "sha256": recovery["raw_manifest_sha256"],
            "size_bytes": recovery["raw_manifest_size_bytes"],
            "entry_id": "entry_recovery_input",
        },
        "recovered_entries": [
            {
                "entry_id": recovered_entry["entry_id"],
                "sha256": recovered_digest,
                "size_bytes": recovered_entry["size_bytes"],
                "logical_source_pointers": [rows[0][1]],
            }
        ],
        "copy_verification_receipts": [
            _safe_receipt(copy_one),
            _safe_receipt(copy_two),
        ],
        "retrieval_receipt": _safe_receipt(retrieval),
        "retention_receipt": _safe_receipt(retention),
        "additional_search_results": [
            {
                "item_id": "python-test-transcript",
                "disposition": "not-found",
                "reason_code": "ORIGINAL_TRANSCRIPT_NOT_FOUND",
                "search_scope_codes": [
                    "source-host-boundary",
                    "verified-copy-one",
                    "verified-copy-two",
                ],
            }
        ],
        "forbidden_private_literals": [
            "example-private-user",
            "example-private-host",
            "10.22.33.44",
        ],
    }
    if unsealed_recovered_input:
        source["recovered_entries"].append(
            {
                "entry_id": "entry_recovered_absent",
                "sha256": "1" * 64,
                "size_bytes": 1,
                "logical_source_pointers": [rows[1][1]],
            }
        )
    if extra_public_receipt:
        source["copy_verification_receipts"].append(
            _safe_receipt(
                _typed_receipt(
                    kind="copy-verification",
                    recovery=recovery,
                    previous_receipt_id=copy_two["receipt_id"],
                    result="passed",
                    details={
                        "copy_id": "copy_" + "3" * 32,
                        "failure_domain_id": "domain_" + "3" * 32,
                        "off_experiment_host": True,
                        "verification_result": "passed",
                    },
                )
            )
        )
    if input_receipt_mismatch:
        source["copy_verification_receipts"][0]["failure_domain_id"] = (
            "domain_" + "9" * 32
        )
    control_path = root / "control"
    control_writer = RawArtifactWriter(
        control_path,
        protected_artifact_id="artifact_" + "b" * 32,
        record_kind="legacy-recovery",
        identity_bindings={"purpose": "recovery-publication-control"},
        capture_tool={"name": "sanitizer-test", "version": "v1"},
        source_bindings={
            "recovery_artifact_id": (
                "artifact_" + "9" * 32
                if control_source_mismatch
                else recovery["protected_artifact_id"]
            )
        },
        provisional_retain_not_before_utc="2028-08-08T12:00:00+00:00",
        required_role_bindings={
            "evidence-receipt": [
                f"entry_receipt_{index:02d}" for index in range(len(control_receipts))
            ],
            "recovery-input": "entry_recovery_input",
            "expected-digest-manifest": "entry_expected_manifest",
        },
    )
    control_writer.write_payload(
        canonical_json_bytes(source),
        "recovery-input.json",
        role="recovery-input",
        media_type="application/json",
        entry_id="entry_recovery_input",
    )
    control_writer.write_payload(
        b"{}\n" if expected_manifest_mismatch else EXPECTED.read_bytes(),
        "expected-digest-manifest.json",
        role="expected-digest-manifest",
        media_type="application/json",
        entry_id="entry_expected_manifest",
    )
    for index, receipt in enumerate(control_receipts):
        receipt_path = f"receipts/{index:02d}-{receipt['kind']}.json"
        if noncanonical_receipt_path and index == 0:
            receipt_path = "receipts/00-copy.json"
        control_writer.write_payload(
            canonical_json_bytes(receipt),
            receipt_path,
            role="evidence-receipt",
            media_type="application/json",
            entry_id=f"entry_receipt_{index:02d}",
        )
    if extra_control_payload:
        control_writer.write_payload(
            b"extraneous\n",
            "extra.txt",
            role="unexpected-control",
            media_type="text/plain",
            entry_id="entry_unexpected_control",
        )
    control_writer.seal()
    return recovery_path, control_path, source


class CudaCampaignSanitizerTests(unittest.TestCase):
    def test_expected_digest_manifest_is_exact_and_preserves_duplicate_rows(
        self,
    ) -> None:
        _manifest, rows = load_expected_digest_manifest(EXPECTED)
        self.assertEqual(EXPECTED_DIGEST_SHA256, EXPECTED_DIGEST_SHA256.lower())
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({row[2] for row in rows}), 39)
        self.assertEqual(len({row[1] for row in rows}), 40)

    def test_projection_derives_all_dispositions_without_replacing_rows(self) -> None:
        projection = project_recovery_supplement(
            expected_digest_manifest_path=EXPECTED,
            protected_input=_protected_input(omit_last=True, mismatch_first=True),
        )
        supplement = projection.supplement
        counts = supplement["summary_counts"]
        self.assertEqual(len(supplement["items"]), 40)
        self.assertEqual(counts["logical_digest_count"], 40)
        self.assertEqual(counts["recovered_mismatched"], 1)
        self.assertEqual(counts["not_found"], 1)
        self.assertEqual(counts["recovered_matching"], 38)
        self.assertEqual(
            counts["logical_digest_count"],
            counts["recovered_matching"]
            + counts["recovered_mismatched"]
            + counts["not_found"],
        )
        self.assertEqual(supplement["independent_review"]["status"], "pending")
        self.assertNotIn("example-private-user", json.dumps(supplement))
        self.assertGreater(len(projection.sanitization_map["entries"]), 40)

    def test_projection_rejects_username_bearing_nonfrozen_original_packet(
        self,
    ) -> None:
        protected_input = _protected_input()
        protected_input["original_packet"] = (
            "users/example-private-user/private/"
            "2026-08-06-smollm2-cuda-lora-single-acceptance"
        )

        with self.assertRaisesRegex(SanitizationError, "frozen Phase 0"):
            project_recovery_supplement(
                expected_digest_manifest_path=EXPECTED,
                protected_input=protected_input,
            )

    def test_projection_rejects_unknown_private_input_fields(self) -> None:
        source = _protected_input()
        source["raw_log"] = "not public"
        with self.assertRaisesRegex(SanitizationError, "fields are not exact"):
            project_recovery_supplement(
                expected_digest_manifest_path=EXPECTED, protected_input=source
            )

    def test_projection_rejects_contextually_invalid_evidence_receipts(self) -> None:
        mutations = (
            lambda source: source["copy_verification_receipts"][0].update(
                kind="retention"
            ),
            lambda source: source["copy_verification_receipts"][0].update(
                failure_domain_id=None
            ),
            lambda source: source["retrieval_receipt"].update(result="failed"),
            lambda source: source["retention_receipt"].update(kind="retrieval"),
            lambda source: source["retention_receipt"].update(
                protected_artifact_id="artifact_" + "9" * 32
            ),
        )
        for mutation in mutations:
            source = _protected_input()
            mutation(source)
            with self.subTest(source=source), self.assertRaises(SanitizationError):
                project_recovery_supplement(
                    expected_digest_manifest_path=EXPECTED,
                    protected_input=source,
                )

    def test_projection_rejects_recovered_pointer_outside_frozen_manifest(self) -> None:
        source = _protected_input()
        source["recovered_entries"][0]["logical_source_pointers"].append(
            "/unfrozen/private-pointer"
        )
        with self.assertRaisesRegex(SanitizationError, "outside the frozen"):
            project_recovery_supplement(
                expected_digest_manifest_path=EXPECTED,
                protected_input=source,
            )

    def test_private_pattern_scan_rejects_all_high_risk_classes(self) -> None:
        unsafe_values = (
            "/Users/private/project/file.json",
            "/mnt/private-vault/evidence.json",
            "/media/custodian/raw.json",
            "/run/user/1000/job-state.json",
            r"C:\\Users\\private\\file.json",
            "name@example.com",
            "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "550e8400-e29b-41d4-a716-446655440000",
            "serial_number=ABC12345",
            "hostname=private-machine",
            "Authorization: Bearer secret-token-value",
            "HOME=/private/home",
            "192.168.50.12",
        )
        for unsafe in unsafe_values:
            with self.subTest(unsafe=unsafe), self.assertRaises(SanitizationError):
                scan_public_value({"safe_field": unsafe})
        with self.assertRaisesRegex(SanitizationError, "Protected public field"):
            scan_public_value({"pid": 123})
        with self.assertRaisesRegex(SanitizationError, "Private literal"):
            scan_public_value(
                {"safe_field": "private-machine-name"},
                forbidden_literals=["private-machine-name"],
            )

    def test_public_reasons_are_fixed_and_arbitrary_text_is_rejected(self) -> None:
        self.assertLessEqual(len(stable_reason("RECOVERED_MATCH")), 240)
        with self.assertRaisesRegex(SanitizationError, "no public template"):
            stable_reason("raw exception text")

    def test_verified_stage_review_and_sealed_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery, control, source = _sealed_context(root)
            context = load_verified_recovery_context(
                recovery_artifact=recovery, control_artifact=control
            )
            self.assertEqual(
                context.recovery_verification["protected_artifact_id"],
                source["recovery_raw_manifest"]["protected_artifact_id"],
            )
            projection = project_verified_recovery_supplement(
                recovery_artifact=recovery, control_artifact=control
            )
            self.assertEqual(
                projection.supplement["summary_counts"]["recovered_mismatched"], 1
            )
            self.assertEqual(projection.supplement["summary_counts"]["not_found"], 39)
            public_receipt = projection.supplement["copy_verification_receipts"][0]
            receipt_entry = context.control_entries[
                context.receipt_entry_ids[public_receipt["receipt_id"]]
            ]
            self.assertEqual(public_receipt["sha256"], receipt_entry["sha256"])
            self.assertEqual(public_receipt["size_bytes"], receipt_entry["size_bytes"])
            self.assertNotEqual(
                public_receipt["sha256"],
                context.recovery_verification["raw_manifest_sha256"],
            )
            stage = root / "stage"
            digests = write_projection_stage(
                stage,
                projection,
                forbidden_literals=context.forbidden_literals,
            )
            self.assertEqual(
                list(digests),
                [
                    "claim-boundary.json",
                    "recovery-supplement.json",
                    "sanitization-map.json",
                ],
            )
            with self.assertRaises(FileExistsError):
                write_projection_stage(stage, projection)
            with self.assertRaisesRegex(SanitizationError, "must differ"):
                verify_projection_stage(
                    stage,
                    recovery_artifact=recovery,
                    control_artifact=control,
                    producer_role_id="phase2-packet-producer",
                    reviewer_role_id="phase2-packet-producer",
                )
            review = verify_projection_stage(
                stage,
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id="phase2-packet-producer",
                reviewer_role_id="phase2-independent-reviewer",
            )
            self.assertEqual(review["result"], "passed")
            reviewed_at = _after_seals(recovery, control)
            sealed_review = seal_projection_review(
                stage,
                root / "sealed-review",
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id="phase2-packet-producer",
                reviewer_role_id="phase2-independent-reviewer",
                review_id="review_" + "4" * 32,
                reviewed_at_utc=reviewed_at,
            )
            finalized_at = _after_timestamp(reviewed_at)
            finalized = finalize_projection_stage(
                stage,
                root / "final-public",
                root / "sealed-review",
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id="phase2-packet-producer",
                reviewer_role_id="phase2-independent-reviewer",
                finalizer_role_id="phase2-candidate-finalizer",
                finalized_at_utc=finalized_at,
            )
            self.assertEqual(finalized["review"]["result"], "passed")
            self.assertEqual(
                json.loads(
                    (root / "final-public/independent-review.json").read_text(
                        encoding="utf-8"
                    )
                )["result"],
                "passed",
            )
            self.assertEqual(
                {path.name for path in (root / "final-public").iterdir()},
                {
                    "SHA256SUMS",
                    "claim-boundary.json",
                    "finalization.json",
                    "independent-review.json",
                    "recovery-supplement.json",
                    "review-bindings.json",
                    "sanitization-map.json",
                },
            )
            verified = verify_finalized_projection(
                stage,
                root / "final-public",
                root / "sealed-review",
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id="phase2-packet-producer",
                reviewer_role_id="phase2-independent-reviewer",
                finalizer_role_id="phase2-candidate-finalizer",
            )
            self.assertEqual(
                verified["finalization_id"],
                finalized["finalization"]["finalization_id"],
            )
            self.assertEqual(
                verified["review_id"], sealed_review["review"]["review_id"]
            )
            second = finalize_projection_stage(
                stage,
                root / "final-public-two",
                root / "sealed-review",
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id="phase2-packet-producer",
                reviewer_role_id="phase2-independent-reviewer",
                finalizer_role_id="phase2-candidate-finalizer",
                finalized_at_utc=finalized_at,
            )
            self.assertEqual(
                second["sealed_review_artifact"]["raw_manifest_sha256"],
                finalized["sealed_review_artifact"]["raw_manifest_sha256"],
            )
            with self.assertRaises(FileExistsError):
                finalize_projection_stage(
                    stage,
                    root / "final-public",
                    root / "sealed-review",
                    recovery_artifact=recovery,
                    control_artifact=control,
                    producer_role_id="phase2-packet-producer",
                    reviewer_role_id="phase2-independent-reviewer",
                    finalizer_role_id="phase2-candidate-finalizer",
                    finalized_at_utc=finalized_at,
                )

    def test_production_review_rejects_test_only_projection_and_public_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery, control, _source = _sealed_context(root)
            projection = project_verified_recovery_supplement(
                recovery_artifact=recovery, control_artifact=control
            )
            stage = root / "stage"
            write_projection_stage(stage, projection)
            supplement_path = stage / "recovery-supplement.json"
            supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
            supplement["summary_counts"]["recovered_matching"] -= 1
            supplement["summary_counts"]["not_found"] += 1
            supplement_path.write_text(
                json.dumps(supplement, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(SanitizationError):
                verify_projection_stage(
                    stage,
                    recovery_artifact=recovery,
                    control_artifact=control,
                    producer_role_id="phase2-packet-producer",
                    reviewer_role_id="phase2-independent-reviewer",
                )

            plain_stage = root / "plain-stage"
            write_projection_stage(
                plain_stage,
                project_recovery_supplement(
                    expected_digest_manifest_path=EXPECTED,
                    protected_input=_protected_input(),
                ),
            )
            with self.assertRaises(SanitizationError):
                verify_projection_stage(
                    plain_stage,
                    recovery_artifact=recovery,
                    control_artifact=control,
                    producer_role_id="phase2-packet-producer",
                    reviewer_role_id="phase2-independent-reviewer",
                )

    def test_finalization_requires_durable_prior_review_and_role_chronology(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery, control, _source = _sealed_context(root)
            stage = root / "stage"
            write_projection_stage(
                stage,
                project_verified_recovery_supplement(
                    recovery_artifact=recovery, control_artifact=control
                ),
            )
            with self.assertRaisesRegex(SanitizationError, "prior review"):
                finalize_projection_stage(
                    stage,
                    root / "candidate-without-review",
                    root / "missing-review",
                    recovery_artifact=recovery,
                    control_artifact=control,
                    producer_role_id="phase2-packet-producer",
                    reviewer_role_id="phase2-independent-reviewer",
                    finalizer_role_id="phase2-candidate-finalizer",
                    finalized_at_utc=_after_seals(recovery, control, seconds=2),
                )
            with self.assertRaisesRegex(SanitizationError, "cannot predate"):
                seal_projection_review(
                    stage,
                    root / "backdated-review",
                    recovery_artifact=recovery,
                    control_artifact=control,
                    producer_role_id="phase2-packet-producer",
                    reviewer_role_id="phase2-independent-reviewer",
                    review_id="review_" + "8" * 32,
                    reviewed_at_utc="2000-01-01T00:00:00+00:00",
                )

            reviewed_at = _after_seals(recovery, control)
            review_artifact = root / "sealed-review"
            seal_projection_review(
                stage,
                review_artifact,
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id="phase2-packet-producer",
                reviewer_role_id="phase2-independent-reviewer",
                review_id="review_" + "9" * 32,
                reviewed_at_utc=reviewed_at,
            )
            with self.assertRaisesRegex(SanitizationError, "distinct"):
                finalize_projection_stage(
                    stage,
                    root / "same-role-candidate",
                    review_artifact,
                    recovery_artifact=recovery,
                    control_artifact=control,
                    producer_role_id="phase2-packet-producer",
                    reviewer_role_id="phase2-independent-reviewer",
                    finalizer_role_id="phase2-independent-reviewer",
                    finalized_at_utc=_after_timestamp(reviewed_at),
                )
            with self.assertRaisesRegex(SanitizationError, "cannot predate"):
                finalize_projection_stage(
                    stage,
                    root / "backdated-finalization",
                    review_artifact,
                    recovery_artifact=recovery,
                    control_artifact=control,
                    producer_role_id="phase2-packet-producer",
                    reviewer_role_id="phase2-independent-reviewer",
                    finalizer_role_id="phase2-candidate-finalizer",
                    finalized_at_utc="2000-01-01T00:00:00+00:00",
                )

    def test_verified_context_rejects_unbound_bytes_and_receipts(self) -> None:
        for keyword in (
            "recovered_binding_mismatch",
            "input_receipt_mismatch",
            "receipt_id_mismatch",
            "expected_manifest_mismatch",
            "extra_control_payload",
            "no_off_host_copy",
        ):
            with (
                self.subTest(keyword=keyword),
                tempfile.TemporaryDirectory() as temporary,
            ):
                recovery, control, _source = _sealed_context(
                    Path(temporary), **{keyword: True}
                )
                with self.assertRaises(SanitizationError):
                    load_verified_recovery_context(
                        recovery_artifact=recovery, control_artifact=control
                    )

        with tempfile.TemporaryDirectory() as temporary:
            recovery, control, _source = _sealed_context(Path(temporary))
            recovered_file = recovery / "recovered/item.bin"
            recovered_file.write_bytes(b"tampered\n")
            recovered_file.chmod(0o600)
            with self.assertRaises(SanitizationError):
                load_verified_recovery_context(
                    recovery_artifact=recovery, control_artifact=control
                )

    def test_verified_context_requires_bidirectional_recovery_inventory(self) -> None:
        cases = (
            (
                "extra_recovery_payload",
                "sealed recovery payload inventories differ",
            ),
            (
                "unsealed_recovered_input",
                "sealed recovery payload inventories differ",
            ),
        )
        for keyword, message in cases:
            with (
                self.subTest(keyword=keyword),
                tempfile.TemporaryDirectory() as temporary,
            ):
                recovery, control, _source = _sealed_context(
                    Path(temporary), **{keyword: True}
                )
                with self.assertRaisesRegex(SanitizationError, message):
                    load_verified_recovery_context(
                        recovery_artifact=recovery, control_artifact=control
                    )

    def test_verified_context_requires_exact_recovery_roles_and_bindings(self) -> None:
        cases = (
            (
                "wrong_recovery_payload_role",
                "non-recovery payload role",
            ),
            (
                "missing_recovery_role_binding",
                "required-role inventory is not exact",
            ),
        )
        for keyword, message in cases:
            with (
                self.subTest(keyword=keyword),
                tempfile.TemporaryDirectory() as temporary,
            ):
                recovery, control, _source = _sealed_context(
                    Path(temporary), **{keyword: True}
                )
                with self.assertRaisesRegex(SanitizationError, message):
                    load_verified_recovery_context(
                        recovery_artifact=recovery, control_artifact=control
                    )

    def test_verified_context_requires_complete_canonical_receipt_inventory(
        self,
    ) -> None:
        cases = (
            ("extra_control_receipt", "extraneous typed receipt"),
            ("extra_public_receipt", "absent from sealed controls"),
            ("noncanonical_receipt_path", "canonically ordered"),
        )
        for keyword, message in cases:
            with (
                self.subTest(keyword=keyword),
                tempfile.TemporaryDirectory() as temporary,
            ):
                recovery, control, _source = _sealed_context(
                    Path(temporary), **{keyword: True}
                )
                with self.assertRaisesRegex(SanitizationError, message):
                    load_verified_recovery_context(
                        recovery_artifact=recovery, control_artifact=control
                    )

    def test_verified_context_binds_retrieval_inventory_and_control_source(
        self,
    ) -> None:
        cases = (
            ("retrieval_inventory_mismatch", "complete sealed recovery artifact"),
            ("control_source_mismatch", "source binding is not exact"),
        )
        for keyword, message in cases:
            with (
                self.subTest(keyword=keyword),
                tempfile.TemporaryDirectory() as temporary,
            ):
                recovery, control, _source = _sealed_context(
                    Path(temporary), **{keyword: True}
                )
                with self.assertRaisesRegex(SanitizationError, message):
                    load_verified_recovery_context(
                        recovery_artifact=recovery, control_artifact=control
                    )

    def test_stage_rejects_extra_directories_hardlinks_and_checksum_mode(self) -> None:
        def prepared(root: Path) -> tuple[Path, Path, Path]:
            recovery, control, _source = _sealed_context(root)
            stage = root / "stage"
            write_projection_stage(
                stage,
                project_verified_recovery_supplement(
                    recovery_artifact=recovery, control_artifact=control
                ),
            )
            return stage, recovery, control

        def verify(stage: Path, recovery: Path, control: Path) -> None:
            verify_projection_stage(
                stage,
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id="phase2-packet-producer",
                reviewer_role_id="phase2-independent-reviewer",
            )

        with tempfile.TemporaryDirectory() as temporary:
            stage, recovery, control = prepared(Path(temporary))
            (stage / "unexpected").mkdir(mode=0o700)
            with self.assertRaises(SanitizationError):
                verify(stage, recovery, control)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage, recovery, control = prepared(root)
            original = stage / "claim-boundary.json"
            hardlink_source = root / "hardlink-source.json"
            os.link(original, hardlink_source)
            with self.assertRaises(SanitizationError):
                verify(stage, recovery, control)

        with tempfile.TemporaryDirectory() as temporary:
            stage, recovery, control = prepared(Path(temporary))
            (stage / "SHA256SUMS").chmod(0o644)
            with self.assertRaises(SanitizationError):
                verify(stage, recovery, control)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage, recovery, control = prepared(root)
            target = stage / "claim-boundary.json"
            outside = root / "outside-claim-boundary.json"
            target.replace(outside)
            target.symlink_to(outside)
            with self.assertRaises(SanitizationError):
                verify(stage, recovery, control)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage, recovery, control = prepared(root)
            target = stage / "claim-boundary.json"
            replacement = root / "replacement-claim-boundary.json"
            replacement.write_bytes(target.read_bytes())
            replacement.chmod(0o600)
            displaced = root / "displaced-claim-boundary.json"
            real_open = os.open
            swapped = False

            def swap_before_open(path: object, flags: int, *args: object) -> int:
                nonlocal swapped
                if not swapped and os.fspath(path) == os.fspath(target):
                    os.replace(target, displaced)
                    os.replace(replacement, target)
                    swapped = True
                return real_open(path, flags, *args)

            with (
                patch(
                    "tools.cuda_campaign.sanitizer.os.open",
                    side_effect=swap_before_open,
                ),
                self.assertRaisesRegex(SanitizationError, "changed while opening"),
            ):
                verify(stage, recovery, control)

    def test_finalized_review_detects_later_stage_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery, control, _source = _sealed_context(root)
            stage = root / "stage"
            write_projection_stage(
                stage,
                project_verified_recovery_supplement(
                    recovery_artifact=recovery, control_artifact=control
                ),
            )
            review_artifact = root / "sealed-review"
            reviewed_at = _after_seals(recovery, control)
            seal_projection_review(
                stage,
                review_artifact,
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id="phase2-packet-producer",
                reviewer_role_id="phase2-independent-reviewer",
                review_id="review_" + "5" * 32,
                reviewed_at_utc=reviewed_at,
            )
            finalize_projection_stage(
                stage,
                root / "final-public",
                review_artifact,
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id="phase2-packet-producer",
                reviewer_role_id="phase2-independent-reviewer",
                finalizer_role_id="phase2-candidate-finalizer",
                finalized_at_utc=_after_timestamp(reviewed_at),
            )
            sums = stage / "SHA256SUMS"
            sums.write_bytes(sums.read_bytes() + b"\n")
            sums.chmod(0o600)
            with self.assertRaises(SanitizationError):
                verify_finalized_projection(
                    stage,
                    root / "final-public",
                    review_artifact,
                    recovery_artifact=recovery,
                    control_artifact=control,
                    producer_role_id="phase2-packet-producer",
                    reviewer_role_id="phase2-independent-reviewer",
                    finalizer_role_id="phase2-candidate-finalizer",
                )


if __name__ == "__main__":
    unittest.main()
