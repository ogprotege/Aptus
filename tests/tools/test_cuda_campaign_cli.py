from __future__ import annotations

import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.cuda_campaign import cli as campaign_cli
from tools.cuda_campaign import eligibility, sanitizer
from tools.cuda_campaign.cli import OperatorCliError, main
from tools.cuda_campaign.contracts import canonical_json_bytes, sha256_bytes
from tools.cuda_campaign.storage import (
    AppendOnlyReceiptStore,
    ArtifactIntegrityError,
    EvidenceStorageError,
    RetrievalError,
)

HOST_ID = "host_" + "1" * 32
ARTIFACT_ID = "artifact_" + "2" * 32
COPY_ID = "copy_" + "3" * 32
DOMAIN_ID = "domain_" + "4" * 32
RESTORE_ID = "restore_" + "5" * 32
SLOT_ID = "slot_" + "6" * 20
XRUN_ID = "xrun_" + "7" * 32
DIGEST = "a" * 64
REVIEW_ID = "review_" + "8" * 32
REVIEWED_AT = "2026-08-08T12:00:00+00:00"
COPY_OPERATION_ID = "operation_" + "9" * 32
RETRIEVE_OPERATION_ID = "operation_" + "a" * 32
SECOND_COPY_OPERATION_ID = "operation_" + "b" * 32
SECOND_RETRIEVE_OPERATION_ID = "operation_" + "c" * 32
RETRIEVAL_INTENT_AT = "2026-08-08T11:59:59+00:00"
RETRIEVAL_RECORDED_AT = "2026-08-08T12:00:02+00:00"


def invoke(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = main(argv)
    return result, stdout.getvalue(), stderr.getvalue()


def retrieval_clock():
    return patch(
        "tools.cuda_campaign.cli.utc_now",
        side_effect=[RETRIEVAL_INTENT_AT, RETRIEVAL_RECORDED_AT],
    )


def private_directory(parent: Path, name: str) -> Path:
    path = parent / name
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def probe_snapshot() -> dict[str, object]:
    return {
        "gpu": {
            "uuid": "GPU-private-uuid-value",
            "memory_used": {"value": "1024", "unit": "MiB"},
            "memory_free": {"value": "7168", "unit": "MiB"},
            "memory_total": {"value": "8192", "unit": "MiB"},
            "utilization_percent": 0.0,
            "temperature_c": 40.0,
            "power_draw_w": 12.5,
            "power_limit_w": 130.0,
            "graphics_clock_mhz": 210.0,
            "memory_clock_mhz": 405.0,
            "performance_state": "P8",
            "throttle_reasons": [],
            "compute_processes": [{"pid": 12345, "managed": False}],
        },
        "host": {
            "mem_available_bytes": 60 * 1024**3,
            "swap_used_bytes": 0,
            "swap_read_bytes": 0,
            "swap_write_bytes": 0,
            "load_1m": 0.1,
            "filesystem_free_bytes": 200 * 1024**3,
            "managed_process_rss_bytes": 0,
            "managed_process_cpu_seconds": 0.0,
            "managed_process_read_bytes": 0,
            "managed_process_write_bytes": 0,
            "disk_growth_bytes": 0,
            "aptus_lease_active": False,
            "cpu_temperature": {"status": "unsupported", "value": None},
            "nvme_temperature": {"status": "unsupported", "value": None},
        },
    }


def sealed_result() -> dict[str, object]:
    return {
        "protected_artifact_id": ARTIFACT_ID,
        "raw_manifest_sha256": DIGEST,
        "raw_manifest_size_bytes": 100,
        "file_count": 3,
        "total_bytes": 200,
        "verification_result": "passed",
        "manifest": {"private_path": "/Users/private/vault"},
        "seal": {"private": True},
    }


def equality_result(
    *, stored_file_count: int = 5, stored_total_bytes: int = 300
) -> dict[str, object]:
    return {
        **sealed_result(),
        "stored_file_count": stored_file_count,
        "stored_total_bytes": stored_total_bytes,
    }


class CudaCampaignCliTests(unittest.TestCase):
    def test_procedural_role_id_grammar_is_shared_without_normalization(self) -> None:
        self.assertIs(campaign_cli.PROCEDURAL_ROLE_ID_RE, eligibility._ROLE_ID)
        self.assertIs(campaign_cli.PROCEDURAL_ROLE_ID_RE, sanitizer._ROLE_ID)
        for value in (
            "evidence-custodian",
            "candidate_producer",
            "a1-b2_c3-d4",
        ):
            self.assertEqual(campaign_cli._require_role_id(value), value)
            self.assertIsNotNone(eligibility._ROLE_ID.fullmatch(value))
            self.assertIsNotNone(sanitizer._ROLE_ID.fullmatch(value))
        for value in ("a", "A-role", "role__id", "role.id", "r" * 65):
            with self.assertRaises(OperatorCliError):
                campaign_cli._require_role_id(value)
            self.assertIsNone(eligibility._ROLE_ID.fullmatch(value))
            self.assertIsNone(sanitizer._ROLE_ID.fullmatch(value))

    def test_operation_journal_rejects_links_path_swaps_and_invalid_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            symlink_root = private_directory(root, "symlink-journal")
            symlink_target = symlink_root / "target.json"
            symlink_target.write_bytes(b"{}\n")
            symlink_target.chmod(0o600)
            (symlink_root / f"{COPY_OPERATION_ID}.intent.json").symlink_to(
                symlink_target
            )
            with self.assertRaisesRegex(OperatorCliError, "RECEIPT_JOURNAL_INVALID"):
                campaign_cli._load_receipt_journal(symlink_root, COPY_OPERATION_ID)

            hardlink_root = private_directory(root, "hardlink-journal")
            hardlink_target = hardlink_root / "target.json"
            hardlink_target.write_bytes(b"{}\n")
            hardlink_target.chmod(0o600)
            os.link(
                hardlink_target,
                hardlink_root / f"{COPY_OPERATION_ID}.intent.json",
            )
            with self.assertRaisesRegex(OperatorCliError, "RECEIPT_JOURNAL_INVALID"):
                campaign_cli._load_receipt_journal(hardlink_root, COPY_OPERATION_ID)

            swap_root = private_directory(root, "swap-journal")
            victim = swap_root / f"{COPY_OPERATION_ID}.intent.json"
            replacement = swap_root / "replacement.json"
            held = swap_root / "held.json"
            victim.write_bytes(b"{}\n")
            replacement.write_bytes(b"{}\n")
            victim.chmod(0o600)
            replacement.chmod(0o600)
            real_open = os.open
            swapped = False

            def swap_then_open(target: object, flags: int, mode: int = 0o777) -> int:
                nonlocal swapped
                if Path(target) == victim and not swapped:
                    swapped = True
                    victim.rename(held)
                    replacement.rename(victim)
                return real_open(target, flags, mode)

            with patch("tools.cuda_campaign.cli.os.open", side_effect=swap_then_open):
                with self.assertRaisesRegex(
                    OperatorCliError, "RECEIPT_JOURNAL_INVALID"
                ):
                    campaign_cli._load_receipt_journal(swap_root, COPY_OPERATION_ID)

            key_root = private_directory(root, "invalid-key-journal")
            key = key_root / ".operation-path-key"
            key.write_bytes(b"not-a-valid-key")
            key.chmod(0o600)
            with self.assertRaisesRegex(OperatorCliError, "RECEIPT_JOURNAL_INVALID"):
                campaign_cli._path_binding(key_root, root / "protected")
            self.assertEqual(key.read_bytes(), b"not-a-valid-key")

    def test_retrieval_outcome_is_exactly_bound_to_its_intent_and_time(self) -> None:
        intent = campaign_cli._validate_operation_intent(
            {
                "format_version": campaign_cli._OPERATION_INTENT_FORMAT,
                "operation_id": RETRIEVE_OPERATION_ID,
                "operation_kind": "retrieve",
                "created_at_utc": RETRIEVAL_INTENT_AT,
                "issuer_role_id": "evidence-custodian",
                "protected_artifact_id": ARTIFACT_ID,
                "raw_manifest_sha256": DIGEST,
                "raw_manifest_size_bytes": 100,
                "expected_raw_manifest_sha256": DIGEST,
                "source_copy_id": COPY_ID,
                "source_failure_domain_id": DOMAIN_ID,
                "destination_copy_id": None,
                "destination_failure_domain_id": None,
                "destination_restore_id": RESTORE_ID,
                "off_experiment_host": None,
                "source_path_binding": "b" * 64,
                "destination_path_binding": "c" * 64,
                "receipt_store_path_binding": "d" * 64,
                "receipt_tail_id": None,
                "receipt_chain_sha256": "e" * 64,
            }
        )
        outcome = {
            "format_version": campaign_cli._OPERATION_OUTCOME_FORMAT,
            "operation_id": RETRIEVE_OPERATION_ID,
            "intent_sha256": sha256_bytes(canonical_json_bytes(intent)),
            "recorded_at_utc": RETRIEVAL_RECORDED_AT,
            "receipt_kind": "retrieval",
            "receipt_created_at_utc": "2026-08-08T12:00:01+00:00",
            "result": "passed",
            "details": {
                "source_copy_id": COPY_ID,
                "source_failure_domain_id": DOMAIN_ID,
                "destination_restore_id": RESTORE_ID,
                "started_at_utc": "2026-08-08T12:00:00+00:00",
                "finished_at_utc": "2026-08-08T12:00:01+00:00",
                "duration_ns": 1_000_000_000,
                "restored_file_count": 5,
                "restored_total_bytes": 300,
                "expected_raw_manifest_sha256": DIGEST,
                "observed_raw_manifest_sha256": DIGEST,
                "mismatch_count": 0,
                "verification_result": "passed",
            },
        }
        self.assertEqual(
            campaign_cli._validate_operation_outcome(outcome, intent=intent), outcome
        )
        mutations = {
            "copy": ("source_copy_id", "copy_" + "f" * 32),
            "domain": ("source_failure_domain_id", "domain_" + "f" * 32),
            "restore": ("destination_restore_id", "restore_" + "f" * 32),
            "digest": ("expected_raw_manifest_sha256", "f" * 64),
        }
        for label, (field, changed_value) in mutations.items():
            with self.subTest(label=label):
                changed = json.loads(json.dumps(outcome))
                changed["details"][field] = changed_value
                if field == "expected_raw_manifest_sha256":
                    changed["details"]["observed_raw_manifest_sha256"] = changed_value
                with self.assertRaisesRegex(
                    OperatorCliError, "RECEIPT_JOURNAL_INVALID"
                ):
                    campaign_cli._validate_operation_outcome(changed, intent=intent)
        predating = json.loads(json.dumps(outcome))
        predating["details"]["started_at_utc"] = "2026-08-08T11:00:00+00:00"
        predating["details"]["finished_at_utc"] = "2026-08-08T11:00:01+00:00"
        predating["receipt_created_at_utc"] = "2026-08-08T11:00:01+00:00"
        with self.assertRaisesRegex(OperatorCliError, "RECEIPT_JOURNAL_INVALID"):
            campaign_cli._validate_operation_outcome(predating, intent=intent)
        with (
            patch(
                "tools.cuda_campaign.cli.verify_copy_equality",
                return_value=equality_result(stored_file_count=4),
            ),
            self.assertRaisesRegex(
                OperatorCliError, "OPERATION_RECONCILIATION_REQUIRED"
            ),
        ):
            campaign_cli._revalidate_passing_outcome_destination(
                intent,
                outcome,
                source=Path("/protected/source"),
                destination=Path("/protected/restore"),
            )

    def test_probe_outputs_only_bounded_nonqualifying_summary(self) -> None:
        with patch("tools.cuda_campaign.cli.LinuxNvidiaHostProbe") as probe:
            probe.return_value.return_value = probe_snapshot()
            code, stdout, stderr = invoke(
                [
                    "probe",
                    "--host-id",
                    HOST_ID,
                    "--filesystem-path",
                    "/private/filesystem/path",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        result = json.loads(stdout)
        self.assertEqual(
            result["qualification_status"], "nonqualifying-read-only-probe"
        )
        self.assertEqual(result["host_id"], HOST_ID)
        self.assertEqual(result["gpu"]["compute_process_count"], 1)
        self.assertNotIn("GPU-private", stdout)
        self.assertNotIn("12345", stdout)
        self.assertNotIn("/private", stdout)

    def test_verify_seal_requires_private_root_and_hides_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = private_directory(root, "private-artifact")
            with patch(
                "tools.cuda_campaign.cli.verify_sealed_artifact",
                return_value=sealed_result(),
            ):
                code, stdout, stderr = invoke(
                    [
                        "verify-seal",
                        "--artifact",
                        str(artifact),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                    ]
                )
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["verification_result"], "passed")
        self.assertNotIn("private_path", stdout)
        self.assertNotIn("/Users", stdout)

    def test_copy_and_retrieve_pass_exact_ids_without_printing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = private_directory(root, "source-secret")
            receipt_root = private_directory(root, "receipt-secret")
            journal_root = private_directory(root, "journal-secret")
            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.copy_sealed_artifact",
                    return_value=sealed_result(),
                ) as copy,
            ):
                code, stdout, stderr = invoke(
                    [
                        "copy-seal",
                        "--operation-id",
                        COPY_OPERATION_ID,
                        "--source",
                        str(source),
                        "--destination",
                        str(root / "copy-secret"),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                        "--copy-id",
                        COPY_ID,
                        "--failure-domain-id",
                        DOMAIN_ID,
                        "--receipt-store",
                        str(receipt_root),
                        "--receipt-journal",
                        str(journal_root),
                        "--issuer-role-id",
                        "evidence-custodian",
                        "--off-experiment-host",
                    ]
                )
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(copy.call_count, 1)
            self.assertEqual(
                copy.call_args.args,
                (source.resolve(), (root / "copy-secret").resolve()),
            )
            self.assertNotIn(str(root), stdout)
            copy_summary = json.loads(stdout)
            self.assertEqual(copy_summary["copy_id"], COPY_ID)
            self.assertTrue(copy_summary["off_experiment_host"])
            copy_receipt = AppendOnlyReceiptStore(receipt_root).read_chain()[0]
            self.assertEqual(copy_summary["receipt_id"], copy_receipt["receipt_id"])
            self.assertEqual(
                copy_receipt["details"],
                {
                    "copy_id": COPY_ID,
                    "failure_domain_id": DOMAIN_ID,
                    "off_experiment_host": True,
                    "verification_result": "passed",
                },
            )
            self.assertEqual(copy_receipt["raw_manifest_sha256"], DIGEST)
            self.assertEqual(copy_receipt["raw_manifest_size_bytes"], 100)

            retrieval = {
                "source_copy_id": COPY_ID,
                "source_failure_domain_id": DOMAIN_ID,
                "destination_restore_id": RESTORE_ID,
                "started_at_utc": "2026-08-08T12:00:00+00:00",
                "finished_at_utc": "2026-08-08T12:00:01+00:00",
                "duration_ns": 12,
                "restored_file_count": 5,
                "restored_total_bytes": 300,
                "expected_raw_manifest_sha256": DIGEST,
                "observed_raw_manifest_sha256": DIGEST,
                "mismatch_count": 0,
                "verification_result": "passed",
            }
            with (
                retrieval_clock(),
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.retrieve_sealed_artifact",
                    return_value=retrieval,
                ) as retrieve,
            ):
                code, stdout, stderr = invoke(
                    [
                        "retrieve",
                        "--operation-id",
                        RETRIEVE_OPERATION_ID,
                        "--source",
                        str(source),
                        "--destination",
                        str(root / "restore-secret"),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                        "--source-copy-id",
                        COPY_ID,
                        "--source-failure-domain-id",
                        DOMAIN_ID,
                        "--expected-raw-manifest-sha256",
                        DIGEST,
                        "--destination-restore-id",
                        RESTORE_ID,
                        "--receipt-store",
                        str(receipt_root),
                        "--receipt-journal",
                        str(journal_root),
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(
                retrieve.call_args.args,
                (source.resolve(), (root / "restore-secret").resolve()),
            )
            self.assertEqual(
                retrieve.call_args.kwargs,
                {
                    "source_copy_id": COPY_ID,
                    "source_failure_domain_id": DOMAIN_ID,
                    "expected_raw_manifest_sha256": DIGEST,
                    "destination_restore_id": RESTORE_ID,
                },
            )
            self.assertNotIn(str(root), stdout)
            retrieval_receipt = AppendOnlyReceiptStore(receipt_root).read_chain()[1]
            self.assertEqual(retrieval_receipt["details"], retrieval)
            self.assertEqual(retrieval_receipt["result"], "passed")
            self.assertEqual(
                json.loads(stdout)["receipt_id"], retrieval_receipt["receipt_id"]
            )

    def test_failed_retrieval_appends_exact_failed_receipt_without_disclosure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = private_directory(root, "off-host-secret")
            receipt_root = private_directory(root, "receipt-secret")
            journal_root = private_directory(root, "journal-secret")
            store = AppendOnlyReceiptStore(receipt_root)
            store.append(
                kind="copy-verification",
                issuer_role_id="evidence-custodian",
                protected_artifact_id=ARTIFACT_ID,
                raw_manifest_sha256=DIGEST,
                raw_manifest_size_bytes=100,
                result="passed",
                details={
                    "copy_id": COPY_ID,
                    "failure_domain_id": DOMAIN_ID,
                    "off_experiment_host": True,
                    "verification_result": "passed",
                },
            )
            failure_details = {
                "source_copy_id": COPY_ID,
                "source_failure_domain_id": DOMAIN_ID,
                "destination_restore_id": RESTORE_ID,
                "started_at_utc": "2026-08-08T12:00:00+00:00",
                "finished_at_utc": "2026-08-08T12:00:01+00:00",
                "duration_ns": 12,
                "restored_file_count": 0,
                "restored_total_bytes": 0,
                "expected_raw_manifest_sha256": DIGEST,
                "observed_raw_manifest_sha256": None,
                "mismatch_count": 1,
                "verification_result": "failed",
            }
            with (
                retrieval_clock(),
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.retrieve_sealed_artifact",
                    side_effect=RetrievalError(
                        "private failure at /Users/private/raw.log",
                        details=failure_details,
                    ),
                ),
            ):
                code, stdout, stderr = invoke(
                    [
                        "retrieve",
                        "--operation-id",
                        RETRIEVE_OPERATION_ID,
                        "--source",
                        str(source),
                        "--destination",
                        str(root / "restore-secret"),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                        "--source-copy-id",
                        COPY_ID,
                        "--source-failure-domain-id",
                        DOMAIN_ID,
                        "--expected-raw-manifest-sha256",
                        DIGEST,
                        "--destination-restore-id",
                        RESTORE_ID,
                        "--receipt-store",
                        str(receipt_root),
                        "--receipt-journal",
                        str(journal_root),
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stdout), (5, ""))
            self.assertEqual(
                json.loads(stderr),
                {"error_code": "RETRIEVAL_FAILURE", "ok": False},
            )
            self.assertNotIn(str(root), stderr)
            receipt = AppendOnlyReceiptStore(receipt_root).read_chain()[-1]
            self.assertEqual(receipt["result"], "failed")
            self.assertEqual(receipt["details"], failure_details)
            self.assertEqual(receipt["raw_manifest_sha256"], DIGEST)
            self.assertEqual(receipt["raw_manifest_size_bytes"], 100)

    def test_copy_receipt_append_failure_resumes_without_recopy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = private_directory(root, "source-secret")
            receipts = private_directory(root, "receipts-secret")
            journals = private_directory(root, "journals-secret")
            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.copy_sealed_artifact",
                    return_value=sealed_result(),
                ) as copy,
                patch(
                    "tools.cuda_campaign.cli._complete_operation",
                    side_effect=EvidenceStorageError(
                        "private receipt failure at /Users/private/receipts"
                    ),
                ),
            ):
                code, stdout, stderr = invoke(
                    [
                        "copy-seal",
                        "--operation-id",
                        COPY_OPERATION_ID,
                        "--source",
                        str(source),
                        "--destination",
                        str(root / "copied-secret"),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                        "--copy-id",
                        COPY_ID,
                        "--failure-domain-id",
                        DOMAIN_ID,
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--issuer-role-id",
                        "evidence-custodian",
                        "--off-experiment-host",
                    ]
                )
            self.assertEqual((code, stdout), (5, ""))
            failure = json.loads(stderr)
            self.assertEqual(failure["error_code"], "RECEIPT_APPEND_PENDING")
            self.assertEqual(failure["operation_id"], COPY_OPERATION_ID)
            self.assertNotIn(str(root), stderr)
            self.assertEqual(copy.call_count, 1)
            intent_path = journals / f"{COPY_OPERATION_ID}.intent.json"
            outcome_path = journals / f"{COPY_OPERATION_ID}.outcome.json"
            self.assertEqual(stat.S_IMODE(intent_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(outcome_path.stat().st_mode), 0o600)
            self.assertNotIn(str(root), intent_path.read_text(encoding="utf-8"))

            resume = [
                "resume-operation",
                "--source",
                str(source),
                "--destination",
                str(root / "copied-secret"),
                "--receipt-store",
                str(receipts),
                "--receipt-journal",
                str(journals),
                "--operation-id",
                COPY_OPERATION_ID,
                "--issuer-role-id",
                "evidence-custodian",
            ]
            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.verify_copy_equality",
                    return_value=equality_result(),
                ),
            ):
                code, stdout, stderr = invoke(resume)
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(json.loads(stdout)["receipt_result"], "passed")
            self.assertEqual(len(AppendOnlyReceiptStore(receipts).read_chain()), 1)
            self.assertEqual(copy.call_count, 1)

            with patch(
                "tools.cuda_campaign.cli.verify_sealed_artifact",
                return_value=sealed_result(),
            ):
                code, stdout, stderr = invoke(resume)
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(len(AppendOnlyReceiptStore(receipts).read_chain()), 1)
            self.assertEqual(json.loads(stdout)["operation_id"], COPY_OPERATION_ID)

            changed = json.loads(outcome_path.read_text(encoding="utf-8"))
            changed["details"]["copy_id"] = "copy_" + "f" * 32
            outcome_path.write_bytes(canonical_json_bytes(changed))
            outcome_path.chmod(0o600)
            with patch(
                "tools.cuda_campaign.cli.verify_sealed_artifact",
                return_value=sealed_result(),
            ):
                code, stdout, stderr = invoke(resume)
            self.assertEqual((code, stdout), (5, ""))
            self.assertEqual(
                json.loads(stderr)["error_code"], "RECEIPT_JOURNAL_INVALID"
            )
            self.assertNotIn("copy_" + "f" * 32, stderr)

    def test_copy_crash_before_outcome_reconciles_without_recopy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = private_directory(root, "source-secret")
            receipts = private_directory(root, "receipts-secret")
            journals = private_directory(root, "journals-secret")
            destination = root / "copied-secret"

            def copy_once(_source: Path, target: Path) -> dict[str, object]:
                target.mkdir(mode=0o700)
                target.chmod(0o700)
                return sealed_result()

            command = [
                "copy-seal",
                "--operation-id",
                COPY_OPERATION_ID,
                "--source",
                str(source),
                "--destination",
                str(destination),
                "--protected-artifact-id",
                ARTIFACT_ID,
                "--copy-id",
                COPY_ID,
                "--failure-domain-id",
                DOMAIN_ID,
                "--receipt-store",
                str(receipts),
                "--receipt-journal",
                str(journals),
                "--issuer-role-id",
                "evidence-custodian",
                "--off-experiment-host",
            ]
            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.copy_sealed_artifact",
                    side_effect=copy_once,
                ) as copy,
                patch(
                    "tools.cuda_campaign.cli._write_operation_outcome",
                    side_effect=OSError("private crash detail"),
                ),
            ):
                code, stdout, stderr = invoke(command)
            self.assertEqual((code, stdout), (5, ""))
            self.assertEqual(
                json.loads(stderr),
                {
                    "error_code": "OPERATION_OUTCOME_PENDING",
                    "ok": False,
                    "operation_id": COPY_OPERATION_ID,
                },
            )
            self.assertEqual(copy.call_count, 1)
            self.assertTrue(destination.is_dir())

            mismatch = list(command)
            mismatch[mismatch.index(str(destination))] = str(root / "other-copy")
            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch("tools.cuda_campaign.cli.copy_sealed_artifact") as recopy,
            ):
                code, stdout, stderr = invoke(mismatch)
            self.assertEqual((code, stdout), (5, ""))
            self.assertEqual(
                json.loads(stderr)["error_code"], "OPERATION_INTENT_MISMATCH"
            )
            recopy.assert_not_called()

            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.verify_copy_equality",
                    return_value={
                        "protected_artifact_id": ARTIFACT_ID,
                        "raw_manifest_sha256": DIGEST,
                        "raw_manifest_size_bytes": 100,
                        "stored_file_count": 3,
                        "stored_total_bytes": 300,
                        "verification_result": "passed",
                    },
                ),
                patch("tools.cuda_campaign.cli.copy_sealed_artifact") as recopy,
            ):
                code, stdout, stderr = invoke(
                    [
                        "resume-operation",
                        "--source",
                        str(source),
                        "--destination",
                        str(destination),
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--operation-id",
                        COPY_OPERATION_ID,
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(json.loads(stdout)["receipt_result"], "passed")
            recopy.assert_not_called()
            self.assertEqual(len(AppendOnlyReceiptStore(receipts).read_chain()), 1)

    def test_resume_refuses_a_receipt_chain_that_moved_after_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = private_directory(root, "source-secret")
            receipts = private_directory(root, "receipts-secret")
            journals = private_directory(root, "journals-secret")
            destination = root / "copied-secret"
            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.copy_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli._complete_operation",
                    side_effect=EvidenceStorageError("receipt append crashed"),
                ),
            ):
                code, stdout, stderr = invoke(
                    [
                        "copy-seal",
                        "--operation-id",
                        SECOND_COPY_OPERATION_ID,
                        "--source",
                        str(source),
                        "--destination",
                        str(destination),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                        "--copy-id",
                        COPY_ID,
                        "--failure-domain-id",
                        DOMAIN_ID,
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--issuer-role-id",
                        "evidence-custodian",
                        "--off-experiment-host",
                    ]
                )
            self.assertEqual((code, stdout), (5, ""))
            self.assertEqual(json.loads(stderr)["error_code"], "RECEIPT_APPEND_PENDING")

            AppendOnlyReceiptStore(receipts).append(
                kind="retention",
                issuer_role_id="evidence-custodian",
                protected_artifact_id=ARTIFACT_ID,
                raw_manifest_sha256=DIGEST,
                raw_manifest_size_bytes=100,
                result="active",
                details={"retain_not_before_utc": "2028-08-08T12:00:00+00:00"},
            )
            with patch(
                "tools.cuda_campaign.cli.verify_sealed_artifact",
                return_value=sealed_result(),
            ):
                code, stdout, stderr = invoke(
                    [
                        "resume-operation",
                        "--source",
                        str(source),
                        "--destination",
                        str(destination),
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--operation-id",
                        SECOND_COPY_OPERATION_ID,
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stdout), (5, ""))
            self.assertEqual(json.loads(stderr)["error_code"], "RECEIPT_CHAIN_MOVED")
            self.assertEqual(len(AppendOnlyReceiptStore(receipts).read_chain()), 1)

    def test_successful_retrieval_receipt_failure_resumes_without_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = private_directory(root, "off-host-secret")
            receipts = private_directory(root, "receipts-secret")
            journals = private_directory(root, "journals-secret")
            AppendOnlyReceiptStore(receipts).append(
                kind="copy-verification",
                issuer_role_id="evidence-custodian",
                protected_artifact_id=ARTIFACT_ID,
                raw_manifest_sha256=DIGEST,
                raw_manifest_size_bytes=100,
                result="passed",
                details={
                    "copy_id": COPY_ID,
                    "failure_domain_id": DOMAIN_ID,
                    "off_experiment_host": True,
                    "verification_result": "passed",
                },
            )
            details = {
                "source_copy_id": COPY_ID,
                "source_failure_domain_id": DOMAIN_ID,
                "destination_restore_id": RESTORE_ID,
                "started_at_utc": "2026-08-08T12:00:00+00:00",
                "finished_at_utc": "2026-08-08T12:00:01+00:00",
                "duration_ns": 1_000_000_000,
                "restored_file_count": 5,
                "restored_total_bytes": 300,
                "expected_raw_manifest_sha256": DIGEST,
                "observed_raw_manifest_sha256": DIGEST,
                "mismatch_count": 0,
                "verification_result": "passed",
            }
            with (
                retrieval_clock(),
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.retrieve_sealed_artifact",
                    return_value=details,
                ) as retrieve,
                patch(
                    "tools.cuda_campaign.cli._complete_operation",
                    side_effect=EvidenceStorageError("private append failure"),
                ),
            ):
                code, stdout, stderr = invoke(
                    [
                        "retrieve",
                        "--operation-id",
                        RETRIEVE_OPERATION_ID,
                        "--source",
                        str(source),
                        "--destination",
                        str(root / "restored-secret"),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                        "--source-copy-id",
                        COPY_ID,
                        "--source-failure-domain-id",
                        DOMAIN_ID,
                        "--expected-raw-manifest-sha256",
                        DIGEST,
                        "--destination-restore-id",
                        RESTORE_ID,
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stdout), (5, ""))
            failure = json.loads(stderr)
            self.assertEqual(failure["error_code"], "RECEIPT_APPEND_PENDING")
            self.assertEqual(failure["operation_id"], RETRIEVE_OPERATION_ID)
            self.assertEqual(retrieve.call_count, 1)
            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch("tools.cuda_campaign.cli.retrieve_sealed_artifact") as rerun,
            ):
                code, stdout, stderr = invoke(
                    [
                        "retrieve",
                        "--operation-id",
                        RETRIEVE_OPERATION_ID,
                        "--source",
                        str(source),
                        "--destination",
                        str(root / "restored-secret"),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                        "--source-copy-id",
                        COPY_ID,
                        "--source-failure-domain-id",
                        DOMAIN_ID,
                        "--expected-raw-manifest-sha256",
                        DIGEST,
                        "--destination-restore-id",
                        RESTORE_ID,
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stdout), (5, ""))
            self.assertEqual(
                json.loads(stderr),
                {
                    "error_code": "OPERATION_RESUME_REQUIRED",
                    "ok": False,
                    "operation_id": RETRIEVE_OPERATION_ID,
                },
            )
            rerun.assert_not_called()
            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.verify_copy_equality",
                    return_value=equality_result(),
                ),
            ):
                code, stdout, stderr = invoke(
                    [
                        "resume-operation",
                        "--source",
                        str(source),
                        "--destination",
                        str(root / "restored-secret"),
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--operation-id",
                        RETRIEVE_OPERATION_ID,
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stderr), (0, ""))
            receipt = AppendOnlyReceiptStore(receipts).read_chain()[-1]
            self.assertEqual(receipt["details"], details)
            self.assertEqual(receipt["result"], "passed")
            self.assertEqual(retrieve.call_count, 1)
            self.assertNotIn(str(root), stdout)

    def test_retrieval_crash_before_outcome_never_invents_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = private_directory(root, "off-host-secret")
            receipts = private_directory(root, "receipts-secret")
            journals = private_directory(root, "journals-secret")
            destination = root / "restored-secret"
            AppendOnlyReceiptStore(receipts).append(
                kind="copy-verification",
                issuer_role_id="evidence-custodian",
                protected_artifact_id=ARTIFACT_ID,
                raw_manifest_sha256=DIGEST,
                raw_manifest_size_bytes=100,
                result="passed",
                details={
                    "copy_id": COPY_ID,
                    "failure_domain_id": DOMAIN_ID,
                    "off_experiment_host": True,
                    "verification_result": "passed",
                },
            )
            measured = {
                "source_copy_id": COPY_ID,
                "source_failure_domain_id": DOMAIN_ID,
                "destination_restore_id": RESTORE_ID,
                "started_at_utc": "2026-08-08T12:00:00+00:00",
                "finished_at_utc": "2026-08-08T12:00:01+00:00",
                "duration_ns": 812_345_678,
                "restored_file_count": 5,
                "restored_total_bytes": 300,
                "expected_raw_manifest_sha256": DIGEST,
                "observed_raw_manifest_sha256": DIGEST,
                "mismatch_count": 0,
                "verification_result": "passed",
            }

            def restore_once(
                _source: Path, target: Path, **_kwargs: object
            ) -> dict[str, object]:
                target.mkdir(mode=0o700)
                target.chmod(0o700)
                return measured

            with (
                retrieval_clock(),
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.retrieve_sealed_artifact",
                    side_effect=restore_once,
                ) as retrieve,
                patch(
                    "tools.cuda_campaign.cli._write_operation_outcome",
                    side_effect=OSError("crash after restore"),
                ),
            ):
                code, stdout, stderr = invoke(
                    [
                        "retrieve",
                        "--operation-id",
                        RETRIEVE_OPERATION_ID,
                        "--source",
                        str(source),
                        "--destination",
                        str(destination),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                        "--source-copy-id",
                        COPY_ID,
                        "--source-failure-domain-id",
                        DOMAIN_ID,
                        "--expected-raw-manifest-sha256",
                        DIGEST,
                        "--destination-restore-id",
                        RESTORE_ID,
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stdout), (5, ""))
            self.assertEqual(
                json.loads(stderr)["error_code"], "OPERATION_OUTCOME_PENDING"
            )
            self.assertEqual(retrieve.call_count, 1)

            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch("tools.cuda_campaign.cli.verify_copy_equality") as equality,
                patch("tools.cuda_campaign.cli.retrieve_sealed_artifact") as rerun,
            ):
                code, stdout, stderr = invoke(
                    [
                        "resume-operation",
                        "--source",
                        str(source),
                        "--destination",
                        str(destination),
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--operation-id",
                        RETRIEVE_OPERATION_ID,
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stdout), (5, ""))
            self.assertEqual(
                json.loads(stderr),
                {
                    "error_code": "OPERATION_RECONCILIATION_REQUIRED",
                    "ok": False,
                    "operation_id": RETRIEVE_OPERATION_ID,
                },
            )
            equality.assert_not_called()
            rerun.assert_not_called()
            chain = AppendOnlyReceiptStore(receipts).read_chain()
            self.assertEqual(len(chain), 1)
            self.assertEqual(chain[0]["kind"], "copy-verification")
            self.assertNotIn("duration_ns", chain[0])

    def test_failed_retrieval_receipt_failure_preserves_exact_resume_details(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = private_directory(root, "off-host-secret")
            receipts = private_directory(root, "receipts-secret")
            journals = private_directory(root, "journals-secret")
            AppendOnlyReceiptStore(receipts).append(
                kind="copy-verification",
                issuer_role_id="evidence-custodian",
                protected_artifact_id=ARTIFACT_ID,
                raw_manifest_sha256=DIGEST,
                raw_manifest_size_bytes=100,
                result="passed",
                details={
                    "copy_id": COPY_ID,
                    "failure_domain_id": DOMAIN_ID,
                    "off_experiment_host": True,
                    "verification_result": "passed",
                },
            )
            details = {
                "source_copy_id": COPY_ID,
                "source_failure_domain_id": DOMAIN_ID,
                "destination_restore_id": RESTORE_ID,
                "started_at_utc": "2026-08-08T12:00:00+00:00",
                "finished_at_utc": "2026-08-08T12:00:01+00:00",
                "duration_ns": 1_000_000_000,
                "restored_file_count": 0,
                "restored_total_bytes": 0,
                "expected_raw_manifest_sha256": DIGEST,
                "observed_raw_manifest_sha256": None,
                "mismatch_count": 1,
                "verification_result": "failed",
            }
            with (
                retrieval_clock(),
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch(
                    "tools.cuda_campaign.cli.retrieve_sealed_artifact",
                    side_effect=RetrievalError(
                        "private failure at /Users/private/raw.log", details=details
                    ),
                ) as retrieve,
                patch(
                    "tools.cuda_campaign.cli._complete_operation",
                    side_effect=EvidenceStorageError("private append failure"),
                ),
            ):
                code, stdout, stderr = invoke(
                    [
                        "retrieve",
                        "--operation-id",
                        RETRIEVE_OPERATION_ID,
                        "--source",
                        str(source),
                        "--destination",
                        str(root / "partial-restore-secret"),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                        "--source-copy-id",
                        COPY_ID,
                        "--source-failure-domain-id",
                        DOMAIN_ID,
                        "--expected-raw-manifest-sha256",
                        DIGEST,
                        "--destination-restore-id",
                        RESTORE_ID,
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stdout), (5, ""))
            failure = json.loads(stderr)
            self.assertEqual(failure["error_code"], "RECEIPT_APPEND_PENDING")
            self.assertEqual(failure["operation_id"], RETRIEVE_OPERATION_ID)
            self.assertNotIn("/Users/private", stderr)
            with patch(
                "tools.cuda_campaign.cli.verify_sealed_artifact",
                return_value=sealed_result(),
            ):
                code, stdout, stderr = invoke(
                    [
                        "resume-operation",
                        "--source",
                        str(source),
                        "--destination",
                        str(root / "partial-restore-secret"),
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--operation-id",
                        RETRIEVE_OPERATION_ID,
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stderr), (0, ""))
            receipt = AppendOnlyReceiptStore(receipts).read_chain()[-1]
            self.assertEqual(receipt["result"], "failed")
            self.assertEqual(receipt["details"], details)
            self.assertEqual(retrieve.call_count, 1)
            self.assertNotIn(str(root), stdout)

    def test_retrieve_requires_matching_off_host_copy_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = private_directory(root, "source-secret")
            receipts = private_directory(root, "receipts")
            journals = private_directory(root, "journals")
            AppendOnlyReceiptStore(receipts).append(
                kind="copy-verification",
                issuer_role_id="evidence-custodian",
                protected_artifact_id=ARTIFACT_ID,
                raw_manifest_sha256=DIGEST,
                raw_manifest_size_bytes=100,
                result="passed",
                details={
                    "copy_id": COPY_ID,
                    "failure_domain_id": DOMAIN_ID,
                    "off_experiment_host": False,
                    "verification_result": "passed",
                },
            )
            with (
                patch(
                    "tools.cuda_campaign.cli.verify_sealed_artifact",
                    return_value=sealed_result(),
                ),
                patch("tools.cuda_campaign.cli.retrieve_sealed_artifact") as retrieve,
            ):
                code, stdout, stderr = invoke(
                    [
                        "retrieve",
                        "--operation-id",
                        RETRIEVE_OPERATION_ID,
                        "--source",
                        str(source),
                        "--destination",
                        str(root / "restore"),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                        "--source-copy-id",
                        COPY_ID,
                        "--source-failure-domain-id",
                        DOMAIN_ID,
                        "--expected-raw-manifest-sha256",
                        DIGEST,
                        "--destination-restore-id",
                        RESTORE_ID,
                        "--receipt-store",
                        str(receipts),
                        "--receipt-journal",
                        str(journals),
                        "--issuer-role-id",
                        "evidence-custodian",
                    ]
                )
            self.assertEqual((code, stdout), (5, ""))
            self.assertEqual(
                json.loads(stderr)["error_code"],
                "OFF_HOST_COPY_ATTESTATION_REQUIRED",
            )
            retrieve.assert_not_called()

    def test_capture_command_requires_explicit_nonqualifying_telemetry_waiver(
        self,
    ) -> None:
        base = [
            "capture-command",
            "--state-root",
            "/not-reached/state",
            "--artifact-directory",
            "/not-reached/artifact",
            "--working-directory",
            "/not-reached/work",
            "--attempt-slot-id",
            SLOT_ID,
            "--experiment-run-id",
            XRUN_ID,
            "--retain-not-before-utc",
            "2028-08-08T00:00:00+00:00",
            "--timeout-seconds",
            "5",
        ]
        code, stdout, stderr = invoke(base + ["--mode", "nonqualifying", "--", "true"])
        self.assertEqual((code, stdout), (7, ""))
        self.assertEqual(
            json.loads(stderr)["error_code"], "TELEMETRY_SIDECAR_UNAVAILABLE"
        )

        code, stdout, stderr = invoke(
            base
            + [
                "--mode",
                "qualifying",
                "--without-telemetry",
                "--",
                "true",
            ]
        )
        self.assertEqual((code, stdout), (7, ""))
        self.assertEqual(
            json.loads(stderr)["error_code"],
            "TELEMETRY_REQUIRED_FOR_QUALIFYING_CAPTURE",
        )

    def test_nonqualifying_capture_passes_exact_argv_without_a_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = private_directory(root, "state")
            work = private_directory(root, "private-work")
            outcome = SimpleNamespace(
                native_outcome="passed",
                sealed=True,
                attempt_slot_id=SLOT_ID,
                experiment_run_id=XRUN_ID,
                reason_code="NONE",
                exit_code=0,
                timed_out=False,
                submission_blocked=False,
                telemetry_healthy=None,
                seal_verification={
                    "protected_artifact_id": ARTIFACT_ID,
                    "raw_manifest_sha256": DIGEST,
                },
            )
            with patch("tools.cuda_campaign.cli.CaptureHarness") as harness:
                harness.return_value.run_command.return_value = outcome
                code, stdout, stderr = invoke(
                    [
                        "capture-command",
                        "--state-root",
                        str(state),
                        "--artifact-directory",
                        str(root / "private-artifact"),
                        "--working-directory",
                        str(work),
                        "--attempt-slot-id",
                        SLOT_ID,
                        "--experiment-run-id",
                        XRUN_ID,
                        "--retain-not-before-utc",
                        "2028-08-08T00:00:00+00:00",
                        "--timeout-seconds",
                        "5",
                        "--mode",
                        "nonqualifying",
                        "--without-telemetry",
                        "--",
                        sys.executable,
                        "-c",
                        "print('private output')",
                    ]
                )
        self.assertEqual((code, stderr), (0, ""))
        call = harness.return_value.run_command.call_args
        self.assertEqual(
            call.args[0],
            [sys.executable, "-c", "print('private output')"],
        )
        self.assertIsNone(call.kwargs["telemetry_session"])
        self.assertNotIn("private output", stdout)
        self.assertNotIn(str(root), stdout)

    def test_sanitize_and_review_use_sealed_provenance_without_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery = private_directory(root, "private-recovery-artifact")
            control = private_directory(root, "private-control-artifact")
            stage_parent = private_directory(root, "stages")
            projection = object()
            with (
                patch(
                    "tools.cuda_campaign.cli.project_verified_recovery_supplement",
                    return_value=projection,
                ) as project,
                patch(
                    "tools.cuda_campaign.cli.write_projection_stage",
                    return_value={
                        "claim-boundary.json": "b" * 64,
                        "recovery-supplement.json": "c" * 64,
                        "sanitization-map.json": "d" * 64,
                    },
                ),
            ):
                code, stdout, stderr = invoke(
                    [
                        "sanitize-recovery-stage",
                        "--recovery-artifact",
                        str(recovery),
                        "--control-artifact",
                        str(control),
                        "--stage-parent",
                        str(stage_parent),
                        "--stage-name",
                        "review-stage",
                    ]
                )
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(
                project.call_args.kwargs,
                {
                    "recovery_artifact": recovery.resolve(),
                    "control_artifact": control.resolve(),
                },
            )
            self.assertEqual(
                project.call_count,
                1,
            )
            self.assertNotIn(str(root), stdout)

            stage = private_directory(root, "review-stage")
            review = {
                "review_id": REVIEW_ID,
                "producer_role_id": "capture-producer",
                "reviewer_role_id": "privacy-reviewer",
                "reviewed_at_utc": REVIEWED_AT,
                "result": "passed",
                "reason_code": "NONE",
                "checks": {"private-value-absence": True},
            }
            with patch(
                "tools.cuda_campaign.cli.verify_projection_stage",
                return_value=review,
            ) as verify:
                code, stdout, stderr = invoke(
                    [
                        "review-recovery-stage",
                        "--stage",
                        str(stage),
                        "--recovery-artifact",
                        str(recovery),
                        "--control-artifact",
                        str(control),
                        "--producer-role-id",
                        "capture-producer",
                        "--reviewer-role-id",
                        "privacy-reviewer",
                    ]
                )
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(
                verify.call_args.args,
                (stage.resolve(),),
            )
            self.assertEqual(
                verify.call_args.kwargs,
                {
                    "recovery_artifact": recovery.resolve(),
                    "control_artifact": control.resolve(),
                    "producer_role_id": "capture-producer",
                    "reviewer_role_id": "privacy-reviewer",
                },
            )
            self.assertNotIn(str(root), stdout)

    def test_publication_lifecycle_commands_are_distinct_and_old_names_absent(
        self,
    ) -> None:
        parser = campaign_cli.build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, campaign_cli.argparse._SubParsersAction)
        )
        lifecycle = (
            "seal-projection-review",
            "finalize-publication-candidate",
            "seal-publication-candidate",
            "verify-publication-candidate",
            "evaluate-publication",
            "publish-candidate",
        )
        handlers = [
            subparsers.choices[name].get_default("handler") for name in lifecycle
        ]
        self.assertEqual(len(set(handlers)), len(lifecycle))
        publish_arguments = {
            action.dest for action in subparsers.choices["publish-candidate"]._actions
        }
        self.assertNotIn("now_utc", publish_arguments)
        for read_only_command in (
            "verify-publication-candidate",
            "evaluate-publication",
        ):
            self.assertIn(
                "now_utc",
                {
                    action.dest
                    for action in subparsers.choices[read_only_command]._actions
                },
            )
        for command in (
            "finalize-recovery-stage",
            "verify-finalized-recovery",
            "publish-recovery-supplement",
        ):
            with self.subTest(command=command):
                code, stdout, stderr = invoke([command])
                self.assertEqual((code, stdout), (2, ""))
                self.assertEqual(json.loads(stderr)["error_code"], "INVALID_ARGUMENT")

    def test_recovery_lifecycle_destinations_are_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recovery = private_directory(root, "recovery")
            control = private_directory(root, "control")
            stages = private_directory(root, "stages")
            private_directory(stages, "already-present")
            with patch(
                "tools.cuda_campaign.cli.project_verified_recovery_supplement"
            ) as project:
                code, stdout, stderr = invoke(
                    [
                        "sanitize-recovery-stage",
                        "--recovery-artifact",
                        str(recovery),
                        "--control-artifact",
                        str(control),
                        "--stage-parent",
                        str(stages),
                        "--stage-name",
                        "already-present",
                    ]
                )
            self.assertEqual((code, stdout), (4, ""))
            self.assertEqual(json.loads(stderr)["error_code"], "DESTINATION_NOT_FRESH")
            project.assert_not_called()

            with patch(
                "tools.cuda_campaign.cli.project_verified_recovery_supplement"
            ) as project:
                code, stdout, stderr = invoke(
                    [
                        "sanitize-recovery-stage",
                        "--recovery-artifact",
                        str(recovery),
                        "--control-artifact",
                        str(control),
                        "--stage-parent",
                        str(recovery),
                        "--stage-name",
                        "nested-stage",
                    ]
                )
            self.assertEqual((code, stdout), (4, ""))
            self.assertEqual(
                json.loads(stderr)["error_code"], "STORAGE_BOUNDARY_COLLISION"
            )
            project.assert_not_called()

    def test_legacy_dictionary_sanitizer_arguments_are_not_exposed(self) -> None:
        code, stdout, stderr = invoke(
            [
                "sanitize-recovery-stage",
                "--expected-manifest",
                "/private/expected.json",
                "--protected-root",
                "/private/protected",
                "--protected-input-relative",
                "input.json",
            ]
        )
        self.assertEqual((code, stdout), (2, ""))
        self.assertEqual(json.loads(stderr)["error_code"], "INVALID_ARGUMENT")
        self.assertNotIn("/private", stderr)

    def test_errors_are_stable_and_never_include_raw_exception_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = private_directory(Path(temporary), "artifact")
            with patch(
                "tools.cuda_campaign.cli.verify_sealed_artifact",
                side_effect=ArtifactIntegrityError(
                    "private failure at /Users/private/vault/raw.log"
                ),
            ):
                code, stdout, stderr = invoke(
                    [
                        "verify-seal",
                        "--artifact",
                        str(artifact),
                        "--protected-artifact-id",
                        ARTIFACT_ID,
                    ]
                )
        self.assertEqual((code, stdout), (5, ""))
        self.assertEqual(
            json.loads(stderr),
            {"error_code": "ARTIFACT_INTEGRITY_FAILURE", "ok": False},
        )
        self.assertNotIn("/Users/private", stderr)
        self.assertNotIn("raw.log", stderr)

    def test_invalid_arguments_return_bounded_json_not_argparse_text(self) -> None:
        code, stdout, stderr = invoke(["verify-seal", "--artifact", "/secret/path"])
        self.assertEqual((code, stdout), (2, ""))
        self.assertEqual(json.loads(stderr)["error_code"], "INVALID_ARGUMENT")
        self.assertNotIn("/secret/path", stderr)
        self.assertNotIn("usage:", stderr)

        code, stdout, stderr = invoke(
            [
                "resume-receipt",
                "--source",
                "/not-reached/source",
                "--destination",
                "/not-reached/destination",
                "--receipt-store",
                "/not-reached/receipts",
                "--receipt-journal",
                "/not-reached/journals",
                "--operation-id",
                COPY_OPERATION_ID,
                "--issuer-role-id",
                "a" * 65,
            ]
        )
        self.assertEqual((code, stdout), (2, ""))
        self.assertEqual(json.loads(stderr)["error_code"], "INVALID_ROLE_ID")
        self.assertNotIn("a" * 65, stderr)


if __name__ == "__main__":
    unittest.main()
