from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tools.cuda_campaign import eligibility
from tools.cuda_campaign.contracts import (
    canonical_json_bytes,
    compact_canonical_json_bytes,
    sha256_bytes,
)
from tools.cuda_campaign.eligibility import (
    EXTERNAL_EVIDENCE_SCHEMA,
    EXTERNAL_RECOVERY_ATTESTATION_SCHEMA,
    FinalizedSanitizerBinding,
    PublicationCandidateBinding,
    evaluate_publication_eligibility,
    seal_publication_candidate,
    verify_publication_candidate,
)
from tools.cuda_campaign.sanitizer import (
    finalize_projection_stage,
    project_verified_recovery_supplement,
    seal_projection_review,
    write_projection_stage,
)
from tools.cuda_campaign.storage import RawArtifactWriter, verify_sealed_artifact


ARTIFACT_ID = "artifact_" + "a" * 32
COPY_ONE = "copy_" + "1" * 32
COPY_TWO = "copy_" + "2" * 32
DOMAIN_ONE = "domain_" + "1" * 32
DOMAIN_TWO = "domain_" + "2" * 32
NOW = datetime(2026, 8, 8, 12, tzinfo=timezone.utc)
PRODUCER = "phase2-packet-producer"
REVIEWER = "phase2-independent-reviewer"
FINALIZER = "phase2-candidate-finalizer"
CANDIDATE_PRODUCER = "phase2-candidate-assembler"
CAMPAIGN_ID = "campaign_" + "c" * 20
CLAIM_KEY = "august-6-protected-raw-recovery-integrity"


class EligibilityFilesystemHardeningTests(unittest.TestCase):
    def test_regular_file_read_rejects_same_byte_inode_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            victim = root / "victim.json"
            replacement = root / "replacement.json"
            held = root / "held.json"
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

            with patch(
                "tools.cuda_campaign.eligibility.os.open",
                side_effect=swap_then_open,
            ):
                with self.assertRaisesRegex(ValueError, "changed while opening"):
                    eligibility._regular_file_bytes(victim)

    def test_directory_binding_rejects_swap_add_and_remove_during_inventory(
        self,
    ) -> None:
        for attack in ("same-byte-swap", "add", "remove"):
            with (
                self.subTest(attack=attack),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = _private_root(Path(temporary))
                stage = root / "stage"
                stage.mkdir(mode=0o700)
                first = stage / "a.json"
                second = stage / "b.json"
                first.write_bytes(b"{}\n")
                second.write_bytes(b"[]\n")
                first.chmod(0o600)
                second.chmod(0o600)
                original = eligibility._regular_file_bytes_at
                changed = False

                def mutate_then_read(
                    directory_descriptor: int,
                    name: str,
                    **kwargs: object,
                ):
                    nonlocal changed
                    if not changed:
                        changed = True
                        if attack == "same-byte-swap":
                            held = stage / "held.json"
                            replacement = stage / "replacement.json"
                            replacement.write_bytes(first.read_bytes())
                            replacement.chmod(0o600)
                            first.rename(held)
                            replacement.rename(first)
                        elif attack == "add":
                            added = stage / "c.json"
                            added.write_bytes(b"{}\n")
                            added.chmod(0o600)
                        else:
                            second.unlink()
                    return original(directory_descriptor, name, **kwargs)

                with patch(
                    "tools.cuda_campaign.eligibility._regular_file_bytes_at",
                    side_effect=mutate_then_read,
                ):
                    with self.assertRaises(ValueError):
                        eligibility._private_directory_bindings(
                            stage, label="candidate stage"
                        )

    def test_directory_snapshot_rejects_parent_path_swap_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            stage = root / "stage"
            stage.mkdir(mode=0o700)
            original_child = stage / "record.json"
            original_payload = canonical_json_bytes({"source": "pinned-original"})
            original_child.write_bytes(original_payload)
            original_child.chmod(0o600)

            replacement = root / "replacement"
            replacement.mkdir(mode=0o700)
            replacement_child = replacement / "record.json"
            replacement_child.write_bytes(
                canonical_json_bytes({"source": "path-replacement"})
            )
            replacement_child.chmod(0o600)
            held = root / "held-original"
            original = eligibility._regular_file_bytes_at
            swapped = False
            observed_child_payload: bytes | None = None

            def swap_parent_then_read(
                directory_descriptor: int,
                name: str,
                **kwargs: object,
            ):
                nonlocal observed_child_payload, swapped
                if swapped:
                    return original(directory_descriptor, name, **kwargs)
                swapped = True
                stage.rename(held)
                replacement.rename(stage)
                try:
                    result = original(directory_descriptor, name, **kwargs)
                    observed_child_payload = result[0]
                    return result
                finally:
                    stage.rename(replacement)
                    held.rename(stage)

            with (
                patch(
                    "tools.cuda_campaign.eligibility._regular_file_bytes_at",
                    side_effect=swap_parent_then_read,
                ),
                self.assertRaisesRegex(ValueError, "changed while hashing"),
            ):
                eligibility._private_directory_snapshot(stage, label="candidate stage")

            self.assertTrue(swapped)
            self.assertEqual(observed_child_payload, original_payload)


def _private_root(path: Path) -> Path:
    path.chmod(0o700)
    return path


def _sealed_artifact(
    root: Path,
    *,
    artifact_id: str = ARTIFACT_ID,
    payload: bytes = b"protected evidence\n",
) -> tuple[Path, dict]:
    artifact = root / "artifact"
    writer = RawArtifactWriter(
        artifact,
        protected_artifact_id=artifact_id,
        record_kind="legacy-recovery",
        identity_bindings={"purpose": "eligibility-test"},
        capture_tool={"name": "eligibility-test", "version": "v1"},
        source_bindings={"source": "fixture"},
        provisional_retain_not_before_utc="2028-08-08T12:00:00+00:00",
    )
    writer.write_payload(
        payload,
        "evidence.bin",
        role="protected-evidence",
        entry_id="entry_evidence",
    )
    return artifact, writer.seal()


def _receipt(
    *,
    kind: str,
    created_at_utc: str,
    previous_receipt_id: str | None,
    artifact: dict,
    result: str,
    details: dict,
    issuer_role_id: str = "phase2-evidence-custodian",
) -> dict:
    without_id = {
        "schema_version": "aptus.experiment-evidence-receipt.v1",
        "kind": kind,
        "created_at_utc": created_at_utc,
        "issuer_role_id": issuer_role_id,
        "protected_artifact_id": artifact["protected_artifact_id"],
        "raw_manifest_sha256": artifact["raw_manifest_sha256"],
        "raw_manifest_size_bytes": artifact["raw_manifest_size_bytes"],
        "previous_receipt_id": previous_receipt_id,
        "result": result,
        "details": details,
    }
    receipt_id = (
        "receipt_" + sha256_bytes(compact_canonical_json_bytes(without_id))[:32]
    )
    return {"receipt_id": receipt_id, **without_id}


def _receipt_chain(
    artifact: dict,
    *,
    copy_ids: tuple[str, str] = (COPY_ONE, COPY_TWO),
    domains: tuple[str, str] = (DOMAIN_ONE, DOMAIN_TWO),
    retention_deadline: str = "2028-08-09T12:00:00+00:00",
) -> list[dict]:
    first = _receipt(
        kind="copy-verification",
        created_at_utc="2026-08-07T10:00:00+00:00",
        previous_receipt_id=None,
        artifact=artifact,
        result="passed",
        details={
            "copy_id": copy_ids[0],
            "failure_domain_id": domains[0],
            "off_experiment_host": False,
            "verification_result": "passed",
        },
    )
    second = _receipt(
        kind="copy-verification",
        created_at_utc="2026-08-07T11:00:00+00:00",
        previous_receipt_id=first["receipt_id"],
        artifact=artifact,
        result="passed",
        details={
            "copy_id": copy_ids[1],
            "failure_domain_id": domains[1],
            "off_experiment_host": True,
            "verification_result": "passed",
        },
    )
    retrieval = _receipt(
        kind="retrieval",
        created_at_utc="2026-08-07T12:00:00+00:00",
        previous_receipt_id=second["receipt_id"],
        artifact=artifact,
        result="passed",
        details={
            "source_copy_id": copy_ids[1],
            "source_failure_domain_id": domains[1],
            "destination_restore_id": "restore_" + "3" * 32,
            "started_at_utc": "2026-08-07T11:59:59+00:00",
            "finished_at_utc": "2026-08-07T12:00:00+00:00",
            "duration_ns": 1_000_000_000,
            "restored_file_count": artifact["file_count"] + 2,
            "restored_total_bytes": (
                artifact["total_bytes"]
                + artifact["raw_manifest_size_bytes"]
                + len(canonical_json_bytes(artifact["seal"]))
            ),
            "expected_raw_manifest_sha256": artifact["raw_manifest_sha256"],
            "observed_raw_manifest_sha256": artifact["raw_manifest_sha256"],
            "mismatch_count": 0,
            "verification_result": "passed",
        },
    )
    retention = _receipt(
        kind="retention",
        created_at_utc="2026-08-07T13:00:00+00:00",
        previous_receipt_id=retrieval["receipt_id"],
        artifact=artifact,
        result="active",
        details={
            "retention_policy_id": "cuda-v02-public-claim-evidence-24m-v1",
            "retain_not_before_utc": retention_deadline,
            "verification_result": "passed",
        },
    )
    return [first, second, retrieval, retention]


def _attestation(
    artifact: dict,
    chain: list[dict],
    *,
    copy_id: str | None = None,
    failure_domain_id: str | None = None,
    attested_at_utc: str = "2026-08-07T13:00:00+00:00",
) -> dict:
    record = {
        "schema_version": EXTERNAL_RECOVERY_ATTESTATION_SCHEMA,
        "attester_role_id": "external-recovery-attester",
        "evidence_custodian_role_id": "phase2-evidence-custodian",
        "attested_at_utc": attested_at_utc,
        "protected_artifact_id": artifact["protected_artifact_id"],
        "raw_manifest_sha256": artifact["raw_manifest_sha256"],
        "raw_manifest_size_bytes": artifact["raw_manifest_size_bytes"],
        "copy_id": copy_id or chain[1]["details"]["copy_id"],
        "failure_domain_id": (
            failure_domain_id or chain[1]["details"]["failure_domain_id"]
        ),
        "copy_verification_receipt_id": chain[1]["receipt_id"],
        "retrieval_receipt_id": chain[2]["receipt_id"],
        "off_host_storage_evidence": {
            "reference_id": "evidence_" + "1" * 32,
            "sha256": "1" * 64,
        },
        "encryption_in_transit_evidence": {
            "reference_id": "evidence_" + "2" * 32,
            "sha256": "2" * 64,
        },
        "encryption_at_rest_evidence": {
            "reference_id": "evidence_" + "3" * 32,
            "sha256": "3" * 64,
        },
        "key_custodian_role_id": "phase2-key-custodian",
        "key_custody_evidence": {
            "reference_id": "evidence_" + "4" * 32,
            "sha256": "4" * 64,
        },
        "recovery_procedure_id": "procedure_" + "5" * 32,
        "recovery_procedure_evidence": {
            "reference_id": "evidence_" + "5" * 32,
            "sha256": "5" * 64,
        },
    }
    record["attestation_id"] = (
        "attest_" + sha256_bytes(compact_canonical_json_bytes(record))[:32]
    )
    return record


def _readdress_attestation(record: dict) -> None:
    identity = {key: value for key, value in record.items() if key != "attestation_id"}
    record["attestation_id"] = (
        "attest_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]
    )


def _sanitizer_binding(root: Path) -> FinalizedSanitizerBinding:
    return FinalizedSanitizerBinding(
        projection_stage=root / "projection-stage",
        finalized_candidate_output=root / "final-candidate",
        review_artifact=root / "sealed-review",
        recovery_artifact=root / "recovery-artifact",
        control_artifact=root / "control-artifact",
        producer_role_id=PRODUCER,
        reviewer_role_id=REVIEWER,
        finalizer_role_id=FINALIZER,
    )


def _materialize_candidate_sanitizer(
    root: Path, artifact_path: Path
) -> FinalizedSanitizerBinding:
    binding = FinalizedSanitizerBinding(
        projection_stage=root / "projection-stage",
        finalized_candidate_output=root / "final-candidate",
        review_artifact=artifact_path,
        recovery_artifact=artifact_path,
        control_artifact=artifact_path,
        producer_role_id=PRODUCER,
        reviewer_role_id=REVIEWER,
        finalizer_role_id=FINALIZER,
    )
    binding.projection_stage.mkdir(mode=0o700)
    stage_path = binding.projection_stage / "stage.json"
    stage_path.write_bytes(canonical_json_bytes({"stage": "reviewed"}))
    stage_path.chmod(0o600)
    binding.finalized_candidate_output.mkdir(mode=0o700)
    boundary_path = binding.finalized_candidate_output / "claim-boundary.json"
    boundary_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "aptus.experiment-claim-boundary.v1",
                "campaign_id": CAMPAIGN_ID,
                "claim_key": CLAIM_KEY,
            }
        )
    )
    boundary_path.chmod(0o600)
    return binding


def _passed_review() -> dict:
    return {
        "review_id": "review_" + "7" * 32,
        "result": "passed",
        "reason_code": "NONE",
        "producer_role_id": PRODUCER,
        "reviewer_role_id": REVIEWER,
        "reviewed_at_utc": "2026-08-08T11:00:00+00:00",
        "finalization_id": "finalization_" + "6" * 32,
        "finalizer_role_id": FINALIZER,
        "finalized_at_utc": "2026-08-08T11:15:00+00:00",
    }


def _materialize_external_evidence(root: Path, attestation: dict) -> dict[str, Path]:
    evidence_root = root / "external-attestation-evidence"
    evidence_root.mkdir(mode=0o700, exist_ok=True)
    result: dict[str, Path] = {}
    kinds = {
        "off_host_storage_evidence": (
            "off-host-storage",
            {
                "copy_verification_receipt_id": attestation[
                    "copy_verification_receipt_id"
                ],
                "storage_control_id": "storage_" + "1" * 32,
                "off_experiment_host": True,
            },
        ),
        "encryption_in_transit_evidence": (
            "encryption-in-transit",
            {
                "copy_verification_receipt_id": attestation[
                    "copy_verification_receipt_id"
                ],
                "transport_control_id": "transport_" + "2" * 32,
                "transport_security": "encrypted-authenticated-channel",
            },
        ),
        "encryption_at_rest_evidence": (
            "encryption-at-rest",
            {
                "copy_verification_receipt_id": attestation[
                    "copy_verification_receipt_id"
                ],
                "encryption_control_id": "encryption_" + "3" * 32,
                "encryption_state": "encrypted-at-rest",
            },
        ),
        "key_custody_evidence": (
            "key-custody",
            {
                "key_custodian_role_id": attestation["key_custodian_role_id"],
                "key_control_id": "key-control_" + "4" * 32,
                "custody_state": "assigned",
            },
        ),
        "recovery_procedure_evidence": (
            "recovery-procedure",
            {
                "retrieval_receipt_id": attestation["retrieval_receipt_id"],
                "recovery_procedure_id": attestation["recovery_procedure_id"],
                "procedure_state": "verified-by-full-retrieval",
            },
        ),
    }
    for field, (kind, specific) in kinds.items():
        reference = attestation.get(field)
        if not isinstance(reference, dict):
            continue
        issuer_role_id = (
            attestation["key_custodian_role_id"]
            if field == "key_custody_evidence"
            else attestation["evidence_custodian_role_id"]
        )
        record = {
            "schema_version": EXTERNAL_EVIDENCE_SCHEMA,
            "evidence_kind": kind,
            "created_at_utc": attestation["attested_at_utc"],
            "issuer_role_id": issuer_role_id,
            "protected_artifact_id": attestation["protected_artifact_id"],
            "raw_manifest_sha256": attestation["raw_manifest_sha256"],
            "raw_manifest_size_bytes": attestation["raw_manifest_size_bytes"],
            "copy_id": attestation["copy_id"],
            "failure_domain_id": attestation["failure_domain_id"],
            "verification_result": "passed",
            **specific,
        }
        reference_id = (
            "evidence_" + sha256_bytes(compact_canonical_json_bytes(record))[:32]
        )
        record = {"reference_id": reference_id, **record}
        payload = canonical_json_bytes(record)
        path = evidence_root / f"{reference_id}.bin"
        path.write_bytes(payload)
        path.chmod(0o600)
        attestation[field] = {
            "reference_id": reference_id,
            "sha256": sha256_bytes(payload),
        }
        result[reference_id] = path
    _readdress_attestation(attestation)
    return result


def _evaluate(
    *,
    root: Path,
    artifact_path: Path,
    artifact: dict,
    chain: list[dict],
    attestation: dict,
    now: datetime = NOW,
    expected_artifact_id: str | None = None,
    expected_digest: str | None = None,
    expected_size: int | None = None,
    review: dict | None = None,
    review_error: Exception | None = None,
    candidate_error: Exception | None = None,
    external_evidence: dict[str, Path] | None = None,
):
    binding = _sanitizer_binding(root)
    if external_evidence is None:
        external_evidence = _materialize_external_evidence(root, attestation)
    patcher = (
        patch(
            "tools.cuda_campaign.eligibility.verify_finalized_projection",
            side_effect=review_error,
        )
        if review_error is not None
        else patch(
            "tools.cuda_campaign.eligibility.verify_finalized_projection",
            return_value=review or _passed_review(),
        )
    )
    candidate_patcher = patch(
        "tools.cuda_campaign.eligibility._verify_publication_candidate",
        side_effect=candidate_error,
        return_value={"candidate_id": "candidate_" + "8" * 32},
    )
    with patcher as verifier, candidate_patcher:
        result = evaluate_publication_eligibility(
            artifact=artifact_path,
            expected_protected_artifact_id=(
                expected_artifact_id or artifact["protected_artifact_id"]
            ),
            expected_raw_manifest_sha256=(
                expected_digest or artifact["raw_manifest_sha256"]
            ),
            expected_raw_manifest_size_bytes=(
                expected_size or artifact["raw_manifest_size_bytes"]
            ),
            receipts=chain,
            external_recovery_attestation=attestation,
            external_evidence=external_evidence,
            now_utc=now,
            sanitizer=binding,
            publication_candidate=PublicationCandidateBinding(
                artifact=root / "publication-candidate",
                campaign_id=CAMPAIGN_ID,
                claim_key=CLAIM_KEY,
                candidate_producer_role_id=CANDIDATE_PRODUCER,
            ),
        )
    return result, verifier, binding


def _append_receipt(
    chain: list[dict],
    artifact: dict,
    *,
    kind: str,
    created_at_utc: str,
    result: str,
    details: dict,
    issuer_role_id: str = "phase2-evidence-custodian",
) -> dict:
    receipt = _receipt(
        kind=kind,
        created_at_utc=created_at_utc,
        previous_receipt_id=chain[-1]["receipt_id"],
        artifact=artifact,
        result=result,
        details=details,
        issuer_role_id=issuer_role_id,
    )
    chain.append(receipt)
    return receipt


def _replace_retrieval_details(
    chain: list[dict], artifact: dict, **detail_updates: object
) -> None:
    retrieval = chain[2]
    details = {**retrieval["details"], **detail_updates}
    replacement = _receipt(
        kind="retrieval",
        created_at_utc=retrieval["created_at_utc"],
        previous_receipt_id=chain[1]["receipt_id"],
        artifact=artifact,
        result="passed",
        details=details,
        issuer_role_id=retrieval["issuer_role_id"],
    )
    retention = chain[3]
    chain[2] = replacement
    chain[3] = _receipt(
        kind="retention",
        created_at_utc=retention["created_at_utc"],
        previous_receipt_id=replacement["receipt_id"],
        artifact=artifact,
        result="active",
        details=dict(retention["details"]),
        issuer_role_id=retention["issuer_role_id"],
    )


def _append_valid_restoration_evidence(
    chain: list[dict], artifact: dict
) -> tuple[dict, dict]:
    _append_receipt(
        chain,
        artifact,
        kind="claim-suspension",
        created_at_utc="2026-08-07T14:00:00+00:00",
        result="suspended",
        details={"reason_code": "copy-state-revalidation"},
    )
    _append_receipt(
        chain,
        artifact,
        kind="copy-verification",
        created_at_utc="2026-08-07T14:10:00+00:00",
        result="passed",
        details={
            "copy_id": COPY_ONE,
            "failure_domain_id": DOMAIN_ONE,
            "off_experiment_host": False,
            "verification_result": "passed",
        },
    )
    off_host_copy = _append_receipt(
        chain,
        artifact,
        kind="copy-verification",
        created_at_utc="2026-08-07T14:20:00+00:00",
        result="passed",
        details={
            "copy_id": COPY_TWO,
            "failure_domain_id": DOMAIN_TWO,
            "off_experiment_host": True,
            "verification_result": "passed",
        },
    )
    retrieval = _append_receipt(
        chain,
        artifact,
        kind="retrieval",
        created_at_utc="2026-08-07T14:30:00+00:00",
        result="passed",
        details={
            "source_copy_id": COPY_TWO,
            "source_failure_domain_id": DOMAIN_TWO,
            "destination_restore_id": "restore_" + "6" * 32,
            "started_at_utc": "2026-08-07T14:29:59+00:00",
            "finished_at_utc": "2026-08-07T14:30:00+00:00",
            "duration_ns": 1_000_000_000,
            "restored_file_count": artifact["file_count"] + 2,
            "restored_total_bytes": (
                artifact["total_bytes"]
                + artifact["raw_manifest_size_bytes"]
                + len(canonical_json_bytes(artifact["seal"]))
            ),
            "expected_raw_manifest_sha256": artifact["raw_manifest_sha256"],
            "observed_raw_manifest_sha256": artifact["raw_manifest_sha256"],
            "mismatch_count": 0,
            "verification_result": "passed",
        },
    )
    _append_receipt(
        chain,
        artifact,
        kind="claim-restoration",
        created_at_utc="2026-08-07T14:40:00+00:00",
        result="restored",
        details={"reason_code": "redundancy-and-retrieval-restored"},
    )
    return off_host_copy, retrieval


class PublicationEligibilityTests(unittest.TestCase):
    def test_complete_bound_evidence_is_eligible_and_reverified_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)

            result, verifier, binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
            )

            self.assertTrue(result.eligible)
            self.assertEqual(result.reason_codes, ())
            self.assertEqual(
                result.raw_manifest_sha256, artifact["raw_manifest_sha256"]
            )
            verifier.assert_called_once_with(
                binding.projection_stage,
                binding.finalized_candidate_output,
                binding.review_artifact,
                recovery_artifact=binding.recovery_artifact,
                control_artifact=binding.control_artifact,
                producer_role_id=PRODUCER,
                reviewer_role_id=REVIEWER,
                finalizer_role_id=FINALIZER,
            )
            self.assertFalse(binding.projection_stage.exists())

    def test_expected_artifact_digest_and_size_must_match_exact_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=_attestation(artifact, chain),
                expected_digest="0" * 64,
                expected_size=artifact["raw_manifest_size_bytes"] + 1,
            )

            self.assertFalse(result.eligible)
            self.assertEqual(
                result.reason_codes,
                (
                    "ARTIFACT_MANIFEST_DIGEST_MISMATCH",
                    "ARTIFACT_MANIFEST_SIZE_MISMATCH",
                ),
            )

    def test_passing_retrieval_must_account_for_exact_sealed_artifact(self) -> None:
        mutations = {
            "self-consistent-zero-digest": {
                "expected_raw_manifest_sha256": "0" * 64,
                "observed_raw_manifest_sha256": "0" * 64,
            },
            "incomplete-file-count": {"restored_file_count": 0},
            "zero-restored-bytes": {"restored_total_bytes": 0},
            "seal-bytes-omitted": "omit-seal",
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = _private_root(Path(temporary))
                artifact_path, artifact = _sealed_artifact(root)
                chain = _receipt_chain(artifact)
                updates = (
                    {
                        "restored_total_bytes": artifact["total_bytes"]
                        + artifact["raw_manifest_size_bytes"]
                    }
                    if mutation == "omit-seal"
                    else mutation
                )
                assert isinstance(updates, dict)
                _replace_retrieval_details(chain, artifact, **updates)
                attestation = _attestation(artifact, chain)

                result, _verifier, _binding = _evaluate(
                    root=root,
                    artifact_path=artifact_path,
                    artifact=artifact,
                    chain=chain,
                    attestation=attestation,
                )

                self.assertEqual(result.reason_codes, ("RECEIPT_CHAIN_INVALID",))

    def test_finalized_receipts_are_an_exact_live_chain_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            _artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            supplement = {
                "schema_version": eligibility.RECOVERY_SUPPLEMENT_SCHEMA,
                "copy_verification_receipts": [
                    eligibility._live_receipt_projection(receipt)
                    for receipt in chain[:2]
                ],
                "retrieval_receipt": eligibility._live_receipt_projection(chain[2]),
                "retention_receipt": eligibility._live_receipt_projection(chain[3]),
            }
            payloads = {"recovery-supplement.json": canonical_json_bytes(supplement)}
            eligibility._bind_finalized_receipt_prefix(payloads, chain)

            appended = list(chain)
            _append_receipt(
                appended,
                artifact,
                kind="claim-suspension",
                created_at_utc="2026-08-07T14:00:00+00:00",
                result="suspended",
                details={"reason_code": "operator-control"},
            )
            eligibility._bind_finalized_receipt_prefix(payloads, appended)

            reordered = dict(supplement)
            reordered["copy_verification_receipts"] = list(
                reversed(supplement["copy_verification_receipts"])
            )
            with self.assertRaisesRegex(ValueError, "not the finalized provenance"):
                eligibility._bind_finalized_receipt_prefix(
                    {"recovery-supplement.json": canonical_json_bytes(reordered)},
                    chain,
                )

            malformed = json.loads(canonical_json_bytes(supplement))
            malformed["retrieval_receipt"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "not the finalized provenance"):
                eligibility._bind_finalized_receipt_prefix(
                    {"recovery-supplement.json": canonical_json_bytes(malformed)},
                    chain,
                )

    def test_protocol_valid_capture_primitive_is_not_publication_qualifying(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            primitive = dict(artifact)
            primitive["manifest"] = {
                **artifact["manifest"],
                "record_kind": "experiment-run",
                "identity_bindings": {
                    "capture_kind": "command",
                    "evidence_status": "protocol-valid",
                },
            }
            with patch(
                "tools.cuda_campaign.eligibility.verify_sealed_artifact",
                return_value=primitive,
            ):
                result, _verifier, _binding = _evaluate(
                    root=root,
                    artifact_path=artifact_path,
                    artifact=artifact,
                    chain=chain,
                    attestation=_attestation(artifact, chain),
                )

            self.assertEqual(
                result.reason_codes,
                ("CAPTURE_KIND_NOT_PUBLICATION_QUALIFYING",),
            )

    def test_protocol_valid_managed_sequence_must_have_passed_native_outcome(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            sequence = artifact_path / "sequence"
            sequence.mkdir(mode=0o700)
            summary = {
                "record_kind": "aptus-cuda-campaign-managed-sequence-v1",
                "native_outcome": "failed",
                "reason_code": "JOB_FAILED",
                "evidence_status": "protocol-valid",
                "capture_reason_code": "NONE",
                "stopped_early": True,
                "started_actions": [
                    {
                        "native_outcome": "failed",
                        "reason_code": "JOB_FAILED",
                        "capture_reason_code": "NONE",
                        "terminal": True,
                    }
                ],
            }
            summary_payload = canonical_json_bytes(summary)
            summary_path = sequence / "summary.json"
            summary_path.write_bytes(summary_payload)
            summary_path.chmod(0o600)
            failed = dict(artifact)
            failed["manifest"] = {
                **artifact["manifest"],
                "record_kind": "experiment-run",
                "files": [
                    *artifact["manifest"]["files"],
                    {
                        "relative_path": "sequence/summary.json",
                        "role": "sequence-summary",
                        "size_bytes": len(summary_payload),
                        "sha256": sha256_bytes(summary_payload),
                    },
                ],
                "identity_bindings": {
                    "capture_kind": "managed-sequence",
                    "evidence_status": "protocol-valid",
                },
            }
            with patch(
                "tools.cuda_campaign.eligibility.verify_sealed_artifact",
                return_value=failed,
            ):
                result, _verifier, _binding = _evaluate(
                    root=root,
                    artifact_path=artifact_path,
                    artifact=artifact,
                    chain=chain,
                    attestation=_attestation(artifact, chain),
                )
            self.assertEqual(
                result.reason_codes,
                ("NATIVE_OUTCOME_NOT_PASSED",),
            )

    def test_mutated_artifact_fails_deep_seal_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            (artifact_path / "evidence.bin").write_bytes(b"mutated\n")
            (artifact_path / "evidence.bin").chmod(0o600)
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=_attestation(artifact, chain),
            )

            self.assertEqual(result.reason_codes, ("ARTIFACT_VERIFICATION_FAILED",))

    def test_copy_ids_and_failure_domains_must_both_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            same_copy = _receipt_chain(artifact, copy_ids=(COPY_ONE, COPY_ONE))
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=same_copy,
                attestation=_attestation(artifact, same_copy),
            )
            self.assertEqual(
                result.reason_codes,
                ("RECEIPT_CHAIN_INVALID",),
            )

            same_domain = _receipt_chain(artifact, domains=(DOMAIN_ONE, DOMAIN_ONE))
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=same_domain,
                attestation=_attestation(artifact, same_domain),
            )
            self.assertEqual(
                result.reason_codes,
                (
                    "FAILURE_DOMAIN_COUNT_INSUFFICIENT",
                    "COPY_VERIFICATION_NOT_CURRENT",
                ),
            )

    def test_boolean_off_host_claim_cannot_replace_external_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            attestation["off_host_storage_evidence"] = True
            _readdress_attestation(attestation)
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
            )

            self.assertEqual(
                result.reason_codes,
                ("EXTERNAL_RECOVERY_ATTESTATION_INVALID",),
            )

    def test_external_attestation_must_cross_bind_copy_and_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            attestation["copy_id"] = COPY_ONE
            _readdress_attestation(attestation)
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
            )

            self.assertEqual(
                result.reason_codes,
                ("EXTERNAL_RECOVERY_ATTESTATION_UNBOUND",),
            )

    def test_external_evidence_bytes_must_exist_and_match_every_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            missing_attestation = _attestation(artifact, chain)
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=missing_attestation,
                external_evidence={},
            )
            self.assertEqual(
                result.reason_codes,
                ("EXTERNAL_RECOVERY_EVIDENCE_INVALID",),
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            mismatched_attestation = _attestation(artifact, chain)
            evidence = _materialize_external_evidence(root, mismatched_attestation)
            first = next(iter(evidence.values()))
            first.write_bytes(b"mutated external attestation evidence\n")
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=mismatched_attestation,
                external_evidence=evidence,
            )
            self.assertEqual(
                result.reason_codes,
                ("EXTERNAL_RECOVERY_EVIDENCE_INVALID",),
            )

    def test_external_evidence_references_must_resolve_to_distinct_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            evidence = _materialize_external_evidence(root, attestation)
            references = list(evidence)
            evidence[references[1]] = evidence[references[0]]
            attestation["encryption_in_transit_evidence"]["sha256"] = attestation[
                "off_host_storage_evidence"
            ]["sha256"]
            _readdress_attestation(attestation)

            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
                external_evidence=evidence,
            )

            self.assertEqual(
                result.reason_codes,
                ("EXTERNAL_RECOVERY_EVIDENCE_INVALID",),
            )

    def test_external_evidence_kind_semantics_are_not_opaque_digest_blobs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            evidence = _materialize_external_evidence(root, attestation)
            reference = attestation["encryption_in_transit_evidence"]
            old_id = reference["reference_id"]
            path = evidence.pop(old_id)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["transport_security"] = "plaintext"
            identity = {
                key: value for key, value in record.items() if key != "reference_id"
            }
            new_id = (
                "evidence_" + sha256_bytes(compact_canonical_json_bytes(identity))[:32]
            )
            record["reference_id"] = new_id
            payload = canonical_json_bytes(record)
            replacement = path.with_name(f"{new_id}.bin")
            replacement.write_bytes(payload)
            replacement.chmod(0o600)
            evidence[new_id] = replacement
            attestation["encryption_in_transit_evidence"] = {
                "reference_id": new_id,
                "sha256": sha256_bytes(payload),
            }
            _readdress_attestation(attestation)

            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
                external_evidence=evidence,
            )

            self.assertEqual(
                result.reason_codes,
                ("EXTERNAL_RECOVERY_EVIDENCE_INVALID",),
            )

    def test_later_failed_copy_supersedes_an_earlier_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            _append_receipt(
                chain,
                artifact,
                kind="copy-verification",
                created_at_utc="2026-08-07T14:00:00+00:00",
                result="failed",
                details={
                    "copy_id": COPY_TWO,
                    "failure_domain_id": DOMAIN_TWO,
                    "off_experiment_host": True,
                    "verification_result": "failed",
                },
            )

            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
            )

            self.assertFalse(result.eligible)
            self.assertIn("VERIFIED_COPY_COUNT_INSUFFICIENT", result.reason_codes)
            self.assertIn("COPY_VERIFICATION_NOT_CURRENT", result.reason_codes)
            self.assertIn("OFF_HOST_RETRIEVAL_NOT_CURRENT", result.reason_codes)

    def test_later_failed_retrieval_supersedes_an_earlier_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            details = dict(chain[2]["details"])
            details.update(
                {
                    "started_at_utc": "2026-08-07T13:59:59+00:00",
                    "finished_at_utc": "2026-08-07T14:00:00+00:00",
                    "mismatch_count": 1,
                    "verification_result": "failed",
                }
            )
            _append_receipt(
                chain,
                artifact,
                kind="retrieval",
                created_at_utc="2026-08-07T14:00:00+00:00",
                result="failed",
                details=details,
            )

            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
            )

            self.assertFalse(result.eligible)
            self.assertIn("OFF_HOST_RETRIEVAL_NOT_CURRENT", result.reason_codes)
            self.assertIn("EXTERNAL_RECOVERY_ATTESTATION_UNBOUND", result.reason_codes)

    def test_claim_withdrawal_and_suspension_are_publication_terminal_states(
        self,
    ) -> None:
        for kind, receipt_result, expected_reason in (
            ("claim-withdrawal", "withdrawn", "CLAIM_WITHDRAWN"),
            ("claim-suspension", "suspended", "CLAIM_SUSPENDED"),
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = _private_root(Path(temporary))
                artifact_path, artifact = _sealed_artifact(root)
                chain = _receipt_chain(artifact)
                attestation = _attestation(artifact, chain)
                _append_receipt(
                    chain,
                    artifact,
                    kind=kind,
                    created_at_utc="2026-08-07T14:00:00+00:00",
                    result=receipt_result,
                    details={"reason_code": "operator-control"},
                )

                result, _verifier, _binding = _evaluate(
                    root=root,
                    artifact_path=artifact_path,
                    artifact=artifact,
                    chain=chain,
                    attestation=attestation,
                )

                self.assertFalse(result.eligible)
                self.assertIn(expected_reason, result.reason_codes)

    def test_restoration_requires_post_suspension_evidence_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            _append_receipt(
                chain,
                artifact,
                kind="claim-suspension",
                created_at_utc="2026-08-07T14:00:00+00:00",
                result="suspended",
                details={"reason_code": "copy-state-revalidation"},
            )
            _append_receipt(
                chain,
                artifact,
                kind="claim-restoration",
                created_at_utc="2026-08-07T14:10:00+00:00",
                result="restored",
                details={"reason_code": "unsupported-restoration"},
            )

            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
            )
            self.assertIn("CLAIM_STATE_INVALID", result.reason_codes)

        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            off_host_copy, retrieval = _append_valid_restoration_evidence(
                chain, artifact
            )
            attestation = _attestation(artifact, chain)
            attestation.update(
                {
                    "attested_at_utc": "2026-08-07T14:35:00+00:00",
                    "copy_verification_receipt_id": off_host_copy["receipt_id"],
                    "retrieval_receipt_id": retrieval["receipt_id"],
                }
            )
            _readdress_attestation(attestation)

            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
            )
            self.assertTrue(result.eligible)

            review_before_restoration = {
                **_passed_review(),
                "reviewed_at_utc": "2026-08-07T14:35:00+00:00",
            }
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
                review=review_before_restoration,
            )
            self.assertIn("CLAIM_STATE_INVALID", result.reason_codes)

    def test_external_attestation_requires_custodian_and_time_chronology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            attestation["evidence_custodian_role_id"] = "different-evidence-custodian"
            _readdress_attestation(attestation)
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
            )
            self.assertIn("EXTERNAL_RECOVERY_ATTESTATION_UNBOUND", result.reason_codes)

        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            attestation["attested_at_utc"] = "2026-08-07T11:59:59+00:00"
            _readdress_attestation(attestation)
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
            )
            self.assertIn("EXTERNAL_RECOVERY_ATTESTATION_UNBOUND", result.reason_codes)

    def test_future_dated_independent_review_is_not_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            review = {
                **_passed_review(),
                "reviewed_at_utc": "2026-08-09T12:00:00+00:00",
            }

            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
                review=review,
            )

            self.assertEqual(
                result.reason_codes,
                (
                    "SANITIZER_FINALIZATION_INVALID",
                    "INDEPENDENT_REVIEW_NOT_PASSED",
                ),
            )

    def test_missing_or_mismatched_publication_candidate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)

            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
                candidate_error=ValueError("candidate is absent or cross-bound"),
            )

            self.assertEqual(
                result.reason_codes,
                ("PUBLICATION_CANDIDATE_INVALID",),
            )

    def test_sealed_candidate_pins_artifact_packet_receipts_and_typed_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            source_root = root / "source-a"
            source_root.mkdir(mode=0o700)
            artifact_path, artifact = _sealed_artifact(source_root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            evidence = _materialize_external_evidence(source_root, attestation)
            sanitizer = _materialize_candidate_sanitizer(source_root, artifact_path)
            candidate_path = root / "sealed-publication-candidate"
            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch("tools.cuda_campaign.eligibility._bind_finalized_receipt_prefix"),
            ):
                sealed = seal_publication_candidate(
                    candidate_path,
                    campaign_id=CAMPAIGN_ID,
                    claim_key=CLAIM_KEY,
                    candidate_producer_role_id=CANDIDATE_PRODUCER,
                    created_at_utc="2026-08-08T11:30:00+00:00",
                    artifact=artifact_path,
                    receipts=chain,
                    external_recovery_attestation=attestation,
                    external_evidence=evidence,
                    sanitizer=sanitizer,
                )
                observed = verify_publication_candidate(
                    PublicationCandidateBinding(
                        artifact=candidate_path,
                        campaign_id=CAMPAIGN_ID,
                        claim_key=CLAIM_KEY,
                        candidate_producer_role_id=CANDIDATE_PRODUCER,
                    ),
                    artifact=artifact_path,
                    receipts=chain,
                    external_recovery_attestation=attestation,
                    external_evidence=evidence,
                    sanitizer=sanitizer,
                    now_utc=NOW,
                )
            self.assertEqual(
                observed["candidate_id"],
                sealed["publication_candidate"]["candidate_id"],
            )
            self.assertEqual(
                [item["evidence_kind"] for item in observed["external_evidence"]],
                [
                    "off-host-storage",
                    "encryption-in-transit",
                    "encryption-at-rest",
                    "key-custody",
                    "recovery-procedure",
                ],
            )

            source_b = root / "source-b"
            source_b.mkdir(mode=0o700)
            artifact_b_path, artifact_b = _sealed_artifact(
                source_b,
                artifact_id="artifact_" + "b" * 32,
                payload=b"different protected evidence\n",
            )
            chain_b = _receipt_chain(artifact_b)
            attestation_b = _attestation(artifact_b, chain_b)
            evidence_b = _materialize_external_evidence(source_b, attestation_b)
            sanitizer_b = _materialize_candidate_sanitizer(source_b, artifact_b_path)
            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch("tools.cuda_campaign.eligibility._bind_finalized_receipt_prefix"),
                self.assertRaises(ValueError),
            ):
                verify_publication_candidate(
                    PublicationCandidateBinding(
                        artifact=candidate_path,
                        campaign_id=CAMPAIGN_ID,
                        claim_key=CLAIM_KEY,
                        candidate_producer_role_id=CANDIDATE_PRODUCER,
                    ),
                    artifact=artifact_b_path,
                    receipts=chain_b,
                    external_recovery_attestation=attestation_b,
                    external_evidence=evidence_b,
                    sanitizer=sanitizer_b,
                    now_utc=NOW,
                )

    def test_real_finalization_candidate_and_eligibility_integrate_end_to_end(
        self,
    ) -> None:
        from tests.tools.test_cuda_campaign_sanitizer import (  # noqa: PLC0415
            _after_seals,
            _after_timestamp,
            _sealed_context,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            recovery, control, _source = _sealed_context(root)
            verified_recovery = verify_sealed_artifact(recovery)
            receipts = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((control / "receipts").iterdir())
            ]
            projection = project_verified_recovery_supplement(
                recovery_artifact=recovery, control_artifact=control
            )
            stage = root / "projection-stage"
            write_projection_stage(stage, projection)
            reviewed_at = _after_seals(recovery, control)
            review_artifact = root / "sealed-review"
            seal_projection_review(
                stage,
                review_artifact,
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id=PRODUCER,
                reviewer_role_id=REVIEWER,
                review_id="review_" + "a" * 32,
                reviewed_at_utc=reviewed_at,
            )
            finalized_at = _after_timestamp(reviewed_at)
            finalized_candidate = root / "finalized-candidate"
            finalize_projection_stage(
                stage,
                finalized_candidate,
                review_artifact,
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id=PRODUCER,
                reviewer_role_id=REVIEWER,
                finalizer_role_id=FINALIZER,
                finalized_at_utc=finalized_at,
            )
            sanitizer = FinalizedSanitizerBinding(
                projection_stage=stage,
                finalized_candidate_output=finalized_candidate,
                review_artifact=review_artifact,
                recovery_artifact=recovery,
                control_artifact=control,
                producer_role_id=PRODUCER,
                reviewer_role_id=REVIEWER,
                finalizer_role_id=FINALIZER,
            )
            attestation = _attestation(
                verified_recovery,
                receipts,
                attested_at_utc=reviewed_at,
            )
            evidence = _materialize_external_evidence(root, attestation)

            with self.assertRaises(ValueError):
                seal_publication_candidate(
                    root / "premature-publication-candidate",
                    campaign_id=CAMPAIGN_ID,
                    claim_key=CLAIM_KEY,
                    candidate_producer_role_id=CANDIDATE_PRODUCER,
                    created_at_utc=reviewed_at,
                    artifact=recovery,
                    receipts=receipts,
                    external_recovery_attestation=attestation,
                    external_evidence=evidence,
                    sanitizer=sanitizer,
                )

            candidate_created_at = _after_timestamp(finalized_at)
            candidate_path = root / "sealed-publication-candidate"
            seal_publication_candidate(
                candidate_path,
                campaign_id=CAMPAIGN_ID,
                claim_key=CLAIM_KEY,
                candidate_producer_role_id=CANDIDATE_PRODUCER,
                created_at_utc=candidate_created_at,
                artifact=recovery,
                receipts=receipts,
                external_recovery_attestation=attestation,
                external_evidence=evidence,
                sanitizer=sanitizer,
            )
            evaluated_at = (
                datetime.fromisoformat(candidate_created_at) + timedelta(seconds=1)
            ).isoformat()
            result = evaluate_publication_eligibility(
                artifact=recovery,
                expected_protected_artifact_id=verified_recovery[
                    "protected_artifact_id"
                ],
                expected_raw_manifest_sha256=verified_recovery["raw_manifest_sha256"],
                expected_raw_manifest_size_bytes=verified_recovery[
                    "raw_manifest_size_bytes"
                ],
                receipts=receipts,
                external_recovery_attestation=attestation,
                external_evidence=evidence,
                now_utc=evaluated_at,
                sanitizer=sanitizer,
                publication_candidate=PublicationCandidateBinding(
                    artifact=candidate_path,
                    campaign_id=CAMPAIGN_ID,
                    claim_key=CLAIM_KEY,
                    candidate_producer_role_id=CANDIDATE_PRODUCER,
                ),
            )

            self.assertTrue(result.eligible, result.reason_codes)

            zero_retrieval_receipts = list(receipts)
            _replace_retrieval_details(
                zero_retrieval_receipts,
                verified_recovery,
                expected_raw_manifest_sha256="0" * 64,
                observed_raw_manifest_sha256="0" * 64,
                restored_file_count=0,
                restored_total_bytes=0,
            )
            zero_attestation = _attestation(
                verified_recovery,
                zero_retrieval_receipts,
                attested_at_utc=reviewed_at,
            )
            zero_evidence = _materialize_external_evidence(root, zero_attestation)
            with self.assertRaisesRegex(ValueError, "retrieval binding is invalid"):
                seal_publication_candidate(
                    root / "zero-retrieval-publication-candidate",
                    campaign_id=CAMPAIGN_ID,
                    claim_key=CLAIM_KEY,
                    candidate_producer_role_id=CANDIDATE_PRODUCER,
                    created_at_utc=candidate_created_at,
                    artifact=recovery,
                    receipts=zero_retrieval_receipts,
                    external_recovery_attestation=zero_attestation,
                    external_evidence=zero_evidence,
                    sanitizer=sanitizer,
                )

            alternate_receipts = list(receipts)
            _replace_retrieval_details(
                alternate_receipts,
                verified_recovery,
                destination_restore_id="restore_" + "9" * 32,
            )
            alternate_attestation = _attestation(
                verified_recovery,
                alternate_receipts,
                attested_at_utc=reviewed_at,
            )
            alternate_evidence = _materialize_external_evidence(
                root,
                alternate_attestation,
            )
            with self.assertRaisesRegex(ValueError, "finalized provenance prefix"):
                seal_publication_candidate(
                    root / "cross-bound-publication-candidate",
                    campaign_id=CAMPAIGN_ID,
                    claim_key=CLAIM_KEY,
                    candidate_producer_role_id=CANDIDATE_PRODUCER,
                    created_at_utc=candidate_created_at,
                    artifact=recovery,
                    receipts=alternate_receipts,
                    external_recovery_attestation=alternate_attestation,
                    external_evidence=alternate_evidence,
                    sanitizer=sanitizer,
                )

            with self.assertRaises(ValueError):
                verify_publication_candidate(
                    PublicationCandidateBinding(
                        artifact=candidate_path,
                        campaign_id=CAMPAIGN_ID,
                        claim_key=CLAIM_KEY,
                        candidate_producer_role_id=CANDIDATE_PRODUCER,
                    ),
                    artifact=control,
                    receipts=receipts,
                    external_recovery_attestation=attestation,
                    external_evidence=evidence,
                    sanitizer=sanitizer,
                    now_utc=evaluated_at,
                )

            withdrawn_receipts = list(receipts)
            withdrawal_at = (
                datetime.fromisoformat(candidate_created_at) + timedelta(seconds=1)
            ).isoformat()
            _append_receipt(
                withdrawn_receipts,
                verified_recovery,
                kind="claim-withdrawal",
                created_at_utc=withdrawal_at,
                result="withdrawn",
                details={"reason_code": "operator-control"},
            )
            withdrawn_result = evaluate_publication_eligibility(
                artifact=recovery,
                expected_protected_artifact_id=verified_recovery[
                    "protected_artifact_id"
                ],
                expected_raw_manifest_sha256=verified_recovery["raw_manifest_sha256"],
                expected_raw_manifest_size_bytes=verified_recovery[
                    "raw_manifest_size_bytes"
                ],
                receipts=withdrawn_receipts,
                external_recovery_attestation=attestation,
                external_evidence=evidence,
                now_utc=(
                    datetime.fromisoformat(withdrawal_at) + timedelta(seconds=1)
                ).isoformat(),
                sanitizer=sanitizer,
                publication_candidate=PublicationCandidateBinding(
                    artifact=candidate_path,
                    campaign_id=CAMPAIGN_ID,
                    claim_key=CLAIM_KEY,
                    candidate_producer_role_id=CANDIDATE_PRODUCER,
                ),
            )
            self.assertIn("CLAIM_WITHDRAWN", withdrawn_result.reason_codes)
            self.assertIn(
                "PUBLICATION_CANDIDATE_INVALID", withdrawn_result.reason_codes
            )

    def test_stale_copy_and_retrieval_receipts_fail_current_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
                now=datetime(2027, 2, 5, 13, tzinfo=timezone.utc),
            )

            self.assertEqual(
                result.reason_codes,
                (
                    "COPY_VERIFICATION_NOT_CURRENT",
                    "EXTERNAL_RECOVERY_ATTESTATION_UNBOUND",
                    "OFF_HOST_RETRIEVAL_NOT_CURRENT",
                ),
            )

    def test_retention_expiry_and_renewal_lead_are_independent_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            due = _receipt_chain(
                artifact,
                retention_deadline="2028-08-08T12:00:00+00:00",
            )
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=due,
                attestation=_attestation(artifact, due),
                now=datetime(2028, 6, 1, 12, tzinfo=timezone.utc),
            )
            self.assertIn("RETENTION_RENEWAL_NOT_CURRENT", result.reason_codes)

            expired = _receipt_chain(
                artifact,
                retention_deadline="2028-08-08T12:00:00+00:00",
            )
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=expired,
                attestation=_attestation(artifact, expired),
                now=datetime(2028, 8, 9, 12, tzinfo=timezone.utc),
            )
            self.assertIn("RETENTION_NOT_CURRENT", result.reason_codes)
            self.assertNotIn("RETENTION_RENEWAL_NOT_CURRENT", result.reason_codes)

    def test_receipt_content_id_and_chain_are_independently_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            chain[0]["receipt_id"] = "receipt_" + "9" * 32
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
            )

            self.assertEqual(result.reason_codes, ("RECEIPT_CHAIN_INVALID",))

    def test_finalized_sanitizer_and_independent_review_both_gate_publication(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            artifact_path, artifact = _sealed_artifact(root)
            chain = _receipt_chain(artifact)
            attestation = _attestation(artifact, chain)
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
                review_error=ValueError("tampered finalized packet"),
            )
            self.assertEqual(result.reason_codes, ("SANITIZER_FINALIZATION_INVALID",))

            failed_review = {
                "result": "failed",
                "reason_code": "SANITIZATION_FAILURE",
                "producer_role_id": PRODUCER,
                "reviewer_role_id": REVIEWER,
            }
            result, _verifier, _binding = _evaluate(
                root=root,
                artifact_path=artifact_path,
                artifact=artifact,
                chain=chain,
                attestation=attestation,
                review=failed_review,
            )
            self.assertEqual(result.reason_codes, ("INDEPENDENT_REVIEW_NOT_PASSED",))


if __name__ == "__main__":
    unittest.main()
