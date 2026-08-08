from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from tests.tools.test_cuda_campaign_eligibility import (
    ARTIFACT_ID,
    CAMPAIGN_ID,
    CANDIDATE_PRODUCER,
    CLAIM_KEY,
    NOW,
    _attestation,
    _materialize_candidate_sanitizer,
    _materialize_external_evidence,
    _passed_review,
    _private_root,
    _receipt_chain,
    _sealed_artifact,
)
from tools.cuda_campaign.contracts import canonical_json_bytes, sha256_bytes
from tools.cuda_campaign import publication
from tools.cuda_campaign.eligibility import (
    RECOVERY_SUPPLEMENT_SCHEMA,
    PublicationCandidateBinding,
    _live_receipt_projection,
    evaluate_publication_eligibility,
    seal_publication_candidate,
)
from tools.cuda_campaign.publication import (
    FINALIZED_CANDIDATE_ALLOWLIST,
    PublicationError,
    publish_candidate,
    verify_published_output,
)
from tools.cuda_campaign.storage import AppendOnlyReceiptStore, ReceiptChainError


class PublicationCommitTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, object]:
        artifact_path, artifact = _sealed_artifact(root)
        receipts = _receipt_chain(artifact)
        receipt_store = AppendOnlyReceiptStore(root / "receipts")
        for receipt in receipts:
            receipt_store.append(
                kind=receipt["kind"],
                issuer_role_id=receipt["issuer_role_id"],
                protected_artifact_id=receipt["protected_artifact_id"],
                raw_manifest_sha256=receipt["raw_manifest_sha256"],
                raw_manifest_size_bytes=receipt["raw_manifest_size_bytes"],
                result=receipt["result"],
                details=receipt["details"],
                receipt_id=receipt["receipt_id"],
                previous_receipt_id=receipt["previous_receipt_id"],
                created_at_utc=receipt["created_at_utc"],
            )
        attestation = _attestation(artifact, receipts)
        evidence = _materialize_external_evidence(root, attestation)
        sanitizer = _materialize_candidate_sanitizer(root, artifact_path)
        for name in sorted(FINALIZED_CANDIDATE_ALLOWLIST - {"claim-boundary.json"}):
            path = sanitizer.finalized_candidate_output / name
            if name == "SHA256SUMS":
                payload = b"placeholder checksums\n"
            elif name == "recovery-supplement.json":
                payload = canonical_json_bytes(
                    {
                        "schema_version": RECOVERY_SUPPLEMENT_SCHEMA,
                        "copy_verification_receipts": [
                            _live_receipt_projection(receipt)
                            for receipt in receipts[:2]
                        ],
                        "retrieval_receipt": _live_receipt_projection(receipts[2]),
                        "retention_receipt": _live_receipt_projection(receipts[3]),
                    }
                )
            else:
                payload = canonical_json_bytes({"reviewed_fixture": name})
            path.write_bytes(payload)
            path.chmod(0o600)
        candidate_path = root / "sealed-publication-candidate"
        with patch(
            "tools.cuda_campaign.eligibility.verify_finalized_projection",
            return_value=_passed_review(),
        ):
            seal_publication_candidate(
                candidate_path,
                campaign_id=CAMPAIGN_ID,
                claim_key=CLAIM_KEY,
                candidate_producer_role_id=CANDIDATE_PRODUCER,
                created_at_utc="2026-08-08T11:30:00+00:00",
                artifact=artifact_path,
                receipts=receipts,
                external_recovery_attestation=attestation,
                external_evidence=evidence,
                sanitizer=sanitizer,
            )
        return {
            "artifact_path": artifact_path,
            "artifact": artifact,
            "receipts": receipts,
            "receipt_store": receipt_store,
            "attestation": attestation,
            "evidence": evidence,
            "sanitizer": sanitizer,
            "candidate": PublicationCandidateBinding(
                artifact=candidate_path,
                campaign_id=CAMPAIGN_ID,
                claim_key=CLAIM_KEY,
                candidate_producer_role_id=CANDIDATE_PRODUCER,
            ),
        }

    def _publish(
        self,
        root: Path,
        fixture: dict[str, object],
        *,
        destination_name: str = "published",
        decision_name: str = "sealed-publication-decision",
        **kwargs: object,
    ):
        artifact = fixture["artifact"]
        assert isinstance(artifact, dict)
        clock_values = kwargs.pop("_clock_values", (NOW.isoformat(), NOW.isoformat()))
        receipt_store = fixture["receipt_store"]
        assert isinstance(receipt_store, AppendOnlyReceiptStore)
        values = {
            "destination_id": "destination_" + "d" * 32,
            "evaluator_role_id": "publication-evaluator",
            "tool_source_sha256": "f" * 64,
            "artifact": fixture["artifact_path"],
            "expected_protected_artifact_id": ARTIFACT_ID,
            "expected_raw_manifest_sha256": artifact["raw_manifest_sha256"],
            "expected_raw_manifest_size_bytes": artifact["raw_manifest_size_bytes"],
            "receipt_store": receipt_store,
            "external_recovery_attestation": fixture["attestation"],
            "external_evidence": fixture["evidence"],
            "sanitizer": fixture["sanitizer"],
            "publication_candidate": fixture["candidate"],
        }
        values.update(kwargs)

        with patch("tools.cuda_campaign.publication.utc_now", side_effect=clock_values):
            return publish_candidate(
                root / destination_name,
                root / decision_name,
                **values,
            )

    def test_evaluate_is_read_only_and_distinct_from_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            artifact = fixture["artifact"]
            assert isinstance(artifact, dict)
            with patch(
                "tools.cuda_campaign.eligibility.verify_finalized_projection",
                return_value=_passed_review(),
            ):
                result = evaluate_publication_eligibility(
                    artifact=fixture["artifact_path"],
                    expected_protected_artifact_id=ARTIFACT_ID,
                    expected_raw_manifest_sha256=artifact["raw_manifest_sha256"],
                    expected_raw_manifest_size_bytes=artifact[
                        "raw_manifest_size_bytes"
                    ],
                    receipts=fixture["receipts"],
                    external_recovery_attestation=fixture["attestation"],
                    external_evidence=fixture["evidence"],
                    now_utc=NOW,
                    sanitizer=fixture["sanitizer"],
                    publication_candidate=fixture["candidate"],
                )
            self.assertTrue(result.eligible, result.reason_codes)
            self.assertFalse((root / "published").exists())
            self.assertFalse((root / "sealed-publication-decision").exists())

    def test_publish_reverifies_and_commits_exact_allowlisted_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            with patch(
                "tools.cuda_campaign.eligibility.verify_finalized_projection",
                return_value=_passed_review(),
            ):
                result = self._publish(root, fixture)
            self.assertEqual(result["publication_status"], "published")
            self.assertEqual(
                verify_published_output(
                    root / "published",
                    decision_artifact=root / "sealed-publication-decision",
                )["decision_id"],
                result["decision_id"],
            )
            self.assertTrue((root / "sealed-publication-decision").is_dir())
            self.assertFalse(
                (root / f".publication-stage-{result['decision_id']}").exists()
            )
            decision = json.loads(
                (root / "published" / "publication-decision.json").read_bytes()
            )
            self.assertTrue(decision["eligible"])
            self.assertEqual(decision["reason_codes"], [])
            self.assertEqual(
                decision["receipt_chain"]["head_receipt_id"],
                fixture["receipts"][-1]["receipt_id"],
            )
            self.assertEqual(
                {
                    item["relative_path"]
                    for item in decision["candidate"][
                        "finalized_candidate_file_inventory"
                    ]
                },
                FINALIZED_CANDIDATE_ALLOWLIST,
            )

    def test_public_checksums_cannot_replace_the_protected_decision_anchor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            with patch(
                "tools.cuda_campaign.eligibility.verify_finalized_projection",
                return_value=_passed_review(),
            ):
                self._publish(root, fixture)
            published = root / "published"
            binding_path = published / "publication-decision-binding.json"
            binding = json.loads(binding_path.read_bytes())
            binding["decision_artifact_id"] = "artifact_" + "e" * 32
            binding_path.write_bytes(canonical_json_bytes(binding))
            binding_path.chmod(0o600)
            names = sorted(
                item.name
                for item in published.iterdir()
                if item.name != publication.PUBLIC_CHECKSUM_NAME
            )
            checksum = "".join(
                f"{sha256_bytes((published / name).read_bytes())}  {name}\n"
                for name in names
            ).encode("utf-8")
            checksum_path = published / publication.PUBLIC_CHECKSUM_NAME
            checksum_path.write_bytes(checksum)
            checksum_path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "decision artifact binding"):
                verify_published_output(
                    published,
                    decision_artifact=root / "sealed-publication-decision",
                )

    def test_commit_time_uses_a_fresh_clock_and_rejects_expired_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            times = iter(
                (
                    NOW.isoformat(),
                    (NOW + timedelta(days=181)).isoformat(),
                )
            )
            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                self.assertRaisesRegex(ValueError, "eligibility did not pass"),
            ):
                self._publish(root, fixture, _clock_values=times)
            self.assertFalse((root / "published").exists())
            self.assertFalse((root / "sealed-publication-decision").exists())

    def test_commit_time_cannot_move_backward(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            times = iter(
                (
                    NOW.isoformat(),
                    (NOW - timedelta(seconds=1)).isoformat(),
                )
            )
            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                self.assertRaisesRegex(ValueError, "moved backward"),
            ):
                self._publish(root, fixture, _clock_values=times)
            self.assertFalse((root / "published").exists())

    def test_candidate_swap_between_evaluation_and_commit_cannot_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            candidate = fixture["candidate"]
            assert isinstance(candidate, PublicationCandidateBinding)

            def mutate_candidate() -> None:
                path = candidate.artifact / "publication-candidate.json"
                payload = bytearray(path.read_bytes())
                payload[-2] = ord("0") if payload[-2] != ord("0") else ord("1")
                path.write_bytes(bytes(payload))
                path.chmod(0o600)

            real_eligibility = publication._eligibility_and_candidate
            evaluation_count = 0

            def mutate_before_final_evaluation(**kwargs: object):
                nonlocal evaluation_count
                evaluation_count += 1
                if evaluation_count == 2:
                    mutate_candidate()
                return real_eligibility(**kwargs)

            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch(
                    "tools.cuda_campaign.publication._eligibility_and_candidate",
                    side_effect=mutate_before_final_evaluation,
                ),
                self.assertRaises(ValueError),
            ):
                self._publish(root, fixture)
            self.assertFalse((root / "published").exists())
            self.assertFalse((root / "sealed-publication-decision").exists())

    def test_exported_publish_api_has_no_caller_controlled_clock(self) -> None:
        parameters = inspect.signature(publish_candidate).parameters
        self.assertNotIn("now_utc", parameters)
        self.assertNotIn("_utc_now", parameters)
        self.assertNotIn("before_commit", parameters)
        self.assertNotIn("receipts", parameters)
        self.assertNotIn("receipt_supplier", parameters)
        self.assertNotIn("receipt_transaction", parameters)
        self.assertEqual(parameters["receipt_store"].default, inspect.Parameter.empty)

    def test_direct_forged_exited_and_foreign_transactions_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            store = fixture["receipt_store"]
            assert isinstance(store, AppendOnlyReceiptStore)
            with self.assertRaises(TypeError):
                publication._LockedReceiptTransaction(store)
            forged = object.__new__(publication._LockedReceiptTransaction)
            with self.assertRaisesRegex(ReceiptChainError, "inactive"):
                store._read_chain_locked(forged)

            with store.transaction() as transaction:
                self.assertEqual(transaction.read_chain(), fixture["receipts"])
            with self.assertRaisesRegex(ReceiptChainError, "inactive"):
                transaction.read_chain()

            foreign = AppendOnlyReceiptStore(root / "foreign-receipts")
            with store.transaction() as transaction:
                with self.assertRaisesRegex(ReceiptChainError, "exact store"):
                    foreign._read_chain_locked(transaction)

    def test_transaction_rejects_receipt_store_root_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            store = fixture["receipt_store"]
            assert isinstance(store, AppendOnlyReceiptStore)
            displaced = root / "displaced-receipts"
            with self.assertRaisesRegex(ReceiptChainError, "identity changed"):
                with store.transaction() as transaction:
                    store.root.rename(displaced)
                    store.root.mkdir(mode=0o700)
                    transaction.read_chain()

    def test_withdrawal_or_suspension_appended_before_publish_is_not_stale(
        self,
    ) -> None:
        for kind, result in (
            ("claim-withdrawal", "withdrawn"),
            ("claim-suspension", "suspended"),
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = _private_root(Path(temporary))
                fixture = self._fixture(root)
                artifact = fixture["artifact"]
                store = fixture["receipt_store"]
                assert isinstance(artifact, dict)
                assert isinstance(store, AppendOnlyReceiptStore)
                store.append(
                    kind=kind,
                    issuer_role_id="phase2-evidence-custodian",
                    protected_artifact_id=artifact["protected_artifact_id"],
                    raw_manifest_sha256=artifact["raw_manifest_sha256"],
                    raw_manifest_size_bytes=artifact["raw_manifest_size_bytes"],
                    result=result,
                    details={"reason_code": "operator-control"},
                    created_at_utc="2026-08-08T11:59:00+00:00",
                )
                with (
                    patch(
                        "tools.cuda_campaign.eligibility.verify_finalized_projection",
                        return_value=_passed_review(),
                    ),
                    self.assertRaises(ValueError),
                ):
                    self._publish(root, fixture)
                self.assertFalse((root / "published").exists())
                self.assertFalse((root / "sealed-publication-decision").exists())

    def test_receipt_tail_movement_between_evaluation_and_commit_is_refused(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            artifact = fixture["artifact"]
            store = fixture["receipt_store"]
            assert isinstance(artifact, dict)
            assert isinstance(store, AppendOnlyReceiptStore)
            real_snapshot = publication._snapshot_public_payloads
            real_read_chain = AppendOnlyReceiptStore._read_chain_locked
            active_transaction: list[object] = []

            def capture_transaction(
                active_store: AppendOnlyReceiptStore, transaction: object
            ):
                if not active_transaction:
                    active_transaction.append(transaction)
                return real_read_chain(active_store, transaction)

            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch.object(
                    AppendOnlyReceiptStore,
                    "_read_chain_locked",
                    autospec=True,
                    side_effect=capture_transaction,
                ),
                patch(
                    "tools.cuda_campaign.publication._snapshot_public_payloads"
                ) as snapshot,
                self.assertRaises(ValueError),
            ):

                def snapshot_then_move(**kwargs: object):
                    payloads = real_snapshot(**kwargs)
                    transaction = active_transaction[0]
                    transaction.append(
                        kind="claim-withdrawal",
                        issuer_role_id="phase2-evidence-custodian",
                        protected_artifact_id=artifact["protected_artifact_id"],
                        raw_manifest_sha256=artifact["raw_manifest_sha256"],
                        raw_manifest_size_bytes=artifact["raw_manifest_size_bytes"],
                        result="withdrawn",
                        details={"reason_code": "operator-control"},
                        created_at_utc="2026-08-08T11:59:00+00:00",
                    )
                    return payloads

                snapshot.side_effect = snapshot_then_move
                self._publish(
                    root,
                    fixture,
                )
            self.assertFalse((root / "published").exists())

    def test_existing_destination_and_older_decision_never_authorize_new_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            existing = root / "published"
            existing.mkdir(mode=0o700)
            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                self.assertRaises(FileExistsError),
            ):
                self._publish(root, fixture)
            self.assertEqual(list(existing.iterdir()), [])
            self.assertFalse((root / "sealed-publication-decision").exists())

    def test_atomic_commit_never_replaces_a_racing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            real_atomic = publication._atomic_no_replace

            def create_destination_then_commit(source: Path, destination: Path) -> None:
                destination.mkdir(mode=0o700)
                marker = destination / "existing.txt"
                marker.write_bytes(b"preexisting destination\n")
                marker.chmod(0o600)
                real_atomic(source, destination)

            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch(
                    "tools.cuda_campaign.publication._atomic_no_replace",
                    side_effect=create_destination_then_commit,
                ),
                self.assertRaises(FileExistsError),
            ):
                self._publish(root, fixture)
            self.assertEqual(
                (root / "published" / "existing.txt").read_bytes(),
                b"preexisting destination\n",
            )

    def test_atomic_helper_signal_after_real_rename_is_reconciled_and_rolled_back(
        self,
    ) -> None:
        for signal_type in (KeyboardInterrupt, SystemExit):
            with (
                self.subTest(signal=signal_type.__name__),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = _private_root(Path(temporary))
                fixture = self._fixture(root)
                real_atomic = publication._atomic_no_replace

                def rename_then_signal(source: Path, destination: Path) -> None:
                    real_atomic(source, destination)
                    if destination == root / "published":
                        raise signal_type("simulated atomic-boundary signal")

                with (
                    patch(
                        "tools.cuda_campaign.eligibility.verify_finalized_projection",
                        return_value=_passed_review(),
                    ),
                    patch(
                        "tools.cuda_campaign.publication._atomic_no_replace",
                        side_effect=rename_then_signal,
                    ),
                    self.assertRaises(PublicationError) as raised,
                ):
                    self._publish(root, fixture)
                self.assertIsInstance(raised.exception.__cause__, signal_type)
                self.assertFalse((root / "published").exists())

    def test_stage_mutation_at_atomic_boundary_is_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            real_atomic = publication._atomic_no_replace

            def mutate_stage_then_commit(source: Path, destination: Path) -> None:
                if source.name.startswith(".publication-stage-"):
                    changed = source / "claim-boundary.json"
                    changed.write_bytes(canonical_json_bytes({"raced": True}))
                    changed.chmod(0o600)
                real_atomic(source, destination)

            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch(
                    "tools.cuda_campaign.publication._atomic_no_replace",
                    side_effect=mutate_stage_then_commit,
                ),
                self.assertRaisesRegex(ValueError, "rolled back"),
            ):
                self._publish(root, fixture)
            self.assertFalse((root / "published").exists())
            self.assertTrue(
                any(
                    path.name.startswith(".publication-stage-")
                    for path in root.iterdir()
                )
            )

    def test_stage_directory_replacement_cannot_cross_commit_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            real_atomic = publication._atomic_no_replace

            def replace_stage_then_commit(source: Path, destination: Path) -> None:
                if source.name.startswith(".publication-stage-"):
                    displaced = source.parent / ".displaced-verified-stage"
                    source.rename(displaced)
                    source.mkdir(mode=0o700)
                    marker = source / "unauthorized.txt"
                    marker.write_bytes(b"unauthorized\n")
                    marker.chmod(0o600)
                real_atomic(source, destination)

            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch(
                    "tools.cuda_campaign.publication._atomic_no_replace",
                    side_effect=replace_stage_then_commit,
                ),
                self.assertRaisesRegex(ValueError, "rolled back"),
            ):
                self._publish(root, fixture)
            self.assertFalse((root / "published").exists())
            self.assertTrue((root / ".displaced-verified-stage").is_dir())
            self.assertTrue(
                any(
                    path.name.startswith(".publication-stage-")
                    for path in root.iterdir()
                )
            )

    def test_parent_fsync_failure_after_rename_is_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            real_fsync = publication._fsync_directory
            fsync_count = 0

            def fail_commit_fsync(path: Path) -> None:
                nonlocal fsync_count
                fsync_count += 1
                if fsync_count == 2:
                    raise OSError("simulated commit durability failure")
                real_fsync(path)

            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch(
                    "tools.cuda_campaign.publication._fsync_directory",
                    side_effect=fail_commit_fsync,
                ),
                self.assertRaisesRegex(ValueError, "rolled back"),
            ):
                self._publish(root, fixture)
            self.assertFalse((root / "published").exists())
            self.assertTrue(
                any(
                    path.name.startswith(".publication-stage-")
                    for path in root.iterdir()
                )
            )

    def test_keyboard_interrupt_after_rename_is_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            real_fsync = publication._fsync_directory
            fsync_count = 0

            def interrupt_commit_fsync(path: Path) -> None:
                nonlocal fsync_count
                fsync_count += 1
                if fsync_count == 2:
                    raise KeyboardInterrupt("simulated operator interrupt")
                real_fsync(path)

            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch(
                    "tools.cuda_campaign.publication._fsync_directory",
                    side_effect=interrupt_commit_fsync,
                ),
                self.assertRaises(PublicationError) as raised,
            ):
                self._publish(root, fixture)
            self.assertIsInstance(raised.exception.__cause__, KeyboardInterrupt)
            self.assertFalse((root / "published").exists())

    def test_system_exit_before_locked_publisher_returns_uses_commit_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            real_publish_locked = publication._publish_candidate_locked

            def publish_then_exit(*args: object, **kwargs: object):
                real_publish_locked(*args, **kwargs)
                raise SystemExit("simulated shutdown before return")

            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch(
                    "tools.cuda_campaign.publication._publish_candidate_locked",
                    side_effect=publish_then_exit,
                ),
                self.assertRaises(PublicationError) as raised,
            ):
                self._publish(root, fixture)
            self.assertIsInstance(raised.exception.__cause__, SystemExit)
            self.assertFalse((root / "published").exists())

    def test_system_exit_at_transaction_exit_rolls_back_completed_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            real_transaction = AppendOnlyReceiptStore.transaction

            @contextmanager
            def interrupted_transaction(store: AppendOnlyReceiptStore):
                with real_transaction(store) as transaction:
                    yield transaction
                raise SystemExit("simulated transaction-exit shutdown")

            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch.object(
                    AppendOnlyReceiptStore,
                    "transaction",
                    new=interrupted_transaction,
                ),
                self.assertRaises(PublicationError) as raised,
            ):
                self._publish(root, fixture)
            self.assertIsInstance(raised.exception.__cause__, SystemExit)
            self.assertFalse((root / "published").exists())

    def test_reclaimed_stage_path_cannot_block_post_commit_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            real_atomic = publication._atomic_no_replace
            real_verify = publication.verify_published_output
            reclaimed_stage: list[Path] = []

            def commit_then_reclaim(source: Path, destination: Path) -> None:
                real_atomic(source, destination)
                if destination == root / "published":
                    source.mkdir(mode=0o700)
                    reclaimed_stage.append(source)

            def fail_public_verification(
                path: Path, *, decision_artifact: Path
            ) -> dict[str, object]:
                if path == root / "published":
                    raise ValueError("simulated post-commit verification failure")
                return real_verify(path, decision_artifact=decision_artifact)

            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                patch(
                    "tools.cuda_campaign.publication._atomic_no_replace",
                    side_effect=commit_then_reclaim,
                ),
                patch(
                    "tools.cuda_campaign.publication.verify_published_output",
                    side_effect=fail_public_verification,
                ),
                self.assertRaisesRegex(ValueError, "rolled back"),
            ):
                self._publish(root, fixture)
            self.assertFalse((root / "published").exists())
            self.assertEqual(len(reclaimed_stage), 1)
            self.assertTrue(reclaimed_stage[0].is_dir())

    def test_receipt_root_or_lock_swap_at_transaction_exit_rolls_back_commit(
        self,
    ) -> None:
        for attack in ("root", "lock"):
            with (
                self.subTest(attack=attack),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = _private_root(Path(temporary))
                fixture = self._fixture(root)
                receipt_store = fixture["receipt_store"]
                assert isinstance(receipt_store, AppendOnlyReceiptStore)
                real_publish_locked = publication._publish_candidate_locked

                def publish_then_swap(*args: object, **kwargs: object):
                    result = real_publish_locked(*args, **kwargs)
                    if attack == "root":
                        receipt_store.root.rename(root / "displaced-receipt-root")
                        receipt_store.root.mkdir(mode=0o700)
                    else:
                        lock = receipt_store.root / ".receipts.lock"
                        lock.rename(receipt_store.root / ".displaced-receipts.lock")
                        lock.write_bytes(b"")
                        lock.chmod(0o600)
                    return result

                with (
                    patch(
                        "tools.cuda_campaign.eligibility.verify_finalized_projection",
                        return_value=_passed_review(),
                    ),
                    patch(
                        "tools.cuda_campaign.publication._publish_candidate_locked",
                        side_effect=publish_then_swap,
                    ),
                    self.assertRaisesRegex(ValueError, "rolled back"),
                ):
                    self._publish(root, fixture)
                self.assertFalse((root / "published").exists())

    def test_older_passing_decision_cannot_authorize_changed_source_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _private_root(Path(temporary))
            fixture = self._fixture(root)
            with patch(
                "tools.cuda_campaign.eligibility.verify_finalized_projection",
                return_value=_passed_review(),
            ):
                first = self._publish(root, fixture)
            self.assertTrue((root / "sealed-publication-decision").exists())

            sanitizer = fixture["sanitizer"]
            changed = sanitizer.finalized_candidate_output / "recovery-supplement.json"
            changed.write_bytes(canonical_json_bytes({"changed": True}))
            changed.chmod(0o600)
            with (
                patch(
                    "tools.cuda_campaign.eligibility.verify_finalized_projection",
                    return_value=_passed_review(),
                ),
                self.assertRaises(ValueError),
            ):
                self._publish(
                    root,
                    fixture,
                    destination_name="published-second",
                    decision_name="sealed-publication-decision-second",
                    destination_id="destination_" + "e" * 32,
                )
            self.assertFalse((root / "published-second").exists())
            self.assertEqual(
                verify_published_output(
                    root / "published",
                    decision_artifact=root / "sealed-publication-decision",
                )["decision_id"],
                first["decision_id"],
            )


if __name__ == "__main__":
    unittest.main()
