import hashlib
import json
import multiprocessing
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from aptus.local_store import atomic_write_json, quarantine_file
from aptus.projects import (
    CURRENT_PROJECT_SCHEMA_VERSION,
    PROJECT_REVISION_SCHEMA_VERSION,
    PROJECT_SCHEMA_VERSION,
    ProjectRepository,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision_worker(
    state_root: str,
    project_id: str,
    reason: str,
    barrier: Any,
    connection: Any,
) -> None:
    try:
        repository = ProjectRepository(Path(state_root))
        barrier.wait(timeout=15)
        revision = repository.create_revision(project_id, reason=reason)
        connection.send(
            {
                "pid": os.getpid(),
                "revision_id": revision["revision_id"],
                "ordinal": revision["ordinal"],
            }
        )
    except BaseException as error:
        connection.send({"error": repr(error)})
        raise
    finally:
        connection.close()


class ProjectRepositoryTests(unittest.TestCase):
    def _saved_plan(self, state: Path, plan_id: str) -> dict[str, Any]:
        plan = {
            "schema_version": "aptus.training-plan.v3",
            "plan_id": plan_id,
            "recommended": {"candidate_id": "candidate_a"},
        }
        plans = state / "plans"
        plans.mkdir(parents=True, exist_ok=True)
        atomic_write_json(plans / f"{plan_id}.json", plan, mode=0o600)
        return plan

    def _bundle(self, root: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
        root.mkdir(parents=True)
        plan_path = root / "plan.json"
        payload_path = root / "payload.txt"
        atomic_write_json(plan_path, plan, mode=0o600)
        payload_path.write_text("immutable bundle input\n", encoding="utf-8")
        entries = [
            {
                "path": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (payload_path, plan_path)
        ]
        manifest = {
            "schema_version": "aptus.bundle.v2",
            "plan_id": plan["plan_id"],
            "plan_sha256": _sha256(plan_path),
            "candidate_id": plan["recommended"]["candidate_id"],
            "files": entries,
        }
        manifest_path = root / "bundle-manifest.json"
        atomic_write_json(manifest_path, manifest, mode=0o600)
        manifest_digest = _sha256(manifest_path)
        return (
            {
                "bundle_dir": str(root.resolve()),
                "files": sorted(
                    ["bundle-manifest.json", *[item["path"] for item in entries]]
                ),
                "artifact_fingerprint": manifest_digest,
            },
            manifest_digest,
        )

    def _orphan_revision(
        self, repository: ProjectRepository, project_id: str, reason: str
    ) -> str:
        revisions_root = repository.root / project_id / "revisions"
        before = set(revisions_root.glob("revision_*.json"))
        write_count = 0

        def interrupt_after_revision(
            path: Path,
            value: dict[str, Any],
            *,
            mode: int | None = None,
        ) -> None:
            nonlocal write_count
            atomic_write_json(path, value, mode=mode)
            write_count += 1
            if write_count == 2:
                raise RuntimeError("simulated process interruption")

        with patch(
            "aptus.projects.atomic_write_json", side_effect=interrupt_after_revision
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
                repository.create_revision(project_id, reason=reason)
        after = set(revisions_root.glob("revision_*.json"))
        return (after - before).pop().stem

    def test_named_project_revisions_are_immutable_and_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            repository = ProjectRepository(state)
            project = repository.create("Parish corpus adapter")
            first = repository.create_revision(
                project["project_id"],
                reason="plan-created",
                facts={"dataset_path": "/tmp/training.jsonl"},
                plan_id="plan_" + "a" * 20,
                selected_candidate_id="candidate_a",
            )
            first_path = (
                repository.root
                / project["project_id"]
                / "revisions"
                / f"{first['revision_id']}.json"
            )
            first_bytes = first_path.read_bytes()
            second = repository.create_revision(
                project["project_id"],
                reason="bundle-compiled",
                bundle={
                    "bundle_dir": "/tmp/bundle",
                    "artifact_fingerprint": "a" * 64,
                },
            )
            restarted = ProjectRepository(state)
            restored = restarted.get(project["project_id"])
            history = restarted.history(project["project_id"])
            first_unchanged = first_path.read_bytes() == first_bytes
            first_mode = stat.S_IMODE(first_path.stat().st_mode)

        self.assertTrue(first_unchanged)
        self.assertEqual(first["schema_version"], PROJECT_REVISION_SCHEMA_VERSION)
        self.assertEqual(second["parent_revision_id"], first["revision_id"])
        self.assertEqual(restored["schema_version"], PROJECT_SCHEMA_VERSION)
        self.assertEqual(restored["latest_revision_id"], second["revision_id"])
        self.assertEqual([item["ordinal"] for item in history], [2, 1])
        self.assertEqual(first_mode, 0o600)

    def test_recovery_creates_a_new_revision_and_never_restores_authorization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            plan_id = "plan_" + "b" * 20
            plan = self._saved_plan(state, plan_id)
            repository = ProjectRepository(state)
            project = repository.create("Recovery test")
            source = repository.create_revision(
                project["project_id"],
                reason="plan-created",
                plan_id=plan_id,
                plan_snapshot=plan,
                selected_candidate_id="candidate_a",
                validation={"authorization_current": True},
            )
            recovered = repository.recover(project["project_id"], source["revision_id"])

        self.assertNotEqual(recovered["revision_id"], source["revision_id"])
        self.assertEqual(recovered["parent_revision_id"], source["revision_id"])
        self.assertFalse(recovered["training_authorization"]["current"])
        self.assertFalse(recovered["validation"]["training_authorization_current"])
        self.assertNotIn("authorization_current", recovered["validation"])

    def test_corrupt_manifest_is_quarantined_without_hiding_other_projects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = ProjectRepository(Path(temporary) / "state")
            healthy = repository.create("Healthy")
            corrupt = repository.create("Corrupt")
            corrupt_path = repository.root / corrupt["project_id"] / "project.json"
            corrupt_path.write_text("{", encoding="utf-8")
            projects = repository.list()
            quarantined = list(repository.quarantine_root.glob("*project.json"))

        self.assertEqual(
            [item["project_id"] for item in projects], [healthy["project_id"]]
        )
        self.assertEqual(len(quarantined), 1)

    def test_corrupt_latest_revision_is_quarantined_and_previous_safe_state_loads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = ProjectRepository(Path(temporary) / "state")
            project = repository.create("Safe fallback")
            safe = repository.create_revision(
                project["project_id"], reason="plan-created"
            )
            corrupt = repository.create_revision(
                project["project_id"], reason="bundle-compiled"
            )
            corrupt_path = (
                repository.root
                / project["project_id"]
                / "revisions"
                / f"{corrupt['revision_id']}.json"
            )
            corrupt_path.write_text("{}", encoding="utf-8")
            restored = repository.get(project["project_id"])
            current = repository.current()
            quarantined = list(
                repository.quarantine_root.glob(f"*{corrupt['revision_id']}.json")
            )

        self.assertEqual(restored["latest_revision_id"], safe["revision_id"])
        self.assertEqual(restored["revision_count"], 1)
        self.assertEqual(current["latest_revision_id"], safe["revision_id"])
        self.assertIn(
            corrupt["revision_id"], restored["recovered_corrupt_revision_ids"]
        )
        self.assertEqual(len(quarantined), 1)

    def test_two_processes_crossing_a_barrier_retain_both_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            repository = ProjectRepository(state)
            project = repository.create("Concurrent writers")
            context = multiprocessing.get_context("spawn")
            barrier = context.Barrier(2)
            receivers = []
            senders = []
            processes = []
            for reason in ("writer-one", "writer-two"):
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_revision_worker,
                    args=(
                        str(state),
                        project["project_id"],
                        reason,
                        barrier,
                        sender,
                    ),
                )
                receivers.append(receiver)
                senders.append(sender)
                processes.append(process)
            for process in processes:
                process.start()
            for sender in senders:
                sender.close()
            for process in processes:
                process.join(20)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                    self.fail("A concurrent project writer did not finish.")
            results = [receiver.recv() for receiver in receivers]
            for receiver in receivers:
                receiver.close()
            restored = ProjectRepository(state).get(project["project_id"])
            history = ProjectRepository(state).history(project["project_id"])

        self.assertTrue(all(process.exitcode == 0 for process in processes))
        self.assertTrue(all("error" not in result for result in results), results)
        self.assertEqual(len({result["pid"] for result in results}), 2)
        self.assertEqual(len({result["revision_id"] for result in results}), 2)
        self.assertEqual(sorted(result["ordinal"] for result in results), [1, 2])
        self.assertEqual(restored["revision_count"], 2)
        self.assertEqual(sorted(item["ordinal"] for item in history), [1, 2])

    def test_interruption_after_each_transaction_write_recovers_deterministically(
        self,
    ) -> None:
        original_write = atomic_write_json
        for interrupted_write in range(1, 5):
            with self.subTest(interrupted_write=interrupted_write):
                with tempfile.TemporaryDirectory() as temporary:
                    state = Path(temporary) / "state"
                    repository = ProjectRepository(state)
                    project = repository.create("Crash recovery")
                    base = repository.create_revision(
                        project["project_id"], reason="base"
                    )
                    write_count = 0

                    def interrupting_write(
                        path: Path,
                        value: dict[str, Any],
                        *,
                        mode: int | None = None,
                    ) -> None:
                        nonlocal write_count
                        original_write(path, value, mode=mode)
                        write_count += 1
                        if write_count == interrupted_write:
                            raise RuntimeError("simulated process interruption")

                    with patch(
                        "aptus.projects.atomic_write_json",
                        side_effect=interrupting_write,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "simulated process interruption"
                        ):
                            repository.create_revision(
                                project["project_id"], reason="interrupted"
                            )

                    restarted = ProjectRepository(state)
                    restored = restarted.get(project["project_id"])
                    current = restarted.current()
                    transaction_path = (
                        restarted.root
                        / project["project_id"]
                        / "revision-transaction.json"
                    )
                    transaction_exists = transaction_path.exists()

                expected_count = 1 if interrupted_write == 1 else 2
                self.assertEqual(restored["revision_count"], expected_count)
                self.assertEqual(
                    current["latest_revision_id"], restored["latest_revision_id"]
                )
                self.assertFalse(transaction_exists)
                if expected_count == 1:
                    self.assertEqual(
                        restored["latest_revision_id"], base["revision_id"]
                    )
                else:
                    self.assertEqual(
                        restored["latest_revision"]["reason"], "interrupted"
                    )

    def test_valid_orphan_revision_is_adopted_without_a_transaction_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            repository = ProjectRepository(state)
            project = repository.create("Orphan recovery")
            repository.create_revision(project["project_id"], reason="base")
            before = set(
                (repository.root / project["project_id"] / "revisions").glob("*.json")
            )
            write_count = 0

            def interrupt_after_revision(
                path: Path,
                value: dict[str, Any],
                *,
                mode: int | None = None,
            ) -> None:
                nonlocal write_count
                atomic_write_json(path, value, mode=mode)
                write_count += 1
                if write_count == 2:
                    raise RuntimeError("simulated process interruption")

            with patch(
                "aptus.projects.atomic_write_json",
                side_effect=interrupt_after_revision,
            ):
                with self.assertRaises(RuntimeError):
                    repository.create_revision(project["project_id"], reason="orphaned")
            transaction_path = (
                repository.root / project["project_id"] / "revision-transaction.json"
            )
            transaction_path.unlink()
            after = set(
                (repository.root / project["project_id"] / "revisions").glob("*.json")
            )
            orphan_id = (after - before).pop().stem
            restored = ProjectRepository(state).get(project["project_id"])

        self.assertEqual(restored["revision_count"], 2)
        self.assertEqual(restored["latest_revision_id"], orphan_id)
        self.assertIn(orphan_id, restored["recovered_orphan_revision_ids"])

    def test_ambiguous_orphan_rejection_survives_interruption_mid_quarantine(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            repository = ProjectRepository(state)
            project = repository.create("Ambiguous orphan recovery")
            base = repository.create_revision(project["project_id"], reason="base")
            first_orphan = self._orphan_revision(
                repository, project["project_id"], "competitor-one"
            )
            repository._revision_transaction_path(project["project_id"]).unlink()
            first_path = repository._revision_path(project["project_id"], first_orphan)
            competitor = json.loads(first_path.read_text(encoding="utf-8"))
            second_orphan = "revision_" + "e" * 32
            competitor["revision_id"] = second_orphan
            competitor["reason"] = "competitor-two"
            competitor["content_sha256"] = hashlib.sha256(
                json.dumps(
                    {
                        key: value
                        for key, value in competitor.items()
                        if key != "content_sha256"
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            atomic_write_json(
                repository._revision_path(project["project_id"], second_orphan),
                competitor,
                mode=0o600,
            )
            quarantine_count = 0

            def interrupt_after_first_quarantine(
                path: Path, root: Path, *, reason: str
            ) -> Path:
                nonlocal quarantine_count
                destination = quarantine_file(path, root, reason=reason)
                quarantine_count += 1
                if quarantine_count == 1:
                    raise RuntimeError("simulated quarantine interruption")
                return destination

            with patch(
                "aptus.projects.quarantine_file",
                side_effect=interrupt_after_first_quarantine,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "simulated quarantine interruption"
                ):
                    ProjectRepository(state).get(project["project_id"])

            restored_repository = ProjectRepository(state)
            restored = restored_repository.get(project["project_id"])
            remaining = list(
                (restored_repository.root / project["project_id"] / "revisions").glob(
                    "revision_*.json"
                )
            )

        self.assertEqual(restored["latest_revision_id"], base["revision_id"])
        self.assertEqual(restored["revision_count"], 1)
        self.assertEqual(
            set(restored["rejected_orphan_revision_ids"]),
            {first_orphan, second_orphan},
        )
        self.assertEqual([path.stem for path in remaining], [base["revision_id"]])

    def test_mismatched_transaction_revision_cannot_reappear_as_an_orphan(self) -> None:
        for field in ("content_sha256", "base_revision_id", "ordinal"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                state = Path(temporary) / "state"
                repository = ProjectRepository(state)
                project = repository.create("Rejected transaction")
                base = repository.create_revision(project["project_id"], reason="base")
                rejected_id = self._orphan_revision(
                    repository, project["project_id"], "rejected"
                )
                receipt_path = repository._revision_transaction_path(
                    project["project_id"]
                )
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if field == "ordinal":
                    receipt[field] += 1
                elif field == "base_revision_id":
                    receipt[field] = "revision_" + "f" * 32
                else:
                    receipt[field] = "0" * 64
                atomic_write_json(receipt_path, receipt, mode=0o600)

                restored = ProjectRepository(state).get(project["project_id"])

                self.assertEqual(restored["latest_revision_id"], base["revision_id"])
                self.assertEqual(restored["revision_count"], 1)
                self.assertIn(rejected_id, restored["rejected_orphan_revision_ids"])

    def test_unreadable_transaction_rejects_its_unindexed_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            repository = ProjectRepository(state)
            project = repository.create("Unreadable transaction")
            base = repository.create_revision(project["project_id"], reason="base")
            rejected_id = self._orphan_revision(
                repository, project["project_id"], "unreadable-receipt"
            )
            repository._revision_transaction_path(project["project_id"]).write_text(
                "{", encoding="utf-8"
            )

            restored = ProjectRepository(state).get(project["project_id"])

        self.assertEqual(restored["latest_revision_id"], base["revision_id"])
        self.assertEqual(restored["revision_count"], 1)
        self.assertIn(rejected_id, restored["rejected_orphan_revision_ids"])

    def test_stale_writer_is_rejected_inside_the_transaction_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            repository = ProjectRepository(state)
            project = repository.create("Optimistic concurrency")
            first = repository.create_revision(project["project_id"], reason="first")
            stale_writer = ProjectRepository(state)
            fresh_writer = ProjectRepository(state)
            second = fresh_writer.create_revision(
                project["project_id"],
                reason="second",
                base_revision_id=first["revision_id"],
                expected_latest_revision_id=first["revision_id"],
            )
            with self.assertRaisesRegex(ValueError, "changed after it was loaded"):
                stale_writer.create_revision(
                    project["project_id"],
                    reason="stale-third",
                    base_revision_id=first["revision_id"],
                    expected_latest_revision_id=first["revision_id"],
                )
            restored = repository.get(project["project_id"])
            revision_files = list(
                (repository.root / project["project_id"] / "revisions").glob(
                    "revision_*.json"
                )
            )

        self.assertEqual(restored["revision_count"], 2)
        self.assertEqual(restored["latest_revision_id"], second["revision_id"])
        self.assertEqual(len(revision_files), 2)

    def test_stale_current_pointer_is_repaired_to_the_manifest_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            repository = ProjectRepository(state)
            project = repository.create("Pointer repair")
            first = repository.create_revision(project["project_id"], reason="first")
            second = repository.create_revision(project["project_id"], reason="second")
            atomic_write_json(
                repository.current_path,
                {
                    "schema_version": CURRENT_PROJECT_SCHEMA_VERSION,
                    "project_id": project["project_id"],
                    "revision_id": first["revision_id"],
                    "selected_at": "stale",
                },
                mode=0o600,
            )
            ProjectRepository(state).get(project["project_id"])
            pointer = json.loads(repository.current_path.read_text(encoding="utf-8"))

        self.assertEqual(pointer["revision_id"], second["revision_id"])

    def test_older_recovery_does_not_replace_a_newer_project_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            repository = ProjectRepository(state)
            older = repository.create("Older interrupted project")
            repository.create_revision(older["project_id"], reason="base")
            self._orphan_revision(repository, older["project_id"], "interrupted")
            newer = repository.create("Newer selected project")
            newer_revision = repository.create_revision(
                newer["project_id"], reason="committed"
            )

            ProjectRepository(state).list()
            current = ProjectRepository(state).current()

        self.assertEqual(current["project_id"], newer["project_id"])
        self.assertEqual(current["latest_revision_id"], newer_revision["revision_id"])

    def test_newer_transaction_intent_can_replace_an_older_project_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            repository = ProjectRepository(state)
            recovering = repository.create("Newer interrupted project")
            older = repository.create("Older selected project")
            repository.create_revision(older["project_id"], reason="committed")
            interrupted_id = self._orphan_revision(
                repository, recovering["project_id"], "interrupted"
            )

            ProjectRepository(state).get(recovering["project_id"])
            current = ProjectRepository(state).current()

        self.assertEqual(current["project_id"], recovering["project_id"])
        self.assertEqual(current["latest_revision_id"], interrupted_id)

    def test_recovery_rejects_tampered_saved_plan_and_bundle_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "plan-state"
            plan_id = "plan_" + "d" * 20
            plan = self._saved_plan(state, plan_id)
            repository = ProjectRepository(state)
            project = repository.create("Tampered plan")
            source = repository.create_revision(
                project["project_id"],
                reason="plan-created",
                plan_id=plan_id,
                plan_snapshot=plan,
                selected_candidate_id="candidate_a",
                bundle={},
            )
            atomic_write_json(
                state / "plans" / f"{plan_id}.json",
                {**plan, "tampered": True},
                mode=0o600,
            )
            with self.assertRaisesRegex(ValueError, "immutable snapshot"):
                repository.recover(project["project_id"], source["revision_id"])
            plan_project = repository.get(project["project_id"])
            plan_source_id = source["revision_id"]

        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "bundle-state"
            plan_id = "plan_" + "e" * 20
            plan = self._saved_plan(state, plan_id)
            bundle, manifest_digest = self._bundle(Path(temporary) / "bundle", plan)
            repository = ProjectRepository(state)
            project = repository.create("Tampered manifest")
            source = repository.create_revision(
                project["project_id"],
                reason="bundle-compiled",
                plan_id=plan_id,
                plan_snapshot=plan,
                selected_candidate_id="candidate_a",
                bundle=bundle,
                validation={"report": {"artifact_fingerprint": manifest_digest}},
            )
            manifest_path = Path(bundle["bundle_dir"]) / "bundle-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["plan_sha256"] = "0" * 64
            atomic_write_json(manifest_path, manifest, mode=0o600)
            with self.assertRaisesRegex(ValueError, "manifest or file digests"):
                repository.recover(project["project_id"], source["revision_id"])
            bundle_project = repository.get(project["project_id"])

        self.assertEqual(plan_project["revision_count"], 1)
        self.assertEqual(plan_project["latest_revision_id"], plan_source_id)
        self.assertEqual(bundle_project["revision_count"], 1)

    def test_recovery_rejects_self_consistent_bundle_replacement_and_archive(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            plan_id = "plan_" + "b" * 20
            plan = self._saved_plan(state, plan_id)
            bundle, _ = self._bundle(root / "bundle", plan)
            archive = root / "bundle.zip"
            archive.write_bytes(b"original archive")
            bundle.update(
                archive_path=str(archive),
                archive_sha256=_sha256(archive),
                archive_size_bytes=archive.stat().st_size,
            )
            repository = ProjectRepository(state)
            project = repository.create("Pinned bundle")
            source = repository.create_revision(
                project["project_id"],
                reason="bundle-compiled",
                plan_id=plan_id,
                plan_snapshot=plan,
                selected_candidate_id="candidate_a",
                bundle=bundle,
            )

            payload_path = Path(bundle["bundle_dir"]) / "payload.txt"
            payload_path.write_text("replacement payload\n", encoding="utf-8")
            manifest_path = Path(bundle["bundle_dir"]) / "bundle-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload_entry = next(
                item for item in manifest["files"] if item["path"] == "payload.txt"
            )
            payload_entry["sha256"] = _sha256(payload_path)
            payload_entry["size_bytes"] = payload_path.stat().st_size
            atomic_write_json(manifest_path, manifest, mode=0o600)

            with self.assertRaisesRegex(ValueError, "artifact fingerprint"):
                repository.recover(project["project_id"], source["revision_id"])

            archive.write_bytes(b"replacement archive")
            atomic_write_json(
                manifest_path,
                {
                    **manifest,
                },
                mode=0o600,
            )
            bundle["artifact_fingerprint"] = _sha256(manifest_path)
            archive_project = repository.create("Pinned archive")
            archive_source = repository.create_revision(
                archive_project["project_id"],
                reason="bundle-compiled",
                plan_id=plan_id,
                plan_snapshot=plan,
                selected_candidate_id="candidate_a",
                bundle={
                    **bundle,
                    "archive_sha256": "0" * 64,
                    "archive_size_bytes": archive.stat().st_size,
                },
            )
            with self.assertRaisesRegex(ValueError, "archive"):
                repository.recover(
                    archive_project["project_id"], archive_source["revision_id"]
                )

    def test_recovering_plan_only_revision_does_not_infer_newer_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            plan_id = "plan_" + "f" * 20
            plan = self._saved_plan(state, plan_id)
            bundle, _ = self._bundle(Path(temporary) / "bundle", plan)
            repository = ProjectRepository(state)
            project = repository.create("Plan-only recovery")
            plan_revision = repository.create_revision(
                project["project_id"],
                reason="plan-created",
                plan_id=plan_id,
                plan_snapshot=plan,
                selected_candidate_id="candidate_a",
                bundle={},
            )
            repository.create_revision(
                project["project_id"],
                reason="bundle-compiled",
                bundle=bundle,
            )
            recovered = repository.recover(
                project["project_id"], plan_revision["revision_id"]
            )

        self.assertEqual(recovered["bundle"], {})
        self.assertEqual(recovered["parent_revision_id"], plan_revision["revision_id"])

    def test_legacy_workspace_import_preserves_sources_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            plans = state / "plans"
            plans.mkdir(parents=True)
            plan_id = "plan_" + "c" * 20
            plan = {
                "plan_id": plan_id,
                "model": {"model_id": "example/model"},
                "dataset": {"source_path": "data.jsonl"},
                "hardware": {},
                "target": {},
                "recommended": {"candidate_id": "candidate_c"},
            }
            plan_path = plans / f"{plan_id}.json"
            atomic_write_json(plan_path, plan)
            bundle, bundle_fingerprint = self._bundle(
                Path(temporary) / "legacy-bundle", plan
            )
            current_bundle_path = state / "current-bundle.json"
            atomic_write_json(
                current_bundle_path,
                {"bundle_dir": bundle["bundle_dir"], "archive_path": None},
                mode=0o600,
            )
            repository = ProjectRepository(state)
            imported = repository.import_legacy(
                plans_dir=plans,
                current_bundle_path=current_bundle_path,
                jobs=[],
            )
            repeated = repository.import_legacy(
                plans_dir=plans,
                current_bundle_path=current_bundle_path,
                jobs=[],
            )
            source_preserved = plan_path.is_file()
            receipt_written = repository.import_receipt_path.is_file()

            payload_path = Path(bundle["bundle_dir"]) / "payload.txt"
            payload_path.write_text("legacy replacement\n", encoding="utf-8")
            manifest_path = Path(bundle["bundle_dir"]) / "bundle-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload_entry = next(
                item for item in manifest["files"] if item["path"] == "payload.txt"
            )
            payload_entry["sha256"] = _sha256(payload_path)
            payload_entry["size_bytes"] = payload_path.stat().st_size
            atomic_write_json(manifest_path, manifest, mode=0o600)
            with self.assertRaisesRegex(ValueError, "Replan required"):
                repository.recover(imported["project_id"], imported["revision_id"])
            revision_count = repository.get(imported["project_id"])["revision_count"]

        self.assertIsNotNone(imported)
        self.assertIsNone(repeated)
        self.assertTrue(source_preserved)
        self.assertEqual(imported["plan_snapshot"], plan)
        self.assertEqual(imported["bundle"]["artifact_fingerprint"], bundle_fingerprint)
        self.assertTrue(receipt_written)
        self.assertEqual(revision_count, 1)


if __name__ == "__main__":
    unittest.main()
