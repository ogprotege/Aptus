import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aptus.execution import (
    JobPrerequisiteError,
    JobService,
    _actual_runtime_snapshot,
    _environment_binding,
    _json_hash,
    _verify_safetensors_structure,
)
from aptus.plan_contract import sha256_file


def write_validation_state(bundle: Path, state: str) -> None:
    (bundle / "validation-report.json").write_text(
        json.dumps({"schema_version": "aptus.validation.v2", "state": state}),
        encoding="utf-8",
    )


def fake_bundle(
    root: Path, *, validation_state: str | None = "model-data-pass"
) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    plan = {
        "plan_id": "plan_" + "a" * 20,
        "model": {"revision": "b" * 40},
        "dataset": {"source_sha256": "c" * 64},
        "hardware": {"reserve_per_device_bytes": 0},
        "recommended": {
            "candidate_id": "cand_test",
            "distribution": "single",
            "world_size": 1,
            "required_host_ram_bytes": 1,
            "checkpoint_retention_bytes": 1,
        },
    }
    (bundle / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (bundle / "validate.py").write_text(
        'print("validation job passed")\n', encoding="utf-8"
    )
    (bundle / "train.py").write_text('print("training job passed")\n', encoding="utf-8")
    (bundle / "requirements.txt").write_text("", encoding="utf-8")
    (bundle / "config").mkdir()
    (bundle / "config" / "accelerate.yaml").write_text(
        "distributed_type: NO\n", encoding="utf-8"
    )
    paths = [item for item in bundle.rglob("*") if item.is_file()]
    manifest = {
        "schema_version": "aptus.bundle.v2",
        "plan_sha256": sha256_file(bundle / "plan.json"),
        "files": [
            {
                "path": path.relative_to(bundle).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(paths)
        ],
    }
    (bundle / "bundle-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if validation_state is not None:
        write_validation_state(bundle, validation_state)
    return bundle


def wait_for(service: JobService, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    record = service.get(job_id)
    while (
        record["state"] in {"queued", "running", "cancelling"}
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
        record = service.get(job_id)
    return record


def make_slow(bundle: Path) -> None:
    (bundle / "validate.py").write_text(
        "import time\ntime.sleep(30)\n", encoding="utf-8"
    )
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    for item in manifest["files"]:
        if item["path"] == "validate.py":
            item["sha256"] = sha256_file(bundle / "validate.py")
            item["size_bytes"] = (bundle / "validate.py").stat().st_size
    (bundle / "bundle-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class ExecutionJobTests(unittest.TestCase):
    def test_parent_runtime_probe_selects_candidate_device_indices(self) -> None:
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                {
                    "hardware": {
                        "cuda_runtime": "test",
                        "driver_version": "test",
                        "devices": [{"index": 0, "uuid": "GPU-selected"}],
                    },
                    "free_cuda_bytes": [1234],
                    "host_ram_free_bytes": 5678,
                }
            ),
            stderr="",
        )
        with patch("aptus.execution.subprocess.run", return_value=completed) as runner:
            snapshot = _actual_runtime_snapshot(1, [2])
        command = runner.call_args.args[0]
        self.assertEqual(json.loads(command[-1]), [2])
        self.assertEqual(snapshot["free_cuda_bytes"], [1234])

    def test_parent_independently_verifies_safetensors_keys_and_indexes(self) -> None:
        class FakeSafeTensorFile:
            def __init__(self, keys: list[str]) -> None:
                self._keys = keys

            def __enter__(self) -> "FakeSafeTensorFile":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def keys(self) -> list[str]:
                return list(self._keys)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_keys: dict[str, list[str] | Exception] = {}
            fake_safetensors = types.ModuleType("safetensors")

            def safe_open(path: str, **_kwargs: object) -> FakeSafeTensorFile:
                value = shard_keys[Path(path).name]
                if isinstance(value, Exception):
                    raise value
                return FakeSafeTensorFile(value)

            fake_safetensors.safe_open = safe_open  # type: ignore[attr-defined]

            def make_export(
                name: str,
                shards: dict[str, list[str] | Exception],
                weight_map: dict[str, str] | None = None,
            ) -> tuple[Path, list[Path]]:
                directory = root / name
                directory.mkdir()
                paths = []
                for shard_name, keys in shards.items():
                    path = directory / shard_name
                    path.write_bytes(b"safetensors-placeholder")
                    shard_keys[shard_name] = keys
                    paths.append(path)
                if weight_map is not None:
                    (directory / "model.safetensors.index.json").write_text(
                        json.dumps({"weight_map": weight_map}), encoding="utf-8"
                    )
                return directory, paths

            with patch.dict(sys.modules, {"safetensors": fake_safetensors}):
                valid, valid_paths = make_export(
                    "valid", {"model.safetensors": ["model.weight"]}
                )
                _verify_safetensors_structure(valid, valid_paths)

                malformed, malformed_paths = make_export(
                    "malformed",
                    {"model.safetensors": RuntimeError("bad header")},
                )
                with self.assertRaisesRegex(ValueError, "parent structural loading"):
                    _verify_safetensors_structure(malformed, malformed_paths)

                empty, empty_paths = make_export("empty", {"model.safetensors": []})
                with self.assertRaisesRegex(ValueError, "no tensor keys"):
                    _verify_safetensors_structure(empty, empty_paths)

                duplicate, duplicate_paths = make_export(
                    "duplicate",
                    {
                        "model-00001-of-00002.safetensors": ["shared.weight"],
                        "model-00002-of-00002.safetensors": ["shared.weight"],
                    },
                    {"shared.weight": "model-00001-of-00002.safetensors"},
                )
                with self.assertRaisesRegex(ValueError, "duplicate tensor key"):
                    _verify_safetensors_structure(duplicate, duplicate_paths)

                misindexed, misindexed_paths = make_export(
                    "misindexed",
                    {
                        "model-00001-of-00002.safetensors": ["weight.a"],
                        "model-00002-of-00002.safetensors": ["weight.b"],
                    },
                    {
                        "weight.a": "model-00002-of-00002.safetensors",
                        "weight.b": "model-00001-of-00002.safetensors",
                    },
                )
                with self.assertRaisesRegex(ValueError, "wrong shards"):
                    _verify_safetensors_structure(misindexed, misindexed_paths)

                (misindexed / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "weight_map": {
                                "weight.a": "model-00001-of-00002.safetensors",
                                "weight.b": "model-00002-of-00002.safetensors",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                _verify_safetensors_structure(misindexed, misindexed_paths)

    def test_direct_action_jumps_are_rejected_with_typed_prerequisites(self) -> None:
        cases = (
            ("dependency", None, "static-pass"),
            ("model-data", "static-pass", "dependency-pass"),
            ("preflight", "dependency-pass", "model-data-pass"),
            ("pilot", "model-data-pass", "measured-preflight-pass"),
            ("train", "measured-preflight-pass", "pilot-pass"),
        )
        for action, current_state, required_state in cases:
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                bundle = fake_bundle(root, validation_state=current_state)
                with self.assertRaises(JobPrerequisiteError) as raised:
                    JobService(root / "jobs").submit(
                        bundle,
                        action=action,  # type: ignore[arg-type]
                        confirm_full_train=action == "train",
                    )
                error = raised.exception
                self.assertEqual(error.code, "job_prerequisite_not_met")
                self.assertEqual(error.action, action)
                self.assertEqual(error.current_state, current_state)
                self.assertEqual(error.required_state, required_state)

    def test_ordered_actions_accept_persisted_state_across_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root, validation_state="static-pass")
            stages = (
                ("dependency", "dependency-pass"),
                ("model-data", "model-data-pass"),
                ("preflight", "measured-preflight-pass"),
                ("pilot", "pilot-pass"),
            )
            for action, promoted_state in stages:
                with self.subTest(action=action):
                    service = JobService(root / "jobs")
                    submitted = service.submit(
                        bundle,
                        action=action,  # type: ignore[arg-type]
                    )
                    finished = wait_for(service, submitted["id"])
                    self.assertEqual(finished["state"], "completed")
                    write_validation_state(bundle, promoted_state)

            service = JobService(root / "jobs")
            with patch.object(
                service,
                "_require_current_pilot",
                return_value={"checked_at": "test"},
            ):
                submitted = service.submit(
                    bundle, action="train", confirm_full_train=True
                )
            finished = wait_for(service, submitted["id"])
        self.assertEqual(submitted["prelaunch_capacity_check"], {"checked_at": "test"})
        self.assertEqual(finished["state"], "failed")
        self.assertIn("completion verification failed", finished["error"])

    def test_unknown_action_cannot_fall_through_to_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            with self.assertRaisesRegex(ValueError, "Unsupported job action"):
                service.submit(fake_bundle(root), action="typo")  # type: ignore[arg-type]

    def test_train_only_fields_are_rejected_for_validation_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            service = JobService(root / "jobs")
            with self.assertRaisesRegex(ValueError, "confirm_full_train"):
                service.submit(bundle, action="preflight", confirm_full_train=True)
            with self.assertRaisesRegex(ValueError, "resume_from"):
                service.submit(bundle, action="pilot", resume_from="checkpoint-1")

    def test_corrupt_job_record_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            jobs.mkdir()
            job_id = "job_" + "f" * 32
            (jobs / f"{job_id}.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unreadable Aptus job record"):
                JobService(jobs)

    def test_job_record_and_log_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            submitted = service.submit(fake_bundle(root), action="preflight")
            finished = wait_for(service, submitted["id"])
            restarted = JobService(root / "jobs")
            reloaded = restarted.get(submitted["id"])
            listed = restarted.list()
        self.assertEqual(finished["state"], "completed")
        self.assertEqual(finished["return_code"], 0)
        self.assertEqual(finished["id"], finished["job_id"])
        self.assertEqual(reloaded["state"], "completed")
        self.assertEqual([item["id"] for item in listed], [submitted["id"]])
        self.assertIn("validation job passed", reloaded["log_tail"])

    def test_managed_child_inherits_the_matching_global_lease_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            token_path = bundle / "observed-lease-token.txt"
            (bundle / "validate.py").write_text(
                "import os\nfrom pathlib import Path\n"
                "Path('observed-lease-token.txt').write_text("
                "os.environ.get('APTUS_GPU_LEASE_TOKEN', ''), encoding='utf-8')\n",
                encoding="utf-8",
            )
            manifest = json.loads(
                (bundle / "bundle-manifest.json").read_text(encoding="utf-8")
            )
            for item in manifest["files"]:
                if item["path"] == "validate.py":
                    item["sha256"] = sha256_file(bundle / "validate.py")
                    item["size_bytes"] = (bundle / "validate.py").stat().st_size
            (bundle / "bundle-manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            finished = wait_for(service, submitted["id"])
            observed_token = token_path.read_text(encoding="utf-8")
        self.assertEqual(finished["state"], "completed")
        self.assertEqual(observed_token, submitted["id"])

    def test_job_refresh_includes_current_bundle_validation_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            finished = wait_for(service, submitted["id"])
            (bundle / "validation-report.json").write_text(
                json.dumps(
                    {
                        "schema_version": "aptus.validation.v2",
                        "state": "measured-preflight-pass",
                    }
                ),
                encoding="utf-8",
            )
            refreshed = service.get(finished["id"])
        self.assertEqual(
            refreshed["validation_report"]["state"],
            "measured-preflight-pass",
        )

    def test_missing_current_report_is_an_explicit_refresh_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            finished = wait_for(service, submitted["id"])
            (bundle / "validation-report.json").unlink()
            refreshed = service.get(finished["id"])
        self.assertIn(
            "validation report is missing", refreshed["validation_report_error"]
        )

    def test_full_training_requires_confirmation_and_current_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            service = JobService(root / "jobs")
            with self.assertRaisesRegex(ValueError, "confirm"):
                service.submit(bundle, action="train")
            with self.assertRaisesRegex(ValueError, "pilot"):
                service.submit(bundle, action="train", confirm_full_train=True)
            plan = json.loads((bundle / "plan.json").read_text())
            (bundle / "pilot-output").mkdir()
            (bundle / "pilot-output" / "metrics.json").write_text(
                json.dumps(
                    {
                        "checkpoint_continuation_observed": True,
                        "phase_one": {"measured_reserved_cuda_bytes": 100},
                        "phase_two_resumed": {"measured_reserved_cuda_bytes": 120},
                    }
                ),
                encoding="utf-8",
            )
            bindings = {
                "bundle": sha256_file(bundle / "bundle-manifest.json"),
                "dataset": plan["dataset"]["source_sha256"],
                "model_revision": plan["model"]["revision"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "environment": _environment_binding(bundle),
                "hardware": _json_hash({"identity": "hardware-test"}),
                "pilot_metrics": sha256_file(bundle / "pilot-output" / "metrics.json"),
            }
            (bundle / "validation-report.json").write_text(
                json.dumps({"state": "pilot-pass", "bindings": bindings}),
                encoding="utf-8",
            )
            with (
                patch(
                    "aptus.execution.validate_plan_payload",
                    return_value=(),
                ),
                patch(
                    "aptus.execution._actual_runtime_snapshot",
                    return_value={
                        "hardware": {"identity": "hardware-test"},
                        "free_cuda_bytes": [10_000],
                        "host_ram_free_bytes": 10_000,
                    },
                ),
                patch(
                    "aptus.execution._verify_pilot_artifacts",
                    return_value=(100, 100),
                ),
            ):
                submitted = service.submit(
                    bundle, action="train", confirm_full_train=True
                )
                finished = wait_for(service, submitted["id"])
        self.assertEqual(finished["state"], "failed")
        self.assertIn("completion verification failed", finished["error"])

    def test_malformed_pilot_metrics_fail_closed_without_service_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            plan = json.loads((bundle / "plan.json").read_text())
            (bundle / "pilot-output").mkdir()
            metrics_path = bundle / "pilot-output" / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "checkpoint_continuation_observed": True,
                        "phase_one": {"measured_reserved_cuda_bytes": "100"},
                        "phase_two_resumed": {"measured_reserved_cuda_bytes": 120},
                    }
                ),
                encoding="utf-8",
            )
            bindings = {
                "bundle": sha256_file(bundle / "bundle-manifest.json"),
                "dataset": plan["dataset"]["source_sha256"],
                "model_revision": plan["model"]["revision"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "environment": _environment_binding(bundle),
                "hardware": _json_hash({"identity": "hardware-test"}),
                "pilot_metrics": sha256_file(metrics_path),
            }
            (bundle / "validation-report.json").write_text(
                json.dumps({"state": "pilot-pass", "bindings": bindings}),
                encoding="utf-8",
            )
            with (
                patch("aptus.execution.validate_plan_payload", return_value=()),
                patch(
                    "aptus.execution._actual_runtime_snapshot",
                    return_value={
                        "hardware": {"identity": "hardware-test"},
                        "free_cuda_bytes": [10_000],
                        "host_ram_free_bytes": 10_000,
                    },
                ),
                patch(
                    "aptus.execution._verify_pilot_artifacts",
                    return_value=(100, 100),
                ),
            ):
                authorization = JobService(root / "jobs").pilot_authorization(bundle)
        self.assertFalse(authorization["current"])
        self.assertIn("non-negative integer", authorization["error"])

    def test_terminal_train_job_defers_deep_authorization_until_submission(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            (bundle / "validation-report.json").write_text(
                json.dumps({"state": "pilot-pass", "bindings": {}}),
                encoding="utf-8",
            )
            service = JobService(root / "jobs")
            job_id = "job_" + "c" * 32
            service._write(
                {
                    "id": job_id,
                    "job_id": job_id,
                    "state": "completed",
                    "action": "train",
                    "bundle_dir": str(bundle),
                    "created_at": "2026-07-21T00:00:00+00:00",
                    "prelaunch_capacity_check": {"checked_at": "stale"},
                }
            )
            with patch.object(
                service,
                "pilot_authorization",
                return_value={
                    "current": False,
                    "error": "current environment drifted",
                    "capacity": None,
                },
            ) as authorization:
                refreshed = service.get(job_id)
        authorization.assert_not_called()
        self.assertFalse(refreshed["validation_report"]["authorization_current"])
        self.assertIn(
            "performed atomically when full training is submitted",
            refreshed["validation_report"]["authorization_error"],
        )

    def test_manifested_change_invalidates_pilot_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            plan = json.loads((bundle / "plan.json").read_text())
            (bundle / "pilot-output").mkdir()
            (bundle / "pilot-output" / "metrics.json").write_text(
                '{"resume_verified":true}\n', encoding="utf-8"
            )
            bindings = {
                "bundle": sha256_file(bundle / "bundle-manifest.json"),
                "dataset": plan["dataset"]["source_sha256"],
                "model_revision": plan["model"]["revision"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "environment": _environment_binding(bundle),
                "hardware": "hardware-test-binding",
                "pilot_metrics": sha256_file(bundle / "pilot-output" / "metrics.json"),
            }
            (bundle / "validation-report.json").write_text(
                json.dumps({"state": "pilot-pass", "bindings": bindings}),
                encoding="utf-8",
            )
            (bundle / "train.py").write_text('print("changed")\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                JobService(root / "jobs").submit(
                    bundle, action="train", confirm_full_train=True
                )

    def test_unmanifested_import_shadow_invalidates_pilot_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            (bundle / "torch.py").write_text(
                "raise RuntimeError('shadowed')\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unmanifested"):
                JobService(root / "jobs").submit(
                    bundle, action="train", confirm_full_train=True
                )

    def test_cancel_terminates_process_group_and_persists_cancelled_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            make_slow(bundle)
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            deadline = time.monotonic() + 2
            current = service.get(submitted["id"])
            while current["state"] == "queued" and time.monotonic() < deadline:
                time.sleep(0.01)
                current = service.get(submitted["id"])
            cancelled = service.cancel(submitted["id"])
        self.assertEqual(cancelled["state"], "cancelled")

    def test_same_bundle_cannot_launch_overlapping_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            make_slow(bundle)
            service = JobService(root / "jobs")
            first = service.submit(bundle, action="preflight")
            with self.assertRaisesRegex(ValueError, "active job"):
                service.submit(bundle, action="preflight")
            service.cancel(first["id"])

    def test_foreign_service_preserves_live_owner_and_refuses_cancellation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            make_slow(bundle)
            owner = JobService(root / "jobs")
            submitted = owner.submit(bundle, action="preflight")
            foreign = JobService(root / "jobs")
            observed = foreign.get(submitted["id"])
            self.assertIn(observed["state"], {"queued", "running"})
            self.assertFalse(observed["cancellable"])
            self.assertEqual(observed["owner_status"], "external-service")
            with self.assertRaisesRegex(ValueError, "does not own"):
                foreign.cancel(submitted["id"])
            self.assertIn(owner.get(submitted["id"])["state"], {"queued", "running"})
            owner.cancel(submitted["id"])

    def test_global_job_guard_blocks_a_different_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root, second_root = root / "first", root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = fake_bundle(first_root)
            second = fake_bundle(second_root)
            make_slow(first)
            service = JobService(root / "jobs")
            submitted = service.submit(first, action="preflight")
            with self.assertRaisesRegex(ValueError, "one local GPU job"):
                service.submit(second, action="preflight")
            service.cancel(submitted["id"])

    def test_host_global_guard_spans_different_state_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root, second_root = root / "first", root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = fake_bundle(first_root)
            second = fake_bundle(second_root)
            make_slow(first)
            owner = JobService(root / "jobs-a")
            competitor = JobService(root / "jobs-b")
            submitted = owner.submit(first, action="preflight")
            with self.assertRaisesRegex(ValueError, "across all state roots"):
                competitor.submit(second, action="preflight")
            owner.cancel(submitted["id"])

    def test_worker_start_failure_releases_global_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            service = JobService(root / "jobs")
            with (
                patch.object(service, "_process_identity", return_value="owner"),
                patch(
                    "aptus.execution.threading.Thread.start",
                    side_effect=RuntimeError("injected thread failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected thread failure"),
            ):
                service.submit(bundle, action="preflight")
            replacement = JobService(root / "replacement-jobs")
            submitted = replacement.submit(bundle, action="preflight")
            finished = wait_for(replacement, submitted["id"])
        self.assertEqual(finished["state"], "completed")

    def test_orphan_child_and_cancelling_record_remain_globally_active(self) -> None:
        for state in ("running", "cancelling"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                jobs = root / "jobs"
                jobs.mkdir()
                bundle = fake_bundle(root)
                job_id = "job_" + ("a" if state == "running" else "b") * 32
                record = {
                    "id": job_id,
                    "job_id": job_id,
                    "state": state,
                    "bundle_dir": str(bundle),
                    "created_at": "2026-07-21T00:00:00+00:00",
                    "owner_pid": 999_999_999,
                    "process_pid": os.getpid(),
                    "process_identity": JobService._process_identity(os.getpid()),
                }
                (jobs / f"{job_id}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
                service = JobService(jobs)
                observed = service.get(job_id)
                self.assertEqual(observed["state"], state)
                self.assertEqual(observed["owner_status"], "orphan-child")
                self.assertFalse(observed["cancellable"])
                with self.assertRaisesRegex(ValueError, "active job"):
                    service.submit(bundle, action="preflight")

    def test_cancel_fails_closed_when_exited_process_has_no_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            job_id = "job_" + "e" * 32
            process = subprocess.Popen(
                [sys.executable, "-c", "pass"],
                text=True,
            )
            self.assertEqual(process.wait(timeout=5), 0)
            service._write(
                {
                    "id": job_id,
                    "job_id": job_id,
                    "state": "running",
                    "bundle_dir": str(root),
                    "return_code": None,
                    "created_at": "2026-07-21T00:00:00+00:00",
                }
            )
            service._processes[job_id] = process
            result = service.cancel(job_id)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["return_code"], 0)
        self.assertIn("verifier is unavailable", result["error"])

    def test_restart_marks_unattachable_running_job_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "jobs"
            root.mkdir()
            job_id = "job_" + "d" * 32
            (root / f"{job_id}.json").write_text(
                json.dumps({"id": job_id, "job_id": job_id, "state": "running"}),
                encoding="utf-8",
            )
            record = JobService(root).get(job_id)
        self.assertEqual(record["state"], "failed")
        self.assertIn("no longer live", record["error"])

    def test_unexpected_worker_exception_persists_terminal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            with (
                patch.object(service, "_process_identity", return_value="test-process"),
                patch(
                    "aptus.execution.subprocess.Popen",
                    side_effect=RuntimeError("injected launcher failure"),
                ),
            ):
                submitted = service.submit(fake_bundle(root), action="preflight")
                finished = wait_for(service, submitted["id"])
        self.assertEqual(finished["state"], "failed")
        self.assertIn("injected launcher failure", finished["error"])


if __name__ == "__main__":
    unittest.main()
