import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from aptus.execution import JobDispositionError, JobService


def _write_completed_train(
    root: Path,
    *,
    job_id: str,
    action: str = "train",
    state: str = "completed",
    plan_id: str = "plan_abc",
    candidate_id: str = "cand_abc",
    run_id: str | None = "run_abc",
    validation_state: str | None = "measured-run-pass",
    with_metrics: bool = False,
    evaluation_decision: str | None = None,
) -> Path:
    """Write a completed-train job JSON the same way execution tests do."""

    jobs = root / "jobs"
    jobs.mkdir(exist_ok=True)
    bundle = root / "bundle"
    bundle.mkdir(exist_ok=True)
    if validation_state is not None:
        (bundle / "validation-report.json").write_text(
            json.dumps(
                {"schema_version": "aptus.validation.v2", "state": validation_state}
            ),
            encoding="utf-8",
        )
    run_dir = None
    if run_id is not None:
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if with_metrics:
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "train_loss_observations": [1.0, 0.4],
                        "validation_loss_observations": [0.9, 1.1],
                    }
                ),
                encoding="utf-8",
            )
    record = {
        "schema_version": "aptus.job-record.v1",
        "id": job_id,
        "job_id": job_id,
        "state": state,
        "created_at": "2026-01-01T00:00:00+00:00",
        "action": action,
        "bundle_dir": str(bundle),
        "run_output_dir": str(run_dir) if run_dir is not None else None,
        "plan_id": plan_id,
        "candidate_id": candidate_id,
        "run_id": run_id,
    }
    if evaluation_decision is not None:
        record["evaluation_decision"] = evaluation_decision
    (jobs / f"{job_id}.json").write_text(json.dumps(record), encoding="utf-8")
    return jobs


class JobDispositionTests(unittest.TestCase):
    def test_save_disposition_refuses_non_train_and_non_completed(self) -> None:
        cases = (
            ("pilot", "completed"),
            ("preflight", "completed"),
            ("train", "running"),
            ("train", "failed"),
        )
        for action, state in cases:
            with self.subTest(action=action, state=state):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    job_id = "job_" + "a" * 32
                    jobs = _write_completed_train(
                        root,
                        job_id=job_id,
                        action=action,
                        state=state,
                    )
                    service = JobService(jobs)
                    with self.assertRaises(JobDispositionError) as raised:
                        service.save_disposition(job_id, "use")
                    self.assertEqual(raised.exception.code, "job_disposition_refused")
                    self.assertFalse((jobs / f"{job_id}.disposition.json").exists())

    def test_save_disposition_writes_sibling_and_get_attaches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job_" + "b" * 32
            plan_id = "plan_keep_me"
            jobs = _write_completed_train(
                root,
                job_id=job_id,
                plan_id=plan_id,
                with_metrics=True,
            )
            service = JobService(jobs)
            got = service.save_disposition(job_id, "use")
            sibling = jobs / f"{job_id}.disposition.json"
            persisted = json.loads(
                (jobs / f"{job_id}.json").read_text(encoding="utf-8")
            )
            payload = json.loads(sibling.read_text(encoding="utf-8"))
            self.assertTrue(sibling.is_file())
            self.assertFalse(sibling.is_symlink())
            self.assertEqual(stat.S_IMODE(sibling.stat().st_mode), 0o600)
            self.assertEqual(payload["schema_version"], "aptus.run-disposition.v1")
            self.assertEqual(payload["kind"], "use")
            self.assertEqual(payload["source"], "operator-attested")
            self.assertEqual(payload["job_id"], job_id)
            self.assertEqual(payload["plan_id"], plan_id)
            self.assertIsNone(payload["previous_kind"])
            self.assertEqual(
                payload["evidence"]["validation_state"], "measured-run-pass"
            )
            self.assertEqual(payload["evidence"]["run_correction_kind"], "eval-rose")
            self.assertEqual(payload["evidence"]["evaluation_decision"], "omitted")
            self.assertEqual(got["run_disposition"]["kind"], "use")
            self.assertEqual(
                got["run_disposition"]["operator_next_step"]["action"],
                "load-adapter",
            )
            self.assertEqual(persisted["plan_id"], plan_id)
            self.assertNotIn("run_disposition", persisted)

    def test_second_attest_sets_previous_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job_" + "c" * 32
            jobs = _write_completed_train(root, job_id=job_id)
            service = JobService(jobs)
            first = service.save_disposition(job_id, "use")
            second = service.save_disposition(job_id, "done")
            persisted = json.loads(
                (jobs / f"{job_id}.json").read_text(encoding="utf-8")
            )

        self.assertEqual(first["run_disposition"]["kind"], "use")
        self.assertIsNone(first["run_disposition"]["previous_kind"])
        self.assertEqual(second["run_disposition"]["kind"], "done")
        self.assertEqual(second["run_disposition"]["previous_kind"], "use")
        self.assertEqual(
            second["run_disposition"]["operator_next_step"]["action"], "none"
        )
        self.assertEqual(persisted["plan_id"], "plan_abc")

    def test_missing_sibling_is_absent_not_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "job_" + "d" * 32
            jobs = _write_completed_train(root, job_id=job_id)
            got = JobService(jobs).get(job_id, include_validation_report=False)

        self.assertNotIn("run_disposition", got)
        self.assertNotIn("run_disposition_error", got)
        self.assertNotEqual(got.get("run_disposition", {}).get("kind"), "use")

    def test_corrupt_sibling_sets_error_not_use(self) -> None:
        cases = ("invalid-json", "invalid-schema-use", "symlink")
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    job_id = "job_" + "e" * 32
                    jobs = _write_completed_train(root, job_id=job_id)
                    sibling = jobs / f"{job_id}.disposition.json"
                    if case == "invalid-json":
                        sibling.write_text("{", encoding="utf-8")
                    elif case == "invalid-schema-use":
                        sibling.write_text(
                            json.dumps(
                                {
                                    "schema_version": "aptus.run-disposition.v1",
                                    "kind": "use",
                                    "source": "operator-attested",
                                }
                            ),
                            encoding="utf-8",
                        )
                    else:
                        target = root / "elsewhere.json"
                        target.write_text("{}", encoding="utf-8")
                        os.symlink(target, sibling)
                    got = JobService(jobs).get(job_id, include_validation_report=False)
                    self.assertNotIn("run_disposition", got)
                    self.assertIn("run_disposition_error", got)
                    self.assertIsInstance(got["run_disposition_error"], str)
                    self.assertTrue(got["run_disposition_error"].strip())
                    self.assertNotEqual(
                        got.get("run_disposition", {}).get("kind"), "use"
                    )


if __name__ == "__main__":
    unittest.main()
