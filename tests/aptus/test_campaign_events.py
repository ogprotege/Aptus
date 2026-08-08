from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aptus._bundle_programs.cuda.campaign_events import emit_boundary


RUN_ID = "xrun_" + "a" * 32
JOB_ID = "job_" + "b" * 32
ENV_NAMES = {
    "APTUS_CUDA_CAMPAIGN_EVENT_SINK",
    "APTUS_CUDA_CAMPAIGN_EVENT_SINK_IDENTITY",
    "APTUS_CUDA_CAMPAIGN_EXPERIMENT_RUN_ID",
    "APTUS_CUDA_CAMPAIGN_JOB_ID",
    "RANK",
    "LOCAL_RANK",
}


class CampaignRuntimeEventTests(unittest.TestCase):
    def _environment(self, path: Path) -> dict[str, str]:
        metadata = path.stat()
        return {
            "APTUS_CUDA_CAMPAIGN_EVENT_SINK": str(path),
            "APTUS_CUDA_CAMPAIGN_EVENT_SINK_IDENTITY": (
                f"{metadata.st_dev}:{metadata.st_ino}"
            ),
            "APTUS_CUDA_CAMPAIGN_EXPERIMENT_RUN_ID": RUN_ID,
            "APTUS_CUDA_CAMPAIGN_JOB_ID": JOB_ID,
        }

    def test_ordinary_runtime_without_binding_is_noop(self) -> None:
        clean = {name: os.environ[name] for name in ENV_NAMES if name in os.environ}
        with patch.dict(os.environ, {}, clear=True):
            emit_boundary("training.started", phase="training", action="train")
        self.assertIsInstance(clean, dict)

    def test_enabled_runtime_appends_one_canonical_bound_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.touch(mode=0o600)
            path.chmod(0o600)
            with patch.dict(os.environ, self._environment(path), clear=True):
                emit_boundary(
                    "training.finished",
                    phase="training",
                    action="train",
                    native_outcome="passed",
                )

            payload = path.read_bytes()
            self.assertTrue(payload.endswith(b"\n"))
            self.assertEqual(len(payload.splitlines()), 1)
            record = json.loads(payload)
            self.assertEqual(record["experiment_run_id"], RUN_ID)
            self.assertEqual(record["job_id"], JOB_ID)
            self.assertEqual(record["event_type"], "training.finished")
            self.assertEqual(record["reason_code"], "NONE")
            self.assertEqual(
                payload,
                (
                    json.dumps(
                        record,
                        ensure_ascii=True,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8"),
            )

    def test_partial_binding_and_replaced_sink_fail_closed(self) -> None:
        with patch.dict(
            os.environ,
            {"APTUS_CUDA_CAMPAIGN_EXPERIMENT_RUN_ID": RUN_ID},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                emit_boundary("training.started", phase="training", action="train")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "events.jsonl"
            path.touch(mode=0o600)
            path.chmod(0o600)
            original_metadata = path.stat()
            environment = self._environment(path)
            replacement = root / "replacement.jsonl"
            replacement.touch(mode=0o600)
            replacement.chmod(0o600)
            replacement_metadata = replacement.stat()
            self.assertNotEqual(
                (original_metadata.st_dev, original_metadata.st_ino),
                (replacement_metadata.st_dev, replacement_metadata.st_ino),
            )
            os.replace(replacement, path)
            with patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(RuntimeError, "integrity"):
                    emit_boundary("training.started", phase="training", action="train")

    def test_semantically_impossible_boundaries_are_never_appended(self) -> None:
        cases = (
            {
                "event_type": "training.started",
                "phase": "training",
                "action": "train",
                "native_outcome": "passed",
            },
            {
                "event_type": "training.finished",
                "phase": "training",
                "action": "train",
            },
            {
                "event_type": "export.finished",
                "phase": "final-export",
                "action": "train",
                "native_outcome": "passed",
                "reason_code": "CUDA_OOM",
            },
            {
                "event_type": "pilot.phase-started",
                "phase": "training",
                "action": "train",
            },
            {
                "event_type": "pilot.phase-finished",
                "phase": "pilot-phase-1",
                "action": "pilot",
                "native_outcome": "failed",
                "reason_code": "UNFROZEN_PRIVATE_REASON",
            },
        )
        for index, fields in enumerate(cases):
            with self.subTest(index), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "events.jsonl"
                path.touch(mode=0o600)
                path.chmod(0o600)
                with (
                    patch.dict(os.environ, self._environment(path), clear=True),
                    self.assertRaises(RuntimeError),
                ):
                    emit_boundary(**fields)
                self.assertEqual(path.read_bytes(), b"")

    def test_nonzero_distributed_rank_does_not_duplicate_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.touch(mode=0o600)
            path.chmod(0o600)
            environment = self._environment(path)
            environment["RANK"] = "1"
            with patch.dict(os.environ, environment, clear=True):
                emit_boundary("export.started", phase="final-export", action="train")
            self.assertEqual(path.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
