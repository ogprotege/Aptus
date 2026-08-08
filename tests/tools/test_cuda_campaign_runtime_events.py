from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - qualifying capture is Linux-only.
    fcntl = None

from tools.cuda_campaign.runtime_events import (
    RUNTIME_BOUNDARY_SCHEMA,
    RuntimeBoundaryError,
    RuntimeBoundaryJournalReader,
)


RUN_ID = "xrun_" + "a" * 32
JOB_ID = "job_" + "b" * 32
WALL = "2026-08-08T12:00:00+00:00"


def encoded_lines(records: list[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def write_lines(path: Path, records: list[dict[str, object]]) -> None:
    payload = encoded_lines(records)
    path.write_bytes(payload)
    path.chmod(0o600)


def boundary(
    event_type: str, monotonic_ns: int, **changes: object
) -> dict[str, object]:
    finished = event_type.endswith("finished")
    value: dict[str, object] = {
        "schema_version": RUNTIME_BOUNDARY_SCHEMA,
        "experiment_run_id": RUN_ID,
        "job_id": JOB_ID,
        "monotonic_ns": monotonic_ns,
        "wall_time_utc": WALL,
        "event_type": event_type,
        "phase": "training",
        "action": "train",
        "native_outcome": "passed" if finished else None,
        "reason_code": "NONE",
    }
    value.update(changes)
    return value


class RuntimeBoundaryJournalTests(unittest.TestCase):
    def _reader(
        self, path: Path, *, action: str = "train"
    ) -> RuntimeBoundaryJournalReader:
        metadata = path.stat()
        return RuntimeBoundaryJournalReader(
            path,
            expected_file_identity=f"{metadata.st_dev}:{metadata.st_ino}",
            experiment_run_id=RUN_ID,
            job_id=JOB_ID,
            action=action,
        )

    def test_reader_drains_only_new_identity_bound_canonical_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            write_lines(path, [boundary("training.started", 10)])
            reader = self._reader(path)
            first = reader.drain()
            self.assertEqual([item.event_type for item in first], ["training.started"])
            write_lines(
                path,
                [
                    boundary("training.started", 10),
                    boundary("training.finished", 20),
                ],
            )
            second = reader.drain()
            self.assertEqual(
                [item.event_type for item in second], ["training.finished"]
            )
            self.assertEqual(reader.verified_bytes, path.read_bytes())
            self.assertEqual(reader.drain(), ())

    @unittest.skipIf(fcntl is None, "POSIX journal locking is required")
    def test_exclusive_drain_samples_before_a_blocked_emitter_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            write_lines(path, [boundary("training.started", 10)])
            reader = self._reader(path)
            parent_sampling = threading.Event()
            writer_blocked = threading.Event()
            writer_acquired = threading.Event()
            writer_errors: list[BaseException] = []

            def append_after_lock() -> None:
                assert fcntl is not None
                descriptor = -1
                try:
                    if not parent_sampling.wait(2):
                        raise AssertionError("parent never sampled under its lock")
                    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
                    try:
                        fcntl.flock(
                            descriptor,
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    except BlockingIOError:
                        writer_blocked.set()
                    else:
                        raise AssertionError(
                            "emitter bypassed the parent exclusive lock"
                        )
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    writer_acquired.set()
                    emitted = boundary("training.finished", 200)
                    payload = encoded_lines([emitted])
                    self.assertEqual(os.write(descriptor, payload), len(payload))
                    os.fsync(descriptor)
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except BaseException as error:
                    writer_errors.append(error)
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)

            writer = threading.Thread(target=append_after_lock, daemon=True)
            writer.start()

            observations: dict[str, bool] = {}

            def sample_parent() -> int:
                parent_sampling.set()
                observations["writer_blocked"] = writer_blocked.wait(2)
                observations["writer_acquired_while_locked"] = writer_acquired.is_set()
                return 100

            new_records, sampled_ns = reader.drain_and_sample_monotonic(sample_parent)
            writer.join(2)
            self.assertFalse(writer.is_alive())
            self.assertEqual(writer_errors, [])
            self.assertTrue(observations["writer_blocked"])
            self.assertFalse(observations["writer_acquired_while_locked"])
            self.assertTrue(writer_acquired.is_set())
            self.assertEqual(sampled_ns, 100)
            self.assertEqual(
                [item.event_type for item in new_records], ["training.started"]
            )
            appended = reader.drain()
            self.assertEqual(
                [item.event_type for item in appended], ["training.finished"]
            )
            self.assertGreaterEqual(appended[0].monotonic_ns, sampled_ns)

    @unittest.skipIf(fcntl is None, "POSIX journal locking is required")
    def test_exclusive_drain_clock_error_unlocks_without_committing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            write_lines(path, [boundary("training.started", 10)])
            reader = self._reader(path)

            def broken_clock() -> int:
                raise LookupError("clock unavailable")

            with self.assertRaisesRegex(RuntimeBoundaryError, "clock failed"):
                reader.drain_and_sample_monotonic(broken_clock)
            self.assertEqual(reader.records, ())
            self.assertEqual(reader.verified_bytes, b"")

            descriptor = os.open(path, os.O_RDONLY)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

            records, sampled_ns = reader.drain_and_sample_monotonic(lambda: 50)
            self.assertEqual(
                [item.event_type for item in records], ["training.started"]
            )
            self.assertEqual(sampled_ns, 50)

    def test_exclusive_drain_rejects_invalid_clock_backward_and_partial_input(
        self,
    ) -> None:
        for invalid in (True, -1, 1.5):
            with (
                self.subTest(invalid=invalid),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path = Path(temporary) / "events.jsonl"
                write_lines(path, [boundary("training.started", 10)])
                reader = self._reader(path)
                with self.assertRaisesRegex(RuntimeBoundaryError, "invalid value"):
                    reader.drain_and_sample_monotonic(lambda: invalid)  # type: ignore[return-value]
                self.assertEqual(reader.records, ())

        clock_calls = 0

        def sampled_clock() -> int:
            nonlocal clock_calls
            clock_calls += 1
            return 100

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            write_lines(
                path,
                [
                    boundary("training.started", 20),
                    boundary("training.finished", 10),
                ],
            )
            with self.assertRaisesRegex(RuntimeBoundaryError, "backward"):
                self._reader(path).drain_and_sample_monotonic(sampled_clock)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            write_lines(path, [boundary("training.started", 10)])
            path.write_bytes(path.read_bytes() + b"{")
            with self.assertRaisesRegex(RuntimeBoundaryError, "partial"):
                self._reader(path).drain_and_sample_monotonic(sampled_clock)

        self.assertEqual(clock_calls, 0)

    def test_reader_rejects_identity_action_canonical_and_time_tampering(self) -> None:
        mutations = (
            {"experiment_run_id": "xrun_" + "c" * 32},
            {"action": "pilot"},
            {"phase": "wrong"},
        )
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path = Path(temporary) / "events.jsonl"
                write_lines(path, [boundary("training.started", 10, **mutation)])
                with self.assertRaises(RuntimeBoundaryError):
                    self._reader(path).drain()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            write_lines(
                path,
                [
                    boundary("training.started", 20),
                    boundary("training.finished", 10),
                ],
            )
            with self.assertRaisesRegex(RuntimeBoundaryError, "backward"):
                self._reader(path).drain()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text(
                json.dumps(boundary("training.started", 10)) + "\n",
                encoding="utf-8",
            )
            path.chmod(0o600)
            with self.assertRaisesRegex(RuntimeBoundaryError, "canonical"):
                self._reader(path).drain()

    def test_reader_rejects_partial_replaced_and_nonprivate_journals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "events.jsonl"
            write_lines(path, [boundary("training.started", 10)])
            reader = self._reader(path)
            reader.drain()
            path.write_bytes(path.read_bytes() + b"{")
            with self.assertRaisesRegex(RuntimeBoundaryError, "partial"):
                reader.drain()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "events.jsonl"
            write_lines(path, [boundary("training.started", 10)])
            reader = self._reader(path)
            path.unlink()
            write_lines(path, [boundary("training.started", 10)])
            with self.assertRaisesRegex(RuntimeBoundaryError, "integrity"):
                reader.drain()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            write_lines(path, [boundary("training.started", 10)])
            path.chmod(0o644)
            with self.assertRaisesRegex(RuntimeBoundaryError, "integrity"):
                self._reader(path).drain()

    def test_reader_rejects_post_capture_rewrite_even_with_same_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            write_lines(path, [boundary("training.started", 10)])
            reader = self._reader(path)
            reader.drain()
            write_lines(path, [boundary("training.started", 11)])
            with self.assertRaisesRegex(RuntimeBoundaryError, "append-only"):
                reader.drain()


if __name__ == "__main__":
    unittest.main()
