import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.aptus_audit.checks import run_check


class CheckRunnerTests(unittest.TestCase):
    def test_run_check_captures_reproducible_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            record = run_check(
                check_id="unit-success",
                command=[sys.executable, "-c", "print('ok')"],
                cwd=Path(temp_dir),
                timeout_seconds=5,
            )

            self.assertEqual(record["exit_code"], 0)
            self.assertEqual(record["status"], "passed")
            self.assertEqual(record["stdout_preview"], "ok\n")
            self.assertEqual(len(record["stdout_sha256"]), 64)
            self.assertEqual(record["command"][0], sys.executable)
            self.assertGreaterEqual(record["duration_ms"], 0)

    def test_run_check_records_failure_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            record = run_check(
                check_id="unit-failure",
                command=[sys.executable, "-c", "import sys; sys.exit(3)"],
                cwd=Path(temp_dir),
                timeout_seconds=5,
            )

            self.assertEqual(record["exit_code"], 3)
            self.assertEqual(record["status"], "failed")
            self.assertFalse(record["timed_out"])

    def test_run_check_forwards_proxy_only_when_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                "os.environ",
                {"HTTPS_PROXY": "http://proxy.invalid:8080"},
                clear=False,
            ):
                record = run_check(
                    check_id="unit-proxy",
                    command=[
                        sys.executable,
                        "-c",
                        "import os; print(os.environ.get('HTTPS_PROXY', 'missing'))",
                    ],
                    cwd=Path(temp_dir),
                    timeout_seconds=5,
                    inherit_proxy=True,
                )

            self.assertEqual(record["stdout_preview"], "http://proxy.invalid:8080\n")

    def test_run_check_drops_proxy_from_legacy_process_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                "os.environ",
                {"HTTPS_PROXY": "http://user:password@proxy.invalid:8080"},
                clear=False,
            ):
                record = run_check(
                    check_id="unit-no-proxy",
                    command=[
                        sys.executable,
                        "-c",
                        "import os; print(os.environ.get('HTTPS_PROXY', 'missing'))",
                    ],
                    cwd=Path(temp_dir),
                    timeout_seconds=5,
                )

            self.assertEqual(record["stdout_preview"], "missing\n")
            self.assertNotIn("password", str(record))

    def test_run_check_normalizes_partial_timeout_output(self) -> None:
        command = [sys.executable, "-c", "print('ignored')"]
        timeout = subprocess.TimeoutExpired(
            cmd=command,
            timeout=0.05,
            output=b"partial\n",
            stderr=b"",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "tools.aptus_audit.checks.subprocess.run",
                side_effect=timeout,
            ) as run_mock:
                record = run_check(
                    check_id="unit-timeout",
                    command=command,
                    cwd=Path(temp_dir),
                    timeout_seconds=0.05,
                )

            run_mock.assert_called_once()
            positional, keyword = run_mock.call_args
            self.assertEqual(positional, (command,))
            self.assertEqual(keyword["cwd"], Path(temp_dir))
            self.assertEqual(keyword["timeout"], 0.05)
            self.assertTrue(keyword["capture_output"])
            self.assertTrue(keyword["text"])
            self.assertFalse(keyword["check"])
            self.assertEqual(record["status"], "timed_out")
            self.assertTrue(record["timed_out"])
            self.assertEqual(record["stdout_preview"], "partial\n")
            self.assertEqual(len(record["stdout_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
