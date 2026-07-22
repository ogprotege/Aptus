import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aptus.runtime_lease as runtime_lease
from aptus.runtime_lease import (
    LEASE_ENV,
    _lease_paths,
    portable_execution_lease,
    require_execution_lease,
    run_with_lease,
)


class RuntimeLeaseTests(unittest.TestCase):
    def test_portable_lease_blocks_an_unrelated_direct_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "lease-parent"
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            inherited = os.environ.pop(LEASE_ENV, None)
            try:
                with portable_execution_lease(
                    bundle, action="pilot", _lease_parent=parent
                ):
                    token = os.environ.pop(LEASE_ENV)
                    try:
                        with self.assertRaisesRegex(RuntimeError, "active GPU action"):
                            with portable_execution_lease(
                                bundle, action="train", _lease_parent=parent
                            ):
                                self.fail("conflicting lease unexpectedly succeeded")
                    finally:
                        os.environ[LEASE_ENV] = token
                _root, lease_path, _lock_path = _lease_paths(parent)
                self.assertFalse(lease_path.exists())
            finally:
                os.environ.pop(LEASE_ENV, None)
                if inherited is not None:
                    os.environ[LEASE_ENV] = inherited

    def test_child_inherits_token_and_direct_unleased_execution_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "lease-parent"
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            inherited = os.environ.pop(LEASE_ENV, None)
            try:
                with self.assertRaisesRegex(RuntimeError, "Direct train.py"):
                    require_execution_lease(_lease_parent=parent)
                with portable_execution_lease(
                    bundle, action="train", _lease_parent=parent
                ):
                    completed = run_with_lease(
                        [
                            sys.executable,
                            "-c",
                            (
                                "import os,sys; "
                                f"sys.exit(0 if os.environ.get('{LEASE_ENV}') else 9)"
                            ),
                        ],
                        cwd=bundle,
                        _lease_parent=parent,
                    )
                    self.assertEqual(completed.returncode, 0)
            finally:
                os.environ.pop(LEASE_ENV, None)
                if inherited is not None:
                    os.environ[LEASE_ENV] = inherited

    @unittest.skipUnless(os.name == "posix", "POSIX process-group contract")
    def test_orphaned_child_group_is_terminated_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "lease-parent"
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            inherited = os.environ.pop(LEASE_ENV, None)
            try:
                with portable_execution_lease(
                    bundle, action="pilot", _lease_parent=parent
                ):
                    with self.assertRaisesRegex(RuntimeError, "descendants remained"):
                        run_with_lease(
                            [
                                sys.executable,
                                "-c",
                                (
                                    "import subprocess,sys; "
                                    "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"
                                ),
                            ],
                            cwd=bundle,
                            _lease_parent=parent,
                        )
            finally:
                os.environ.pop(LEASE_ENV, None)
                if inherited is not None:
                    os.environ[LEASE_ENV] = inherited

    @unittest.skipUnless(os.name == "posix", "POSIX process-group contract")
    def test_registration_failure_terminates_the_spawned_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "lease-parent"
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            inherited = os.environ.pop(LEASE_ENV, None)
            spawned = []
            original_popen = runtime_lease.subprocess.Popen
            original_write = runtime_lease._write_lease

            def tracked_popen(*args, **kwargs):
                process = original_popen(*args, **kwargs)
                command = args[0] if args else kwargs.get("args", [])
                if (
                    command
                    and command[0] == sys.executable
                    and "time.sleep(30)" in str(command)
                ):
                    spawned.append(process)
                return process

            def fail_child_registration(path, value):
                if value.get("process_pid") != os.getpid():
                    raise OSError("lease registration failed")
                original_write(path, value)

            try:
                with portable_execution_lease(
                    bundle, action="pilot", _lease_parent=parent
                ):
                    with (
                        patch.object(
                            runtime_lease.subprocess, "Popen", side_effect=tracked_popen
                        ),
                        patch.object(
                            runtime_lease,
                            "_write_lease",
                            side_effect=fail_child_registration,
                        ),
                    ):
                        with self.assertRaisesRegex(OSError, "registration failed"):
                            run_with_lease(
                                [
                                    sys.executable,
                                    "-c",
                                    "import time; time.sleep(30)",
                                ],
                                cwd=bundle,
                                _lease_parent=parent,
                            )
                self.assertEqual(len(spawned), 1)
                self.assertIsNotNone(spawned[0].poll())
                self.assertFalse(runtime_lease._process_group_alive(spawned[0].pid))
            finally:
                for process in spawned:
                    if process.poll() is None:
                        os.killpg(process.pid, 9)
                        process.wait(timeout=2)
                os.environ.pop(LEASE_ENV, None)
                if inherited is not None:
                    os.environ[LEASE_ENV] = inherited


if __name__ == "__main__":
    unittest.main()
