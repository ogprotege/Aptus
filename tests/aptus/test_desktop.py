import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aptus.desktop import _private_directory, _write_ready_file, main


class DesktopRuntimeTests(unittest.TestCase):
    def test_ready_file_is_private_and_contains_only_endpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "session" / "ready.json"
            _write_ready_file(ready, port=43119)
            payload = json.loads(ready.read_text(encoding="utf-8"))
            mode = ready.stat().st_mode & 0o777
        self.assertEqual(payload["host"], "127.0.0.1")
        self.assertEqual(payload["port"], 43119)
        self.assertEqual(payload["version"], "0.2.0")
        self.assertNotIn("token", payload)
        self.assertEqual(mode, 0o600)

    def test_private_directory_is_created_with_user_only_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "state"
            resolved = _private_directory(target)
            mode = resolved.stat().st_mode & 0o777
        self.assertEqual(resolved, target.resolve())
        self.assertEqual(mode, 0o700)

    def test_private_directory_fails_closed_when_permissions_cannot_be_set(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "state"
            with patch.object(Path, "chmod", side_effect=OSError("chmod denied")):
                with self.assertRaisesRegex(OSError, "chmod denied"):
                    _private_directory(target)

    def test_missing_desktop_token_fails_before_starting_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {}, clear=True):
                result = main(
                    [
                        "--state-dir",
                        str(Path(temporary) / "state"),
                        "--ready-file",
                        str(Path(temporary) / "ready.json"),
                    ]
                )
        self.assertEqual(result, 2)
