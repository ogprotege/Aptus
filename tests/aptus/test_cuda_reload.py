import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
RELOAD = REPOSITORY / "src" / "aptus" / "_bundle_programs" / "cuda" / "reload.py"


class CudaReloadTests(unittest.TestCase):
    def test_reload_program_exists(self) -> None:
        self.assertTrue(RELOAD.is_file(), "CUDA reload.py must exist for M7-C")

    def test_rejects_unexpected_parent_pid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            adapter = root / "final"
            adapter.mkdir()
            export = root / "final-export.json"
            export.write_text("{}\n", encoding="utf-8")
            output = root / "reload-evidence.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RELOAD),
                    "--adapter-path",
                    str(adapter),
                    "--final-export",
                    str(export),
                    "--output",
                    str(output),
                    "--expected-parent-pid",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("fresh child process", completed.stderr.lower())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
