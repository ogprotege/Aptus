import hashlib
import json
import os
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from aptus.diagnostics import (
    DIAGNOSTIC_SCHEMA_VERSION,
    DOCTOR_SCHEMA_VERSION,
    build_diagnostics,
    build_doctor_report,
    create_diagnostic_archive,
)


def _inventory(path: str) -> dict[str, object]:
    return {
        "schema_version": "aptus.runtime-inventory.v1",
        "interpreters": [
            {
                "path": path,
                "source": "current-process",
                "python_version": "3.12.12",
                "runtimes": {
                    "mlx-lm": {
                        "available": True,
                        "compatible": True,
                        "versions": {"mlx": "0.31.2", "mlx-lm": "0.31.3"},
                        "expected_versions": {
                            "mlx": "0.31.2",
                            "mlx-lm": "0.31.3",
                        },
                    }
                },
                "error": None,
            }
        ],
        "available": {
            "mlx-lm": [path],
            "pytorch-mps": [],
            "transformers-peft-cuda": [],
        },
        "compatible": {
            "mlx-lm": [path],
            "pytorch-mps": [],
            "transformers-peft-cuda": [],
        },
        "configuration": {
            "mlx-lm": "APTUS_MLX_PYTHON",
            "pytorch-mps": "APTUS_PYTORCH_PYTHON",
            "transformers-peft-cuda": "APTUS_CUDA_PYTHON",
        },
    }


class DiagnosticTests(unittest.TestCase):
    def test_doctor_reports_readiness_without_exposing_environment_values(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            root = Path(temporary) / "state"
            root.mkdir(mode=0o700)
            interpreter = str(Path(temporary) / "runtime" / "bin" / "python")
            with (
                patch("aptus.diagnostics.platform.system", return_value="Darwin"),
                patch("aptus.diagnostics.platform.machine", return_value="arm64"),
                patch.dict(os.environ, {"APTUS_MLX_PYTHON": interpreter}),
            ):
                report = build_doctor_report(root, inventory=_inventory(interpreter))

        self.assertEqual(report["schema_version"], DOCTOR_SCHEMA_VERSION)
        self.assertEqual(report["status"], "ready")
        self.assertTrue(report["configured_runtime_keys"]["mlx-lm"])
        self.assertFalse(report["installation_performed"])
        self.assertTrue(
            report["runtime_inventory"]["interpreters"][0]["runtimes"]["mlx-lm"][
                "compatible"
            ]
        )
        serialized = json.dumps(report)
        self.assertNotIn(str(Path.home()), serialized)
        self.assertNotIn(interpreter, serialized)
        self.assertIn("$HOME", serialized)

    def test_doctor_fails_closed_for_importable_but_unpinned_mlx(self) -> None:
        inventory = _inventory("/usr/bin/python3")
        inventory["compatible"]["mlx-lm"] = []
        inventory["interpreters"][0]["runtimes"]["mlx-lm"].update(
            {
                "compatible": False,
                "versions": {"mlx": "0.31.1", "mlx-lm": "0.31.3"},
            }
        )
        with (
            patch("aptus.diagnostics.platform.system", return_value="Darwin"),
            patch("aptus.diagnostics.platform.machine", return_value="arm64"),
        ):
            report = build_doctor_report(Path("/tmp/aptus-state"), inventory=inventory)

        self.assertEqual(report["status"], "action-required")
        self.assertEqual(report["runtime_inventory"]["compatible_counts"]["mlx-lm"], 0)

    def test_diagnostics_summarize_state_without_project_names_or_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "state"
            jobs = root / "jobs"
            revisions = root / "projects" / ("project_" + "a" * 32) / "revisions"
            jobs.mkdir(parents=True, mode=0o700)
            root.chmod(0o700)
            revisions.mkdir(parents=True, mode=0o700)
            (jobs / "job.json").write_text(
                json.dumps(
                    {
                        "state": "completed",
                        "action": "preflight",
                        "log": "private log body",
                    }
                ),
                encoding="utf-8",
            )
            (revisions.parent / "project.json").write_text(
                json.dumps({"name": "Private project name"}), encoding="utf-8"
            )
            (revisions / ("revision_" + "b" * 32 + ".json")).write_text(
                "{}", encoding="utf-8"
            )
            with patch(
                "aptus.diagnostics.probe_apple_platform",
                side_effect=ValueError("not Apple silicon"),
            ):
                report = build_diagnostics(
                    root, inventory=_inventory("/usr/bin/python3")
                )

        self.assertEqual(report["schema_version"], DIAGNOSTIC_SCHEMA_VERSION)
        self.assertEqual(report["doctor"]["state"]["jobs"]["records"], 1)
        self.assertEqual(
            report["doctor"]["state"]["jobs"]["by_state"], {"completed": 1}
        )
        self.assertEqual(report["doctor"]["state"]["projects"]["records"], 1)
        self.assertEqual(report["doctor"]["state"]["projects"]["revisions"], 1)
        serialized = json.dumps(report)
        self.assertNotIn("private log body", serialized)
        self.assertNotIn("Private project name", serialized)

    def test_diagnostics_redact_paths_outside_home_and_runtime_errors(self) -> None:
        inventory = _inventory("/Volumes/Private Runtime/bin/python")
        inventory["interpreters"][0]["error"] = (
            "Probe failed while reading /Volumes/Private Runtime/secret.json"
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary) / "state"
            root.mkdir(mode=0o700)
            jobs = root / "jobs"
            jobs.mkdir(mode=0o700)
            (jobs / "job.json").write_text(
                json.dumps(
                    {
                        "state": "completed",
                        "action": "/Volumes/Private Runtime/secret.json",
                    }
                ),
                encoding="utf-8",
            )
            report = build_diagnostics(root, inventory=inventory)

        serialized = json.dumps(report)
        self.assertNotIn("/Volumes/Private Runtime", serialized)
        self.assertNotIn("Runtime/secret.json", serialized)
        self.assertNotIn("Probe failed", serialized)
        self.assertNotIn(str(root), serialized)
        self.assertNotIn("bin/python", serialized)
        self.assertFalse(report["doctor"]["runtime_inventory"]["paths_included"])

    def test_archive_is_private_no_clobber_and_self_verifying(self) -> None:
        diagnostics = {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "privacy": {"logs_included": False},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "support.zip"
            result = create_diagnostic_archive(
                Path(temporary) / "state", output, diagnostics=diagnostics
            )
            self.assertEqual(result, output)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    ["README.txt", "diagnostics.json", "manifest.json"],
                )
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(
                    manifest["files"]["diagnostics.json"],
                    hashlib.sha256(archive.read("diagnostics.json")).hexdigest(),
                )
            with self.assertRaises(FileExistsError):
                create_diagnostic_archive(
                    Path(temporary) / "state", output, diagnostics=diagnostics
                )


if __name__ == "__main__":
    unittest.main()
