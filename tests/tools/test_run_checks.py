import tempfile
import unittest
from pathlib import Path

from tools.aptus_audit.run_checks import build_check_specs, execute_legacy_checks


class LegacyCheckPlanTests(unittest.TestCase):
    def test_build_check_specs_is_explicit_and_shell_free(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / "legacy-copy"
            node_workspace = root / "node"
            legacy.mkdir()
            node_workspace.mkdir()

            specs = build_check_specs(
                repository_root=root,
                legacy_copy=legacy,
                node_workspace=node_workspace,
                python_executable=Path("/sandbox/bin/python"),
            )

            self.assertEqual(
                [spec["check_id"] for spec in specs],
                [
                    "node-server-parse",
                    "typescript-project-check",
                    "node-lock-resolution",
                    "python-requirements-resolution",
                    "python-test-collection",
                    "python-resource-scanner-smoke",
                    "python-resource-scanner-salvage-probe",
                    "python-v2-generator-construction",
                    "python-v2-generator-salvage-probe",
                ],
            )
            for spec in specs:
                self.assertIsInstance(spec["command"], list)
                self.assertTrue(spec["command"])
                self.assertNotIn("shell", spec)
            python_specs = [
                spec for spec in specs if spec["check_id"].startswith("python-")
            ]
            self.assertTrue(python_specs)
            self.assertTrue(
                all(
                    spec["command"][0] == "/sandbox/bin/python" for spec in python_specs
                )
            )
            collection = next(
                spec for spec in specs if spec["check_id"] == "python-test-collection"
            )
            self.assertEqual(
                collection["requires"],
                ["python-requirements-resolution"],
            )
            self.assertEqual(
                collection["execution_policy"],
                "blocked_without_installed_environment",
            )
            registry_checks = {
                spec["check_id"] for spec in specs if spec["registry_network"]
            }
            self.assertEqual(
                registry_checks,
                {
                    "typescript-project-check",
                    "node-lock-resolution",
                    "python-requirements-resolution",
                },
            )

    def test_execute_checks_requires_explicit_host_process_acknowledgement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with self.assertRaisesRegex(
                RuntimeError,
                "does not enforce OS-level isolation",
            ):
                execute_legacy_checks(
                    root,
                    root / "sandbox-results.jsonl",
                    allow_host_subprocesses=False,
                )

    def test_execute_checks_rejects_output_inside_legacy_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / "HyperTune"
            legacy.mkdir()

            with self.assertRaisesRegex(
                ValueError,
                "outside the legacy source tree",
            ):
                execute_legacy_checks(
                    root,
                    legacy / "sandbox-results.jsonl",
                    allow_host_subprocesses=True,
                )


if __name__ == "__main__":
    unittest.main()
