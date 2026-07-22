import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aptus.cli import main
from aptus.execution import JobPrerequisiteError


def fact_arguments(dataset: Path) -> list[str]:
    return [
        "--model-id",
        "example/model",
        "--revision",
        "a" * 40,
        "--family",
        "llama",
        "--parameters-b",
        "1",
        "--hidden-size",
        "2048",
        "--intermediate-size",
        "8192",
        "--layers",
        "24",
        "--context-length",
        "4096",
        "--license",
        "apache-2.0",
        "--confirm-training-allowed",
        "--dataset",
        str(dataset),
        "--gpu-count",
        "1",
        "--vram-gib",
        "24",
        "--bf16",
        "--four-bit",
        "--host-ram-gib",
        "64",
        "--disk-free-gib",
        "500",
        "--objective",
        "memory",
        "--sequence-length",
        "128",
        "--effective-batch-size",
        "8",
        "--epochs",
        "1",
        "--checkpoint-steps",
        "10",
    ]


class CliIntegrationTests(unittest.TestCase):
    def test_profile_spec_plan_and_compile_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            profile_path, plan_path, bundle = (
                root / "profile.json",
                root / "plan.json",
                root / "bundle",
            )
            self.assertEqual(
                main(
                    [
                        "profile",
                        "--dataset",
                        str(dataset),
                        "--output",
                        str(profile_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    ["spec-plan", *fact_arguments(dataset), "--output", str(plan_path)]
                ),
                0,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        ["compile", "--plan", str(plan_path), "--output", str(bundle)]
                    ),
                    0,
                )
            self.assertTrue(profile_path.is_file())
            self.assertEqual(
                json.loads(plan_path.read_text())["schema_version"],
                "aptus.training-plan.v2",
            )
            self.assertFalse(
                json.loads(plan_path.read_text())["hardware"]["devices"][0][
                    "supports_8bit"
                ]
            )
            self.assertTrue((bundle / "bundle-manifest.json").is_file())

    def test_combined_plan_flow_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"example"}\n', encoding="utf-8")
            output = root / "bundle"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["plan", *fact_arguments(dataset), "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertTrue(output.with_suffix(".zip").is_file())

    def test_explicit_zero_sequence_length_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            arguments = fact_arguments(dataset)
            index = arguments.index("--sequence-length") + 1
            arguments[index] = "0"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["plan", *arguments, "--output", str(root / "bundle")])
            self.assertEqual(code, 2)
            self.assertIn("positive", stderr.getvalue())

    def test_sequence_length_is_an_explicit_required_fact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            arguments = fact_arguments(dataset)
            index = arguments.index("--sequence-length")
            del arguments[index : index + 2]
            with self.assertRaises(SystemExit):
                main(["spec-plan", *arguments, "--output", str(root / "plan.json")])

    def test_negative_intermediate_size_and_checkpoint_steps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            for option, value in (
                ("--intermediate-size", "-1"),
                ("--checkpoint-steps", "-1"),
            ):
                arguments = fact_arguments(dataset)
                index = arguments.index(option) + 1
                arguments[index] = value
                with (
                    self.subTest(option=option),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(
                        main(
                            [
                                "spec-plan",
                                *arguments,
                                "--output",
                                str(root / f"{option}.json"),
                            ]
                        ),
                        2,
                    )

    def test_runtime_validation_uses_persisted_job_service(self) -> None:
        service = MagicMock()
        service.submit.return_value = {
            "id": "job_" + "a" * 32,
            "state": "completed",
        }
        with patch("aptus.execution.JobService", return_value=service) as job_service:
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "validate",
                        "/tmp/bundle",
                        "--level",
                        "model-data",
                        "--run",
                        "--state-dir",
                        "/tmp/aptus-state-test",
                    ]
                )
        self.assertEqual(code, 0)
        job_service.assert_called_once_with(Path("/tmp/aptus-state-test") / "jobs")
        service.submit.assert_called_once_with(Path("/tmp/bundle"), action="model-data")

    def test_job_prerequisite_failure_is_a_stable_cli_error(self) -> None:
        service = MagicMock()
        service.submit.side_effect = JobPrerequisiteError(
            action="pilot",
            required_state="measured-preflight-pass",
            current_state="model-data-pass",
            reason="insufficient_state",
        )
        stderr = io.StringIO()
        with (
            patch("aptus.execution.JobService", return_value=service),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(["run", "/tmp/bundle", "--action", "pilot"])
        self.assertEqual(code, 2)
        self.assertIn("Aptus error: Cannot start pilot", stderr.getvalue())
        self.assertIn("measured-preflight-pass", stderr.getvalue())

    def test_ctrl_c_requests_owned_job_cancellation(self) -> None:
        service = MagicMock()
        job_id = "job_" + "b" * 32
        service.submit.return_value = {"id": job_id, "state": "queued"}
        service.get.side_effect = KeyboardInterrupt
        service.cancel.return_value = {"id": job_id, "state": "cancelled"}
        with patch("aptus.execution.JobService", return_value=service):
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["run", "/tmp/bundle"])
        self.assertEqual(code, 130)
        service.cancel.assert_called_once_with(job_id)

    def test_missing_training_permission_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            arguments = fact_arguments(dataset)
            arguments.remove("--confirm-training-allowed")
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["plan", *arguments, "--output", str(root / "bundle")])
            self.assertEqual(code, 2)

    def test_serve_blocks_accidental_non_loopback_execution_api(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["serve", "--host", "0.0.0.0"])
        self.assertEqual(code, 2)
        self.assertIn("Non-loopback serving is blocked", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
