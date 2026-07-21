import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from aptus.cli import main


class CliIntegrationTests(unittest.TestCase):
    def test_plan_command_profiles_plans_generates_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "data.jsonl"
            dataset.write_text(
                '{"text":"A concise fine-tuning example."}\n',
                encoding="utf-8",
            )
            output = root / "bundle"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "plan",
                        "--model-id",
                        "meta-llama/example-7b",
                        "--revision",
                        "a" * 40,
                        "--family",
                        "llama",
                        "--parameters-b",
                        "7",
                        "--hidden-size",
                        "4096",
                        "--layers",
                        "32",
                        "--context-length",
                        "4096",
                        "--license",
                        "example",
                        "--confirm-training-allowed",
                        "--dataset",
                        str(dataset),
                        "--backend",
                        "cuda",
                        "--gpu-count",
                        "1",
                        "--vram-gib",
                        "12",
                        "--bf16",
                        "--four-bit",
                        "--host-ram-gib",
                        "64",
                        "--reserve-gib",
                        "2",
                        "--objective",
                        "quality",
                        "--sequence-length",
                        "64",
                        "--effective-batch-size",
                        "8",
                        "--epochs",
                        "1",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            plan = json.loads((output / "plan.json").read_text())
            report = json.loads((output / "validation-report.json").read_text())
            self.assertEqual(plan["recommended"]["method"], "qlora")
            self.assertEqual(report["state"], "static-pass")
            self.assertIn("qlora", stdout.getvalue().lower())

    def test_plan_command_fails_without_explicit_training_permission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "plan",
                        "--model-id",
                        "example/model",
                        "--revision",
                        "a" * 40,
                        "--family",
                        "llama",
                        "--parameters-b",
                        "1",
                        "--hidden-size",
                        "1024",
                        "--layers",
                        "16",
                        "--context-length",
                        "2048",
                        "--license",
                        "unknown",
                        "--dataset",
                        str(dataset),
                        "--backend",
                        "cuda",
                        "--gpu-count",
                        "1",
                        "--vram-gib",
                        "24",
                        "--four-bit",
                        "--host-ram-gib",
                        "32",
                        "--reserve-gib",
                        "2",
                        "--objective",
                        "quality",
                        "--output",
                        str(root / "bundle"),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("training permission", stderr.getvalue().lower())

    def test_plan_command_rejects_explicit_zero_sequence_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "data.txt"
            dataset.write_text("example\n", encoding="utf-8")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "plan",
                        "--model-id",
                        "example/model",
                        "--revision",
                        "a" * 40,
                        "--family",
                        "llama",
                        "--parameters-b",
                        "1",
                        "--hidden-size",
                        "1024",
                        "--layers",
                        "16",
                        "--context-length",
                        "2048",
                        "--license",
                        "apache-2.0",
                        "--confirm-training-allowed",
                        "--dataset",
                        str(dataset),
                        "--backend",
                        "cuda",
                        "--gpu-count",
                        "1",
                        "--vram-gib",
                        "24",
                        "--four-bit",
                        "--host-ram-gib",
                        "32",
                        "--reserve-gib",
                        "2",
                        "--objective",
                        "quality",
                        "--sequence-length",
                        "0",
                        "--output",
                        str(root / "bundle"),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("sequence_length must be positive", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
