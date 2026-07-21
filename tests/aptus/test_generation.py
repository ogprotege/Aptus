import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aptus.domain import (
    Backend,
    DatasetProfile,
    DeviceSpec,
    HardwareSpec,
    MeasurementKind,
    ModelSpec,
    Objective,
    TrainingTarget,
    ValidationState,
    gibibytes,
)
from aptus.generation import generate_bundle
from aptus.planning import plan_training
from aptus.validation import validate_bundle


def make_plan(dataset_path: Path, *, vram_gb: int = 12):
    dataset = DatasetProfile(
        source_path=dataset_path,
        source_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        source_format="jsonl",
        schema_name="text",
        example_count=10,
        total_estimated_tokens=1_000,
        sequence_p50=32,
        sequence_p95=64,
        sequence_max=64,
        measurement=MeasurementKind.ESTIMATED,
    )
    model = ModelSpec(
        model_id="meta-llama/example-7b",
        revision="a" * 40,
        family="llama",
        parameters=7_000_000_000,
        hidden_size=4096,
        layers=32,
        context_length=4096,
        license_name="example",
        training_allowed=True,
    )
    hardware = HardwareSpec(
        devices=(
            DeviceSpec(
                name="CUDA GPU 0",
                backend=Backend.CUDA,
                total_vram_bytes=gibibytes(vram_gb),
                supports_bf16=True,
                supports_4bit=True,
            ),
        ),
        host_ram_bytes=gibibytes(64),
        reserve_per_device_bytes=gibibytes(2),
    )
    target = TrainingTarget(
        objective=Objective.QUALITY,
        sequence_length=64,
        effective_batch_size=8,
        max_epochs=1,
    )
    return plan_training(
        model=model,
        dataset=dataset,
        hardware=hardware,
        target=target,
    )


class BundleGenerationTests(unittest.TestCase):
    def test_generates_complete_static_validated_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data.jsonl"
            dataset_path.write_text('{"text":"hello world"}\n', encoding="utf-8")
            output = root / "bundle"
            plan = make_plan(dataset_path)

            generate_bundle(plan, output)
            report = validate_bundle(output)

            self.assertEqual(report.state, ValidationState.STATIC_PASS)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "README.md",
                    "plan.json",
                    "plan_contract.py",
                    "requirements.txt",
                    "train.py",
                    "validate.py",
                    "validation-report.json",
                },
            )
            ast.parse((output / "train.py").read_text())
            source = (output / "train.py").read_text()
            self.assertNotIn(str(dataset_path), source)
            self.assertNotIn(plan.model.model_id, source)
            self.assertIn("active_tokens = (batch[\"labels\"] != -100).sum()", source)
            self.assertIn("loss = model(**batch).loss * active_tokens", source)
            self.assertIn("parameter.grad.div_(window_tokens)", source)
            self.assertNotIn("parameter.grad.div_(window_steps)", source)
            self.assertNotIn("model(**batch).loss / accumulation", source)
            requirements = (output / "requirements.txt").read_text()
            self.assertIn("bitsandbytes==0.49.2", requirements)

    def test_generated_validate_only_mode_requires_no_ml_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data.jsonl"
            dataset_path.write_text('{"text":"hello"}\n', encoding="utf-8")
            output = root / "bundle"
            generate_bundle(make_plan(dataset_path), output)

            completed = subprocess.run(
                [sys.executable, str(output / "train.py"), "--validate-only"],
                cwd=output,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("validation passed", completed.stdout.lower())
            report = json.loads((output / "validation-report.json").read_text())
            self.assertEqual(report["state"], "environment-pass")

    def test_validator_rejects_tampered_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data.jsonl"
            dataset_path.write_text('{"text":"hello"}\n', encoding="utf-8")
            output = root / "bundle"
            generate_bundle(make_plan(dataset_path), output)
            (output / "train.py").write_text("def broken(:\n", encoding="utf-8")

            report = validate_bundle(output)
            completed = subprocess.run(
                [sys.executable, str(output / "train.py"), "--validate-only"],
                cwd=output,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(report.state, ValidationState.INVALID)
            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(
                any(finding.code == "PYTHON_PARSE_ERROR" for finding in report.findings)
            )

    def test_validator_and_generated_entrypoint_reject_empty_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data.jsonl"
            dataset_path.write_text('{"text":"hello"}\n', encoding="utf-8")
            output = root / "bundle"
            generate_bundle(make_plan(dataset_path), output)
            (output / "plan.json").write_text("{}\n", encoding="utf-8")

            report = validate_bundle(output)
            completed = subprocess.run(
                [sys.executable, str(output / "train.py"), "--validate-only"],
                cwd=output,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(report.state, ValidationState.INVALID)
            self.assertNotEqual(completed.returncode, 0)

    def test_validator_requires_exact_method_dependency_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data.jsonl"
            dataset_path.write_text('{"text":"hello"}\n', encoding="utf-8")
            output = root / "bundle"
            generate_bundle(make_plan(dataset_path), output)
            requirements = output / "requirements.txt"
            requirements.write_text(
                requirements.read_text().replace(
                    "bitsandbytes==0.49.2\n",
                    "",
                ),
                encoding="utf-8",
            )

            report = validate_bundle(output)
            completed = subprocess.run(
                [sys.executable, str(output / "train.py"), "--validate-only"],
                cwd=output,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(report.state, ValidationState.INVALID)
            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(
                any(
                    finding.code == "DEPENDENCY_SET_MISMATCH"
                    for finding in report.findings
                )
            )

    def test_generated_runtime_state_can_persist_smoke_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data.jsonl"
            dataset_path.write_text('{"text":"hello"}\n', encoding="utf-8")
            output = root / "bundle"
            generate_bundle(make_plan(dataset_path, vram_gb=24), output)
            train_path = output / "train.py"
            spec = importlib.util.spec_from_file_location(
                "generated_train",
                train_path,
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.path.insert(0, str(output))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.remove(str(output))

            module.update_validation_state(
                "smoke-pass",
                "offline optimizer step passed",
            )

            report = json.loads((output / "validation-report.json").read_text())
            self.assertEqual(report["state"], "smoke-pass")
            self.assertIn(
                "offline optimizer step passed",
                report["runtime_evidence"],
            )
            self.assertEqual(len(report["artifact_fingerprint"]), 64)

            module.update_validation_state(
                "environment-pass",
                "later validation passed",
            )
            report = json.loads((output / "validation-report.json").read_text())
            self.assertEqual(report["state"], "smoke-pass")

    def test_artifact_change_invalidates_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data.jsonl"
            dataset_path.write_text('{"text":"hello"}\n', encoding="utf-8")
            output = root / "bundle"
            generate_bundle(make_plan(dataset_path, vram_gb=24), output)
            train_path = output / "train.py"
            spec = importlib.util.spec_from_file_location(
                "generated_train_changed",
                train_path,
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.path.insert(0, str(output))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.remove(str(output))
            module.update_validation_state("smoke-pass", "smoke passed")
            readme = output / "README.md"
            readme.write_text(readme.read_text() + "\nChanged.\n", encoding="utf-8")

            report = validate_bundle(output)

            self.assertEqual(report.state, ValidationState.STATIC_PASS)
            self.assertEqual(report.runtime_evidence, ())

    def test_invalid_static_report_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data.jsonl"
            dataset_path.write_text('{"text":"hello"}\n', encoding="utf-8")
            output = root / "bundle"
            generate_bundle(make_plan(dataset_path, vram_gb=24), output)
            train_path = output / "train.py"
            spec = importlib.util.spec_from_file_location(
                "generated_train_invalid",
                train_path,
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.path.insert(0, str(output))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.remove(str(output))
            report_path = output / "validation-report.json"
            report = json.loads(report_path.read_text())
            report["state"] = "invalid"
            report["findings"] = [
                {"code": "TEST", "message": "invalid", "severity": "error"}
            ]
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid"):
                module.update_validation_state(
                    "environment-pass",
                    "must not promote",
                )

    def test_identical_plan_generates_byte_identical_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data.jsonl"
            dataset_path.write_text('{"text":"hello"}\n', encoding="utf-8")
            plan = make_plan(dataset_path)
            first = root / "first"
            second = root / "second"

            generate_bundle(plan, first)
            generate_bundle(plan, second)

            first_files = {
                path.name: path.read_bytes()
                for path in first.iterdir()
                if path.is_file()
            }
            second_files = {
                path.name: path.read_bytes()
                for path in second.iterdir()
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)


if __name__ == "__main__":
    unittest.main()
