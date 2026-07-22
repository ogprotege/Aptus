import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aptus.domain import ValidationState
from aptus.generation import generate_bundle
from aptus.plan_contract import sha256_file
from aptus.validation import _read_preflight_metrics, validate_bundle

from tests.aptus.helpers import make_plan


def install_measured_run_attestation(bundle: Path) -> dict:
    plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
    candidate = plan["recommended"]
    preflight_metrics = {
        "schema_version": "aptus.preflight-metrics.v1",
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        "precision": candidate["precision"],
        "quantization": candidate.get("quantization"),
        "distribution": candidate["distribution"],
        "world_size": candidate["world_size"],
        "measured_peak_cuda_bytes": 4096,
        "scope": "synthetic-method-preflight-not-model-data-pilot",
        "trainable_parameter_census": {
            "schema_version": "aptus.trainable-parameter-census.v1",
            "method": candidate["method"],
            "parameter_scope": (
                "all-parameters"
                if candidate["method"] == "full"
                else "lora-adapter-only"
            ),
            "trainable_parameter_count": 8192,
            "trainable_tensor_count": 2,
            "frozen_parameter_count": (
                0 if candidate["method"] == "full" else 2_000_000
            ),
            "frozen_tensor_count": 0 if candidate["method"] == "full" else 1,
            "unexpected_trainable_tensor_count": 0,
            "expected_adapter_target_match_count": (
                0 if candidate["method"] == "full" else 1
            ),
            "adapter_target_instance_count": (
                0 if candidate["method"] == "full" else 1
            ),
            "incomplete_adapter_target_instance_count": 0,
            "all_values_finite": True,
            "descriptor_sha256": "b" * 64,
        },
    }
    preflight_path = bundle / "preflight-metrics.json"
    preflight_path.write_text(json.dumps(preflight_metrics), encoding="utf-8")
    pilot_path = bundle / "pilot-output" / "metrics.json"
    pilot_path.parent.mkdir()
    pilot_metrics = {"pilot": True, "candidate_id": candidate["candidate_id"]}
    pilot_path.write_text(json.dumps(pilot_metrics), encoding="utf-8")

    run_dir = bundle / "runs" / ("run_" + "a" * 32)
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True)
    artifact_path = final_dir / "artifact.bin"
    artifact_path.write_bytes(b"verified-final-artifact")
    export = {
        "schema_version": "aptus.final-export.v1",
        "verification_level": "structural-file-tree",
        "method": candidate["method"],
        "distribution": candidate["distribution"],
        "world_size": candidate["world_size"],
        "total_bytes": artifact_path.stat().st_size,
        "files": [
            {
                "path": "artifact.bin",
                "size_bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
            }
        ],
    }
    export_path = run_dir / "final-export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    per_rank = [
        {
            "rank": rank,
            "measured_peak_cuda_bytes": 2048,
            "measured_reserved_cuda_bytes": 4096,
        }
        for rank in range(candidate["world_size"])
    ]
    metrics = {
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "distribution": candidate["distribution"],
        "actual_world_size": candidate["world_size"],
        "global_step": 1,
        "per_rank_cuda_peaks": per_rank,
        "final_export": export,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    report_path = bundle / "validation-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        state="measured-run-pass",
        validation_level="measured-run",
        preflight_metrics=preflight_metrics,
        pilot_metrics=pilot_metrics,
        measured_run_completed_at="2026-07-21T00:00:00+00:00",
        final_export={
            "path": str(final_dir.resolve()),
            "manifest_sha256": sha256_file(export_path),
            "total_bytes": export["total_bytes"],
            "plan_id": plan["plan_id"],
            "candidate_id": candidate["candidate_id"],
            "distribution": candidate["distribution"],
            "world_size": candidate["world_size"],
        },
        measured_run={
            "output_dir": str(run_dir.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "global_step": 1,
            "plan_id": plan["plan_id"],
            "candidate_id": candidate["candidate_id"],
            "distribution": candidate["distribution"],
            "world_size": candidate["world_size"],
            "per_rank_cuda_peaks": per_rank,
        },
    )
    report["bindings"].update(
        hardware="selected-hardware-binding",
        preflight_metrics=sha256_file(preflight_path),
        pilot_metrics=sha256_file(pilot_path),
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report


class ValidationAttestationTests(unittest.TestCase):
    def test_host_preflight_reader_requires_trainable_census(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root), bundle)
            plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
            candidate = plan["recommended"]
            metrics_path = bundle / "preflight-metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "schema_version": "aptus.preflight-metrics.v1",
                        "candidate_id": candidate["candidate_id"],
                        "method": candidate["method"],
                        "precision": candidate["precision"],
                        "quantization": candidate.get("quantization"),
                        "distribution": candidate["distribution"],
                        "world_size": candidate["world_size"],
                        "measured_peak_cuda_bytes": 1,
                        "scope": "synthetic-method-preflight-not-model-data-pilot",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "census"):
                _read_preflight_metrics(metrics_path, plan)

    def test_lower_recheck_preserves_stronger_same_bundle_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root)
            bundle = root / "bundle"
            generate_bundle(plan, bundle)
            report_path = bundle / "validation-report.json"
            stronger = json.loads(report_path.read_text(encoding="utf-8"))
            stronger.update(
                state="dependency-pass",
                runtime_evidence=["dependency validation completed"],
                validation_level="dependency",
                validator_version="aptus-portable-validator-v2",
            )
            report_path.write_text(json.dumps(stronger), encoding="utf-8")

            host_result = validate_bundle(bundle, level="static", run=False)
            portable = subprocess.run(
                [sys.executable, str(bundle / "validate.py"), "--level", "static"],
                cwd=bundle,
                check=False,
                capture_output=True,
                text=True,
            )
            persisted = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(host_result.state, ValidationState.DEPENDENCY_PASS)
        self.assertEqual(portable.returncode, 0, portable.stderr)
        self.assertEqual(persisted["state"], "dependency-pass")
        self.assertIn("Preserved stronger", portable.stdout)

    def test_host_static_recheck_preserves_current_measured_run_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root), bundle)
            expected = install_measured_run_attestation(bundle)
            with patch(
                "aptus.validation._actual_hardware_binding",
                return_value="selected-hardware-binding",
            ):
                result = validate_bundle(bundle, level="static", run=False)
            persisted = json.loads(
                (bundle / "validation-report.json").read_text(encoding="utf-8")
            )
        self.assertEqual(result.state, ValidationState.MEASURED_RUN_PASS)
        self.assertEqual(persisted["state"], "measured-run-pass")
        self.assertEqual(persisted["final_export"], expected["final_export"])
        self.assertEqual(persisted["measured_run"], expected["measured_run"])
        self.assertEqual(persisted["latest_recheck"]["state"], "static-pass")

    def test_generated_dependency_recheck_preserves_measured_run_but_tamper_downgrades(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root), bundle)
            expected = install_measured_run_attestation(bundle)
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "aptus_generated_measured_run_preservation", bundle / "validate.py"
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            previous = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            sys.path.insert(0, str(bundle))
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.remove(str(bundle))
                sys.dont_write_bytecode = previous

            with (
                patch.object(
                    module,
                    "environment_binding",
                    return_value=expected["bindings"]["environment"],
                ),
                patch.object(
                    module,
                    "actual_hardware_binding",
                    return_value="selected-hardware-binding",
                ),
            ):
                module._write_attestation("dependency")
            report_path = bundle / "validation-report.json"
            preserved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(preserved["state"], "measured-run-pass")
            self.assertEqual(preserved["final_export"], expected["final_export"])
            self.assertEqual(preserved["latest_recheck"]["state"], "dependency-pass")

            preserved["final_export"]["manifest_sha256"] = "0" * 64
            report_path.write_text(json.dumps(preserved), encoding="utf-8")
            with patch(
                "aptus.validation._actual_hardware_binding",
                return_value="selected-hardware-binding",
            ):
                result = validate_bundle(bundle, level="static", run=False)
            downgraded = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(result.state, ValidationState.STATIC_PASS)
        self.assertEqual(downgraded["state"], "static-pass")
        self.assertIsNone(downgraded["final_export"])


if __name__ == "__main__":
    unittest.main()
