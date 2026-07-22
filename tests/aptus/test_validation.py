import json
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aptus.domain import Backend, ValidationReport, ValidationState
from aptus.generation import generate_bundle
from aptus.plan_contract import sha256_file
from aptus.plan_contract import bundle_fingerprint
from aptus.planning import plan_training
from aptus.profiling import build_hardware_spec
from aptus.validation import (
    _completed_run_evidence_is_current,
    _read_mlx_runtime_metrics,
    _read_preflight_metrics,
    validate_bundle,
)

from tests.aptus.helpers import make_plan


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_generated_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def install_mlx_completed_run(
    root: Path,
    *,
    action: str = "bounded-smoke",
    bundle: Path | None = None,
    identifier: str = "a",
) -> tuple[Path, dict, Path, dict]:
    if bundle is None:
        base = make_plan(root)
        hardware = build_hardware_spec(
            backend=Backend.MPS,
            gpu_count=1,
            vram_gib=64,
            supports_bf16=False,
            supports_4bit=False,
            host_ram_gib=64,
            host_ram_free_gib=48,
            reserve_gib=8,
            disk_free_gib=500,
        )
        mlx_plan = plan_training(
            model=base.model,
            dataset=base.dataset,
            hardware=hardware,
            target=base.target,
        )
        bundle = root / "mlx-bundle"
        generate_bundle(mlx_plan, bundle)
    plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
    candidate = plan["recommended"]
    run_parent = bundle / ("runs" if action == "full" else "pilot-output")
    prefix = "run" if action == "full" else action
    run_root = run_parent / f"{prefix}_{identifier * 32}"
    adapter_dir = run_root / ("final" if action == "full" else "adapters")
    adapter_dir.mkdir(parents=True)
    marker = {
        "schema_version": "aptus.mlx-run-output.v1",
        "run_id": run_root.name,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "created_at": "2026-07-22T00:00:00+00:00",
    }
    marker_path = run_root / ".aptus-run.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"lora_parameters": {"rank": candidate["rank"]}}),
        encoding="utf-8",
    )
    (adapter_dir / "adapters.safetensors").write_bytes(b"mlx-adapter-fixture")
    adapter_manifest = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(adapter_dir.iterdir())
    ]
    split = json.loads(
        (bundle / "data" / "mlx" / "split-contract.json").read_text(encoding="utf-8")
    )
    accumulation = candidate["gradient_accumulation_steps"]
    train_count = split["splits"]["train"]["compiled_row_count"]
    if action == "pilot":
        iterations = 2 * accumulation
    elif action == "full":
        batches = train_count // candidate["micro_batch_size"]
        epoch_iterations = batches * plan["target"]["max_epochs"]
        iterations = (
            (epoch_iterations + accumulation - 1) // accumulation
        ) * accumulation
    else:
        iterations = accumulation
    layers = plan["model"]["layers"]
    targets = candidate["target_modules"]
    binding = {
        "schema_version": "aptus.mlx-trainable-target-binding.v1",
        "planned_target_modules": targets,
        "resolved_layer_keys": [f"resolved.{target}" for target in targets],
        "transformer_layer_count": layers,
        "expected_adapter_target_instance_count": layers * len(targets),
        "adapter_target_instance_count": layers * len(targets),
        "trainable_tensor_count": layers * len(targets) * 2,
        "target_instance_counts": {target: layers for target in targets},
    }
    binding["descriptor_sha256"] = _json_digest(binding)
    reserve = max(plan["hardware"]["reserve_per_device_bytes"], 8 * 1024**3)
    point = candidate["memory"]["point_estimate_bytes"]
    upper = candidate["memory"]["upper_estimate_bytes"]
    admission = {
        "schema_version": "aptus.mlx-unified-memory-admission.v1",
        "available_unified_memory_bytes": max(point, upper) + reserve + 1,
        "point_estimate_bytes": point,
        "upper_estimate_bytes": upper,
        "reserve_bytes": reserve,
        "required_available_bytes": max(point, upper) + reserve,
    }
    scope = {
        "bounded-smoke": "bounded-compiler-smoke-not-pilot-evidence",
        "pilot": "uninterrupted-pilot",
        "full": "uninterrupted-full-train",
    }[action]
    training_metrics = {
        "schema_version": "aptus.runtime-metrics.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "method": candidate["method"],
        "training_runtime": "mlx-lm",
        "compute_backend": "mps",
        "compiler_id": candidate["runtime_contract"]["compiler_id"],
        "scope": scope,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "micro_iterations": iterations,
        "global_step": iterations,
        "gradient_accumulation_steps": accumulation,
        "optimizer_update_opportunities": iterations // accumulation,
        "completed_optimizer_updates": iterations // accumulation,
        "train_examples": train_count,
        "validation_examples": split["splits"]["valid"]["compiled_row_count"],
        "source_train_examples": split["splits"]["train"]["source_row_count"],
        "source_validation_examples": split["splits"]["valid"]["source_row_count"],
        "max_epochs": plan["target"]["max_epochs"],
        "distribution": "single",
        "actual_world_size": 1,
        "measured_peak_bytes": 4096,
        "active_memory_bytes": 2048,
        "cache_memory_bytes": 1024,
        "memory_metric_backend": "mlx",
        "model_load_binding": {
            "schema_version": "aptus.mlx-model-load-binding.v1",
            "model_id": plan["model"]["model_id"],
            "model_revision": plan["model"]["revision"],
            "resolved_local_snapshot": True,
            "trust_remote_code": False,
        },
        "unified_memory_admission": admission,
        "finite_train_loss": True,
        "train_loss_observations": [1.25, 1.0],
        "finite_validation_loss": True,
        "validation_loss_observations": [1.1],
        "optimizer_update_observed": True,
        "trainable_target_binding": binding,
        "adapter_delta_l1": 0.5,
        "changed_adapter_tensor_count": 2,
        "adapter_path": adapter_dir.relative_to(bundle).as_posix(),
        "adapter_manifest": adapter_manifest,
        "completed_at": "2026-07-22T00:00:00+00:00",
    }
    training_path = run_root / "training-metrics.json"
    training_path.write_text(json.dumps(training_metrics), encoding="utf-8")
    reload_evidence = None
    reload_path = run_root / "reload-evidence.json"
    if action in {"pilot", "full"}:
        reload_evidence = {
            "schema_version": "aptus.mlx-reload-evidence.v1",
            "plan_id": plan["plan_id"],
            "candidate_id": candidate["candidate_id"],
            "model_revision": plan["model"]["revision"],
            "dataset_sha256": plan["dataset"]["source_sha256"],
            "method": candidate["method"],
            "training_runtime": "mlx-lm",
            "compute_backend": "mps",
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
            "fresh_process_observed": True,
            "parent_pid": 100,
            "verifier_pid": 101,
            "adapter_manifest_sha256": _json_digest(adapter_manifest),
            "generation_max_tokens": 4,
            "generation_tokens": 2,
            "generation_text_sha256": "a" * 64,
            "measured_peak_bytes": 2048,
            "unified_memory_admission": admission,
            "verified_at": "2026-07-22T00:00:01+00:00",
        }
        reload_path.write_text(json.dumps(reload_evidence), encoding="utf-8")
    proof_paths = [
        marker_path,
        training_path,
        adapter_dir / "adapter_config.json",
        adapter_dir / "adapters.safetensors",
    ]
    if reload_evidence is not None:
        proof_paths.append(reload_path)
    artifact_files = sorted(
        (
            {
                "path": path.relative_to(run_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in proof_paths
        ),
        key=lambda item: item["path"],
    )
    artifact_manifest = {
        "schema_version": "aptus.mlx-artifact-manifest.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "files": artifact_files,
        "total_bytes": sum(item["size_bytes"] for item in artifact_files),
    }
    artifact_path = run_root / "artifact-manifest.json"
    artifact_path.write_text(json.dumps(artifact_manifest), encoding="utf-8")
    final_export = None
    if action == "full":
        final_export = {
            "schema_version": "aptus.mlx-final-export.v1",
            "verification_level": "immutable-adapter-file-tree",
            "plan_id": plan["plan_id"],
            "candidate_id": candidate["candidate_id"],
            "model_revision": plan["model"]["revision"],
            "dataset_sha256": plan["dataset"]["source_sha256"],
            "method": candidate["method"],
            "training_runtime": "mlx-lm",
            "compute_backend": "mps",
            "distribution": "single",
            "world_size": 1,
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
            "files": adapter_manifest,
            "total_bytes": sum(item["size_bytes"] for item in adapter_manifest),
            "artifact_manifest_sha256": sha256_file(artifact_path),
            "reload_evidence_sha256": sha256_file(reload_path),
        }
        (run_root / "final-export.json").write_text(
            json.dumps(final_export), encoding="utf-8"
        )
    metrics = {
        **training_metrics,
        "run_id": run_root.name,
        "output_dir": str(run_root.resolve()),
        "run_marker_sha256": sha256_file(marker_path),
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_sha256": sha256_file(artifact_path),
        "reload_evidence": reload_evidence,
        "reload_evidence_sha256": (
            sha256_file(reload_path) if reload_evidence is not None else None
        ),
        "final_export": final_export,
        "run_completed": True,
    }
    metrics_path = run_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    if action == "bounded-smoke":
        (bundle / "preflight-metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
    elif action == "pilot":
        (bundle / "pilot-output" / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
    return bundle, plan, run_root, metrics


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
    def test_host_preflight_reader_accepts_bound_mlx_memory_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle, plan, _run_root, metrics = install_mlx_completed_run(
                Path(temporary)
            )
            path = bundle / "preflight-metrics.json"
            result = _read_preflight_metrics(path, plan)

        self.assertEqual(result, metrics)

    def test_host_mlx_reader_rejects_forged_semantic_evidence(self) -> None:
        mutations = (
            (
                "unified_memory_admission",
                lambda metrics: metrics["unified_memory_admission"].update(
                    available_unified_memory_bytes=0
                ),
                "unified-memory admission",
            ),
            (
                "completed_optimizer_updates",
                lambda metrics: metrics.update(completed_optimizer_updates=0),
                "optimizer updates",
            ),
            (
                "trainable_target_binding",
                lambda metrics: metrics["trainable_target_binding"].update(
                    adapter_target_instance_count=0
                ),
                "trainable-target binding",
            ),
            (
                "model_load_binding",
                lambda metrics: metrics["model_load_binding"].update(
                    trust_remote_code=True
                ),
                "safe model load",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                bundle, plan, _run_root, _metrics = install_mlx_completed_run(
                    Path(temporary)
                )
                path = bundle / "preflight-metrics.json"
                forged = json.loads(path.read_text(encoding="utf-8"))
                mutate(forged)
                path.write_text(json.dumps(forged), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    _read_preflight_metrics(path, plan)

    def test_host_mlx_reader_rejects_forged_reload_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle, plan, run_root, metrics = install_mlx_completed_run(
                Path(temporary), action="pilot"
            )
            pilot_path = bundle / "pilot-output" / "metrics.json"
            forged = json.loads(json.dumps(metrics))
            forged["reload_evidence"]["fresh_process_observed"] = False
            pilot_path.write_text(json.dumps(forged), encoding="utf-8")
            (run_root / "metrics.json").write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fresh-process"):
                _read_mlx_runtime_metrics(pilot_path, plan, action="pilot")

        with tempfile.TemporaryDirectory() as temporary:
            _bundle, plan, run_root, _metrics = install_mlx_completed_run(
                Path(temporary), action="pilot"
            )
            (run_root / "adapters" / "adapters.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "manifest"):
                _read_mlx_runtime_metrics(
                    run_root / "metrics.json", plan, action="pilot"
                )

    def test_generated_mlx_completed_tree_rejects_extra_and_symlink_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, plan, run_root, _metrics = install_mlx_completed_run(
                root, action="pilot"
            )
            module = _load_generated_module(
                bundle / "validate.py", "aptus_generated_mlx_exact_tree"
            )
            extra = run_root / "unmanifested.bin"
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "unexpected or missing file"):
                module.require_completed_run(plan, run_root, action="pilot")
            extra.unlink()

            training_path = run_root / "training-metrics.json"
            external = root / "training-metrics-external.json"
            external.write_bytes(training_path.read_bytes())
            training_path.unlink()
            training_path.symlink_to(external)
            with self.assertRaisesRegex(RuntimeError, "cannot be a symlink"):
                module.require_completed_run(plan, run_root, action="pilot")

    def test_host_preserves_only_exact_mlx_full_run_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle, plan, run_root, metrics = install_mlx_completed_run(
                Path(temporary), action="full"
            )
            export_path = run_root / "final-export.json"
            export = json.loads(export_path.read_text(encoding="utf-8"))
            final_report = {
                "path": str((run_root / "final").resolve()),
                "manifest_sha256": sha256_file(export_path),
                "total_bytes": export["total_bytes"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "distribution": "single",
                "world_size": 1,
                "training_runtime": "mlx-lm",
                "artifact_manifest_sha256": metrics["artifact_manifest_sha256"],
                "reload_evidence_sha256": metrics["reload_evidence_sha256"],
                "export_contract": export,
            }
            measured_report = {
                "output_dir": str(run_root.resolve()),
                "metrics_sha256": sha256_file(run_root / "metrics.json"),
                "global_step": metrics["global_step"],
                "completed_optimizer_updates": metrics["completed_optimizer_updates"],
                "measured_peak_bytes": metrics["measured_peak_bytes"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "distribution": "single",
                "world_size": 1,
                "training_runtime": "mlx-lm",
                "execution_semantics": "uninterrupted",
                "resume_supported": False,
            }
            previous = ValidationReport(
                state=ValidationState.MEASURED_RUN_PASS,
                findings=(),
                checked_files=(),
                artifact_fingerprint="fixture",
                final_export=final_report,
                measured_run=measured_report,
                measured_run_completed_at="2026-07-22T00:00:02+00:00",
            )
            self.assertTrue(_completed_run_evidence_is_current(previous, bundle, plan))
            (run_root / "final" / "adapters.safetensors").write_bytes(b"tampered")
            self.assertFalse(_completed_run_evidence_is_current(previous, bundle, plan))

    def test_generated_mlx_monotonic_recheck_and_new_pilot_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, plan, _preflight_root, preflight = install_mlx_completed_run(root)
            _bundle, _plan, _pilot_root, pilot = install_mlx_completed_run(
                root, action="pilot", bundle=bundle
            )
            _bundle, _plan, full_root, full = install_mlx_completed_run(
                root, action="full", bundle=bundle
            )
            export_path = full_root / "final-export.json"
            export = full["final_export"]
            final_report = {
                "path": str((full_root / "final").resolve()),
                "manifest_sha256": sha256_file(export_path),
                "total_bytes": export["total_bytes"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "distribution": "single",
                "world_size": 1,
                "training_runtime": "mlx-lm",
                "artifact_manifest_sha256": full["artifact_manifest_sha256"],
                "reload_evidence_sha256": full["reload_evidence_sha256"],
                "export_contract": export,
            }
            measured_report = {
                "output_dir": str(full_root.resolve()),
                "metrics_sha256": sha256_file(full_root / "metrics.json"),
                "global_step": full["global_step"],
                "completed_optimizer_updates": full["completed_optimizer_updates"],
                "measured_peak_bytes": full["measured_peak_bytes"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "distribution": "single",
                "world_size": 1,
                "training_runtime": "mlx-lm",
                "execution_semantics": "uninterrupted",
                "resume_supported": False,
            }
            report = {
                "state": "measured-run-pass",
                "findings": [],
                "checked_files": [],
                "artifact_fingerprint": bundle_fingerprint(bundle),
                "runtime_evidence": [],
                "validation_level": "measured-run",
                "bindings": {
                    "bundle": bundle_fingerprint(bundle),
                    "dataset": plan["dataset"]["source_sha256"],
                    "plan_id": plan["plan_id"],
                    "candidate_id": plan["recommended"]["candidate_id"],
                    "model_revision": plan["model"]["revision"],
                    "preflight_metrics": sha256_file(bundle / "preflight-metrics.json"),
                    "pilot_metrics": sha256_file(
                        bundle / "pilot-output" / "metrics.json"
                    ),
                },
                "preflight_metrics": preflight,
                "pilot_metrics": pilot,
                "final_export": final_report,
                "measured_run": measured_report,
                "measured_run_completed_at": "2026-07-22T00:00:02+00:00",
                "latest_recheck": None,
            }
            report_path = bundle / "validation-report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(bundle / "validate.py"), "--level", "static"],
                cwd=bundle,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            preserved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(preserved["state"], "measured-run-pass")
            self.assertEqual(preserved["latest_recheck"]["state"], "static-pass")

            _bundle, _plan, _new_pilot_root, new_pilot = install_mlx_completed_run(
                root,
                action="pilot",
                bundle=bundle,
                identifier="b",
            )
            module = _load_generated_module(
                bundle / "validate.py", "aptus_generated_mlx_monotonic"
            )
            module.promote(
                plan,
                "pilot-pass",
                preflight_metrics=preflight,
                pilot_metrics=new_pilot,
            )
            refreshed = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(refreshed["state"], "pilot-pass")
            self.assertEqual(refreshed["pilot_metrics"], new_pilot)
            self.assertEqual(
                refreshed["bindings"]["pilot_metrics"],
                sha256_file(bundle / "pilot-output" / "metrics.json"),
            )
            self.assertIsNone(refreshed["measured_run"])

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
