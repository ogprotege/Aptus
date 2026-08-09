import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aptus.execution import (
    JobPrerequisiteError,
    JobService,
    JobSubmissionFailure,
    _actual_runtime_snapshot,
    _environment_binding,
    _json_hash,
    _promote_mlx_train_attestation,
    _promote_train_attestation,
    _read_json_object,
    _read_json_object_bytes,
    _require_current_bundle_model_policy,
    _verify_mlx_completed_run,
    _verify_mlx_pilot_attestation,
    _verify_mlx_runtime_metrics,
    _verify_mlx_train_artifacts,
    _verify_pilot_artifacts,
    _verify_safetensors_structure,
)
from aptus.domain import to_primitive
from aptus.generation import generate_bundle
from aptus.model_compatibility import (
    current_model_policy_snapshot,
    current_model_policy_snapshot_bytes,
    current_model_policy_snapshot_sha256,
)
from aptus.plan_contract import (
    StaleModelPolicyError,
    _current_model_policy_decision,
    expected_model_architecture_contract,
    sha256_file,
)

from tests.aptus.helpers import make_plan


def write_validation_state(bundle: Path, state: str) -> None:
    (bundle / "validation-report.json").write_text(
        json.dumps({"schema_version": "aptus.validation.v2", "state": state}),
        encoding="utf-8",
    )


def changed_model_policy_snapshot() -> dict:
    snapshot = copy.deepcopy(current_model_policy_snapshot())
    snapshot["dense_families"] = sorted(
        [*snapshot["dense_families"], "future-dense-family"]
    )
    return snapshot


def fake_bundle(
    root: Path, *, validation_state: str | None = "model-data-pass"
) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    policy_plan = to_primitive(make_plan(root, gpu_count=1))
    plan = {
        "schema_version": policy_plan["schema_version"],
        "plan_id": "plan_" + "a" * 20,
        "model_policy_snapshot_sha256": current_model_policy_snapshot_sha256(),
        "model": policy_plan["model"],
        "model_policy_decision": policy_plan["model_policy_decision"],
        "dataset": {"source_sha256": "c" * 64},
        "hardware": {"reserve_per_device_bytes": 0},
        "recommended": {
            "candidate_id": "cand_test",
            "method": "lora",
            "distribution": "single",
            "world_size": 1,
            "required_host_ram_bytes": 1,
            "checkpoint_retention_bytes": 1,
        },
    }
    (bundle / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (bundle / "validate.py").write_text(
        'print("validation job passed")\n', encoding="utf-8"
    )
    (bundle / "train.py").write_text('print("training job passed")\n', encoding="utf-8")
    (bundle / "requirements.txt").write_text("", encoding="utf-8")
    snapshot_path = bundle / "policy" / "model-policy-snapshot.v1.json"
    snapshot_path.parent.mkdir()
    snapshot_path.write_bytes(current_model_policy_snapshot_bytes())
    (bundle / "config").mkdir()
    (bundle / "config" / "accelerate.yaml").write_text(
        "distributed_type: NO\n", encoding="utf-8"
    )
    paths = [item for item in bundle.rglob("*") if item.is_file()]
    manifest = {
        "schema_version": "aptus.bundle.v3",
        "plan_sha256": sha256_file(bundle / "plan.json"),
        "policy_snapshot_path": "policy/model-policy-snapshot.v1.json",
        "policy_snapshot_sha256": current_model_policy_snapshot_sha256(),
        "files": [
            {
                "path": path.relative_to(bundle).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(paths)
        ],
    }
    (bundle / "bundle-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if validation_state is not None:
        write_validation_state(bundle, validation_state)
    return bundle


def wait_for(service: JobService, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    record = service.get(job_id)
    while (
        record["state"] in {"queued", "running", "cancelling"}
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
        record = service.get(job_id)
    return record


def make_slow(bundle: Path) -> None:
    (bundle / "validate.py").write_text(
        "import time\ntime.sleep(30)\n", encoding="utf-8"
    )
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    for item in manifest["files"]:
        if item["path"] == "validate.py":
            item["sha256"] = sha256_file(bundle / "validate.py")
            item["size_bytes"] = (bundle / "validate.py").stat().st_size
    (bundle / "bundle-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def mlx_model_load_binding(plan: dict) -> dict:
    model = plan["model"]
    total = model["parameters"]
    active = model.get("active_parameters", total)
    census = {
        "schema_version": "aptus.mlx-model-parameter-census.v1",
        "census_method": "mlx-lm.get_total_parameters.v1",
        "declared_total_parameters": total,
        "observed_total_parameters": total,
        "total_parameter_delta": 0,
        "total_parameter_tolerance": max(1_000_000, round(total * 0.02)),
        "declared_active_parameters": active,
        "observed_active_parameters": active,
        "sparse_layer_count": 0,
        "routed_expert_parameters": 0,
        "active_routed_expert_parameters": 0,
        "inactive_expert_parameters": 0,
    }
    census["descriptor_sha256"] = _json_hash(census)
    expected_weight_bytes = round(total * 2.0)
    observed_safetensors_bytes = expected_weight_bytes + 4096
    packed = {
        "schema_version": "aptus.mlx-packed-checkpoint.v1",
        "observed_safetensors_bytes": observed_safetensors_bytes,
        "observed_logical_parameters": total,
        "expected_weight_bytes": expected_weight_bytes,
        "expected_quantization_metadata_bytes": 0,
        "expected_packed_tensor_bytes": expected_weight_bytes,
        "container_overhead_bytes": 4096,
        "container_overhead_limit_bytes": max(
            1024**2, round(expected_weight_bytes * 0.0001)
        ),
    }
    packed["descriptor_sha256"] = _json_hash(packed)
    binding = {
        "schema_version": "aptus.mlx-model-load-binding.v3",
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "resolved_local_snapshot": True,
        "trust_remote_code": False,
        "architecture_contract": expected_model_architecture_contract(model),
        "parameter_census": census,
        "packed_checkpoint_binding": packed,
    }
    binding["descriptor_sha256"] = _json_hash(binding)
    return binding


def fake_mlx_plan() -> dict:
    reserve = 8 * 1024**3
    plan = {
        "schema_version": "aptus.training-plan.v6",
        "plan_id": "plan_" + "a" * 20,
        "model_policy_snapshot_sha256": current_model_policy_snapshot_sha256(),
        "model": {
            "model_id": "example/model",
            "revision": "b" * 40,
            "family": "llama",
            "parameters": 1_000_000_000,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "layers": 2,
            "context_length": 4096,
            "architecture": "causal-lm",
            "model_type": None,
            "quantization_bits": None,
            "quantization_layout": None,
            "moe": None,
            "sparse_layer_count": 0,
            "active_parameters": 1_000_000_000,
        },
        "dataset": {"source_sha256": "c" * 64},
        "hardware": {"reserve_per_device_bytes": reserve},
        "target": {"max_epochs": 1},
        "recommended": {
            "candidate_id": "cand_" + "d" * 24,
            "method": "lora",
            "distribution": "single",
            "world_size": 1,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "rank": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "memory": {
                "base_weights_bytes": 2_000_000_000,
                "quantization_metadata_bytes": 0,
                "point_estimate_bytes": 1024,
                "upper_estimate_bytes": 2048,
            },
            "final_export_bytes": 4096,
            "required_disk_bytes": 8192,
            "runtime_contract": {
                "training_runtime": "mlx-lm",
                "compute_backend": "mps",
                "compiler_id": "mlx-lm.lora.v1",
            },
        },
    }
    plan["model_policy_decision"] = _current_model_policy_decision(plan["model"])
    return plan


def write_fake_mlx_split_contract(bundle: Path) -> None:
    split_root = bundle / "data" / "mlx"
    split_root.mkdir(parents=True, exist_ok=True)
    write_json(
        split_root / "split-contract.json",
        {
            "schema_version": "aptus.mlx-split.v1",
            "micro_batch_size": 1,
            "padding_policy": "repeat-within-disjoint-split-to-complete-final-batch",
            "splits": {
                "train": {"source_row_count": 2, "compiled_row_count": 2},
                "valid": {"source_row_count": 1, "compiled_row_count": 1},
            },
        },
    )


def fake_mlx_metrics(plan: dict, *, action: str) -> dict:
    candidate = plan["recommended"]
    binding = {
        "schema_version": "aptus.mlx-trainable-target-binding.v1",
        "planned_target_modules": candidate["target_modules"],
        "resolved_layer_keys": ["self_attn.q_proj", "self_attn.v_proj"],
        "transformer_layer_count": 2,
        "expected_adapter_target_instance_count": 4,
        "adapter_target_instance_count": 4,
        "trainable_tensor_count": 8,
        "target_instance_counts": {"q_proj": 2, "v_proj": 2},
    }
    binding["descriptor_sha256"] = _json_hash(binding)
    reserve = 8 * 1024**3
    memory = candidate["memory"]
    planned_resident = (
        memory["base_weights_bytes"] + memory["quantization_metadata_bytes"]
    )
    observed = mlx_model_load_binding(plan)["packed_checkpoint_binding"][
        "observed_safetensors_bytes"
    ]
    adjustment = max(0, observed - planned_resident)
    adjusted_point = memory["point_estimate_bytes"] + adjustment
    adjusted_upper = memory["upper_estimate_bytes"] + adjustment
    required = max(adjusted_point, adjusted_upper) + reserve
    admission = {
        "schema_version": "aptus.mlx-unified-memory-admission.v2",
        "available_unified_memory_bytes": required + 1,
        "planned_resident_bytes": planned_resident,
        "observed_safetensors_bytes": observed,
        "resident_adjustment_bytes": adjustment,
        "adjusted_point_estimate_bytes": adjusted_point,
        "adjusted_upper_estimate_bytes": adjusted_upper,
        "reserve_bytes": reserve,
        "required_available_bytes": required,
    }
    updates = 2
    return {
        "schema_version": "aptus.runtime-metrics.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "method": "lora",
        "training_runtime": "mlx-lm",
        "compute_backend": "mps",
        "compiler_id": "mlx-lm.lora.v1",
        "model_load_binding": mlx_model_load_binding(plan),
        "scope": f"uninterrupted-{action}"
        if action == "pilot"
        else "uninterrupted-full-train",
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "micro_iterations": updates,
        "global_step": updates,
        "gradient_accumulation_steps": 1,
        "optimizer_update_opportunities": updates,
        "completed_optimizer_updates": updates,
        "train_examples": 2,
        "validation_examples": 1,
        "source_train_examples": 2,
        "source_validation_examples": 1,
        "max_epochs": 1,
        "distribution": "single",
        "actual_world_size": 1,
        "measured_peak_bytes": 4096,
        "active_memory_bytes": 2048,
        "cache_memory_bytes": 1024,
        "memory_metric_backend": "mlx",
        "unified_memory_admission": admission,
        "finite_train_loss": True,
        "train_loss_observations": [1.0, 0.5],
        "finite_validation_loss": True,
        "validation_loss_observations": [0.75],
        "optimizer_update_observed": True,
        "trainable_target_binding": binding,
        "adapter_delta_l1": 1.0,
        "changed_adapter_tensor_count": 2,
        "adapter_path": f"pilot-output/{action}_test/adapters",
        "adapter_manifest": [
            {
                "path": "adapters.safetensors",
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
        ],
        "completed_at": "2026-07-22T00:00:00+00:00",
    }


def create_mlx_completed_run(
    bundle: Path, plan: dict, *, action: str
) -> tuple[Path, dict]:
    write_fake_mlx_split_contract(bundle)
    root = (
        bundle / "pilot-output" / "pilot_test"
        if action == "pilot"
        else bundle / "runs" / "run_test"
    )
    adapter_name = "adapters" if action == "pilot" else "final"
    adapter_dir = root / adapter_name
    adapter_dir.mkdir(parents=True)
    adapter_config = adapter_dir / "adapter_config.json"
    write_json(
        adapter_config,
        {
            "lora_parameters": {
                "keys": ["self_attn.q_proj", "self_attn.v_proj"],
                "rank": 8,
                "scale": 2.0,
            }
        },
    )
    adapter_weights = adapter_dir / "adapters.safetensors"
    adapter_weights.write_bytes(b"mlx-adapter")
    adapter_manifest = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (adapter_config, adapter_weights)
    ]

    training_metrics = fake_mlx_metrics(plan, action=action)
    training_metrics["adapter_path"] = adapter_dir.relative_to(bundle).as_posix()
    training_metrics["adapter_manifest"] = adapter_manifest
    training_metrics_path = root / "training-metrics.json"
    write_json(training_metrics_path, training_metrics)

    marker = {
        "schema_version": "aptus.mlx-run-output.v1",
        "run_id": root.name,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "created_at": "2026-07-22T00:00:00+00:00",
    }
    marker_path = root / ".aptus-run.json"
    write_json(marker_path, marker)

    reload_evidence = {
        "schema_version": "aptus.mlx-reload-evidence.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "method": "lora",
        "training_runtime": "mlx-lm",
        "compute_backend": "mps",
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "fresh_process_observed": True,
        "parent_pid": 100,
        "verifier_pid": 101,
        "adapter_manifest_sha256": _json_hash(adapter_manifest),
        "generation_max_tokens": 4,
        "generation_tokens": 2,
        "generation_text_sha256": "e" * 64,
        "measured_peak_bytes": 2048,
        "unified_memory_admission": training_metrics["unified_memory_admission"],
        "verified_at": "2026-07-22T00:00:01+00:00",
    }
    reload_path = root / "reload-evidence.json"
    write_json(reload_path, reload_evidence)

    manifest_paths = (
        marker_path,
        training_metrics_path,
        adapter_config,
        adapter_weights,
        reload_path,
    )
    manifest_entries = sorted(
        (
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifest_paths
        ),
        key=lambda entry: entry["path"],
    )
    artifact_manifest = {
        "schema_version": "aptus.mlx-artifact-manifest.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": plan["recommended"]["candidate_id"],
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "files": manifest_entries,
        "total_bytes": sum(entry["size_bytes"] for entry in manifest_entries),
    }
    artifact_manifest_path = root / "artifact-manifest.json"
    write_json(artifact_manifest_path, artifact_manifest)

    final_export = None
    if action == "full":
        final_export = {
            "schema_version": "aptus.mlx-final-export.v1",
            "verification_level": "immutable-adapter-file-tree",
            "plan_id": plan["plan_id"],
            "candidate_id": plan["recommended"]["candidate_id"],
            "model_revision": plan["model"]["revision"],
            "dataset_sha256": plan["dataset"]["source_sha256"],
            "method": "lora",
            "training_runtime": "mlx-lm",
            "compute_backend": "mps",
            "distribution": "single",
            "world_size": 1,
            "execution_semantics": "uninterrupted",
            "resume_supported": False,
            "files": adapter_manifest,
            "total_bytes": sum(entry["size_bytes"] for entry in adapter_manifest),
            "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
            "reload_evidence_sha256": sha256_file(reload_path),
        }
        write_json(root / "final-export.json", final_export)

    completed_metrics = {
        **training_metrics,
        "run_id": root.name,
        "output_dir": str(root.resolve()),
        "run_marker_sha256": sha256_file(marker_path),
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "reload_evidence": reload_evidence,
        "reload_evidence_sha256": sha256_file(reload_path),
        "final_export": final_export,
        "run_completed": True,
    }
    write_json(root / "metrics.json", completed_metrics)
    return root, completed_metrics


class ExecutionJobTests(unittest.TestCase):
    def test_json_object_loaders_reject_resource_hostile_inputs_cleanly(self) -> None:
        documents = (
            ("oversized-integer", '{"value":' + "9" * 5_000 + "}"),
            (
                "excessive-nesting",
                '{"value":' + "[" * 10_000 + "0" + "]" * 10_000 + "}",
            ),
        )
        loaders = (_read_json_object, _read_json_object_bytes)
        for name, contents in documents:
            for loader in loaders:
                with (
                    self.subTest(name=name, loader=loader.__name__),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    path = Path(temporary) / "document.json"
                    path.write_text(contents, encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, "is unreadable"):
                        loader(path, "Test document")

    def test_cuda_jobs_use_the_configured_external_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            plan_path = bundle / "plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["recommended"]["runtime_contract"] = {
                "schema_version": "aptus.runtime-contract.v1",
                "compute_backend": "cuda",
                "training_runtime": "transformers-peft-cuda",
                "compiler_id": "transformers.peft-lora.v2",
                "estimator_id": "aptus-memory-v2",
                "evidence_requirement": "pilot-required",
                "export_kind": "peft-adapter-safetensors",
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            service = JobService(
                root / "jobs",
                runtime_environment={
                    "APTUS_CUDA_PYTHON": "/managed/cuda-python",
                },
            )
            with patch(
                "aptus.execution.resolve_runtime_interpreter",
                return_value=types.SimpleNamespace(path="/managed/cuda-python"),
            ) as resolve:
                command = service._command(bundle, "dependency", resume_from=None)

        self.assertEqual(command[0], "/managed/cuda-python")
        resolve.assert_called_once_with(
            "transformers-peft-cuda",
            environment={"APTUS_CUDA_PYTHON": "/managed/cuda-python"},
        )

    def test_mlx_jobs_use_the_resolved_external_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            plan_path = bundle / "plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["recommended"]["runtime_contract"] = {
                "schema_version": "aptus.runtime-contract.v1",
                "compute_backend": "mps",
                "training_runtime": "mlx-lm",
                "compiler_id": "mlx-lm.lora.v1",
                "estimator_id": "aptus-memory-mlx-v2",
                "evidence_requirement": "pilot-required",
                "export_kind": "mlx-lm-adapter",
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            service = JobService(root / "jobs")
            with patch(
                "aptus.execution.resolve_runtime_interpreter",
                return_value=types.SimpleNamespace(path="/managed/mlx-python"),
            ):
                command = service._command(bundle, "dependency", resume_from=None)

        self.assertEqual(command[0], "/managed/mlx-python")
        self.assertEqual(command[-2:], ["--level", "dependency"])

    def test_mlx_full_job_dispatches_through_the_uninterrupted_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            plan_path = bundle / "plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["recommended"]["runtime_contract"] = {
                "schema_version": "aptus.runtime-contract.v1",
                "compute_backend": "mps",
                "training_runtime": "mlx-lm",
                "compiler_id": "mlx-lm.lora.v1",
                "estimator_id": "aptus-memory-mlx-v2",
                "evidence_requirement": "pilot-required",
                "export_kind": "mlx-lm-adapter",
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            (bundle / "run.py").write_text(
                'parser.add_argument("--defer-parent-promotion")\n',
                encoding="utf-8",
            )
            service = JobService(root / "jobs")
            with patch(
                "aptus.execution.resolve_runtime_interpreter",
                return_value=types.SimpleNamespace(path="/managed/mlx-python"),
            ):
                command = service._command(
                    bundle,
                    "train",
                    resume_from=None,
                    run_id="run_test",
                )
                (bundle / "run.py").write_text(
                    "# Legacy MLX runner without managed deferral.\n",
                    encoding="utf-8",
                )
                legacy_command = service._command(
                    bundle,
                    "train",
                    resume_from=None,
                    run_id="run_legacy",
                )

        self.assertEqual(command[0], "/managed/mlx-python")
        self.assertEqual(command[1], "run.py")
        self.assertEqual(
            command[2:5],
            [
                "--confirm-full-train",
                "--defer-parent-promotion",
                "--output-dir",
            ],
        )
        self.assertEqual(command[5], str(Path("runs") / "run_test"))
        self.assertEqual(
            legacy_command[2:],
            [
                "--confirm-full-train",
                "--output-dir",
                str(Path("runs") / "run_legacy"),
            ],
        )

    def test_mlx_runtime_metrics_reject_tampered_and_stale_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            write_fake_mlx_split_contract(bundle)
            plan = fake_mlx_plan()
            metrics = fake_mlx_metrics(plan, action="pilot")
            metrics["adapter_manifest"] = [
                {
                    "path": "adapters.safetensors",
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                }
            ]
            self.assertIs(
                _verify_mlx_runtime_metrics(bundle, plan, metrics, action="pilot"),
                metrics,
            )
            for name, value in (
                ("compiler_id", "invented.compiler"),
                ("candidate_id", "cand_stale"),
                ("execution_semantics", "resumed"),
                ("completed_optimizer_updates", 1),
                ("adapter_delta_l1", 0.0),
                ("measured_peak_bytes", 0),
                (
                    "model_load_binding",
                    {
                        **metrics["model_load_binding"],
                        "model_revision": "f" * 40,
                    },
                ),
            ):
                with self.subTest(name=name):
                    tampered = json.loads(json.dumps(metrics))
                    tampered[name] = value
                    with self.assertRaises(ValueError):
                        _verify_mlx_runtime_metrics(
                            bundle, plan, tampered, action="pilot"
                        )

            forged_packed = json.loads(json.dumps(metrics))
            packed = forged_packed["model_load_binding"]["packed_checkpoint_binding"]
            packed["expected_weight_bytes"] += 1
            packed["expected_packed_tensor_bytes"] += 1
            packed["container_overhead_bytes"] -= 1
            packed["descriptor_sha256"] = _json_hash(
                {
                    key: value
                    for key, value in packed.items()
                    if key != "descriptor_sha256"
                }
            )
            model_binding = forged_packed["model_load_binding"]
            model_binding["descriptor_sha256"] = _json_hash(
                {
                    key: value
                    for key, value in model_binding.items()
                    if key != "descriptor_sha256"
                }
            )
            with self.assertRaisesRegex(ValueError, "safe model load"):
                _verify_mlx_runtime_metrics(bundle, plan, forged_packed, action="pilot")

            forged_adjustment = json.loads(json.dumps(metrics))
            forged_admission = forged_adjustment["unified_memory_admission"]
            forged_admission["resident_adjustment_bytes"] += 1
            forged_admission["adjusted_point_estimate_bytes"] += 1
            forged_admission["adjusted_upper_estimate_bytes"] += 1
            forged_admission["required_available_bytes"] += 1
            with self.assertRaisesRegex(ValueError, "memory contract"):
                _verify_mlx_runtime_metrics(
                    bundle, plan, forged_adjustment, action="pilot"
                )

            mismatched_measurement = json.loads(json.dumps(metrics))
            mismatch_admission = mismatched_measurement["unified_memory_admission"]
            mismatch_admission["observed_safetensors_bytes"] += 1
            mismatch_admission["resident_adjustment_bytes"] += 1
            mismatch_admission["adjusted_point_estimate_bytes"] += 1
            mismatch_admission["adjusted_upper_estimate_bytes"] += 1
            mismatch_admission["required_available_bytes"] += 1
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                _verify_mlx_runtime_metrics(
                    bundle, plan, mismatched_measurement, action="pilot"
                )

            invented = json.loads(json.dumps(metrics))
            invented["free_vram_bytes"] = 1
            with self.assertRaisesRegex(ValueError, "unexpected runtime field"):
                _verify_mlx_runtime_metrics(bundle, plan, invented, action="pilot")

            split_path = bundle / "data" / "mlx" / "split-contract.json"
            split = json.loads(split_path.read_text(encoding="utf-8"))
            split["splits"]["train"]["compiled_row_count"] = 3
            write_json(split_path, split)
            with self.assertRaisesRegex(ValueError, "compiled data split"):
                _verify_mlx_runtime_metrics(bundle, plan, metrics, action="pilot")
            write_fake_mlx_split_contract(bundle)

            shortened = fake_mlx_metrics(plan, action="full")
            shortened["adapter_manifest"] = metrics["adapter_manifest"]
            shortened["micro_iterations"] = 1
            shortened["global_step"] = 1
            shortened["optimizer_update_opportunities"] = 1
            shortened["completed_optimizer_updates"] = 1
            with self.assertRaisesRegex(ValueError, "dataset-derived epoch schedule"):
                _verify_mlx_runtime_metrics(
                    bundle,
                    plan,
                    shortened,
                    action="full",
                )

    def test_mlx_completed_full_run_verifies_runtime_neutral_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            plan = fake_mlx_plan()
            run, metrics = create_mlx_completed_run(bundle, plan, action="full")

            evidence = _verify_mlx_completed_run(
                bundle,
                plan,
                run,
                action="full",
            )

        self.assertEqual(evidence["metrics"], metrics)
        self.assertEqual(evidence["measured_peak_bytes"], 4096)
        self.assertGreater(evidence["adapter_total_bytes"], 0)

    def test_mlx_completed_run_rejects_tampering_and_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            plan = fake_mlx_plan()
            run, _metrics = create_mlx_completed_run(bundle, plan, action="full")
            reload_path = run / "reload-evidence.json"
            reload_evidence = json.loads(reload_path.read_text(encoding="utf-8"))
            reload_evidence["generation_tokens"] = 0
            write_json(reload_path, reload_evidence)
            with self.assertRaisesRegex(ValueError, "reload evidence"):
                _verify_mlx_completed_run(bundle, plan, run, action="full")

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            plan = fake_mlx_plan()
            run, _metrics = create_mlx_completed_run(bundle, plan, action="full")
            (run / "final-export.json").unlink()
            with self.assertRaises(ValueError):
                _verify_mlx_completed_run(bundle, plan, run, action="full")

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            plan = fake_mlx_plan()
            run, _metrics = create_mlx_completed_run(bundle, plan, action="full")
            (run / "unexpected-empty-directory").mkdir()
            with self.assertRaisesRegex(ValueError, "unexpected or partial"):
                _verify_mlx_completed_run(bundle, plan, run, action="full")

    def test_mlx_pilot_attestation_rejects_stale_report_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            plan = fake_mlx_plan()
            pilot_run, metrics = create_mlx_completed_run(bundle, plan, action="pilot")
            pilot_copy = bundle / "pilot-output" / "metrics.json"
            write_json(pilot_copy, metrics)
            (bundle / "bundle-manifest.json").write_text("manifest", encoding="utf-8")
            bindings = {
                "bundle": sha256_file(bundle / "bundle-manifest.json"),
                "dataset": plan["dataset"]["source_sha256"],
                "model_revision": plan["model"]["revision"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "pilot_metrics": sha256_file(pilot_copy),
            }
            report = {"bindings": bindings, "pilot_metrics": metrics}
            evidence = _verify_mlx_pilot_attestation(
                bundle,
                plan,
                report,
                pilot_copy,
            )
            self.assertEqual(evidence["root"], str(pilot_run.resolve()))

            report["bindings"] = {**bindings, "candidate_id": "cand_stale"}
            with self.assertRaisesRegex(ValueError, "stale"):
                _verify_mlx_pilot_attestation(
                    bundle,
                    plan,
                    report,
                    pilot_copy,
                )

    def test_mlx_train_admission_rejects_insufficient_live_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            plan = fake_mlx_plan()
            write_json(bundle / "plan.json", plan)
            write_json(
                bundle / "validation-report.json",
                {"state": "pilot-pass", "bindings": {}},
            )
            pilot_root = bundle / "pilot-output"
            pilot_root.mkdir()
            write_json(pilot_root / "metrics.json", {})
            service = JobService(Path(temporary) / "jobs")
            reserve = 8 * 1024**3
            with (
                patch(
                    "aptus.execution._require_current_bundle_model_policy",
                    return_value=(plan, "a" * 64),
                ),
                patch("aptus.execution.validate_plan_payload", return_value=()),
                patch(
                    "aptus.execution._verify_mlx_pilot_attestation",
                    return_value={
                        "measured_peak_bytes": 4096,
                        "adapter_total_bytes": 1024,
                        "artifact_total_bytes": 2048,
                    },
                ),
                patch(
                    "aptus.execution._current_available_unified_memory_bytes",
                    return_value=reserve + 4095,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "unified memory"):
                    service._require_current_pilot(bundle)

    def test_mlx_parent_promotion_allows_measured_run_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            bindings = {"pilot_metrics": "a" * 64}
            final_export = {"manifest_sha256": "b" * 64}
            measured_run = {"metrics_sha256": "c" * 64}
            write_json(
                bundle / "validation-report.json",
                {
                    "state": "measured-run-pass",
                    "validation_level": "measured-run",
                    "bindings": bindings,
                    "runtime_evidence": [],
                    "final_export": final_export,
                    "measured_run": measured_run,
                },
            )
            evidence = {
                "training_runtime": "mlx-lm",
                "source_report_state": "measured-run-pass",
                "source_bindings": bindings,
                "final_export": final_export,
                "measured_run": measured_run,
                "source_report_sha256": sha256_file(bundle / "validation-report.json"),
            }
            promoted = _promote_mlx_train_attestation(
                {"bundle_dir": str(bundle)}, evidence
            )
            report = json.loads(
                (bundle / "validation-report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(promoted["state"], "measured-run-pass")
        self.assertEqual(report["state"], "measured-run-pass")
        self.assertEqual(report["final_export"], evidence["final_export"])
        self.assertEqual(report["measured_run"], evidence["measured_run"])

    def test_mlx_parent_receipts_child_promoted_verified_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            plan = fake_mlx_plan()
            write_json(bundle / "plan.json", plan)
            write_json(bundle / "bundle-manifest.json", {})

            _pilot_run, pilot_metrics = create_mlx_completed_run(
                bundle, plan, action="pilot"
            )
            pilot_copy = bundle / "pilot-output" / "metrics.json"
            write_json(pilot_copy, pilot_metrics)
            full_run, _full_metrics = create_mlx_completed_run(
                bundle, plan, action="full"
            )
            artifact_fingerprint = sha256_file(bundle / "bundle-manifest.json")
            bindings = {
                "bundle": artifact_fingerprint,
                "dataset": plan["dataset"]["source_sha256"],
                "model_revision": plan["model"]["revision"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "pilot_metrics": sha256_file(pilot_copy),
            }
            report_path = bundle / "validation-report.json"
            write_json(
                report_path,
                {
                    "state": "pilot-pass",
                    "validation_level": "pilot",
                    "bindings": bindings,
                    "pilot_metrics": pilot_metrics,
                },
            )
            record = {
                "id": "job_" + "a" * 32,
                "run_id": full_run.name,
                "bundle_dir": str(bundle),
                "run_output_dir": str(full_run),
                "artifact_fingerprint": artifact_fingerprint,
                "command": [
                    "/managed/mlx-python",
                    "run.py",
                    "--confirm-full-train",
                    "--output-dir",
                    str(Path("runs") / full_run.name),
                ],
            }

            with (
                patch("aptus.execution.validate_bundle_manifest", return_value=()),
                patch("aptus.execution.validate_plan_payload", return_value=()),
            ):
                expected = _verify_mlx_train_artifacts(record)
                child_report = json.loads(report_path.read_text(encoding="utf-8"))
                child_report.update(
                    state="measured-run-pass",
                    validation_level="measured-run",
                    measured_run_completed_at="2026-08-05T12:10:57+00:00",
                    final_export=expected["final_export"],
                    measured_run=expected["measured_run"],
                )
                write_json(report_path, child_report)
                evidence = _verify_mlx_train_artifacts(record)

            self.assertEqual(evidence["source_report_state"], "measured-run-pass")
            self.assertNotIn("parent_promotion", child_report)
            with patch(
                "aptus.execution._require_current_bundle_model_policy",
                return_value=(plan, artifact_fingerprint),
            ):
                record["command"] = [
                    "/managed/mlx-python",
                    "run.py",
                    "--confirm-full-train",
                    "--defer-parent-promotion",
                    "--output-dir",
                    str(Path("runs") / full_run.name),
                ]
                with self.assertRaisesRegex(
                    ValueError, "valid parent-promotion receipt"
                ):
                    _promote_train_attestation(
                        record,
                        evidence,
                        _allow_legacy_mlx_child_completion=True,
                    )

                record["command"].remove("--defer-parent-promotion")
                changed_report = {**child_report, "runtime_evidence": ["changed"]}
                write_json(report_path, changed_report)
                with self.assertRaisesRegex(
                    ValueError, "valid parent-promotion receipt"
                ):
                    _promote_train_attestation(
                        record,
                        evidence,
                        _allow_legacy_mlx_child_completion=True,
                    )

                write_json(report_path, child_report)
                promoted = _promote_train_attestation(
                    record,
                    evidence,
                    _allow_legacy_mlx_child_completion=True,
                )

            promoted_report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(promoted["state"], "measured-run-pass")
        self.assertEqual(promoted_report["parent_promotion"]["job_id"], record["id"])
        self.assertEqual(
            promoted_report["parent_promotion"]["run_id"], record["run_id"]
        )
        self.assertEqual(
            promoted_report["parent_promotion"]["evidence_sha256"],
            _json_hash(evidence),
        )

    def test_mlx_parent_receipts_managed_deferred_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            write_json(bundle / "bundle-manifest.json", {})
            report_path = bundle / "validation-report.json"
            bindings = {"pilot_metrics": "a" * 64}
            run_id = "run_" + "b" * 32
            source_active_run = {
                "output_dir": str((bundle / "runs" / run_id).resolve()),
                "run_id": run_id,
                "plan_id": "plan_test",
                "candidate_id": "cand_test",
            }
            write_json(
                report_path,
                {
                    "state": "execution-approved",
                    "validation_level": "pilot",
                    "bindings": bindings,
                    "active_run": source_active_run,
                    "runtime_evidence": [],
                },
            )
            evidence = {
                "training_runtime": "mlx-lm",
                "source_report_state": "execution-approved",
                "source_bindings": bindings,
                "source_active_run": source_active_run,
                "source_report_sha256": sha256_file(report_path),
                "final_export": {"manifest_sha256": "b" * 64},
                "measured_run": {"metrics_sha256": "c" * 64},
            }
            record = {
                "id": "job_" + "b" * 32,
                "run_id": run_id,
                "bundle_dir": str(bundle),
                "artifact_fingerprint": sha256_file(bundle / "bundle-manifest.json"),
            }
            with patch(
                "aptus.execution._require_current_bundle_model_policy",
                return_value=({}, record["artifact_fingerprint"]),
            ):
                promoted = _promote_train_attestation(record, evidence)
            promoted_report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(promoted["state"], "measured-run-pass")
        self.assertEqual(promoted_report["validation_level"], "measured-run")
        self.assertNotIn("active_run", promoted_report)
        self.assertEqual(
            promoted_report["parent_promotion"],
            {
                "schema_version": "aptus.parent-promotion.v1",
                "job_id": record["id"],
                "run_id": record["run_id"],
                "artifact_fingerprint": record["artifact_fingerprint"],
                "evidence_sha256": _json_hash(evidence),
                "promoted_at": promoted["measured_run_completed_at"],
            },
        )

    def test_mlx_host_accepts_only_the_exact_portable_measured_attestation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            plan = fake_mlx_plan()
            write_json(bundle / "plan.json", plan)
            run = bundle / "runs" / "run_test"
            final_contract = {"schema_version": "aptus.mlx-final-export.v1"}
            completed = {
                "final_export": final_contract,
                "final_export_sha256": "a" * 64,
                "adapter_total_bytes": 128,
                "artifact_manifest_sha256": "b" * 64,
                "reload_evidence_sha256": "c" * 64,
                "metrics_sha256": "d" * 64,
                "metrics": {
                    "global_step": 2,
                    "completed_optimizer_updates": 2,
                    "measured_peak_bytes": 4096,
                },
            }
            final_report = {
                "path": str(run.resolve() / "final"),
                "manifest_sha256": "a" * 64,
                "total_bytes": 128,
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "distribution": "single",
                "world_size": 1,
                "training_runtime": "mlx-lm",
                "artifact_manifest_sha256": "b" * 64,
                "reload_evidence_sha256": "c" * 64,
                "export_contract": final_contract,
            }
            measured_report = {
                "output_dir": str(run.resolve()),
                "metrics_sha256": "d" * 64,
                "global_step": 2,
                "completed_optimizer_updates": 2,
                "measured_peak_bytes": 4096,
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "distribution": "single",
                "world_size": 1,
                "training_runtime": "mlx-lm",
                "execution_semantics": "uninterrupted",
                "resume_supported": False,
            }
            report_path = bundle / "validation-report.json"
            write_json(
                report_path,
                {
                    "state": "measured-run-pass",
                    "bindings": {},
                    "pilot_metrics": {},
                    "final_export": {"stale": True},
                    "measured_run": measured_report,
                },
            )
            record = {
                "bundle_dir": str(bundle),
                "run_output_dir": str(run),
            }
            with (
                patch("aptus.execution.validate_bundle_manifest", return_value=()),
                patch("aptus.execution.validate_plan_payload", return_value=()),
                patch(
                    "aptus.execution._verify_mlx_completed_run",
                    return_value=completed,
                ),
                patch("aptus.execution._verify_mlx_pilot_attestation"),
            ):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    _verify_mlx_train_artifacts(record)

                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["final_export"] = final_report
                write_json(report_path, report)
                evidence = _verify_mlx_train_artifacts(record)

        self.assertEqual(evidence["final_export"], final_report)
        self.assertEqual(evidence["measured_run"], measured_report)

    def test_mlx_parent_promotion_rejects_report_toctou_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            report_path = bundle / "validation-report.json"
            bindings = {"pilot_metrics": "a" * 64}
            write_json(
                report_path,
                {
                    "state": "pilot-pass",
                    "bindings": bindings,
                    "runtime_evidence": [],
                },
            )
            evidence = {
                "training_runtime": "mlx-lm",
                "source_report_state": "pilot-pass",
                "source_bindings": bindings,
                "source_report_sha256": sha256_file(report_path),
                "final_export": {"manifest_sha256": "b" * 64},
                "measured_run": {"metrics_sha256": "c" * 64},
            }
            changed = json.loads(report_path.read_text(encoding="utf-8"))
            changed["runtime_evidence"] = ["changed after verification"]
            write_json(report_path, changed)

            with self.assertRaisesRegex(ValueError, "changed before parent"):
                _promote_mlx_train_attestation({"bundle_dir": str(bundle)}, evidence)

    def test_host_pilot_verifier_requires_matching_phase_censuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = fake_bundle(Path(temporary))
            plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
            pilot_id = "pilot_" + "d" * 32
            pilot_root = bundle / "runs" / pilot_id
            pilot_root.mkdir(parents=True)
            (pilot_root / ".aptus-pilot-run.json").write_text(
                json.dumps(
                    {
                        "schema_version": "aptus.pilot-run.v1",
                        "pilot_run_id": pilot_id,
                        "plan_id": plan["plan_id"],
                        "candidate_id": plan["recommended"]["candidate_id"],
                        "model_revision": plan["model"]["revision"],
                        "dataset_sha256": plan["dataset"]["source_sha256"],
                    }
                ),
                encoding="utf-8",
            )

            def checkpoint_contract(phase: str, checkpoint: str) -> dict:
                root = pilot_root / phase / checkpoint
                root.mkdir(parents=True)
                state = root / "state.bin"
                state.write_bytes(b"x")
                files = [
                    {
                        "path": "state.bin",
                        "size_bytes": 1,
                        "sha256": sha256_file(state),
                    }
                ]
                return {
                    "files": files,
                    "manifest_sha256": _json_hash(files),
                    "total_bytes": 1,
                }

            phase_one_contract = checkpoint_contract("phase-1", "checkpoint-1")
            phase_two_contract = checkpoint_contract("phase-2", "checkpoint-2")
            metrics = {
                "checkpoint_continuation_observed": True,
                "pilot_run_dir": f"runs/{pilot_id}",
                "pilot_run_id": pilot_id,
                "phase_one_checkpoint": phase_one_contract,
                "phase_two_checkpoint": phase_two_contract,
                "measured_checkpoint_bytes": 1,
                "measured_final_export_bytes": 1,
                "phase_one": {"final_export": {"total_bytes": 1}},
                "phase_two_resumed": {"final_export": {"total_bytes": 1}},
            }
            with self.assertRaisesRegex(ValueError, "census"):
                _verify_pilot_artifacts(bundle, metrics)

            census = {
                "schema_version": "aptus.trainable-parameter-census.v1",
                "method": "lora",
                "parameter_scope": "lora-adapter-only",
                "trainable_parameter_count": 8,
                "trainable_tensor_count": 2,
                "frozen_parameter_count": 100,
                "frozen_tensor_count": 1,
                "unexpected_trainable_tensor_count": 0,
                "expected_adapter_target_match_count": 1,
                "adapter_target_instance_count": 1,
                "incomplete_adapter_target_instance_count": 0,
                "all_values_finite": True,
                "descriptor_sha256": "a" * 64,
            }
            metrics["phase_one"]["trainable_parameter_census"] = census
            metrics["phase_two_resumed"]["trainable_parameter_census"] = {
                **census,
                "descriptor_sha256": "b" * 64,
            }
            with self.assertRaisesRegex(ValueError, "same trainable parameter set"):
                _verify_pilot_artifacts(bundle, metrics)

            metrics["phase_two_resumed"]["trainable_parameter_census"] = census
            self.assertEqual(_verify_pilot_artifacts(bundle, metrics), (1, 1))

    def test_parent_runtime_probe_selects_candidate_device_indices(self) -> None:
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                {
                    "hardware": {
                        "cuda_runtime": "test",
                        "driver_version": "test",
                        "devices": [{"index": 0, "uuid": "GPU-selected"}],
                    },
                    "free_cuda_bytes": [1234],
                    "host_ram_free_bytes": 5678,
                }
            ),
            stderr="",
        )
        with patch("aptus.execution.subprocess.run", return_value=completed) as runner:
            snapshot = _actual_runtime_snapshot(1, [2])
        command = runner.call_args.args[0]
        self.assertEqual(json.loads(command[-1]), [2])
        self.assertEqual(snapshot["free_cuda_bytes"], [1234])

    def test_parent_independently_verifies_safetensors_keys_and_indexes(self) -> None:
        class FakeSafeTensorFile:
            def __init__(self, keys: list[str]) -> None:
                self._keys = keys

            def __enter__(self) -> "FakeSafeTensorFile":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def keys(self) -> list[str]:
                return list(self._keys)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_keys: dict[str, list[str] | Exception] = {}
            fake_safetensors = types.ModuleType("safetensors")

            def safe_open(path: str, **_kwargs: object) -> FakeSafeTensorFile:
                value = shard_keys[Path(path).name]
                if isinstance(value, Exception):
                    raise value
                return FakeSafeTensorFile(value)

            fake_safetensors.safe_open = safe_open  # type: ignore[attr-defined]

            def make_export(
                name: str,
                shards: dict[str, list[str] | Exception],
                weight_map: dict[str, str] | None = None,
            ) -> tuple[Path, list[Path]]:
                directory = root / name
                directory.mkdir()
                paths = []
                for shard_name, keys in shards.items():
                    path = directory / shard_name
                    path.write_bytes(b"safetensors-placeholder")
                    shard_keys[shard_name] = keys
                    paths.append(path)
                if weight_map is not None:
                    (directory / "model.safetensors.index.json").write_text(
                        json.dumps({"weight_map": weight_map}), encoding="utf-8"
                    )
                return directory, paths

            with patch.dict(sys.modules, {"safetensors": fake_safetensors}):
                valid, valid_paths = make_export(
                    "valid", {"model.safetensors": ["model.weight"]}
                )
                _verify_safetensors_structure(valid, valid_paths)

                malformed, malformed_paths = make_export(
                    "malformed",
                    {"model.safetensors": RuntimeError("bad header")},
                )
                with self.assertRaisesRegex(ValueError, "parent structural loading"):
                    _verify_safetensors_structure(malformed, malformed_paths)

                empty, empty_paths = make_export("empty", {"model.safetensors": []})
                with self.assertRaisesRegex(ValueError, "no tensor keys"):
                    _verify_safetensors_structure(empty, empty_paths)

                duplicate, duplicate_paths = make_export(
                    "duplicate",
                    {
                        "model-00001-of-00002.safetensors": ["shared.weight"],
                        "model-00002-of-00002.safetensors": ["shared.weight"],
                    },
                    {"shared.weight": "model-00001-of-00002.safetensors"},
                )
                with self.assertRaisesRegex(ValueError, "duplicate tensor key"):
                    _verify_safetensors_structure(duplicate, duplicate_paths)

                misindexed, misindexed_paths = make_export(
                    "misindexed",
                    {
                        "model-00001-of-00002.safetensors": ["weight.a"],
                        "model-00002-of-00002.safetensors": ["weight.b"],
                    },
                    {
                        "weight.a": "model-00002-of-00002.safetensors",
                        "weight.b": "model-00001-of-00002.safetensors",
                    },
                )
                with self.assertRaisesRegex(ValueError, "wrong shards"):
                    _verify_safetensors_structure(misindexed, misindexed_paths)

                (misindexed / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "weight_map": {
                                "weight.a": "model-00001-of-00002.safetensors",
                                "weight.b": "model-00002-of-00002.safetensors",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                _verify_safetensors_structure(misindexed, misindexed_paths)

    def test_host_submission_rejects_a_bundle_after_policy_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            write_validation_state(bundle, "pilot-pass")
            service = JobService(root / "jobs")

            with (
                patch(
                    "aptus.model_compatibility.current_model_policy_snapshot",
                    return_value=changed_model_policy_snapshot(),
                ),
                self.assertRaises(StaleModelPolicyError) as raised,
            ):
                service.submit(bundle, action="train", confirm_full_train=True)

            self.assertIn("replan_required", str(raised.exception))
            self.assertEqual(service.list(), [])
            self.assertEqual(service._threads, {})
            self.assertFalse(service._lease_path.exists())

    def test_pilot_authorization_is_non_current_after_policy_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            write_validation_state(bundle, "pilot-pass")

            with patch(
                "aptus.model_compatibility.current_model_policy_snapshot",
                return_value=changed_model_policy_snapshot(),
            ):
                authorization = JobService(root / "jobs").pilot_authorization(bundle)

        self.assertFalse(authorization["current"])
        self.assertIn("replan_required", authorization["error"])
        self.assertIsNone(authorization["capacity"])

    def test_recovery_does_not_promote_evidence_under_a_new_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            service = JobService(root / "jobs")
            job_id = "job_" + "f" * 32
            service._write(
                {
                    "id": job_id,
                    "job_id": job_id,
                    "state": "running",
                    "action": "train",
                    "bundle_dir": str(bundle),
                    "artifact_fingerprint": sha256_file(
                        bundle / "bundle-manifest.json"
                    ),
                    "return_code": 0,
                    "owner_pid": -1,
                    "process_pid": None,
                    "created_at": "2026-08-02T00:00:00+00:00",
                    "verified_pending_evidence": {
                        "training_runtime": "transformers-peft-cuda"
                    },
                }
            )

            with (
                patch(
                    "aptus.model_compatibility.current_model_policy_snapshot",
                    return_value=changed_model_policy_snapshot(),
                ),
                patch(
                    "aptus.execution._promote_cuda_train_attestation",
                    return_value={"state": "measured-run-pass"},
                ) as promote,
            ):
                recovered = service.get(job_id, include_validation_report=False)

        promote.assert_not_called()
        self.assertEqual(recovered["state"], "failed")
        self.assertIn("replan_required", recovered["error"])
        self.assertNotIn("completion_attestation", recovered)

    def test_host_guard_rejects_policy_state_incoherent_with_embedded_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            plan_path = bundle / "plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["model_policy_decision"]["unexpected"] = "not-contract-state"
            write_json(plan_path, plan)
            manifest_path = bundle / "bundle-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["plan_sha256"] = sha256_file(plan_path)
            plan_entry = next(
                item for item in manifest["files"] if item["path"] == "plan.json"
            )
            plan_entry["sha256"] = sha256_file(plan_path)
            plan_entry["size_bytes"] = plan_path.stat().st_size
            write_json(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "embedded model policy"):
                _require_current_bundle_model_policy(bundle)

    def test_host_guard_detects_plan_change_during_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            plan_path = bundle / "plan.json"

            def swap_plan(_bundle: Path) -> tuple[str, ...]:
                write_json(
                    plan_path,
                    {
                        "model_policy_snapshot_sha256": (
                            current_model_policy_snapshot_sha256()
                        )
                    },
                )
                return ()

            with (
                patch(
                    "aptus.execution.validate_bundle_manifest",
                    side_effect=swap_plan,
                ),
                self.assertRaisesRegex(ValueError, "changed while"),
            ):
                _require_current_bundle_model_policy(bundle)

    def test_host_guard_detects_change_during_current_policy_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            plan_path = bundle / "plan.json"

            def mutate_after_embedded_check(_plan: object, **_kwargs: object) -> None:
                plan_path.write_bytes(plan_path.read_bytes() + b" ")

            with (
                patch(
                    "aptus.execution.require_current_model_policy_snapshot",
                    side_effect=mutate_after_embedded_check,
                ),
                self.assertRaisesRegex(ValueError, "changed while"),
            ):
                _require_current_bundle_model_policy(bundle)

    def test_submission_builds_command_from_the_admitted_plan_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            plan_path = bundle / "plan.json"
            admitted_bytes = plan_path.read_bytes()
            tampered = json.loads(admitted_bytes)
            tampered["recommended"]["runtime_contract"] = {"training_runtime": "mlx-lm"}
            service = JobService(root / "jobs")
            admitted_command = service._command

            def swap_during_command(
                command_bundle: Path,
                action: str,
                **arguments: object,
            ) -> list[str]:
                write_json(plan_path, tampered)
                try:
                    return admitted_command(
                        command_bundle,
                        action,  # type: ignore[arg-type]
                        **arguments,  # type: ignore[arg-type]
                    )
                finally:
                    plan_path.write_bytes(admitted_bytes)

            submitted: dict | None = None
            try:
                with (
                    patch.object(service, "_command", side_effect=swap_during_command),
                    patch(
                        "aptus.execution.resolve_runtime_interpreter",
                        return_value=types.SimpleNamespace(path="/tampered/mlx-python"),
                    ),
                    patch("aptus.execution.threading.Thread.start"),
                ):
                    submitted = service.submit(bundle, action="preflight")
                self.assertEqual(submitted["command"][0], sys.executable)
            finally:
                if submitted is not None:
                    service._threads.pop(submitted["id"], None)
                    service._clear_global_lease(submitted["id"])

    def test_train_pilot_check_is_bound_to_the_admitted_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root, validation_state="pilot-pass")
            fingerprint = sha256_file(bundle / "bundle-manifest.json")
            service = JobService(root / "jobs")
            submitted: dict | None = None
            try:
                with (
                    patch.object(
                        service,
                        "_require_current_pilot",
                        return_value={"measured_peak_bytes": 1},
                    ) as require_pilot,
                    patch("aptus.execution.threading.Thread.start"),
                ):
                    submitted = service.submit(
                        bundle,
                        action="train",
                        confirm_full_train=True,
                    )
                require_pilot.assert_called_once_with(
                    bundle.resolve(),
                    expected_artifact_fingerprint=fingerprint,
                )
            finally:
                if submitted is not None:
                    service._threads.pop(submitted["id"], None)
                    service._clear_global_lease(submitted["id"])

    def test_worker_launch_rechecks_current_policy_and_observed_fingerprint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            service = JobService(root / "jobs")
            submitted: dict | None = None
            try:
                with patch("aptus.execution.threading.Thread.start"):
                    submitted = service.submit(bundle, action="dependency")
                self.assertEqual(
                    submitted["artifact_fingerprint"],
                    sha256_file(bundle / "bundle-manifest.json"),
                )
                with (
                    patch(
                        "aptus.model_compatibility.current_model_policy_snapshot",
                        return_value=changed_model_policy_snapshot(),
                    ),
                    self.assertRaises(StaleModelPolicyError),
                ):
                    service._require_record_bundle_binding(submitted)
            finally:
                if submitted is not None:
                    service._threads.pop(submitted["id"], None)
                    service._clear_global_lease(submitted["id"])

    def test_recovery_preserves_evidence_already_promoted_before_policy_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            job_id = "job_" + "e" * 32
            run_id = "run_" + "e" * 32
            active_run = {"run_id": run_id}
            pending_at = "2026-08-02T00:00:01+00:00"
            final_export = {
                "schema_version": "aptus.final-export.v1",
                "manifest_sha256": "a" * 64,
            }
            measured_run = {
                "schema_version": "aptus.measured-run.v1",
                "metrics_sha256": "b" * 64,
            }
            write_json(
                bundle / "validation-report.json",
                {
                    "state": "execution-approved",
                    "active_run": active_run,
                    "measured_run_pending_at": pending_at,
                    "pending_final_export": final_export,
                    "pending_measured_run": measured_run,
                },
            )
            service = JobService(root / "jobs")
            evidence = {
                "training_runtime": "transformers-peft-cuda",
                "active_run": active_run,
                "pending_at": pending_at,
                "final_export": final_export,
                "measured_run": measured_run,
            }
            record = {
                "id": job_id,
                "job_id": job_id,
                "run_id": run_id,
                "state": "running",
                "action": "train",
                "bundle_dir": str(bundle),
                "artifact_fingerprint": sha256_file(bundle / "bundle-manifest.json"),
                "return_code": 0,
                "owner_pid": -1,
                "process_pid": None,
                "created_at": "2026-08-02T00:00:00+00:00",
                "verified_pending_evidence": evidence,
            }
            promoted = _promote_train_attestation(record, evidence)
            promoted_report = json.loads(
                (bundle / "validation-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(promoted["state"], "measured-run-pass")
            self.assertEqual(
                promoted_report["parent_promotion"]["artifact_fingerprint"],
                record["artifact_fingerprint"],
            )
            service._write(record)

            with (
                patch(
                    "aptus.model_compatibility.current_model_policy_snapshot",
                    return_value=changed_model_policy_snapshot(),
                ),
                patch("aptus.execution._promote_cuda_train_attestation") as promote,
            ):
                recovered = service.get(job_id, include_validation_report=False)

        promote.assert_not_called()
        self.assertEqual(recovered["state"], "completed")
        self.assertEqual(
            recovered["completion_attestation"]["state"], "measured-run-pass"
        )

    def test_recovery_rejects_unreceipted_matching_terminal_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            final_export = {"manifest_sha256": "a" * 64}
            measured_run = {"metrics_sha256": "b" * 64}
            write_json(
                bundle / "validation-report.json",
                {
                    "schema_version": "aptus.validation.v2",
                    "state": "measured-run-pass",
                    "final_export": final_export,
                    "measured_run": measured_run,
                },
            )
            service = JobService(root / "jobs")
            job_id = "job_" + "d" * 32
            service._write(
                {
                    "id": job_id,
                    "job_id": job_id,
                    "run_id": "run_" + "d" * 32,
                    "state": "running",
                    "action": "train",
                    "bundle_dir": str(bundle),
                    "artifact_fingerprint": sha256_file(
                        bundle / "bundle-manifest.json"
                    ),
                    "return_code": 0,
                    "owner_pid": -1,
                    "process_pid": None,
                    "created_at": "2026-08-02T00:00:00+00:00",
                    "verified_pending_evidence": {
                        "training_runtime": "transformers-peft-cuda",
                        "final_export": final_export,
                        "measured_run": measured_run,
                    },
                }
            )

            with patch(
                "aptus.model_compatibility.current_model_policy_snapshot",
                return_value=changed_model_policy_snapshot(),
            ):
                recovered = service.get(job_id, include_validation_report=False)

        self.assertEqual(recovered["state"], "failed")
        self.assertIn("parent-promotion receipt", recovered["error"])
        self.assertNotIn("completion_attestation", recovered)

    def test_parent_promotion_rechecks_bundle_before_committing_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            job_id = "job_" + "c" * 32
            run_id = "run_" + "c" * 32
            active_run = {"run_id": run_id}
            pending_at = "2026-08-02T00:00:01+00:00"
            final_export = {"manifest_sha256": "a" * 64}
            measured_run = {"metrics_sha256": "b" * 64}
            write_json(
                bundle / "validation-report.json",
                {
                    "schema_version": "aptus.validation.v2",
                    "state": "execution-approved",
                    "active_run": active_run,
                    "measured_run_pending_at": pending_at,
                    "pending_final_export": final_export,
                    "pending_measured_run": measured_run,
                },
            )
            record = {
                "id": job_id,
                "run_id": run_id,
                "bundle_dir": str(bundle),
                "artifact_fingerprint": sha256_file(bundle / "bundle-manifest.json"),
            }
            evidence = {
                "training_runtime": "transformers-peft-cuda",
                "active_run": active_run,
                "pending_at": pending_at,
                "final_export": final_export,
                "measured_run": measured_run,
            }
            manifest_path = bundle / "bundle-manifest.json"
            original_guard = _require_current_bundle_model_policy
            changed = False

            def change_after_current_check(*args: object, **kwargs: object) -> object:
                nonlocal changed
                result = original_guard(*args, **kwargs)  # type: ignore[arg-type]
                if kwargs.get("enforce_current_policy", True) and not changed:
                    changed = True
                    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
                return result

            with (
                patch(
                    "aptus.execution._require_current_bundle_model_policy",
                    side_effect=change_after_current_check,
                ),
                self.assertRaisesRegex(ValueError, "project-bound artifact"),
            ):
                _promote_train_attestation(record, evidence)

            report = json.loads(
                (bundle / "validation-report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report["state"], "execution-approved")
        self.assertNotIn("parent_promotion", report)

    def test_parent_promotion_crash_before_receipt_keeps_pending_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            generate_bundle(make_plan(root, gpu_count=1), bundle)
            job_id = "job_" + "b" * 32
            run_id = "run_" + "b" * 32
            active_run = {"run_id": run_id}
            pending_at = "2026-08-02T00:00:01+00:00"
            final_export = {"manifest_sha256": "a" * 64}
            measured_run = {"metrics_sha256": "b" * 64}
            write_json(
                bundle / "validation-report.json",
                {
                    "schema_version": "aptus.validation.v2",
                    "state": "execution-approved",
                    "active_run": active_run,
                    "measured_run_pending_at": pending_at,
                    "pending_final_export": final_export,
                    "pending_measured_run": measured_run,
                },
            )
            record = {
                "id": job_id,
                "run_id": run_id,
                "bundle_dir": str(bundle),
                "artifact_fingerprint": sha256_file(bundle / "bundle-manifest.json"),
            }
            evidence = {
                "training_runtime": "transformers-peft-cuda",
                "active_run": active_run,
                "pending_at": pending_at,
                "final_export": final_export,
                "measured_run": measured_run,
            }
            original_guard = _require_current_bundle_model_policy
            guard_calls = 0

            def interrupt_final_check(*args: object, **kwargs: object) -> object:
                nonlocal guard_calls
                guard_calls += 1
                if guard_calls == 3:
                    raise KeyboardInterrupt("simulated promotion crash")
                return original_guard(*args, **kwargs)  # type: ignore[arg-type]

            with (
                patch(
                    "aptus.execution._require_current_bundle_model_policy",
                    side_effect=interrupt_final_check,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                _promote_train_attestation(record, evidence)

            report = json.loads(
                (bundle / "validation-report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report["state"], "execution-approved")
        self.assertNotIn("parent_promotion", report)

    def test_direct_action_jumps_are_rejected_with_typed_prerequisites(self) -> None:
        cases = (
            ("dependency", None, "static-pass"),
            ("model-data", "static-pass", "dependency-pass"),
            ("preflight", "dependency-pass", "model-data-pass"),
            ("pilot", "model-data-pass", "measured-preflight-pass"),
            ("train", "measured-preflight-pass", "pilot-pass"),
        )
        for action, current_state, required_state in cases:
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                bundle = fake_bundle(root, validation_state=current_state)
                with self.assertRaises(JobPrerequisiteError) as raised:
                    JobService(root / "jobs").submit(
                        bundle,
                        action=action,  # type: ignore[arg-type]
                        confirm_full_train=action == "train",
                    )
                error = raised.exception
                self.assertEqual(error.code, "job_prerequisite_not_met")
                self.assertEqual(error.action, action)
                self.assertEqual(error.current_state, current_state)
                self.assertEqual(error.required_state, required_state)

    def test_ordered_actions_accept_persisted_state_across_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root, validation_state="static-pass")
            stages = (
                ("dependency", "dependency-pass"),
                ("model-data", "model-data-pass"),
                ("preflight", "measured-preflight-pass"),
                ("pilot", "pilot-pass"),
            )
            for action, promoted_state in stages:
                with self.subTest(action=action):
                    service = JobService(root / "jobs")
                    submitted = service.submit(
                        bundle,
                        action=action,  # type: ignore[arg-type]
                    )
                    finished = wait_for(service, submitted["id"])
                    self.assertEqual(finished["state"], "completed")
                    write_validation_state(bundle, promoted_state)

            service = JobService(root / "jobs")
            with patch.object(
                service,
                "_require_current_pilot",
                return_value={"checked_at": "test"},
            ):
                submitted = service.submit(
                    bundle, action="train", confirm_full_train=True
                )
            finished = wait_for(service, submitted["id"])
        self.assertEqual(submitted["prelaunch_capacity_check"], {"checked_at": "test"})
        self.assertEqual(finished["state"], "failed")
        self.assertIn("completion verification failed", finished["error"])

    def test_unknown_action_cannot_fall_through_to_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            with self.assertRaisesRegex(ValueError, "Unsupported job action"):
                service.submit(fake_bundle(root), action="typo")  # type: ignore[arg-type]

    def test_train_only_fields_are_rejected_for_validation_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            service = JobService(root / "jobs")
            with self.assertRaisesRegex(ValueError, "confirm_full_train"):
                service.submit(bundle, action="preflight", confirm_full_train=True)
            with self.assertRaisesRegex(ValueError, "resume_from"):
                service.submit(bundle, action="pilot", resume_from="checkpoint-1")

    def test_corrupt_job_record_is_quarantined_and_other_jobs_remain_available(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            jobs.mkdir()
            job_id = "job_" + "f" * 32
            (jobs / f"{job_id}.json").write_text("{", encoding="utf-8")
            service = JobService(jobs)
            quarantined = list((jobs / "quarantine").glob(f"*{job_id}.json"))
            self.assertEqual(service.list(), [])
            self.assertEqual(len(quarantined), 1)

    def test_job_corrupted_after_startup_does_not_hide_healthy_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            jobs.mkdir()
            healthy_id = "job_" + "a" * 32
            corrupt_id = "job_" + "b" * 32
            record = {
                "schema_version": "aptus.job-record.v1",
                "id": healthy_id,
                "job_id": healthy_id,
                "state": "completed",
                "created_at": "2026-01-01T00:00:00+00:00",
                "action": "preflight",
                "bundle_dir": "/tmp/bundle",
            }
            (jobs / f"{healthy_id}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            service = JobService(jobs)
            corrupt = jobs / f"{corrupt_id}.json"
            corrupt.write_text("{", encoding="utf-8")

            listed = service.list()

            self.assertEqual([item["id"] for item in listed], [healthy_id])
            receipts = list((jobs / "quarantine").glob(f"*{corrupt_id}*.reason.json"))
            self.assertEqual(len(receipts), 1)

    def test_legacy_job_record_is_migrated_to_the_versioned_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            jobs.mkdir()
            job_id = "job_" + "e" * 32
            path = jobs / f"{job_id}.json"
            path.write_text(
                json.dumps(
                    {
                        "id": job_id,
                        "job_id": job_id,
                        "state": "completed",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "authorization_status": "current",
                        "authorization_current": True,
                    }
                ),
                encoding="utf-8",
            )
            record = JobService(jobs).get(job_id, include_validation_report=False)
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["schema_version"], "aptus.job-record.v1")
        self.assertEqual(persisted["schema_version"], "aptus.job-record.v1")
        self.assertEqual(
            persisted["persistence_migrated_from"], "aptus.job-record.legacy"
        )
        self.assertNotIn("authorization_status", persisted)
        self.assertNotIn("authorization_current", persisted)
        self.assertIsNone(record["child_process_started_monotonic_ns"])
        self.assertIsNone(record["child_process_finished_monotonic_ns"])
        self.assertIsNone(record["queued_monotonic_ns"])
        self.assertIsNone(record["terminal_monotonic_ns"])
        self.assertIsNone(record["monotonic_clock_binding"])
        self.assertIsNone(persisted["child_process_started_monotonic_ns"])
        self.assertIsNone(persisted["child_process_finished_monotonic_ns"])

    def test_child_process_monotonic_boundaries_are_persisted_for_success(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            submitted = service.submit(fake_bundle(root), action="preflight")
            finished = wait_for(service, submitted["id"])
            persisted = json.loads(
                (root / "jobs" / f"{submitted['id']}.json").read_text(encoding="utf-8")
            )

        started = finished["child_process_started_monotonic_ns"]
        stopped = finished["child_process_finished_monotonic_ns"]
        queued = finished["queued_monotonic_ns"]
        terminal = finished["terminal_monotonic_ns"]
        self.assertEqual(finished["state"], "completed")
        self.assertRegex(
            finished["monotonic_clock_binding"],
            r"^(?:linux-boot-sha256:[0-9a-f]{64}|process-monotonic:[0-9a-f]{32})$",
        )
        self.assertIsInstance(started, int)
        self.assertNotIsInstance(started, bool)
        self.assertIsInstance(stopped, int)
        self.assertNotIsInstance(stopped, bool)
        self.assertLessEqual(queued, started)
        self.assertLessEqual(started, stopped)
        self.assertLessEqual(stopped, terminal)
        self.assertEqual(persisted["queued_monotonic_ns"], queued)
        self.assertEqual(persisted["child_process_started_monotonic_ns"], started)
        self.assertEqual(persisted["child_process_finished_monotonic_ns"], stopped)
        self.assertEqual(persisted["terminal_monotonic_ns"], terminal)

    def test_child_process_monotonic_boundaries_are_persisted_for_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            validate_path = bundle / "validate.py"
            validate_path.write_text("raise SystemExit(7)\n", encoding="utf-8")
            manifest_path = bundle / "bundle-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_entry = next(
                item for item in manifest["files"] if item["path"] == "validate.py"
            )
            validate_entry["sha256"] = sha256_file(validate_path)
            validate_entry["size_bytes"] = validate_path.stat().st_size
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            finished = wait_for(service, submitted["id"])

        self.assertEqual(finished["state"], "failed")
        self.assertEqual(finished["return_code"], 7)
        self.assertLessEqual(
            finished["child_process_started_monotonic_ns"],
            finished["child_process_finished_monotonic_ns"],
        )
        self.assertLessEqual(
            finished["queued_monotonic_ns"],
            finished["child_process_started_monotonic_ns"],
        )
        self.assertLessEqual(
            finished["child_process_finished_monotonic_ns"],
            finished["terminal_monotonic_ns"],
        )

    def test_child_finish_boundary_precedes_parent_completion_verification(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root, validation_state="pilot-pass")
            service = JobService(root / "jobs")
            verification_started = threading.Event()
            verification_release = threading.Event()

            def block_verification(_record: object) -> dict:
                verification_started.set()
                verification_release.wait(timeout=5)
                return {}

            try:
                with (
                    patch.object(
                        service,
                        "_require_current_pilot",
                        return_value={"measured_peak_bytes": 1},
                    ),
                    patch(
                        "aptus.execution._verify_train_artifacts",
                        side_effect=block_verification,
                    ),
                    patch(
                        "aptus.execution._promote_train_attestation", return_value={}
                    ),
                ):
                    submitted = service.submit(
                        bundle, action="train", confirm_full_train=True
                    )
                    self.assertTrue(verification_started.wait(timeout=5))
                    verifying = service.get(submitted["id"])
                    child_finished = verifying["child_process_finished_monotonic_ns"]

                    self.assertEqual(verifying["phase"], "verifying")
                    self.assertEqual(verifying["return_code"], 0)
                    self.assertIsInstance(child_finished, int)
                    self.assertIsNone(verifying["terminal_monotonic_ns"])
                    self.assertIsNone(verifying["finished_at"])
                    self.assertIsNotNone(
                        verifying["completion_verification_started_at"]
                    )

                    verification_release.set()
                    finished = wait_for(service, submitted["id"])
            finally:
                verification_release.set()

        self.assertEqual(finished["state"], "completed")
        self.assertEqual(
            finished["child_process_finished_monotonic_ns"], child_finished
        )
        self.assertIsNotNone(finished["finished_at"])
        self.assertGreaterEqual(
            finished["terminal_monotonic_ns"],
            finished["child_process_finished_monotonic_ns"],
        )

    def test_restart_preserves_child_process_monotonic_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            submitted = service.submit(fake_bundle(root), action="preflight")
            finished = wait_for(service, submitted["id"])

            reloaded = JobService(root / "jobs").get(submitted["id"])

        self.assertEqual(
            reloaded["child_process_started_monotonic_ns"],
            finished["child_process_started_monotonic_ns"],
        )
        self.assertEqual(
            reloaded["child_process_finished_monotonic_ns"],
            finished["child_process_finished_monotonic_ns"],
        )

    def test_restart_fails_closed_without_exact_child_finish_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            jobs.mkdir()
            job_id = "job_" + "c" * 32
            (jobs / f"{job_id}.json").write_text(
                json.dumps(
                    {
                        "schema_version": "aptus.job-record.v1",
                        "id": job_id,
                        "job_id": job_id,
                        "state": "running",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "action": "train",
                        "bundle_dir": str(Path(temporary) / "missing-bundle"),
                        "owner_pid": 999_999_997,
                        "process_pid": 999_999_998,
                        "return_code": 0,
                        "verified_pending_evidence": {},
                        "child_process_started_monotonic_ns": 100,
                        "child_process_finished_monotonic_ns": None,
                    }
                ),
                encoding="utf-8",
            )

            recovered = JobService(jobs).get(job_id)

        self.assertEqual(recovered["state"], "failed")
        self.assertIsNone(recovered["child_process_finished_monotonic_ns"])
        self.assertIn("exact child-process finish boundary", recovered["error"])
        self.assertNotIn("completion_attestation", recovered)

    def test_malformed_child_process_monotonic_boundaries_are_quarantined(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            jobs.mkdir()
            job_id = "job_" + "d" * 32
            (jobs / f"{job_id}.json").write_text(
                json.dumps(
                    {
                        "schema_version": "aptus.job-record.v1",
                        "id": job_id,
                        "job_id": job_id,
                        "state": "completed",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "action": "preflight",
                        "bundle_dir": "/tmp/bundle",
                        "child_process_started_monotonic_ns": 200,
                        "child_process_finished_monotonic_ns": 199,
                    }
                ),
                encoding="utf-8",
            )

            service = JobService(jobs)
            listed = service.list()
            quarantined = list((jobs / "quarantine").glob(f"*{job_id}.json"))

        self.assertEqual(listed, [])
        self.assertEqual(len(quarantined), 1)

    def test_malformed_job_lifetime_monotonic_boundaries_are_quarantined(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            jobs.mkdir()
            job_id = "job_" + "f" * 32
            (jobs / f"{job_id}.json").write_text(
                json.dumps(
                    {
                        "schema_version": "aptus.job-record.v1",
                        "id": job_id,
                        "job_id": job_id,
                        "state": "completed",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "action": "preflight",
                        "bundle_dir": "/tmp/bundle",
                        "monotonic_clock_binding": ("linux-boot-sha256:" + "a" * 64),
                        "queued_monotonic_ns": 100,
                        "child_process_started_monotonic_ns": 110,
                        "child_process_finished_monotonic_ns": 120,
                        "terminal_monotonic_ns": 119,
                    }
                ),
                encoding="utf-8",
            )

            service = JobService(jobs)
            listed = service.list()
            quarantined = list((jobs / "quarantine").glob(f"*{job_id}.json"))

        self.assertEqual(listed, [])
        self.assertEqual(len(quarantined), 1)

    def test_job_record_and_log_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            submitted = service.submit(fake_bundle(root), action="preflight")
            finished = wait_for(service, submitted["id"])
            restarted = JobService(root / "jobs")
            reloaded = restarted.get(submitted["id"])
            listed = restarted.list()
        self.assertEqual(finished["state"], "completed")
        self.assertEqual(finished["return_code"], 0)
        self.assertEqual(finished["id"], finished["job_id"])
        self.assertEqual(reloaded["state"], "completed")
        self.assertEqual([item["id"] for item in listed], [submitted["id"]])
        self.assertIn("validation job passed", reloaded["log_tail"])

    def test_pre_start_persistence_failure_never_starts_worker_or_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")

            def fail_persistence(_record: object) -> None:
                raise RuntimeError("project revision write failed")

            with (
                patch.object(service, "_run") as run_worker,
                patch("aptus.execution.threading.Thread.start") as start_worker,
                self.assertRaises(JobSubmissionFailure) as raised,
            ):
                service.submit(
                    fake_bundle(root),
                    action="preflight",
                    before_start=fail_persistence,
                )

            self.assertEqual(
                raised.exception.failure_code, "PRE_START_PERSISTENCE_FAILED"
            )
            self.assertEqual(raised.exception.terminal_record["state"], "failed")
            self.assertRegex(raised.exception.job_id, r"^job_[0-9a-f]{32}$")
            records = service.list()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["state"], "failed")
            self.assertIn("Job pre-start persistence failed", records[0]["error"])
            self.assertFalse(service._lease_path.exists())
            start_worker.assert_not_called()
            run_worker.assert_not_called()

    def test_admission_binding_failure_creates_no_job_record_or_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")

            def reject_binding() -> None:
                raise RuntimeError("project artifact binding changed")

            with (
                patch("aptus.execution.threading.Thread.start") as start_worker,
                self.assertRaisesRegex(RuntimeError, "artifact binding changed"),
            ):
                service.submit(
                    fake_bundle(root),
                    action="preflight",
                    admission_check=reject_binding,
                )

            self.assertEqual(service.list(), [])
            self.assertFalse(service._lease_path.exists())
            start_worker.assert_not_called()

    def test_post_persist_setup_failure_returns_typed_terminal_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            with (
                patch.object(
                    service,
                    "_create_global_lease",
                    side_effect=OSError("private host detail"),
                ),
                self.assertRaises(JobSubmissionFailure) as raised,
            ):
                service.submit(fake_bundle(root), action="preflight")

            failure = raised.exception
            records = service.list()
            self.assertEqual(failure.failure_code, "SUBMISSION_SETUP_FAILED")
            self.assertEqual(failure.terminal_record["state"], "failed")
            self.assertNotIn("private host detail", str(failure))
            self.assertEqual(records[0]["id"], failure.job_id)
            self.assertEqual(records[0]["state"], "failed")
            self.assertFalse(service._lease_path.exists())

    def test_post_start_handoff_failure_cancels_and_returns_typed_terminal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            with (
                patch.object(
                    service,
                    "get",
                    side_effect=RuntimeError("private handoff detail"),
                ),
                self.assertRaises(JobSubmissionFailure) as raised,
            ):
                service.submit(fake_bundle(root), action="preflight")

            failure = raised.exception
            persisted = service._read(failure.job_id)
            self.assertEqual(failure.failure_code, "SUBMISSION_HANDOFF_FAILED")
            self.assertIn(failure.terminal_record["state"], {"failed", "cancelled"})
            self.assertIn(persisted["state"], {"failed", "cancelled"})
            self.assertNotIn("private handoff detail", str(failure))
            self.assertFalse(service._lease_path.exists())

    def test_opt_in_campaign_event_sink_is_private_and_record_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            experiment_run_id = "xrun_" + "a" * 32
            bundle = fake_bundle(root)
            observed_path = bundle / "campaign-environment.json"
            (bundle / "validate.py").write_text(
                "import json\n"
                "import os\n"
                "from pathlib import Path\n"
                "Path('campaign-environment.json').write_text(json.dumps({"
                "'sink': os.environ.get('APTUS_CUDA_CAMPAIGN_EVENT_SINK'), "
                "'identity': os.environ.get('APTUS_CUDA_CAMPAIGN_EVENT_SINK_IDENTITY'), "
                "'xrun': os.environ.get('APTUS_CUDA_CAMPAIGN_EXPERIMENT_RUN_ID'), "
                "'job': os.environ.get('APTUS_CUDA_CAMPAIGN_JOB_ID')}), encoding='utf-8')\n",
                encoding="utf-8",
            )
            manifest_path = bundle / "bundle-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_entry = next(
                item for item in manifest["files"] if item["path"] == "validate.py"
            )
            validate_entry["sha256"] = sha256_file(bundle / "validate.py")
            validate_entry["size_bytes"] = (bundle / "validate.py").stat().st_size
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            submitted = service.submit(
                bundle,
                action="preflight",
                campaign_event_capture=True,
                campaign_experiment_run_id=experiment_run_id,
            )
            finished = wait_for(service, submitted["id"])
            sink = Path(finished["campaign_event_sink"])
            observed = json.loads(observed_path.read_text(encoding="utf-8"))
            self.assertTrue(finished["campaign_event_capture"])
            self.assertEqual(finished["campaign_experiment_run_id"], experiment_run_id)
            self.assertEqual(sink.name, f"{finished['id']}.jsonl")
            self.assertEqual(sink.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                finished["campaign_event_sink_identity"],
                f"{sink.stat().st_dev}:{sink.stat().st_ino}",
            )
            self.assertEqual(
                observed,
                {
                    "sink": str(sink),
                    "identity": finished["campaign_event_sink_identity"],
                    "xrun": experiment_run_id,
                    "job": finished["id"],
                },
            )

    def test_ordinary_job_scrubs_inherited_campaign_event_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            bundle = fake_bundle(root)
            observed_path = bundle / "campaign-environment.json"
            (bundle / "validate.py").write_text(
                "import json\n"
                "import os\n"
                "from pathlib import Path\n"
                "names = ("
                "'APTUS_CUDA_CAMPAIGN_EVENT_SINK', "
                "'APTUS_CUDA_CAMPAIGN_EVENT_SINK_IDENTITY', "
                "'APTUS_CUDA_CAMPAIGN_EXPERIMENT_RUN_ID', "
                "'APTUS_CUDA_CAMPAIGN_JOB_ID')\n"
                "Path('campaign-environment.json').write_text("
                "json.dumps({name: os.environ.get(name) for name in names}), "
                "encoding='utf-8')\n",
                encoding="utf-8",
            )
            manifest_path = bundle / "bundle-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_entry = next(
                item for item in manifest["files"] if item["path"] == "validate.py"
            )
            validate_entry["sha256"] = sha256_file(bundle / "validate.py")
            validate_entry["size_bytes"] = (bundle / "validate.py").stat().st_size
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            poisoned = {
                "APTUS_CUDA_CAMPAIGN_EVENT_SINK": str(root / "foreign.jsonl"),
                "APTUS_CUDA_CAMPAIGN_EVENT_SINK_IDENTITY": "1:1",
                "APTUS_CUDA_CAMPAIGN_EXPERIMENT_RUN_ID": "xrun_" + "a" * 32,
                "APTUS_CUDA_CAMPAIGN_JOB_ID": "job_" + "b" * 32,
            }
            with patch.dict(os.environ, poisoned, clear=False):
                submitted = service.submit(bundle, action="preflight")
                finished = wait_for(service, submitted["id"])

            observed = json.loads(observed_path.read_text(encoding="utf-8"))
            self.assertEqual(finished["state"], "completed")
            self.assertEqual(observed, {name: None for name in poisoned})

    def test_campaign_event_sink_arguments_are_fail_closed_pre_persist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            bundle = fake_bundle(root)
            with self.assertRaisesRegex(ValueError, "requires an exact"):
                service.submit(bundle, campaign_event_capture=True)
            with self.assertRaisesRegex(ValueError, "requires campaign_event_capture"):
                service.submit(
                    bundle,
                    campaign_experiment_run_id="xrun_" + "a" * 32,
                )
            self.assertEqual(service.list(), [])

    def test_parent_verification_boundaries_use_the_pinned_campaign_sink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            sink = service.root / ".campaign-events" / ("job_" + "b" * 32 + ".jsonl")
            identity = service._create_campaign_event_sink(sink)
            record = {
                "id": "job_" + "b" * 32,
                "job_id": "job_" + "b" * 32,
                "action": "train",
                "campaign_event_capture": True,
                "campaign_experiment_run_id": "xrun_" + "a" * 32,
                "campaign_event_sink": str(sink),
                "campaign_event_sink_identity": identity,
            }
            service._append_campaign_verification_boundary(
                record, event_type="verification.started"
            )
            service._append_campaign_verification_boundary(
                record,
                event_type="verification.finished",
                native_outcome="passed",
            )
            boundaries = [
                json.loads(line)
                for line in sink.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [item["event_type"] for item in boundaries],
                ["verification.started", "verification.finished"],
            )
            self.assertEqual(
                {item["phase"] for item in boundaries}, {"parent-verification"}
            )
            self.assertEqual(boundaries[-1]["native_outcome"], "passed")
            self.assertEqual(boundaries[-1]["reason_code"], "NONE")

    def test_parent_verification_rechecks_sink_metadata_after_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            sink = service.root / ".campaign-events" / ("job_" + "b" * 32 + ".jsonl")
            identity = service._create_campaign_event_sink(sink)
            record = {
                "id": "job_" + "b" * 32,
                "job_id": "job_" + "b" * 32,
                "action": "train",
                "campaign_event_capture": True,
                "campaign_experiment_run_id": "xrun_" + "a" * 32,
                "campaign_event_sink": str(sink),
                "campaign_event_sink_identity": identity,
            }
            lock_calls = 0

            def mutate_after_lock(*_arguments: object) -> None:
                nonlocal lock_calls
                lock_calls += 1
                if lock_calls == 1:
                    sink.chmod(0o644)

            with (
                patch("aptus.execution.fcntl.flock", side_effect=mutate_after_lock),
                self.assertRaisesRegex(RuntimeError, "changed before append"),
            ):
                service._append_campaign_verification_boundary(
                    record, event_type="verification.started"
                )
            self.assertEqual(sink.read_bytes(), b"")

    def test_parent_verification_rechecks_sink_metadata_after_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            sink = service.root / ".campaign-events" / ("job_" + "b" * 32 + ".jsonl")
            identity = service._create_campaign_event_sink(sink)
            record = {
                "id": "job_" + "b" * 32,
                "job_id": "job_" + "b" * 32,
                "action": "train",
                "campaign_event_capture": True,
                "campaign_experiment_run_id": "xrun_" + "a" * 32,
                "campaign_event_sink": str(sink),
                "campaign_event_sink_identity": identity,
            }
            real_fsync = os.fsync

            def mutate_after_fsync(descriptor: int) -> None:
                real_fsync(descriptor)
                sink.chmod(0o644)

            with (
                patch("aptus.execution.os.fsync", side_effect=mutate_after_fsync),
                self.assertRaisesRegex(RuntimeError, "changed during append"),
            ):
                service._append_campaign_verification_boundary(
                    record, event_type="verification.started"
                )
            self.assertTrue(sink.read_bytes())

    def test_worker_rechecks_project_fingerprint_immediately_before_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            launch_marker = bundle / "worker-launched.txt"
            (bundle / "validate.py").write_text(
                "from pathlib import Path\n"
                "Path('worker-launched.txt').write_text('launched', encoding='utf-8')\n",
                encoding="utf-8",
            )
            initial_manifest_path = bundle / "bundle-manifest.json"
            initial_manifest = json.loads(
                initial_manifest_path.read_text(encoding="utf-8")
            )
            validate_entry = next(
                item
                for item in initial_manifest["files"]
                if item["path"] == "validate.py"
            )
            validate_entry["sha256"] = sha256_file(bundle / "validate.py")
            validate_entry["size_bytes"] = (bundle / "validate.py").stat().st_size
            initial_manifest_path.write_text(
                json.dumps(initial_manifest), encoding="utf-8"
            )
            expected_fingerprint = sha256_file(bundle / "bundle-manifest.json")
            service = JobService(root / "jobs")

            def replace_with_self_consistent_bundle() -> None:
                replacement = bundle / "replacement.txt"
                replacement.write_text("substituted\n", encoding="utf-8")
                manifest_path = bundle / "bundle-manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["files"].append(
                    {
                        "path": "replacement.txt",
                        "sha256": sha256_file(replacement),
                        "size_bytes": replacement.stat().st_size,
                    }
                )
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            submitted = service.submit(
                bundle,
                action="preflight",
                admission_check=replace_with_self_consistent_bundle,
                expected_artifact_fingerprint=expected_fingerprint,
            )
            finished = wait_for(service, submitted["id"])

            self.assertEqual(finished["state"], "failed")
            self.assertIn("project-bound artifact", finished["error"])
            self.assertFalse(launch_marker.exists())

    def test_same_path_swap_after_launcher_spawn_never_receives_a_permit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            expected_fingerprint = sha256_file(bundle / "bundle-manifest.json")
            replacement_root = root / "replacement-root"
            replacement_root.mkdir()
            replacement = fake_bundle(replacement_root)
            replacement_marker = root / "replacement-executed.txt"
            (replacement / "validate.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(replacement_marker)!r}).write_text('executed', encoding='utf-8')\n",
                encoding="utf-8",
            )
            replacement_manifest_path = replacement / "bundle-manifest.json"
            replacement_manifest = json.loads(
                replacement_manifest_path.read_text(encoding="utf-8")
            )
            replacement_validate = next(
                item
                for item in replacement_manifest["files"]
                if item["path"] == "validate.py"
            )
            replacement_validate["sha256"] = sha256_file(replacement / "validate.py")
            replacement_validate["size_bytes"] = (
                (replacement / "validate.py").stat().st_size
            )
            replacement_manifest_path.write_text(
                json.dumps(replacement_manifest), encoding="utf-8"
            )
            service = JobService(root / "jobs")
            original_bundle = root / "original-bundle-moved"

            def swap_after_spawn(*_arguments: object) -> None:
                bundle.rename(original_bundle)
                shutil.copytree(replacement, bundle)

            with patch.object(
                service,
                "_bind_global_lease_to_process",
                side_effect=swap_after_spawn,
            ):
                submitted = service.submit(
                    bundle,
                    action="preflight",
                    expected_artifact_fingerprint=expected_fingerprint,
                )
                finished = wait_for(service, submitted["id"])

            self.assertEqual(finished["state"], "failed")
            self.assertIn("project-bound artifact", finished["error"])
            self.assertFalse(replacement_marker.exists())

    def test_managed_child_inherits_lease_and_host_policy_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            token_path = bundle / "observed-lease-token.txt"
            (bundle / "validate.py").write_text(
                "import json\nimport os\nfrom pathlib import Path\n"
                "Path('observed-lease-token.txt').write_text(json.dumps({"
                "'lease': os.environ.get('APTUS_GPU_LEASE_TOKEN'), "
                "'artifact': os.environ.get('APTUS_EXPECTED_ARTIFACT_FINGERPRINT'), "
                "'policy': os.environ.get("
                "'APTUS_AUTHORIZED_MODEL_POLICY_SNAPSHOT_SHA256')}), encoding='utf-8')\n",
                encoding="utf-8",
            )
            manifest = json.loads(
                (bundle / "bundle-manifest.json").read_text(encoding="utf-8")
            )
            for item in manifest["files"]:
                if item["path"] == "validate.py":
                    item["sha256"] = sha256_file(bundle / "validate.py")
                    item["size_bytes"] = (bundle / "validate.py").stat().st_size
            (bundle / "bundle-manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            finished = wait_for(service, submitted["id"])
            observed = json.loads(token_path.read_text(encoding="utf-8"))
        self.assertEqual(finished["state"], "completed")
        self.assertEqual(observed["lease"], submitted["id"])
        self.assertEqual(observed["artifact"], submitted["artifact_fingerprint"])
        self.assertEqual(
            observed["policy"],
            submitted["authorized_model_policy_snapshot_sha256"],
        )

    def test_job_refresh_includes_current_bundle_validation_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            finished = wait_for(service, submitted["id"])
            (bundle / "validation-report.json").write_text(
                json.dumps(
                    {
                        "schema_version": "aptus.validation.v2",
                        "state": "measured-preflight-pass",
                    }
                ),
                encoding="utf-8",
            )
            refreshed = service.get(finished["id"])
        self.assertEqual(
            refreshed["validation_report"]["state"],
            "measured-preflight-pass",
        )

    def test_missing_current_report_is_an_explicit_refresh_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            finished = wait_for(service, submitted["id"])
            (bundle / "validation-report.json").unlink()
            refreshed = service.get(finished["id"])
        self.assertIn(
            "validation report is missing", refreshed["validation_report_error"]
        )

    def test_full_training_requires_confirmation_and_current_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            service = JobService(root / "jobs")
            with self.assertRaisesRegex(ValueError, "confirm"):
                service.submit(bundle, action="train")
            with self.assertRaisesRegex(ValueError, "pilot"):
                service.submit(bundle, action="train", confirm_full_train=True)
            plan = json.loads((bundle / "plan.json").read_text())
            (bundle / "pilot-output").mkdir()
            (bundle / "pilot-output" / "metrics.json").write_text(
                json.dumps(
                    {
                        "checkpoint_continuation_observed": True,
                        "phase_one": {"measured_reserved_cuda_bytes": 100},
                        "phase_two_resumed": {"measured_reserved_cuda_bytes": 120},
                    }
                ),
                encoding="utf-8",
            )
            bindings = {
                "bundle": sha256_file(bundle / "bundle-manifest.json"),
                "dataset": plan["dataset"]["source_sha256"],
                "model_revision": plan["model"]["revision"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "environment": _environment_binding(bundle),
                "hardware": _json_hash({"identity": "hardware-test"}),
                "pilot_metrics": sha256_file(bundle / "pilot-output" / "metrics.json"),
            }
            (bundle / "validation-report.json").write_text(
                json.dumps({"state": "pilot-pass", "bindings": bindings}),
                encoding="utf-8",
            )
            with (
                patch(
                    "aptus.execution.validate_plan_payload",
                    return_value=(),
                ),
                patch(
                    "aptus.execution._actual_runtime_snapshot",
                    return_value={
                        "hardware": {"identity": "hardware-test"},
                        "free_cuda_bytes": [10_000],
                        "host_ram_free_bytes": 10_000,
                    },
                ),
                patch(
                    "aptus.execution._verify_pilot_artifacts",
                    return_value=(100, 100),
                ),
            ):
                submitted = service.submit(
                    bundle, action="train", confirm_full_train=True
                )
                finished = wait_for(service, submitted["id"])
        self.assertEqual(finished["state"], "failed")
        self.assertIn("completion verification failed", finished["error"])

    def test_malformed_pilot_metrics_fail_closed_without_service_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            plan = json.loads((bundle / "plan.json").read_text())
            (bundle / "pilot-output").mkdir()
            metrics_path = bundle / "pilot-output" / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "checkpoint_continuation_observed": True,
                        "phase_one": {"measured_reserved_cuda_bytes": "100"},
                        "phase_two_resumed": {"measured_reserved_cuda_bytes": 120},
                    }
                ),
                encoding="utf-8",
            )
            bindings = {
                "bundle": sha256_file(bundle / "bundle-manifest.json"),
                "dataset": plan["dataset"]["source_sha256"],
                "model_revision": plan["model"]["revision"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "environment": _environment_binding(bundle),
                "hardware": _json_hash({"identity": "hardware-test"}),
                "pilot_metrics": sha256_file(metrics_path),
            }
            (bundle / "validation-report.json").write_text(
                json.dumps({"state": "pilot-pass", "bindings": bindings}),
                encoding="utf-8",
            )
            with (
                patch("aptus.execution.validate_plan_payload", return_value=()),
                patch(
                    "aptus.execution._actual_runtime_snapshot",
                    return_value={
                        "hardware": {"identity": "hardware-test"},
                        "free_cuda_bytes": [10_000],
                        "host_ram_free_bytes": 10_000,
                    },
                ),
                patch(
                    "aptus.execution._verify_pilot_artifacts",
                    return_value=(100, 100),
                ),
            ):
                authorization = JobService(root / "jobs").pilot_authorization(bundle)
        self.assertFalse(authorization["current"])
        self.assertIn("non-negative integer", authorization["error"])

    def test_terminal_train_job_defers_deep_authorization_until_submission(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            (bundle / "validation-report.json").write_text(
                json.dumps({"state": "pilot-pass", "bindings": {}}),
                encoding="utf-8",
            )
            service = JobService(root / "jobs")
            job_id = "job_" + "c" * 32
            service._write(
                {
                    "id": job_id,
                    "job_id": job_id,
                    "state": "completed",
                    "action": "train",
                    "bundle_dir": str(bundle),
                    "created_at": "2026-07-21T00:00:00+00:00",
                    "prelaunch_capacity_check": {"checked_at": "stale"},
                }
            )
            with patch.object(
                service,
                "pilot_authorization",
                return_value={
                    "current": False,
                    "error": "current environment drifted",
                    "capacity": None,
                },
            ) as authorization:
                refreshed = service.get(job_id)
        authorization.assert_not_called()
        self.assertEqual(
            refreshed["validation_report"]["authorization_status"], "deferred"
        )
        self.assertFalse(refreshed["validation_report"]["authorization_current"])
        self.assertIn(
            "performed atomically when full training is submitted",
            refreshed["validation_report"]["authorization_error"],
        )

    def test_active_job_reports_current_or_blocked_authorization_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            (bundle / "validation-report.json").write_text(
                json.dumps({"state": "pilot-pass", "bindings": {}}),
                encoding="utf-8",
            )
            service = JobService(root / "jobs")
            cases = (
                (
                    "job_" + "a" * 32,
                    "train",
                    {"checked_at": "current"},
                    "current",
                    True,
                ),
                ("job_" + "b" * 32, "pilot", None, "blocked", False),
            )
            for job_id, action, capacity, expected_status, expected_current in cases:
                with self.subTest(status=expected_status):
                    service._write(
                        {
                            "id": job_id,
                            "job_id": job_id,
                            "state": "running",
                            "action": action,
                            "bundle_dir": str(bundle),
                            "created_at": "2026-07-21T00:00:00+00:00",
                            "prelaunch_capacity_check": capacity,
                        }
                    )
                    with patch.object(
                        service,
                        "_reconcile_external_record",
                        side_effect=lambda record: record,
                    ):
                        refreshed = service.get(job_id)
                    report = refreshed["validation_report"]
                    self.assertEqual(report["authorization_status"], expected_status)
                    self.assertIs(report["authorization_current"], expected_current)
                    if expected_current:
                        self.assertIsNone(report["authorization_error"])
                    else:
                        self.assertTrue(report["authorization_error"].strip())

    def test_manifested_change_invalidates_pilot_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            plan = json.loads((bundle / "plan.json").read_text())
            (bundle / "pilot-output").mkdir()
            (bundle / "pilot-output" / "metrics.json").write_text(
                '{"resume_verified":true}\n', encoding="utf-8"
            )
            bindings = {
                "bundle": sha256_file(bundle / "bundle-manifest.json"),
                "dataset": plan["dataset"]["source_sha256"],
                "model_revision": plan["model"]["revision"],
                "plan_id": plan["plan_id"],
                "candidate_id": plan["recommended"]["candidate_id"],
                "environment": _environment_binding(bundle),
                "hardware": "hardware-test-binding",
                "pilot_metrics": sha256_file(bundle / "pilot-output" / "metrics.json"),
            }
            (bundle / "validation-report.json").write_text(
                json.dumps({"state": "pilot-pass", "bindings": bindings}),
                encoding="utf-8",
            )
            (bundle / "train.py").write_text('print("changed")\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                JobService(root / "jobs").submit(
                    bundle, action="train", confirm_full_train=True
                )

    def test_unmanifested_import_shadow_invalidates_pilot_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            (bundle / "torch.py").write_text(
                "raise RuntimeError('shadowed')\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unmanifested"):
                JobService(root / "jobs").submit(
                    bundle, action="train", confirm_full_train=True
                )

    def test_cancel_terminates_process_group_and_persists_cancelled_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            make_slow(bundle)
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            deadline = time.monotonic() + 2
            current = service.get(submitted["id"])
            while current["state"] == "queued" and time.monotonic() < deadline:
                time.sleep(0.01)
                current = service.get(submitted["id"])
            detected = time.monotonic_ns()
            cancelled = service.cancel(
                submitted["id"],
                reason_code="WATCHDOG_HEARTBEAT_LOST",
                trigger_detected_monotonic_ns=detected,
            )
            persisted = json.loads(
                (root / "jobs" / f"{submitted['id']}.json").read_text(encoding="utf-8")
            )
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertEqual(cancelled["cancel_reason_code"], "WATCHDOG_HEARTBEAT_LOST")
        self.assertEqual(cancelled["cancel_trigger_detected_monotonic_ns"], detected)
        self.assertGreaterEqual(cancelled["cancel_requested_monotonic_ns"], detected)
        self.assertGreaterEqual(
            cancelled["process_group_terminated_monotonic_ns"],
            cancelled["cancel_requested_monotonic_ns"],
        )
        self.assertLessEqual(
            cancelled["child_process_started_monotonic_ns"],
            cancelled["child_process_finished_monotonic_ns"],
        )
        self.assertLessEqual(
            cancelled["child_process_finished_monotonic_ns"],
            cancelled["process_group_terminated_monotonic_ns"],
        )
        self.assertGreaterEqual(
            cancelled["lease_reconciled_monotonic_ns"],
            cancelled["process_group_terminated_monotonic_ns"],
        )
        self.assertLessEqual(
            cancelled["queued_monotonic_ns"],
            cancelled["child_process_started_monotonic_ns"],
        )
        self.assertLessEqual(
            cancelled["child_process_finished_monotonic_ns"],
            cancelled["terminal_monotonic_ns"],
        )
        self.assertEqual(
            persisted["lease_reconciled_monotonic_ns"],
            cancelled["lease_reconciled_monotonic_ns"],
        )

    def test_cancel_rejects_invalid_campaign_milestone_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            make_slow(bundle)
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            with self.assertRaisesRegex(ValueError, "reason_code"):
                service.cancel(submitted["id"], reason_code="")
            with self.assertRaisesRegex(ValueError, "monotonic"):
                service.cancel(submitted["id"], trigger_detected_monotonic_ns=True)
            with self.assertRaisesRegex(ValueError, "future"):
                service.cancel(
                    submitted["id"],
                    trigger_detected_monotonic_ns=time.monotonic_ns() + 1_000_000_000,
                )
            service.cancel(submitted["id"])

    def test_completion_verification_is_noncancellable_and_cannot_contradict_pass(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root, validation_state="pilot-pass")
            service = JobService(root / "jobs")
            verification_started = threading.Event()
            verification_release = threading.Event()

            def block_verification(_record: object) -> dict:
                verification_started.set()
                verification_release.wait(timeout=5)
                return {}

            def promote(record: dict, _pending: object, **_kwargs: object) -> dict:
                report_path = Path(record["bundle_dir"]) / "validation-report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["state"] = "measured-run-pass"
                report_path.write_text(json.dumps(report), encoding="utf-8")
                return {}

            with (
                patch.object(
                    service,
                    "_require_current_pilot",
                    return_value={"measured_peak_bytes": 1},
                ),
                patch(
                    "aptus.execution._verify_train_artifacts",
                    side_effect=block_verification,
                ),
                patch(
                    "aptus.execution._promote_train_attestation",
                    side_effect=promote,
                ),
            ):
                submitted = service.submit(
                    bundle,
                    action="train",
                    confirm_full_train=True,
                    campaign_event_capture=True,
                    campaign_experiment_run_id="xrun_" + "a" * 32,
                )
                self.assertTrue(verification_started.wait(timeout=5))
                verifying = service.get(submitted["id"])
                self.assertEqual(verifying["phase"], "verifying")
                self.assertFalse(verifying["cancellable"])
                with self.assertRaisesRegex(ValueError, "noncancellable"):
                    service.cancel(
                        submitted["id"],
                        reason_code="WATCHDOG_HEARTBEAT_LOST",
                        trigger_detected_monotonic_ns=time.monotonic_ns(),
                    )
                self.assertTrue(service.campaign_lease_active())
                verification_release.set()
                finished = wait_for(service, submitted["id"])

            self.assertEqual(finished["state"], "completed")
            self.assertNotIn("cancel_requested_at", finished)
            report = json.loads(
                (bundle / "validation-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["state"], "measured-run-pass")
            boundaries = [
                json.loads(line)
                for line in Path(finished["campaign_event_sink"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [item["event_type"] for item in boundaries],
                ["verification.started", "verification.finished"],
            )
            self.assertEqual(boundaries[-1]["native_outcome"], "passed")
            self.assertFalse(service.campaign_lease_active())

    def test_cancel_does_not_claim_a_mismatched_lease_was_reconciled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            make_slow(bundle)
            service = JobService(root / "jobs")
            submitted = service.submit(bundle, action="preflight")
            deadline = time.monotonic() + 5
            current = service.get(submitted["id"])
            while current["state"] == "queued" and time.monotonic() < deadline:
                time.sleep(0.01)
                current = service.get(submitted["id"])
            original_clear = service._clear_global_lease
            with patch.object(service, "_clear_global_lease", return_value=False):
                cancelled = service.cancel(
                    submitted["id"],
                    reason_code="LEASE_RECONCILIATION_FAILURE",
                    trigger_detected_monotonic_ns=time.monotonic_ns(),
                )
            self.assertNotIn("lease_reconciled_monotonic_ns", cancelled)
            self.assertIn("lease_reconciliation_error", cancelled)
            original_clear(submitted["id"])

    def test_same_bundle_cannot_launch_overlapping_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            make_slow(bundle)
            service = JobService(root / "jobs")
            first = service.submit(bundle, action="preflight")
            with self.assertRaisesRegex(ValueError, "active job"):
                service.submit(bundle, action="preflight")
            service.cancel(first["id"])

    def test_foreign_service_preserves_live_owner_and_refuses_cancellation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            make_slow(bundle)
            owner = JobService(root / "jobs")
            submitted = owner.submit(bundle, action="preflight")
            foreign = JobService(root / "jobs")
            observed = foreign.get(submitted["id"])
            self.assertIn(observed["state"], {"queued", "running"})
            self.assertFalse(observed["cancellable"])
            self.assertEqual(observed["owner_status"], "external-service")
            with self.assertRaisesRegex(ValueError, "does not own"):
                foreign.cancel(submitted["id"])
            self.assertIn(owner.get(submitted["id"])["state"], {"queued", "running"})
            owner.cancel(submitted["id"])

    def test_global_job_guard_blocks_a_different_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root, second_root = root / "first", root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = fake_bundle(first_root)
            second = fake_bundle(second_root)
            make_slow(first)
            service = JobService(root / "jobs")
            submitted = service.submit(first, action="preflight")
            with self.assertRaisesRegex(ValueError, "one local GPU job"):
                service.submit(second, action="preflight")
            service.cancel(submitted["id"])

    def test_host_global_guard_spans_different_state_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root, second_root = root / "first", root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = fake_bundle(first_root)
            second = fake_bundle(second_root)
            make_slow(first)
            owner = JobService(root / "jobs-a")
            competitor = JobService(root / "jobs-b")
            submitted = owner.submit(first, action="preflight")
            with self.assertRaisesRegex(ValueError, "across all state roots"):
                competitor.submit(second, action="preflight")
            owner.cancel(submitted["id"])

    def test_campaign_lease_state_is_read_only_and_tracks_live_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            make_slow(bundle)
            service = JobService(root / "jobs")
            self.assertFalse(service.campaign_lease_active())
            submitted = service.submit(bundle, action="preflight")
            self.assertTrue(service.campaign_lease_active())
            service.cancel(submitted["id"])
            self.assertFalse(service.campaign_lease_active())

    def test_campaign_lease_telemetry_snapshot_never_waits_on_worker_lock(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = JobService(Path(temporary) / "jobs")
            completed = threading.Event()
            observed: list[bool] = []

            def read_snapshot() -> None:
                observed.append(service.campaign_lease_active())
                completed.set()

            with patch.object(service, "_read_global_lease", return_value=None):
                service._lock.acquire()
                try:
                    reader = threading.Thread(target=read_snapshot)
                    reader.start()
                    self.assertTrue(completed.wait(timeout=0.5))
                finally:
                    service._lock.release()
                reader.join(timeout=1)

            self.assertEqual(observed, [False])

    def test_campaign_lease_transition_is_conservatively_active(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = JobService(Path(temporary) / "jobs")
            lease = {"job_id": "job_" + "1" * 32}
            with patch.object(service, "_read_global_lease", side_effect=(None, lease)):
                self.assertTrue(service.campaign_lease_active())
            with (
                patch.object(service, "_read_global_lease", side_effect=(lease, None)),
                patch.object(service, "_lease_snapshot_active", return_value=False),
            ):
                self.assertTrue(service.campaign_lease_active())

    def test_worker_start_failure_releases_global_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = fake_bundle(root)
            service = JobService(root / "jobs")
            with (
                patch.object(service, "_process_identity", return_value="owner"),
                patch(
                    "aptus.execution.threading.Thread.start",
                    side_effect=RuntimeError("injected thread failure"),
                ),
                self.assertRaises(JobSubmissionFailure) as raised,
            ):
                service.submit(bundle, action="preflight")
            self.assertEqual(raised.exception.failure_code, "WORKER_START_FAILED")
            self.assertEqual(raised.exception.terminal_record["state"], "failed")
            replacement = JobService(root / "replacement-jobs")
            submitted = replacement.submit(bundle, action="preflight")
            finished = wait_for(replacement, submitted["id"])
        self.assertEqual(finished["state"], "completed")

    def test_orphan_child_and_cancelling_record_remain_globally_active(self) -> None:
        for state in ("running", "cancelling"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                jobs = root / "jobs"
                jobs.mkdir()
                bundle = fake_bundle(root)
                job_id = "job_" + ("a" if state == "running" else "b") * 32
                record = {
                    "id": job_id,
                    "job_id": job_id,
                    "state": state,
                    "bundle_dir": str(bundle),
                    "created_at": "2026-07-21T00:00:00+00:00",
                    "owner_pid": 999_999_999,
                    "process_pid": os.getpid(),
                    "process_identity": JobService._process_identity(os.getpid()),
                }
                (jobs / f"{job_id}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
                service = JobService(jobs)
                observed = service.get(job_id)
                self.assertEqual(observed["state"], state)
                self.assertEqual(observed["owner_status"], "orphan-child")
                self.assertFalse(observed["cancellable"])
                with self.assertRaisesRegex(ValueError, "active job"):
                    service.submit(bundle, action="preflight")

    def test_cancel_fails_closed_when_exited_process_has_no_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            job_id = "job_" + "e" * 32
            process = subprocess.Popen(
                [sys.executable, "-c", "pass"],
                text=True,
            )
            self.assertEqual(process.wait(timeout=5), 0)
            service._write(
                {
                    "id": job_id,
                    "job_id": job_id,
                    "state": "running",
                    "bundle_dir": str(root),
                    "return_code": None,
                    "created_at": "2026-07-21T00:00:00+00:00",
                }
            )
            service._processes[job_id] = process
            result = service.cancel(job_id)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["return_code"], 0)
        self.assertIn("verifier is unavailable", result["error"])

    def test_restart_marks_unattachable_running_job_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "jobs"
            root.mkdir()
            job_id = "job_" + "d" * 32
            (root / f"{job_id}.json").write_text(
                json.dumps({"id": job_id, "job_id": job_id, "state": "running"}),
                encoding="utf-8",
            )
            record = JobService(root).get(job_id)
        self.assertEqual(record["state"], "failed")
        self.assertIn("no longer live", record["error"])

    def test_unexpected_worker_exception_persists_terminal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = JobService(root / "jobs")
            with (
                patch.object(service, "_process_identity", return_value="test-process"),
                patch(
                    "aptus.execution.subprocess.Popen",
                    side_effect=RuntimeError("injected launcher failure"),
                ),
            ):
                submitted = service.submit(fake_bundle(root), action="preflight")
                finished = wait_for(service, submitted["id"])
        self.assertEqual(finished["state"], "failed")
        self.assertIn("injected launcher failure", finished["error"])


if __name__ == "__main__":
    unittest.main()
