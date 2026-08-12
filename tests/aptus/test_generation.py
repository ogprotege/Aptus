import importlib.util
import ast
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aptus.domain import Backend, QuantizationLayout, ValidationState
from aptus.generation import (
    _BUNDLE_PROGRAMS,
    _accelerate_config,
    _bundle_program_bytes,
    create_bundle_archive,
    generate_bundle,
    validate_bundle_archive_bytes,
    verify_bundle_archive,
)
from aptus.model_compatibility import (
    current_model_policy_snapshot_bytes,
    current_model_policy_snapshot_sha256,
)
from aptus.planning import plan_training
from aptus.plan_contract import (
    expected_model_architecture_contract,
    mlx_quantized_storage_bytes_for_contract,
    sha256_file,
)
from aptus.profiling import build_hardware_spec, profile_dataset
from aptus.validation import validate_bundle
from tools.generate_cuda_campaign_fixture import (
    DEFAULT_OUTPUT as CUDA_CAMPAIGN_FIXTURE,
    FIXTURE_ID as CUDA_CAMPAIGN_FIXTURE_ID,
    FIXTURE_SEED as CUDA_CAMPAIGN_FIXTURE_SEED,
    GENERATOR_VERSION as CUDA_CAMPAIGN_GENERATOR_VERSION,
    _display_path as cuda_campaign_display_path,
    render_fixture as render_cuda_campaign_fixture,
)

from tests.aptus.helpers import (
    make_plan,
    make_qwen2_runtime_footprint_plan,
    make_qwen3_moe_plan,
    qwen3_moe_quantization_config,
)


def _mlx_model_load_binding(plan: dict) -> dict:
    model = plan["model"]
    total = model["parameters"]
    active = model.get("active_parameters", total)
    sparse = model.get("sparse_layer_count", 0)
    moe = model.get("moe")
    if moe is None:
        routed = active_routed = inactive = 0
        method = "mlx-lm.get_total_parameters.v1"
    else:
        routed = (
            sparse
            * moe["expert_count"]
            * 3
            * model["hidden_size"]
            * moe["expert_intermediate_size"]
        )
        active_routed = routed * moe["experts_per_token"] // moe["expert_count"]
        inactive = routed - active_routed
        method = "mlx-lm.get_total_parameters-plus-exact-qwen3-moe-routing.v1"
    census = {
        "schema_version": "aptus.mlx-model-parameter-census.v1",
        "census_method": method,
        "declared_total_parameters": total,
        "observed_total_parameters": total,
        "total_parameter_delta": 0,
        "total_parameter_tolerance": max(1_000_000, round(total * 0.02)),
        "declared_active_parameters": active,
        "observed_active_parameters": active,
        "sparse_layer_count": sparse,
        "routed_expert_parameters": routed,
        "active_routed_expert_parameters": active_routed,
        "inactive_expert_parameters": inactive,
    }
    census["descriptor_sha256"] = hashlib.sha256(
        json.dumps(census, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if plan["recommended"]["method"] == "qlora":
        expected_weight_bytes, expected_metadata_bytes = (
            mlx_quantized_storage_bytes_for_contract(model, logical_parameters=total)
        )
    else:
        expected_weight_bytes = round(total * 2.0)
        expected_metadata_bytes = 0
    expected_packed_bytes = expected_weight_bytes + expected_metadata_bytes
    observed_safetensors_bytes = expected_packed_bytes + 4096
    packed = {
        "schema_version": "aptus.mlx-packed-checkpoint.v1",
        "observed_safetensors_bytes": observed_safetensors_bytes,
        "observed_logical_parameters": total,
        "expected_weight_bytes": expected_weight_bytes,
        "expected_quantization_metadata_bytes": expected_metadata_bytes,
        "expected_packed_tensor_bytes": expected_packed_bytes,
        "container_overhead_bytes": 4096,
        "container_overhead_limit_bytes": max(
            1024**2, round(expected_packed_bytes * 0.0001)
        ),
    }
    packed["descriptor_sha256"] = hashlib.sha256(
        json.dumps(packed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
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
    binding["descriptor_sha256"] = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return binding


def _mlx_unified_memory_admission(plan: dict) -> dict:
    memory = plan["recommended"]["memory"]
    planned_resident = (
        memory["base_weights_bytes"] + memory["quantization_metadata_bytes"]
    )
    observed = _mlx_model_load_binding(plan)["packed_checkpoint_binding"][
        "observed_safetensors_bytes"
    ]
    adjustment = max(0, observed - planned_resident)
    point = memory["point_estimate_bytes"] + adjustment
    upper = memory["upper_estimate_bytes"] + adjustment
    reserve = max(plan["hardware"].get("reserve_per_device_bytes", 0), 8 * 1024**3)
    required = max(point, upper) + reserve
    return {
        "schema_version": "aptus.mlx-unified-memory-admission.v2",
        "available_unified_memory_bytes": required + 1,
        "planned_resident_bytes": planned_resident,
        "observed_safetensors_bytes": observed,
        "resident_adjustment_bytes": adjustment,
        "adjusted_point_estimate_bytes": point,
        "adjusted_upper_estimate_bytes": upper,
        "reserve_bytes": reserve,
        "required_available_bytes": required,
    }


class FakeTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        values = [10 + (ord(character) % 50) for character in text]
        return ([1] if add_special_tokens else []) + values

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return (
            "".join(f"<{item['role']}>{item['content']}" for item in messages)
            + "<assistant>"
        )


class NonPrefixChatTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prefix = "prompt:" if add_generation_prompt else "full:"
        return prefix + super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )


class FakePretrainedLoader:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class FakeSafeTensorFile:
    def __init__(self, keys):
        self._keys = keys

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def keys(self):
        return list(self._keys)


class FakeModelParameter:
    def __init__(self, count: int, *, quantized_shape: tuple[int, ...] | None = None):
        self._count = count
        self.quant_state = (
            types.SimpleNamespace(shape=quantized_shape)
            if quantized_shape is not None
            else None
        )

    def numel(self):
        return self._count


class FakeCensusParameter:
    def __init__(
        self,
        count: int,
        *,
        shape: tuple[int, ...],
        dtype: str = "fixture-float32",
        requires_grad: bool = True,
        finite: bool = True,
    ):
        self._count = count
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.finite = finite

    def numel(self):
        return self._count

    def detach(self):
        return self


class FakeCensusModel:
    def __init__(self, parameters):
        self._parameters = tuple(parameters)

    def named_parameters(self):
        return self._parameters


class FakeFiniteResult:
    def __init__(self, value: bool):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class FakeLoadedModel:
    def __init__(
        self,
        parameter: FakeModelParameter,
        *,
        module_names: tuple[str, ...],
    ):
        self._parameter = parameter
        self._module_names = module_names
        self.mutations = []

    def parameters(self):
        return (self._parameter,)

    def named_modules(self):
        return tuple((name, object()) for name in self._module_names)

    def train(self, *args, **kwargs):
        self.mutations.append(("train", args, kwargs))
        raise AssertionError("model-data validation must not enter training mode")

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.mutations.append(("gradient_checkpointing_enable", args, kwargs))
        raise AssertionError("model-data validation must not mutate checkpointing")


class BundleGenerationTests(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        output = root / "bundle"
        report = generate_bundle(make_plan(root), output)
        self.assertEqual(report.state, ValidationState.STATIC_PASS)
        return output

    def _run_package_free_static_validation(
        self, output: Path
    ) -> subprocess.CompletedProcess[str]:
        return self._run_package_free_entrypoint(
            output, "validate.py", "--level", "static"
        )

    def _run_package_free_entrypoint(
        self, output: Path, relative: str, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        return subprocess.run(
            [sys.executable, "-S", str(output / relative), *arguments],
            cwd=output,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def _refresh_manifested_file(self, output: Path, relative: str) -> None:
        path = output / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path = output / "bundle-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(item for item in manifest["files"] if item["path"] == relative)
        entry["sha256"] = digest
        entry["size_bytes"] = path.stat().st_size
        if relative == "plan.json":
            manifest["plan_sha256"] = digest
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def _load_generated(self, output: Path, name: str):
        return self._load_generated_path(output, "train.py", name)

    def _load_generated_path(self, output: Path, relative: str, name: str):
        spec = importlib.util.spec_from_file_location(name, output / relative)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(output))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(output))
            sys.dont_write_bytecode = previous
        return module

    def _mlx_plan(self, root: Path, *, dataset_content: str | None = None):
        base = make_plan(root)
        dataset = base.dataset
        if dataset_content is not None:
            dataset_path = root / "mlx-source.jsonl"
            dataset_path.write_text(dataset_content, encoding="utf-8")
            dataset = profile_dataset(
                dataset_path, sample_limit=64, sequence_length=128
            )
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
        return plan_training(
            model=base.model,
            dataset=dataset,
            hardware=hardware,
            target=base.target,
        )

    def _mlx_bundle(self, root: Path) -> Path:
        output = root / "mlx-bundle"
        report = generate_bundle(self._mlx_plan(root), output)
        self.assertEqual(report.state, ValidationState.STATIC_PASS)
        return output

    def _run_model_data_fixture(
        self,
        *,
        method: str,
        expected_parameters: int,
        actual_parameters: int,
        target_modules: tuple[str, ...],
        module_names: tuple[str, ...],
        quantized: bool = False,
        config_values: dict[str, int | None] | None = None,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(
                output, f"aptus_generated_model_data_{method.replace('-', '_')}"
            )
            trainer_config = json.loads(
                (output / "config" / "trainer.json").read_text(encoding="utf-8")
            )
            plan = {
                "recommended": {
                    "method": method,
                    "precision": "bf16",
                    "distribution": "single",
                    "target_modules": list(target_modules),
                },
                "model": {
                    "model_id": "example/model-fixture",
                    "revision": "a" * 40,
                    "tokenizer_id": "example/tokenizer-fixture",
                    "parameters": expected_parameters,
                    "hidden_size": 2048,
                    "intermediate_size": 8192,
                    "layers": 24,
                    "context_length": 4096,
                },
            }
            parameter = FakeModelParameter(
                1 if quantized else actual_parameters,
                quantized_shape=(actual_parameters,) if quantized else None,
            )
            loaded_model = FakeLoadedModel(parameter, module_names=module_names)
            calls = {}
            encoded_rows = []
            cleanup_calls = []
            preparation_events = []
            expected_target_matches = (
                0
                if method == "full"
                else sum(
                    1
                    for name in module_names
                    if any(
                        name == target or name.endswith("." + target)
                        for target in target_modules
                    )
                )
            )
            census = {
                "schema_version": "aptus.trainable-parameter-census.v1",
                "method": method,
                "parameter_scope": (
                    "all-parameters" if method == "full" else "lora-adapter-only"
                ),
                "trainable_parameter_count": 8192,
                "trainable_tensor_count": 2,
                "frozen_parameter_count": 0 if method == "full" else actual_parameters,
                "frozen_tensor_count": 0 if method == "full" else 1,
                "unexpected_trainable_tensor_count": 0,
                "expected_adapter_target_match_count": expected_target_matches,
                "adapter_target_instance_count": expected_target_matches,
                "incomplete_adapter_target_instance_count": 0,
                "all_values_finite": True,
                "descriptor_sha256": "b" * 64,
            }

            class FakeAutoTokenizer:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    calls["tokenizer"] = (args, kwargs)
                    return FakeTokenizer()

            class FakeAutoConfig:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    calls["config"] = (args, kwargs)
                    values = dict(
                        hidden_size=2048,
                        intermediate_size=8192,
                        num_hidden_layers=24,
                        max_position_embeddings=4096,
                    )
                    values.update(config_values or {})
                    return types.SimpleNamespace(**values)

            class FakeAutoModelForCausalLM:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    calls["model"] = (args, kwargs)
                    return loaded_model

            class FakeBitsAndBytesConfig:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoConfig = FakeAutoConfig
            fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
            fake_transformers.AutoTokenizer = FakeAutoTokenizer
            fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
            fake_torch = types.ModuleType("torch")
            fake_torch.bfloat16 = "fixture-bf16"
            fake_torch.float16 = "fixture-fp16"
            fake_torch.cuda = types.SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: cleanup_calls.append("empty_cache"),
            )
            fake_bitsandbytes = types.ModuleType("bitsandbytes")

            with (
                patch.dict(
                    sys.modules,
                    {
                        "bitsandbytes": fake_bitsandbytes,
                        "torch": fake_torch,
                        "transformers": fake_transformers,
                    },
                ),
                patch.object(module, "initialize_and_require_strategy"),
                patch.object(module, "require_hardware_parity"),
                patch.object(
                    module,
                    "prepare_model_for_training",
                    side_effect=lambda model, _plan: (
                        preparation_events.append("prepare") or model
                    ),
                ),
                patch.object(
                    module,
                    "require_trainable_parameter_census",
                    side_effect=lambda _model, **_kwargs: (
                        preparation_events.append("census") or census
                    ),
                ),
                patch.object(
                    module,
                    "encode_record",
                    side_effect=lambda row, *_args: encoded_rows.append(row),
                ),
            ):
                observed_census = module.model_data_preflight(
                    plan, trainer_config, local_files_only=True
                )
            return (
                calls,
                encoded_rows,
                cleanup_calls,
                loaded_model,
                preparation_events,
                observed_census,
            )

    def test_compiles_canonical_portable_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root)
            output = root / "bundle"
            report = generate_bundle(plan, output)
            files = {
                item.relative_to(output).as_posix()
                for item in output.rglob("*")
                if item.is_file()
            }
        required = {
            "plan.json",
            "policy_snapshot.py",
            "policy/model-policy-snapshot.v1.json",
            "profiles/model.json",
            "profiles/dataset.json",
            "profiles/hardware.json",
            "candidates.json",
            "evidence.jsonl",
            "decision-report.md",
            "bundle-manifest.json",
            "requirements.txt",
            "config/accelerate.yaml",
            "config/trainer.json",
            "campaign_events.py",
            "train.py",
            "preflight.py",
            "runtime_lease.py",
            "validate.py",
            "runbook.md",
            "run.py",
            "data/dataset.jsonl",
            "data/pilot-sample.jsonl",
            "data/training.jsonl",
        }
        self.assertTrue(required <= files)
        self.assertEqual(report.state, ValidationState.STATIC_PASS)

    def test_bundle_binds_phase3_controls_and_counter_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bundle"
            plan = make_plan(
                root, optimizer_steps=128, training_seed=101, data_order_seed=1_000_101
            )
            generate_bundle(plan, output)
            trainer = json.loads(
                (output / "config" / "trainer.json").read_text(encoding="utf-8")
            )
            module = self._load_generated(output, "aptus_generated_phase3")

        self.assertEqual(trainer["schema_version"], "aptus.trainer-config.v3")
        self.assertEqual(trainer["optimizer_steps"], 128)
        self.assertEqual(
            (
                trainer["split_seed"],
                trainer["training_seed"],
                trainer["data_order_seed"],
            ),
            (424242, 101, 1_000_101),
        )
        self.assertEqual(
            trainer["counter_contract"]["schema_version"],
            "aptus.training-counters.v1",
        )
        self.assertEqual(
            module.resolve_max_steps(pilot=False, max_steps=None, optimizer_steps=128),
            128,
        )

    def test_bundle_binds_canonical_policy_snapshot_across_plan_and_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bundle"
            generate_bundle(make_plan(root), output)
            snapshot_path = output / "policy/model-policy-snapshot.v1.json"
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (output / "bundle-manifest.json").read_text(encoding="utf-8")
            )
            entries = {item["path"]: item for item in manifest["files"]}
            snapshot_bytes = snapshot_path.read_bytes()

        expected_bytes = current_model_policy_snapshot_bytes()
        expected_digest = current_model_policy_snapshot_sha256()
        self.assertEqual(snapshot_bytes, expected_bytes)
        self.assertEqual(manifest["schema_version"], "aptus.bundle.v3")
        self.assertEqual(
            manifest["policy_snapshot_path"],
            "policy/model-policy-snapshot.v1.json",
        )
        self.assertEqual(manifest["policy_snapshot_sha256"], expected_digest)
        self.assertEqual(plan["model_policy_snapshot_sha256"], expected_digest)
        self.assertEqual(
            entries[manifest["policy_snapshot_path"]]["sha256"], expected_digest
        )

    def test_compilation_rejects_plan_bound_to_another_policy_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = replace(make_plan(root), model_policy_snapshot_sha256="0" * 64)

            with self.assertRaisesRegex(ValueError, "current host policy"):
                generate_bundle(plan, root / "bundle")

    def test_compiles_honest_mlx_bundle_slice_without_cuda_assumptions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            plan = plan_training(
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )
            output = root / "mlx-bundle"
            report = generate_bundle(plan, output)
            requirements = (output / "requirements.txt").read_text(encoding="utf-8")
            train_source = (output / "train.py").read_text(encoding="utf-8")
            validate_source = (output / "validate.py").read_text(encoding="utf-8")
            run_source = (output / "run.py").read_text(encoding="utf-8")
            reload_source = (output / "reload.py").read_text(encoding="utf-8")
            readme = (output / "README.md").read_text(encoding="utf-8")
            runbook = (output / "runbook.md").read_text(encoding="utf-8")
            config = (output / "config" / "mlx-lm.yaml").read_text(encoding="utf-8")
            trainer_config = json.loads(
                (output / "config" / "trainer.json").read_text(encoding="utf-8")
            )
            decision_report = (output / "decision-report.md").read_text(
                encoding="utf-8"
            )
            files = {
                item.relative_to(output).as_posix()
                for item in output.rglob("*")
                if item.is_file()
            }
            compiled_rows = []
            for name in ("train.jsonl", "valid.jsonl"):
                compiled_rows.extend(
                    json.loads(line)
                    for line in (output / "data" / "mlx" / name)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                )
            split_contract = json.loads(
                (output / "data" / "mlx" / "split-contract.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(report.state, ValidationState.STATIC_PASS)
        self.assertEqual(requirements, "mlx==0.31.2\nmlx-lm==0.31.3\n")
        self.assertNotIn("bitsandbytes", requirements)
        self.assertIn("measured_peak_bytes", train_source)
        self.assertNotIn("measured_peak_cuda_bytes", train_source)
        self.assertIn('"execution_semantics": "uninterrupted"', train_source)
        self.assertIn('"resume_supported": False', train_source)
        self.assertIn("load_pinned_local_model", train_source)
        self.assertIn('{"trust_remote_code": False}', train_source)
        self.assertIn("--confirm-full-train", run_source)
        self.assertIn("--output-dir", run_source)
        self.assertIn("fresh_process_observed", reload_source)
        self.assertNotIn("current pilot fails", readme.lower())
        self.assertNotIn("current pilot fails", runbook.lower())
        self.assertIn("`aptus.bundle.v3`", readme)
        self.assertIn("`aptus.training-plan.v6`", readme)
        self.assertIn("`policy/model-policy-snapshot.v1.json`", readme)
        self.assertIn(current_model_policy_snapshot_sha256(), readme)
        self.assertIn("proves only the integrity of the frozen policy", readme)
        self.assertIn("Installed Aptus checks its current registry", readme)
        self.assertIn("`replan_required`", readme)
        self.assertIn("runs uninterrupted", runbook)
        self.assertIn("measured-preflight", validate_source)
        self.assertIn("num_layers: -1", config)
        self.assertIn("mask_prompt: true", config)
        self.assertIn("scale: 2.0", config)
        self.assertIn(
            'keys: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]',
            config,
        )
        self.assertEqual(trainer_config["optimizer"], "adamw")
        self.assertIsNone(trainer_config["lr_scheduler_type"])
        self.assertNotIn("adamw_torch", decision_report)
        self.assertTrue(
            {
                "data/mlx/train.jsonl",
                "data/mlx/valid.jsonl",
                "data/mlx/split-contract.json",
                "config/mlx-lm.yaml",
                "reload.py",
            }
            <= files
        )
        for source in (train_source, validate_source, run_source, reload_source):
            ast.parse(source)

        micro_batch = plan.recommended.micro_batch_size
        self.assertEqual(
            len(compiled_rows),
            sum(
                value["compiled_row_count"]
                for value in split_contract["splits"].values()
            ),
        )
        self.assertEqual(
            sum(
                value["source_row_count"] for value in split_contract["splits"].values()
            ),
            2,
        )
        self.assertTrue(
            all(
                value["compiled_row_count"] >= micro_batch
                and value["compiled_row_count"] % micro_batch == 0
                for value in split_contract["splits"].values()
            )
        )
        self.assertTrue(all(set(row) == {"messages"} for row in compiled_rows))
        self.assertTrue(
            all(row["messages"][-1]["role"] == "assistant" for row in compiled_rows)
        )

    def test_compiles_dense_mlx_bundle_with_explicit_empty_override_layout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = make_plan(root)
            model = replace(
                base.model,
                parameters=494_000_000,
                quantization_bits=4,
                quantization_layout=QuantizationLayout(
                    default_bits=4,
                    default_group_size=64,
                ),
            )
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
            plan = plan_training(
                model=model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )
            output = root / "dense-quantized-mlx-bundle"
            report = generate_bundle(plan, output)
            payload = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            portable_contract = self._load_generated_path(
                output,
                "plan_contract.py",
                "aptus_generated_dense_quantized_plan_contract",
            )
            runtime = self._load_generated(
                output,
                "aptus_generated_dense_quantized_runtime",
            )
            model_load_binding = _mlx_model_load_binding(payload)

        self.assertEqual(report.state, ValidationState.STATIC_PASS)
        self.assertEqual(payload["recommended"]["method"], "qlora")
        self.assertEqual(
            payload["model"]["quantization_layout"],
            {
                "default_bits": 4,
                "default_group_size": 64,
                "module_overrides": [],
            },
        )
        self.assertEqual(
            portable_contract.mlx_quantized_storage_bytes_for_contract(
                payload["model"]
            ),
            (247_000_000, 30_875_000),
        )
        self.assertEqual(
            payload["recommended"]["memory"]["base_weights_bytes"],
            247_000_000,
        )
        self.assertEqual(
            payload["recommended"]["memory"]["quantization_metadata_bytes"],
            30_875_000,
        )
        self.assertEqual(
            model_load_binding["packed_checkpoint_binding"]["expected_weight_bytes"],
            247_000_000,
        )
        self.assertEqual(
            model_load_binding["packed_checkpoint_binding"][
                "expected_quantization_metadata_bytes"
            ],
            30_875_000,
        )
        self.assertIs(
            runtime.require_mlx_model_load_binding(payload, model_load_binding),
            model_load_binding,
        )

    def test_qwen3_moe_bundle_documents_attention_only_resident_base_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "qwen3-moe-bundle"
            report = generate_bundle(make_qwen3_moe_plan(root), output)
            config = (output / "config" / "mlx-lm.yaml").read_text(encoding="utf-8")
            readme = (output / "README.md").read_text(encoding="utf-8")
            runbook = (output / "runbook.md").read_text(encoding="utf-8")
            decision = (output / "decision-report.md").read_text(encoding="utf-8")

        self.assertEqual(report.state, ValidationState.STATIC_PASS)
        self.assertIn("attention projections only", config)
        self.assertIn('keys: ["q_proj", "k_proj", "v_proj", "o_proj"]', config)
        self.assertIn("Expert and router weights remain frozen", config)
        self.assertIn("full quantized base still resides", readme)
        self.assertIn("logical active parameters", runbook)
        self.assertIn("MoE adapter policy: attention-only QLoRA", decision)

    def test_qwen2_runtime_footprint_compiles_and_binds_dense_model_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "qwen2-runtime-footprint-bundle"
            report = generate_bundle(
                make_qwen2_runtime_footprint_plan(root),
                output,
            )
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            runtime = self._load_generated(
                output,
                "aptus_generated_qwen2_runtime_footprint",
            )
            loaded_plan, candidate = runtime.load_contract()
            model_path = root / "model"
            model_path.mkdir()
            config = {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 896,
                "intermediate_size": 4864,
                "num_hidden_layers": 24,
                "max_position_embeddings": 32768,
                "quantization": {"bits": 4, "group_size": 64},
            }
            (model_path / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )

            runtime.require_method_model(loaded_plan, candidate, model_path)
            config["num_hidden_layers"] = 25
            (model_path / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "architecture"):
                runtime.require_method_model(loaded_plan, candidate, model_path)

            config["num_hidden_layers"] = 24
            config["quantization"]["group_size"] = 128
            (model_path / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "architecture"):
                runtime.require_method_model(loaded_plan, candidate, model_path)

            config["quantization"]["group_size"] = 64
            config.update(
                {
                    "num_experts": 8,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 1024,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                }
            )
            (model_path / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "architecture"):
                runtime.require_method_model(loaded_plan, candidate, model_path)

            dense_layers = [
                types.SimpleNamespace(mlp=types.SimpleNamespace()) for _ in range(24)
            ]
            dense_layers[5] = types.SimpleNamespace(
                mlp=types.SimpleNamespace(
                    num_experts=8,
                    top_k=2,
                    switch_mlp=object(),
                )
            )
            loaded_model = types.SimpleNamespace(layers=dense_layers)
            with self.assertRaisesRegex(RuntimeError, "requires dense topology"):
                runtime.build_mlx_model_load_binding(
                    loaded_model,
                    loaded_plan,
                    observed_safetensors_bytes=0,
                    parameter_counter=lambda value: loaded_plan["model"]["parameters"],
                )

        self.assertEqual(report.state, ValidationState.STATIC_PASS)
        self.assertEqual(
            plan["model_policy_decision"]["policy_id"],
            "model.qwen2-24l.mlx-qlora",
        )
        self.assertEqual(
            plan["recommended"]["policy_binding"]["path_id"],
            "mlx-lm.qlora.single.dense-causal-lm.v1",
        )
        self.assertEqual(
            plan["recommended"]["target_modules"],
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    def test_generated_plan_contract_recomputes_qwen3_moe_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "qwen3-moe-bundle"
            generate_bundle(make_qwen3_moe_plan(root), output)
            module = self._load_generated_path(
                output, "plan_contract.py", "aptus_generated_plan_contract"
            )
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))

        candidate = next(
            item
            for item in plan["candidates"]
            if item["method"] == "qlora" and item["distribution"] == "single"
        )
        memory = candidate["memory"]
        point_delta = memory["activations_bytes"]
        upper_delta = memory["component_upper_bounds"]["activations_bytes"]
        memory["activations_bytes"] = 0
        memory["point_estimate_bytes"] -= point_delta
        memory["estimated_peak_bytes"] -= point_delta
        memory["component_upper_bounds"]["activations_bytes"] = 0
        memory["upper_estimate_bytes"] -= upper_delta
        candidate["candidate_id"] = module.candidate_id_for_payload(
            candidate,
            model=plan["model"],
            dataset=plan["dataset"],
            hardware=plan["hardware"],
            target=plan["target"],
        )
        plan["recommended"] = copy.deepcopy(candidate)
        plan["plan_id"] = module.plan_id_for_payload(plan)

        errors = module.validate_plan_payload(plan, verify_dataset=False)

        self.assertTrue(
            any("deterministic recomputation" in item for item in errors), errors
        )

    def test_generated_mlx_binds_exact_qwen3_moe_quantization_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "qwen3-moe-bundle"
            generate_bundle(make_qwen3_moe_plan(root), output)
            module = self._load_generated(
                output, "aptus_generated_mlx_quantization_layout"
            )
            plan, candidate = module.load_contract()
            model_path = root / "model"
            model_path.mkdir()
            config = {
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 48,
                "max_position_embeddings": 262144,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 768,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "quantization": qwen3_moe_quantization_config(),
            }
            (model_path / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )

            module.require_method_model(plan, candidate, model_path)
            config["quantization"]["model.layers.0.mlp.gate"]["bits"] = 4
            (model_path / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "architecture"):
                module.require_method_model(plan, candidate, model_path)

    def test_packaged_program_resources_match_generated_files_and_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = {
                "cuda": self._bundle(root),
                "mlx": self._mlx_bundle(root),
            }
            for runtime, output in bundles.items():
                manifest = json.loads(
                    (output / "bundle-manifest.json").read_text(encoding="utf-8")
                )
                manifest_files = {entry["path"]: entry for entry in manifest["files"]}
                for name in _BUNDLE_PROGRAMS[runtime]:
                    resource = _bundle_program_bytes(runtime, name)
                    generated = (output / name).read_bytes()
                    self.assertEqual(generated, resource)
                    self.assertEqual(
                        manifest_files[name]["sha256"],
                        hashlib.sha256(resource).hexdigest(),
                    )
                    self.assertEqual(manifest_files[name]["size_bytes"], len(resource))

    def test_generated_mlx_binding_proves_every_planned_lora_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_mlx_binding")
            plan = module.load_contract()[0]
            candidate = plan["recommended"]

            class FakeLayer:
                def named_modules(self):
                    return [
                        ("self_attn.q_proj", object()),
                        ("self_attn.k_proj", object()),
                        ("self_attn.v_proj", object()),
                        ("self_attn.o_proj", object()),
                        ("mlp.gate_proj", object()),
                        ("mlp.up_proj", object()),
                        ("mlp.down_proj", object()),
                    ]

            model = types.SimpleNamespace(
                layers=[FakeLayer() for _index in range(plan["model"]["layers"])]
            )
            resolved, binding = module.resolve_lora_keys(model, candidate)
            names = [
                f"model.layers.{layer_index}.{key}.{suffix}"
                for layer_index in range(plan["model"]["layers"])
                for key in resolved
                for suffix in ("lora_a", "lora_b")
            ]
            census = module.require_trainable_binding(names, binding)

            self.assertEqual(
                census["planned_target_modules"], candidate["target_modules"]
            )
            self.assertEqual(
                census["adapter_target_instance_count"],
                plan["model"]["layers"] * len(candidate["target_modules"]),
            )
            self.assertEqual(census["trainable_tensor_count"], len(names))
            with self.assertRaisesRegex(RuntimeError, "non-LoRA"):
                module.require_trainable_binding(
                    [*names, "model.layers.0.norm.weight"], binding
                )

    def test_generated_mlx_iteration_schedule_is_bounded_and_epoch_derived(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            module = self._load_generated(
                output, "aptus_generated_mlx_iteration_schedule"
            )
            plan, candidate = module.load_contract()
            micro_batch = candidate["micro_batch_size"]
            accumulation = candidate["gradient_accumulation_steps"]
            train_examples = micro_batch * 3 - 1
            full_iterations = module.derive_iterations(
                action="full",
                requested_iterations=999,
                candidate=candidate,
                plan=plan,
                train_examples=train_examples,
            )
            complete_batches = train_examples // micro_batch
            epoch_iterations = complete_batches * plan["target"]["max_epochs"]
            expected = math.ceil(epoch_iterations / accumulation) * accumulation

            self.assertEqual(full_iterations, expected)
            self.assertEqual(
                module.derive_iterations(
                    action="pilot",
                    requested_iterations=999,
                    candidate=candidate,
                    plan=plan,
                    train_examples=train_examples,
                ),
                2 * accumulation,
            )
            with self.assertRaisesRegex(RuntimeError, "no complete micro-batch"):
                module.derive_iterations(
                    action="full",
                    requested_iterations=1,
                    candidate=candidate,
                    plan=plan,
                    train_examples=micro_batch - 1,
                )

    def test_generated_mlx_model_loader_forces_safe_pinned_local_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_mlx_safe_model_load")
            plan = module.load_contract()[0]
            if plan["recommended"]["method"] == "qlora":
                plan = json.loads(json.dumps(plan))
                plan["model"]["quantization_bits"] = 4
            model_path = Path(temporary) / "pinned-model"
            model_path.mkdir()
            model_config = {
                "hidden_size": plan["model"]["hidden_size"],
                "num_hidden_layers": plan["model"]["layers"],
                "max_position_embeddings": plan["model"]["context_length"],
            }
            if plan["model"].get("intermediate_size") is not None:
                model_config["intermediate_size"] = plan["model"]["intermediate_size"]
            if plan["model"].get("quantization_bits") is not None:
                model_config["quantization_config"] = {
                    "bits": plan["model"]["quantization_bits"]
                }
            (model_path / "config.json").write_text(
                json.dumps(model_config), encoding="utf-8"
            )
            calls = []
            observed_safetensors_bytes = _mlx_model_load_binding(plan)[
                "packed_checkpoint_binding"
            ]["observed_safetensors_bytes"]
            loaded_model = types.SimpleNamespace(
                layers=[object() for _ in range(plan["model"]["layers"])]
            )

            def loader(*args, **kwargs):
                calls.append((args, kwargs))
                return loaded_model, object()

            loaded, binding = module.load_pinned_local_model(
                loader,
                str(model_path),
                model_path=model_path,
                plan=plan,
                observed_safetensors_bytes=observed_safetensors_bytes,
                parameter_counter=lambda _model: plan["model"]["parameters"],
                tokenizer_config={"trust_remote_code": True},
            )
            self.assertEqual(len(loaded), 2)
            self.assertEqual(
                calls,
                [
                    (
                        (str(model_path.resolve()),),
                        {"tokenizer_config": {"trust_remote_code": False}},
                    )
                ],
            )
            self.assertIs(binding["resolved_local_snapshot"], True)
            self.assertIs(binding["trust_remote_code"], False)

            other_path = Path(temporary) / "other-model"
            other_path.mkdir()
            with self.assertRaisesRegex(RuntimeError, "unbound loader arguments"):
                module.load_pinned_local_model(
                    loader,
                    str(other_path),
                    model_path=model_path,
                    plan=plan,
                    observed_safetensors_bytes=observed_safetensors_bytes,
                    tokenizer_config={"trust_remote_code": True},
                )
            with self.assertRaisesRegex(RuntimeError, "unbound loader arguments"):
                module.load_pinned_local_model(
                    loader,
                    str(model_path),
                    model_path=model_path,
                    plan=plan,
                    observed_safetensors_bytes=observed_safetensors_bytes,
                    tokenizer_config={"trust_remote_code": True},
                    lazy=True,
                )
            with self.assertRaisesRegex(RuntimeError, "unbound loader arguments"):
                module.load_pinned_local_model(
                    loader,
                    str(model_path),
                    model_path=model_path,
                    plan=plan,
                    observed_safetensors_bytes=observed_safetensors_bytes,
                    tokenizer_config={"trust_remote_code": False},
                )

    def test_generated_mlx_census_proves_moe_topology_and_rejects_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_mlx_moe_census")
            hidden_size = 4
            expert_count = 4
            experts_per_token = 2
            expert_intermediate_size = 3
            layer_count = 2
            per_layer = expert_count * 3 * hidden_size * expert_intermediate_size
            total = 10_000_000
            inactive = per_layer * layer_count // 2
            model_contract = {
                "model_id": "mlx-community/Qwen3-MoE-fixture-4bit",
                "revision": "a" * 40,
                "family": "qwen3_moe",
                "parameters": total,
                "hidden_size": hidden_size,
                "intermediate_size": 12,
                "layers": layer_count,
                "context_length": 128,
                "architecture": "Qwen3MoeForCausalLM",
                "model_type": "qwen3_moe",
                "quantization_bits": 4,
                "quantization_layout": {
                    "default_bits": 4,
                    "default_group_size": 64,
                    "module_overrides": [
                        {
                            "module_path": f"model.layers.{index}.mlp.gate",
                            "bits": 8,
                            "group_size": 64,
                        }
                        for index in range(layer_count)
                    ],
                },
                "moe": {
                    "expert_count": expert_count,
                    "experts_per_token": experts_per_token,
                    "expert_intermediate_size": expert_intermediate_size,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                    "shared_expert_intermediate_size": None,
                },
                "sparse_layer_count": layer_count,
                "active_parameters": total - inactive,
            }
            switches = [object() for _ in range(layer_count)]
            layers = [
                types.SimpleNamespace(
                    mlp=types.SimpleNamespace(
                        num_experts=expert_count,
                        top_k=experts_per_token,
                        switch_mlp=switch,
                    )
                )
                for switch in switches
            ]
            loaded_model = types.SimpleNamespace(layers=layers)

            def count_parameters(value):
                return total if value is loaded_model else per_layer

            plan = {"model": model_contract, "recommended": {"method": "qlora"}}
            expected_weight_bytes, expected_metadata_bytes = (
                mlx_quantized_storage_bytes_for_contract(model_contract)
            )
            observed_safetensors_bytes = (
                expected_weight_bytes + expected_metadata_bytes + 4096
            )
            binding = module.build_mlx_model_load_binding(
                loaded_model,
                plan,
                observed_safetensors_bytes=observed_safetensors_bytes,
                parameter_counter=count_parameters,
            )
            census = binding["parameter_census"]
            self.assertEqual(
                binding["schema_version"], "aptus.mlx-model-load-binding.v3"
            )
            self.assertEqual(census["observed_active_parameters"], total - inactive)
            self.assertEqual(census["routed_expert_parameters"], per_layer * 2)
            self.assertEqual(
                binding["packed_checkpoint_binding"]["observed_safetensors_bytes"],
                observed_safetensors_bytes,
            )
            self.assertIs(module.require_mlx_model_load_binding(plan, binding), binding)

            forged_packed = json.loads(json.dumps(binding))
            packed = forged_packed["packed_checkpoint_binding"]
            packed["expected_weight_bytes"] += 1
            packed["expected_packed_tensor_bytes"] += 1
            packed["container_overhead_bytes"] -= 1
            packed["descriptor_sha256"] = module._descriptor_sha256(
                {
                    key: value
                    for key, value in packed.items()
                    if key != "descriptor_sha256"
                }
            )
            forged_packed["descriptor_sha256"] = module._descriptor_sha256(
                {
                    key: value
                    for key, value in forged_packed.items()
                    if key != "descriptor_sha256"
                }
            )
            with self.assertRaisesRegex(RuntimeError, "logical parameter census"):
                module.require_mlx_model_load_binding(plan, forged_packed)

            tampered = json.loads(json.dumps(binding))
            tampered["parameter_census"]["observed_active_parameters"] += 1
            tampered["parameter_census"]["descriptor_sha256"] = (
                module._descriptor_sha256(
                    {
                        key: value
                        for key, value in tampered["parameter_census"].items()
                        if key != "descriptor_sha256"
                    }
                )
            )
            tampered["descriptor_sha256"] = module._descriptor_sha256(
                {
                    key: value
                    for key, value in tampered.items()
                    if key != "descriptor_sha256"
                }
            )
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                module.require_mlx_model_load_binding(plan, tampered)

            with self.assertRaisesRegex(RuntimeError, "two-percent"):
                module.build_mlx_model_load_binding(
                    loaded_model,
                    plan,
                    observed_safetensors_bytes=observed_safetensors_bytes,
                    parameter_counter=lambda value: (
                        total + 1_000_001 if value is loaded_model else per_layer
                    ),
                )

    def test_generated_mlx_rejects_method_model_quantization_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_mlx_method_model")
            plan = module.load_contract()[0]
            quantized_plan = json.loads(json.dumps(plan))
            quantized_plan["model"]["quantization_bits"] = 4
            unquantized = Path(temporary) / "unquantized"
            quantized = Path(temporary) / "quantized"
            custom_model = Path(temporary) / "custom-model"
            unquantized.mkdir()
            quantized.mkdir()
            custom_model.mkdir()
            base_config = {
                "hidden_size": plan["model"]["hidden_size"],
                "intermediate_size": plan["model"]["intermediate_size"],
                "num_hidden_layers": plan["model"]["layers"],
                "max_position_embeddings": plan["model"]["context_length"],
            }
            (unquantized / "config.json").write_text(
                json.dumps(base_config), encoding="utf-8"
            )
            (quantized / "config.json").write_text(
                json.dumps({**base_config, "quantization_config": {"bits": 4}}),
                encoding="utf-8",
            )
            (custom_model / "config.json").write_text(
                json.dumps({**base_config, "model_file": "model.py"}),
                encoding="utf-8",
            )

            module.require_method_model(plan, {"method": "lora"}, unquantized)
            module.require_method_model(quantized_plan, {"method": "qlora"}, quantized)
            with self.assertRaisesRegex(RuntimeError, "unquantized pinned base"):
                module.require_method_model(
                    quantized_plan, {"method": "lora"}, quantized
                )
            with self.assertRaisesRegex(RuntimeError, "four-bit"):
                module.require_method_model(plan, {"method": "qlora"}, unquantized)
            with self.assertRaisesRegex(RuntimeError, "custom model_file"):
                module.require_method_model(plan, {"method": "lora"}, custom_model)

    def test_generated_mlx_resume_arguments_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            train_module = self._load_generated(
                output, "aptus_generated_mlx_resume_train"
            )
            run_module = self._load_generated_path(
                output, "run.py", "aptus_generated_mlx_resume_run"
            )
            for module, argv in (
                (
                    train_module,
                    [
                        "train.py",
                        "--pilot",
                        "--adapter-path",
                        "pilot-output/rejected",
                        "--resume-from",
                        "checkpoint",
                    ],
                ),
                (
                    run_module,
                    ["run.py", "--pilot", "--resume-from", "checkpoint"],
                ),
            ):
                with (
                    self.subTest(module=module.__name__),
                    patch.object(sys, "argv", argv),
                    self.assertRaises(SystemExit),
                ):
                    module.main()

    def test_generated_mlx_managed_full_defers_only_parent_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            help_result = self._run_package_free_entrypoint(output, "run.py", "--help")
            module = self._load_generated_path(
                output, "run.py", "aptus_generated_mlx_managed_parent"
            )
            plan = module.load_plan()
            run_output = output / "runs" / "run_managed"
            metrics = {"run_completed": True}
            completed = subprocess.CompletedProcess([], 0)
            verified_calls: list[tuple[dict, Path, str]] = []
            fake_validate = types.ModuleType("validate")

            def require_completed_run(
                observed_plan: dict, observed_output: Path, *, action: str
            ) -> dict:
                verified_calls.append((observed_plan, observed_output, action))
                return metrics

            fake_validate.require_completed_run = require_completed_run
            pilot_path = output / "pilot-output" / "metrics.json"
            pilot_path.parent.mkdir(exist_ok=True)
            pilot_path.write_text("{}\n", encoding="utf-8")
            admission = {"pilot_metrics_sha256": module.sha256(pilot_path)}
            report_path = output / "validation-report.json"
            source_report = json.loads(report_path.read_text(encoding="utf-8"))
            source_report.update(
                state="measured-run-pass",
                validation_level="measured-run",
                artifact_fingerprint=module.bundle_fingerprint(output),
                bindings={
                    "bundle": module.bundle_fingerprint(output),
                    "dataset": plan["dataset"]["source_sha256"],
                    "plan_id": plan["plan_id"],
                    "candidate_id": plan["recommended"]["candidate_id"],
                    "model_revision": plan["model"]["revision"],
                    "pilot_metrics": admission["pilot_metrics_sha256"],
                },
                measured_run_completed_at="2026-08-05T12:10:57+00:00",
                final_export={"stale": True},
                measured_run={"stale": True},
                parent_promotion={"schema_version": "aptus.parent-promotion.v1"},
            )
            report_path.write_text(
                json.dumps(source_report, sort_keys=True) + "\n", encoding="utf-8"
            )
            argument_values = {
                "bounded_smoke": False,
                "pilot": False,
                "confirm_full_train": True,
                "resume_from": None,
                "output_dir": run_output,
                "model": None,
                "data": None,
                "iters": 2,
            }

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "run.py",
                        "--confirm-full-train",
                        "--defer-parent-promotion",
                        "--output-dir",
                        str(run_output),
                    ],
                ),
                patch.object(module, "portable_execution_lease"),
                patch.object(module, "launch", return_value=0) as parsed_launch,
            ):
                self.assertEqual(module.main(), 0)
            self.assertTrue(parsed_launch.call_args.args[0].defer_parent_promotion)

            with (
                patch.dict(sys.modules, {"validate": fake_validate}),
                patch.object(module, "load_plan", return_value=plan),
                patch.object(module, "require_full_admission", return_value=admission),
                patch.object(module, "claim_output", return_value=run_output),
                patch.object(module, "run_with_lease", return_value=completed),
                patch.object(module, "finalize", return_value=metrics),
            ):
                with patch.object(module, "promote_full_completion") as promote:
                    result = module.launch(
                        types.SimpleNamespace(
                            **argument_values, defer_parent_promotion=True
                        )
                    )
                    promote.assert_not_called()
                self.assertEqual(result, 0)
                managed_report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(managed_report["state"], "execution-approved")
                self.assertEqual(managed_report["validation_level"], "pilot")
                self.assertEqual(
                    managed_report["active_run"],
                    {
                        "output_dir": str(run_output.resolve()),
                        "run_id": run_output.name,
                        "plan_id": plan["plan_id"],
                        "candidate_id": plan["recommended"]["candidate_id"],
                    },
                )
                self.assertEqual(managed_report["prelaunch_capacity_check"], admission)
                for stale_name in (
                    "measured_run_completed_at",
                    "final_export",
                    "measured_run",
                    "parent_promotion",
                ):
                    self.assertNotIn(stale_name, managed_report)

                with patch.object(module, "promote_full_completion") as promote:
                    result = module.launch(
                        types.SimpleNamespace(
                            **argument_values, defer_parent_promotion=False
                        )
                    )
                    promote.assert_called_once_with(
                        plan, run_output, metrics, admission
                    )

                run_output.mkdir(parents=True)
                (run_output / "final-export.json").write_text("{}\n", encoding="utf-8")
                (run_output / "metrics.json").write_text("{}\n", encoding="utf-8")
                managed_report["parent_promotion"] = {"stale": True}
                report_path.write_text(
                    json.dumps(managed_report, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                module.promote_full_completion(
                    plan,
                    run_output,
                    {
                        "final_export": {"total_bytes": 0},
                        "artifact_manifest_sha256": "a" * 64,
                        "reload_evidence_sha256": "b" * 64,
                        "global_step": 1,
                        "completed_optimizer_updates": 1,
                        "measured_peak_bytes": 1,
                    },
                    admission,
                )
                standalone_report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(standalone_report["state"], "measured-run-pass")
                self.assertNotIn("parent_promotion", standalone_report)

        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertNotIn("--defer-parent-promotion", help_result.stdout)
        self.assertEqual(result, 0)
        self.assertEqual(
            verified_calls,
            [(plan, run_output, "full"), (plan, run_output, "full")],
        )

    def test_generated_mlx_full_refuses_missing_or_stale_pilot_before_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            module = self._load_generated_path(
                output, "run.py", "aptus_generated_mlx_full_gate"
            )
            plan = module.load_plan()
            with self.assertRaisesRegex(RuntimeError, "pilot-pass"):
                module.require_full_admission(plan)

            pilot_path = output / "pilot-output" / "metrics.json"
            pilot_path.parent.mkdir(exist_ok=True)
            pilot_path.write_text("{}", encoding="utf-8")
            report_path = output / "validation-report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update(
                state="pilot-pass",
                pilot_metrics={},
            )
            report["bindings"]["candidate_id"] = "stale-candidate"
            report["bindings"]["pilot_metrics"] = module.sha256(pilot_path)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact current pilot bindings"):
                module.require_full_admission(plan)

    def test_generated_mlx_runtime_metrics_require_loss_update_binding_delta_and_headroom(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            module = self._load_generated_path(
                output, "validate.py", "aptus_generated_mlx_metric_proof"
            )
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            candidate = plan["recommended"]
            layers = plan["model"]["layers"]
            targets = candidate["target_modules"]
            split_contract = json.loads(
                (output / "data" / "mlx" / "split-contract.json").read_text(
                    encoding="utf-8"
                )
            )
            accumulation = candidate["gradient_accumulation_steps"]
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
            binding["descriptor_sha256"] = module.hashlib.sha256(
                json.dumps(binding, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            metrics = {
                "schema_version": "aptus.runtime-metrics.v1",
                "plan_id": plan["plan_id"],
                "candidate_id": candidate["candidate_id"],
                "model_revision": plan["model"]["revision"],
                "dataset_sha256": plan["dataset"]["source_sha256"],
                "method": candidate["method"],
                "training_runtime": "mlx-lm",
                "compute_backend": "mps",
                "compiler_id": candidate["runtime_contract"]["compiler_id"],
                "scope": "bounded-compiler-smoke-not-pilot-evidence",
                "action": "bounded-smoke",
                "execution_semantics": "uninterrupted",
                "resume_supported": False,
                "micro_iterations": accumulation,
                "global_step": accumulation,
                "gradient_accumulation_steps": accumulation,
                "measured_peak_bytes": 4096,
                "active_memory_bytes": 2048,
                "cache_memory_bytes": 1024,
                "memory_metric_backend": "mlx",
                "model_load_binding": _mlx_model_load_binding(plan),
                "finite_train_loss": True,
                "train_loss_observations": [1.25, 1.0],
                "optimizer_update_opportunities": 1,
                "completed_optimizer_updates": 1,
                "optimizer_update_observed": True,
                "train_examples": split_contract["splits"]["train"][
                    "compiled_row_count"
                ],
                "validation_examples": split_contract["splits"]["valid"][
                    "compiled_row_count"
                ],
                "source_train_examples": split_contract["splits"]["train"][
                    "source_row_count"
                ],
                "source_validation_examples": split_contract["splits"]["valid"][
                    "source_row_count"
                ],
                "max_epochs": plan["target"]["max_epochs"],
                "distribution": "single",
                "actual_world_size": 1,
                "finite_validation_loss": True,
                "validation_loss_observations": [1.1],
                "trainable_target_binding": binding,
                "adapter_delta_l1": 0.5,
                "changed_adapter_tensor_count": 7,
                "unified_memory_admission": _mlx_unified_memory_admission(plan),
            }
            self.assertIs(module.require_runtime_metrics(plan, metrics), metrics)
            for name, value, message in (
                ("finite_train_loss", False, "proof scope"),
                ("optimizer_update_observed", False, "proof scope"),
                ("adapter_delta_l1", 0.0, "adapter delta"),
                ("trainable_target_binding", {}, "binding"),
                ("unified_memory_admission", {}, "unified-memory"),
            ):
                with self.subTest(name=name):
                    changed = {**metrics, name: value}
                    with self.assertRaisesRegex(RuntimeError, message):
                        module.require_runtime_metrics(plan, changed)

    def test_generated_mlx_reprobes_live_unified_memory_without_fake_vram(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            module = self._load_generated(
                output, "aptus_generated_mlx_memory_admission"
            )
            plan = module.load_contract()[0]
            model_path = Path(temporary) / "pinned-admission-model"
            model_path.mkdir()
            (model_path / "model.safetensors").write_bytes(b"x")
            vm_stat = (
                "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
                "Pages free:                               5000000.\n"
                "Pages inactive:                                 0.\n"
                "Pages speculative:                              0.\n"
            )
            completed = subprocess.CompletedProcess(
                ["/usr/bin/vm_stat"], 0, stdout=vm_stat, stderr=""
            )
            with patch.object(module.subprocess, "run", return_value=completed):
                admission = module.require_unified_memory_admission(plan, model_path)
            self.assertGreaterEqual(
                admission["available_unified_memory_bytes"],
                admission["required_available_bytes"],
            )
            self.assertEqual(admission["reserve_bytes"], 8 * 1024**3)
            self.assertEqual(admission["observed_safetensors_bytes"], 1)
            self.assertEqual(admission["resident_adjustment_bytes"], 0)
            self.assertNotIn("free_vram_bytes", admission)

            memory = plan["recommended"]["memory"]
            planned_resident = (
                memory["base_weights_bytes"] + memory["quantization_metadata_bytes"]
            )
            observed = planned_resident + 12_345
            adjusted_required = (
                max(
                    memory["point_estimate_bytes"] + 12_345,
                    memory["upper_estimate_bytes"] + 12_345,
                )
                + 8 * 1024**3
            )
            with (
                patch.object(
                    module, "snapshot_safetensors_bytes", return_value=observed
                ),
                patch.object(
                    module,
                    "current_available_unified_memory_bytes",
                    return_value=adjusted_required,
                ),
            ):
                adjusted = module.require_unified_memory_admission(plan, model_path)
            self.assertEqual(adjusted["resident_adjustment_bytes"], 12_345)
            self.assertEqual(adjusted["required_available_bytes"], adjusted_required)

            low_vm_stat = vm_stat.replace("5000000", "1")
            failed = subprocess.CompletedProcess(
                ["/usr/bin/vm_stat"], 0, stdout=low_vm_stat, stderr=""
            )
            with (
                patch.object(module.subprocess, "run", return_value=failed),
                self.assertRaisesRegex(RuntimeError, "8 GiB Aptus reserve") as raised,
            ):
                module.require_unified_memory_admission(plan, model_path)
            required = (
                max(memory["point_estimate_bytes"], memory["upper_estimate_bytes"])
                + 8 * 1024**3
            )
            self.assertIn(f"required={required} bytes", str(raised.exception))
            self.assertIn("available=4096 bytes", str(raised.exception))
            self.assertIn(f"shortfall={required - 4096} bytes", str(raised.exception))

    def test_generated_mlx_model_data_refuses_right_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._mlx_bundle(Path(temporary))
            module = self._load_generated_path(
                output, "validate.py", "aptus_generated_mlx_dataset_contract"
            )
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            plan["model"]["quantization_bits"] = 4
            observed = {}

            class FakeDataset:
                def __init__(self, token_count):
                    self.token_count = token_count

                def __bool__(self):
                    return True

                def __len__(self):
                    return 1

                def __getitem__(self, _index):
                    return {
                        "messages": [
                            {"role": "user", "content": "prompt"},
                            {"role": "assistant", "content": "completion"},
                        ]
                    }

                def process(self, _row):
                    return [1] * self.token_count, 5

            token_count = plan["target"]["sequence_length"] + 1

            def load_dataset(args, _tokenizer):
                observed["mask_prompt"] = args.mask_prompt
                dataset = FakeDataset(token_count)
                return dataset, dataset, []

            fake_root = types.ModuleType("mlx_lm")
            fake_root.__path__ = []
            fake_tuner = types.ModuleType("mlx_lm.tuner")
            fake_tuner.__path__ = []
            fake_datasets = types.ModuleType("mlx_lm.tuner.datasets")
            fake_datasets.load_dataset = load_dataset
            fake_utils = types.ModuleType("mlx_lm.utils")
            fake_model = types.SimpleNamespace(
                layers=[object() for _ in range(plan["model"]["layers"])]
            )
            provider_config = {
                "hidden_size": plan["model"]["hidden_size"],
                "intermediate_size": plan["model"]["intermediate_size"],
                "num_hidden_layers": plan["model"]["layers"],
                "max_position_embeddings": plan["model"]["context_length"],
                "quantization_config": {"bits": 4},
            }

            def safe_load(*_args, **kwargs):
                observed["tokenizer_config"] = kwargs.get("tokenizer_config")
                return (
                    fake_model,
                    object(),
                    provider_config,
                )

            fake_utils.load = safe_load
            fake_utils.get_total_parameters = lambda _model: plan["model"]["parameters"]
            pinned_model = Path(temporary) / "pinned-model-data"
            pinned_model.mkdir()
            (pinned_model / "config.json").write_text(
                json.dumps(provider_config),
                encoding="utf-8",
            )
            fake_huggingface = types.ModuleType("huggingface_hub")
            fake_huggingface.snapshot_download = lambda **_kwargs: str(pinned_model)
            with (
                patch.dict(
                    sys.modules,
                    {
                        "huggingface_hub": fake_huggingface,
                        "mlx_lm": fake_root,
                        "mlx_lm.tuner": fake_tuner,
                        "mlx_lm.tuner.datasets": fake_datasets,
                        "mlx_lm.utils": fake_utils,
                    },
                ),
                patch.object(
                    module,
                    "require_unified_memory_admission",
                    return_value=_mlx_unified_memory_admission(plan),
                ),
                self.assertRaisesRegex(RuntimeError, "right-truncates"),
            ):
                module.require_model_data(plan)
            self.assertIs(observed["mask_prompt"], True)
            self.assertEqual(observed["tokenizer_config"], {"trust_remote_code": False})

    def test_mlx_compilation_refuses_unmasked_text_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self._mlx_plan(
                root,
                dataset_content=(
                    '{"text":"first fully supervised row"}\n'
                    '{"text":"second fully supervised row"}\n'
                ),
            )
            with self.assertRaisesRegex(ValueError, "refuses text rows"):
                generate_bundle(plan, root / "mlx-text-bundle")

    def test_bundle_documents_parent_runner_and_fail_closed_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            readme = (output / "README.md").read_text(encoding="utf-8")
            runbook = (output / "runbook.md").read_text(encoding="utf-8")
            manifest = json.loads(
                (output / "bundle-manifest.json").read_text(encoding="utf-8")
            )
        self.assertIn("python run.py --confirm-full-train", readme)
        self.assertIn("python run.py --confirm-full-train", runbook)
        self.assertIn("`aptus.bundle.v3`", readme)
        self.assertIn("`aptus.training-plan.v6`", readme)
        self.assertIn("`policy/model-policy-snapshot.v1.json`", readme)
        self.assertIn(current_model_policy_snapshot_sha256(), readme)
        self.assertIn("proves only the integrity of the frozen policy", readme)
        self.assertIn("Installed Aptus checks its current registry", readme)
        self.assertIn("`replan_required`", readme)
        self.assertIn(
            "checks plan-driving architecture\nfacts and target modules", runbook
        )
        self.assertIn("It constructs no optimizer and takes no training step", runbook)
        self.assertIn("beside it, never\ninside it", runbook)
        self.assertIn("Full-training resume is fail-closed", runbook)
        self.assertNotIn("--resume-from", runbook)
        self.assertEqual(manifest["entrypoints"]["run"], "run.py")

    def test_generated_runtime_contract_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            source = (output / "train.py").read_text(encoding="utf-8")
            preflight = (output / "preflight.py").read_text(encoding="utf-8")
            requirements = (output / "requirements.txt").read_text(encoding="utf-8")
            plan = make_plan(Path(temporary))
            fsdp_candidate = next(
                item for item in plan.candidates if item.distribution.value == "fsdp"
            )
            accelerate = _accelerate_config(replace(plan, recommended=fsdp_candidate))
        self.assertNotIn("overwrite_output_dir", source)
        self.assertIn('warmup_steps=trainer_config["warmup_steps"]', source)
        self.assertIn('revision=model_spec["revision"]', source)
        self.assertIn("class FiniteGuardTrainer(Trainer)", source)
        self.assertIn("def create_optimizer(self)", source)
        self.assertIn("exactly match the validated trainable set", source)
        self.assertIn("logging_nan_inf_filter=False", source)
        self.assertIn("def _clip_grad_norm(self, model", source)
        self.assertIn("require_trainable_parameters_finite", source)
        self.assertIn("optimizer_step_was_skipped", source)
        self.assertIn('candidate["method"] == "qlora" and capability < (6, 0)', source)
        self.assertIn(
            'candidate["method"] == "int8-lora" and capability < (7, 5)',
            source,
        )
        self.assertIn("accelerate.commands.accelerate_cli", preflight)
        self.assertIn("safetensors==0.8.0", requirements)
        self.assertIn("fsdp_use_orig_params: true", accelerate)

    def test_generated_linux_capacity_uses_kernel_memavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._bundle(root)
            module = self._load_generated(output, "aptus_generated_linux_memory")
            meminfo = root / "meminfo"
            meminfo.write_text(
                "MemTotal: 65536000 kB\n"
                "MemFree: 1048576 kB\n"
                "MemAvailable: 50331648 kB\n",
                encoding="utf-8",
            )
            with patch.object(module.platform, "system", return_value="Linux"):
                self.assertEqual(
                    module.linux_available_memory_bytes(meminfo),
                    50331648 * 1024,
                )

            meminfo.write_text("MemFree: 1048576 kB\n", encoding="utf-8")
            with patch.object(module.platform, "system", return_value="Linux"):
                self.assertIsNone(module.linux_available_memory_bytes(meminfo))

    def test_trainable_parameter_census_is_exact_stable_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_census")
            first_parameters = (
                (
                    "decoder.q_proj.lora_B.default.weight",
                    FakeCensusParameter(6, shape=(2, 3), dtype="fixture-float16"),
                ),
                (
                    "decoder.frozen",
                    FakeCensusParameter(100, shape=(10, 10), requires_grad=False),
                ),
                (
                    "decoder.q_proj.lora_A.default.weight",
                    FakeCensusParameter(2, shape=(2,), dtype="fixture-float32"),
                ),
            )
            second_parameters = tuple(reversed(first_parameters))
            fake_torch = types.ModuleType("torch")
            fake_torch.isfinite = lambda parameter: FakeFiniteResult(parameter.finite)
            lora_contract = {
                "method": "lora",
                "target_modules": ("q_proj",),
                "expected_adapter_target_match_count": 1,
            }
            with patch.dict(sys.modules, {"torch": fake_torch}):
                first = module.require_trainable_parameter_census(
                    FakeCensusModel(first_parameters), **lora_contract
                )
                second = module.require_trainable_parameter_census(
                    FakeCensusModel(second_parameters), **lora_contract
                )
                changed_dtype = module.require_trainable_parameter_census(
                    FakeCensusModel(
                        (
                            (
                                "decoder.q_proj.lora_B.default.weight",
                                FakeCensusParameter(
                                    6, shape=(2, 3), dtype="fixture-bfloat16"
                                ),
                            ),
                            first_parameters[2],
                            first_parameters[1],
                        )
                    ),
                    **lora_contract,
                )
                with self.assertRaisesRegex(RuntimeError, "zero trainable"):
                    module.require_trainable_parameter_census(
                        FakeCensusModel(
                            (
                                (
                                    "decoder.frozen",
                                    FakeCensusParameter(
                                        4, shape=(2, 2), requires_grad=False
                                    ),
                                ),
                            )
                        ),
                        **lora_contract,
                    )
                with self.assertRaisesRegex(FloatingPointError, "non-finite"):
                    module.require_trainable_parameter_census(
                        FakeCensusModel(
                            (
                                (
                                    "decoder.q_proj.lora_A.bad",
                                    FakeCensusParameter(1, shape=(1,), finite=False),
                                ),
                            )
                        ),
                        **lora_contract,
                    )
                with self.assertRaisesRegex(RuntimeError, "non-LoRA"):
                    module.require_trainable_parameter_census(
                        FakeCensusModel(
                            (
                                (
                                    "decoder.q_proj.weight",
                                    FakeCensusParameter(4, shape=(2, 2)),
                                ),
                            )
                        ),
                        **lora_contract,
                    )
                with self.assertRaisesRegex(RuntimeError, "non-LoRA"):
                    module.require_trainable_parameter_census(
                        FakeCensusModel(
                            (
                                (
                                    "decoder.lora_decoy.weight",
                                    FakeCensusParameter(4, shape=(2, 2)),
                                ),
                                first_parameters[1],
                            )
                        ),
                        **lora_contract,
                    )
                with self.assertRaisesRegex(RuntimeError, "non-LoRA"):
                    module.require_trainable_parameter_census(
                        FakeCensusModel(
                            (
                                (
                                    "decoder.unplanned_proj.lora_A.default.weight",
                                    FakeCensusParameter(2, shape=(2,)),
                                ),
                                (
                                    "decoder.unplanned_proj.lora_B.default.weight",
                                    FakeCensusParameter(2, shape=(2,)),
                                ),
                                first_parameters[1],
                            )
                        ),
                        **lora_contract,
                    )
                with self.assertRaisesRegex(RuntimeError, "frozen base"):
                    module.require_trainable_parameter_census(
                        FakeCensusModel(
                            (
                                (
                                    "decoder.q_proj.lora_A.default.weight",
                                    FakeCensusParameter(4, shape=(2, 2)),
                                ),
                            )
                        ),
                        **lora_contract,
                    )
                with self.assertRaisesRegex(RuntimeError, "exactly one LoRA A/B pair"):
                    module.require_trainable_parameter_census(
                        FakeCensusModel((first_parameters[2], first_parameters[1])),
                        **lora_contract,
                    )
                with self.assertRaisesRegex(RuntimeError, "left one or more"):
                    module.require_trainable_parameter_census(
                        FakeCensusModel(
                            (
                                (
                                    "decoder.weight",
                                    FakeCensusParameter(4, shape=(2, 2)),
                                ),
                                (
                                    "decoder.frozen",
                                    FakeCensusParameter(
                                        4, shape=(2, 2), requires_grad=False
                                    ),
                                ),
                            )
                        ),
                        method="full",
                        target_modules=(),
                        expected_adapter_target_match_count=0,
                    )

        self.assertEqual(first["trainable_parameter_count"], 8)
        self.assertEqual(first["trainable_tensor_count"], 2)
        self.assertIs(first["all_values_finite"], True)
        self.assertEqual(first["adapter_target_instance_count"], 1)
        self.assertEqual(first["expected_adapter_target_match_count"], 1)
        self.assertEqual(first["incomplete_adapter_target_instance_count"], 0)
        self.assertEqual(first["descriptor_sha256"], second["descriptor_sha256"])
        self.assertNotEqual(
            first["descriptor_sha256"], changed_dtype["descriptor_sha256"]
        )
        serialized = json.dumps(first, sort_keys=True)
        self.assertNotIn("decoder.q_proj.lora_B.default.weight", serialized)
        self.assertNotIn("decoder.q_proj.lora_A.default.weight", serialized)

    def test_selected_cuda_visibility_maps_once_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_cuda_visibility")
            plan = module.load_plan()
            plan["recommended"]["device_indices"] = [1]
            plan["recommended"]["world_size"] = 1
            candidate_id = plan["recommended"]["candidate_id"]
            with patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "GPU-a,MIG-b"},
                clear=False,
            ):
                os.environ.pop("APTUS_BOUND_DEVICE_CANDIDATE", None)
                module.bind_visible_cuda_devices(plan)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "MIG-b")
                self.assertEqual(
                    os.environ["APTUS_BOUND_DEVICE_CANDIDATE"], candidate_id
                )
                module.bind_visible_cuda_devices(plan)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "MIG-b")

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
                os.environ.pop("APTUS_BOUND_DEVICE_CANDIDATE", None)
                with self.assertRaisesRegex(RuntimeError, "no selectable"):
                    module.bind_visible_cuda_devices(plan)

            with patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
                    "APTUS_BOUND_DEVICE_CANDIDATE": candidate_id,
                },
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "missing or malformed"):
                    module.bind_visible_cuda_devices(plan)

    def test_measured_preflight_metrics_are_validated_hash_bound_and_carried(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated_path(
                output, "validate.py", "aptus_generated_preflight_attestation"
            )
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            candidate = plan["recommended"]
            metrics_path = output / "preflight-metrics.json"

            def write_metrics(**overrides):
                metrics = {
                    "schema_version": "aptus.preflight-metrics.v1",
                    "candidate_id": candidate["candidate_id"],
                    "method": candidate["method"],
                    "precision": candidate["precision"],
                    "quantization": candidate.get("quantization"),
                    "distribution": candidate["distribution"],
                    "world_size": candidate["world_size"],
                    "measured_peak_cuda_bytes": 4096,
                    "trainable_parameter_census": {
                        "schema_version": "aptus.trainable-parameter-census.v1",
                        "method": candidate["method"],
                        "parameter_scope": (
                            "all-parameters"
                            if candidate["method"] == "full"
                            else "lora-adapter-only"
                        ),
                        "trainable_parameter_count": 8,
                        "trainable_tensor_count": 2,
                        "frozen_parameter_count": (
                            0 if candidate["method"] == "full" else 2_000_000
                        ),
                        "frozen_tensor_count": (
                            0 if candidate["method"] == "full" else 1
                        ),
                        "unexpected_trainable_tensor_count": 0,
                        "expected_adapter_target_match_count": (
                            0 if candidate["method"] == "full" else 1
                        ),
                        "adapter_target_instance_count": (
                            0 if candidate["method"] == "full" else 1
                        ),
                        "incomplete_adapter_target_instance_count": 0,
                        "all_values_finite": True,
                        "descriptor_sha256": "a" * 64,
                    },
                    "scope": "synthetic-method-preflight-not-model-data-pilot",
                    **overrides,
                }
                metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
                return metrics

            valid = write_metrics()
            with patch.object(
                module, "actual_hardware_binding", return_value="hardware-binding"
            ):
                module._write_attestation("measured-preflight")
            report = json.loads(
                (output / "validation-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["preflight_metrics"], valid)
            self.assertEqual(
                report["bindings"]["preflight_metrics"], module.sha256(metrics_path)
            )

            metrics_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "unreadable"):
                module.require_preflight_metrics(plan)

            write_metrics(measured_peak_cuda_bytes=0)
            with self.assertRaisesRegex(RuntimeError, "positive"):
                module.require_preflight_metrics(plan)

            write_metrics(method="wrong-method")
            with self.assertRaisesRegex(RuntimeError, "bind method"):
                module.require_preflight_metrics(plan)

            write_metrics(measured_peak_cuda_bytes=8192)
            with patch.object(
                module, "actual_hardware_binding", return_value="hardware-binding"
            ):
                module._write_attestation("static")
            downgraded = json.loads(
                (output / "validation-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(downgraded["state"], "static-pass")

    def test_model_data_preflight_rejects_wrong_loaded_parameter_count(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "parameter count differs"):
            self._run_model_data_fixture(
                method="full",
                expected_parameters=4_000_000,
                actual_parameters=1_000_000,
                target_modules=(),
                module_names=("", "model.layers.0.self_attn.q_proj"),
            )

    def test_model_data_preflight_rejects_missing_adapter_target_module(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "down_proj"):
            self._run_model_data_fixture(
                method="lora",
                expected_parameters=2_000_000,
                actual_parameters=2_000_000,
                target_modules=("q_proj", "down_proj"),
                module_names=("", "model.layers.0.self_attn.q_proj"),
            )

    def test_model_data_preflight_rejects_each_wrong_structural_model_fact(
        self,
    ) -> None:
        cases = {
            "hidden_size": {"hidden_size": 1024},
            "layers": {"num_hidden_layers": 12},
            "context_length": {"max_position_embeddings": 2048},
            "intermediate_size": {"intermediate_size": 4096},
        }
        for fact, config_values in cases.items():
            with self.subTest(fact=fact), self.assertRaisesRegex(RuntimeError, fact):
                self._run_model_data_fixture(
                    method="full",
                    expected_parameters=2_000_000,
                    actual_parameters=2_000_000,
                    target_modules=(),
                    module_names=("",),
                    config_values=config_values,
                )

    def test_model_data_preflight_normalizes_config_aliases_and_rejects_absence(
        self,
    ) -> None:
        self._run_model_data_fixture(
            method="full",
            expected_parameters=2_000_000,
            actual_parameters=2_000_000,
            target_modules=(),
            module_names=("",),
            config_values={
                "hidden_size": None,
                "d_model": 2048,
                "num_hidden_layers": None,
                "n_layer": 24,
                "max_position_embeddings": None,
                "n_positions": 4096,
                "intermediate_size": None,
                "ffn_dim": 8192,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "unavailable: hidden_size"):
            self._run_model_data_fixture(
                method="full",
                expected_parameters=2_000_000,
                actual_parameters=2_000_000,
                target_modules=(),
                module_names=("",),
                config_values={"hidden_size": None},
            )

    def test_model_data_preflight_accepts_exact_quantized_structure_without_mutation(
        self,
    ) -> None:
        (
            calls,
            encoded_rows,
            cleanup_calls,
            loaded_model,
            preparation_events,
            census,
        ) = self._run_model_data_fixture(
            method="qlora",
            expected_parameters=2_000_000,
            actual_parameters=2_000_000,
            target_modules=("q_proj", "down_proj"),
            module_names=(
                "",
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.mlp.down_proj",
            ),
            quantized=True,
        )
        model_args, model_kwargs = calls["model"]
        self.assertEqual(model_args, ("example/model-fixture",))
        self.assertEqual(model_kwargs["revision"], "a" * 40)
        self.assertIs(model_kwargs["trust_remote_code"], False)
        self.assertIs(model_kwargs["local_files_only"], True)
        self.assertEqual(model_kwargs["dtype"], "fixture-bf16")
        self.assertEqual(model_kwargs["device_map"], {"": 0})
        self.assertEqual(
            model_kwargs["quantization_config"].kwargs,
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "fixture-bf16",
            },
        )
        self.assertEqual(len(encoded_rows), 2)
        self.assertEqual(cleanup_calls, ["empty_cache"])
        self.assertEqual(loaded_model.mutations, [])
        self.assertEqual(preparation_events, ["prepare", "census"])
        self.assertEqual(census["trainable_parameter_count"], 8192)

    def test_final_export_binds_every_tensor_key_to_its_actual_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._bundle(root)
            module = self._load_generated(output, "aptus_generated_export_verifier")
            candidate = {
                "method": "full",
                "distribution": "single",
                "world_size": 1,
            }
            model = {"model_id": "example/model-1b", "revision": "a" * 40}
            shard_keys = {}
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoConfig = FakePretrainedLoader
            fake_transformers.AutoTokenizer = FakePretrainedLoader
            fake_safetensors = types.ModuleType("safetensors")
            fake_safetensors.safe_open = lambda path, **kwargs: FakeSafeTensorFile(
                shard_keys[Path(path).name]
            )

            def export_dir(name, shards):
                directory = root / name
                directory.mkdir()
                (directory / "config.json").write_text("{}\n", encoding="utf-8")
                for shard_name, keys in shards.items():
                    (directory / shard_name).write_bytes(b"safetensors-placeholder")
                    shard_keys[shard_name] = keys
                return directory

            modules = {
                "transformers": fake_transformers,
                "safetensors": fake_safetensors,
            }
            with patch.dict(sys.modules, modules):
                single = export_dir("single", {"model.safetensors": ["weight"]})
                evidence = module.verify_final_export(single, candidate, model)
                self.assertEqual(evidence["weight_files"], ["model.safetensors"])

                missing_index = export_dir(
                    "missing-index",
                    {
                        "model-00001-of-00002.safetensors": ["weight.a"],
                        "model-00002-of-00002.safetensors": ["weight.b"],
                    },
                )
                with self.assertRaisesRegex(
                    RuntimeError, "requires one safetensors index"
                ):
                    module.verify_final_export(missing_index, candidate, model)

                duplicate = export_dir(
                    "duplicate",
                    {
                        "model-00001-of-00002.safetensors": ["shared.weight"],
                        "model-00002-of-00002.safetensors": ["shared.weight"],
                    },
                )
                (duplicate / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "weight_map": {
                                "shared.weight": "model-00001-of-00002.safetensors"
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "duplicate tensor key"):
                    module.verify_final_export(duplicate, candidate, model)

                misindexed = export_dir(
                    "misindexed",
                    {
                        "model-00001-of-00002.safetensors": ["weight.a"],
                        "model-00002-of-00002.safetensors": ["weight.b"],
                    },
                )
                index_path = misindexed / "model.safetensors.index.json"
                index_path.write_text(
                    json.dumps(
                        {
                            "weight_map": {
                                "weight.a": "model-00002-of-00002.safetensors",
                                "weight.b": "model-00001-of-00002.safetensors",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "wrong shards"):
                    module.verify_final_export(misindexed, candidate, model)

                index_path.write_text(
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
                evidence = module.verify_final_export(misindexed, candidate, model)
                self.assertEqual(len(evidence["weight_files"]), 2)

    def test_generation_rejects_a_syntax_error_in_portable_run_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bundle"

            def program_with_broken_run(runtime: str, name: str) -> bytes:
                if runtime == "cuda" and name == "run.py":
                    return b"def broken(:\n"
                return _bundle_program_bytes(runtime, name)

            with (
                patch(
                    "aptus.generation._bundle_program_bytes",
                    side_effect=program_with_broken_run,
                ),
                self.assertRaisesRegex(ValueError, "failed static validation"),
            ):
                generate_bundle(make_plan(root), output)
            self.assertFalse(output.exists())

    def test_static_validation_parses_the_portable_policy_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            policy_source = output / "policy_snapshot.py"
            policy_source.write_text("def broken(:\n", encoding="utf-8")
            manifest_path = output / "bundle-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(
                item
                for item in manifest["files"]
                if item["path"] == "policy_snapshot.py"
            )
            entry["sha256"] = hashlib.sha256(policy_source.read_bytes()).hexdigest()
            entry["size_bytes"] = policy_source.stat().st_size
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_bundle(output, level="static", run=False)

            self.assertEqual(report.state, ValidationState.INVALID)
            self.assertTrue(
                any(
                    finding.path == "policy_snapshot.py"
                    and finding.code == "PYTHON_PARSE_ERROR"
                    for finding in report.findings
                )
            )

    def test_cuda_preflight_parses_the_portable_policy_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            policy_source = output / "policy_snapshot.py"
            policy_source.write_text("def broken(:\n", encoding="utf-8")
            manifest_path = output / "bundle-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(
                item
                for item in manifest["files"]
                if item["path"] == "policy_snapshot.py"
            )
            entry["sha256"] = hashlib.sha256(policy_source.read_bytes()).hexdigest()
            entry["size_bytes"] = policy_source.stat().st_size
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, str(output / "preflight.py"), "--level", "static"],
                cwd=output,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("policy_snapshot.py", completed.stderr)
        self.assertIn("SyntaxError", completed.stderr)

    def test_cuda_dependency_pins_accept_pep440_local_labels(self) -> None:
        """CUDA wheels report local labels (2.13.0+cu130); pins stay public."""

        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated_path(
                output, "preflight.py", "aptus_generated_preflight_local_versions"
            )
            self.assertEqual(module._public_version("2.13.0+cu130"), "2.13.0")
            self.assertEqual(module._public_version("2.13.0"), "2.13.0")

            requirements = (output / "requirements.txt").read_text(encoding="utf-8")
            pinned = {
                line.split("==", 1)[0]: line.split("==", 1)[1]
                for line in requirements.splitlines()
                if line.strip() and "==" in line
            }

            def fake_version(name: str) -> str:
                base = pinned[name]
                if name == "torch":
                    return f"{base}+cu130"
                return base

            with patch.object(module, "version", side_effect=fake_version):
                module.require_dependencies()

            def wrong_torch(name: str) -> str:
                base = pinned[name]
                if name == "torch":
                    return "2.12.0+cu130"
                return base

            with patch.object(module, "version", side_effect=wrong_torch):
                with self.assertRaisesRegex(RuntimeError, "Dependency mismatch for torch"):
                    module.require_dependencies()

    def test_pilot_retention_deletes_only_marked_aptus_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated_path(
                output, "preflight.py", "aptus_generated_preflight_retention"
            )
            plan = module.require_contract()
            user_directory = output / "runs" / "pilot_notes"
            user_directory.mkdir(parents=True)
            (user_directory / "keep.txt").write_text("keep", encoding="utf-8")
            owned = output / "runs" / ("pilot_" + "a" * 32)
            module.claim_pilot_root(owned, plan)
            module.prune_pilot_runs(plan=plan, preserve=set())
            self.assertTrue(user_directory.is_dir())
            self.assertFalse(owned.exists())

    def test_pilot_checkpoint_failure_cannot_emit_a_passing_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated_path(
                output, "preflight.py", "aptus_generated_preflight_boundaries"
            )
            events: list[tuple[str, dict[str, object]]] = []

            def record_event(event_type: str, **fields: object) -> None:
                events.append((event_type, fields))

            with (
                patch.object(module, "require_contract", return_value={}),
                patch.object(module, "require_static"),
                patch.object(module, "require_dependencies"),
                patch.object(module, "training_command", return_value=["train"]),
                patch.object(
                    module,
                    "run_with_lease",
                    return_value=types.SimpleNamespace(returncode=0),
                ),
                patch.object(module, "prune_pilot_runs"),
                patch.object(module, "claim_pilot_root"),
                patch.object(
                    module,
                    "checkpoint_contract",
                    side_effect=RuntimeError(
                        "Checkpoint continuation artifact changed."
                    ),
                ),
                patch.object(module, "emit_boundary", side_effect=record_event),
                self.assertRaisesRegex(RuntimeError, "Checkpoint continuation"),
            ):
                module.run_validation(
                    types.SimpleNamespace(level="pilot", local_files_only=False)
                )

        self.assertEqual(
            events,
            [
                (
                    "pilot.phase-started",
                    {"phase": "pilot-phase-1", "action": "pilot"},
                ),
                (
                    "pilot.phase-finished",
                    {
                        "phase": "pilot-phase-1",
                        "action": "pilot",
                        "native_outcome": "failed",
                        "reason_code": "CHECKPOINT_CONTINUATION_FAILURE",
                    },
                ),
            ],
        )

    def test_pilot_child_projects_specific_terminal_failure_reasons(self) -> None:
        cases = (
            (RuntimeError("CUDA out of memory."), "CUDA_OOM"),
            (RuntimeError("Training produced a nonfinite loss."), "NONFINITE_VALUE"),
        )
        for index, (failure, expected_reason) in enumerate(cases):
            with (
                self.subTest(expected_reason),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = self._bundle(Path(temporary))
                module = self._load_generated(
                    output, f"aptus_generated_pilot_reason_{index}"
                )
                events: list[tuple[str, dict[str, object]]] = []

                def record_event(event_type: str, **fields: object) -> None:
                    events.append((event_type, fields))

                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            "train.py",
                            "--pilot",
                            "--max-steps",
                            "1",
                            "--campaign-pilot-phase",
                            "pilot-phase-1",
                            "--output-dir",
                            str(output / "pilot"),
                        ],
                    ),
                    patch.object(module, "require_execution_lease"),
                    patch.object(module, "load_plan", return_value={}),
                    patch.object(module, "bind_visible_cuda_devices"),
                    patch.object(
                        module, "load_trainer_config", return_value={"seed": 17}
                    ),
                    patch.object(module, "require_compiler_contract"),
                    patch.object(
                        module, "claim_output_dir", return_value=output / "pilot"
                    ),
                    patch.object(module, "run_training", side_effect=failure),
                    patch.object(module, "emit_boundary", side_effect=record_event),
                    self.assertRaises(type(failure)),
                ):
                    module.main()

                self.assertEqual(
                    events,
                    [
                        (
                            "pilot.phase-finished",
                            {
                                "phase": "pilot-phase-1",
                                "action": "pilot",
                                "native_outcome": "failed",
                                "reason_code": expected_reason,
                            },
                        )
                    ],
                )

    def test_pilot_child_emits_failure_when_setup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_pilot_setup_failure")
            events: list[tuple[str, dict[str, object]]] = []

            def record_event(event_type: str, **fields: object) -> None:
                events.append((event_type, fields))

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "train.py",
                        "--pilot",
                        "--campaign-pilot-phase",
                        "pilot-phase-2",
                    ],
                ),
                patch.object(module, "require_execution_lease"),
                patch.object(
                    module,
                    "load_plan",
                    side_effect=RuntimeError("Pilot setup failed."),
                ),
                patch.object(module, "emit_boundary", side_effect=record_event),
                self.assertRaisesRegex(RuntimeError, "Pilot setup failed"),
            ):
                module.main()

            self.assertEqual(
                events,
                [
                    (
                        "pilot.phase-finished",
                        {
                            "phase": "pilot-phase-2",
                            "action": "pilot",
                            "native_outcome": "failed",
                            "reason_code": "PROCESS_EXIT_NONZERO",
                        },
                    )
                ],
            )

    def test_user_values_stay_in_data_not_generated_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root)
            output = self._bundle(root)
            source = (output / "train.py").read_text(encoding="utf-8")
            compiled_plan = json.loads(
                (output / "plan.json").read_text(encoding="utf-8")
            )
        self.assertNotIn(plan.model.model_id, source)
        self.assertNotIn(str(plan.dataset.source_path), source)
        self.assertEqual(compiled_plan["dataset"]["source_path"], "data/dataset.jsonl")
        self.assertEqual(
            compiled_plan["dataset"]["source_sha256"], plan.dataset.source_sha256
        )

    def test_generated_masking_supervises_only_completion_for_structured_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_train")
            structured = module.encode_record(
                {"prompt": "prompt", "completion": "answer"}, FakeTokenizer(), 128
            )
            plain = module.encode_record(
                {"text": "all supervised"}, FakeTokenizer(), 128
            )
            messages = module.encode_record(
                {
                    "messages": [
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ]
                },
                FakeTokenizer(),
                128,
            )
        self.assertIn(-100, structured["labels"])
        self.assertNotIn(-100, plain["labels"])
        self.assertIn(-100, messages["labels"])
        self.assertTrue(any(value != -100 for value in messages["labels"]))
        with self.assertRaisesRegex(ValueError, "not prefix-separable"):
            module.encode_record(
                {
                    "messages": [
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ]
                },
                NonPrefixChatTokenizer(),
                128,
            )

    def test_generated_loader_matches_profiler_empty_row_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_rows")
            dataset = output / "data" / "rows.jsonl"
            dataset.write_text(
                '{"text":""}\n'
                '{"prompt":"question","completion":"answer"}\n'
                '{"instruction":"ignored without output"}\n',
                encoding="utf-8",
            )
            rows = module.load_rows(dataset)
        self.assertEqual(rows, [{"prompt": "question", "completion": "answer"}])

    def test_generated_schema_precedence_matches_the_profiler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_schema_precedence")
            tokenizer = FakeTokenizer()
            messages = [
                {"role": "user", "content": "chat prompt"},
                {"role": "assistant", "content": "chat answer"},
            ]

            self.assertEqual(
                module.record_to_parts(
                    {
                        "text": "plain text",
                        "prompt": "prompt",
                        "completion": "completion",
                        "messages": messages,
                    },
                    tokenizer,
                ),
                ("", "plain text", True),
            )
            self.assertEqual(
                module.record_to_parts(
                    {
                        "prompt": "prompt",
                        "completion": "completion",
                        "messages": messages,
                        "content": "content alias",
                    },
                    tokenizer,
                ),
                ("prompt", "completion", False),
            )
            instruction_prompt, completion, supervise_all = module.record_to_parts(
                {
                    "instruction": "instruction",
                    "output": "output",
                    "messages": messages,
                },
                tokenizer,
            )
            self.assertIn("instruction", instruction_prompt)
            self.assertEqual((completion, supervise_all), ("output", False))
            message_prompt, completion, supervise_all = module.record_to_parts(
                {"messages": messages, "content": "content alias"}, tokenizer
            )
            self.assertIn("chat prompt", message_prompt)
            self.assertEqual((completion, supervise_all), ("chat answer", False))

            encoded = module.encode_record(
                {"text": "plain text", "messages": messages}, tokenizer, 128
            )
            self.assertNotIn(-100, encoded["labels"])

    def test_generated_pilot_uses_bounded_deterministic_pressure_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_pilot_rows")
            rows = [{"text": "x" * length} for length in range(1, 50)]
            selected = module.select_pilot_rows(rows, limit=3)
            trainer = json.loads(
                (output / "config" / "trainer.json").read_text(encoding="utf-8")
            )
            source = (output / "train.py").read_text(encoding="utf-8")
        self.assertEqual([len(item["text"]) for item in selected], [49, 48, 47])
        self.assertEqual(trainer["pilot_row_limit"], 32)
        self.assertIn("eval_dataset = None", source)

    def test_cuda_campaign_fixture_is_reproducible_and_group_split_exact(self) -> None:
        fixture_bytes = CUDA_CAMPAIGN_FIXTURE.read_bytes()
        self.assertEqual(fixture_bytes, render_cuda_campaign_fixture())
        self.assertEqual(len(fixture_bytes), 1_635_765)
        self.assertEqual(
            hashlib.sha256(fixture_bytes).hexdigest(),
            "6d90599e949bf2698b940e0c159e1fa24f3dc0c162005546bd270fc761aac7f2",
        )
        self.assertEqual(
            CUDA_CAMPAIGN_GENERATOR_VERSION,
            "aptus.cuda-campaign-fixture-generator.v1",
        )
        self.assertEqual(CUDA_CAMPAIGN_FIXTURE_ID, "aptus.cuda-campaign-sft.v1")
        self.assertEqual(CUDA_CAMPAIGN_FIXTURE_SEED, 20260808)
        self.assertEqual(
            cuda_campaign_display_path(CUDA_CAMPAIGN_FIXTURE),
            "examples/cuda-campaign-sft-v1.jsonl",
        )
        self.assertEqual(
            cuda_campaign_display_path(
                Path(tempfile.gettempdir()) / "aptus-sensitive-parent" / "fixture.jsonl"
            ),
            "fixture.jsonl",
        )

        rows = [json.loads(line) for line in fixture_bytes.splitlines()]
        self.assertEqual(len(rows), 512)
        self.assertEqual(
            [row["row_id"] for row in rows],
            [f"cuda-campaign-row-{index:04d}" for index in range(512)],
        )
        self.assertEqual(
            Counter(row["target_content_words"] for row in rows),
            Counter({128: 256, 256: 128, 512: 64, 1024: 32, 2048: 32}),
        )
        self.assertEqual(
            set(Counter(row["split_group"] for row in rows).values()),
            {4},
        )
        self.assertEqual(len({row["split_group"] for row in rows}), 128)
        self.assertEqual(len({row["prompt"] for row in rows}), 512)
        for row in rows:
            prompt = row["prompt"].removesuffix(".\nAgent:")
            completion = row["completion"].removesuffix(".")
            self.assertEqual(
                len(prompt.split()) + len(completion.split()),
                row["target_content_words"],
            )

        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_campaign_split")
            train_offsets, evaluation_offsets, evidence = (
                module.split_jsonl_offsets_with_evidence(
                    CUDA_CAMPAIGN_FIXTURE,
                    evaluation_fraction=0.125,
                    seed=424242,
                )
            )
        self.assertEqual((len(train_offsets), len(evaluation_offsets)), (448, 64))
        self.assertEqual(evidence["training_declared_group_count"], 112)
        self.assertEqual(evidence["evaluation_declared_group_count"], 16)
        self.assertEqual(evidence["evaluation_row_error"], 0)
        self.assertEqual(
            evidence["canonical_jsonl_sha256"],
            "6d90599e949bf2698b940e0c159e1fa24f3dc0c162005546bd270fc761aac7f2",
        )
        self.assertEqual(
            evidence["assignment_sha256"],
            "7e9e747a6e69868d2d542137468cd1baf3d81d7aaac1de29ed14e4dd83b428ed",
        )

        def groups_at(offsets) -> set[str]:
            groups: set[str] = set()
            with CUDA_CAMPAIGN_FIXTURE.open("rb") as source:
                for offset in offsets:
                    source.seek(offset)
                    groups.add(json.loads(source.readline())["split_group"])
            return groups

        train_groups = groups_at(train_offsets)
        evaluation_groups = groups_at(evaluation_offsets)
        self.assertEqual(len(train_groups), 112)
        self.assertEqual(len(evaluation_groups), 16)
        self.assertFalse(train_groups & evaluation_groups)

        def target_counts_at(offsets) -> Counter[int]:
            counts: Counter[int] = Counter()
            with CUDA_CAMPAIGN_FIXTURE.open("rb") as source:
                for offset in offsets:
                    source.seek(offset)
                    counts[json.loads(source.readline())["target_content_words"]] += 1
            return counts

        self.assertEqual(
            target_counts_at(train_offsets),
            Counter({128: 232, 256: 112, 512: 56, 1024: 24, 2048: 24}),
        )
        self.assertEqual(
            target_counts_at(evaluation_offsets),
            Counter({128: 24, 256: 16, 512: 8, 1024: 8, 2048: 8}),
        )

    def test_generated_full_dataset_split_and_lazy_random_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_lazy_rows")
            dataset = output / "data" / "lazy.jsonl"
            dataset.write_text(
                "".join(
                    json.dumps({"text": f"row-{index}"}) + "\n" for index in range(20)
                ),
                encoding="utf-8",
            )
            first = module.split_jsonl_offsets_with_evidence(
                dataset, evaluation_fraction=0.2, seed=17
            )
            second = module.split_jsonl_offsets_with_evidence(
                dataset, evaluation_fraction=0.2, seed=17
            )
            self.assertEqual(tuple(first[0]), tuple(second[0]))
            self.assertEqual(tuple(first[1]), tuple(second[1]))
            self.assertTrue(first[0])
            self.assertTrue(first[1])
            self.assertEqual(len(first[0]) + len(first[1]), 20)
            lazy = module.LazyJsonlDataset(
                dataset,
                first[0],
                FakeTokenizer(),
                64,
                first[2]["canonical_jsonl_sha256"],
            )
            encoded = lazy[len(lazy) - 1]
            self.assertTrue(encoded["input_ids"])
            self.assertIsNone(lazy.__getstate__()["_source"])

            changed = output / "data" / "changed.jsonl"
            changed.write_text('{"text":"old-a"}\n{"text":"old-b"}\n', encoding="utf-8")
            changed_train, _changed_eval, changed_evidence = (
                module.split_jsonl_offsets_with_evidence(
                    changed, evaluation_fraction=0.5, seed=17
                )
            )
            changed.write_text('{"text":"new-a"}\n{"text":"new-b"}\n', encoding="utf-8")
            changed_lazy = module.LazyJsonlDataset(
                changed,
                changed_train,
                FakeTokenizer(),
                64,
                changed_evidence["canonical_jsonl_sha256"],
            )
            with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                changed_lazy[0]

            race = output / "data" / "race.jsonl"
            original_race = b'{"text":"old-a"}\n{"text":"old-b"}\n'
            mutated_race = b'{"text":"new-a"}\n{"text":"new-b"}\n'
            self.assertEqual(len(original_race), len(mutated_race))
            race.write_bytes(original_race)
            race_train, _race_eval, race_evidence = (
                module.split_jsonl_offsets_with_evidence(
                    race, evaluation_fraction=0.5, seed=17
                )
            )
            race_lazy = module.LazyJsonlDataset(
                race,
                race_train,
                FakeTokenizer(),
                64,
                race_evidence["canonical_jsonl_sha256"],
            )
            race_lazy._open_verified_source()
            verified_source = race_lazy._source

            class MutatingReadSource:
                def fileno(self):
                    return verified_source.fileno()

                def seek(self, offset):
                    return verified_source.seek(offset)

                def readline(self):
                    race.write_bytes(mutated_race)
                    # Same-size rewrites can share a timestamp tick on some
                    # filesystems. Force the metadata identity to change so
                    # this test deterministically exercises the consumption
                    # guard instead of depending on timestamp granularity.
                    mutated_stat = race.stat()
                    os.utime(
                        race,
                        ns=(
                            mutated_stat.st_atime_ns,
                            mutated_stat.st_mtime_ns + 2_000_000_000,
                        ),
                    )
                    return verified_source.readline()

                def close(self):
                    return verified_source.close()

            race_lazy._source = MutatingReadSource()
            with self.assertRaisesRegex(RuntimeError, "during consumption"):
                race_lazy[0]

            one = output / "data" / "one.jsonl"
            one.write_text('{"text":"only"}\n', encoding="utf-8")
            train, evaluation = module.split_jsonl_offsets(
                one, evaluation_fraction=0.9, seed=17
            )
            self.assertEqual(len(train), 1)
            self.assertEqual(len(evaluation), 0)

    def test_generated_group_aware_split_never_crosses_train_eval_boundary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_group_split")
            dataset = output / "data" / "grouped.jsonl"
            rows = [
                {"text": "alpha-1", "split_group": "source-alpha"},
                {"text": "free-1"},
                {"text": "beta-1", "metadata": {"split_group": "source-beta"}},
                {"text": "alpha-2", "split_group": "source-alpha"},
                {"text": "free-2"},
                {
                    "text": "gamma-1",
                    "split_group": "source-gamma",
                    "metadata": {"split_group": "source-gamma"},
                },
                {"text": "beta-2", "metadata": {"split_group": "source-beta"}},
                {"text": "alpha-3", "split_group": "source-alpha"},
                {"text": "free-3"},
                {
                    "text": "gamma-2",
                    "split_group": "source-gamma",
                    "metadata": {"split_group": "source-gamma"},
                },
            ]
            dataset.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            first = module.split_jsonl_offsets_with_evidence(
                dataset, evaluation_fraction=0.4, seed=29
            )
            second = module.split_jsonl_offsets_with_evidence(
                dataset, evaluation_fraction=0.4, seed=29
            )
            simple_train, simple_eval = module.split_jsonl_offsets(
                dataset, evaluation_fraction=0.4, seed=29
            )

            def rows_at(offsets):
                values = []
                with dataset.open("rb") as source:
                    for offset in offsets:
                        source.seek(offset)
                        values.append(json.loads(source.readline()))
                return values

            def group_of(row):
                return row.get("split_group") or row.get("metadata", {}).get(
                    "split_group"
                )

            train_rows = rows_at(first[0])
            eval_rows = rows_at(first[1])
            train_groups = {group_of(row) for row in train_rows if group_of(row)}
            eval_groups = {group_of(row) for row in eval_rows if group_of(row)}
            evidence = first[2]

            self.assertFalse(train_groups & eval_groups)
            for group in {"source-alpha", "source-beta", "source-gamma"}:
                locations = {
                    "train" if row in train_rows else "evaluation"
                    for row in rows
                    if group_of(row) == group
                }
                self.assertEqual(len(locations), 1)
            self.assertEqual(tuple(first[0]), tuple(second[0]))
            self.assertEqual(tuple(first[1]), tuple(second[1]))
            self.assertEqual(first[2], second[2])
            self.assertEqual(tuple(first[0]), tuple(simple_train))
            self.assertEqual(tuple(first[1]), tuple(simple_eval))
            self.assertEqual(
                evidence["strategy"], "deterministic-size-aware-group-sha256"
            )
            self.assertEqual(evidence["total_row_count"], len(rows))
            self.assertEqual(evidence["training_row_count"], len(train_rows))
            self.assertEqual(evidence["evaluation_row_count"], len(eval_rows))
            self.assertEqual(evidence["declared_group_count"], 3)
            self.assertEqual(evidence["ungrouped_row_count"], 3)
            self.assertEqual(len(evidence["assignment_sha256"]), 64)
            self.assertNotIn("source-alpha", json.dumps(evidence, sort_keys=True))

            one_group = output / "data" / "one-group.jsonl"
            one_group.write_text(
                "".join(
                    json.dumps({"text": f"same-{index}", "split_group": "same"}) + "\n"
                    for index in range(4)
                ),
                encoding="utf-8",
            )
            train, evaluation, one_group_evidence = (
                module.split_jsonl_offsets_with_evidence(
                    one_group, evaluation_fraction=0.9, seed=29
                )
            )
            self.assertEqual(len(train), 4)
            self.assertEqual(len(evaluation), 0)
            self.assertEqual(one_group_evidence["training_declared_group_count"], 1)

            conflicting = output / "data" / "conflicting-groups.jsonl"
            conflicting.write_text(
                json.dumps(
                    {
                        "text": "conflict",
                        "split_group": "top",
                        "metadata": {"split_group": "nested"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflicting split_group"):
                module.split_jsonl_offsets_with_evidence(
                    conflicting, evaluation_fraction=0.2, seed=29
                )

            imbalanced = output / "data" / "imbalanced-groups.jsonl"
            imbalanced.write_text(
                "".join(
                    json.dumps({"text": f"huge-{index}", "split_group": "huge"}) + "\n"
                    for index in range(900)
                )
                + "".join(
                    json.dumps({"text": f"small-{index}", "split_group": "small"})
                    + "\n"
                    for index in range(100)
                ),
                encoding="utf-8",
            )
            imbalanced_train, imbalanced_eval, imbalanced_evidence = (
                module.split_jsonl_offsets_with_evidence(
                    imbalanced, evaluation_fraction=0.1, seed=0
                )
            )
            self.assertEqual(len(imbalanced_train), 900)
            self.assertEqual(len(imbalanced_eval), 100)
            self.assertEqual(imbalanced_evidence["target_evaluation_row_count"], 100)
            self.assertEqual(imbalanced_evidence["evaluation_row_error"], 0)

            adversarial = output / "data" / "adversarial-groups.jsonl"
            adversarial.write_text(
                "".join(
                    json.dumps({"text": f"{group}-{index}", "split_group": group})
                    + "\n"
                    for group, count in (("a", 51), ("b", 97), ("c", 49), ("d", 3))
                    for index in range(count)
                ),
                encoding="utf-8",
            )
            adversarial_train, adversarial_eval, adversarial_evidence = (
                module.split_jsonl_offsets_with_evidence(
                    adversarial, evaluation_fraction=0.5, seed=0
                )
            )
            self.assertEqual(len(adversarial_train), 100)
            self.assertEqual(len(adversarial_eval), 100)
            self.assertEqual(adversarial_evidence["evaluation_row_error"], 0)

    def test_portable_static_entrypoint_needs_no_ml_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            environment = os.environ.copy()
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            completed = subprocess.run(
                [sys.executable, str(output / "validate.py"), "--level", "static"],
                cwd=output,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            report = json.loads(
                (output / "validation-report.json").read_text(encoding="utf-8")
            )
            cache_created = (output / "__pycache__").exists()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(report["state"], "static-pass")
        self.assertFalse(cache_created)

    def test_package_free_entrypoint_rejects_a_non_object_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            (output / "bundle-manifest.json").write_text("null\n", encoding="utf-8")

            completed = self._run_package_free_static_validation(output)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Bundle manifest must be a JSON object.", completed.stderr)
        self.assertNotIn("AttributeError", completed.stderr)

    def test_package_free_entrypoint_rejects_non_object_plans(self) -> None:
        invalid_roots = (
            ("null", "null\n"),
            ("array", "[]\n"),
            ("number", "7\n"),
            ("string", '"plan"\n'),
            ("boolean", "true\n"),
        )
        for name, contents in invalid_roots:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                output = self._bundle(Path(temporary))
                (output / "plan.json").write_text(contents, encoding="utf-8")

                completed = self._run_package_free_static_validation(output)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(
                    "Bundle plan does not bind the emitted model policy snapshot.",
                    completed.stderr,
                )
                self.assertNotIn("AttributeError", completed.stderr)

    def test_package_free_entrypoint_rejects_resource_hostile_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            plan_path = output / "plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            nested: object = "leaf"
            for _ in range(500):
                nested = [nested]
            plan["unexpected_nested_value"] = nested
            plan_path.write_text(
                json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            self._refresh_manifested_file(output, "plan.json")

            completed = self._run_package_free_static_validation(output)

            self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Plan structure is malformed:", completed.stderr)
        self.assertNotIn("RecursionError", completed.stderr)

    def test_package_free_entrypoints_normalize_json_parser_resource_errors(
        self,
    ) -> None:
        invalid_documents = (
            ("oversized-integer", '{"value":' + "9" * 5000 + "}\n"),
            (
                "excessive-nesting",
                '{"value":' + "[" * 10000 + "0" + "]" * 10000 + "}\n",
            ),
        )
        entrypoints = (
            ("cuda-preflight", self._bundle, "preflight.py", ("--level", "static")),
            ("mlx-validate", self._mlx_bundle, "validate.py", ("--level", "static")),
        )
        for entrypoint, make_bundle, relative, arguments in entrypoints:
            for document, contents in invalid_documents:
                with (
                    self.subTest(entrypoint=entrypoint, document=document),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    output = make_bundle(Path(temporary))
                    (output / "plan.json").write_text(contents, encoding="utf-8")

                    completed = self._run_package_free_entrypoint(
                        output, relative, *arguments
                    )

                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(
                        "Bundle plan is unreadable or invalid JSON.",
                        completed.stderr,
                    )
                    self.assertNotIn("RecursionError", completed.stderr)
                    self.assertNotIn("integer string conversion", completed.stderr)

    def test_package_free_cuda_entrypoints_validate_plan_before_device_binding(
        self,
    ) -> None:
        entrypoints = (
            ("validate.py", ("--level", "static")),
            ("run.py", ("--confirm-full-train",)),
        )
        invalid_roots = (
            ("null", None),
            ("array", []),
            ("number", 7),
            ("string", "invalid"),
            ("boolean", True),
        )
        for relative, arguments in entrypoints:
            for name, invalid_value in invalid_roots:
                with (
                    self.subTest(relative=relative, name=name),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    output = self._bundle(Path(temporary))
                    plan_path = output / "plan.json"
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    plan["recommended"] = invalid_value
                    plan_path.write_text(
                        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="utf-8",
                    )
                    self._refresh_manifested_file(output, "plan.json")

                    completed = self._run_package_free_entrypoint(
                        output, relative, *arguments
                    )

                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("Invalid Aptus plan:", completed.stderr)
                    self.assertNotIn("TypeError", completed.stderr)
                    self.assertNotIn("AttributeError", completed.stderr)

    def test_package_free_entrypoint_rejects_corrupted_policy_snapshot_data(
        self,
    ) -> None:
        mutations = (
            (
                "missing",
                lambda path: path.unlink(),
                "Bundle model policy snapshot is missing.",
            ),
            (
                "malformed",
                lambda path: path.write_text("{", encoding="utf-8"),
                "Bundle model policy snapshot is malformed:",
            ),
            (
                "noncanonical",
                lambda path: path.write_text(
                    json.dumps(json.loads(path.read_text(encoding="utf-8"))),
                    encoding="utf-8",
                ),
                "Bundle model policy snapshot is not canonical JSON.",
            ),
            (
                "tampered",
                lambda path: path.write_bytes(
                    path.read_bytes().replace(
                        b"malformed or contradictory",
                        b"malformed or inconsistent",
                        1,
                    )
                ),
                "Bundle model policy snapshot digest does not match.",
            ),
        )
        for name, mutate, expected_error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                output = self._bundle(Path(temporary))
                mutate(output / "policy/model-policy-snapshot.v1.json")

                completed = self._run_package_free_static_validation(output)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stderr)

    def test_standalone_preflight_uses_snapshot_after_host_policy_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            source_root = Path(__file__).resolve().parents[2] / "src"
            script = """
import copy
import json
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
source_root = Path(sys.argv[1])
bundle = Path(sys.argv[2])
sys.path.insert(0, str(source_root))
from aptus import model_compatibility as host_policy

changed_snapshot = copy.deepcopy(host_policy.current_model_policy_snapshot())
changed_snapshot["dense_families"] = sorted(
    [*changed_snapshot["dense_families"], "future-dense-family"]
)
plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
assert (
    host_policy.model_policy_snapshot_sha256(changed_snapshot)
    != plan["model_policy_snapshot_sha256"]
)
host_policy.current_model_policy_snapshot = lambda: changed_snapshot
sys.path.insert(0, str(bundle))
sys.argv = [str(bundle / "preflight.py"), "--level", "static"]
runpy.run_path(bundle / "preflight.py", run_name="__main__")
"""
            completed = subprocess.run(
                [sys.executable, "-c", script, str(source_root), str(output)],
                cwd=output,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Aptus static validation passed.", completed.stdout)

    def test_managed_preflight_rejects_mismatched_host_launch_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            manifest_digest = sha256_file(output / "bundle-manifest.json")
            policy_digest = current_model_policy_snapshot_sha256()
            cases = (
                ("artifact", "0" * 64, policy_digest),
                ("policy", manifest_digest, "0" * 64),
            )
            for label, artifact_binding, policy_binding in cases:
                with self.subTest(label=label):
                    environment = {
                        **os.environ,
                        "APTUS_EXPECTED_ARTIFACT_FINGERPRINT": artifact_binding,
                        "APTUS_AUTHORIZED_MODEL_POLICY_SNAPSHOT_SHA256": (
                            policy_binding
                        ),
                    }
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(output / "preflight.py"),
                            "--level",
                            "static",
                        ],
                        cwd=output,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                    )

                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("host-authorized", completed.stderr)

    def test_managed_cuda_train_load_plan_rejects_mismatched_host_launch_bindings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            manifest_digest = sha256_file(output / "bundle-manifest.json")
            policy_digest = current_model_policy_snapshot_sha256()
            cases = (
                ("artifact", "0" * 64, policy_digest),
                ("policy", manifest_digest, "0" * 64),
            )
            for label, artifact_binding, policy_binding in cases:
                with self.subTest(label=label):
                    environment = {
                        **os.environ,
                        "APTUS_EXPECTED_ARTIFACT_FINGERPRINT": artifact_binding,
                        "APTUS_AUTHORIZED_MODEL_POLICY_SNAPSHOT_SHA256": (
                            policy_binding
                        ),
                    }
                    completed = subprocess.run(
                        [sys.executable, "-B", "-c", "import train; train.load_plan()"],
                        cwd=output,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                    )

                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("host-authorized", completed.stderr)

    def test_generated_entrypoint_rejects_preexisting_bytecode_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            (output / "__pycache__").mkdir()
            completed = subprocess.run(
                [sys.executable, str(output / "validate.py"), "--level", "static"],
                cwd=output,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unmanifested __pycache__", completed.stderr)

    def test_static_validator_detects_manifest_and_lock_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            lock = output / "requirements.txt"
            lock.write_text(
                lock.read_text(encoding="utf-8") + "invented==1\n", encoding="utf-8"
            )
            report = validate_bundle(output)
        self.assertEqual(report.state, ValidationState.INVALID)
        self.assertTrue(
            any(
                item.code in {"MANIFEST_MISMATCH", "DEPENDENCY_SET_MISMATCH"}
                for item in report.findings
            )
        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_unmanifested_mutable_directory_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._bundle(root)
            external = root / "external-runs"
            external.mkdir()
            (output / "runs").symlink_to(external, target_is_directory=True)
            report = validate_bundle(output)
        self.assertEqual(report.state, ValidationState.INVALID)
        self.assertTrue(
            any(
                item.code == "MANIFEST_INTEGRITY" and "symlink" in item.message
                for item in report.findings
            )
        )

    def test_dataset_change_aborts_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root)
            plan.dataset.source_path.write_text(
                '{"text":"changed"}\n', encoding="utf-8"
            )
            output = root / "bundle"
            with self.assertRaisesRegex(ValueError, "changed"):
                generate_bundle(plan, output)
            self.assertFalse(output.exists())

    def test_archive_is_deterministic_and_excludes_time_varying_attestation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root)
            first, second = root / "first", root / "second"
            generate_bundle(plan, first)
            generate_bundle(plan, second)
            first_zip = create_bundle_archive(first)
            second_zip = create_bundle_archive(second)
            self.assertEqual(first_zip.read_bytes(), second_zip.read_bytes())
            inventory = validate_bundle_archive_bytes(first_zip.read_bytes())
            self.assertIn("bundle-manifest.json", inventory)
            self.assertIn("plan.json", inventory)
            self.assertTrue(verify_bundle_archive(first, first_zip))

    def test_archive_rejects_arbitrary_zip_symlink_and_hardlink_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arbitrary = root / "arbitrary.zip"
            with zipfile.ZipFile(
                arbitrary,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                info = zipfile.ZipInfo("plan.json", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, b"{}\n")
            with self.assertRaisesRegex(ValueError, "canonical Aptus"):
                validate_bundle_archive_bytes(arbitrary.read_bytes())

        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = self._bundle(root)
                source = output / "plan.json"
                linked = output / "linked-plan.json"
                if kind == "symlink":
                    linked.symlink_to(source)
                else:
                    os.link(source, linked)
                with self.assertRaisesRegex(
                    ValueError, "symlink|regular non-hardlinked"
                ):
                    create_bundle_archive(output, root / "release.zip")

    def test_archive_rejects_a_self_consistent_but_incomplete_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = self._bundle(root)
            incomplete = root / "incomplete"
            (incomplete / "policy").mkdir(parents=True)
            retained = {
                "plan.json",
                "policy/model-policy-snapshot.v1.json",
            }
            for relative in retained:
                destination = incomplete / relative
                destination.write_bytes((complete / relative).read_bytes())
            manifest = json.loads(
                (complete / "bundle-manifest.json").read_text(encoding="utf-8")
            )
            manifest["files"] = [
                item for item in manifest["files"] if item["path"] in retained
            ]
            (incomplete / "bundle-manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            archive = create_bundle_archive(incomplete, root / "incomplete.zip")

            with self.assertRaisesRegex(ValueError, "canonical Aptus"):
                validate_bundle_archive_bytes(archive.read_bytes())

    def test_archive_verification_detects_bundle_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._bundle(root)
            archive = create_bundle_archive(output)
            self.assertTrue(verify_bundle_archive(output, archive))
            (output / "plan.json").write_text("{}\n", encoding="utf-8")
            self.assertFalse(verify_bundle_archive(output, archive))

    def test_archive_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._bundle(root)
            archive = root / "release.zip"
            archive.write_bytes(b"keep")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                create_bundle_archive(output, archive)
            self.assertEqual(archive.read_bytes(), b"keep")

    def test_archive_publish_does_not_clobber_a_concurrent_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._bundle(root)
            archive = root / "release.zip"
            original_link = os.link

            def competing_link(source: str | Path, target: str | Path) -> None:
                Path(target).write_bytes(b"competitor")
                original_link(source, target)

            with patch("aptus.generation.os.link", side_effect=competing_link):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    create_bundle_archive(output, archive)
            self.assertEqual(archive.read_bytes(), b"competitor")

    def test_full_training_requires_current_pilot_attestation_before_ml_imports(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            module = self._load_generated(output, "aptus_generated_gate")
            plan = module.load_plan()
            self.assertEqual(module.resolve_max_steps(pilot=True, max_steps=2), 2)
            first_output = module.default_output_dir(plan, pilot=False).resolve()
            second_output = module.default_output_dir(plan, pilot=False).resolve()
            self.assertEqual(first_output.parent, (output / "runs").resolve())
            self.assertTrue(first_output.name.startswith("run_"))
            self.assertNotEqual(first_output, second_output)
            with self.assertRaisesRegex(RuntimeError, "finite train_loss"):
                module.assert_measured_training_metrics(
                    {"global_step": 1, "train_loss": math.nan},
                    candidate=plan["recommended"],
                    pilot=True,
                )
            with self.assertRaisesRegex(RuntimeError, "non skipped optimizer steps"):
                module.assert_measured_training_metrics(
                    {
                        "global_step": 1,
                        "train_loss": 1.0,
                        "finite_raw_loss_checks": 1,
                        "finite_backward_loss_checks": 1,
                        "finite_gradient_norm_checks": 1,
                        "finite_trainable_parameter_scans": 1,
                        "optimizer_parameter_binding_checks": 1,
                    },
                    candidate=plan["recommended"],
                    pilot=True,
                )
            with self.assertRaisesRegex(RuntimeError, "pilot"):
                module.require_pilot_and_record_approval(plan, output / "runs" / "test")

    def test_portable_parent_recovers_pending_success_before_new_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            run_dir = output / "runs" / ("run_" + "c" * 32)
            run_dir.mkdir(parents=True)
            (output / "validation-report.json").write_text(
                json.dumps(
                    {
                        "state": "execution-approved",
                        "active_run": {"output_dir": str(run_dir)},
                        "measured_run_pending_at": "2026-07-21T00:00:00+00:00",
                        "pending_final_export": {},
                        "pending_measured_run": {},
                    }
                ),
                encoding="utf-8",
            )
            module = self._load_generated_path(
                output, "run.py", "aptus_generated_run_recovery"
            )
            with self.assertRaisesRegex(RuntimeError, "ROOT/runs/run_"):
                module.normalize_run_output(Path(temporary) / "outside", fresh=True)
            completed = subprocess.CompletedProcess([], 0)
            with patch.object(
                module, "run_with_lease", return_value=completed
            ) as runner:
                result = module.launch_full_training(
                    types.SimpleNamespace(output_dir=None, local_files_only=False)
                )
        self.assertEqual(result, 0)
        command = runner.call_args.args[0]
        self.assertIn("--promote-pending", command)
        self.assertEqual(Path(command[-1]), run_dir.resolve())

    def test_lock_contains_only_imported_method_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = self._bundle(Path(temporary))
            lock = (output / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("datasets==", lock)
        self.assertNotIn("trl==", lock)
        self.assertIn("torch==2.13.0", lock)


if __name__ == "__main__":
    unittest.main()
