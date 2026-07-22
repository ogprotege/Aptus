import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aptus.domain import ValidationState
from aptus.generation import _accelerate_config, create_bundle_archive, generate_bundle
from aptus.validation import validate_bundle

from tests.aptus.helpers import make_plan


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
                    "encode_record",
                    side_effect=lambda row, *_args: encoded_rows.append(row),
                ),
            ):
                module.model_data_preflight(plan, trainer_config, local_files_only=True)
            return calls, encoded_rows, cleanup_calls, loaded_model

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
        self.assertIn(
            "verify parameter count and adapter target-module presence", runbook
        )
        self.assertIn("does not mutate weights or claim calibrated fit", runbook)
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
        calls, encoded_rows, cleanup_calls, loaded_model = self._run_model_data_fixture(
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
            with (
                patch("aptus.generation.RUN_SCRIPT", "def broken(:\n"),
                self.assertRaisesRegex(ValueError, "failed static validation"),
            ):
                generate_bundle(make_plan(root), output)
            self.assertFalse(output.exists())

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
            first = module.split_jsonl_offsets(
                dataset, evaluation_fraction=0.2, seed=17
            )
            second = module.split_jsonl_offsets(
                dataset, evaluation_fraction=0.2, seed=17
            )
            self.assertEqual(tuple(first[0]), tuple(second[0]))
            self.assertEqual(tuple(first[1]), tuple(second[1]))
            self.assertTrue(first[0])
            self.assertTrue(first[1])
            self.assertEqual(len(first[0]) + len(first[1]), 20)
            lazy = module.LazyJsonlDataset(dataset, first[0], FakeTokenizer(), 64)
            encoded = lazy[len(lazy) - 1]
            self.assertTrue(encoded["input_ids"])
            self.assertIsNone(lazy.__getstate__()["_source"])

            one = output / "data" / "one.jsonl"
            one.write_text('{"text":"only"}\n', encoding="utf-8")
            train, evaluation = module.split_jsonl_offsets(
                one, evaluation_fraction=0.9, seed=17
            )
            self.assertEqual(len(train), 1)
            self.assertEqual(len(evaluation), 0)

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
