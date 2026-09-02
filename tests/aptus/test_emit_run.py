from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aptus import cli as aptus_cli
from aptus.cli import main
from aptus.domain import Backend
from aptus.emit_run import (
    EMIT_RUN_SCHEMA_VERSION,
    fill_namespace_from_hardware,
    spec_plan_argv,
)
from aptus.profiling import build_hardware_spec

from tests.aptus.test_cli import fact_arguments


def _without_hardware_flags(arguments: list[str]) -> list[str]:
    skip = {
        "--gpu-count",
        "--vram-gib",
        "--free-vram-gib",
        "--host-ram-gib",
        "--host-ram-free-gib",
        "--disk-free-gib",
        "--bf16",
        "--four-bit",
        "--eight-bit",
    }
    out: list[str] = []
    consume_value = False
    for item in arguments:
        if consume_value:
            consume_value = False
            continue
        if item in skip:
            consume_value = item not in {
                "--bf16",
                "--four-bit",
                "--eight-bit",
            }
            continue
        out.append(item)
    return out


def _cuda_hardware():
    return build_hardware_spec(
        backend=Backend.CUDA,
        gpu_count=1,
        vram_gib=24,
        supports_bf16=True,
        supports_4bit=True,
        supports_8bit=False,
        free_vram_gib=22,
        host_ram_gib=64,
        host_ram_free_gib=56,
        reserve_gib=2,
        disk_free_gib=500,
    )


def _mps_hardware():
    return build_hardware_spec(
        backend=Backend.MPS,
        gpu_count=1,
        vram_gib=48,
        supports_bf16=False,
        supports_4bit=False,
        host_ram_gib=48,
        host_ram_free_gib=36,
        reserve_gib=8,
        disk_free_gib=400,
    )


def _blank_namespace(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "gpu_count": None,
        "vram_gib": None,
        "free_vram_gib": None,
        "host_ram_gib": None,
        "host_ram_free_gib": None,
        "disk_free_gib": None,
        "reserve_gib": 2.0,
        "backend": "cuda",
        "training_runtime": None,
        "bf16": False,
        "four_bit": False,
        "eight_bit": False,
        "model_id": "example/model",
        "revision": "a" * 40,
        "family": "llama",
        "parameters_b": 1.0,
        "model_type": None,
        "architecture": None,
        "quantization_bits": None,
        "quantization_group_size": None,
        "quantization_layout_profile": None,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "moe_expert_count": None,
        "moe_experts_per_token": None,
        "moe_expert_intermediate_size": None,
        "moe_decoder_sparse_step": None,
        "moe_mlp_only_layer": [],
        "moe_shared_expert_intermediate_size": None,
        "layers": 24,
        "context_length": 4096,
        "license": "apache-2.0",
        "confirm_training_allowed": True,
        "confirm_unreviewed_runtime": False,
        "inspection_receipt": None,
        "dataset": Path("train.jsonl"),
        "sample_limit": 512,
        "objective": "quality",
        "sequence_length": 128,
        "effective_batch_size": 8,
        "epochs": 3,
        "prefer_method": None,
        "evaluation_fraction": 0.1,
        "checkpoint_steps": 100,
        "optimizer_steps": None,
        "split_seed": 424242,
        "training_seed": 17,
        "data_order_seed": 1000017,
        "micro_batch_size": None,
        "gradient_accumulation_steps": None,
        "packing": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class EmitRunTests(unittest.TestCase):
    def test_fill_uses_probed_cuda_capacity_and_measured_flags(self) -> None:
        arguments = _blank_namespace()
        notes = fill_namespace_from_hardware(arguments, _cuda_hardware())
        self.assertEqual(arguments.gpu_count, 1)
        self.assertAlmostEqual(float(arguments.vram_gib), 24.0, places=4)
        self.assertAlmostEqual(float(arguments.free_vram_gib), 22.0, places=4)
        self.assertAlmostEqual(float(arguments.host_ram_gib), 64.0, places=4)
        self.assertTrue(arguments.bf16)
        self.assertTrue(arguments.four_bit)
        self.assertFalse(arguments.eight_bit)
        self.assertEqual(arguments.backend, "cuda")
        self.assertIsNone(arguments.training_runtime)
        self.assertTrue(any("probe_local_hardware" in note for note in notes))

    def test_fill_selects_mlx_on_apple_silicon_when_backend_is_parser_default(
        self,
    ) -> None:
        arguments = _blank_namespace()
        notes = fill_namespace_from_hardware(arguments, _mps_hardware())
        self.assertEqual(arguments.backend, "mps")
        self.assertEqual(arguments.training_runtime, "mlx-lm")
        self.assertEqual(arguments.reserve_gib, 8.0)
        self.assertFalse(arguments.four_bit)
        self.assertTrue(any("mlx-lm" in note for note in notes))

    def test_include_is_refused_on_cuda_this_host(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            gold = root / "gold.jsonl"
            dataset.write_text(
                '{"prompt":"Question 1?","completion":"Answer 1."}\n'
                '{"prompt":"Question 2?","completion":"Answer 2."}\n',
                encoding="utf-8",
            )
            gold.write_text(
                '{"prompt":"Question 1?","completion":"Answer 1."}\n',
                encoding="utf-8",
            )
            with patch(
                "aptus.emit_run.probe_local_hardware", return_value=_cuda_hardware()
            ):
                self.assertEqual(
                    main(
                        [
                            "emit-run",
                            *fact_arguments(dataset),
                            "--include",
                            str(gold),
                            "--output",
                            str(root / "run"),
                        ]
                    ),
                    2,
                )

    def test_fill_refuses_explicit_cuda_runtime_on_probed_mps_host(self) -> None:
        arguments = _blank_namespace(training_runtime="transformers-peft-cuda")
        with self.assertRaisesRegex(ValueError, "this machine"):
            fill_namespace_from_hardware(arguments, _mps_hardware())

    def test_operator_hardware_facts_win_over_probe(self) -> None:
        arguments = _blank_namespace(gpu_count=2, vram_gib=80.0, four_bit=True)
        fill_namespace_from_hardware(arguments, _cuda_hardware())
        self.assertEqual(arguments.gpu_count, 2)
        self.assertEqual(arguments.vram_gib, 80.0)
        self.assertTrue(arguments.four_bit)

    def test_spec_plan_argv_is_a_runnable_spec_plan_command(self) -> None:
        arguments = _blank_namespace()
        fill_namespace_from_hardware(arguments, _cuda_hardware())
        argv = spec_plan_argv(arguments, Path("/tmp/plan.json"))
        self.assertEqual(argv[1:3], ["-m", "aptus"])
        self.assertIn("spec-plan", argv)
        self.assertIn("--confirm-training-allowed", argv)
        self.assertNotIn("--confirm-full-train", argv)
        self.assertIn("--objective", argv)
        self.assertEqual(argv[argv.index("--objective") + 1], "quality")

    def test_cli_writes_scripts_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text(
                '{"prompt":"Question 1?","completion":"Answer 1."}\n'
                '{"prompt":"Question 2?","completion":"Answer 2."}\n',
                encoding="utf-8",
            )
            output = root / "run"
            hardware = _cuda_hardware()
            argv = [
                "emit-run",
                *fact_arguments(dataset),
                "--output",
                str(output),
            ]
            # fact_arguments already supplies hardware; probe still runs.
            with patch("aptus.emit_run.probe_local_hardware", return_value=hardware):
                self.assertEqual(main(argv), 0)
            spec_plan = (output / "spec-plan.sh").read_text(encoding="utf-8")
            self.assertIn("spec-plan", spec_plan)
            self.assertNotIn("--confirm-full-train", spec_plan)
            self.assertTrue((output / "hardware.json").is_file())
            payload = json.loads((output / "emit-run.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], EMIT_RUN_SCHEMA_VERSION)
            self.assertFalse(payload["trained"])
            self.assertIsNone(payload["bundle"])
            self.assertNotIn("job_id", payload)

    def test_cli_fills_omitted_hardware_from_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text(
                '{"prompt":"Question 1?","completion":"Answer 1."}\n'
                '{"prompt":"Question 2?","completion":"Answer 2."}\n',
                encoding="utf-8",
            )
            output = root / "run"
            with patch(
                "aptus.emit_run.probe_local_hardware", return_value=_cuda_hardware()
            ):
                self.assertEqual(
                    main(
                        [
                            "emit-run",
                            *_without_hardware_flags(fact_arguments(dataset)),
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            script = (output / "spec-plan.sh").read_text(encoding="utf-8")
            self.assertIn("--gpu-count", script)
            self.assertIn("--vram-gib", script)
            self.assertIn("--host-ram-gib", script)
            self.assertIn("--four-bit", script)

    def test_cli_refuses_to_overwrite_existing_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text('{"prompt":"q","completion":"a"}\n', encoding="utf-8")
            output = root / "run"
            output.mkdir()
            (output / "spec-plan.sh").write_text("already\n", encoding="utf-8")
            with patch(
                "aptus.emit_run.probe_local_hardware", return_value=_cuda_hardware()
            ):
                self.assertEqual(
                    main(
                        [
                            "emit-run",
                            *fact_arguments(dataset),
                            "--output",
                            str(output),
                        ]
                    ),
                    2,
                )

    def test_compile_flag_does_not_teach_a_dead_run_plan_requirement(self) -> None:
        source = Path(aptus_cli.__file__).read_text(encoding="utf-8")
        self.assertNotIn(
            "pass --compile with --run-plan",
            source,
        )


if __name__ == "__main__":
    unittest.main()
