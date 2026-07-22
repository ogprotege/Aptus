import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aptus.domain import Backend, MeasurementKind, gibibytes
from aptus.profiling import (
    _apple_silicon_chip_name,
    _bitsandbytes_capabilities,
    _nvidia_smi_devices,
    build_hardware_spec,
    build_model_spec,
    canonical_training_rows,
    pilot_sample_rows,
    probe_local_hardware,
    profile_dataset,
)


class DatasetProfilingTests(unittest.TestCase):
    def test_profiles_all_supported_schemas_and_deterministic_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.jsonl"
            rows = [
                {"text": "plain text"},
                {"prompt": "p", "completion": "c"},
                {"instruction": "i", "input": "x", "output": "o"},
                {
                    "messages": [
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ]
                },
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            first = profile_dataset(path, sample_limit=2, sequence_length=2)
            second = profile_dataset(path, sample_limit=2, sequence_length=2)
            expected_canonical_size = sum(
                len(
                    (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                )
                for row in rows
            )
        self.assertEqual(first, second)
        self.assertEqual(first.schema_name, "mixed")
        self.assertEqual(
            set(first.schema_counts),
            {"text", "prompt-completion", "instruction-output", "messages"},
        )
        self.assertEqual(first.sampled_examples, 2)
        self.assertTrue(first.truncation_count)
        self.assertEqual(first.canonical_size_bytes, expected_canonical_size)
        self.assertEqual(
            first.max_canonical_row_bytes,
            max(
                len(
                    (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                )
                for row in rows
            ),
        )

    def test_detects_empty_and_duplicate_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            path.write_text(
                '{"text":"same"}\n{"text":" same "}\n{"text":""}\n', encoding="utf-8"
            )
            profile = profile_dataset(path)
        self.assertEqual(profile.example_count, 2)
        self.assertEqual(profile.duplicate_count, 1)
        self.assertEqual(profile.empty_count, 1)
        self.assertTrue(any("duplicate" in item.lower() for item in profile.warnings))

    def test_tokenizer_measurement_is_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.txt"
            path.write_text("alpha beta gamma\n", encoding="utf-8")
            profile = profile_dataset(path, tokenizer=lambda text: text.split())
        self.assertEqual(profile.measurement, MeasurementKind.TOKENIZER_MEASURED)
        self.assertEqual(profile.total_estimated_tokens, 3)

    def test_unsupported_row_fails_even_when_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            path.write_text('{"text":"ok"}\n{"unknown":"bad"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no supported"):
                profile_dataset(path, sample_limit=1)

    def test_messages_schema_requires_the_generated_transform_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            path.write_text(
                '{"messages":[{"role":"assistant","content":"answer"},'
                '{"role":"user","content":"follow-up"}]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "end with a non-empty assistant"):
                profile_dataset(path)

    def test_canonical_and_pilot_rows_cover_supported_file_formats(self) -> None:
        fixtures = {
            "jsonl": '{"text":"short"}\n{"text":"the longest row"}\n',
            "json": '[{"text":"short"},{"text":"the longest row"}]',
            "csv": "text\nshort\nthe longest row\n",
            "txt": "short\nthe longest row\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix, content in fixtures.items():
                with self.subTest(suffix=suffix):
                    path = root / f"data.{suffix}"
                    path.write_text(content, encoding="utf-8")
                    profile = profile_dataset(path, sample_limit=2)
                    canonical = list(canonical_training_rows(profile))
                    pilot = pilot_sample_rows(profile, limit=1)
                    self.assertEqual(len(canonical), 2)
                    self.assertEqual(len(pilot), 1)
                    self.assertIn("longest", str(pilot[0]))


class HardwareInspectionTests(unittest.TestCase):
    def test_bitsandbytes_feature_thresholds_are_not_conflated(self) -> None:
        self.assertEqual(_bitsandbytes_capabilities(5, 9), (False, False))
        self.assertEqual(_bitsandbytes_capabilities(6, 0), (True, False))
        self.assertEqual(_bitsandbytes_capabilities(7, 4), (True, False))
        self.assertEqual(_bitsandbytes_capabilities(7, 5), (True, True))

    def test_nvidia_smi_uses_bounded_argument_vector_without_shell(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "GPU X, 24576, 23000, 8.9\n", "")
        with (
            patch("aptus.profiling.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("aptus.profiling.subprocess.run", return_value=completed) as run,
        ):
            rows = _nvidia_smi_devices()
        self.assertEqual(rows[0][0], "GPU X")
        arguments, keywords = run.call_args
        self.assertEqual(arguments[0][0], "/usr/bin/nvidia-smi")
        self.assertNotIn("shell", keywords)
        self.assertEqual(keywords["timeout"], 8)

    def test_nvidia_smi_timeout_is_an_explicit_probe_failure(self) -> None:
        with (
            patch("aptus.profiling.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "aptus.profiling.subprocess.run",
                side_effect=subprocess.TimeoutExpired("nvidia-smi", 8),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "probe failed"):
                _nvidia_smi_devices()

    def test_apple_chip_probe_uses_fixed_bounded_argument_vector(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "Apple M5 Pro\n", "")
        with patch("aptus.profiling.subprocess.run", return_value=completed) as run:
            chip_name = _apple_silicon_chip_name()
        self.assertEqual(chip_name, "Apple M5 Pro")
        arguments, keywords = run.call_args
        self.assertEqual(
            arguments[0],
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
        )
        self.assertNotIn("shell", keywords)
        self.assertEqual(keywords["timeout"], 3)

    def test_apple_chip_probe_does_not_admit_serial_or_uuid_output(self) -> None:
        completed = subprocess.CompletedProcess(
            [], 0, "Hardware UUID: 00000000-0000-0000-0000-000000000000\n", ""
        )
        with patch("aptus.profiling.subprocess.run", return_value=completed):
            self.assertIsNone(_apple_silicon_chip_name())

    def test_darwin_arm64_falls_back_to_measured_shared_memory(self) -> None:
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch(
                "aptus.profiling._host_memory",
                return_value=(gibibytes(64), gibibytes(48)),
            ),
            patch("aptus.profiling._nvidia_smi_devices", return_value=[]),
            patch("aptus.profiling.platform.system", return_value="Darwin"),
            patch("aptus.profiling.platform.machine", return_value="arm64"),
            patch(
                "aptus.profiling._apple_silicon_chip_name",
                return_value="Apple M5 Pro",
            ),
            patch(
                "aptus.profiling.shutil.disk_usage",
                return_value=SimpleNamespace(free=gibibytes(100)),
            ),
        ):
            hardware = probe_local_hardware(disk_path=Path("/tmp"))
        self.assertEqual(hardware.host_ram_bytes, gibibytes(64))
        self.assertEqual(hardware.host_ram_free_bytes, gibibytes(48))
        self.assertEqual(len(hardware.devices), 1)
        device = hardware.devices[0]
        self.assertEqual(device.backend, Backend.MPS)
        self.assertEqual(device.name, "Apple M5 Pro (shared unified memory)")
        self.assertEqual(device.total_vram_bytes, gibibytes(64))
        self.assertEqual(device.free_vram_bytes, gibibytes(48))
        self.assertFalse(device.supports_bf16)
        self.assertFalse(device.supports_4bit)
        self.assertFalse(device.supports_8bit)
        self.assertIn("shared unified memory", device.provenance.detail)
        self.assertIn("not dedicated VRAM", device.provenance.detail)
        self.assertIn("fail-closed for MPS", device.provenance.detail)
        self.assertIn("not dedicated VRAM", hardware.provenance.detail)

    def test_darwin_arm64_omits_unmeasured_available_memory(self) -> None:
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch("aptus.profiling._host_memory", return_value=(gibibytes(64), None)),
            patch("aptus.profiling._nvidia_smi_devices", return_value=[]),
            patch("aptus.profiling.platform.system", return_value="Darwin"),
            patch("aptus.profiling.platform.machine", return_value="arm64"),
            patch("aptus.profiling._apple_silicon_chip_name", return_value=None),
            patch(
                "aptus.profiling.shutil.disk_usage",
                return_value=SimpleNamespace(free=gibibytes(100)),
            ),
        ):
            hardware = probe_local_hardware(disk_path=Path("/tmp"))
        self.assertIsNone(hardware.host_ram_free_bytes)
        self.assertIsNone(hardware.devices[0].free_vram_bytes)

    def test_non_darwin_no_device_remains_an_explicit_failure(self) -> None:
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch("aptus.profiling._host_memory", return_value=(gibibytes(64), None)),
            patch("aptus.profiling._nvidia_smi_devices", return_value=[]),
            patch("aptus.profiling.platform.system", return_value="Linux"),
            patch("aptus.profiling.platform.machine", return_value="aarch64"),
            patch("aptus.profiling._apple_silicon_chip_name") as chip_probe,
            patch(
                "aptus.profiling.shutil.disk_usage",
                return_value=SimpleNamespace(free=gibibytes(100)),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                probe_local_hardware(disk_path=Path("/tmp"))
        chip_probe.assert_not_called()

    def test_manual_hardware_and_model_facts_remain_available(self) -> None:
        hardware = build_hardware_spec(
            backend=Backend.CUDA,
            gpu_count=2,
            vram_gib=24,
            supports_bf16=True,
            supports_4bit=True,
            host_ram_gib=64,
            reserve_gib=2,
        )
        model = build_model_spec(
            model_id="example/model",
            revision="a" * 40,
            family="llama",
            parameters_b=7,
            hidden_size=4096,
            layers=32,
            context_length=4096,
            license_name="apache-2.0",
            training_allowed=True,
        )
        self.assertEqual(hardware.limiting_vram_bytes, gibibytes(22))
        self.assertEqual(model.parameters, 7_000_000_000)

    def test_manual_available_memory_and_eight_bit_are_explicit(self) -> None:
        hardware = build_hardware_spec(
            backend=Backend.CUDA,
            gpu_count=2,
            vram_gib=24,
            free_vram_gib=18,
            supports_bf16=True,
            supports_4bit=False,
            supports_8bit=True,
            host_ram_gib=64,
            host_ram_free_gib=20,
            reserve_gib=2,
        )
        self.assertEqual(hardware.limiting_vram_bytes, gibibytes(16))
        self.assertEqual(hardware.host_ram_free_bytes, gibibytes(20))
        self.assertTrue(hardware.devices[0].supports_8bit)
        self.assertFalse(hardware.devices[0].supports_4bit)


if __name__ == "__main__":
    unittest.main()
