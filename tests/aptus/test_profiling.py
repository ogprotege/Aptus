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
    RuntimeCapability,
    _apple_silicon_chip_name,
    _apple_metal_gpu_core_count,
    _bitsandbytes_capabilities,
    _darwin_available_memory,
    _linux_available_memory,
    _nvidia_smi_devices,
    _probe_mlx_runtime,
    build_hardware_spec,
    build_model_spec,
    canonical_training_rows,
    pilot_sample_rows,
    probe_apple_platform,
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
    def test_linux_host_memory_uses_kernel_memavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            meminfo = Path(temporary) / "meminfo"
            meminfo.write_text(
                "MemTotal: 65536000 kB\n"
                "MemFree: 1048576 kB\n"
                "MemAvailable: 50331648 kB\n",
                encoding="utf-8",
            )
            with patch("aptus.profiling.platform.system", return_value="Linux"):
                self.assertEqual(_linux_available_memory(meminfo), 50331648 * 1024)

            meminfo.write_text("MemAvailable: invalid kB\n", encoding="utf-8")
            with patch("aptus.profiling.platform.system", return_value="Linux"):
                self.assertIsNone(_linux_available_memory(meminfo))

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

    def test_apple_chip_probe_uses_privacy_reduced_system_profiler_fallback(
        self,
    ) -> None:
        failed_sysctl = subprocess.CompletedProcess([], 1, "", "denied")
        safe_fallback = subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "SPHardwareDataType": [
                        {"chip_type": "Apple M5 Pro", "machine_model": "Mac17,9"}
                    ]
                }
            ),
            "",
        )
        with patch(
            "aptus.profiling.subprocess.run",
            side_effect=[failed_sysctl, safe_fallback],
        ) as run:
            self.assertEqual(_apple_silicon_chip_name(), "Apple M5 Pro")
        self.assertEqual(
            run.call_args_list[1].args[0],
            [
                "/usr/sbin/system_profiler",
                "-detailLevel",
                "mini",
                "SPHardwareDataType",
                "-json",
            ],
        )

    def test_mlx_probe_requires_an_actual_device_query(self) -> None:
        fake_metal = SimpleNamespace(is_available=lambda: True)
        fake_mlx = SimpleNamespace(
            metal=fake_metal,
            device_info=lambda: (_ for _ in ()).throw(RuntimeError("no device")),
        )
        with (
            patch("aptus.profiling._module_is_installed", return_value=True),
            patch("aptus.profiling._distribution_version", return_value="0.32.0"),
            patch("aptus.profiling.importlib.import_module", return_value=fake_mlx),
        ):
            capability = _probe_mlx_runtime()
        self.assertTrue(capability.installed)
        self.assertFalse(capability.available)
        self.assertIn("no device", capability.detail)

    def test_apple_gpu_core_probe_reads_only_the_builtin_gpu_record(self) -> None:
        payload = json.dumps(
            {
                "SPDisplaysDataType": [
                    {
                        "_name": "External display",
                        "sppci_bus": "spdisplays_external",
                        "sppci_device_type": "spdisplays_display",
                        "sppci_cores": "999",
                    },
                    {
                        "_name": "Apple M5 Pro",
                        "sppci_bus": "spdisplays_builtin",
                        "sppci_device_type": "spdisplays_gpu",
                        "sppci_cores": "20",
                    },
                ]
            }
        )
        with patch("aptus.profiling._fixed_command_text", return_value=payload):
            self.assertEqual(_apple_metal_gpu_core_count(), 20)

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
                "aptus.profiling._metal_recommended_working_set_bytes",
                return_value=None,
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
        self.assertIsNone(device.free_vram_bytes)
        self.assertFalse(device.supports_bf16)
        self.assertFalse(device.supports_4bit)
        self.assertFalse(device.supports_8bit)
        self.assertIn("shared unified memory", device.provenance.detail)
        self.assertIn("not dedicated VRAM", device.provenance.detail)
        self.assertIn(
            "free_vram_bytes is intentionally omitted", device.provenance.detail
        )
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
                "aptus.profiling._metal_recommended_working_set_bytes",
                return_value=None,
            ),
            patch(
                "aptus.profiling.shutil.disk_usage",
                return_value=SimpleNamespace(free=gibibytes(100)),
            ),
        ):
            hardware = probe_local_hardware(disk_path=Path("/tmp"))
        self.assertIsNone(hardware.host_ram_free_bytes)
        self.assertIsNone(hardware.devices[0].free_vram_bytes)

    def test_darwin_vm_stat_available_memory_is_conservative(self) -> None:
        output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages inactive: 200.
Pages speculative: 50.
Pages purgeable: 900.
"""
        with patch("aptus.profiling._fixed_command_text", return_value=output):
            measured = _darwin_available_memory(gibibytes(64))
        self.assertEqual(measured, 350 * 16384)

    def test_apple_platform_report_keeps_runtime_capabilities_separate(self) -> None:
        mlx = RuntimeCapability(True, True, "0.31.2", "ready")
        mlx_lm = RuntimeCapability(True, True, "0.31.3", "ready")
        torch_mps = RuntimeCapability(True, False, "2.9.0", "not available")

        def command(arguments, *, timeout=3):
            del timeout
            if arguments[-1] == "-productVersion":
                return "26.5.2"
            if arguments[-1] == "-buildVersion":
                return "25F84"
            return None

        with (
            patch("aptus.profiling.platform.system", return_value="Darwin"),
            patch("aptus.profiling.platform.machine", return_value="arm64"),
            patch(
                "aptus.profiling._host_memory",
                return_value=(gibibytes(64), gibibytes(23)),
            ),
            patch("aptus.profiling._fixed_command_text", side_effect=command),
            patch(
                "aptus.profiling._apple_silicon_chip_name",
                return_value="Apple M5 Pro",
            ),
            patch("aptus.profiling.os.cpu_count", return_value=18),
            patch("aptus.profiling._apple_metal_gpu_core_count", return_value=20),
            patch("aptus.profiling._darwin_memory_free_percent", return_value=36),
            patch(
                "aptus.profiling._darwin_swap_usage",
                return_value=(gibibytes(4), gibibytes(1), gibibytes(3)),
            ),
            patch(
                "aptus.profiling._metal_recommended_working_set_bytes",
                return_value=gibibytes(48),
            ),
            patch("aptus.profiling._probe_mlx_runtime", return_value=mlx),
            patch("aptus.profiling._probe_mlx_lm_runtime", return_value=mlx_lm),
            patch(
                "aptus.profiling._probe_pytorch_mps_runtime",
                return_value=torch_mps,
            ),
        ):
            profile = probe_apple_platform()

        self.assertEqual(profile.os_version, "26.5.2")
        self.assertEqual(profile.os_build, "25F84")
        self.assertEqual(profile.chip_name, "Apple M5 Pro")
        self.assertEqual(profile.logical_cpu_count, 18)
        self.assertEqual(profile.metal_gpu_core_count, 20)
        self.assertEqual(profile.unified_memory_bytes, gibibytes(64))
        self.assertEqual(profile.available_memory_bytes, gibibytes(23))
        self.assertEqual(profile.metal_recommended_working_set_bytes, gibibytes(48))
        self.assertTrue(profile.mlx.available)
        self.assertTrue(profile.mlx_lm.available)
        self.assertFalse(profile.pytorch_mps.available)
        self.assertEqual(profile.to_dict()["mlx"]["version"], "0.31.2")

    def test_apple_platform_probe_fails_closed_off_apple_silicon(self) -> None:
        with (
            patch("aptus.profiling.platform.system", return_value="Linux"),
            patch("aptus.profiling.platform.machine", return_value="aarch64"),
        ):
            with self.assertRaisesRegex(ValueError, "Darwin arm64"):
                probe_apple_platform()

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
        self.assertEqual(hardware.limiting_vram_bytes, 0)
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
