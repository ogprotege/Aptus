import json
import tempfile
import unittest
from pathlib import Path

from aptus.domain import Backend, MeasurementKind, gibibytes
from aptus.profiling import (
    build_hardware_spec,
    build_model_spec,
    profile_dataset,
)


class DatasetProfilingTests(unittest.TestCase):
    def test_profiles_jsonl_text_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "data.jsonl"
            rows = [
                {"text": "a" * 40},
                {"text": "b" * 80},
                {"text": "c" * 120},
            ]
            dataset.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            first = profile_dataset(dataset)
            second = profile_dataset(dataset)

            self.assertEqual(first, second)
            self.assertEqual(first.example_count, 3)
            self.assertEqual(first.total_estimated_tokens, 60)
            self.assertEqual(first.sequence_p50, 20)
            self.assertEqual(first.sequence_p95, 30)
            self.assertEqual(first.sequence_max, 30)
            self.assertEqual(first.schema_name, "text")
            self.assertEqual(first.measurement, MeasurementKind.ESTIMATED)

    def test_profiles_conversation_for_analysis_but_warns_generation_is_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "chat.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "messages": [
                                {"role": "user", "content": "Hello"},
                                {"role": "assistant", "content": "Hi"},
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )

            profile = profile_dataset(dataset)

            self.assertEqual(profile.schema_name, "messages")
            self.assertTrue(
                any("plain text" in warning.lower() for warning in profile.warnings)
            )

    def test_content_field_is_analysis_only_not_trainable_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "content.jsonl"
            dataset.write_text(
                '{"content":"analysis content"}\n',
                encoding="utf-8",
            )

            profile = profile_dataset(dataset)

            self.assertEqual(profile.schema_name, "content")
            self.assertTrue(
                any("plain text" in warning.lower() for warning in profile.warnings)
            )

    def test_sampling_still_validates_every_row_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "mixed.jsonl"
            dataset.write_text(
                '{"text":"valid first row"}\n{"unknown":"unsupported later row"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "no supported"):
                profile_dataset(dataset, sample_limit=1)

    def test_empty_dataset_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "empty.jsonl"
            dataset.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no examples"):
                profile_dataset(dataset)


class ExplicitSpecBuilderTests(unittest.TestCase):
    def test_builds_per_device_cuda_hardware(self) -> None:
        hardware = build_hardware_spec(
            backend=Backend.CUDA,
            gpu_count=2,
            vram_gib=24,
            supports_bf16=True,
            supports_4bit=True,
            host_ram_gib=64,
            reserve_gib=2,
        )

        self.assertEqual(hardware.gpu_count, 2)
        self.assertEqual(hardware.devices[0].total_vram_bytes, gibibytes(24))
        self.assertEqual(hardware.limiting_vram_bytes, gibibytes(22))

    def test_builds_model_parameters_from_billions(self) -> None:
        model = build_model_spec(
            model_id="meta-llama/example",
            revision="a" * 40,
            family="llama",
            parameters_b=7,
            hidden_size=4096,
            layers=32,
            context_length=4096,
            license_name="example-license",
            training_allowed=True,
        )

        self.assertEqual(model.parameters, 7_000_000_000)


if __name__ == "__main__":
    unittest.main()
