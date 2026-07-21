import unittest
from pathlib import Path

from aptus.domain import (
    Backend,
    DatasetProfile,
    DeviceSpec,
    HardwareSpec,
    MeasurementKind,
    Method,
    ModelSpec,
    Objective,
    TrainingTarget,
    gibibytes,
    to_primitive,
)


class DomainContractTests(unittest.TestCase):
    def test_gibibytes_uses_binary_units(self) -> None:
        self.assertEqual(gibibytes(1), 1024**3)
        self.assertEqual(gibibytes(24), 24 * 1024**3)

    def test_hardware_uses_per_device_vram_not_aggregate_vram(self) -> None:
        hardware = HardwareSpec(
            devices=(
                DeviceSpec(
                    name="GPU 0",
                    backend=Backend.CUDA,
                    total_vram_bytes=gibibytes(24),
                    supports_bf16=True,
                    supports_4bit=True,
                ),
                DeviceSpec(
                    name="GPU 1",
                    backend=Backend.CUDA,
                    total_vram_bytes=gibibytes(24),
                    supports_bf16=True,
                    supports_4bit=True,
                ),
            ),
            host_ram_bytes=gibibytes(64),
            reserve_per_device_bytes=gibibytes(2),
        )

        self.assertEqual(hardware.gpu_count, 2)
        self.assertEqual(hardware.limiting_vram_bytes, gibibytes(22))
        self.assertNotEqual(hardware.limiting_vram_bytes, gibibytes(44))

    def test_invalid_model_facts_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "parameters"):
            ModelSpec(
                model_id="example/model",
                revision="a" * 40,
                family="llama",
                parameters=0,
                hidden_size=4096,
                layers=32,
                context_length=4096,
                license_name="unknown",
                training_allowed=True,
            )

    def test_model_requires_immutable_revision_and_license(self) -> None:
        common = {
            "model_id": "example/model",
            "family": "llama",
            "parameters": 1_000_000_000,
            "hidden_size": 1024,
            "layers": 16,
            "context_length": 2048,
            "training_allowed": True,
        }
        with self.assertRaisesRegex(ValueError, "immutable"):
            ModelSpec(revision="main", license_name="apache-2.0", **common)
        with self.assertRaisesRegex(ValueError, "license"):
            ModelSpec(revision="a" * 40, license_name=" ", **common)

    def test_contracts_serialize_to_json_safe_primitives(self) -> None:
        profile = DatasetProfile(
            source_path=Path("/tmp/data.jsonl"),
            source_sha256="a" * 64,
            source_format="jsonl",
            schema_name="text",
            example_count=10,
            total_estimated_tokens=100,
            sequence_p50=10,
            sequence_p95=12,
            sequence_max=15,
            measurement=MeasurementKind.ESTIMATED,
            warnings=("Tokenizer not supplied.",),
        )
        target = TrainingTarget(
            objective=Objective.MEMORY,
            sequence_length=512,
            effective_batch_size=16,
            max_epochs=3,
            method_preference=Method.QLORA,
        )

        value = to_primitive({"profile": profile, "target": target})

        self.assertEqual(value["profile"]["source_path"], "/tmp/data.jsonl")
        self.assertEqual(value["profile"]["measurement"], "estimated")
        self.assertEqual(value["target"]["objective"], "memory")
        self.assertEqual(value["target"]["method_preference"], "qlora")


if __name__ == "__main__":
    unittest.main()
