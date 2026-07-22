import json
import tempfile
import unittest
from pathlib import Path

from aptus.domain import (
    Backend,
    DeviceSpec,
    HardwareSpec,
    ModelSpec,
    gibibytes,
    to_primitive,
    training_plan_from_primitive,
)

from tests.aptus.helpers import make_plan


class DomainContractTests(unittest.TestCase):
    def test_gibibytes_uses_binary_units(self) -> None:
        self.assertEqual(gibibytes(1), 1024**3)

    def test_hardware_uses_limiting_per_device_vram_minus_user_reserve(self) -> None:
        hardware = HardwareSpec(
            devices=(
                DeviceSpec("GPU 0", Backend.CUDA, gibibytes(24), True, True),
                DeviceSpec("GPU 1", Backend.CUDA, gibibytes(16), True, True),
            ),
            host_ram_bytes=gibibytes(64),
            reserve_per_device_bytes=gibibytes(2),
        )
        self.assertEqual(hardware.limiting_vram_bytes, gibibytes(14))

    def test_model_requires_positive_named_facts_and_immutable_revision(self) -> None:
        common = dict(
            model_id="example/model",
            family="llama",
            parameters=1_000_000_000,
            hidden_size=1024,
            layers=16,
            context_length=2048,
            license_name="apache-2.0",
            training_allowed=True,
        )
        with self.assertRaisesRegex(ValueError, "immutable"):
            ModelSpec(revision="main", **common)
        with self.assertRaisesRegex(ValueError, "parameters"):
            ModelSpec(revision="a" * 40, **{**common, "parameters": 0})
        for local_path in (
            "/models/example",
            "../models/example",
            "C:\\models\\example",
        ):
            with self.subTest(local_path=local_path):
                with self.assertRaisesRegex(ValueError, "provider repository"):
                    ModelSpec(
                        revision="a" * 40,
                        **{**common, "model_id": local_path},
                    )

    def test_memory_json_exposes_transparent_point_upper_and_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary))
            memory = to_primitive(plan.recommended.memory)
        point_names = (
            "base_weights_bytes",
            "quantization_metadata_bytes",
            "adapter_weights_bytes",
            "adapter_gradients_bytes",
            "optimizer_states_bytes",
            "activations_bytes",
            "temporary_overhead_bytes",
            "communication_bytes",
            "workspace_bytes",
            "allocator_bytes",
            "load_transient_bytes",
        )
        self.assertEqual(
            memory["point_estimate_bytes"], sum(memory[name] for name in point_names)
        )
        self.assertEqual(
            memory["upper_estimate_bytes"],
            sum(memory["component_upper_bounds"].values()),
        )
        self.assertEqual(memory["uncertainty_bytes"], memory["safety_margin_bytes"])
        self.assertEqual(memory["formula_version"], "aptus-memory-v2")

    def test_plan_round_trips_through_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary))
            payload = json.loads(json.dumps(to_primitive(plan), allow_nan=False))
            restored = training_plan_from_primitive(payload)
        self.assertEqual(restored, plan)


if __name__ == "__main__":
    unittest.main()
