import json
import tempfile
import unittest
from pathlib import Path

from aptus.domain import (
    Backend,
    DeviceSpec,
    HardwareSpec,
    MoETopology,
    ModelSpec,
    QuantizationLayout,
    QuantizationOverride,
    UnsupportedPlanSchemaError,
    gibibytes,
    to_primitive,
    training_plan_from_primitive,
)

from tests.aptus.helpers import make_plan, make_qwen3_moe_plan


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

    def test_moe_topology_derives_sparse_and_active_parameters(self) -> None:
        topology = MoETopology(
            expert_count=128,
            experts_per_token=8,
            expert_intermediate_size=768,
            decoder_sparse_step=2,
            mlp_only_layers=(1,),
        )
        model = ModelSpec(
            model_id="example/qwen3-moe",
            revision="c" * 40,
            family="qwen3_moe",
            parameters=30_500_000_000,
            hidden_size=2048,
            intermediate_size=6144,
            layers=48,
            context_length=262144,
            license_name="apache-2.0",
            training_allowed=True,
            architecture="Qwen3MoeForCausalLM",
            model_type="qwen3_moe",
            quantization_bits=4,
            moe=topology,
        )

        self.assertEqual(model.sparse_layer_count, 23)
        expected_active = 30_500_000_000 - 23 * 120 * 3 * 2048 * 768
        self.assertEqual(model.active_parameters, expected_active)
        primitive = to_primitive(model)
        self.assertEqual(primitive["sparse_layer_count"], 23)
        self.assertEqual(primitive["active_parameters"], expected_active)

    def test_moe_topology_rejects_invalid_routing(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            MoETopology(
                expert_count=4,
                experts_per_token=5,
                expert_intermediate_size=128,
                decoder_sparse_step=1,
            )
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            MoETopology(
                expert_count=4,
                experts_per_token=2,
                expert_intermediate_size=128,
                decoder_sparse_step=1,
                mlp_only_layers=(2, 1),
            )

    def test_quantization_layout_requires_canonical_unique_overrides(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            QuantizationLayout(
                default_bits=4,
                default_group_size=64,
                module_overrides=(
                    QuantizationOverride("model.layers.1.mlp.gate", 8, 64),
                    QuantizationOverride("model.layers.0.mlp.gate", 8, 64),
                ),
            )
        with self.assertRaisesRegex(ValueError, "must equal"):
            ModelSpec(
                model_id="example/quantized",
                revision="d" * 40,
                family="qwen",
                parameters=1_000_000_000,
                hidden_size=1024,
                layers=16,
                context_length=2048,
                license_name="apache-2.0",
                training_allowed=True,
                quantization_bits=8,
                quantization_layout=QuantizationLayout(4, 64),
            )

    def test_plan_round_trips_through_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary))
            payload = json.loads(json.dumps(to_primitive(plan), allow_nan=False))
            restored = training_plan_from_primitive(payload)
        self.assertEqual(restored, plan)

    def test_plan_requires_lowercase_policy_snapshot_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary))
        for invalid_digest in ("", "A" * 64, "g" * 64, "a" * 63):
            with self.subTest(digest=invalid_digest):
                with self.assertRaisesRegex(ValueError, "snapshot SHA-256"):
                    plan.__class__(
                        **{
                            **plan.__dict__,
                            "model_policy_snapshot_sha256": invalid_digest,
                        }
                    )

    def test_persisted_plan_requires_policy_snapshot_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = to_primitive(make_plan(Path(temporary)))
        payload.pop("model_policy_snapshot_sha256")
        with self.assertRaisesRegex(ValueError, "model_policy_snapshot_sha256"):
            training_plan_from_primitive(payload)

    def test_moe_quantization_layout_round_trips_through_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_qwen3_moe_plan(Path(temporary))
            payload = json.loads(json.dumps(to_primitive(plan), allow_nan=False))
            restored = training_plan_from_primitive(payload)
        self.assertEqual(restored, plan)

    def test_legacy_plan_requires_replan_without_mutating_source(self) -> None:
        for legacy_schema in (
            "aptus.training-plan.v4",
            "aptus.training-plan.v3",
            "aptus.training-plan.v2",
            None,
        ):
            with self.subTest(schema=legacy_schema):
                with tempfile.TemporaryDirectory() as temporary:
                    payload = to_primitive(make_plan(Path(temporary)))
                    if legacy_schema is None:
                        payload.pop("schema_version")
                    else:
                        payload["schema_version"] = legacy_schema
                    before = json.dumps(payload, sort_keys=True, separators=(",", ":"))

                    with self.assertRaises(UnsupportedPlanSchemaError) as raised:
                        training_plan_from_primitive(payload)

                    after = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                self.assertEqual(raised.exception.found_schema, legacy_schema)
                self.assertEqual(
                    raised.exception.required_schema, "aptus.training-plan.v5"
                )
                self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
