import unittest

from aptus.attestation import require_trainable_parameter_census


def valid_lora_census() -> dict:
    return {
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


class TrainableCensusAttestationTests(unittest.TestCase):
    def test_accepts_complete_plan_bound_adapter_census(self) -> None:
        census = valid_lora_census()
        self.assertIs(require_trainable_parameter_census(census, method="lora"), census)

    def test_rejects_boolean_counter_and_truthy_integer_finiteness(self) -> None:
        for field, value in (
            ("unexpected_trainable_tensor_count", False),
            ("frozen_tensor_count", True),
            ("all_values_finite", 1),
        ):
            with self.subTest(field=field):
                census = {**valid_lora_census(), field: value}
                with self.assertRaises(ValueError):
                    require_trainable_parameter_census(census, method="lora")

    def test_rejects_incomplete_or_unbound_adapter_targets(self) -> None:
        for overrides in (
            {"adapter_target_instance_count": 2},
            {"incomplete_adapter_target_instance_count": 1},
            {"expected_adapter_target_match_count": 0},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    require_trainable_parameter_census(
                        {**valid_lora_census(), **overrides}, method="lora"
                    )


if __name__ == "__main__":
    unittest.main()
