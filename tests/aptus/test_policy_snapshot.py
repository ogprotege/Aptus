from __future__ import annotations

import copy
import unittest

from aptus.policy_snapshot import (
    evaluate_model_policy_snapshot,
    model_policy_snapshot_bytes,
    model_policy_snapshot_payload,
    model_policy_snapshot_sha256,
    validate_model_policy_snapshot,
)


def _registry() -> dict:
    reasons = {
        "identity": "identity blocked",
        "layout": "layout blocked",
        "topology": "topology blocked",
        "shared": "shared expert blocked",
        "four_bit": "four bit blocked",
        "invalid": "invalid facts",
        "matched": "matched",
        "dense": "dense recognized",
        "sparse": "sparse unreviewed",
        "unknown": "unknown",
    }
    return {
        "compatibility_schema_version": "aptus.model-compatibility.v2",
        "dense_families": ["llama", "qwen"],
        "sparse_identity_markers": ["mixtral", "moe"],
        "reasons": reasons,
        "policies": [
            {
                "policy_id": "model.qwen3-moe.mlx-qlora",
                "policy_version": "1.0.0",
                "family": "qwen3_moe",
                "claims": {
                    "any_identity": {
                        "family": ["qwen3_moe"],
                        "model_type": ["qwen3_moe"],
                        "architecture": ["Qwen3MoeForCausalLM"],
                    }
                },
                "constraints": [
                    {
                        "kind": "exact_identity",
                        "values": {
                            "family": "qwen3_moe",
                            "model_type": "qwen3_moe",
                            "architecture": "Qwen3MoeForCausalLM",
                        },
                        "reason": "identity",
                        "reason_code": "identity-mismatch",
                    },
                    {
                        "kind": "quantization_layout",
                        "default_bits": 4,
                        "default_group_size": 64,
                        "override_module_template": "model.layers.{layer}.mlp.gate",
                        "override_bits": 8,
                        "override_group_size": 64,
                        "reason": "layout",
                        "reason_code": "quantization-layout-mismatch",
                    },
                    {
                        "kind": "sparse_topology",
                        "reason": "topology",
                        "reason_code": "topology-incomplete",
                    },
                    {
                        "kind": "no_shared_expert",
                        "reason": "shared",
                        "reason_code": "shared-expert-unsupported",
                    },
                    {
                        "kind": "field_equals",
                        "field": "quantization_bits",
                        "value": 4,
                        "reason": "four_bit",
                        "reason_code": "four-bit-required",
                    },
                ],
                "paths": [{"path_id": "mlx-lm.qlora.single.attention-qkvo.v1"}],
                "matched_reason": "matched",
                "matched_reason_codes": [
                    "exact-reviewed-artifact",
                    "pilot-not-yet-proven",
                ],
                "evidence_ids": ["policy.qwen3-moe.mlx-qlora.v1"],
            }
        ],
    }


def _subject() -> dict:
    return {
        "family": "qwen3_moe",
        "model_type": "qwen3_moe",
        "architecture": "Qwen3MoeForCausalLM",
        "layers": 2,
        "quantization_bits": 4,
        "quantization_layout": {
            "default_bits": 4,
            "default_group_size": 64,
            "module_overrides": [
                {"module_path": "model.layers.0.mlp.gate", "bits": 8, "group_size": 64},
                {"module_path": "model.layers.1.mlp.gate", "bits": 8, "group_size": 64},
            ],
        },
        "moe": {
            "expert_count": 8,
            "experts_per_token": 2,
            "expert_intermediate_size": 64,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "shared_expert_intermediate_size": None,
        },
        "fact_errors": [],
    }


class PolicySnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = model_policy_snapshot_payload(_registry())

    def test_canonical_bytes_and_digest_are_stable(self) -> None:
        reversed_registry = dict(reversed(list(_registry().items())))
        other = model_policy_snapshot_payload(reversed_registry)
        self.assertEqual(self.snapshot, other)
        encoded = model_policy_snapshot_bytes(self.snapshot)
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertFalse(encoded.endswith(b"\n\n"))
        self.assertEqual(encoded, model_policy_snapshot_bytes(other))
        self.assertEqual(len(model_policy_snapshot_sha256(self.snapshot)), 64)

    def test_validation_rejects_unknown_constraint_and_malformed_snapshot(self) -> None:
        validate_model_policy_snapshot(self.snapshot)
        malformed = copy.deepcopy(self.snapshot)
        malformed["policies"][0]["constraints"][0]["kind"] = "python_callback"
        with self.assertRaisesRegex(ValueError, "constraint"):
            validate_model_policy_snapshot(malformed)
        missing = copy.deepcopy(self.snapshot)
        del missing["reasons"]
        with self.assertRaises(ValueError):
            validate_model_policy_snapshot(missing)

    def test_generic_evaluator_covers_policy_and_fallback_states(self) -> None:
        matched = evaluate_model_policy_snapshot(self.snapshot, _subject())
        self.assertEqual(matched["kind"], "path-matched")
        self.assertEqual(matched["policy_id"], "model.qwen3-moe.mlx-qlora")
        self.assertEqual(
            matched["paths"], [{"path_id": "mlx-lm.qlora.single.attention-qkvo.v1"}]
        )
        cases = [
            (
                "identity",
                ("identity", {"model_type": "qwen3", "architecture": "Other"}),
                "identity-mismatch",
            ),
            ("layout", ("quantization_layout", None), "quantization-layout-mismatch"),
            ("topology", ("moe", None), "topology-incomplete"),
            (
                "shared",
                ("moe.shared_expert_intermediate_size", 32),
                "shared-expert-unsupported",
            ),
            ("four bit", ("quantization_bits", 8), "four-bit-required"),
        ]
        for name, (key, value), reason_code in cases:
            with self.subTest(name=name):
                subject = copy.deepcopy(_subject())
                if key == "identity":
                    subject.update(value)
                    decision = evaluate_model_policy_snapshot(self.snapshot, subject)
                    self.assertEqual(decision["kind"], "blocked")
                    self.assertEqual(decision["reason_codes"], [reason_code])
                    continue
                if "." in key:
                    outer, inner = key.split(".")
                    subject[outer][inner] = value
                else:
                    subject[key] = value
                decision = evaluate_model_policy_snapshot(self.snapshot, subject)
                self.assertEqual(decision["kind"], "blocked")
                self.assertEqual(decision["reason_codes"], [reason_code])

        dense = {
            **_subject(),
            "family": "llama",
            "model_type": "llama",
            "architecture": "LlamaForCausalLM",
            "moe": None,
        }
        unknown = {**dense, "family": "custom", "model_type": "custom"}
        sparse = {**unknown, "family": "mixtral"}
        self.assertEqual(
            evaluate_model_policy_snapshot(self.snapshot, dense)["kind"],
            "family-recognized",
        )
        self.assertEqual(
            evaluate_model_policy_snapshot(self.snapshot, sparse)["reason_codes"],
            ["unreviewed-sparse-model"],
        )
        self.assertEqual(
            evaluate_model_policy_snapshot(self.snapshot, unknown)["kind"], "unknown"
        )

    def test_fact_errors_fail_closed(self) -> None:
        subject = _subject()
        subject["fact_errors"] = ["quantization: contradictory"]
        decision = evaluate_model_policy_snapshot(self.snapshot, subject)
        self.assertEqual(decision["kind"], "blocked")
        self.assertEqual(decision["reason_codes"], ["invalid-compatibility-facts"])

    def test_subject_identity_uses_only_normalized_compatibility_fields(self) -> None:
        subject = _subject()
        subject["fact_errors"] = [
            "quantization: conflicting provider facts",
            "moe: conflicting provider facts",
        ]
        reordered = copy.deepcopy(subject)
        reordered["fact_errors"] = list(reversed(subject["fact_errors"]))
        with_metadata = copy.deepcopy(subject)
        with_metadata.update(
            context_length=4096,
            parameters=1_000_000,
            future_fact={"ignored": True},
        )

        expected = evaluate_model_policy_snapshot(self.snapshot, subject)
        for name, candidate in {
            "reordered errors": reordered,
            "caller-only metadata": with_metadata,
        }.items():
            with self.subTest(name=name):
                self.assertEqual(
                    evaluate_model_policy_snapshot(self.snapshot, candidate),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
