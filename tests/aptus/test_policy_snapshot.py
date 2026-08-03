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
                "paths": [
                    {
                        "path_id": "mlx-lm.qlora.single.attention-qkvo.v1",
                        "method": "qlora",
                        "distribution": "single",
                        "adapter_profile_id": "attention-qkvo.v1",
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "runtime_contract": {
                            "compute_backend": "mps",
                            "training_runtime": "mlx-lm",
                            "compiler_id": "mlx-lm.qlora.v1",
                            "estimator_id": "aptus-memory-mlx-v2",
                            "evidence_requirement": "pilot-required",
                            "export_kind": "mlx-lm-adapter",
                            "schema_version": "aptus.runtime-contract.v1",
                        },
                        "required_validation_levels": [
                            "model-data",
                            "measured-preflight",
                            "pilot",
                        ],
                        "evidence_ids": ["policy.qwen3-moe.mlx-qlora.v1"],
                    }
                ],
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

    def _constraint(self, snapshot: dict, kind: str) -> dict:
        return next(
            constraint
            for constraint in snapshot["policies"][0]["constraints"]
            if constraint["kind"] == kind
        )

    def _assert_evaluation_rejects(self, snapshot: dict, pattern: str) -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            evaluate_model_policy_snapshot(snapshot, _subject())

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

    def test_validation_rejects_malformed_quantization_constraint_operands(
        self,
    ) -> None:
        for field in (
            "default_bits",
            "default_group_size",
            "override_bits",
            "override_group_size",
        ):
            for value in (None, True, 0, -1, 1.5, "4"):
                with self.subTest(field=field, value=value):
                    malformed = copy.deepcopy(self.snapshot)
                    self._constraint(malformed, "quantization_layout")[field] = value
                    self._assert_evaluation_rejects(malformed, "positive integer")

    def test_validation_rejects_malformed_quantization_module_templates(self) -> None:
        for value in (
            42,
            "",
            "model.layers.gate",
            "model.layers.{other}.gate",
            "model.layers.{layer.gate}",
            "model.layers.{layer[0]}.gate",
            "model.layers.{layer!r}.gate",
            "model.layers.{layer:03d}.gate",
            "model.layers.{layer}.{layer}.gate",
            "model.{{layers}}.{layer}.gate",
            " model.layers.{layer}.gate",
            "model.layers.{}.gate",
            "model.layers.{layer:{width}}.gate",
            "model.layers.{layer.gate",
        ):
            with self.subTest(value=value):
                malformed = copy.deepcopy(self.snapshot)
                self._constraint(malformed, "quantization_layout")[
                    "override_module_template"
                ] = value
                self._assert_evaluation_rejects(malformed, "module template")

    def test_validation_rejects_malformed_field_and_identity_operands(self) -> None:
        for value in (None, "", " ", " quantization_bits", [], {}):
            with self.subTest(kind="field_equals", value=value):
                malformed = copy.deepcopy(self.snapshot)
                self._constraint(malformed, "field_equals")["field"] = value
                self._assert_evaluation_rejects(malformed, "field")

        for field in ("family", "model_type", "architecture"):
            for value in (
                None,
                "",
                " ",
                f" {self._constraint(self.snapshot, 'exact_identity')['values'][field]}",
                [],
                {},
            ):
                with self.subTest(kind="exact_identity", field=field, value=value):
                    malformed = copy.deepcopy(self.snapshot)
                    self._constraint(malformed, "exact_identity")["values"][field] = (
                        value
                    )
                    self._assert_evaluation_rejects(malformed, "exact identity")

    def test_validation_requires_exactly_one_exact_identity_constraint(self) -> None:
        missing = copy.deepcopy(self.snapshot)
        missing["policies"][0]["constraints"] = [
            constraint
            for constraint in missing["policies"][0]["constraints"]
            if constraint["kind"] != "exact_identity"
        ]
        missing_subject = _subject()
        missing_subject["fact_errors"] = ["quantization: contradictory"]
        with self.assertRaisesRegex(ValueError, "exact identity"):
            evaluate_model_policy_snapshot(missing, missing_subject)

        duplicate = copy.deepcopy(self.snapshot)
        duplicate["policies"][0]["constraints"].append(
            copy.deepcopy(self._constraint(duplicate, "exact_identity"))
        )
        self._assert_evaluation_rejects(duplicate, "exact identity")

    def test_validation_rejects_malformed_claim_shapes_and_references(self) -> None:
        mutations = {
            "claims list": lambda policy: policy.update(claims=[]),
            "unknown claims key": lambda policy: policy["claims"].update(
                all_identity={}
            ),
            "claim value is not text": lambda policy: policy["claims"]["any_identity"][
                "family"
            ].append(42),
            "claim value is empty": lambda policy: policy["claims"]["any_identity"][
                "family"
            ].append(""),
            "claim value is whitespace": lambda policy: policy["claims"][
                "any_identity"
            ]["family"].append(" "),
            "claim value is padded": lambda policy: policy["claims"]["any_identity"][
                "family"
            ].append(" qwen3_moe"),
            "duplicate claim value": lambda policy: policy["claims"]["any_identity"][
                "family"
            ].append("qwen3_moe"),
            "claim identity field missing": lambda policy: policy["claims"][
                "any_identity"
            ].pop("architecture"),
            "identity not claimed": lambda policy: policy["claims"]["any_identity"][
                "family"
            ].remove("qwen3_moe"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                malformed = copy.deepcopy(self.snapshot)
                mutate(malformed["policies"][0])
                self._assert_evaluation_rejects(malformed, "claim")

    def test_validation_rejects_malformed_paths_reasons_and_evidence(self) -> None:
        mutations = {
            "path is not a mapping": lambda policy, snapshot: policy["paths"].append(
                "path-id"
            ),
            "path id missing": lambda policy, snapshot: policy["paths"].append({}),
            "path id empty": lambda policy, snapshot: policy["paths"][0].update(
                path_id=""
            ),
            "path id whitespace": lambda policy, snapshot: policy["paths"][0].update(
                path_id=" "
            ),
            "path id non-text": lambda policy, snapshot: policy["paths"][0].update(
                path_id=[]
            ),
            "duplicate path id": lambda policy, snapshot: policy["paths"].append(
                copy.deepcopy(policy["paths"][0])
            ),
            "path method non-text": lambda policy, snapshot: policy["paths"][0].update(
                method=[]
            ),
            "path distribution empty": lambda policy, snapshot: policy["paths"][
                0
            ].update(distribution=""),
            "path adapter profile padded": lambda policy, snapshot: policy["paths"][
                0
            ].update(adapter_profile_id=" profile"),
            "path adapter profile non-text": lambda policy, snapshot: policy["paths"][
                0
            ].update(adapter_profile_id=[]),
            "path target modules malformed": lambda policy, snapshot: policy["paths"][
                0
            ].update(target_modules="q_proj"),
            "path target modules duplicate": lambda policy, snapshot: policy["paths"][
                0
            ].update(target_modules=["q_proj", "q_proj"]),
            "path runtime contract malformed": lambda policy, snapshot: policy["paths"][
                0
            ].update(runtime_contract=[]),
            "path runtime contract value non-text": lambda policy, snapshot: policy[
                "paths"
            ][0].update(runtime_contract={"compiler_id": []}),
            "path validation levels empty": lambda policy, snapshot: policy["paths"][
                0
            ].update(required_validation_levels=[]),
            "constraint reason missing": lambda policy, snapshot: policy["constraints"][
                0
            ].update(reason="not-defined"),
            "matched reason missing": lambda policy, snapshot: policy.update(
                matched_reason="not-defined"
            ),
            "reason code non-text": lambda policy, snapshot: policy["constraints"][
                0
            ].update(reason_code=[]),
            "reason code whitespace": lambda policy, snapshot: policy["constraints"][
                0
            ].update(reason_code=" "),
            "matched reason code empty": lambda policy, snapshot: policy[
                "matched_reason_codes"
            ].append(""),
            "matched reason code duplicate": lambda policy, snapshot: policy[
                "matched_reason_codes"
            ].append(policy["matched_reason_codes"][0]),
            "evidence id non-text": lambda policy, snapshot: policy[
                "evidence_ids"
            ].append({}),
            "evidence id whitespace": lambda policy, snapshot: policy[
                "evidence_ids"
            ].append(" "),
            "evidence id duplicate": lambda policy, snapshot: policy[
                "evidence_ids"
            ].append(policy["evidence_ids"][0]),
            "path evidence id empty": lambda policy, snapshot: policy["paths"][
                0
            ].update(evidence_ids=[""]),
            "path evidence id duplicate": lambda policy, snapshot: policy["paths"][
                0
            ].update(evidence_ids=["evidence", "evidence"]),
            "reason text non-text": lambda policy, snapshot: snapshot["reasons"].update(
                matched=[]
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                malformed = copy.deepcopy(self.snapshot)
                mutate(malformed["policies"][0], malformed)
                self._assert_evaluation_rejects(
                    malformed,
                    "path|reason|evidence",
                )

    def test_validation_requires_exact_path_and_runtime_contract_shapes(self) -> None:
        path_fields = set(self.snapshot["policies"][0]["paths"][0])
        for field in path_fields:
            with self.subTest(missing_path_field=field):
                malformed = copy.deepcopy(self.snapshot)
                del malformed["policies"][0]["paths"][0][field]
                self._assert_evaluation_rejects(malformed, "path")

        runtime_fields = set(
            self.snapshot["policies"][0]["paths"][0]["runtime_contract"]
        )
        for field in runtime_fields:
            with self.subTest(missing_runtime_field=field):
                malformed = copy.deepcopy(self.snapshot)
                del malformed["policies"][0]["paths"][0]["runtime_contract"][field]
                self._assert_evaluation_rejects(malformed, "runtime contract")
            with self.subTest(non_text_runtime_field=field):
                malformed = copy.deepcopy(self.snapshot)
                malformed["policies"][0]["paths"][0]["runtime_contract"][field] = []
                self._assert_evaluation_rejects(malformed, "runtime contract")

        mutations = {
            "extra path field": lambda path: path.update(unexpected="value"),
            "extra runtime field": lambda path: path["runtime_contract"].update(
                unexpected="value"
            ),
            "export kind padded": lambda path: path["runtime_contract"].update(
                export_kind=" export"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                malformed = copy.deepcopy(self.snapshot)
                mutate(malformed["policies"][0]["paths"][0])
                self._assert_evaluation_rejects(malformed, "path|runtime contract")

        nullable = copy.deepcopy(self.snapshot)
        runtime_contract = nullable["policies"][0]["paths"][0]["runtime_contract"]
        runtime_contract["compiler_id"] = None
        runtime_contract["export_kind"] = None
        validate_model_policy_snapshot(nullable)

    def test_validation_rejects_malformed_constraint_kinds_and_provenance(self) -> None:
        for value in ([], {}):
            with self.subTest(kind=value):
                malformed = copy.deepcopy(self.snapshot)
                malformed["policies"][0]["constraints"][0]["kind"] = value
                self._assert_evaluation_rejects(malformed, "constraint")

        unexpected = copy.deepcopy(self.snapshot)
        unexpected["policies"][0]["constraints"][0]["unexpected"] = "operand"
        self._assert_evaluation_rejects(unexpected, "constraint shape")

        for value in ([], [""], [" "], ["layers", "layers"], "layers"):
            with self.subTest(required_provenance_fields=value):
                malformed = copy.deepcopy(self.snapshot)
                malformed["policies"][0]["required_provenance_fields"] = value
                self._assert_evaluation_rejects(malformed, "provenance")

    def test_generic_evaluator_covers_policy_and_fallback_states(self) -> None:
        matched = evaluate_model_policy_snapshot(self.snapshot, _subject())
        self.assertEqual(matched["kind"], "path-matched")
        self.assertEqual(matched["policy_id"], "model.qwen3-moe.mlx-qlora")
        self.assertEqual(matched["paths"], self.snapshot["policies"][0]["paths"])
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
