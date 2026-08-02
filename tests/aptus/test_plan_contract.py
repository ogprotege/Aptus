import copy
from dataclasses import replace
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aptus.domain import (
    Backend,
    Objective,
    TrainingRuntime,
    TrainingTarget,
    to_primitive,
)
from aptus.methods import selectable_method_descriptors
from aptus.model_compatibility import (
    create_model_inspection_receipt,
    current_model_policy_snapshot,
    current_model_policy_snapshot_bytes,
    evaluate_model_compatibility,
    subject_from_model,
)
from aptus.plan_contract import (
    MODEL_TARGET_MODULES,
    RUNTIME_BINDING_IDENTITIES,
    StaleModelPolicyError,
    _current_model_policy_decision,
    candidate_id_for_payload,
    expected_model_architecture_contract,
    plan_id_for_payload,
    require_current_model_policy,
    validate_model_config_against_plan,
    validate_plan_payload,
)
from aptus.planning import plan_training
from aptus.profiling import (
    build_hardware_spec,
    build_model_spec,
    profile_dataset,
)

from tests.aptus.helpers import (
    make_plan,
    make_qwen3_moe_plan,
    qwen3_moe_quantization_config,
)


class PlanContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.payload = to_primitive(make_plan(self.root))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_real_v5_plan_is_valid(self) -> None:
        self.assertEqual(validate_plan_payload(self.payload, verify_dataset=True), ())
        self.assertEqual(self.payload["schema_version"], "aptus.training-plan.v5")

    def test_installed_host_uses_current_policy_before_bundle_snapshot(self) -> None:
        snapshot_path = self.root / "policy" / "model-policy-snapshot.v1.json"
        snapshot_path.parent.mkdir()
        snapshot_path.write_bytes(current_model_policy_snapshot_bytes())
        changed_snapshot = copy.deepcopy(current_model_policy_snapshot())
        changed_snapshot["dense_families"] = sorted(
            [*changed_snapshot["dense_families"], "future-dense-family"]
        )

        with patch(
            "aptus.model_compatibility.current_model_policy_snapshot",
            return_value=changed_snapshot,
        ):
            errors = validate_plan_payload(
                self.payload,
                root=self.root,
                verify_dataset=False,
            )

        self.assertTrue(any("replan_required" in error for error in errors), errors)

    def test_explicit_policy_snapshot_overrides_current_host_policy(self) -> None:
        snapshot = current_model_policy_snapshot()
        changed_snapshot = copy.deepcopy(snapshot)
        changed_snapshot["dense_families"] = sorted(
            [*changed_snapshot["dense_families"], "future-dense-family"]
        )

        with patch(
            "aptus.model_compatibility.current_model_policy_snapshot",
            return_value=changed_snapshot,
        ):
            errors = validate_plan_payload(
                self.payload,
                verify_dataset=False,
                policy_snapshot=snapshot,
            )

        self.assertEqual(errors, ())

    def test_host_and_portable_policy_decisions_match_every_supported_state(
        self,
    ) -> None:
        qwen_model = make_qwen3_moe_plan(self.root).model
        qwen_subject = subject_from_model(qwen_model)
        qwen_payload = to_primitive(qwen_model)
        assert qwen_subject.quantization_layout is not None
        assert qwen_subject.moe is not None

        dense_model = make_plan(self.root).model
        dense_subject = subject_from_model(dense_model)
        dense_payload = to_primitive(dense_model)

        cases = [("exact-qwen", qwen_subject, qwen_payload)]

        identity_subject = replace(qwen_subject, architecture="OtherForCausalLM")
        identity_payload = copy.deepcopy(qwen_payload)
        identity_payload["architecture"] = "OtherForCausalLM"
        cases.append(("identity-mismatch", identity_subject, identity_payload))

        layout = replace(qwen_subject.quantization_layout, default_group_size=128)
        layout_subject = replace(qwen_subject, quantization_layout=layout)
        layout_payload = copy.deepcopy(qwen_payload)
        layout_payload["quantization_layout"]["default_group_size"] = 128
        cases.append(("layout-mismatch", layout_subject, layout_payload))

        topology = replace(qwen_subject.moe, decoder_sparse_step=49)
        topology_subject = replace(qwen_subject, moe=topology)
        topology_payload = copy.deepcopy(qwen_payload)
        topology_payload["moe"]["decoder_sparse_step"] = 49
        cases.append(("topology-incomplete", topology_subject, topology_payload))

        shared = replace(qwen_subject.moe, shared_expert_intermediate_size=256)
        shared_subject = replace(qwen_subject, moe=shared)
        shared_payload = copy.deepcopy(qwen_payload)
        shared_payload["moe"]["shared_expert_intermediate_size"] = 256
        cases.append(("shared-expert", shared_subject, shared_payload))

        four_bit_subject = replace(qwen_subject, quantization_bits=8)
        four_bit_payload = copy.deepcopy(qwen_payload)
        four_bit_payload["quantization_bits"] = 8
        cases.append(("four-bit-required", four_bit_subject, four_bit_payload))

        cases.append(("dense-recognized", dense_subject, dense_payload))

        sparse_subject = replace(
            dense_subject,
            family="mixtral",
            model_type="mixtral",
            architecture="MixtralForCausalLM",
        )
        sparse_payload = copy.deepcopy(dense_payload)
        sparse_payload.update(
            {
                "family": "mixtral",
                "model_type": "mixtral",
                "architecture": "MixtralForCausalLM",
            }
        )
        cases.append(("unreviewed-sparse", sparse_subject, sparse_payload))

        unknown_subject = replace(
            dense_subject,
            family="unknown-family",
            model_type=None,
            architecture=None,
        )
        unknown_payload = copy.deepcopy(dense_payload)
        unknown_payload.update(
            {
                "family": "unknown-family",
                "model_type": None,
                "architecture": None,
            }
        )
        cases.append(("unknown", unknown_subject, unknown_payload))

        for label, subject, model in cases:
            with self.subTest(label=label):
                self.assertEqual(
                    _current_model_policy_decision(
                        model, current_model_policy_snapshot()
                    ),
                    to_primitive(evaluate_model_compatibility(subject)),
                )

    def _provider_qwen_payload(self) -> dict:
        base = make_qwen3_moe_plan(self.root)
        model = base.model
        facts = to_primitive(model)
        observed_at = "2026-07-29T12:00:00+00:00"
        receipt_fields = (
            "architecture",
            "context_length",
            "family",
            "hidden_size",
            "intermediate_size",
            "layers",
            "license_name",
            "model_type",
            "moe",
            "quantization_bits",
            "quantization_layout",
        )
        provenance = {
            field: {
                "kind": "inferred" if field == "family" else "provider-declared",
                "source": "https://huggingface.co/provider/pinned-config",
                "observed_at": observed_at,
                "resolved_revision": model.revision,
            }
            for field in receipt_fields
        }
        receipt = create_model_inspection_receipt(
            model_id=model.model_id,
            resolved_revision=model.revision,
            facts=facts,
            provenance=provenance,
            subject=subject_from_model(model),
            evaluated_at=observed_at,
        )
        return to_primitive(
            plan_training(
                model=model,
                dataset=base.dataset,
                hardware=base.hardware,
                target=base.target,
                inspection_receipt=receipt,
            )
        )

    def test_pre_v4_plan_requires_replanning(self) -> None:
        value = copy.deepcopy(self.payload)
        value["schema_version"] = "aptus.training-plan.v3"
        value["plan_id"] = plan_id_for_payload(value)

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(any("replan_required" in item for item in errors), errors)

    def test_recomputes_current_policy_after_consistent_plan_id_tampering(self) -> None:
        value = copy.deepcopy(self.payload)
        value["model_policy_decision"]["reason_codes"] = ["no-policy-match"]
        value["plan_id"] = plan_id_for_payload(value)

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(any("current registered policy" in item for item in errors))
        self.assertFalse(any("Plan immutable ID" in item for item in errors))

    def test_stale_registered_policy_has_a_dedicated_replan_error(self) -> None:
        value = to_primitive(make_qwen3_moe_plan(self.root))
        decision = value["model_policy_decision"]
        recommended_key = (
            value["recommended"]["method"],
            value["recommended"]["distribution"],
        )
        decision["policy_version"] = "0.9.0"
        identity = {
            key: decision[key]
            for key in (
                "schema_version",
                "subject_facts_sha256",
                "kind",
                "family",
                "policy_id",
                "policy_version",
                "paths",
                "reason_codes",
                "evidence_ids",
            )
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        decision["decision_id"] = "compat_" + hashlib.sha256(encoded).hexdigest()[:20]
        for candidate in value["candidates"]:
            candidate["model_policy_decision_id"] = decision["decision_id"]
            binding = candidate["policy_binding"]
            if binding is not None:
                binding["decision_id"] = decision["decision_id"]
                binding["policy_version"] = decision["policy_version"]
            candidate["candidate_id"] = candidate_id_for_payload(
                candidate,
                model=value["model"],
                dataset=value["dataset"],
                hardware=value["hardware"],
                target=value["target"],
            )
        value["recommended"] = copy.deepcopy(
            next(
                candidate
                for candidate in value["candidates"]
                if (candidate["method"], candidate["distribution"]) == recommended_key
            )
        )
        value["plan_id"] = plan_id_for_payload(value)

        with self.assertRaisesRegex(StaleModelPolicyError, "replan_required"):
            require_current_model_policy(value)

    def test_partial_stale_policy_rewrite_is_tampering_not_replan(self) -> None:
        value = to_primitive(make_qwen3_moe_plan(self.root))
        decision = value["model_policy_decision"]
        decision["policy_version"] = "0.9.0"
        identity = {
            key: decision[key]
            for key in (
                "schema_version",
                "subject_facts_sha256",
                "kind",
                "family",
                "policy_id",
                "policy_version",
                "paths",
                "reason_codes",
                "evidence_ids",
            )
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        decision["decision_id"] = "compat_" + hashlib.sha256(encoded).hexdigest()[:20]

        with self.assertRaises(ValueError) as caught:
            require_current_model_policy(value)

        self.assertNotIsInstance(caught.exception, StaleModelPolicyError)

    def test_internally_valid_plan_requires_replan_when_a_policy_is_added(self) -> None:
        value = copy.deepcopy(self.payload)
        future = copy.deepcopy(value["model_policy_decision"])
        future["policy_id"] = "model.llama.future-policy"
        future["policy_version"] = "1.0.0"
        future["reason_codes"] = ["exact-reviewed-artifact"]
        identity = {
            key: future[key]
            for key in (
                "schema_version",
                "subject_facts_sha256",
                "kind",
                "family",
                "policy_id",
                "policy_version",
                "paths",
                "reason_codes",
                "evidence_ids",
            )
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        future["decision_id"] = "compat_" + hashlib.sha256(encoded).hexdigest()[:20]

        with (
            patch(
                "aptus.plan_contract._current_model_policy_decision",
                return_value=future,
            ),
            patch.dict(MODEL_TARGET_MODULES, {"llama": ["future_proj"]}),
            self.assertRaisesRegex(StaleModelPolicyError, "replan_required"),
        ):
            require_current_model_policy(value)

    def test_generated_unknown_family_plan_round_trips_portable_validation(
        self,
    ) -> None:
        base = make_plan(self.root)
        plan = plan_training(
            model=replace(base.model, family="unregistered-family"),
            dataset=base.dataset,
            hardware=base.hardware,
            target=base.target,
        )
        value = to_primitive(plan)
        adapter_candidates = [
            candidate
            for candidate in value["candidates"]
            if candidate["method"] != "full"
        ]

        self.assertEqual(value["model_policy_decision"]["kind"], "unknown")
        self.assertTrue(adapter_candidates)
        self.assertTrue(
            all(
                candidate["status"] == "unsupported"
                and candidate["target_modules"] == []
                and candidate["checkpoint_retention_bytes"] == 0
                for candidate in adapter_candidates
            )
        )
        self.assertEqual(validate_plan_payload(value, verify_dataset=False), ())

    def test_malformed_json_scalar_types_return_errors_instead_of_raising(
        self,
    ) -> None:
        mutations = (
            ("model family", lambda value: value["model"].__setitem__("family", [])),
            (
                "target method",
                lambda value: value["target"].__setitem__("method_preference", []),
            ),
            (
                "target runtime",
                lambda value: value["target"].__setitem__("training_runtime", []),
            ),
            (
                "candidate method",
                lambda value: value["candidates"][0].__setitem__("method", []),
            ),
            (
                "candidate runtime",
                lambda value: value["candidates"][0]["runtime_contract"].__setitem__(
                    "training_runtime", []
                ),
            ),
            (
                "candidate status",
                lambda value: value["candidates"][0].__setitem__("status", []),
            ),
            (
                "oversized evaluation fraction",
                lambda value: value["target"].__setitem__(
                    "evaluation_fraction", 10**400
                ),
            ),
            (
                "negative oversized evaluation fraction",
                lambda value: value["target"].__setitem__(
                    "evaluation_fraction", -(10**400)
                ),
            ),
            (
                "oversized learning rate",
                lambda value: value["candidates"][0].__setitem__(
                    "learning_rate", 10**400
                ),
            ),
            (
                "negative oversized learning rate",
                lambda value: value["candidates"][0].__setitem__(
                    "learning_rate", -(10**400)
                ),
            ),
        )

        for label, mutate in mutations:
            with self.subTest(label=label):
                value = copy.deepcopy(self.payload)
                mutate(value)
                errors = validate_plan_payload(value, verify_dataset=False)
                self.assertTrue(errors)

    def test_uppercase_family_is_rejected_at_both_policy_boundaries(self) -> None:
        qwen_model = make_qwen3_moe_plan(self.root).model
        qwen_subject = subject_from_model(qwen_model)
        qwen_payload = to_primitive(qwen_model)
        qwen_payload["family"] = "QWEN3_MOE"

        with self.assertRaisesRegex(ValueError, "canonical lowercase"):
            replace(qwen_model, family="QWEN3_MOE")
        with self.assertRaisesRegex(ValueError, "canonical lowercase"):
            replace(qwen_subject, family="QWEN3_MOE")
        with self.assertRaisesRegex(ValueError, "canonical lowercase"):
            _current_model_policy_decision(qwen_payload)

    def test_coherently_reidentified_non_scalar_family_fails_validation(self) -> None:
        base = make_plan(self.root)
        value = to_primitive(
            plan_training(
                model=replace(base.model, family="unregistered-family"),
                dataset=base.dataset,
                hardware=base.hardware,
                target=base.target,
            )
        )
        recommended_key = (
            value["recommended"]["method"],
            value["recommended"]["distribution"],
        )
        value["model"]["family"] = ["unregistered-family"]
        decision = _current_model_policy_decision(value["model"])
        value["model_policy_decision"] = decision
        for candidate in value["candidates"]:
            candidate["model_policy_decision_id"] = decision["decision_id"]
            candidate["policy_binding"] = None
            candidate["candidate_id"] = candidate_id_for_payload(
                candidate,
                model=value["model"],
                dataset=value["dataset"],
                hardware=value["hardware"],
                target=value["target"],
            )
        value["recommended"] = copy.deepcopy(
            next(
                candidate
                for candidate in value["candidates"]
                if (candidate["method"], candidate["distribution"]) == recommended_key
            )
        )
        value["plan_id"] = plan_id_for_payload(value)

        require_current_model_policy(value)
        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(any("Model family is required" in item for item in errors))

    def test_policy_tampering_is_not_classified_as_a_stale_version(self) -> None:
        value = to_primitive(make_qwen3_moe_plan(self.root))
        value["model_policy_decision"]["reason_codes"] = ["no-policy-match"]

        with self.assertRaises(ValueError) as caught:
            require_current_model_policy(value)

        self.assertNotIsInstance(caught.exception, StaleModelPolicyError)

    def test_explanatory_policy_prose_is_not_part_of_identity(self) -> None:
        value = copy.deepcopy(self.payload)
        original_plan_id = value["plan_id"]
        value["model_policy_decision"]["reason"] = "Clearer explanatory prose."

        self.assertEqual(plan_id_for_payload(value), original_plan_id)
        self.assertEqual(validate_plan_payload(value, verify_dataset=False), ())

    def test_evidence_record_content_is_identity_bound_and_canonical(self) -> None:
        original = to_primitive(make_qwen3_moe_plan(self.root))
        mutations = {
            "claim": "Full production training passed.",
            "source": "https://attacker.invalid/forged",
            "source_kind": "production-attestation",
            "scope": "All models and all hosts",
            "confidence": "production-passed",
            "revision": "forged-revision",
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                value = copy.deepcopy(original)
                record = next(
                    item
                    for item in value["evidence_records"]
                    if item["evidence_id"]
                    == "admission.qwen3-30b-a3b.memory-blocked.2026-07-28"
                )
                record[field] = replacement

                self.assertNotEqual(plan_id_for_payload(value), original["plan_id"])
                value["plan_id"] = plan_id_for_payload(value)
                errors = validate_plan_payload(value, verify_dataset=False)
                self.assertTrue(
                    any("canonical evidence registry" in item for item in errors),
                    errors,
                )

    def test_unknown_family_cannot_replay_known_adapter_targets(self) -> None:
        value = copy.deepcopy(self.payload)
        recommended_key = (
            value["recommended"]["method"],
            value["recommended"]["distribution"],
        )
        value["model"]["family"] = "unknown-model"
        decision = _current_model_policy_decision(value["model"])
        self.assertEqual(decision["kind"], "unknown")
        value["model_policy_decision"] = decision
        for candidate in value["candidates"]:
            candidate["model_policy_decision_id"] = decision["decision_id"]
            candidate["policy_binding"] = None
            candidate["candidate_id"] = candidate_id_for_payload(
                candidate,
                model=value["model"],
                dataset=value["dataset"],
                hardware=value["hardware"],
                target=value["target"],
            )
        value["recommended"] = copy.deepcopy(
            next(
                candidate
                for candidate in value["candidates"]
                if (candidate["method"], candidate["distribution"]) == recommended_key
            )
        )
        value["plan_id"] = plan_id_for_payload(value)

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(
            any("unregistered model families" in item for item in errors), errors
        )
        require_current_model_policy(value)

    def test_candidate_identity_and_link_bind_the_policy_decision(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        original_id = candidate["candidate_id"]
        candidate["model_policy_decision_id"] = "compat_" + "0" * 20
        candidate["candidate_id"] = candidate_id_for_payload(
            candidate,
            model=value["model"],
            dataset=value["dataset"],
            hardware=value["hardware"],
            target=value["target"],
        )
        if value["recommended"]["candidate_id"] == original_id:
            value["recommended"] = copy.deepcopy(candidate)
        value["plan_id"] = plan_id_for_payload(value)

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(
            any("current model policy decision ID" in item for item in errors)
        )
        self.assertFalse(any("immutable candidate ID" in item for item in errors))

    def test_registered_path_requires_binding_and_other_paths_forbid_it(self) -> None:
        for mutation in ("remove-matching", "add-nonmatching"):
            with self.subTest(mutation=mutation):
                value = to_primitive(make_qwen3_moe_plan(self.root))
                bound = next(
                    item for item in value["candidates"] if item["policy_binding"]
                )
                if mutation == "remove-matching":
                    candidate = bound
                    candidate["policy_binding"] = None
                else:
                    candidate = next(
                        item
                        for item in value["candidates"]
                        if item["policy_binding"] is None
                    )
                    candidate["policy_binding"] = copy.deepcopy(bound["policy_binding"])
                original_id = candidate["candidate_id"]
                candidate["candidate_id"] = candidate_id_for_payload(
                    candidate,
                    model=value["model"],
                    dataset=value["dataset"],
                    hardware=value["hardware"],
                    target=value["target"],
                )
                if value["recommended"]["candidate_id"] == original_id:
                    value["recommended"] = copy.deepcopy(candidate)
                value["plan_id"] = plan_id_for_payload(value)

                errors = validate_plan_payload(value, verify_dataset=False)

                expected = (
                    "requires a policy_binding"
                    if mutation == "remove-matching"
                    else "policy_binding must be null"
                )
                self.assertTrue(any(expected in item for item in errors), errors)

    def test_provider_receipt_is_recomputed_from_every_observed_plan_fact(self) -> None:
        value = self._provider_qwen_payload()
        self.assertEqual(validate_plan_payload(value, verify_dataset=False), ())

        value["model"]["context_length"] -= 1
        value["plan_id"] = plan_id_for_payload(value)

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(any("observed-facts digest" in item for item in errors), errors)

    def test_receipt_decision_prose_is_not_part_of_identity(self) -> None:
        value = self._provider_qwen_payload()
        original_plan_id = value["plan_id"]
        value["model_policy_decision"]["reason"] = "Plan explanation."
        value["inspection_receipt"]["decision"]["reason"] = "Receipt explanation."

        self.assertEqual(plan_id_for_payload(value), original_plan_id)
        self.assertEqual(validate_plan_payload(value, verify_dataset=False), ())

    def test_receipt_backed_model_provenance_is_verified(self) -> None:
        for field in ("architecture", "parameters", "training_allowed"):
            with self.subTest(field=field):
                value = self._provider_qwen_payload()
                value["model"]["provenance"][field]["kind"] = "unknown"

                errors = validate_plan_payload(value, verify_dataset=False)

                self.assertTrue(
                    any(f"provenance for {field}" in item for item in errors),
                    errors,
                )

    def test_portable_receipt_rejects_non_provider_provenance_kinds(self) -> None:
        for kind in ("measured", "unknown", "user-attested"):
            with self.subTest(kind=kind):
                value = self._provider_qwen_payload()
                provenance = value["inspection_receipt"]["provenance_summary"]
                provenance[0]["kind"] = kind
                value["plan_id"] = plan_id_for_payload(value)

                errors = validate_plan_payload(value, verify_dataset=False)

                self.assertTrue(
                    any(
                        "kind must be provider-declared or inferred" in item
                        for item in errors
                    ),
                    errors,
                )

    def test_portable_receipt_requires_complete_compatibility_subject_coverage(
        self,
    ) -> None:
        value = self._provider_qwen_payload()
        provenance = value["inspection_receipt"]["provenance_summary"]
        value["inspection_receipt"]["provenance_summary"] = [
            item for item in provenance if item["field"] != "architecture"
        ]
        value["plan_id"] = plan_id_for_payload(value)

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(
            any(
                "provenance does not cover compatibility subject facts: architecture"
                in item
                for item in errors
            ),
            errors,
        )

    def test_malformed_provider_receipt_does_not_downgrade_to_user_attested(
        self,
    ) -> None:
        value = self._provider_qwen_payload()
        value["inspection_receipt"] = {}
        value["plan_id"] = plan_id_for_payload(value)

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(any("inspection receipt" in item.lower() for item in errors))
        self.assertTrue(any("exact v1 fields" in item for item in errors))

    def test_receipt_free_plan_cannot_claim_provider_model_provenance(self) -> None:
        value = copy.deepcopy(self.payload)
        value["model"]["provenance"]["all"]["kind"] = "provider-declared"
        value["plan_id"] = plan_id_for_payload(value)

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(
            any("must remain user-attested" in item for item in errors),
            errors,
        )

    def test_plan_identity_binds_decision_source_and_full_receipt(self) -> None:
        for field, replacement in (
            ("model_policy_decision_source", "user-attested"),
            ("inspection_receipt", None),
        ):
            with self.subTest(field=field):
                value = self._provider_qwen_payload()
                value[field] = replacement

                errors = validate_plan_payload(value, verify_dataset=False)

                self.assertTrue(any("Plan immutable ID" in item for item in errors))

    def test_embedded_runtime_identities_match_the_registered_bindings(self) -> None:
        registered = {
            (
                descriptor.method_id,
                binding.training_runtime,
                binding.compute_backend,
            ): (
                binding.compiler_id,
                binding.estimator_id,
                binding.export_kind,
                binding.evidence_requirement,
            )
            for descriptor in selectable_method_descriptors()
            for binding in descriptor.runtime_bindings
        }
        self.assertEqual(RUNTIME_BINDING_IDENTITIES, registered)

    def test_rejects_falsey_and_non_object_payloads(self) -> None:
        for value in ({}, None, [], ""):
            self.assertTrue(validate_plan_payload(value, verify_dataset=False))

    def test_rejects_formula_or_candidate_identity_tampering(self) -> None:
        for mutate in (
            lambda value: value.update(formula_version="hidden-multiplier-v1"),
            lambda value: value["candidates"][0].update(candidate_id="cand_tampered"),
        ):
            value = copy.deepcopy(self.payload)
            mutate(value)
            self.assertTrue(validate_plan_payload(value, verify_dataset=False))

    def test_rejects_plan_id_tampering(self) -> None:
        value = copy.deepcopy(self.payload)
        value["plan_id"] = "plan_" + "0" * 20
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("Plan immutable ID" in item for item in errors))

    def test_candidate_identity_binds_learning_rate(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        candidate["learning_rate"] *= 2
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("normalized execution contract" in item for item in errors))

    def test_candidate_identity_binds_target_modules(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = next(item for item in value["candidates"] if item["target_modules"])
        candidate["target_modules"] = [*candidate["target_modules"], "lm_head"]
        if candidate["candidate_id"] == value["recommended"]["candidate_id"]:
            value["recommended"] = copy.deepcopy(candidate)
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("normalized execution contract" in item for item in errors))

    def test_plan_identity_catches_consistent_memory_recomputation(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        original_candidate_id = candidate["candidate_id"]
        delta = 4096
        memory = candidate["memory"]
        memory["base_weights_bytes"] += delta
        memory["point_estimate_bytes"] += delta
        memory["estimated_peak_bytes"] += delta
        memory["component_upper_bounds"]["base_weights_bytes"] += delta
        memory["upper_estimate_bytes"] += delta
        candidate["candidate_id"] = candidate_id_for_payload(
            candidate,
            model=value["model"],
            dataset=value["dataset"],
            hardware=value["hardware"],
            target=value["target"],
        )
        if value["recommended"]["candidate_id"] == original_candidate_id:
            value["recommended"] = copy.deepcopy(candidate)

        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertFalse(
            any(
                "point memory" in item or "component upper bounds" in item
                for item in errors
            )
        )
        self.assertFalse(any("candidate ID" in item for item in errors))
        self.assertTrue(any("Plan immutable ID" in item for item in errors))

    def test_candidate_identity_binds_normalized_input_facts(self) -> None:
        value = copy.deepcopy(self.payload)
        value["target"]["checkpoint_steps"] += 1
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("normalized execution contract" in item for item in errors))
        self.assertTrue(any("Plan immutable ID" in item for item in errors))

    def test_rejects_global_batch_tampering(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        candidate["effective_batch_size"] += 1
        if candidate["candidate_id"] == value["recommended"]["candidate_id"]:
            value["recommended"] = copy.deepcopy(candidate)
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("batch arithmetic" in item for item in errors))

    def test_rejects_hidden_or_inconsistent_upper_bound(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        candidate["memory"]["upper_estimate_bytes"] += 1
        if candidate["candidate_id"] == value["recommended"]["candidate_id"]:
            value["recommended"] = copy.deepcopy(candidate)
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("component upper bounds" in item for item in errors))

    def test_rejects_boolean_memory_components(self) -> None:
        value = copy.deepcopy(self.payload)
        value["candidates"][0]["memory"]["communication_bytes"] = False

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(
            any(
                "memory components must be non-negative integers" in item
                for item in errors
            ),
            errors,
        )

    def test_rejects_dataset_hash_mismatch(self) -> None:
        self.payload["dataset"]["source_sha256"] = "0" * 64
        errors = validate_plan_payload(self.payload, verify_dataset=True)
        self.assertTrue(any("hash" in item.lower() for item in errors))

    def test_rejects_invalid_precision_learning_rate_and_nonfinite_numbers(
        self,
    ) -> None:
        value = copy.deepcopy(self.payload)
        value["candidates"][0]["precision"] = "banana"
        value["candidates"][1]["learning_rate"] = float("nan")
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("precision" in item for item in errors))
        self.assertTrue(any("learning_rate" in item for item in errors))
        self.assertTrue(any("finite JSON" in item for item in errors))

    def test_rejects_invalid_target_and_hardware_ranges(self) -> None:
        value = copy.deepcopy(self.payload)
        value["target"]["objective"] = "guess"
        value["target"]["method_preference"] = "invented"
        value["hardware"]["devices"][0]["free_vram_bytes"] = (
            value["hardware"]["devices"][0]["total_vram_bytes"] + 1
        )
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("objective" in item for item in errors))
        self.assertTrue(any("method_preference" in item for item in errors))
        self.assertTrue(any("free_vram_bytes" in item for item in errors))

    def test_rejects_unenforced_wall_time_contract(self) -> None:
        value = copy.deepcopy(self.payload)
        value["target"]["max_wall_time_minutes"] = 30
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("must be null in Aptus v0.2" in item for item in errors))

    def test_rejects_malformed_duplicate_and_missing_evidence(self) -> None:
        value = copy.deepcopy(self.payload)
        duplicate = copy.deepcopy(value["evidence_records"][0])
        duplicate["claim"] = ""
        value["evidence_records"].append(duplicate)
        value["candidates"][0]["evidence"] = ["missing.evidence"]
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("Duplicate evidence ID" in item for item in errors))
        self.assertTrue(any("non-empty string claim" in item for item in errors))
        self.assertTrue(
            any("references missing evidence ID" in item for item in errors)
        )

    def test_rejects_non_list_evidence_shapes_without_crashing(self) -> None:
        value = copy.deepcopy(self.payload)
        value["evidence_records"] = "not-a-list"
        value["candidates"][0]["evidence"] = "method.lora.paper"
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(
            any("evidence_records must be a list" in item for item in errors)
        )
        self.assertTrue(any("evidence must be a list" in item for item in errors))

    def test_runtime_contract_is_identity_bound_and_backend_checked(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        candidate["runtime_contract"]["compute_backend"] = "mps"
        if candidate["candidate_id"] == value["recommended"]["candidate_id"]:
            value["recommended"] = copy.deepcopy(candidate)
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("runtime and compute backend" in item for item in errors))
        self.assertTrue(any("normalized execution contract" in item for item in errors))

    def test_rejects_invented_runtime_binding_identities(self) -> None:
        candidate_index = next(
            index
            for index, item in enumerate(self.payload["candidates"])
            if item["status"] in {"feasible", "conditional"}
        )
        mutations = (
            ("compiler_id", "invented.compiler.v99"),
            ("estimator_id", "invented-estimator-v99"),
            ("export_kind", "invented-export"),
            ("evidence_requirement", "implementation-required"),
            ("training_runtime", "invented-runtime"),
            ("compute_backend", "rocm"),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):
                value = copy.deepcopy(self.payload)
                value["candidates"][candidate_index]["runtime_contract"][field] = (
                    replacement
                )
                errors = validate_plan_payload(value, verify_dataset=False)
                self.assertTrue(
                    any(
                        "registered compiler, estimator, export, and evidence identity"
                        in item
                        or "exact unavailable identity" in item
                        for item in errors
                    ),
                    errors,
                )

    def test_mps_static_fit_uses_live_host_memory_headroom(self) -> None:
        dataset_path = self.root / "apple-source.jsonl"
        dataset_path.write_text('{"text":"apple example"}\n', encoding="utf-8")
        dataset = profile_dataset(
            dataset_path,
            sample_limit=64,
            sequence_length=128,
        )
        model = build_model_spec(
            model_id="example/apple-model-1b",
            revision="b" * 40,
            family="llama",
            parameters_b=1,
            hidden_size=2048,
            intermediate_size=8192,
            layers=24,
            context_length=4096,
            license_name="apache-2.0",
            training_allowed=True,
        )
        hardware = build_hardware_spec(
            backend=Backend.MPS,
            gpu_count=1,
            vram_gib=64,
            supports_bf16=True,
            supports_4bit=False,
            supports_8bit=False,
            host_ram_gib=64,
            host_ram_free_gib=48,
            reserve_gib=8,
            disk_free_gib=500,
        )
        target = TrainingTarget(
            objective=Objective.MEMORY,
            sequence_length=128,
            effective_batch_size=8,
            max_epochs=1,
            task="sft",
            checkpoint_steps=10,
            training_runtime=TrainingRuntime.MLX_LM,
        )
        value = to_primitive(
            plan_training(
                model=model,
                dataset=dataset,
                hardware=hardware,
                target=target,
            )
        )
        self.assertEqual(validate_plan_payload(value, verify_dataset=False), ())

        value["hardware"]["host_ram_free_bytes"] = int(8.25 * 1024**3)
        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(
            any("exceeds usable per-device memory" in item for item in errors),
            errors,
        )

    def test_qwen3_moe_topology_is_identity_bound_and_policy_checked(self) -> None:
        value = to_primitive(make_qwen3_moe_plan(self.root))
        self.assertEqual(validate_plan_payload(value, verify_dataset=False), ())

        topology_tamper = copy.deepcopy(value)
        topology_tamper["model"]["moe"]["experts_per_token"] = 7
        errors = validate_plan_payload(topology_tamper, verify_dataset=False)
        self.assertTrue(any("active_parameters" in item for item in errors), errors)
        self.assertTrue(any("normalized execution contract" in item for item in errors))

        target_tamper = copy.deepcopy(value)
        candidate = next(
            item
            for item in target_tamper["candidates"]
            if item["method"] == "qlora" and item["distribution"] == "single"
        )
        candidate["target_modules"].append("gate_proj")
        target_tamper["recommended"] = copy.deepcopy(candidate)
        errors = validate_plan_payload(target_tamper, verify_dataset=False)
        self.assertTrue(any("model-family policy" in item for item in errors), errors)

        status_tamper = copy.deepcopy(value)
        status_tamper["recommended"]["status"] = "feasible"
        recommended_id = status_tamper["recommended"]["candidate_id"]
        next(
            item
            for item in status_tamper["candidates"]
            if item["candidate_id"] == recommended_id
        )["status"] = "feasible"
        errors = validate_plan_payload(status_tamper, verify_dataset=False)
        self.assertTrue(
            any("must remain conditional" in item for item in errors), errors
        )

    def test_qwen3_moe_memory_is_recomputed_after_consistent_id_tampering(self) -> None:
        value = to_primitive(make_qwen3_moe_plan(self.root))
        candidate = next(
            item
            for item in value["candidates"]
            if item["method"] == "qlora" and item["distribution"] == "single"
        )
        memory = candidate["memory"]
        point_delta = memory["activations_bytes"]
        upper_delta = memory["component_upper_bounds"]["activations_bytes"]
        memory["activations_bytes"] = 0
        memory["point_estimate_bytes"] -= point_delta
        memory["estimated_peak_bytes"] -= point_delta
        memory["component_upper_bounds"]["activations_bytes"] = 0
        memory["upper_estimate_bytes"] -= upper_delta
        candidate["candidate_id"] = candidate_id_for_payload(
            candidate,
            model=value["model"],
            dataset=value["dataset"],
            hardware=value["hardware"],
            target=value["target"],
        )
        value["recommended"] = copy.deepcopy(candidate)
        value["plan_id"] = plan_id_for_payload(value)

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(
            any("deterministic recomputation" in item for item in errors), errors
        )
        self.assertFalse(
            any("immutable candidate ID" in item for item in errors), errors
        )
        self.assertFalse(any("immutable ID" in item for item in errors), errors)

    def test_qwen3_moe_architecture_contract_matches_pinned_config(self) -> None:
        model = to_primitive(make_qwen3_moe_plan(self.root))["model"]
        config = {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 48,
            "max_position_embeddings": 262144,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "quantization": qwen3_moe_quantization_config(),
        }

        expected = expected_model_architecture_contract(model)
        self.assertEqual(validate_model_config_against_plan(model, config), expected)
        self.assertEqual(
            expected["schema_version"], "aptus.model-architecture-contract.v1"
        )
        self.assertEqual(len(expected["contract_sha256"]), 64)

        config["num_experts_per_tok"] = 4
        with self.assertRaisesRegex(ValueError, "topology"):
            validate_model_config_against_plan(model, config)

    def test_qwen3_moe_architecture_contract_rejects_quantization_layout_drift(
        self,
    ) -> None:
        model = to_primitive(make_qwen3_moe_plan(self.root))["model"]
        base_config = {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 48,
            "max_position_embeddings": 262144,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
        }
        cases = {
            "missing override": lambda value: value.pop("model.layers.47.mlp.gate"),
            "wrong override bits": lambda value: value[
                "model.layers.0.mlp.gate"
            ].update(bits=4),
            "extra override": lambda value: value.update(
                {
                    "model.layers.0.self_attn.q_proj": {
                        "bits": 8,
                        "group_size": 64,
                    }
                }
            ),
            "wrong default group": lambda value: value.update(group_size=32),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                quantization = qwen3_moe_quantization_config()
                mutate(quantization)
                with self.assertRaisesRegex(ValueError, "quantization layout"):
                    validate_model_config_against_plan(
                        model,
                        {**base_config, "quantization": quantization},
                    )

    def test_v2_plan_cannot_smuggle_moe_fields(self) -> None:
        value = to_primitive(make_qwen3_moe_plan(self.root))
        value["schema_version"] = "aptus.training-plan.v2"
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("v2 plan cannot contain v3" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
