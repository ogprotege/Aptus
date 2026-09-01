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
    model_policy_snapshot_sha256,
    subject_from_model,
)
from aptus.plan_contract import (
    MODEL_TARGET_MODULES,
    mlx_packed_checkpoint_overhead_limit,
    mlx_trainable_target_instance_total,
    RUNTIME_BINDING_IDENTITIES,
    StaleModelPolicyError,
    _current_model_policy_decision,
    candidate_id_for_payload,
    expected_model_architecture_contract,
    mlx_quantized_storage_bytes_for_contract,
    plan_id_for_payload,
    require_current_model_policy,
    validate_bundle_manifest,
    validate_model_config_against_plan,
    validate_plan_payload,
)
from aptus.planning import plan_training, select_candidate
from aptus.profiling import (
    build_hardware_spec,
    build_model_spec,
    profile_dataset,
)

from tests.aptus.helpers import (
    make_plan,
    make_qwen2_runtime_footprint_plan,
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

    def test_real_v6_plan_is_valid(self) -> None:
        self.assertEqual(validate_plan_payload(self.payload, verify_dataset=True), ())
        self.assertEqual(self.payload["schema_version"], "aptus.training-plan.v6")

    def test_phase3_target_values_mutate_candidate_and_plan_identity(self) -> None:
        candidate = self.payload["recommended"]
        changed = copy.deepcopy(self.payload)
        changed["target"]["training_seed"] = 101
        changed["target"]["data_order_seed"] = 1_000_101

        changed_candidate_id = candidate_id_for_payload(
            candidate,
            model=changed["model"],
            dataset=changed["dataset"],
            hardware=changed["hardware"],
            target=changed["target"],
        )

        self.assertNotEqual(changed_candidate_id, candidate["candidate_id"])
        self.assertNotEqual(plan_id_for_payload(changed), self.payload["plan_id"])

    def test_select_candidate_creates_a_new_bound_plan(self) -> None:
        plan = make_plan(self.root)
        alternative = next(
            item
            for item in plan.candidates
            if item.feasible and item.candidate_id != plan.recommended.candidate_id
        )

        selected = select_candidate(plan, alternative.candidate_id)

        self.assertEqual(selected.recommended, alternative)
        self.assertNotEqual(selected.plan_id, plan.plan_id)
        self.assertEqual(selected.model_policy_decision, plan.model_policy_decision)
        self.assertEqual(selected.inspection_receipt, plan.inspection_receipt)
        self.assertEqual(selected.evidence_records, plan.evidence_records)
        self.assertEqual(
            validate_plan_payload(to_primitive(selected), verify_dataset=False), ()
        )

    def test_select_candidate_rejects_nonselectable_and_mutated_candidates(
        self,
    ) -> None:
        plan = make_plan(self.root)
        rejected = next(item for item in plan.candidates if not item.feasible)
        with self.assertRaisesRegex(ValueError, "rejected or nonselectable"):
            select_candidate(plan, rejected.candidate_id)

        alternative = next(
            item
            for item in plan.candidates
            if item.feasible and item.candidate_id != plan.recommended.candidate_id
        )
        mutated = replace(alternative, learning_rate=alternative.learning_rate * 2)
        tampered = replace(
            plan,
            candidates=tuple(
                mutated if item.candidate_id == alternative.candidate_id else item
                for item in plan.candidates
            ),
        )
        with self.assertRaisesRegex(ValueError, "current, unmodified plan"):
            select_candidate(tampered, alternative.candidate_id)

    def test_semantically_resource_hostile_plan_returns_a_controlled_error(
        self,
    ) -> None:
        value = copy.deepcopy(self.payload)
        nested: object = "leaf"
        for _ in range(500):
            nested = [nested]
        value["unexpected_nested_value"] = nested

        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("Plan structure is malformed:"), errors)

    def test_current_policy_check_rejects_resource_hostile_decision_cleanly(
        self,
    ) -> None:
        value = copy.deepcopy(self.payload)
        nested: object = "path"
        for _ in range(10_000):
            nested = [nested]
        value["model_policy_decision"]["paths"] = nested

        with self.assertRaisesRegex(ValueError, "malformed"):
            require_current_model_policy(value)

    def test_bundle_manifest_must_be_a_json_object(self) -> None:
        (self.root / "bundle-manifest.json").write_text("null\n", encoding="utf-8")

        self.assertEqual(
            validate_bundle_manifest(self.root),
            ("Bundle manifest must be a JSON object.",),
        )

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

    def test_stale_snapshot_digest_requires_replan_when_decision_is_unchanged(
        self,
    ) -> None:
        value = copy.deepcopy(self.payload)
        changed_snapshot = copy.deepcopy(current_model_policy_snapshot())
        changed_snapshot["dense_families"] = sorted(
            [*changed_snapshot["dense_families"], "future-dense-family"]
        )

        self.assertEqual(validate_plan_payload(value, verify_dataset=False), ())
        self.assertNotEqual(
            value["model_policy_snapshot_sha256"],
            model_policy_snapshot_sha256(changed_snapshot),
        )
        self.assertEqual(
            value["model_policy_decision"],
            _current_model_policy_decision(value["model"], changed_snapshot),
        )

        errors = validate_plan_payload(
            value,
            verify_dataset=False,
            policy_snapshot=changed_snapshot,
        )

        self.assertTrue(any("replan_required" in error for error in errors), errors)
        with self.assertRaisesRegex(StaleModelPolicyError, "replan_required"):
            require_current_model_policy(value, policy_snapshot=changed_snapshot)

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

    def _provider_qwen2_payload(self) -> dict:
        base = make_qwen2_runtime_footprint_plan(self.root)
        model = base.model
        facts = to_primitive(model)
        observed_at = "2026-08-04T12:00:00+00:00"
        receipt_fields = (
            "architecture",
            "context_length",
            "family",
            "hidden_size",
            "intermediate_size",
            "layers",
            "license_name",
            "model_type",
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

    def test_v4_plan_requires_replanning(self) -> None:
        value = copy.deepcopy(self.payload)
        value["schema_version"] = "aptus.training-plan.v4"
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

    def test_stale_provider_policy_uses_its_identity_bound_receipt(self) -> None:
        value = self._provider_qwen2_payload()
        changed_snapshot = copy.deepcopy(current_model_policy_snapshot())
        changed_snapshot["policies"] = [
            policy
            for policy in changed_snapshot["policies"]
            if policy["policy_id"] != "model.qwen2-24l.mlx-qlora"
        ]

        with (
            patch(
                "aptus.model_compatibility.current_model_policy_snapshot",
                return_value=changed_snapshot,
            ),
            self.assertRaisesRegex(StaleModelPolicyError, "replan_required"),
        ):
            require_current_model_policy(value)

    def test_stale_provider_policy_rejects_inferred_path_fact_rewrite(self) -> None:
        value = self._provider_qwen2_payload()
        receipt = value["inspection_receipt"]
        recommended_key = (
            value["recommended"]["method"],
            value["recommended"]["distribution"],
        )
        downgraded_fields = {
            "layers",
            "model_type",
            "quantization_bits",
            "quantization_layout",
        }
        for item in receipt["provenance_summary"]:
            if item["field"] in downgraded_fields:
                item["kind"] = "inferred"

        receipt_identity = {
            "schema_version": receipt["schema_version"],
            "model_id": receipt["model_id"],
            "resolved_revision": receipt["resolved_revision"].lower(),
            "observed_facts_sha256": receipt["observed_facts_sha256"],
            "decision_id": receipt["decision"]["decision_id"],
            "provenance_summary": receipt["provenance_summary"],
            "provenance_requirement": receipt["provenance_requirement"],
            "provenance_requirement_met": receipt["provenance_requirement_met"],
            "evaluated_at": receipt["evaluated_at"],
        }
        encoded = json.dumps(
            receipt_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        receipt["receipt_id"] = "receipt_" + hashlib.sha256(encoded).hexdigest()[:20]
        for item in receipt["provenance_summary"]:
            model_provenance = value["model"]["provenance"][item["field"]]
            model_provenance["kind"] = item["kind"]
            model_provenance["digest"] = receipt["receipt_id"]
        for candidate in value["candidates"]:
            binding = candidate["policy_binding"]
            if binding is not None:
                binding["inspection_receipt_id"] = receipt["receipt_id"]
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

        changed_snapshot = copy.deepcopy(current_model_policy_snapshot())
        changed_snapshot["policies"] = [
            policy
            for policy in changed_snapshot["policies"]
            if policy["policy_id"] != "model.qwen2-24l.mlx-qlora"
        ]
        with patch(
            "aptus.model_compatibility.current_model_policy_snapshot",
            return_value=changed_snapshot,
        ):
            with self.assertRaises(ValueError) as caught:
                require_current_model_policy(value)

        self.assertNotIsInstance(caught.exception, StaleModelPolicyError)

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

    def test_portable_receipt_uses_each_policy_provenance_requirements(
        self,
    ) -> None:
        cases = {
            "Qwen3 MoE artifact policy": (
                self._provider_qwen_payload(),
                "model.qwen3-moe.mlx-qlora",
                {
                    "architecture",
                    "layers",
                    "model_type",
                    "moe",
                    "quantization_bits",
                    "quantization_layout",
                },
            ),
            "Qwen2 reviewed runtime footprint": (
                self._provider_qwen2_payload(),
                "model.qwen2-24l.mlx-qlora",
                {
                    "architecture",
                    "layers",
                    "model_type",
                    "quantization_bits",
                    "quantization_layout",
                },
            ),
        }
        policies = {
            policy["policy_id"]: policy
            for policy in current_model_policy_snapshot()["policies"]
        }

        for name, (value, policy_id, required_fields) in cases.items():
            with self.subTest(name=name):
                receipt = value["inspection_receipt"]
                self.assertEqual(receipt["decision"]["policy_id"], policy_id)
                self.assertEqual(
                    set(policies[policy_id]["required_provenance_fields"]),
                    required_fields,
                )
                summary = {
                    item["field"]: item for item in receipt["provenance_summary"]
                }
                self.assertTrue(required_fields.issubset(summary))
                self.assertTrue(
                    all(
                        summary[field]["kind"] == "provider-declared"
                        for field in required_fields
                    )
                )
                self.assertEqual(
                    receipt["provenance_requirement"],
                    "provider-declared",
                )
                self.assertTrue(receipt["provenance_requirement_met"])
                if policy_id == "model.qwen2-24l.mlx-qlora":
                    self.assertNotIn("moe", summary)
                self.assertEqual(
                    validate_plan_payload(value, verify_dataset=False),
                    (),
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

    def test_plan_id_changes_when_training_policy_version_changes(self) -> None:
        value = copy.deepcopy(self.payload)
        self.assertEqual(value["training_policy_version"], "aptus-training-policy-v1")
        original_plan_id = plan_id_for_payload(value)
        value["training_policy_version"] = "aptus-training-policy-v0-test"
        self.assertNotEqual(plan_id_for_payload(value), original_plan_id)

    def test_rejects_missing_or_wrong_training_policy_version(self) -> None:
        missing = copy.deepcopy(self.payload)
        del missing["training_policy_version"]
        missing_errors = validate_plan_payload(missing, verify_dataset=False)
        self.assertTrue(
            any("training_policy_version must be" in item for item in missing_errors)
        )
        wrong = copy.deepcopy(self.payload)
        wrong["training_policy_version"] = "aptus-training-policy-v0-test"
        wrong_errors = validate_plan_payload(wrong, verify_dataset=False)
        self.assertTrue(
            any("training_policy_version must be" in item for item in wrong_errors)
        )

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

    def test_dense_qwen2_architecture_contract_rejects_moe_config(self) -> None:
        model = to_primitive(make_qwen2_runtime_footprint_plan(self.root))["model"]
        config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "max_position_embeddings": 32768,
            "quantization": {"bits": 4, "group_size": 64},
        }

        validate_model_config_against_plan(model, config)
        config.update(
            {
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 1024,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
            }
        )

        with self.assertRaisesRegex(ValueError, "unexpectedly declares MoE"):
            validate_model_config_against_plan(model, config)

        dense_config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "max_position_embeddings": 32768,
            "quantization": {"bits": 4, "group_size": 64},
            "mlp_only_layers": [],
        }
        validate_model_config_against_plan(model, dense_config)

    def test_dense_mlx_storage_supports_explicit_empty_override_layout(self) -> None:
        model = {
            "parameters": 494_000_000,
            "quantization_layout": {
                "default_bits": 4,
                "default_group_size": 64,
                "module_overrides": [],
            },
        }

        self.assertEqual(
            mlx_quantized_storage_bytes_for_contract(model),
            (247_000_000, 30_875_000),
        )

    def test_dense_mlx_storage_rejects_router_override_without_moe(self) -> None:
        model = {
            "parameters": 494_000_000,
            "hidden_size": 896,
            "quantization_layout": {
                "default_bits": 4,
                "default_group_size": 64,
                "module_overrides": [
                    {
                        "module_path": "model.layers.0.mlp.gate",
                        "bits": 8,
                        "group_size": 64,
                    }
                ],
            },
        }

        with self.assertRaisesRegex(ValueError, "MoE topology"):
            mlx_quantized_storage_bytes_for_contract(model)

    def test_mlx_packed_checkpoint_overhead_limit_covers_gemma4_norm_residual(
        self,
    ) -> None:
        self.assertEqual(
            mlx_packed_checkpoint_overhead_limit(4_197_945_024), 2 * 1024**2
        )
        self.assertGreater(
            mlx_packed_checkpoint_overhead_limit(4_197_945_024), 1_157_462
        )

    def test_mlx_trainable_target_instance_total_allows_absent_kv(self) -> None:
        targets = MODEL_TARGET_MODULES["gemma4"]
        counts = {target: 35 for target in targets}
        counts["k_proj"] = 15
        counts["v_proj"] = 15
        self.assertEqual(
            mlx_trainable_target_instance_total(targets, 35, counts, family="gemma4"),
            205,
        )
        self.assertEqual(
            mlx_trainable_target_instance_total(
                targets, 35, {**counts, "v_proj": 10}, family="gemma4"
            ),
            200,
        )
        moe_targets = MODEL_TARGET_MODULES["gemma4_moe"]
        moe_counts = {target: 30 for target in moe_targets}
        moe_counts["k_proj"] = 30
        moe_counts["v_proj"] = 25
        self.assertEqual(
            mlx_trainable_target_instance_total(
                moe_targets, 30, moe_counts, family="gemma4_moe"
            ),
            115,
        )
        with self.assertRaisesRegex(ValueError, "k_proj and v_proj adapter counts"):
            mlx_trainable_target_instance_total(
                targets, 35, {**counts, "v_proj": 16}, family="gemma4"
            )
        with self.assertRaisesRegex(ValueError, "every transformer layer"):
            mlx_trainable_target_instance_total(targets, 35, counts, family="llama")
        full = {target: 35 for target in targets}
        self.assertEqual(
            mlx_trainable_target_instance_total(targets, 35, full, family="llama"),
            245,
        )
        sparse_qwen3 = {target: 48 for target in MODEL_TARGET_MODULES["qwen3_moe"]}
        sparse_qwen3["k_proj"] = 1
        with self.assertRaisesRegex(ValueError, "every transformer layer"):
            mlx_trainable_target_instance_total(
                MODEL_TARGET_MODULES["qwen3_moe"],
                48,
                sparse_qwen3,
                family="qwen3_moe",
            )
        counts["q_proj"] = 34
        with self.assertRaisesRegex(ValueError, "every transformer layer"):
            mlx_trainable_target_instance_total(targets, 35, counts, family="gemma4")
        counts["q_proj"] = 35
        counts["k_proj"] = 0
        with self.assertRaisesRegex(ValueError, "not exact"):
            mlx_trainable_target_instance_total(targets, 35, counts, family="gemma4")

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
