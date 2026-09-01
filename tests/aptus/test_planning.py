import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aptus.domain import (
    Backend,
    CandidateStatus,
    DeviceSpec,
    Distribution,
    HardwareSpec,
    Method,
    ModelPolicyBindingSource,
    Objective,
    Provenance,
    ProvenanceKind,
    QuantizationLayout,
    TrainingRuntime,
    gibibytes,
)
from aptus.methods import METHOD_REGISTRY
from aptus.model_compatibility import (
    create_model_inspection_receipt,
    subject_from_model,
)
from aptus.planning import NoFeasiblePlanError, estimate_candidate, plan_training
from aptus.profiling import build_hardware_spec

from tests.aptus.helpers import (
    make_gemma4_moe_plan,
    make_plan,
    make_qwen2_runtime_footprint_plan,
    make_qwen3_moe_plan,
)


def _provider_receipt(model):
    observed_at = "2026-07-29T20:00:00+00:00"
    fields = (
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
    facts = {field: getattr(model, field) for field in fields}
    provenance = {
        field: {
            "kind": (
                ProvenanceKind.INFERRED.value
                if field == "family"
                else ProvenanceKind.PROVIDER_DECLARED.value
            ),
            "source": (
                "aptus-family-map"
                if field == "family"
                else f"https://huggingface.co/{model.model_id}/resolve/{model.revision}/config.json"
            ),
            "observed_at": observed_at,
            "resolved_revision": model.revision,
        }
        for field in fields
    }
    return create_model_inspection_receipt(
        model_id=model.model_id,
        resolved_revision=model.revision,
        facts=facts,
        provenance=provenance,
        subject=subject_from_model(model),
        evaluated_at=observed_at,
    )


class PlannerTests(unittest.TestCase):
    def test_plan_binds_policy_snapshot_digest_into_identity(self) -> None:
        with patch(
            "aptus.planning.current_model_policy_snapshot_sha256",
            return_value="a" * 64,
        ):
            with tempfile.TemporaryDirectory() as temporary:
                first = make_plan(Path(temporary))
        with patch(
            "aptus.planning.current_model_policy_snapshot_sha256",
            return_value="b" * 64,
        ):
            with tempfile.TemporaryDirectory() as temporary:
                second = make_plan(Path(temporary))

        self.assertEqual(first.model_policy_snapshot_sha256, "a" * 64)
        self.assertEqual(second.model_policy_snapshot_sha256, "b" * 64)
        self.assertNotEqual(first.plan_id, second.plan_id)

    def test_user_attested_plan_binds_current_policy_to_every_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_qwen3_moe_plan(Path(temporary))

        self.assertEqual(
            plan.model_policy_decision_source,
            ModelPolicyBindingSource.USER_ATTESTED,
        )
        self.assertIsNone(plan.inspection_receipt)
        self.assertTrue(
            all(
                candidate.model_policy_decision_id
                == plan.model_policy_decision.decision_id
                for candidate in plan.candidates
            )
        )
        bound = [candidate for candidate in plan.candidates if candidate.policy_binding]
        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0].method, Method.QLORA)
        self.assertEqual(bound[0].distribution, Distribution.SINGLE)
        self.assertEqual(
            bound[0].policy_binding.source,
            ModelPolicyBindingSource.USER_ATTESTED,
        )
        self.assertIsNone(bound[0].policy_binding.inspection_receipt_id)
        self.assertTrue(
            set(bound[0].policy_binding.evidence_ids).issubset(bound[0].evidence)
        )

    def test_provider_receipt_is_validated_and_preserved_in_policy_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_qwen3_moe_plan(Path(temporary))
            receipt = _provider_receipt(base.model)
            plan = plan_training(
                model=base.model,
                dataset=base.dataset,
                hardware=base.hardware,
                target=base.target,
                inspection_receipt=receipt,
            )

        self.assertEqual(
            plan.model_policy_decision_source,
            ModelPolicyBindingSource.PROVIDER_INSPECTION,
        )
        self.assertEqual(plan.inspection_receipt, receipt)
        self.assertEqual(
            plan.model.provenance["architecture"].kind,
            ProvenanceKind.PROVIDER_DECLARED,
        )
        self.assertEqual(
            plan.model.provenance["family"].kind,
            ProvenanceKind.INFERRED,
        )
        self.assertEqual(
            plan.model.provenance["parameters"].kind,
            ProvenanceKind.USER_ATTESTED,
        )
        self.assertEqual(
            plan.model.provenance["training_allowed"].kind,
            ProvenanceKind.USER_ATTESTED,
        )
        bound = [candidate for candidate in plan.candidates if candidate.policy_binding]
        self.assertEqual(len(bound), 1)
        self.assertEqual(
            bound[0].policy_binding.source,
            ModelPolicyBindingSource.PROVIDER_INSPECTION,
        )
        self.assertEqual(
            bound[0].policy_binding.inspection_receipt_id,
            receipt.receipt_id,
        )

    def test_no_feasible_provider_plan_preserves_policy_receipt_chain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_qwen3_moe_plan(Path(temporary))
            receipt = _provider_receipt(base.model)
            hardware = build_hardware_spec(
                backend=Backend.MPS,
                gpu_count=1,
                vram_gib=9,
                supports_bf16=False,
                supports_4bit=False,
                host_ram_gib=9,
                host_ram_free_gib=9,
                reserve_gib=8,
                disk_free_gib=500,
            )

            with self.assertRaises(NoFeasiblePlanError) as raised:
                plan_training(
                    model=base.model,
                    dataset=base.dataset,
                    hardware=hardware,
                    target=base.target,
                    inspection_receipt=receipt,
                )

        error = raised.exception
        self.assertEqual(
            error.model_policy_decision_source,
            ModelPolicyBindingSource.PROVIDER_INSPECTION,
        )
        self.assertEqual(error.inspection_receipt, receipt)
        self.assertEqual(error.model.model_id, base.model.model_id)
        self.assertEqual(error.model.revision, base.model.revision)
        self.assertEqual(
            error.model_policy_decision.decision_id,
            receipt.decision.decision_id,
        )
        self.assertTrue(
            all(
                candidate.model_policy_decision_id
                == error.model_policy_decision.decision_id
                for candidate in error.candidates
            )
        )
        bound = [
            candidate
            for candidate in error.candidates
            if candidate.policy_binding is not None
        ]
        self.assertEqual(len(bound), 1)
        self.assertEqual(
            bound[0].policy_binding.inspection_receipt_id,
            receipt.receipt_id,
        )
        with self.assertRaisesRegex(TypeError, "require a model subject"):
            NoFeasiblePlanError(
                error.candidates,
                model=None,
                model_policy_decision=error.model_policy_decision,
                model_policy_decision_source=error.model_policy_decision_source,
                inspection_receipt=error.inspection_receipt,
            )
        for mismatched_model in (
            replace(error.model, model_id="different/model"),
            replace(error.model, revision="0" * 40),
        ):
            with (
                self.subTest(
                    model_id=mismatched_model.model_id,
                    revision=mismatched_model.revision,
                ),
                self.assertRaisesRegex(ValueError, "must match the model"),
            ):
                NoFeasiblePlanError(
                    error.candidates,
                    model=mismatched_model,
                    model_policy_decision=error.model_policy_decision,
                    model_policy_decision_source=error.model_policy_decision_source,
                    inspection_receipt=error.inspection_receipt,
                )

    def test_receipt_rejects_user_attested_dense_unknown_and_exact_subjects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dense = make_plan(Path(temporary)).model
            exact = make_qwen3_moe_plan(Path(temporary)).model
        unknown = replace(
            dense,
            family="custom",
            model_type="custom",
            architecture="CustomForCausalLM",
        )
        observed_at = "2026-07-29T20:00:00+00:00"
        fields = (
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

        for model in (dense, unknown, exact):
            facts = {
                field: getattr(model, field)
                for field in fields
                if getattr(model, field) is not None
            }
            provenance = {
                field: {
                    "kind": "user-attested",
                    "source": "operator",
                    "observed_at": observed_at,
                    "resolved_revision": model.revision,
                }
                for field in facts
            }
            with (
                self.subTest(family=model.family),
                self.assertRaisesRegex(ValueError, "provider-declared observation"),
            ):
                create_model_inspection_receipt(
                    model_id=model.model_id,
                    resolved_revision=model.revision,
                    facts=facts,
                    provenance=provenance,
                    subject=subject_from_model(model),
                    evaluated_at=observed_at,
                )

    def test_receipt_rejects_partial_subject_and_noninspection_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = make_plan(Path(temporary)).model
        observed_at = "2026-07-29T20:00:00+00:00"
        license_only = {
            "license_name": {
                "kind": "provider-declared",
                "source": "provider-card",
                "observed_at": observed_at,
                "resolved_revision": model.revision,
            }
        }
        with self.assertRaisesRegex(ValueError, "does not cover"):
            create_model_inspection_receipt(
                model_id=model.model_id,
                resolved_revision=model.revision,
                facts={"license_name": model.license_name},
                provenance=license_only,
                subject=subject_from_model(model),
                evaluated_at=observed_at,
            )

        fields = ("architecture", "family", "layers", "license_name")
        facts = {field: getattr(model, field) for field in fields}
        provenance = {
            field: {
                "kind": "provider-declared",
                "source": "provider-config",
                "observed_at": observed_at,
                "resolved_revision": model.revision,
            }
            for field in fields
        }
        provenance["license_name"]["kind"] = "unknown"
        with self.assertRaisesRegex(
            ValueError, "provider-declared or provider-derived"
        ):
            create_model_inspection_receipt(
                model_id=model.model_id,
                resolved_revision=model.revision,
                facts=facts,
                provenance=provenance,
                subject=subject_from_model(model),
                evaluated_at=observed_at,
            )

    def test_tampered_provider_receipt_is_rejected_instead_of_downgraded(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_qwen3_moe_plan(Path(temporary))
            receipt = replace(
                _provider_receipt(base.model),
                receipt_id="receipt_" + "0" * 20,
            )

            with self.assertRaisesRegex(ValueError, "immutable ID"):
                plan_training(
                    model=base.model,
                    dataset=base.dataset,
                    hardware=base.hardware,
                    target=base.target,
                    inspection_receipt=receipt,
                )

    def test_receipt_replanning_preserves_user_attestation_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_qwen3_moe_plan(Path(temporary))
            model = replace(
                base.model,
                provenance={
                    "all": Provenance(
                        ProvenanceKind.USER_ATTESTED,
                        "operator-intake",
                    )
                },
            )
            receipt = _provider_receipt(model)
            first = plan_training(
                model=model,
                dataset=base.dataset,
                hardware=base.hardware,
                target=base.target,
                inspection_receipt=receipt,
            )
            repeated = plan_training(
                model=first.model,
                dataset=base.dataset,
                hardware=base.hardware,
                target=base.target,
                inspection_receipt=receipt,
            )

        self.assertEqual(
            repeated.model.provenance["parameters"].source,
            "operator-intake",
        )
        self.assertEqual(
            repeated.model.provenance["training_allowed"].source,
            "operator-intake",
        )

    def test_receipt_free_replanning_explicitly_user_attests_all_model_facts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_qwen3_moe_plan(Path(temporary))
            receipt = _provider_receipt(base.model)
            inspected = plan_training(
                model=base.model,
                dataset=base.dataset,
                hardware=base.hardware,
                target=base.target,
                inspection_receipt=receipt,
            )
            direct = plan_training(
                model=inspected.model,
                dataset=base.dataset,
                hardware=base.hardware,
                target=base.target,
            )

        self.assertEqual(
            direct.model_policy_decision_source,
            ModelPolicyBindingSource.USER_ATTESTED,
        )
        self.assertEqual(set(direct.model.provenance), {"all"})
        self.assertEqual(
            direct.model.provenance["all"].kind,
            ProvenanceKind.USER_ATTESTED,
        )

    def test_estimate_candidate_cannot_accept_injected_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_qwen3_moe_plan(Path(temporary))
            receipt = _provider_receipt(base.model)

            with self.assertRaises(TypeError):
                estimate_candidate(
                    method=Method.QLORA,
                    model=base.model,
                    dataset=base.dataset,
                    hardware=base.hardware,
                    target=base.target,
                    inspection_receipt=receipt,  # type: ignore[call-arg]
                )

    def test_enumerates_full_matrix_with_immutable_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary), gpu_count=2)
        self.assertEqual(len(plan.candidates), 12)
        self.assertEqual({item.method for item in plan.candidates}, set(Method))
        self.assertEqual(
            {item.distribution for item in plan.candidates}, set(Distribution)
        )
        self.assertEqual(len({item.candidate_id for item in plan.candidates}), 12)
        self.assertTrue(
            all(item.candidate_id.startswith("cand_") for item in plan.candidates)
        )

    def test_global_batch_includes_world_size_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary), gpu_count=2, effective_batch=12)
        for candidate in plan.candidates:
            self.assertEqual(
                candidate.micro_batch_size
                * candidate.gradient_accumulation_steps
                * candidate.world_size,
                12,
            )

    def test_single_gpu_keeps_multi_gpu_strategies_visible_but_unsupported(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary), gpu_count=1)
        multi = [
            item for item in plan.candidates if item.distribution != Distribution.SINGLE
        ]
        self.assertTrue(multi)
        self.assertTrue(
            all(item.status == CandidateStatus.UNSUPPORTED for item in multi)
        )

    def test_registry_distribution_support_is_an_authoritative_gate(self) -> None:
        restricted = replace(
            METHOD_REGISTRY["lora"], supported_distributions=("single",)
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(METHOD_REGISTRY, {"lora": restricted}),
        ):
            plan = make_plan(Path(temporary), gpu_count=2)
        distributed_lora = [
            item
            for item in plan.candidates
            if item.method == Method.LORA
            and item.distribution in {Distribution.DDP, Distribution.FSDP}
        ]
        self.assertTrue(distributed_lora)
        self.assertTrue(
            all(item.status == CandidateStatus.UNSUPPORTED for item in distributed_lora)
        )
        self.assertTrue(
            all(
                any("registry contract" in reason for reason in item.rejection_reasons)
                for item in distributed_lora
            )
        )

    def test_quantized_fsdp_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary), gpu_count=2)
        quantized_fsdp = [
            item
            for item in plan.candidates
            if item.distribution == Distribution.FSDP
            and item.method in {Method.INT8_LORA, Method.QLORA}
        ]
        self.assertTrue(
            all(item.status == CandidateStatus.UNSUPPORTED for item in quantized_fsdp)
        )

    def test_full_fsdp_is_closed_and_lora_fsdp_requires_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary), gpu_count=2)
        full_fsdp = [
            item
            for item in plan.candidates
            if item.distribution == Distribution.FSDP and item.method == Method.FULL
        ]
        self.assertTrue(full_fsdp)
        self.assertTrue(
            all(item.status == CandidateStatus.UNSUPPORTED for item in full_fsdp)
        )
        fsdp = [
            item
            for item in plan.candidates
            if item.distribution == Distribution.FSDP and item.method == Method.LORA
        ]
        self.assertTrue(fsdp)
        self.assertTrue(
            all(item.status == CandidateStatus.CONDITIONAL for item in fsdp)
        )
        self.assertTrue(
            all(
                any("simplified" in reason for reason in item.rejection_reasons)
                for item in fsdp
            )
        )
        self.assertTrue(
            all(
                any("use_orig_params=true" in item for item in candidate.assumptions)
                for candidate in fsdp
            )
        )

    def test_memory_reserve_is_separate_from_usage_and_bounds_are_named(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary))
        candidate = plan.recommended
        self.assertEqual(candidate.user_reserve_bytes, 2 * 1024**3)
        self.assertNotIn("user_reserve_bytes", candidate.memory.component_upper_bounds)
        self.assertEqual(
            candidate.memory.upper_bytes,
            sum(candidate.memory.component_upper_bounds.values()),
        )
        self.assertGreater(
            candidate.memory.upper_bytes, candidate.memory.point_estimate_bytes
        )

    def test_distributed_host_staging_scales_with_rank_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary), gpu_count=2)
        single = next(
            item
            for item in plan.candidates
            if item.method == Method.LORA and item.distribution == Distribution.SINGLE
        )
        ddp = next(
            item
            for item in plan.candidates
            if item.method == Method.LORA and item.distribution == Distribution.DDP
        )
        self.assertEqual(
            ddp.required_host_ram_bytes, 2 * single.required_host_ram_bytes
        )
        fsdp = next(
            item
            for item in plan.candidates
            if item.method == Method.LORA and item.distribution == Distribution.FSDP
        )
        self.assertEqual(
            fsdp.required_host_ram_bytes, 2 * single.required_host_ram_bytes
        )

    def test_ranking_is_policy_not_fabricated_quality_or_speed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary), objective=Objective.QUALITY)
        self.assertTrue(
            any(
                "not a prediction" in item.lower()
                for item in plan.recommendation_rationale
            )
        )
        self.assertTrue(
            all(
                any("no model-quality" in basis.lower() for basis in item.ranking_basis)
                for item in plan.candidates
            )
        )

    def test_non_sft_and_packing_fail_closed(self) -> None:
        for task, packing in (("dpo", False), ("sft", True)):
            with (
                self.subTest(task=task, packing=packing),
                tempfile.TemporaryDirectory() as temporary,
            ):
                with self.assertRaises(NoFeasiblePlanError):
                    make_plan(Path(temporary), task=task, packing=packing)

    def test_host_ram_and_disk_are_hard_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(NoFeasiblePlanError) as raised:
                make_plan(Path(temporary), host_ram_gib=1, disk_free_gib=0.1)

        error = raised.exception
        self.assertEqual(
            error.model_policy_decision_source,
            ModelPolicyBindingSource.USER_ATTESTED,
        )
        self.assertIsNone(error.inspection_receipt)
        self.assertEqual(error.model.model_id, "example/model-1b")
        self.assertEqual(error.model.revision, "a" * 40)
        self.assertTrue(
            all(
                candidate.model_policy_decision_id
                == error.model_policy_decision.decision_id
                for candidate in error.candidates
            )
        )

    def test_non_divisible_multi_gpu_batch_is_infeasible_but_single_can_survive(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary), gpu_count=2, effective_batch=7)
        self.assertTrue(
            any(
                item.feasible
                for item in plan.candidates
                if item.distribution == Distribution.SINGLE
            )
        )
        self.assertTrue(
            all(
                not item.feasible
                for item in plan.candidates
                if item.distribution != Distribution.SINGLE
            )
        )

    def test_single_candidate_binds_highest_capacity_compatible_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = make_plan(root, gpu_count=2)
            hardware = replace(
                base.hardware,
                devices=(
                    replace(base.hardware.devices[0], supports_8bit=True),
                    replace(
                        base.hardware.devices[1],
                        name="Large adapter-only GPU",
                        total_vram_bytes=gibibytes(48),
                        free_vram_bytes=gibibytes(32),
                        supports_bf16=False,
                        supports_4bit=False,
                        supports_8bit=False,
                    ),
                ),
            )
            plan = plan_training(
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )
        single_lora = next(
            item
            for item in plan.candidates
            if item.method == Method.LORA and item.distribution == Distribution.SINGLE
        )
        distributed_lora = next(
            item
            for item in plan.candidates
            if item.method == Method.LORA and item.distribution == Distribution.DDP
        )
        single_qlora = next(
            item
            for item in plan.candidates
            if item.method == Method.QLORA and item.distribution == Distribution.SINGLE
        )
        single_full = next(
            item
            for item in plan.candidates
            if item.method == Method.FULL and item.distribution == Distribution.SINGLE
        )
        single_int8 = next(
            item
            for item in plan.candidates
            if item.method == Method.INT8_LORA
            and item.distribution == Distribution.SINGLE
        )
        distributed_qlora = next(
            item
            for item in plan.candidates
            if item.method == Method.QLORA and item.distribution == Distribution.DDP
        )
        self.assertEqual(single_lora.device_indices, (1,))
        self.assertEqual(single_lora.precision, "fp16")
        self.assertTrue(
            any(
                "greatest usable VRAM" in assumption
                for assumption in single_lora.assumptions
            )
        )
        self.assertEqual(single_qlora.device_indices, (0,))
        self.assertEqual(single_qlora.status, CandidateStatus.FEASIBLE)
        self.assertEqual(single_int8.device_indices, (0,))
        self.assertEqual(single_int8.status, CandidateStatus.FEASIBLE)
        self.assertEqual(single_full.device_indices, (0,))
        self.assertEqual(distributed_lora.device_indices, (0, 1))
        self.assertEqual(distributed_lora.precision, "fp16")
        self.assertEqual(distributed_qlora.device_indices, (0, 1))
        self.assertEqual(distributed_qlora.status, CandidateStatus.UNSUPPORTED)

    def test_single_device_tie_breaks_by_stable_hardware_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = make_plan(root, gpu_count=2)
            hardware = replace(
                base.hardware,
                devices=(
                    replace(
                        base.hardware.devices[0],
                        name="CUDA zero",
                        free_vram_bytes=gibibytes(12),
                    ),
                    replace(
                        base.hardware.devices[1],
                        name="CUDA one",
                        free_vram_bytes=gibibytes(12),
                    ),
                ),
            )
            plan = plan_training(
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )
        single_lora = next(
            item
            for item in plan.candidates
            if item.method == Method.LORA and item.distribution == Distribution.SINGLE
        )
        self.assertEqual(single_lora.device_indices, (0,))

    def test_apple_unified_memory_yields_only_pilot_required_mlx_candidates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_plan(Path(temporary))
            hardware = build_hardware_spec(
                backend=Backend.MPS,
                gpu_count=1,
                vram_gib=64,
                supports_bf16=False,
                supports_4bit=False,
                host_ram_gib=64,
                host_ram_free_gib=48,
                reserve_gib=8,
                disk_free_gib=500,
            )
            plan = plan_training(
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )
        viable = [item for item in plan.candidates if item.feasible]
        self.assertEqual({item.method for item in viable}, {Method.LORA})
        self.assertTrue(
            all(item.status == CandidateStatus.CONDITIONAL for item in viable)
        )
        self.assertTrue(
            all(
                item.runtime_contract
                and item.runtime_contract.training_runtime == TrainingRuntime.MLX_LM
                and item.runtime_contract.estimator_id == "aptus-memory-mlx-v2"
                and item.memory.formula_version == "aptus-memory-mlx-v2"
                for item in viable
            )
        )
        qlora = next(
            item
            for item in plan.candidates
            if item.method == Method.QLORA and item.distribution == Distribution.SINGLE
        )
        self.assertFalse(qlora.feasible)
        self.assertEqual(qlora.status, CandidateStatus.UNSUPPORTED)
        self.assertTrue(
            any(
                "declared quantization bits" in reason
                for reason in qlora.rejection_reasons
            )
        )
        self.assertFalse(hardware.devices[0].supports_4bit)

    def test_qwen3_moe_allows_only_attention_only_single_mlx_qlora(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_qwen3_moe_plan(Path(temporary))

        viable = [candidate for candidate in plan.candidates if candidate.feasible]
        self.assertEqual(len(viable), 1)
        candidate = viable[0]
        self.assertEqual(candidate.method, Method.QLORA)
        self.assertEqual(candidate.distribution, Distribution.SINGLE)
        self.assertEqual(candidate.status, CandidateStatus.CONDITIONAL)
        self.assertEqual(
            candidate.runtime_contract.training_runtime, TrainingRuntime.MLX_LM
        )
        self.assertEqual(candidate.runtime_contract.estimator_id, "aptus-memory-mlx-v2")
        self.assertEqual(
            candidate.target_modules, ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        self.assertTrue(
            all(
                item.status == CandidateStatus.UNSUPPORTED
                for item in plan.candidates
                if item.candidate_id != candidate.candidate_id
            )
        )
        router_parameters = 48 * 2048 * 128
        self.assertEqual(
            candidate.memory.base_weights_bytes,
            round((30_500_000_000 - router_parameters) * 0.5 + router_parameters),
        )
        self.assertEqual(
            candidate.memory.quantization_metadata_bytes,
            round(30_500_000_000 * 4 / 64),
        )
        dense_activation = round(8 * 128 * 2048 * 48 * 2 * 3.0)
        routed_activation = round(8 * 128 * 48 * 8 * 768 * 2 * 3.0)
        self.assertEqual(
            candidate.memory.activations_bytes,
            dense_activation + routed_activation,
        )
        self.assertLess(plan.model.active_parameters, plan.model.parameters)
        self.assertTrue(
            any(
                "total parameters" in assumption
                for assumption in candidate.memory.assumptions
            )
        )

    def test_gemma4_moe_allows_only_attention_only_single_mlx_qlora(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_gemma4_moe_plan(Path(temporary))

        viable = [candidate for candidate in plan.candidates if candidate.feasible]
        self.assertEqual(len(viable), 1)
        candidate = viable[0]
        self.assertEqual(candidate.method, Method.QLORA)
        self.assertEqual(candidate.distribution, Distribution.SINGLE)
        self.assertEqual(candidate.status, CandidateStatus.CONDITIONAL)
        self.assertEqual(
            candidate.runtime_contract.training_runtime, TrainingRuntime.MLX_LM
        )
        self.assertEqual(
            candidate.target_modules, ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        self.assertEqual(
            plan.model_policy_decision.policy_id,
            "model.gemma4-moe.mlx.v1",
        )
        router_parameters = 30 * 2816 * 128
        self.assertEqual(
            candidate.memory.base_weights_bytes,
            round((25_200_000_000 - router_parameters) * 0.5 + router_parameters),
        )
        self.assertNotEqual(
            candidate.memory.base_weights_bytes,
            round(plan.model.active_parameters * 0.5),
        )
        self.assertLess(plan.model.active_parameters, plan.model.parameters)
        self.assertTrue(
            any(
                "total parameters" in assumption
                for assumption in candidate.memory.assumptions
            )
        )

    def test_qwen2_runtime_footprint_binds_only_dense_single_mlx_qlora(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_qwen2_runtime_footprint_plan(Path(temporary))

        self.assertEqual(
            plan.model_policy_decision.policy_id,
            "model.qwen2-24l.mlx-qlora",
        )
        bound = [
            candidate
            for candidate in plan.candidates
            if candidate.policy_binding is not None
        ]
        self.assertEqual(len(bound), 1)
        candidate = bound[0]
        self.assertEqual(candidate.candidate_id, plan.recommended.candidate_id)
        self.assertTrue(candidate.feasible)
        self.assertEqual(candidate.status, CandidateStatus.CONDITIONAL)
        self.assertEqual(candidate.method, Method.QLORA)
        self.assertEqual(candidate.distribution, Distribution.SINGLE)
        self.assertEqual(
            candidate.runtime_contract.training_runtime,
            TrainingRuntime.MLX_LM,
        )
        self.assertEqual(candidate.runtime_contract.compute_backend, Backend.MPS)
        self.assertEqual(
            candidate.target_modules,
            (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        )
        assert candidate.policy_binding is not None
        self.assertEqual(
            candidate.policy_binding.path_id,
            "mlx-lm.qlora.single.dense-causal-lm.v1",
        )
        self.assertEqual(
            sum(item.feasible for item in plan.candidates),
            1,
        )
        self.assertTrue(
            all(
                item.status == CandidateStatus.UNSUPPORTED
                for item in plan.candidates
                if item.candidate_id != candidate.candidate_id
            )
        )

    def test_qwen2_28_layer_needs_operator_confirm_for_qlora_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reviewed = make_qwen2_runtime_footprint_plan(Path(temporary))
            seven_b = replace(
                reviewed.model,
                layers=28,
                parameters=7_620_000_000,
            )
            with self.assertRaises(NoFeasiblePlanError):
                plan_training(
                    model=seven_b,
                    dataset=reviewed.dataset,
                    hardware=reviewed.hardware,
                    target=reviewed.target,
                )
            attested = plan_training(
                model=seven_b,
                dataset=reviewed.dataset,
                hardware=reviewed.hardware,
                target=replace(reviewed.target, unreviewed_runtime_confirmed=True),
            )
        self.assertEqual(
            [item.value for item in attested.model_policy_decision.reason_codes],
            [
                "unreviewed-runtime-operator-attested",
                "pilot-not-yet-proven",
            ],
        )
        self.assertEqual(attested.recommended.method, Method.QLORA)
        self.assertTrue(attested.recommended.feasible)
        self.assertEqual(attested.recommended.status, CandidateStatus.CONDITIONAL)
        self.assertTrue(attested.target.unreviewed_runtime_confirmed)

    def test_qwen3_moe_near_match_has_no_viable_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_qwen3_moe_plan(Path(temporary))
            cases = {
                "architecture": replace(
                    plan.model,
                    architecture="Qwen3MoeModel",
                ),
                "layout": replace(
                    plan.model,
                    quantization_layout=QuantizationLayout(4, 64),
                ),
                "shared expert": replace(
                    plan.model,
                    moe=replace(
                        plan.model.moe,
                        shared_expert_intermediate_size=1024,
                    ),
                ),
            }
            for name, model in cases.items():
                with self.subTest(name=name):
                    with self.assertRaises(NoFeasiblePlanError):
                        plan_training(
                            model=model,
                            dataset=plan.dataset,
                            hardware=plan.hardware,
                            target=plan.target,
                        )

    def test_sparse_architecture_marker_without_topology_has_no_viable_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_qwen3_moe_plan(Path(temporary))
            model = replace(
                plan.model,
                family="qwen",
                model_type="qwen2",
                architecture="Qwen2MoeForCausalLM",
                moe=None,
                quantization_layout=None,
            )

            with self.assertRaises(NoFeasiblePlanError):
                plan_training(
                    model=model,
                    dataset=plan.dataset,
                    hardware=plan.hardware,
                    target=plan.target,
                )

            candidate = estimate_candidate(
                method=Method.QLORA,
                model=model,
                dataset=plan.dataset,
                hardware=plan.hardware,
                target=plan.target,
            )
            self.assertEqual(candidate.status, CandidateStatus.UNSUPPORTED)
            self.assertIn(
                "Sparse model execution requires an exact reviewed model "
                "compatibility policy.",
                candidate.rejection_reasons,
            )
            with self.assertRaises(TypeError):
                estimate_candidate(
                    method=Method.QLORA,
                    model=model,
                    dataset=plan.dataset,
                    hardware=plan.hardware,
                    target=plan.target,
                    policy_decision=object(),  # type: ignore[call-arg]
                )

    def test_apple_fit_uses_current_unified_memory_headroom_without_fake_vram(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_plan(Path(temporary))
            hardware = build_hardware_spec(
                backend=Backend.MPS,
                gpu_count=1,
                vram_gib=64,
                supports_bf16=False,
                supports_4bit=False,
                host_ram_gib=64,
                host_ram_free_gib=8.25,
                reserve_gib=8,
                disk_free_gib=500,
            )
            candidate = estimate_candidate(
                method=Method.LORA,
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )

        self.assertIsNone(hardware.devices[0].free_vram_bytes)
        self.assertEqual(candidate.status, CandidateStatus.INFEASIBLE)
        self.assertTrue(
            any(
                "usable per-device memory" in reason
                for reason in candidate.rejection_reasons
            )
        )

    def test_unknown_cuda_free_vram_is_infeasible_not_treated_as_total(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_plan(Path(temporary))
            hardware = build_hardware_spec(
                backend=Backend.CUDA,
                gpu_count=1,
                vram_gib=80,
                supports_bf16=True,
                supports_4bit=True,
                host_ram_gib=64,
                host_ram_free_gib=56,
                reserve_gib=2,
                disk_free_gib=500,
            )
            candidate = estimate_candidate(
                method=Method.LORA,
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )

        self.assertIsNone(hardware.devices[0].free_vram_bytes)
        self.assertEqual(candidate.status, CandidateStatus.INFEASIBLE)
        self.assertTrue(
            any(
                "unknown" in reason.lower() and "total" in reason.lower()
                for reason in candidate.rejection_reasons
            )
        )

    def test_unknown_host_ram_free_is_infeasible_not_treated_as_total(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_plan(Path(temporary))
            hardware = build_hardware_spec(
                backend=Backend.CUDA,
                gpu_count=1,
                vram_gib=24,
                free_vram_gib=22,
                supports_bf16=True,
                supports_4bit=True,
                host_ram_gib=64,
                reserve_gib=2,
                disk_free_gib=500,
            )
            candidate = estimate_candidate(
                method=Method.LORA,
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )

        self.assertIsNone(hardware.host_ram_free_bytes)
        self.assertEqual(candidate.status, CandidateStatus.INFEASIBLE)
        self.assertTrue(
            any(
                "host ram" in reason.lower() and "unknown" in reason.lower()
                for reason in candidate.rejection_reasons
            )
        )

    def test_unknown_disk_free_is_infeasible_not_assumed_sufficient(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_plan(Path(temporary))
            hardware = build_hardware_spec(
                backend=Backend.CUDA,
                gpu_count=1,
                vram_gib=24,
                free_vram_gib=22,
                supports_bf16=True,
                supports_4bit=True,
                host_ram_gib=64,
                host_ram_free_gib=56,
                reserve_gib=2,
            )
            candidate = estimate_candidate(
                method=Method.LORA,
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )

        self.assertIsNone(hardware.disk_free_bytes)
        self.assertEqual(candidate.status, CandidateStatus.INFEASIBLE)
        self.assertTrue(
            any(
                "disk" in reason.lower() and "unknown" in reason.lower()
                for reason in candidate.rejection_reasons
            )
        )

    def test_unknown_apple_unified_free_is_infeasible_without_using_total(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_plan(Path(temporary))
            hardware = build_hardware_spec(
                backend=Backend.MPS,
                gpu_count=1,
                vram_gib=64,
                supports_bf16=False,
                supports_4bit=False,
                host_ram_gib=64,
                reserve_gib=8,
                disk_free_gib=500,
            )
            candidate = estimate_candidate(
                method=Method.LORA,
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )

        self.assertIsNone(hardware.host_ram_free_bytes)
        self.assertEqual(candidate.status, CandidateStatus.INFEASIBLE)
        self.assertTrue(
            any("unknown" in reason.lower() for reason in candidate.rejection_reasons)
        )
        self.assertFalse(
            any(
                "exceeds usable per-device memory" in reason
                for reason in candidate.rejection_reasons
            )
        )

    def test_mixed_known_and_unknown_cuda_free_selects_known_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_plan(Path(temporary))
            hardware = HardwareSpec(
                devices=(
                    DeviceSpec(
                        "GPU 0",
                        Backend.CUDA,
                        gibibytes(24),
                        True,
                        True,
                        False,
                        gibibytes(22),
                    ),
                    DeviceSpec("GPU 1", Backend.CUDA, gibibytes(24), True, True),
                ),
                host_ram_bytes=gibibytes(64),
                host_ram_free_bytes=gibibytes(56),
                reserve_per_device_bytes=gibibytes(2),
                disk_free_bytes=gibibytes(500),
            )
            candidate = estimate_candidate(
                method=Method.LORA,
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )

        self.assertEqual(candidate.device_indices, (0,))
        self.assertNotEqual(candidate.status, CandidateStatus.INFEASIBLE)

    def test_mlx_zero_evaluation_fraction_is_infeasible_before_compile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_plan(Path(temporary))
            hardware = build_hardware_spec(
                backend=Backend.MPS,
                gpu_count=1,
                vram_gib=64,
                supports_bf16=False,
                supports_4bit=False,
                host_ram_gib=64,
                host_ram_free_gib=56,
                reserve_gib=8,
                disk_free_gib=500,
            )
            target = replace(base.target, evaluation_fraction=0.0)
            candidate = estimate_candidate(
                method=Method.LORA,
                model=base.model,
                dataset=base.dataset,
                hardware=hardware,
                target=target,
            )

        self.assertEqual(candidate.status, CandidateStatus.INFEASIBLE)
        self.assertTrue(
            any(
                "evaluation_fraction=0" in reason
                for reason in candidate.rejection_reasons
            )
        )

    def test_missing_intermediate_size_refuses_mlp_adapter_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = make_plan(Path(temporary))
            model = replace(base.model, intermediate_size=None)
            hardware = build_hardware_spec(
                backend=Backend.CUDA,
                gpu_count=1,
                vram_gib=24,
                free_vram_gib=22,
                supports_bf16=True,
                supports_4bit=True,
                host_ram_gib=64,
                host_ram_free_gib=56,
                reserve_gib=2,
                disk_free_gib=500,
            )
            candidate = estimate_candidate(
                method=Method.LORA,
                model=model,
                dataset=base.dataset,
                hardware=hardware,
                target=base.target,
            )

        self.assertIsNone(model.intermediate_size)
        self.assertEqual(candidate.status, CandidateStatus.INFEASIBLE)
        self.assertTrue(
            any(
                "intermediate_size" in reason and "4" in reason
                for reason in candidate.rejection_reasons
            )
        )

    def test_four_rows_one_epoch_is_at_least_conditional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(
                Path(temporary),
                dataset_rows=4,
                max_epochs=1,
            )
        self.assertEqual(plan.dataset.example_count, 4)
        self.assertEqual(plan.target.max_epochs, 1)
        self.assertIn(
            plan.recommended.status,
            {CandidateStatus.CONDITIONAL, CandidateStatus.FEASIBLE},
        )
        self.assertEqual(plan.recommended.status, CandidateStatus.CONDITIONAL)
        self.assertTrue(
            any(
                "below the instruction-SFT supervision prior" in reason
                for reason in plan.recommended.rejection_reasons
            )
        )
        self.assertTrue(
            any(
                "training-policy v1 priors" in assumption
                for assumption in plan.recommended.assumptions
            )
        )

    def test_four_rows_ten_epochs_is_no_feasible_plan(self) -> None:
        target_epochs = 10
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(NoFeasiblePlanError) as raised:
                make_plan(
                    Path(temporary),
                    dataset_rows=4,
                    max_epochs=target_epochs,
                )
        # Operator request is not rewritten by policy.
        self.assertEqual(target_epochs, 10)
        error = raised.exception
        self.assertTrue(error.candidates)
        self.assertTrue(
            all(
                candidate.status == CandidateStatus.INFEASIBLE
                or candidate.status == CandidateStatus.UNSUPPORTED
                for candidate in error.candidates
            )
        )
        self.assertTrue(
            any(
                any(
                    "will not endorse training longer than 3 epochs" in reason
                    for reason in candidate.rejection_reasons
                )
                for candidate in error.candidates
            )
        )

    def test_thousand_rows_five_epochs_is_viable_conditional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(
                Path(temporary),
                dataset_rows=1000,
                max_epochs=5,
            )
        self.assertEqual(plan.dataset.example_count, 1000)
        self.assertEqual(plan.target.max_epochs, 5)
        self.assertTrue(plan.recommended.feasible)
        self.assertEqual(plan.recommended.status, CandidateStatus.CONDITIONAL)
        self.assertTrue(
            any(
                "exceeds the instruction-SFT epoch-cap prior of 3" in reason
                for reason in plan.recommended.rejection_reasons
            )
        )


if __name__ == "__main__":
    unittest.main()
