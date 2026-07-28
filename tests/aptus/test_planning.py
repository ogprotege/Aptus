import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aptus.domain import (
    Backend,
    CandidateStatus,
    Distribution,
    Method,
    Objective,
    TrainingRuntime,
    gibibytes,
)
from aptus.methods import METHOD_REGISTRY
from aptus.planning import NoFeasiblePlanError, estimate_candidate, plan_training
from aptus.profiling import build_hardware_spec

from tests.aptus.helpers import make_plan, make_qwen3_moe_plan


class PlannerTests(unittest.TestCase):
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
            with self.assertRaises(NoFeasiblePlanError):
                make_plan(Path(temporary), host_ram_gib=1, disk_free_gib=0.1)

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
        self.assertEqual({item.method for item in viable}, {Method.LORA, Method.QLORA})
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
        qlora = next(item for item in viable if item.method == Method.QLORA)
        self.assertEqual(qlora.quantization, "mlx-4bit-groupwise")
        self.assertFalse(hardware.devices[0].supports_4bit)
        self.assertTrue(any("not bitsandbytes" in item for item in qlora.assumptions))

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

    def test_qwen3_moe_near_match_has_no_viable_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_qwen3_moe_plan(Path(temporary))
            with self.assertRaises(NoFeasiblePlanError):
                plan_training(
                    model=replace(plan.model, architecture="Qwen3MoeModel"),
                    dataset=plan.dataset,
                    hardware=plan.hardware,
                    target=plan.target,
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


if __name__ == "__main__":
    unittest.main()
