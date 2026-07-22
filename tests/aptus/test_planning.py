import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aptus.domain import CandidateStatus, Distribution, Method, Objective, gibibytes
from aptus.methods import METHOD_REGISTRY
from aptus.planning import NoFeasiblePlanError, plan_training

from tests.aptus.helpers import make_plan


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


if __name__ == "__main__":
    unittest.main()
