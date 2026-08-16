from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aptus.domain import Backend, Method, Objective, TrainingTarget
from aptus.planning import plan_training
from aptus.profiling import build_hardware_spec, build_model_spec, profile_dataset
from aptus.refusal import format_candidate_refusal_block, guide_rejection_reason
from tests.aptus.helpers import make_dataset, make_plan


class RefusalGuidanceTests(unittest.TestCase):
    def test_unknown_capacity_maps_to_stable_codes(self) -> None:
        memory = guide_rejection_reason(
            "Usable per-device memory is unknown; Aptus will not treat total capacity as free."
        )
        host = guide_rejection_reason(
            "Host RAM free is unknown; Aptus will not treat total host RAM as free."
        )
        disk = guide_rejection_reason(
            "Free disk is unknown; Aptus will not assume enough staging space."
        )
        width = guide_rejection_reason(
            "Model intermediate_size is required for MLP adapter targets; Aptus will not invent 4 × hidden_size."
        )
        self.assertEqual(memory.reason_code, "unknown_device_free_memory")
        self.assertEqual(host.reason_code, "unknown_host_ram_free")
        self.assertEqual(disk.reason_code, "unknown_disk_free")
        self.assertEqual(width.reason_code, "missing_intermediate_size")
        self.assertTrue(memory.operator_actionable)

    def test_full_fp16_maps_to_stable_code_and_changeable_facts(self) -> None:
        guided = guide_rejection_reason(
            "Full-parameter FP16 training is fail-closed in Aptus v0.2 because the "
            "generated mixed-precision path does not retain verified FP32 trainable "
            "master weights."
        )
        self.assertEqual(guided.reason_code, "full_fp16")
        self.assertIn("hardware.devices[].supports_bf16", guided.changeable_facts)
        self.assertTrue(guided.operator_actionable)
        self.assertFalse(guided.none_in_catalog)

    def test_multi_gpu_on_single_is_not_ready_language(self) -> None:
        guided = guide_rejection_reason("ddp requires at least two GPUs.")
        self.assertEqual(guided.reason_code, "multi_gpu_on_single")
        self.assertIn("hardware.devices", guided.changeable_facts)
        block = format_candidate_refusal_block(
            status="unsupported",
            reasons=["ddp requires at least two GPUs."],
        )
        self.assertIn("What can change", block)
        self.assertNotIn("ready for multi-GPU training", block.lower())

    def test_mlx_pilot_required_states_none_in_catalog(self) -> None:
        guided = guide_rejection_reason(
            "MLX-LM support is pilot-required: the unified-memory estimate is "
            "provisional and cannot guarantee that the exact model and data fit."
        )
        self.assertEqual(guided.reason_code, "conditional_pilot_required")
        self.assertTrue(guided.none_in_catalog)
        self.assertEqual(guided.changeable_facts, ())
        block = format_candidate_refusal_block(
            status="conditional",
            reasons=[guided.source_reason],
        )
        self.assertIn(
            "No supported correction exists in the current Aptus catalog",
            block,
        )

    def test_full_fp16_is_rejected_on_non_bf16_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = make_dataset(root)
            dataset = profile_dataset(
                dataset_path, sample_limit=64, sequence_length=128
            )
            model = build_model_spec(
                model_id="example/model-1b",
                revision="a" * 40,
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
                backend=Backend.CUDA,
                gpu_count=1,
                vram_gib=24,
                free_vram_gib=22,
                supports_bf16=False,
                supports_4bit=True,
                host_ram_gib=64,
                host_ram_free_gib=56,
                reserve_gib=2,
                disk_free_gib=500,
            )
            target = TrainingTarget(
                objective=Objective.QUALITY,
                sequence_length=128,
                effective_batch_size=8,
                max_epochs=1,
                task="sft",
                packing=False,
                checkpoint_steps=10,
            )
            plan = plan_training(
                model=model, dataset=dataset, hardware=hardware, target=target
            )
        full_single = next(
            item
            for item in plan.candidates
            if item.method == Method.FULL and item.distribution.value == "single"
        )
        self.assertEqual(full_single.status.value, "unsupported")
        self.assertTrue(
            any(
                "Full-parameter FP16" in reason
                for reason in full_single.rejection_reasons
            )
        )
        guided = guide_rejection_reason(full_single.rejection_reasons[0])
        self.assertEqual(guided.reason_code, "full_fp16")

    def test_single_gpu_ddp_rows_remain_unsupported_with_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp), gpu_count=1)
        ddp_rows = [
            item for item in plan.candidates if item.distribution.value == "ddp"
        ]
        self.assertTrue(ddp_rows)
        for item in ddp_rows:
            self.assertEqual(item.status.value, "unsupported")
            self.assertTrue(
                any("at least two GPUs" in reason for reason in item.rejection_reasons)
            )
            guided = guide_rejection_reason(item.rejection_reasons[0])
            self.assertEqual(guided.reason_code, "multi_gpu_on_single")

    def test_dataset_below_sft_prior_maps_to_stable_code(self) -> None:
        guided = guide_rejection_reason(
            "Dataset example_count is below the instruction-SFT supervision prior of "
            "100 rows; this is not a justified domain adaptation."
        )
        self.assertEqual(guided.reason_code, "dataset_below_sft_prior")
        self.assertEqual(guided.changeable_facts, ("dataset.example_count",))
        self.assertTrue(guided.operator_actionable)
        self.assertFalse(guided.none_in_catalog)

    def test_dataset_too_small_for_requested_epochs_maps_to_stable_code(self) -> None:
        guided = guide_rejection_reason(
            "Dataset example_count is below 100 rows; Aptus will not endorse training "
            "longer than 3 epochs on that set."
        )
        self.assertEqual(guided.reason_code, "dataset_too_small_for_requested_epochs")
        self.assertEqual(
            guided.changeable_facts,
            ("dataset.example_count", "target.max_epochs"),
        )
        self.assertTrue(guided.operator_actionable)

    def test_epoch_cap_prior_maps_to_stable_code(self) -> None:
        guided = guide_rejection_reason(
            "Requested max_epochs exceeds the instruction-SFT epoch-cap prior of 3; "
            "Aptus will not rewrite the requested epoch count."
        )
        self.assertEqual(guided.reason_code, "epoch_cap_prior")
        self.assertEqual(guided.changeable_facts, ("target.max_epochs",))
        self.assertTrue(guided.operator_actionable)

    def test_small_corpus_high_epoch_maps_to_stable_code(self) -> None:
        guided = guide_rejection_reason(
            "Small instruction corpus (under 300 rows) with max_epochs >= 10 matches "
            "the parrot/sycophancy over-training prior."
        )
        self.assertEqual(guided.reason_code, "small_corpus_high_epoch")
        self.assertEqual(
            guided.changeable_facts,
            ("dataset.example_count", "target.max_epochs"),
        )
        self.assertTrue(guided.operator_actionable)


if __name__ == "__main__":
    unittest.main()
