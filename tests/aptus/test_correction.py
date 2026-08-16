from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aptus.correction import (
    CORRECTION_SCHEMA_VERSION,
    _direction_for_fact,
    attach_correction,
    build_no_path_correction,
    build_plan_correction,
)
from aptus.domain import Backend, Method, Objective, TrainingTarget, to_primitive
from aptus.plan_contract import plan_id_for_payload
from aptus.planning import NoFeasiblePlanError, plan_training
from aptus.profiling import build_hardware_spec, build_model_spec, profile_dataset
from aptus.refusal import guide_rejection_reason
from tests.aptus.helpers import make_dataset, make_plan


class PlanCorrectionTests(unittest.TestCase):
    def test_feasible_plan_selects_recommended_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp), gpu_count=1, objective=Objective.SPEED)
        correction = build_plan_correction(plan)
        payload = correction.to_primitive()
        self.assertEqual(payload["schema_version"], CORRECTION_SCHEMA_VERSION)
        self.assertEqual(payload["kind"], "select-candidate")
        self.assertEqual(
            payload["recommended_candidate_id"], plan.recommended.candidate_id
        )
        self.assertIn(payload["recommended_status"], {"feasible", "conditional"})
        self.assertEqual(payload["ranking_objective"], plan.target.objective.value)
        self.assertIn(
            payload["operator_next_step"]["action"],
            {"compile-recommended", "confirm-pilot-then-train"},
        )
        messages = " ".join(
            item["message"].lower() for item in payload["disallowed_suggestions"]
        )
        self.assertNotIn("invent a fifth training method", messages)

    def test_conditional_plan_marks_pilot_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = profile_dataset(
                make_dataset(root), sample_limit=64, sequence_length=128
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
                backend=Backend.MPS,
                gpu_count=1,
                vram_gib=32,
                supports_bf16=True,
                supports_4bit=True,
                host_ram_gib=64,
                host_ram_free_gib=48,
                reserve_gib=8,
                disk_free_gib=200,
            )
            plan = plan_training(
                model=model,
                dataset=dataset,
                hardware=hardware,
                target=TrainingTarget(
                    task="sft",
                    objective=Objective.MEMORY,
                    sequence_length=128,
                    effective_batch_size=1,
                    max_epochs=1,
                    method_preference=Method.QLORA,
                    evaluation_fraction=0.25,
                    checkpoint_steps=1,
                    optimizer_steps=3,
                ),
            )
        self.assertEqual(plan.recommended.status.value, "conditional")
        correction = build_plan_correction(plan)
        self.assertEqual(correction.kind, "select-candidate")
        self.assertTrue(correction.pilot_required)
        self.assertEqual(
            correction.operator_next_step.action, "confirm-pilot-then-train"
        )
        self.assertEqual(correction.recommended_status, "conditional")

    def test_no_path_emits_fact_hints_without_unsupported_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = profile_dataset(
                make_dataset(root), sample_limit=64, sequence_length=128
            )
            model = build_model_spec(
                model_id="example/model-7b",
                revision="b" * 40,
                family="llama",
                parameters_b=7,
                hidden_size=4096,
                intermediate_size=11008,
                layers=32,
                context_length=4096,
                license_name="apache-2.0",
                training_allowed=True,
            )
            # Tiny VRAM forces analytic infeasible rows.
            hardware = build_hardware_spec(
                backend=Backend.CUDA,
                gpu_count=1,
                vram_gib=1,
                free_vram_gib=1,
                supports_bf16=True,
                supports_4bit=False,
                supports_8bit=False,
                host_ram_gib=8,
                host_ram_free_gib=4,
                reserve_gib=0.5,
                disk_free_gib=5,
            )
            with self.assertRaises(NoFeasiblePlanError) as raised:
                plan_training(
                    model=model,
                    dataset=dataset,
                    hardware=hardware,
                    target=TrainingTarget(
                        task="sft",
                        objective=Objective.MEMORY,
                        sequence_length=2048,
                        effective_batch_size=16,
                        max_epochs=1,
                        method_preference=None,
                        evaluation_fraction=0.1,
                        checkpoint_steps=100,
                    ),
                )
        error = raised.exception
        correction = build_no_path_correction(
            error.candidates,
            ranking_objective=Objective.MEMORY,
        )
        payload = correction.to_primitive()
        self.assertEqual(payload["kind"], "no-path")
        self.assertIsNone(payload["recommended_candidate_id"])
        self.assertEqual(payload["operator_next_step"]["action"], "change-facts")
        self.assertTrue(payload["primary_reason_codes"] or payload["fact_hints"])
        messages = " ".join(
            item["message"].lower() for item in payload["disallowed_suggestions"]
        )
        self.assertIn("do not invent a training method", messages)
        self.assertNotIn("enable fsdp to make it fit", messages)
        # Hints must only cite catalog facts, never a fifth method.
        for hint in payload["fact_hints"]:
            self.assertNotIn("dora", hint["fact"].lower())
            self.assertNotIn("bitfit", hint["fact"].lower())

    def test_attach_correction_does_not_change_plan_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp), gpu_count=1)
        base = to_primitive(plan)
        plan_id = plan_id_for_payload(base)
        correction = build_plan_correction(plan)
        attached = attach_correction(base, correction)
        self.assertIn("correction", attached)
        self.assertEqual(plan_id_for_payload(base), plan_id)
        # Identity must ignore presentation-only correction if present.
        self.assertEqual(plan_id_for_payload(attached), plan_id)

    def test_no_path_rejects_viable_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp), gpu_count=1)
        with self.assertRaisesRegex(ValueError, "viable"):
            build_no_path_correction(plan.candidates)

    def test_sft_policy_fact_hint_directions(self) -> None:
        below = guide_rejection_reason(
            "Dataset example_count is below the instruction-SFT supervision prior of "
            "100 rows; this is not a justified domain adaptation."
        )
        self.assertEqual(
            _direction_for_fact("dataset.example_count", below), "increase"
        )
        # Supervision-only conditional: max_epochs is review, not decrease.
        self.assertEqual(_direction_for_fact("target.max_epochs", below), "review")

        too_small = guide_rejection_reason(
            "Dataset example_count is below 100 rows; Aptus will not endorse training "
            "longer than 3 epochs on that set."
        )
        self.assertEqual(
            _direction_for_fact("dataset.example_count", too_small), "increase"
        )
        self.assertEqual(
            _direction_for_fact("target.max_epochs", too_small), "decrease"
        )

        epoch_cap = guide_rejection_reason(
            "Requested max_epochs exceeds the instruction-SFT epoch-cap prior of 3; "
            "Aptus will not rewrite the requested epoch count."
        )
        self.assertEqual(
            _direction_for_fact("target.max_epochs", epoch_cap), "decrease"
        )

        parrot = guide_rejection_reason(
            "Small instruction corpus (under 300 rows) with max_epochs >= 10 matches "
            "the parrot/sycophancy over-training prior."
        )
        self.assertEqual(
            _direction_for_fact("dataset.example_count", parrot), "increase"
        )
        self.assertEqual(_direction_for_fact("target.max_epochs", parrot), "decrease")


if __name__ == "__main__":
    unittest.main()
