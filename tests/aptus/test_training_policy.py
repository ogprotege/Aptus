from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aptus.domain import to_primitive
from aptus.plan_contract import plan_id_for_payload
from aptus.training_policy import (
    TRAINING_POLICY_SCHEMA_VERSION,
    attach_training_policy,
    build_training_policy_presentation,
    classify_instruction_sft_policy,
)
from tests.aptus.helpers import make_plan


class TrainingPolicyPresentationTests(unittest.TestCase):
    def test_adapter_priors_are_labeled_priors_not_optima(self) -> None:
        body = build_training_policy_presentation(
            method="lora",
            rank=16,
            alpha=32,
            learning_rate=2e-4,
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
            example_count=4,
            max_epochs=1,
            truncation_policy=(
                "completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision"
            ),
        )
        text = json.dumps(body.to_primitive()).lower()
        self.assertIn("prior", text)
        self.assertNotIn("optimal", text)
        self.assertNotIn("best", text)
        names = [k.name for k in body.knobs]
        self.assertEqual(
            names,
            ["rank", "alpha", "learning_rate", "completions_mask"],
        )
        primitive = body.to_primitive()
        self.assertEqual(primitive["schema_version"], TRAINING_POLICY_SCHEMA_VERSION)
        self.assertEqual(primitive["policy_version"], "aptus-training-policy-v1")
        self.assertIn(
            "These knobs are not a prediction of model quality.",
            primitive["non_claims"],
        )

    def test_full_method_uses_full_lr_prior(self) -> None:
        body = build_training_policy_presentation(
            method="full",
            rank=0,
            alpha=0,
            learning_rate=2e-5,
            target_modules=(),
            example_count=1000,
            max_epochs=1,
            truncation_policy=(
                "completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision"
            ),
        )
        lr = next(k for k in body.knobs if k.name == "learning_rate")
        self.assertIn("2e-05", lr.value.replace("0.00002", "2e-05"))
        self.assertEqual(lr.prior_kind, "method-class-prior")

    def test_knob_rationales_match_v02_priors(self) -> None:
        body = build_training_policy_presentation(
            method="lora",
            rank=16,
            alpha=32,
            learning_rate=2e-4,
            target_modules=("q_proj", "v_proj"),
            example_count=4,
            max_epochs=1,
            truncation_policy=(
                "completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision"
            ),
        )
        by_name = {knob.name: knob for knob in body.knobs}
        self.assertEqual(
            by_name["rank"].rationale,
            "Adapter rank 16 is the Aptus v0.2 objective and dataset-volume prior, "
            "not a tuned optimum.",
        )
        self.assertEqual(
            by_name["rank"].prior_kind, "objective-and-token-volume-prior"
        )
        self.assertEqual(
            by_name["alpha"].rationale,
            "Adapter alpha 32 follows the Aptus v0.2 alpha=2*rank policy.",
        )
        self.assertEqual(
            by_name["learning_rate"].rationale,
            "Learning rate 0.0002 is an Aptus v0.2 method-class prior, "
            "not a tuned optimum.",
        )
        self.assertEqual(
            by_name["completions_mask"].rationale,
            "Loss is computed on assistant/completion tokens only; prompt tokens "
            "are masked. Empty supervision is refused.",
        )
        self.assertEqual(by_name["completions_mask"].prior_kind, "compiler-contract")

    def test_attach_training_policy_does_not_change_plan_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp), gpu_count=1)
        base = to_primitive(plan)
        plan_id = plan_id_for_payload(base)
        recommended = plan.recommended
        policy = build_training_policy_presentation(
            method=recommended.method.value,
            rank=recommended.rank,
            alpha=recommended.alpha,
            learning_rate=recommended.learning_rate,
            target_modules=recommended.target_modules,
            example_count=plan.dataset.example_count,
            max_epochs=plan.target.max_epochs,
            truncation_policy=(
                "completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision"
            ),
        )
        attached = attach_training_policy(base, policy)
        self.assertIn("training_policy", attached)
        self.assertEqual(plan_id_for_payload(base), plan_id)
        # Identity must ignore presentation-only training_policy if present.
        self.assertEqual(plan_id_for_payload(attached), plan_id)


class InstructionSftPolicyTests(unittest.TestCase):
    def test_path_alpha_four_rows_one_epoch_is_conditional_not_infeasible(self) -> None:
        verdict = classify_instruction_sft_policy(
            example_count=4, max_epochs=1, task="sft"
        )
        self.assertEqual(verdict.status, "conditional")
        self.assertTrue(
            any("below the instruction-SFT supervision prior" in r for r in verdict.reasons)
        )

    def test_four_rows_ten_epochs_is_infeasible(self) -> None:
        verdict = classify_instruction_sft_policy(
            example_count=4, max_epochs=10, task="sft"
        )
        self.assertEqual(verdict.status, "infeasible")

    def test_thousand_rows_five_epochs_is_conditional_and_keeps_requested_count(
        self,
    ) -> None:
        verdict = classify_instruction_sft_policy(
            example_count=1000, max_epochs=5, task="sft"
        )
        self.assertEqual(verdict.status, "conditional")
        self.assertTrue(any("will not rewrite" in r for r in verdict.reasons))

    def test_two_hundred_rows_ten_epochs_is_infeasible_parrot_prior(self) -> None:
        verdict = classify_instruction_sft_policy(
            example_count=200, max_epochs=10, task="sft"
        )
        self.assertEqual(verdict.status, "infeasible")
        self.assertTrue(any("parrot/sycophancy" in r for r in verdict.reasons))

    def test_two_hundred_rows_three_epochs_is_none(self) -> None:
        verdict = classify_instruction_sft_policy(
            example_count=200, max_epochs=3, task="sft"
        )
        self.assertEqual(verdict.status, "none")
        self.assertEqual(verdict.reasons, ())


if __name__ == "__main__":
    unittest.main()
