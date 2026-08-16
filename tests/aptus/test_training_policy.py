from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aptus.domain import to_primitive
from aptus.plan_contract import plan_id_for_payload
from aptus.training_policy import (
    RUN_CORRECTION_SCHEMA_VERSION,
    TRAINING_POLICY_SCHEMA_VERSION,
    attach_run_correction,
    attach_training_policy,
    build_run_correction_from_metrics,
    build_run_correction_from_metrics_path,
    build_training_policy_presentation,
    classify_instruction_sft_policy,
    classify_run_loss_signal,
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
            [
                "rank",
                "alpha",
                "learning_rate",
                "completions_mask",
                "epochs",
                "dataset_size",
            ],
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

    def test_four_rows_one_epoch_quotes_supervision_prior_on_dataset_and_epochs(
        self,
    ) -> None:
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
            task="sft",
        )
        by_name = {knob.name: knob for knob in body.knobs}
        self.assertEqual(by_name["epochs"].value, "1")
        self.assertEqual(by_name["dataset_size"].value, "4")
        self.assertEqual(by_name["epochs"].prior_kind, "method-class-prior")
        self.assertEqual(by_name["dataset_size"].prior_kind, "method-class-prior")
        for name in ("epochs", "dataset_size"):
            self.assertIn(
                "below the instruction-SFT supervision prior of 100 rows",
                by_name[name].rationale,
            )
            self.assertNotIn("optimal", by_name[name].rationale.lower())
            self.assertNotIn("sycophant", by_name[name].rationale.lower())

    def test_within_prior_states_request_is_within_instruction_sft_prior(self) -> None:
        body = build_training_policy_presentation(
            method="lora",
            rank=16,
            alpha=32,
            learning_rate=2e-4,
            target_modules=("q_proj", "v_proj"),
            example_count=200,
            max_epochs=3,
            truncation_policy=(
                "completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision"
            ),
            task="sft",
        )
        by_name = {knob.name: knob for knob in body.knobs}
        for name in ("epochs", "dataset_size"):
            self.assertIn(
                "within the instruction-SFT prior",
                by_name[name].rationale,
            )

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


class RunLossSignalClassifierTests(unittest.TestCase):
    """TP4 normative fixture series and contract invariants."""

    _DISALLOWED = {
        "no_automl",
        "no_quality_from_loss",
        "no_weight_decay_as_sycophancy_fix",
    }
    _NON_CLAIMS = (
        "Training loss is not model quality.",
        "Validation split loss is not an aptus.evaluation-result.v1 decision.",
    )

    def _assert_common(self, correction) -> None:
        primitive = correction.to_primitive()
        self.assertEqual(primitive["schema_version"], RUN_CORRECTION_SCHEMA_VERSION)
        self.assertEqual(
            primitive["source"],
            "train_loss_observations+validation_loss_observations",
        )
        codes = {item["code"] for item in primitive["disallowed_suggestions"]}
        self.assertEqual(codes, self._DISALLOWED)
        for claim in self._NON_CLAIMS:
            self.assertIn(claim, primitive["non_claims"])
        # Run-correction must not claim to be an evaluation-result decision.
        self.assertIn(
            "Validation split loss is not an aptus.evaluation-result.v1 decision.",
            primitive["non_claims"],
        )
        self.assertNotEqual(primitive["schema_version"], "aptus.evaluation-result.v1")

    def test_fixture_rose(self) -> None:
        correction = classify_run_loss_signal([1.0, 0.4], [0.9, 1.1])
        self.assertEqual(correction.kind, "eval-rose")
        self._assert_common(correction)
        self.assertEqual(correction.operator_next_step.action, "replan-with-fact-hints")
        self.assertEqual(correction.next_plan_hints[0].fact, "target.max_epochs")
        self.assertEqual(correction.next_plan_hints[0].direction, "decrease")

    def test_fixture_collapsed_with_eval_down(self) -> None:
        correction = classify_run_loss_signal([1.0, 0.05], [0.8, 0.4])
        self.assertEqual(correction.kind, "loss-collapsed")
        self._assert_common(correction)
        self.assertEqual(correction.next_plan_hints[0].fact, "target.max_epochs")

    def test_fixture_collapsed_missing_eval(self) -> None:
        correction = classify_run_loss_signal([1.0, 0.05], None)
        self.assertEqual(correction.kind, "loss-collapsed")

    def test_fixture_flat(self) -> None:
        correction = classify_run_loss_signal([1.0, 0.95], None)
        self.assertEqual(correction.kind, "loss-flat")
        self._assert_common(correction)
        self.assertEqual(correction.next_plan_hints[0].direction, "increase")
        hint_facts = [hint.fact for hint in correction.next_plan_hints]
        self.assertNotIn("weight_decay", hint_facts)
        self.assertNotIn("target.weight_decay", hint_facts)

    def test_fixture_single_point(self) -> None:
        correction = classify_run_loss_signal([1.0], [])
        self.assertEqual(correction.kind, "none")
        self.assertEqual(correction.next_plan_hints, ())
        self.assertEqual(correction.operator_next_step.action, "none")
        self._assert_common(correction)

    def test_fixture_empty(self) -> None:
        correction = classify_run_loss_signal(None, None)
        self.assertEqual(correction.kind, "none")
        self._assert_common(correction)

    def test_fixture_both_down_is_none(self) -> None:
        # Not rose (eval did not rise); not collapsed enough (0.4 >= 0.2).
        correction = classify_run_loss_signal([1.0, 0.4], [0.9, 0.5])
        self.assertEqual(correction.kind, "none")
        self._assert_common(correction)

    def test_eval_rose_wins_when_also_collapsed(self) -> None:
        # BiLoRA rule: both series present, train collapsed AND eval rose → rose.
        correction = classify_run_loss_signal([1.0, 0.05], [0.9, 1.2])
        self.assertEqual(correction.kind, "eval-rose")

    def test_non_finite_series_abstains(self) -> None:
        correction = classify_run_loss_signal([1.0, float("nan")], [0.9, 1.1])
        self.assertEqual(correction.kind, "none")

    def test_build_from_metrics_and_path(self) -> None:
        metrics = {
            "train_loss_observations": [1.0, 0.4],
            "validation_loss_observations": [0.9, 1.1],
        }
        from_map = build_run_correction_from_metrics(metrics)
        assert from_map is not None
        self.assertEqual(from_map.kind, "eval-rose")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps(metrics), encoding="utf-8")
            from_path = build_run_correction_from_metrics_path(path)
            assert from_path is not None
            self.assertEqual(from_path.kind, "eval-rose")
            missing = build_run_correction_from_metrics_path(Path(tmp) / "absent.json")
            self.assertIsNone(missing)

    def test_attach_run_correction_does_not_change_plan_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp), gpu_count=1)
        base = to_primitive(plan)
        plan_id = plan_id_for_payload(base)
        correction = classify_run_loss_signal([1.0, 0.05], None)
        attached = attach_run_correction(base, correction)
        self.assertIn("run_correction", attached)
        self.assertEqual(plan_id_for_payload(base), plan_id)
        self.assertEqual(plan_id_for_payload(attached), plan_id)


if __name__ == "__main__":
    unittest.main()
