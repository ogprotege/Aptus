from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from aptus.cli import main
from aptus.domain import to_primitive
from aptus.evaluation import (
    CONTRACT_SCHEMA_VERSION,
    EXACT_MATCH_IMPLEMENTATION,
    RESULT_SCHEMA_VERSION,
    attach_evaluation_contract,
    build_evaluation_contract,
    evaluate_predictions,
    evaluation_contract_from_primitive,
)
from aptus.plan_contract import plan_id_for_payload
from tests.aptus.helpers import make_plan

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class EvaluationContractTests(unittest.TestCase):
    def test_contract_binds_dataset_digest_metric_and_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            _write_jsonl(
                gold,
                [
                    {"id": "a", "completion": "Reset the session and sign in."},
                    {"id": "b", "completion": "Export the archive from settings."},
                ],
            )
            contract = build_evaluation_contract(
                dataset_path=gold,
                claim=(
                    "On this two-row gold set, the bound adapter must exact-match "
                    "every completion."
                ),
                threshold=1.0,
                plan_id="plan_0123456789abcdef0123",
                export_digest="a" * 64,
                export_kind="adapter",
            )
        payload = contract.to_primitive()
        self.assertEqual(payload["schema_version"], CONTRACT_SCHEMA_VERSION)
        self.assertEqual(payload["metric"]["name"], "exact_match")
        self.assertEqual(
            payload["metric"]["implementation_version"], EXACT_MATCH_IMPLEMENTATION
        )
        self.assertEqual(payload["threshold"]["minimum"], 1.0)
        self.assertEqual(payload["dataset"]["row_count"], 2)
        self.assertEqual(len(payload["dataset"]["sha256"]), 64)
        self.assertEqual(payload["artifact_binding"]["export_kind"], "adapter")
        self.assertIn(
            "Training finished is not an evaluation pass.", payload["non_claims"]
        )
        rebuilt = evaluation_contract_from_primitive(payload)
        self.assertEqual(rebuilt.digest(), contract.digest())

    def test_local_dataset_path_is_not_contract_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "one" / "gold.jsonl"
            second = Path(tmp) / "two" / "gold.jsonl"
            first.parent.mkdir()
            second.parent.mkdir()
            rows = [{"id": "1", "completion": "Same gold."}]
            _write_jsonl(first, rows)
            _write_jsonl(second, rows)
            left = build_evaluation_contract(
                dataset_path=first,
                claim="Same bound gold.",
                threshold=0.5,
            )
            right = build_evaluation_contract(
                dataset_path=second,
                claim="Same bound gold.",
                threshold=0.5,
            )
        self.assertEqual(left.digest(), right.digest())
        self.assertNotEqual(left.to_primitive()["dataset"].get("path"), None)

    def test_numeric_identity_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            _write_jsonl(
                gold,
                [
                    {"id": 1, "completion": "one"},
                    {"id": 2, "completion": "two"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                build_evaluation_contract(
                    dataset_path=gold,
                    claim="IDs must be strings.",
                    threshold=1.0,
                )

    def test_from_primitive_rejects_rewritten_literals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            _write_jsonl(gold, [{"id": "a", "completion": "one"}])
            payload = build_evaluation_contract(
                dataset_path=gold,
                claim="Closed literals.",
                threshold=1.0,
            ).to_primitive()
        payload["dataset"]["format"] = "csv"
        with self.assertRaisesRegex(ValueError, "jsonl"):
            evaluation_contract_from_primitive(payload)

    def test_unsupported_metric_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            _write_jsonl(gold, [{"completion": "ok"}])
            with self.assertRaisesRegex(ValueError, "exact_match"):
                build_evaluation_contract(
                    dataset_path=gold,
                    claim="Unsupported metric.",
                    threshold=1.0,
                    metric="bleu",
                )

    def test_threshold_must_be_unit_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            _write_jsonl(gold, [{"completion": "ok"}])
            with self.assertRaisesRegex(ValueError, "threshold"):
                build_evaluation_contract(
                    dataset_path=gold,
                    claim="Bad threshold.",
                    threshold=1.5,
                )


class EvaluationScoringTests(unittest.TestCase):
    def test_exact_match_pass_fail_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            predictions = Path(tmp) / "pred.jsonl"
            _write_jsonl(
                gold,
                [
                    {"id": "a", "completion": "Hello world"},
                    {"id": "b", "completion": "Goodbye"},
                ],
            )
            contract = build_evaluation_contract(
                dataset_path=gold,
                claim="Exact match after whitespace collapse.",
                threshold=1.0,
            )
            _write_jsonl(
                predictions,
                [
                    {"id": "a", "prediction": "  Hello   world  "},
                    {"id": "b", "prediction": "Goodbye"},
                ],
            )
            passed = evaluate_predictions(contract, gold, predictions)
            self.assertEqual(passed.decision, "pass")
            self.assertEqual(passed.score, 1.0)
            self.assertEqual(passed.schema_version, RESULT_SCHEMA_VERSION)
            self.assertEqual(passed.contract_sha256, contract.digest())

            _write_jsonl(
                predictions,
                [
                    {"id": "a", "prediction": "Hello world"},
                    {"id": "b", "prediction": "wrong"},
                ],
            )
            failed = evaluate_predictions(contract, gold, predictions)
            self.assertEqual(failed.decision, "fail")
            self.assertEqual(failed.score, 0.5)

    def test_missing_or_extra_ids_abstain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            predictions = Path(tmp) / "pred.jsonl"
            _write_jsonl(
                gold,
                [
                    {"id": "a", "completion": "one"},
                    {"id": "b", "completion": "two"},
                ],
            )
            contract = build_evaluation_contract(
                dataset_path=gold,
                claim="Require complete alignment.",
                threshold=0.0,
            )
            _write_jsonl(predictions, [{"id": "a", "prediction": "one"}])
            missing = evaluate_predictions(contract, gold, predictions)
            self.assertEqual(missing.decision, "abstain")
            self.assertIsNone(missing.score)
            self.assertTrue(
                any("missing" in reason for reason in missing.decision_reasons)
            )

            _write_jsonl(
                predictions,
                [
                    {"id": "a", "prediction": "one"},
                    {"id": "b", "prediction": "two"},
                    {"id": "c", "prediction": "extra"},
                ],
            )
            extra = evaluate_predictions(contract, gold, predictions)
            self.assertEqual(extra.decision, "abstain")
            self.assertTrue(any("extra" in reason for reason in extra.decision_reasons))

    def test_gold_digest_mismatch_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            other = Path(tmp) / "other.jsonl"
            predictions = Path(tmp) / "pred.jsonl"
            _write_jsonl(gold, [{"id": "a", "completion": "one"}])
            _write_jsonl(other, [{"id": "a", "completion": "changed"}])
            _write_jsonl(predictions, [{"id": "a", "prediction": "one"}])
            contract = build_evaluation_contract(
                dataset_path=gold,
                claim="Bound gold digest.",
                threshold=1.0,
            )
            result = evaluate_predictions(contract, other, predictions)
            self.assertEqual(result.decision, "abstain")
            self.assertTrue(
                any("gold digest" in reason for reason in result.decision_reasons)
            )

    def test_bound_export_digest_requires_caller_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            predictions = Path(tmp) / "pred.jsonl"
            _write_jsonl(gold, [{"id": "a", "completion": "one"}])
            _write_jsonl(predictions, [{"id": "a", "prediction": "one"}])
            contract = build_evaluation_contract(
                dataset_path=gold,
                claim="Export must be presented.",
                threshold=1.0,
                export_digest="b" * 64,
                export_kind="adapter",
            )
            omitted = evaluate_predictions(contract, gold, predictions)
            self.assertEqual(omitted.decision, "abstain")
            self.assertIsNone(omitted.score)
            self.assertTrue(
                any(
                    "export digest is required" in reason
                    for reason in omitted.decision_reasons
                )
            )
            matched = evaluate_predictions(
                contract,
                gold,
                predictions,
                expected_export_digest="b" * 64,
            )
            self.assertEqual(matched.decision, "pass")

    def test_export_digest_mismatch_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            predictions = Path(tmp) / "pred.jsonl"
            _write_jsonl(gold, [{"id": "a", "completion": "one"}])
            _write_jsonl(predictions, [{"id": "a", "prediction": "one"}])
            contract = build_evaluation_contract(
                dataset_path=gold,
                claim="Bind the export.",
                threshold=1.0,
                export_digest="b" * 64,
                export_kind="final-export",
            )
            result = evaluate_predictions(
                contract,
                gold,
                predictions,
                expected_export_digest="c" * 64,
            )
            self.assertEqual(result.decision, "abstain")
            self.assertTrue(
                any("export digest" in reason for reason in result.decision_reasons)
            )

    def test_training_finished_is_not_an_eval_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gold = Path(tmp) / "gold.jsonl"
            predictions = Path(tmp) / "pred.jsonl"
            _write_jsonl(gold, [{"completion": "target"}])
            _write_jsonl(predictions, [{"prediction": "different"}])
            contract = build_evaluation_contract(
                dataset_path=gold,
                claim="Loss is not this decision.",
                threshold=1.0,
            )
            result = evaluate_predictions(contract, gold, predictions)
        self.assertEqual(result.decision, "fail")
        self.assertNotIn(result.decision, {"pass", "measured-run-pass"})
        self.assertIn(
            "Training finished is not an evaluation pass.",
            result.non_claims,
        )


class EvaluationAttachmentTests(unittest.TestCase):
    def test_attach_evaluation_contract_does_not_change_plan_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp), gpu_count=1)
            gold = Path(tmp) / "gold.jsonl"
            _write_jsonl(gold, [{"completion": "ok"}])
            contract = build_evaluation_contract(
                dataset_path=gold,
                claim="Presentation only.",
                threshold=1.0,
            )
        base = to_primitive(plan)
        plan_id = plan_id_for_payload(base)
        attached = attach_evaluation_contract(base, contract)
        self.assertIn("evaluation_contract", attached)
        self.assertEqual(plan_id_for_payload(base), plan_id)
        self.assertEqual(plan_id_for_payload(attached), plan_id)


class EvaluationCliTests(unittest.TestCase):
    def test_eval_contract_and_eval_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            predictions = root / "pred.jsonl"
            contract_path = root / "contract.json"
            result_path = root / "result.json"
            plan = make_plan(root, gpu_count=1)
            plan_path = root / "plan.json"
            attached_path = root / "plan-with-eval.json"
            plan_path.write_text(
                json.dumps(to_primitive(plan), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_jsonl(gold, [{"id": "a", "completion": "Keep the receipt."}])
            _write_jsonl(predictions, [{"id": "a", "prediction": "Keep the receipt."}])
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    [
                        "eval-contract",
                        "--dataset",
                        str(gold),
                        "--claim",
                        "Exact match the receipt instruction.",
                        "--threshold",
                        "1",
                        "--output",
                        str(contract_path),
                        "--attach-plan",
                        str(plan_path),
                        "--plan-output",
                        str(attached_path),
                    ]
                )
            self.assertEqual(status, 0)
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            self.assertEqual(contract["schema_version"], CONTRACT_SCHEMA_VERSION)
            attached = json.loads(attached_path.read_text(encoding="utf-8"))
            self.assertEqual(
                attached["evaluation_contract"]["schema_version"],
                CONTRACT_SCHEMA_VERSION,
            )
            self.assertEqual(attached["plan_id"], to_primitive(plan)["plan_id"])

            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                status = main(
                    [
                        "eval",
                        "--contract",
                        str(contract_path),
                        "--gold",
                        str(gold),
                        "--predictions",
                        str(predictions),
                        "--output",
                        str(result_path),
                    ]
                )
            self.assertEqual(status, 0)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["decision"], "pass")
            self.assertEqual(result["schema_version"], RESULT_SCHEMA_VERSION)


@unittest.skipIf(TestClient is None, "server extra is not installed")
class EvaluationApiTests(unittest.TestCase):
    def test_contract_and_score_endpoints(self) -> None:
        from aptus.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            predictions = root / "pred.jsonl"
            _write_jsonl(gold, [{"id": "a", "completion": "Keep the receipt."}])
            _write_jsonl(predictions, [{"id": "a", "prediction": "Keep the receipt."}])
            client = TestClient(
                create_app(
                    state_dir=root / "state",
                    allow_unauthenticated=True,
                )
            )
            built = client.post(
                "/api/v1/evaluations/contracts",
                json={
                    "dataset_path": str(gold),
                    "claim": "Exact match the receipt instruction.",
                    "threshold": 1,
                },
            )
            self.assertEqual(built.status_code, 200)
            contract = built.json()
            self.assertEqual(contract["schema_version"], CONTRACT_SCHEMA_VERSION)
            scored = client.post(
                "/api/v1/evaluations",
                json={
                    "contract": contract,
                    "gold_path": str(gold),
                    "predictions_path": str(predictions),
                },
            )
            self.assertEqual(scored.status_code, 200)
            result = scored.json()
            self.assertEqual(result["decision"], "pass")
            self.assertEqual(result["schema_version"], RESULT_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
