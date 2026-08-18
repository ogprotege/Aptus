import unittest

from aptus.run_disposition import (
    DISPOSITION_NON_CLAIMS,
    build_run_disposition,
    run_disposition_from_primitive,
)


class RunDispositionTests(unittest.TestCase):
    def test_use_sets_load_adapter_and_required_non_claims(self) -> None:
        body = build_run_disposition(
            kind="use",
            job_id="job_" + "a" * 32,
            plan_id="plan_abc",
            candidate_id="cand_abc",
            run_id="run_abc",
            attested_at="2026-08-18T00:00:00+00:00",
            previous_kind=None,
            validation_state="measured-run-pass",
            run_correction_kind="none",
            evaluation_decision="omitted",
        ).to_primitive()
        self.assertEqual(body["schema_version"], "aptus.run-disposition.v1")
        self.assertEqual(body["kind"], "use")
        self.assertEqual(body["source"], "operator-attested")
        self.assertEqual(body["operator_next_step"]["action"], "load-adapter")
        self.assertEqual(body["operator_next_step"]["label"], "Load this adapter")
        for claim in DISPOSITION_NON_CLAIMS:
            self.assertIn(claim, body["non_claims"])

    def test_done_and_stop_have_no_next_plan(self) -> None:
        done = build_run_disposition(
            kind="done",
            job_id="job_" + "b" * 32,
            plan_id="plan_abc",
            candidate_id="cand_abc",
            run_id=None,
            attested_at="2026-08-18T00:00:00+00:00",
            previous_kind="use",
            validation_state="measured-run-pass",
            run_correction_kind="loss-flat",
            evaluation_decision="fail",
        )
        self.assertEqual(done.operator_next_step.action, "none")
        self.assertEqual(done.previous_kind, "use")
        stop = build_run_disposition(
            kind="stop",
            job_id="job_" + "c" * 32,
            plan_id="plan_abc",
            candidate_id="cand_abc",
            run_id=None,
            attested_at="2026-08-18T00:00:00+00:00",
            previous_kind=None,
            validation_state="measured-run-pass",
            run_correction_kind=None,
            evaluation_decision="omitted",
        )
        self.assertEqual(
            stop.operator_next_step.label, "Don't use this. Don't train this again."
        )

    def test_unknown_kind_and_cut_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_run_disposition(
                kind="cut",
                job_id="job_" + "d" * 32,
                plan_id="plan_abc",
                candidate_id="cand_abc",
                run_id=None,
                attested_at="2026-08-18T00:00:00+00:00",
                previous_kind=None,
                validation_state=None,
                run_correction_kind=None,
                evaluation_decision="omitted",
            )

    def test_from_primitive_requires_all_non_claims(self) -> None:
        payload = build_run_disposition(
            kind="use",
            job_id="job_" + "e" * 32,
            plan_id="plan_abc",
            candidate_id="cand_abc",
            run_id=None,
            attested_at="2026-08-18T00:00:00+00:00",
            previous_kind=None,
            validation_state=None,
            run_correction_kind=None,
            evaluation_decision="omitted",
        ).to_primitive()
        payload["non_claims"] = list(payload["non_claims"])[:-1]
        with self.assertRaises(ValueError):
            run_disposition_from_primitive(payload)
