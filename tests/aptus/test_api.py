import copy
import hashlib
import json
import shutil
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from aptus.api import (
    ApiContext,
    JobRequest,
    ProfileRequest,
    _resolve_static_dir,
    create_app,
)
from aptus.api_contracts import (
    ErrorResponse,
    ModelCompatibilityResponse,
    ModelInspectionReceiptResponse,
    ModelInspectionResponse,
    NoFeasiblePlanResponse,
    PlanCandidateResponse,
    TrainingPlanResponse,
    ValidationResponse,
)
from aptus.catalog import reviewed_qwen3_moe_quantization_layout
from aptus.domain import Backend, ValidationReport, ValidationState, to_primitive
from aptus.execution import ActiveJobError, JobPrerequisiteError
from aptus.local_store import atomic_write_json
from aptus.model_compatibility import (
    create_model_inspection_receipt,
    current_model_policy_snapshot,
    subject_from_model,
    validate_registered_compatibility_path,
)
from aptus.plan_contract import (
    StaleModelPolicyError,
    candidate_id_for_payload,
    plan_id_for_payload,
)
from aptus.profiling import build_hardware_spec, build_model_spec
from aptus.runtime_env import RuntimeInterpreter

from tests.aptus.helpers import (
    QWEN2_5_ACCEPTANCE_MODEL_ID,
    QWEN2_5_ACCEPTANCE_REVISION,
    make_plan,
    make_qwen3_moe_plan,
)

try:
    from fastapi.testclient import TestClient
except ImportError:  # The base package intentionally keeps the server optional.
    TestClient = None


def inspection_receipt_shape(
    model_id: str, resolved_revision: str
) -> dict[str, object]:
    observed_at = "2026-07-29T12:00:00+00:00"
    return {
        "schema_version": "aptus.model-inspection-receipt.v1",
        "receipt_id": "receipt_" + "a" * 20,
        "model_id": model_id,
        "resolved_revision": resolved_revision,
        "observed_facts_sha256": "b" * 64,
        "decision": {
            "schema_version": "aptus.model-compatibility.v2",
            "decision_id": "compat_" + "c" * 20,
            "subject_facts_sha256": "d" * 64,
            "kind": "unknown",
            "family": None,
            "policy_id": None,
            "policy_version": None,
            "paths": [],
            "reason_codes": ["no-policy-match"],
            "evidence_ids": [],
            "reason": "No registered provider policy matches this artifact.",
        },
        "provenance_summary": [
            {
                "field": "family",
                "kind": "provider-declared",
                "source": "Provider model configuration",
                "observed_at": observed_at,
                "resolved_revision": resolved_revision,
            }
        ],
        "provenance_requirement": None,
        "provenance_requirement_met": False,
        "evaluated_at": observed_at,
    }


class ApiContractTests(unittest.TestCase):
    def test_validation_authorization_contract_is_typed_and_coherent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            schema = create_app(
                state_dir=Path(temporary) / "state",
                allow_unauthenticated=True,
            ).openapi()
        validation_schema = schema["components"]["schemas"]["ValidationResponse"]
        status_schema = validation_schema["properties"]["authorization_status"]
        status_variant = next(item for item in status_schema["anyOf"] if "enum" in item)
        self.assertEqual(
            set(status_variant["enum"]), {"current", "deferred", "blocked"}
        )
        self.assertNotIn("authorization_status", validation_schema["required"])

        base = {
            "state": "pilot-pass",
            "project_id": "project_" + "a" * 32,
            "project_revision_id": "revision_" + "b" * 32,
        }
        for status, current, error in (
            ("current", True, None),
            ("deferred", False, "Admission runs at training submission."),
            ("blocked", False, "Another GPU job owns admission."),
        ):
            with self.subTest(status=status):
                validated = ValidationResponse.model_validate(
                    {
                        **base,
                        "authorization_status": status,
                        "authorization_current": current,
                        "authorization_error": error,
                    }
                )
                self.assertEqual(validated.authorization_status, status)

        for state in ("pilot-pass", "execution-approved", "measured-run-pass"):
            with self.subTest(current_state=state):
                ValidationResponse.model_validate(
                    {
                        **base,
                        "state": state,
                        "authorization_status": "current",
                        "authorization_current": True,
                        "authorization_error": None,
                    }
                )

        invalid = (
            {"authorization_current": False},
            {"authorization_status": "deferred"},
            {
                "authorization_status": "current",
                "authorization_current": False,
                "authorization_error": "Not current.",
            },
            {
                "authorization_status": "current",
                "authorization_current": True,
                "authorization_error": "Contradictory error.",
            },
            {
                "state": "static-pass",
                "authorization_status": "current",
                "authorization_current": True,
                "authorization_error": None,
            },
            {
                "authorization_status": "blocked",
                "authorization_current": False,
                "authorization_error": " ",
            },
        )
        for mutation in invalid:
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                ValidationResponse.model_validate({**base, **mutation})

    def test_plan_openapi_declares_typed_no_feasible_policy_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            schema = create_app(
                state_dir=Path(temporary) / "state",
                allow_unauthenticated=True,
            ).openapi()

        response_schema = schema["paths"]["/api/v1/plan"]["post"]["responses"]["422"][
            "content"
        ]["application/json"]["schema"]
        self.assertEqual(
            {item["$ref"] for item in response_schema["anyOf"]},
            {
                "#/components/schemas/ErrorResponse",
                "#/components/schemas/NoFeasiblePlanResponse",
            },
        )
        failure_schema = schema["components"]["schemas"]["NoFeasiblePlanResponse"]
        self.assertFalse(failure_schema["additionalProperties"])
        self.assertTrue(
            {
                "error",
                "message",
                "model",
                "candidates",
                "model_policy_decision",
                "model_policy_decision_source",
                "inspection_receipt",
            }.issubset(failure_schema["required"])
        )
        self.assertEqual(
            failure_schema["properties"]["model"]["$ref"],
            "#/components/schemas/PlanModelSubjectResponse",
        )
        candidate_schema = schema["components"]["schemas"]["PlanCandidateResponse"]
        self.assertTrue(
            {
                "candidate_id",
                "model_policy_decision_id",
                "policy_binding",
                "method",
                "distribution",
                "status",
                "feasible",
                "rejection_reasons",
                "target_modules",
                "runtime_contract",
            }.issubset(candidate_schema["required"])
        )
        self.assertEqual(
            candidate_schema["properties"]["status"]["$ref"],
            "#/components/schemas/CandidateStatus",
        )
        self.assertEqual(
            candidate_schema["properties"]["runtime_contract"]["$ref"],
            "#/components/schemas/InspectedRuntimeContractResponse",
        )
        plan_schema = schema["components"]["schemas"]["TrainingPlanResponse"]
        self.assertIn("model", plan_schema["required"])
        self.assertEqual(
            plan_schema["properties"]["model"]["$ref"],
            "#/components/schemas/PlanModelSubjectResponse",
        )
        model_subject_schema = schema["components"]["schemas"][
            "PlanModelSubjectResponse"
        ]
        self.assertTrue(model_subject_schema["additionalProperties"])
        self.assertEqual(
            set(model_subject_schema["required"]), {"model_id", "revision"}
        )

    def test_candidate_contract_requires_a_coherent_execution_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = to_primitive(make_plan(Path(temporary)))
        candidate = payload["candidates"][0]

        validated = PlanCandidateResponse.model_validate(candidate)

        self.assertEqual(validated.candidate_id, candidate["candidate_id"])
        required_execution_fields = (
            "method",
            "distribution",
            "status",
            "feasible",
            "rejection_reasons",
            "target_modules",
            "runtime_contract",
        )
        for field in required_execution_fields:
            with self.subTest(field=field), self.assertRaises(ValidationError):
                missing = copy.deepcopy(candidate)
                del missing[field]
                PlanCandidateResponse.model_validate(missing)

        incoherent = copy.deepcopy(candidate)
        incoherent["feasible"] = not incoherent["feasible"]
        with self.assertRaisesRegex(ValidationError, "status and feasibility"):
            PlanCandidateResponse.model_validate(incoherent)

    def test_plan_response_requires_an_exact_structural_policy_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = to_primitive(make_qwen3_moe_plan(Path(temporary)))
        TrainingPlanResponse.model_validate(payload)
        missing_model = copy.deepcopy(payload)
        del missing_model["model"]
        with self.assertRaises(ValidationError):
            TrainingPlanResponse.model_validate(missing_model)
        for field, value in (("model_id", " "), ("revision", "mutable-main")):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                invalid_model_subject = copy.deepcopy(payload)
                invalid_model_subject["model"][field] = value
                TrainingPlanResponse.model_validate(invalid_model_subject)

        duplicate_candidates = copy.deepcopy(payload)
        duplicate_candidates["candidates"].append(
            copy.deepcopy(duplicate_candidates["candidates"][0])
        )
        with self.assertRaisesRegex(ValidationError, "candidate IDs must be unique"):
            TrainingPlanResponse.model_validate(duplicate_candidates)

        nonviable_recommendation = copy.deepcopy(payload)
        recommended_id = nonviable_recommendation["recommended"]["candidate_id"]
        nonviable_recommendation["recommended"]["status"] = "infeasible"
        nonviable_recommendation["recommended"]["feasible"] = False
        listed_recommendation = next(
            candidate
            for candidate in nonviable_recommendation["candidates"]
            if candidate["candidate_id"] == recommended_id
        )
        listed_recommendation["status"] = "infeasible"
        listed_recommendation["feasible"] = False
        with self.assertRaisesRegex(ValidationError, "must be viable"):
            TrainingPlanResponse.model_validate(nonviable_recommendation)

        bound_index = next(
            index
            for index, candidate in enumerate(payload["candidates"])
            if candidate["policy_binding"] is not None
        )

        missing_binding = copy.deepcopy(payload)
        bound_id = missing_binding["candidates"][bound_index]["candidate_id"]
        missing_binding["candidates"][bound_index]["policy_binding"] = None
        if missing_binding["recommended"]["candidate_id"] == bound_id:
            missing_binding["recommended"]["policy_binding"] = None
        with self.assertRaisesRegex(ValidationError, "require a binding"):
            TrainingPlanResponse.model_validate(missing_binding)

        mismatched_tuple = copy.deepcopy(payload)
        mismatched_tuple["candidates"][bound_index]["target_modules"] = ["q_proj"]
        if mismatched_tuple["recommended"]["candidate_id"] == bound_id:
            mismatched_tuple["recommended"]["target_modules"] = ["q_proj"]
        with self.assertRaisesRegex(ValidationError, "must be null"):
            TrainingPlanResponse.model_validate(mismatched_tuple)

        divergent_recommendation = copy.deepcopy(payload)
        divergent_recommendation["recommended"]["target_modules"] = ["q_proj"]
        with self.assertRaisesRegex(ValidationError, "equal its listed"):
            TrainingPlanResponse.model_validate(divergent_recommendation)

    def test_path_matched_receipts_require_provider_declared_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = to_primitive(make_qwen3_moe_plan(Path(temporary)))
        receipt = inspection_receipt_shape(
            plan["model"]["model_id"], plan["model"]["revision"]
        )
        receipt["decision"] = plan["model_policy_decision"]
        receipt["provenance_requirement"] = "provider-declared"
        receipt["provenance_requirement_met"] = True
        ModelInspectionReceiptResponse.model_validate(receipt)

        provenance_mutations = {
            "empty": [],
            "duplicate": [
                copy.deepcopy(receipt["provenance_summary"][0]),
                copy.deepcopy(receipt["provenance_summary"][0]),
            ],
            "unsorted": [
                copy.deepcopy(receipt["provenance_summary"][0]),
                {
                    **copy.deepcopy(receipt["provenance_summary"][0]),
                    "field": "architecture",
                },
            ],
            "revision-mismatch": [
                {
                    **copy.deepcopy(receipt["provenance_summary"][0]),
                    "resolved_revision": "0" * 40,
                }
            ],
            "inferred-only": [
                {
                    **copy.deepcopy(receipt["provenance_summary"][0]),
                    "kind": "inferred",
                }
            ],
        }
        for name, provenance in provenance_mutations.items():
            with self.subTest(provenance=name), self.assertRaises(ValidationError):
                incoherent = copy.deepcopy(receipt)
                incoherent["provenance_summary"] = provenance
                ModelInspectionReceiptResponse.model_validate(incoherent)

        for requirement, met in ((None, True), ("provider-declared", False)):
            with (
                self.subTest(requirement=requirement, met=met),
                self.assertRaisesRegex(ValidationError, "provider-declared"),
            ):
                incoherent = copy.deepcopy(receipt)
                incoherent["provenance_requirement"] = requirement
                incoherent["provenance_requirement_met"] = met
                ModelInspectionReceiptResponse.model_validate(incoherent)

    def test_success_and_failure_contracts_reject_incoherent_policy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            success = to_primitive(make_qwen3_moe_plan(Path(temporary)))
        receipt_id = "receipt_" + "f" * 20
        receipt = inspection_receipt_shape(
            success["model"]["model_id"], success["model"]["revision"]
        )
        receipt["decision"] = copy.deepcopy(success["model_policy_decision"])
        receipt["provenance_requirement"] = "provider-declared"
        receipt["provenance_requirement_met"] = True
        success["model_policy_decision_source"] = "provider-inspection"
        success["inspection_receipt"] = receipt
        for candidate in [success["recommended"], *success["candidates"]]:
            binding = candidate["policy_binding"]
            if binding is not None:
                binding["source"] = "provider-inspection"
                binding["inspection_receipt_id"] = receipt_id
        receipt["receipt_id"] = receipt_id
        TrainingPlanResponse.model_validate(success)

        explanatory_difference = copy.deepcopy(success)
        explanatory_difference["inspection_receipt"]["decision"]["reason"] = (
            "Equivalent policy decision with different explanatory text."
        )
        TrainingPlanResponse.model_validate(explanatory_difference)
        semantic_decision_drift = copy.deepcopy(success)
        semantic_decision_drift["inspection_receipt"]["decision"]["family"] = (
            "tampered-family"
        )
        with self.assertRaisesRegex(ValidationError, "semantically match the plan"):
            TrainingPlanResponse.model_validate(semantic_decision_drift)
        for field, value in (
            ("model_id", "different/model"),
            ("resolved_revision", "0" * 40),
        ):
            with (
                self.subTest(receipt_field=field),
                self.assertRaisesRegex(ValidationError, "must match the model subject"),
            ):
                mismatched_receipt_subject = copy.deepcopy(success)
                mismatched_receipt_subject["inspection_receipt"][field] = value
                if field == "resolved_revision":
                    for provenance in mismatched_receipt_subject["inspection_receipt"][
                        "provenance_summary"
                    ]:
                        provenance["resolved_revision"] = value
                TrainingPlanResponse.model_validate(mismatched_receipt_subject)

        receipt_drift = copy.deepcopy(success)
        bound_candidate = next(
            candidate
            for candidate in receipt_drift["candidates"]
            if candidate["policy_binding"] is not None
        )
        drifted_receipt_id = "receipt_" + "e" * 20
        bound_candidate["policy_binding"]["inspection_receipt_id"] = drifted_receipt_id
        if (
            receipt_drift["recommended"]["candidate_id"]
            == bound_candidate["candidate_id"]
        ):
            receipt_drift["recommended"]["policy_binding"]["inspection_receipt_id"] = (
                drifted_receipt_id
            )
        with self.assertRaisesRegex(ValidationError, "exactly match"):
            TrainingPlanResponse.model_validate(receipt_drift)

        rejected = next(
            copy.deepcopy(candidate)
            for candidate in success["candidates"]
            if not candidate["feasible"] and candidate["policy_binding"] is None
        )
        failure = {
            "error": "no_feasible_plan",
            "message": "No candidate passed every hard gate.",
            "model": success["model"],
            "candidates": [rejected],
            "model_policy_decision": success["model_policy_decision"],
            "model_policy_decision_source": "user-attested",
            "inspection_receipt": None,
        }
        NoFeasiblePlanResponse.model_validate(failure)
        missing_failure_subject = copy.deepcopy(failure)
        del missing_failure_subject["model"]
        with self.assertRaises(ValidationError):
            NoFeasiblePlanResponse.model_validate(missing_failure_subject)
        malformed_failure_subject = copy.deepcopy(failure)
        malformed_failure_subject["model"]["revision"] = "main"
        with self.assertRaises(ValidationError):
            NoFeasiblePlanResponse.model_validate(malformed_failure_subject)

        provider_failure = copy.deepcopy(failure)
        provider_failure["model_policy_decision_source"] = "provider-inspection"
        provider_failure["inspection_receipt"] = copy.deepcopy(
            success["inspection_receipt"]
        )
        NoFeasiblePlanResponse.model_validate(provider_failure)
        provider_failure["inspection_receipt"]["model_id"] = "different/model"
        with self.assertRaisesRegex(ValidationError, "must match the model subject"):
            NoFeasiblePlanResponse.model_validate(provider_failure)
        duplicate_failure = copy.deepcopy(failure)
        duplicate_failure["candidates"].append(
            copy.deepcopy(duplicate_failure["candidates"][0])
        )
        with self.assertRaisesRegex(ValidationError, "candidate IDs must be unique"):
            NoFeasiblePlanResponse.model_validate(duplicate_failure)
        for status in ("feasible", "conditional"):
            with (
                self.subTest(status=status),
                self.assertRaisesRegex(
                    ValidationError, "must be infeasible or unsupported"
                ),
            ):
                viable_failure = copy.deepcopy(failure)
                viable_failure["candidates"][0]["status"] = status
                viable_failure["candidates"][0]["feasible"] = True
                NoFeasiblePlanResponse.model_validate(viable_failure)
        missing_reasons = copy.deepcopy(failure)
        missing_reasons["candidates"][0]["rejection_reasons"] = []
        with self.assertRaisesRegex(ValidationError, "with rejection reasons"):
            NoFeasiblePlanResponse.model_validate(missing_reasons)

    def test_request_models_reject_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ProfileRequest(dataset_path="data.jsonl", unknown=True)
        with self.assertRaises(ValidationError):
            JobRequest(
                bundle_dir="bundle",
                project_id="project_" + "a" * 32,
                expected_project_revision_id="revision_" + "b" * 32,
                action="train",
                confirm_full_train=True,
                resume_from="checkpoint-1",
            )

    def test_inspection_response_allows_incomplete_moe_evidence(self) -> None:
        response = ModelInspectionResponse.model_validate(
            {
                "status": "ok",
                "model_id": "provider/incomplete-moe",
                "requested_revision": "main",
                "resolved_revision": "a" * 40,
                "facts": {
                    "model_type": "unknown_moe",
                    "architecture": "UnknownMoeForCausalLM",
                    "moe": {
                        "expert_count": 64,
                        "experts_per_token": None,
                        "expert_intermediate_size": 512,
                        "decoder_sparse_step": 1,
                        "mlp_only_layers": None,
                        "shared_expert_intermediate_size": None,
                    },
                },
                "compatibility": {
                    "status": "unsupported",
                    "family": "unknown_moe",
                    "supported_runtime": None,
                    "supported_methods": [],
                    "compute_backend": None,
                    "distribution": None,
                    "evidence_requirement": "implementation-required",
                    "adapter_profile_id": None,
                    "reason": "The provider topology is incomplete.",
                },
                "provenance": {},
                "inspection_receipt": inspection_receipt_shape(
                    "provider/incomplete-moe", "a" * 40
                ),
            }
        )

        self.assertIsNone(response.facts.moe.experts_per_token)

    def test_model_compatibility_contract_accepts_only_coherent_variants(self) -> None:
        variants = (
            {
                "status": "conditional",
                "family": "qwen3_moe",
                "supported_runtime": "mlx-lm",
                "supported_methods": ["qlora"],
                "compute_backend": "mps",
                "distribution": "single",
                "evidence_requirement": "pilot-required",
                "adapter_profile_id": "attention-qkvo.v1",
                "reason": "A measured pilot remains required.",
            },
            {
                "status": "recognized",
                "family": "llama",
                "supported_runtime": None,
                "supported_methods": [],
                "compute_backend": None,
                "distribution": None,
                "evidence_requirement": "pilot-required",
                "adapter_profile_id": None,
                "reason": "The planner decides the executable path.",
            },
            {
                "status": "unsupported",
                "family": None,
                "supported_runtime": None,
                "supported_methods": [],
                "compute_backend": None,
                "distribution": None,
                "evidence_requirement": "implementation-required",
                "adapter_profile_id": None,
                "reason": "No reviewed policy matches this model.",
            },
        )

        for payload in variants:
            with self.subTest(status=payload["status"]):
                validated = ModelCompatibilityResponse.model_validate(payload)
                self.assertEqual(validated.model_dump(mode="json"), payload)

    def test_inspection_response_requires_receipt_only_on_success(self) -> None:
        success = {
            "status": "ok",
            "model_id": "example/model",
            "requested_revision": "main",
            "resolved_revision": "a" * 40,
            "facts": {},
            "compatibility": {
                "status": "unsupported",
                "family": None,
                "supported_runtime": None,
                "supported_methods": [],
                "compute_backend": None,
                "distribution": None,
                "evidence_requirement": "implementation-required",
                "adapter_profile_id": None,
                "reason": "No reviewed policy matches this model.",
            },
            "provenance": {},
        }
        with self.assertRaises(ValidationError):
            ModelInspectionResponse.model_validate(success)

        unavailable = {
            "status": "unavailable",
            "model_id": "example/model",
            "requested_revision": "main",
            "inspection_receipt": inspection_receipt_shape("example/model", "a" * 40),
        }
        with self.assertRaises(ValidationError):
            ModelInspectionResponse.model_validate(unavailable)

    def test_conditional_response_delegates_to_the_domain_path_validator(self) -> None:
        payload = {
            "status": "conditional",
            "family": "qwen3_moe",
            "supported_runtime": "mlx-lm",
            "supported_methods": ["qlora"],
            "compute_backend": "mps",
            "distribution": "single",
            "evidence_requirement": "pilot-required",
            "adapter_profile_id": "attention-qkvo.v1",
            "reason": "A measured pilot remains required.",
        }

        with patch(
            "aptus.api_contracts.validate_registered_compatibility_path",
            wraps=validate_registered_compatibility_path,
        ) as validate:
            ModelCompatibilityResponse.model_validate(payload)

        validate.assert_called_once()
        self.assertEqual(
            validate.call_args.kwargs["evidence_requirement"],
            "pilot-required",
        )

    def test_conditional_response_rejects_unregistered_model_policy_claims(
        self,
    ) -> None:
        base = {
            "status": "conditional",
            "family": "qwen3_moe",
            "supported_runtime": "mlx-lm",
            "supported_methods": ["qlora"],
            "compute_backend": "mps",
            "distribution": "single",
            "evidence_requirement": "pilot-required",
            "adapter_profile_id": "attention-qkvo.v1",
            "reason": "A measured pilot remains required.",
        }
        claims = (
            {**base, "supported_methods": ["lora"]},
            {**base, "family": "llama"},
        )

        for claim in claims:
            with self.subTest(claim=claim):
                with self.assertRaisesRegex(ValueError, "registered for the model"):
                    ModelCompatibilityResponse.model_validate(claim)

    def test_model_compatibility_contract_rejects_contradictory_evidence(
        self,
    ) -> None:
        conditional = {
            "status": "conditional",
            "family": "qwen3_moe",
            "supported_runtime": "mlx-lm",
            "supported_methods": ["qlora"],
            "compute_backend": "mps",
            "distribution": "single",
            "evidence_requirement": "pilot-required",
            "adapter_profile_id": "attention-qkvo.v1",
            "reason": "A measured pilot remains required.",
        }
        recognized = {
            "status": "recognized",
            "family": "llama",
            "supported_runtime": None,
            "supported_methods": [],
            "compute_backend": None,
            "distribution": None,
            "evidence_requirement": "pilot-required",
            "adapter_profile_id": None,
            "reason": "The planner decides the executable path.",
        }
        unsupported = {
            "status": "unsupported",
            "family": None,
            "supported_runtime": None,
            "supported_methods": [],
            "compute_backend": None,
            "distribution": None,
            "evidence_requirement": "implementation-required",
            "adapter_profile_id": None,
            "reason": "No reviewed policy matches this model.",
        }
        invalid_variants = {
            "null runtime": {**conditional, "supported_runtime": None},
            "empty runtime": {**conditional, "supported_runtime": ""},
            "unknown runtime": {
                **conditional,
                "supported_runtime": "future-runtime",
            },
            "empty methods": {**conditional, "supported_methods": []},
            "empty method name": {**conditional, "supported_methods": [""]},
            "unknown method": {
                **conditional,
                "supported_methods": ["future-method"],
            },
            "duplicate method": {
                **conditional,
                "supported_methods": ["qlora", "qlora"],
            },
            "unregistered method runtime binding": {
                **conditional,
                "supported_methods": ["full"],
            },
            "adapter profile on non-adapter method": {
                **conditional,
                "supported_runtime": "transformers-peft-cuda",
                "supported_methods": ["full"],
                "compute_backend": "cuda",
                "distribution": "single",
            },
            "missing backend": {
                key: value
                for key, value in conditional.items()
                if key != "compute_backend"
            },
            "null backend": {**conditional, "compute_backend": None},
            "empty backend": {**conditional, "compute_backend": ""},
            "unknown backend": {
                **conditional,
                "compute_backend": "future-backend",
            },
            "unregistered runtime backend binding": {
                **conditional,
                "compute_backend": "cuda",
            },
            "null distribution": {**conditional, "distribution": None},
            "empty distribution": {**conditional, "distribution": ""},
            "unknown distribution": {
                **conditional,
                "distribution": "future-distribution",
            },
            "unsupported registered distribution": {
                **conditional,
                "distribution": "ddp",
            },
            "implementation evidence": {
                **conditional,
                "evidence_requirement": "implementation-required",
            },
            "missing adapter profile": {
                key: value
                for key, value in conditional.items()
                if key != "adapter_profile_id"
            },
            "null adapter profile": {
                **conditional,
                "adapter_profile_id": None,
            },
            "empty adapter profile": {
                **conditional,
                "adapter_profile_id": "",
            },
            "unknown adapter profile": {
                **conditional,
                "adapter_profile_id": "future-profile.v1",
            },
            "empty family": {**conditional, "family": ""},
            "blank family": {**conditional, "family": "   "},
            "padded family": {**conditional, "family": " qwen3_moe"},
            "empty reason": {**conditional, "reason": ""},
            "blank reason": {**conditional, "reason": "\t"},
            "padded reason": {
                **conditional,
                "reason": "A measured pilot remains required. ",
            },
            "unknown field": {**conditional, "unreviewed": True},
            "recognized runtime claim": {
                **recognized,
                "supported_runtime": "mlx-lm",
            },
            "recognized method claim": {
                **recognized,
                "supported_methods": ["lora"],
            },
            "recognized backend claim": {
                **recognized,
                "compute_backend": "mps",
            },
            "recognized distribution claim": {
                **recognized,
                "distribution": "single",
            },
            "recognized adapter claim": {
                **recognized,
                "adapter_profile_id": "attention-qkvo.v1",
            },
            "recognized implementation evidence": {
                **recognized,
                "evidence_requirement": "implementation-required",
            },
            "unsupported runtime claim": {
                **unsupported,
                "supported_runtime": "mlx-lm",
            },
            "unsupported method claim": {
                **unsupported,
                "supported_methods": ["qlora"],
            },
            "unsupported backend claim": {
                **unsupported,
                "compute_backend": "mps",
            },
            "unsupported distribution claim": {
                **unsupported,
                "distribution": "single",
            },
            "unsupported adapter claim": {
                **unsupported,
                "adapter_profile_id": "attention-qkvo.v1",
            },
            "unsupported pilot evidence": {
                **unsupported,
                "evidence_requirement": "pilot-required",
            },
        }

        for case, payload in invalid_variants.items():
            with self.subTest(case=case):
                with self.assertRaises(ValidationError):
                    ModelCompatibilityResponse.model_validate(payload)

    def test_plan_store_survives_context_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root)
            first = ApiContext(root / "state")
            first.save_plan(plan)
            restored = ApiContext(root / "state").load_plan(plan.plan_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.plan_id, plan.plan_id)
        self.assertEqual(
            restored.recommended.candidate_id, plan.recommended.candidate_id
        )

    def test_invalid_plan_id_does_not_escape_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = ApiContext(Path(temporary) / "state")
            self.assertIsNone(context.load_plan("../../plan_secret"))

    def test_saved_plan_id_must_match_requested_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = ApiContext(root / "state")
            plan = make_plan(root)
            substitute_id = "plan_" + "f" * 20
            substitute_path = context.plans_dir / f"{substitute_id}.json"
            atomic_write_json(substitute_path, to_primitive(plan), mode=0o600)
            before = substitute_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "requested filename"):
                context.load_plan(substitute_id)

            self.assertEqual(substitute_path.read_bytes(), before)

    def test_runtime_configuration_is_private_and_survives_context_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            interpreter_path = str((root / "mlx-python").resolve())
            probe = RuntimeInterpreter(
                path=interpreter_path,
                source="configured:APTUS_MLX_PYTHON",
                python_version="3.12.9",
                runtimes={"mlx-lm": {"available": True}},
            )
            context = ApiContext(root / "state")
            with patch(
                "aptus.api.validate_runtime_configuration",
                return_value=probe,
            ):
                result = context.configure_runtime("mlx-lm", Path(interpreter_path))
            configuration_path = context.runtime_config_path
            configuration_mode = stat.S_IMODE(configuration_path.stat().st_mode)
            restarted = ApiContext(root / "state")

        self.assertEqual(result["interpreter_path"], interpreter_path)
        self.assertEqual(configuration_mode, 0o600)
        self.assertEqual(restarted.runtime_paths["mlx-lm"], interpreter_path)
        self.assertEqual(
            restarted.jobs.runtime_environment["APTUS_MLX_PYTHON"],
            interpreter_path,
        )

    def test_server_extra_failure_is_explicit_when_fastapi_absent(self) -> None:
        try:
            import fastapi  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(RuntimeError, "server"):
                create_app()

    def test_invalid_explicit_workbench_directory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "must contain index.html"):
                _resolve_static_dir(Path(temporary))

    def test_packaged_workbench_is_discoverable(self) -> None:
        workbench = _resolve_static_dir(None)
        self.assertIsNotNone(workbench)
        self.assertTrue((workbench / "index.html").is_file())

    @unittest.skipIf(
        TestClient is None,
        "Install the server and test extras for endpoint integration tests.",
    )
    def test_desktop_session_token_is_required_and_constant_time_checked(self) -> None:
        token = "desktop-session-token-that-is-long-enough"
        with tempfile.TemporaryDirectory() as temporary:
            client = TestClient(
                create_app(
                    state_dir=Path(temporary) / "state",
                    session_token=token,
                )
            )
            try:
                public_health = client.get("/api/v1/health")
                rejected = client.get("/api/v1/bootstrap")
                client.cookies.set("aptus_desktop_session", token)
                accepted = client.get("/api/v1/bootstrap")
            finally:
                client.close()
        self.assertEqual(public_health.status_code, 200)
        self.assertEqual(rejected.status_code, 403)
        self.assertEqual(rejected.json()["error"], "desktop_session_required")
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json()["version"], "0.2.0")
        self.assertEqual(accepted.headers["x-content-type-options"], "nosniff")
        self.assertIn("default-src 'self'", accepted.headers["content-security-policy"])
        self.assertEqual(rejected.headers["x-frame-options"], "DENY")

    @unittest.skipIf(
        TestClient is None,
        "Install the server and test extras for endpoint integration tests.",
    )
    def test_server_session_partitions_public_static_from_protected_api(self) -> None:
        token = "server-session-token-that-is-long-enough"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            static = root / "web"
            static.mkdir()
            (static / "index.html").write_text("Aptus static", encoding="utf-8")
            client = TestClient(
                create_app(
                    state_dir=root / "state",
                    static_dir=static,
                    session_token=token,
                )
            )
            try:
                static_response = client.get("/")
                platform_response = client.get("/api/v1/platform")
                runtimes_response = client.get("/api/v1/runtimes")
                mutation_response = client.post(
                    "/api/v1/jobs",
                    json={"bundle_dir": "/tmp/bundle", "action": "pilot"},
                )
                bearer_response = client.get(
                    "/api/v1/bootstrap",
                    headers={"Authorization": f"Bearer {token}"},
                )
                exchange_response = client.get(
                    f"/?aptus_session_token={token}",
                    follow_redirects=False,
                )
                cookie_response = client.get("/api/v1/bootstrap")
            finally:
                client.close()

        self.assertEqual(static_response.status_code, 200)
        self.assertIn("Aptus static", static_response.text)
        for response in (platform_response, runtimes_response, mutation_response):
            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["error"], "desktop_session_required")
        self.assertEqual(bearer_response.status_code, 200)
        self.assertEqual(exchange_response.status_code, 303)
        self.assertEqual(exchange_response.headers["location"], "/")
        self.assertIn("HttpOnly", exchange_response.headers["set-cookie"])
        self.assertIn("SameSite=strict", exchange_response.headers["set-cookie"])
        self.assertEqual(cookie_response.status_code, 200)

    def test_create_app_requires_session_token_unless_tests_opt_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "state"
            with self.assertRaisesRegex(ValueError, "session_token"):
                create_app(state_dir=state_dir)
            opted_out = create_app(
                state_dir=state_dir,
                allow_unauthenticated=True,
            )
            self.assertIsNotNone(opted_out)

    def test_desktop_session_token_rejects_short_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 32"):
            create_app(session_token="too-short")

    @unittest.skipIf(
        TestClient is None,
        "Install the server and test extras for endpoint integration tests.",
    )
    def test_desktop_server_enforces_the_no_execution_boundary(self) -> None:
        token = "desktop-session-token-that-is-long-enough"
        with tempfile.TemporaryDirectory() as temporary:
            client = TestClient(
                create_app(
                    state_dir=Path(temporary) / "state",
                    session_token=token,
                    execution_enabled=False,
                )
            )
            client.cookies.set("aptus_desktop_session", token)
            try:
                job_response = client.post(
                    "/api/v1/jobs",
                    json={
                        "bundle_dir": "/tmp/untrusted-bundle",
                        "project_id": "project_" + "a" * 32,
                        "expected_project_revision_id": "revision_" + "b" * 32,
                        "action": "train",
                    },
                )
                validation_response = client.post(
                    "/api/v1/validate",
                    json={
                        "bundle_dir": "/tmp/untrusted-bundle",
                        "project_id": "project_" + "a" * 32,
                        "expected_project_revision_id": "revision_" + "b" * 32,
                        "level": "dependency",
                        "run": True,
                    },
                )
            finally:
                client.close()

        self.assertEqual(job_response.status_code, 403)
        self.assertEqual(
            job_response.json()["error"],
            "desktop_execution_disabled",
        )
        self.assertEqual(validation_response.status_code, 403)
        self.assertEqual(
            validation_response.json()["error"],
            "desktop_execution_disabled",
        )


@unittest.skipIf(
    TestClient is None,
    "Install the server and test extras for endpoint integration tests.",
)
class ApiEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "data.jsonl"
        self.dataset.write_text(
            '{"prompt":"Question?","completion":"Answer."}\n', encoding="utf-8"
        )
        static = self.root / "web"
        static.mkdir()
        (static / "index.html").write_text(
            "<html><body>Aptus workbench</body></html>", encoding="utf-8"
        )
        self.client = TestClient(
            create_app(
                state_dir=self.root / "state",
                static_dir=static,
                allow_unauthenticated=True,
            )
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def plan_payload(self) -> dict[str, object]:
        return {
            "model": {
                "model_id": "example/model-1b",
                "revision": "a" * 40,
                "family": "llama",
                "parameters_b": 1,
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "layers": 24,
                "context_length": 4096,
                "license_name": "apache-2.0",
                "training_allowed": True,
            },
            "hardware": {
                "discovery": "manual",
                "backend": "cuda",
                "gpu_count": 1,
                "vram_gib": 24,
                "free_vram_gib": 22,
                "supports_bf16": True,
                "supports_8bit": True,
                "supports_4bit": True,
                "host_ram_gib": 64,
                "host_ram_free_gib": 56,
                "reserve_gib": 2,
                "disk_free_gib": 500,
            },
            "target": {
                "objective": "memory",
                "sequence_length": 128,
                "effective_batch_size": 8,
                "max_epochs": 1,
                "task": "sft",
                "evaluation_fraction": 0.1,
                "packing": False,
                "checkpoint_steps": 10,
            },
            "dataset_path": str(self.dataset),
            "sample_limit": 64,
        }

    def inspection_receipt(self, payload: dict[str, object]) -> dict[str, object]:
        model_payload = payload["model"]
        assert isinstance(model_payload, dict)
        model = build_model_spec(**model_payload)
        observed_at = "2026-07-29T12:00:00+00:00"
        facts = {
            field: getattr(model, field)
            for field in (
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
        }
        provenance = {
            field: {
                "kind": "inferred" if field == "family" else "provider-declared",
                "source": (
                    "Aptus exact model-type compatibility mapping"
                    if field == "family"
                    else "https://huggingface.co/example/model-1b/config.json"
                ),
                "observed_at": observed_at,
                "resolved_revision": model.revision,
            }
            for field, value in facts.items()
            if value is not None
        }
        return to_primitive(
            create_model_inspection_receipt(
                model_id=model.model_id,
                resolved_revision=model.revision,
                facts=facts,
                provenance=provenance,
                subject=subject_from_model(model),
                evaluated_at=observed_at,
            )
        )

    def seed_job(
        self,
        *,
        job_id: str,
        action: str = "train",
        state: str = "completed",
        plan_id: str = "plan_abc",
        candidate_id: str = "cand_abc",
        run_id: str | None = "run_abc",
    ) -> str:
        jobs = self.client.app.state.aptus.jobs
        bundle = self.root / f"bundle-{job_id}"
        bundle.mkdir(exist_ok=True)
        (bundle / "validation-report.json").write_text(
            json.dumps(
                {
                    "schema_version": "aptus.validation.v2",
                    "state": "measured-run-pass",
                }
            ),
            encoding="utf-8",
        )
        record = {
            "schema_version": "aptus.job-record.v1",
            "id": job_id,
            "job_id": job_id,
            "state": state,
            "created_at": "2026-01-01T00:00:00+00:00",
            "action": action,
            "bundle_dir": str(bundle),
            "plan_id": plan_id,
            "candidate_id": candidate_id,
            "run_id": run_id,
        }
        (jobs.root / f"{job_id}.json").write_text(json.dumps(record), encoding="utf-8")
        return job_id

    def test_plan_accepts_bound_inspection_receipt_and_marks_omission_attested(
        self,
    ) -> None:
        attested = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(attested.status_code, 200, attested.text)
        self.assertEqual(
            attested.json()["model_policy_decision_source"], "user-attested"
        )
        self.assertIsNone(attested.json()["inspection_receipt"])

        payload = self.plan_payload()
        receipt = self.inspection_receipt(payload)
        payload["inspection_receipt"] = receipt
        inspected = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(inspected.status_code, 200, inspected.text)
        result = inspected.json()
        self.assertEqual(result["model_policy_decision_source"], "provider-inspection")
        self.assertEqual(
            result["inspection_receipt"]["receipt_id"], receipt["receipt_id"]
        )
        self.assertTrue(
            all(
                candidate["model_policy_decision_id"]
                == result["model_policy_decision"]["decision_id"]
                for candidate in result["candidates"]
            )
        )

    def test_candidate_selection_creates_new_plan_and_rejects_stale_revision(
        self,
    ) -> None:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(planned.status_code, 200, planned.text)
        source = planned.json()
        alternative = next(
            item
            for item in source["candidates"]
            if item["feasible"]
            and item["candidate_id"] != source["recommended"]["candidate_id"]
        )
        request = {
            "plan_id": source["plan_id"],
            "candidate_id": alternative["candidate_id"],
            "project_id": source["project_id"],
            "expected_project_revision_id": source["project_revision_id"],
        }

        selected = self.client.post("/api/v1/plans/select", json=request)

        self.assertEqual(selected.status_code, 200, selected.text)
        result = selected.json()
        self.assertEqual(
            result["recommended"]["candidate_id"], alternative["candidate_id"]
        )
        self.assertNotEqual(result["plan_id"], source["plan_id"])
        self.assertNotEqual(
            result["project_revision_id"], source["project_revision_id"]
        )
        stale = self.client.post("/api/v1/plans/select", json=request)
        self.assertEqual(stale.status_code, 409, stale.text)
        self.assertEqual(stale.json()["error"], "project_revision_conflict")

    def test_unknown_family_plan_saves_and_reloads_without_reinterpretation(
        self,
    ) -> None:
        payload = self.plan_payload()
        payload["model"]["family"] = "unregistered-family"

        planned = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(planned.status_code, 200, planned.text)
        value = planned.json()
        self.assertEqual(value["model_policy_decision"]["kind"], "unknown")
        self.assertTrue(
            all(
                candidate["checkpoint_retention_bytes"] == 0
                for candidate in value["candidates"]
                if candidate["method"] != "full"
            )
        )

        loaded = self.client.get(f"/api/v1/plans/{value['plan_id']}")

        self.assertEqual(loaded.status_code, 200, loaded.text)
        self.assertEqual(loaded.json()["plan_id"], value["plan_id"])

    def test_plan_rejects_malformed_and_tampered_inspection_receipts(self) -> None:
        malformed = self.plan_payload()
        malformed["inspection_receipt"] = {"schema_version": "wrong"}
        malformed_response = self.client.post("/api/v1/plan", json=malformed)
        self.assertEqual(malformed_response.status_code, 422, malformed_response.text)

        for label, mutate in (
            (
                "digest",
                lambda receipt: receipt.__setitem__("observed_facts_sha256", "0" * 64),
            ),
            (
                "model identity",
                lambda receipt: receipt.__setitem__("model_id", "example/other"),
            ),
        ):
            with self.subTest(label=label):
                payload = self.plan_payload()
                receipt = self.inspection_receipt(payload)
                mutate(receipt)
                payload["inspection_receipt"] = receipt
                response = self.client.post("/api/v1/plan", json=payload)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(response.json()["error"], "invalid_request")

    def test_plan_request_validation_serializes_model_validator_errors(self) -> None:
        payload = self.plan_payload()
        receipt = inspection_receipt_shape("example/model-1b", "a" * 40)
        receipt["provenance_summary"][0]["kind"] = "inferred"
        payload["inspection_receipt"] = receipt

        response = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(response.status_code, 422, response.text)
        failure = ErrorResponse.model_validate(response.json())
        self.assertEqual(failure.error, "request_validation")
        self.assertTrue(
            any("provider-declared" in item.get("msg", "") for item in failure.details)
        )

    def test_every_pre_v5_saved_plan_schema_requires_replanning_without_rewrite(
        self,
    ) -> None:
        context = self.client.app.state.aptus
        for index, found_schema in enumerate(
            (
                "aptus.training-plan.v4",
                "aptus.training-plan.v3",
                "aptus.training-plan.v2",
                None,
            )
        ):
            with self.subTest(found_schema=found_schema):
                plan_id = "plan_" + format(index + 1, "020x")
                payload = {"plan_id": plan_id}
                if found_schema is not None:
                    payload["schema_version"] = found_schema
                path = context.plans_dir / f"{plan_id}.json"
                atomic_write_json(path, payload, mode=0o600)
                before = path.read_bytes()

                response = self.client.get(f"/api/v1/plans/{plan_id}")

                self.assertEqual(response.status_code, 409, response.text)
                self.assertEqual(response.json()["error"], "replan_required")
                self.assertEqual(
                    response.json()["required_schema"], "aptus.training-plan.v6"
                )
                self.assertEqual(response.json()["found_schema"], found_schema)
                self.assertEqual(path.read_bytes(), before)

    def test_saved_plan_json_parser_resource_errors_map_to_invalid_request(
        self,
    ) -> None:
        invalid_documents = (
            ("oversized-integer", '{"value":' + "9" * 5000 + "}\n"),
            (
                "excessive-nesting",
                '{"value":' + "[" * 10000 + "0" + "]" * 10000 + "}\n",
            ),
        )
        context = self.client.app.state.aptus
        for index, (name, contents) in enumerate(invalid_documents):
            with self.subTest(name=name):
                plan_id = "plan_" + format(index + 1, "020x")
                (context.plans_dir / f"{plan_id}.json").write_text(
                    contents, encoding="utf-8"
                )

                response = self.client.get(f"/api/v1/plans/{plan_id}")

                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(response.json()["error"], "invalid_request")
                self.assertEqual(
                    response.json()["details"],
                    "Saved plan is unreadable or invalid JSON.",
                )

    def test_stale_same_schema_policy_maps_to_replan_required(self) -> None:
        plan_id = "plan_" + "e" * 20
        context = self.client.app.state.aptus
        with patch.object(
            context,
            "load_plan",
            side_effect=StaleModelPolicyError("Registered policy is obsolete."),
        ):
            response = self.client.get(f"/api/v1/plans/{plan_id}")

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "replan_required")
        self.assertEqual(response.json()["found_schema"], "aptus.training-plan.v6")
        self.assertEqual(response.json()["required_schema"], "aptus.training-plan.v6")

    def owned_bundle_request(self, bundle: Path) -> dict[str, str]:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(planned.status_code, 200, planned.text)
        plan = planned.json()
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan["plan_id"],
                "output_dir": str(bundle),
                "project_id": plan["project_id"],
                "expected_project_revision_id": plan["project_revision_id"],
            },
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)
        return {
            "project_id": plan["project_id"],
            "expected_project_revision_id": compiled.json()["project_revision_id"],
        }

    def test_health_spa_plan_compile_and_static_validation(self) -> None:
        self.assertEqual(self.client.get("/api/v1/health").json()["status"], "ok")
        self.assertIn("Aptus workbench", self.client.get("/").text)
        capabilities = self.client.get("/api/v1/bootstrap").json()["capabilities"]
        self.assertEqual(capabilities["backends"], ["cuda", "mps"])
        self.assertEqual(capabilities["supported_execution_backends"], ["cuda", "mps"])
        self.assertEqual(
            capabilities["supported_execution_backend"],
            "mps" if sys.platform == "darwin" else "cuda",
        )
        defaults = self.client.get("/api/v1/bootstrap").json()["defaults"]
        self.assertEqual(
            defaults["backend"], capabilities["supported_execution_backend"]
        )
        self.assertEqual(
            defaults["training_runtime"],
            "mlx-lm" if sys.platform == "darwin" else "transformers-peft-cuda",
        )
        self.assertEqual(
            defaults["reserve_gib"], 8.0 if sys.platform == "darwin" else 2.0
        )
        self.assertEqual(
            capabilities["training_runtimes"],
            ["transformers-peft-cuda", "mlx-lm"],
        )
        self.assertEqual(
            set(capabilities["known_backends"]), {"cuda", "rocm", "mps", "cpu"}
        )
        self.assertEqual(
            set(capabilities["methods"]), {"full", "lora", "int8-lora", "qlora"}
        )
        method_catalog = {
            item["method_id"]: item for item in capabilities["method_catalog"]
        }
        self.assertEqual(method_catalog["lora"]["lifecycle"], "gated-executable")
        self.assertTrue(method_catalog["lora"]["selectable"])
        self.assertEqual(method_catalog["dora"]["lifecycle"], "experimental")
        self.assertFalse(method_catalog["dora"]["selectable"])
        self.assertEqual(method_catalog["bitfit"]["lifecycle"], "experimental")
        self.assertEqual(method_catalog["loreft"]["lifecycle"], "research-only")

        planned = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(planned.status_code, 200, planned.text)
        planned_payload = planned.json()
        plan_id = planned_payload["plan_id"]
        project_id = planned_payload["project_id"]
        project_revision_id = planned_payload["project_revision_id"]

        bundle = self.root / "bundle"
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan_id,
                "output_dir": str(bundle),
                "project_id": project_id,
                "expected_project_revision_id": project_revision_id,
            },
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)
        self.assertTrue(compiled.json()["archive_path"].endswith("bundle.zip"))
        compiled_revision_id = compiled.json()["project_revision_id"]

        validated = self.client.post(
            "/api/v1/validate",
            json={
                "bundle_dir": str(bundle),
                "project_id": project_id,
                "expected_project_revision_id": compiled_revision_id,
                "level": "static",
                "run": False,
            },
        )
        self.assertEqual(validated.status_code, 200, validated.text)
        self.assertEqual(validated.json()["state"], "static-pass")
        self.assertNotIn("authorization_status", validated.json())
        self.assertNotIn("authorization_current", validated.json())
        validated_revision_id = validated.json()["project_revision_id"]

        restored = self.client.get("/api/v1/bootstrap")
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertEqual(restored.json()["plan"]["plan_id"], plan_id)
        self.assertEqual(restored.json()["bundle"]["bundle_dir"], str(bundle.resolve()))
        self.assertEqual(restored.json()["bundle"]["report"]["state"], "static-pass")

        runtime = self.client.post(
            "/api/v1/validate",
            json={
                "bundle_dir": str(bundle),
                "project_id": project_id,
                "expected_project_revision_id": validated_revision_id,
                "level": "pilot",
                "run": True,
            },
        )
        self.assertEqual(runtime.status_code, 409)
        self.assertEqual(runtime.json()["suggested_action"], "pilot")

        conflict = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan_id,
                "output_dir": str(bundle),
                "project_id": project_id,
                "expected_project_revision_id": validated_revision_id,
            },
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["error"], "path_conflict")

    def test_bootstrap_marks_unchecked_pilot_authorization_deferred(self) -> None:
        bundle = self.root / "bootstrap-authorization"
        self.owned_bundle_request(bundle)
        (bundle / "validation-report.json").write_text(
            json.dumps({"state": "pilot-pass", "bindings": {}}),
            encoding="utf-8",
        )

        response = self.client.get("/api/v1/bootstrap")

        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()["bundle"]["report"]
        self.assertEqual(report["authorization_status"], "deferred")
        self.assertFalse(report["authorization_current"])
        self.assertIn("full training is submitted", report["authorization_error"])

    def test_exact_qwen3_moe_plan_preserves_topology_and_derived_facts(self) -> None:
        payload = self.plan_payload()
        payload["model"] = {
            **payload["model"],
            "model_id": "Qwen/Qwen3-30B-A3B-MLX-4bit",
            "family": "qwen3_moe",
            "parameters_b": 30.5,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "layers": 48,
            "context_length": 40960,
            "model_type": "qwen3_moe",
            "architecture": "Qwen3MoeForCausalLM",
            "quantization_bits": 4,
            "quantization_layout": to_primitive(
                reviewed_qwen3_moe_quantization_layout(48)
            ),
            "moe": {
                "expert_count": 128,
                "experts_per_token": 8,
                "expert_intermediate_size": 768,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
            },
        }
        payload["hardware"] = {
            **payload["hardware"],
            "backend": "mps",
            "gpu_count": 1,
            "vram_gib": 64,
            "free_vram_gib": 48,
            "supports_bf16": False,
            "supports_8bit": False,
            "supports_4bit": False,
            "host_ram_gib": 64,
            "host_ram_free_gib": 48,
            "reserve_gib": 8,
        }
        payload["target"] = {
            **payload["target"],
            "method_preference": "qlora",
            "training_runtime": "mlx-lm",
        }

        response = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        plan = response.json()
        self.assertEqual(plan["schema_version"], "aptus.training-plan.v6")
        self.assertEqual(plan["model"]["model_type"], "qwen3_moe")
        self.assertEqual(
            len(plan["model"]["quantization_layout"]["module_overrides"]), 48
        )
        self.assertEqual(plan["model"]["moe"]["expert_count"], 128)
        self.assertEqual(plan["model"]["sparse_layer_count"], 48)
        self.assertLess(plan["model"]["active_parameters"], plan["model"]["parameters"])
        self.assertEqual(plan["recommended"]["method"], "qlora")
        self.assertEqual(
            plan["recommended"]["runtime_contract"]["training_runtime"],
            "mlx-lm",
        )

    def test_moe_integer_facts_reject_boolean_coercion(self) -> None:
        payload = self.plan_payload()
        payload["model"] = {
            **payload["model"],
            "quantization_bits": True,
            "quantization_layout": {
                "default_bits": True,
                "default_group_size": 64,
                "module_overrides": [],
            },
            "moe": {
                "expert_count": True,
                "experts_per_token": 1,
                "expert_intermediate_size": 768,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
            },
        }

        response = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(response.status_code, 422, response.text)
        locations = {tuple(item["loc"]) for item in response.json()["details"]}
        self.assertIn(("body", "model", "quantization_bits"), locations)
        self.assertIn(
            ("body", "model", "quantization_layout", "default_bits"), locations
        )
        self.assertIn(("body", "model", "moe", "expert_count"), locations)

    def test_model_inspection_contract_exposes_exact_moe_compatibility(self) -> None:
        inspection = {
            "status": "ok",
            "model_id": "Qwen/Qwen3-30B-A3B-MLX-4bit",
            "requested_revision": "main",
            "resolved_revision": "d" * 40,
            "facts": {
                "architecture": "Qwen3MoeForCausalLM",
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "family": "qwen3_moe",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "layers": 48,
                "context_length": 40960,
                "attention_heads": 32,
                "key_value_heads": 4,
                "vocab_size": 151936,
                "quantization_bits": 4,
                "quantization_layout": to_primitive(
                    reviewed_qwen3_moe_quantization_layout(48)
                ),
                "moe": {
                    "expert_count": 128,
                    "experts_per_token": 8,
                    "expert_intermediate_size": 768,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                    "shared_expert_intermediate_size": None,
                },
                "license_name": "apache-2.0",
                "parameters": None,
                "training_allowed": None,
            },
            "provenance": {},
            "inspection_receipt": inspection_receipt_shape(
                "Qwen/Qwen3-30B-A3B-MLX-4bit", "d" * 40
            ),
            "warnings": [],
            "compatibility": {
                "status": "conditional",
                "family": "qwen3_moe",
                "supported_runtime": "mlx-lm",
                "supported_methods": ["qlora"],
                "compute_backend": "mps",
                "distribution": "single",
                "evidence_requirement": "pilot-required",
                "adapter_profile_id": "attention-qkvo.v1",
                "reason": "Exact model-data and pilot evidence are required.",
            },
            "explicit_user_facts_required": ["parameters", "training_allowed"],
        }
        static = self.root / "inspection-web"
        static.mkdir()
        (static / "index.html").write_text("Aptus", encoding="utf-8")
        with patch(
            "aptus.inspection.inspect_huggingface_model",
            return_value=inspection,
        ):
            client = TestClient(
                create_app(
                    state_dir=self.root / "inspection-state",
                    static_dir=static,
                    allow_unauthenticated=True,
                )
            )
        try:
            response = client.post(
                "/api/v1/models/inspect",
                json={
                    "model_id": "Qwen/Qwen3-30B-A3B-MLX-4bit",
                    "revision": "main",
                },
            )
        finally:
            client.close()

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["facts"]["moe"]["experts_per_token"], 8)
        self.assertEqual(
            len(result["facts"]["quantization_layout"]["module_overrides"]), 48
        )
        self.assertEqual(result["compatibility"]["status"], "conditional")
        self.assertEqual(result["inspection_receipt"]["resolved_revision"], "d" * 40)
        self.assertEqual(result["compatibility"]["compute_backend"], "mps")
        self.assertEqual(
            result["compatibility"]["adapter_profile_id"],
            "attention-qkvo.v1",
        )

    def test_model_inspection_response_rejects_malformed_compatibility(self) -> None:
        inspection = {
            "status": "ok",
            "model_id": "provider/malformed",
            "requested_revision": "main",
            "compatibility": {
                "status": "conditional",
                "family": "qwen3_moe",
                "supported_runtime": None,
                "supported_methods": ["qlora"],
                "compute_backend": "mps",
                "distribution": "single",
                "evidence_requirement": "pilot-required",
                "adapter_profile_id": "attention-qkvo.v1",
                "reason": "The producer omitted the executable runtime.",
            },
        }
        with patch(
            "aptus.inspection.inspect_huggingface_model",
            return_value=inspection,
        ):
            client = TestClient(
                create_app(
                    state_dir=self.root / "malformed-inspection-state",
                    static_dir=self.root / "web",
                    allow_unauthenticated=True,
                ),
                raise_server_exceptions=False,
            )
        try:
            response = client.post(
                "/api/v1/models/inspect",
                json={"model_id": "provider/malformed", "revision": "main"},
            )
        finally:
            client.close()

        self.assertEqual(response.status_code, 500, response.text)

    def test_model_inspection_response_rejects_inferred_only_receipt(self) -> None:
        receipt = inspection_receipt_shape("provider/inferred-only", "a" * 40)
        receipt["provenance_summary"][0]["kind"] = "inferred"
        inspection = {
            "status": "ok",
            "model_id": "provider/inferred-only",
            "requested_revision": "main",
            "resolved_revision": "a" * 40,
            "facts": {},
            "compatibility": {
                "status": "unsupported",
                "family": None,
                "supported_runtime": None,
                "supported_methods": [],
                "compute_backend": None,
                "distribution": None,
                "evidence_requirement": "implementation-required",
                "adapter_profile_id": None,
                "reason": "No reviewed policy matches this model.",
            },
            "provenance": {},
            "inspection_receipt": receipt,
        }
        with patch(
            "aptus.inspection.inspect_huggingface_model",
            return_value=inspection,
        ):
            client = TestClient(
                create_app(
                    state_dir=self.root / "inferred-receipt-state",
                    static_dir=self.root / "web",
                    allow_unauthenticated=True,
                ),
                raise_server_exceptions=False,
            )
        try:
            response = client.post(
                "/api/v1/models/inspect",
                json={"model_id": "provider/inferred-only", "revision": "main"},
            )
        finally:
            client.close()

        self.assertEqual(response.status_code, 500, response.text)

    def test_mutating_workflow_requests_require_exact_project_identity(self) -> None:
        for path, payload in (
            ("/api/v1/compile", {"plan_id": "plan_test", "output_dir": "bundle"}),
            (
                "/api/v1/validate",
                {"bundle_dir": "bundle", "level": "static", "run": False},
            ),
            ("/api/v1/jobs", {"bundle_dir": "bundle", "action": "preflight"}),
        ):
            with self.subTest(path=path):
                response = self.client.post(path, json=payload)
                self.assertEqual(response.status_code, 422, response.text)

    def test_compile_rejects_stale_or_cross_project_plan_identity(self) -> None:
        first = self.client.post("/api/v1/plan", json=self.plan_payload()).json()
        second_payload = self.plan_payload()
        second_payload["model"] = {
            **second_payload["model"],
            "revision": "b" * 40,
        }
        second = self.client.post("/api/v1/plan", json=second_payload).json()
        cross_project = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": first["plan_id"],
                "output_dir": str(self.root / "cross-project"),
                "project_id": second["project_id"],
                "expected_project_revision_id": second["project_revision_id"],
            },
        )
        self.assertEqual(cross_project.status_code, 409, cross_project.text)
        self.assertEqual(cross_project.json()["error"], "project_plan_mismatch")

        payload = self.plan_payload()
        payload["project_id"] = first["project_id"]
        latest = self.client.post("/api/v1/plan", json=payload)
        self.assertEqual(latest.status_code, 200, latest.text)
        stale = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": first["plan_id"],
                "output_dir": str(self.root / "stale"),
                "project_id": first["project_id"],
                "expected_project_revision_id": first["project_revision_id"],
            },
        )
        self.assertEqual(stale.status_code, 409, stale.text)
        self.assertEqual(stale.json()["error"], "project_revision_conflict")

    def test_compile_race_never_publishes_uncommitted_legacy_bundle_pointer(
        self,
    ) -> None:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload()).json()
        projects = self.client.app.state.aptus.projects
        create_revision = projects.create_revision

        def race_project_revision(
            project_id: str, **changes: object
        ) -> dict[str, object]:
            create_revision(
                project_id,
                reason="concurrent-update",
                base_revision_id=planned["project_revision_id"],
                expected_latest_revision_id=planned["project_revision_id"],
            )
            return create_revision(project_id, **changes)

        with patch.object(
            projects,
            "create_revision",
            side_effect=race_project_revision,
        ):
            response = self.client.post(
                "/api/v1/compile",
                json={
                    "plan_id": planned["plan_id"],
                    "output_dir": str(self.root / "uncommitted-bundle"),
                    "project_id": planned["project_id"],
                    "expected_project_revision_id": planned["project_revision_id"],
                },
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "project_revision_conflict")
        self.assertFalse(self.client.app.state.aptus.current_bundle_path.exists())

    def test_named_project_history_and_recovery_are_explicit_contracts(self) -> None:
        created = self.client.post(
            "/api/v1/projects", json={"name": "Parish corpus adapter"}
        )
        self.assertEqual(created.status_code, 201, created.text)
        project_id = created.json()["project_id"]

        payload = self.plan_payload()
        payload["project_id"] = project_id
        payload["project_name"] = "Parish corpus adapter"
        planned = self.client.post("/api/v1/plan", json=payload)
        self.assertEqual(planned.status_code, 200, planned.text)
        self.assertEqual(planned.json()["project_id"], project_id)
        revision_id = planned.json()["project_revision_id"]

        history = self.client.get(f"/api/v1/projects/{project_id}/revisions")
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(history.json()[0]["reason"], "plan-created")
        revision = self.client.get(
            f"/api/v1/projects/{project_id}/revisions/{revision_id}"
        )
        self.assertEqual(revision.status_code, 200, revision.text)
        self.assertFalse(revision.json()["training_authorization"]["current"])

        recovered = self.client.post(
            f"/api/v1/projects/{project_id}/recover",
            json={"revision_id": revision_id},
        )
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertFalse(recovered.json()["training_authorization_current"])
        self.assertNotEqual(recovered.json()["revision"]["revision_id"], revision_id)

        bootstrap = self.client.get("/api/v1/bootstrap")
        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertEqual(bootstrap.json()["project"]["project_id"], project_id)
        self.assertEqual(bootstrap.json()["project_history"][0]["ordinal"], 2)

    def test_legacy_plan_is_preserved_and_requires_explicit_replan(self) -> None:
        context = self.client.app.state.aptus
        plan_id = "plan_" + "c" * 20
        legacy_plan = {
            "schema_version": "aptus.training-plan.v2",
            "plan_id": plan_id,
            "recommended": {"candidate_id": "candidate_legacy"},
        }
        atomic_write_json(
            context.plans_dir / f"{plan_id}.json", legacy_plan, mode=0o600
        )
        project = context.projects.create("Legacy saved plan")
        revision = context.projects.create_revision(
            project["project_id"],
            reason="legacy-plan-imported",
            plan_id=plan_id,
            plan_snapshot=legacy_plan,
            selected_candidate_id="candidate_legacy",
        )
        saved_plan_path = context.plans_dir / f"{plan_id}.json"
        before = saved_plan_path.read_bytes()

        bootstrap = self.client.get("/api/v1/bootstrap")
        loaded = self.client.get(f"/api/v1/plans/{plan_id}")
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan_id,
                "output_dir": str(self.root / "legacy-output"),
                "project_id": project["project_id"],
                "expected_project_revision_id": revision["revision_id"],
            },
        )
        recovered = self.client.post(
            f"/api/v1/projects/{project['project_id']}/recover",
            json={"revision_id": revision["revision_id"]},
        )

        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertIsNone(bootstrap.json().get("plan"))
        self.assertEqual(
            bootstrap.json()["replan_required"],
            {
                "status": "replan_required",
                "plan_id": plan_id,
                "found_schema": "aptus.training-plan.v2",
                "required_schema": "aptus.training-plan.v6",
                "source": "project-revision",
                "project_id": project["project_id"],
                "project_revision_id": revision["revision_id"],
                "message": (
                    "This saved plan predates the current executable contract. "
                    "Create a new plan from its preserved facts before compiling "
                    "or recovering it."
                ),
            },
        )
        for response in (loaded, compiled, recovered):
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()["error"], "replan_required")
            self.assertEqual(
                response.json()["required_schema"], "aptus.training-plan.v6"
            )
        self.assertEqual(saved_plan_path.read_bytes(), before)
        self.assertFalse((self.root / "legacy-output").exists())
        self.assertEqual(
            context.projects.get(project["project_id"])["revision_count"], 1
        )

    def test_coherent_stale_v5_plan_requires_replan_across_saved_workflows(
        self,
    ) -> None:
        context = self.client.app.state.aptus
        stale_plan = to_primitive(make_qwen3_moe_plan(self.root))
        decision = stale_plan["model_policy_decision"]
        recommended_key = (
            stale_plan["recommended"]["method"],
            stale_plan["recommended"]["distribution"],
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
        for candidate in stale_plan["candidates"]:
            candidate["model_policy_decision_id"] = decision["decision_id"]
            binding = candidate["policy_binding"]
            if binding is not None:
                binding["decision_id"] = decision["decision_id"]
                binding["policy_version"] = decision["policy_version"]
            candidate["candidate_id"] = candidate_id_for_payload(
                candidate,
                model=stale_plan["model"],
                dataset=stale_plan["dataset"],
                hardware=stale_plan["hardware"],
                target=stale_plan["target"],
            )
        stale_plan["recommended"] = copy.deepcopy(
            next(
                candidate
                for candidate in stale_plan["candidates"]
                if (candidate["method"], candidate["distribution"]) == recommended_key
            )
        )
        stale_plan["plan_id"] = plan_id_for_payload(stale_plan)
        plan_id = stale_plan["plan_id"]

        saved_plan_path = context.plans_dir / f"{plan_id}.json"
        atomic_write_json(saved_plan_path, stale_plan, mode=0o600)
        project = context.projects.create("Stale policy plan")
        revision = context.projects.create_revision(
            project["project_id"],
            reason="stale-v5-imported",
            plan_id=plan_id,
            plan_snapshot=stale_plan,
            selected_candidate_id=stale_plan["recommended"]["candidate_id"],
        )
        before = saved_plan_path.read_bytes()

        bootstrap = self.client.get("/api/v1/bootstrap")
        loaded = self.client.get(f"/api/v1/plans/{plan_id}")
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan_id,
                "output_dir": str(self.root / "stale-v5-output"),
                "project_id": project["project_id"],
                "expected_project_revision_id": revision["revision_id"],
            },
        )
        recovered = self.client.post(
            f"/api/v1/projects/{project['project_id']}/recover",
            json={"revision_id": revision["revision_id"]},
        )

        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertIsNone(bootstrap.json().get("plan"))
        self.assertEqual(
            bootstrap.json()["replan_required"]["status"], "replan_required"
        )
        self.assertEqual(
            bootstrap.json()["replan_required"]["found_schema"],
            "aptus.training-plan.v6",
        )
        for response in (loaded, compiled, recovered):
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()["error"], "replan_required")
            self.assertEqual(
                response.json()["required_schema"], "aptus.training-plan.v6"
            )
        self.assertEqual(saved_plan_path.read_bytes(), before)
        self.assertFalse((self.root / "stale-v5-output").exists())
        self.assertEqual(
            context.projects.get(project["project_id"])["revision_count"], 1
        )

    def test_plan_only_recovery_ignores_newer_legacy_bundle_pointer(self) -> None:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(planned.status_code, 200, planned.text)
        plan = planned.json()
        bundle = self.root / "newer-global-bundle"
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan["plan_id"],
                "output_dir": str(bundle),
                "project_id": plan["project_id"],
                "expected_project_revision_id": plan["project_revision_id"],
            },
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)
        recovered = self.client.post(
            f"/api/v1/projects/{plan['project_id']}/recover",
            json={"revision_id": plan["project_revision_id"]},
        )
        self.assertEqual(recovered.status_code, 200, recovered.text)

        bootstrap = self.client.get("/api/v1/bootstrap")

        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertEqual(bootstrap.json()["plan"]["plan_id"], plan["plan_id"])
        self.assertEqual(bootstrap.json()["plan"]["project_id"], plan["project_id"])
        self.assertEqual(
            bootstrap.json()["plan"]["project_revision_id"],
            recovered.json()["revision"]["revision_id"],
        )
        self.assertIsNone(bootstrap.json().get("bundle"))

    def test_bootstrap_rejects_valid_bundle_replaced_by_another_identity(self) -> None:
        first = self.client.post("/api/v1/plan", json=self.plan_payload()).json()
        first_bundle = self.root / "first-bundle"
        first_compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": first["plan_id"],
                "output_dir": str(first_bundle),
                "project_id": first["project_id"],
                "expected_project_revision_id": first["project_revision_id"],
            },
        ).json()

        second_payload = self.plan_payload()
        second_payload["model"] = {
            **second_payload["model"],
            "revision": "b" * 40,
        }
        second = self.client.post("/api/v1/plan", json=second_payload).json()
        second_bundle = self.root / "second-bundle"
        second_compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": second["plan_id"],
                "output_dir": str(second_bundle),
                "project_id": second["project_id"],
                "expected_project_revision_id": second["project_revision_id"],
            },
        )
        self.assertEqual(second_compiled.status_code, 200, second_compiled.text)
        recovered = self.client.post(
            f"/api/v1/projects/{first['project_id']}/recover",
            json={"revision_id": first_compiled["project_revision_id"]},
        )
        self.assertEqual(recovered.status_code, 200, recovered.text)

        shutil.rmtree(first_bundle)
        shutil.copytree(second_bundle, first_bundle)
        bootstrap = self.client.get("/api/v1/bootstrap")

        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertEqual(bootstrap.json()["plan"]["plan_id"], first["plan_id"])
        self.assertIsNone(bootstrap.json().get("bundle"))

    def test_restored_project_identity_supports_the_next_validate_and_compile(
        self,
    ) -> None:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload()).json()
        first_bundle = self.root / "restored-first"
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": planned["plan_id"],
                "output_dir": str(first_bundle),
                "project_id": planned["project_id"],
                "expected_project_revision_id": planned["project_revision_id"],
            },
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)

        restored = self.client.get("/api/v1/bootstrap").json()
        self.assertEqual(restored["plan"]["project_id"], planned["project_id"])
        self.assertEqual(
            restored["plan"]["project_revision_id"],
            restored["bundle"]["project_revision_id"],
        )
        validated = self.client.post(
            "/api/v1/validate",
            json={
                "bundle_dir": restored["bundle"]["bundle_dir"],
                "project_id": restored["bundle"]["project_id"],
                "expected_project_revision_id": restored["bundle"][
                    "project_revision_id"
                ],
                "level": "static",
                "run": False,
            },
        )
        self.assertEqual(validated.status_code, 200, validated.text)

        latest = self.client.get("/api/v1/bootstrap").json()["plan"]
        second_compile = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": latest["plan_id"],
                "output_dir": str(self.root / "restored-second"),
                "project_id": latest["project_id"],
                "expected_project_revision_id": latest["project_revision_id"],
            },
        )
        self.assertEqual(second_compile.status_code, 200, second_compile.text)

    def test_identical_plan_ids_remain_bound_to_distinct_projects(self) -> None:
        first_project = self.client.post(
            "/api/v1/projects", json={"name": "First named project"}
        ).json()
        second_project = self.client.post(
            "/api/v1/projects", json={"name": "Second named project"}
        ).json()
        first_payload = {
            **self.plan_payload(),
            "project_id": first_project["project_id"],
            "project_name": "First named project",
        }
        second_payload = {
            **self.plan_payload(),
            "project_id": second_project["project_id"],
            "project_name": "Second named project",
        }
        first_plan = self.client.post("/api/v1/plan", json=first_payload).json()
        second_plan = self.client.post("/api/v1/plan", json=second_payload).json()
        self.assertEqual(first_plan["plan_id"], second_plan["plan_id"])

        compiled: list[dict[str, object]] = []
        for label, plan in (("first", first_plan), ("second", second_plan)):
            bundle = self.root / f"{label}-owned-bundle"
            response = self.client.post(
                "/api/v1/compile",
                json={
                    "plan_id": plan["plan_id"],
                    "output_dir": str(bundle),
                    "project_id": plan["project_id"],
                    "expected_project_revision_id": plan["project_revision_id"],
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            compiled.append(response.json())

        for label, plan, bundle in (
            ("first", first_plan, compiled[0]),
            ("second", second_plan, compiled[1]),
        ):
            revision = self.client.get(
                f"/api/v1/projects/{plan['project_id']}/revisions/"
                f"{bundle['project_revision_id']}"
            ).json()
            self.assertEqual(
                revision["bundle"]["bundle_dir"],
                str((self.root / f"{label}-owned-bundle").resolve()),
            )
            validated = self.client.post(
                "/api/v1/validate",
                json={
                    "bundle_dir": bundle["bundle_dir"],
                    "project_id": plan["project_id"],
                    "expected_project_revision_id": bundle["project_revision_id"],
                    "level": "static",
                    "run": False,
                },
            )
            self.assertEqual(validated.status_code, 200, validated.text)
            self.assertEqual(validated.json()["project_id"], plan["project_id"])

    def test_replaced_bundle_at_same_path_is_rejected_before_validation_or_job(
        self,
    ) -> None:
        first_plan_response = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(first_plan_response.status_code, 200, first_plan_response.text)
        first_plan = first_plan_response.json()
        first_bundle = self.root / "first-bundle"
        first_compile_response = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": first_plan["plan_id"],
                "output_dir": str(first_bundle),
                "project_id": first_plan["project_id"],
                "expected_project_revision_id": first_plan["project_revision_id"],
            },
        )
        self.assertEqual(
            first_compile_response.status_code, 200, first_compile_response.text
        )
        first_compile = first_compile_response.json()

        second_payload = self.plan_payload()
        target = dict(second_payload["target"])  # type: ignore[arg-type]
        target["sequence_length"] = 96
        second_payload["target"] = target
        second_plan_response = self.client.post("/api/v1/plan", json=second_payload)
        self.assertEqual(
            second_plan_response.status_code, 200, second_plan_response.text
        )
        second_plan = second_plan_response.json()
        self.assertNotEqual(first_plan["plan_id"], second_plan["plan_id"])
        second_bundle = self.root / "second-bundle"
        second_compile_response = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": second_plan["plan_id"],
                "output_dir": str(second_bundle),
                "project_id": second_plan["project_id"],
                "expected_project_revision_id": second_plan["project_revision_id"],
            },
        )
        self.assertEqual(
            second_compile_response.status_code, 200, second_compile_response.text
        )

        shutil.rmtree(first_bundle)
        shutil.copytree(second_bundle, first_bundle)
        report_path = first_bundle / "validation-report.json"
        report_before = report_path.read_bytes()
        jobs_before = list(self.client.app.state.aptus.jobs.root.glob("job_*.json"))
        project_before = self.client.app.state.aptus.projects.get(
            first_plan["project_id"]
        )

        validation = self.client.post(
            "/api/v1/validate",
            json={
                "bundle_dir": str(first_bundle),
                "project_id": first_plan["project_id"],
                "expected_project_revision_id": first_compile["project_revision_id"],
                "level": "static",
                "run": False,
            },
        )
        job = self.client.post(
            "/api/v1/jobs",
            json={
                "bundle_dir": str(first_bundle),
                "project_id": first_plan["project_id"],
                "expected_project_revision_id": first_compile["project_revision_id"],
                "action": "preflight",
            },
        )
        project_after = self.client.app.state.aptus.projects.get(
            first_plan["project_id"]
        )
        jobs_after = list(self.client.app.state.aptus.jobs.root.glob("job_*.json"))

        self.assertEqual(validation.status_code, 409, validation.text)
        self.assertEqual(validation.json()["error"], "project_bundle_binding_mismatch")
        self.assertEqual(job.status_code, 409, job.text)
        self.assertEqual(job.json()["error"], "project_bundle_binding_mismatch")
        self.assertEqual(report_path.read_bytes(), report_before)
        self.assertEqual(jobs_after, jobs_before)
        self.assertEqual(
            project_after["latest_revision_id"],
            project_before["latest_revision_id"],
        )

    def test_compile_conflict_removes_uncommitted_bundle_and_archive(self) -> None:
        plan_response = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(plan_response.status_code, 200, plan_response.text)
        plan = plan_response.json()
        output = self.root / "raced-bundle"
        projects = self.client.app.state.aptus.projects
        original_create_revision = projects.create_revision
        raced = False

        def create_with_competing_revision(
            project_id: str, **changes: object
        ) -> dict[str, object]:
            nonlocal raced
            if changes.get("reason") == "bundle-compiled" and not raced:
                raced = True
                original_create_revision(project_id, reason="competing-update")
            return original_create_revision(project_id, **changes)

        with patch.object(
            projects, "create_revision", side_effect=create_with_competing_revision
        ):
            response = self.client.post(
                "/api/v1/compile",
                json={
                    "plan_id": plan["plan_id"],
                    "output_dir": str(output),
                    "project_id": plan["project_id"],
                    "expected_project_revision_id": plan["project_revision_id"],
                },
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "project_revision_conflict")
        self.assertTrue(raced)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".zip").exists())
        self.assertIsNone(self.client.app.state.aptus.load_bundle_reference())

    def test_compile_conflict_never_deletes_same_path_replacements(self) -> None:
        plan_response = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(plan_response.status_code, 200, plan_response.text)
        plan = plan_response.json()
        output = self.root / "swapped-bundle"
        archive = output.with_suffix(".zip")
        moved_output = self.root / "generated-bundle-moved"
        moved_archive = self.root / "generated-archive-moved.zip"
        projects = self.client.app.state.aptus.projects
        original_create_revision = projects.create_revision
        swapped = False

        def create_after_path_swap(
            project_id: str, **changes: object
        ) -> dict[str, object]:
            nonlocal swapped
            if changes.get("reason") == "bundle-compiled" and not swapped:
                swapped = True
                output.rename(moved_output)
                archive.rename(moved_archive)
                output.mkdir()
                (output / "unrelated.txt").write_text(
                    "keep directory\n", encoding="utf-8"
                )
                archive.write_text("keep archive\n", encoding="utf-8")
                original_create_revision(project_id, reason="competing-update")
            return original_create_revision(project_id, **changes)

        with patch.object(
            projects, "create_revision", side_effect=create_after_path_swap
        ):
            response = self.client.post(
                "/api/v1/compile",
                json={
                    "plan_id": plan["plan_id"],
                    "output_dir": str(output),
                    "project_id": plan["project_id"],
                    "expected_project_revision_id": plan["project_revision_id"],
                },
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue(swapped)
        self.assertEqual(
            (output / "unrelated.txt").read_text(encoding="utf-8"),
            "keep directory\n",
        )
        self.assertEqual(archive.read_text(encoding="utf-8"), "keep archive\n")
        self.assertTrue((moved_output / "bundle-manifest.json").is_file())
        self.assertTrue(moved_archive.is_file())

    def test_bootstrap_omits_an_archive_that_changed_after_compile(self) -> None:
        plan_response = self.client.post("/api/v1/plan", json=self.plan_payload())
        self.assertEqual(plan_response.status_code, 200, plan_response.text)
        plan = plan_response.json()
        bundle = self.root / "archive-bound-bundle"
        compile_response = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": plan["plan_id"],
                "output_dir": str(bundle),
                "project_id": plan["project_id"],
                "expected_project_revision_id": plan["project_revision_id"],
            },
        )
        self.assertEqual(compile_response.status_code, 200, compile_response.text)
        archive = Path(compile_response.json()["archive_path"])
        archive.write_bytes(b"substituted archive")

        bootstrap = self.client.get("/api/v1/bootstrap")

        self.assertEqual(bootstrap.status_code, 200, bootstrap.text)
        self.assertEqual(
            bootstrap.json()["bundle"]["bundle_dir"], str(bundle.resolve())
        )
        self.assertIsNone(bootstrap.json()["bundle"]["archive_path"])

    def test_bootstrap_restores_only_jobs_bound_to_the_exact_current_revision(
        self,
    ) -> None:
        planned = self.client.post("/api/v1/plan", json=self.plan_payload()).json()
        job_id = "job_" + "d" * 32
        active_job = {
            "schema_version": "aptus.job-record.v1",
            "id": job_id,
            "job_id": job_id,
            "state": "running",
            "action": "pilot",
            "bundle_dir": str(self.root / "unrelated-bundle"),
            "created_at": "2026-07-27T12:00:00Z",
        }
        jobs = self.client.app.state.aptus.jobs
        with patch.object(jobs, "list", return_value=[active_job]):
            unrelated = self.client.get("/api/v1/bootstrap")
        self.assertEqual(unrelated.status_code, 200, unrelated.text)
        self.assertIsNone(unrelated.json().get("job"))

        projects = self.client.app.state.aptus.projects
        plan_only = projects.create_revision(
            planned["project_id"],
            reason="job-submitted-fixture",
            base_revision_id=planned["project_revision_id"],
            expected_latest_revision_id=planned["project_revision_id"],
            job_ids=[job_id],
        )
        with patch.object(jobs, "list", return_value=[active_job]):
            plan_only_response = self.client.get("/api/v1/bootstrap")
        self.assertEqual(plan_only_response.status_code, 200, plan_only_response.text)
        self.assertIsNone(plan_only_response.json().get("job"))

        bundle_dir = self.root / "bound-job-bundle"
        compiled = self.client.post(
            "/api/v1/compile",
            json={
                "plan_id": planned["plan_id"],
                "output_dir": str(bundle_dir),
                "project_id": planned["project_id"],
                "expected_project_revision_id": plan_only["revision_id"],
            },
        )
        self.assertEqual(compiled.status_code, 200, compiled.text)
        compiled_revision_id = compiled.json()["project_revision_id"]
        projects.create_revision(
            planned["project_id"],
            reason="bundle-and-job-fixture",
            base_revision_id=compiled_revision_id,
            expected_latest_revision_id=compiled_revision_id,
            job_ids=[job_id],
        )
        bound_active_job = {**active_job, "bundle_dir": str(bundle_dir.resolve())}
        with patch.object(jobs, "list", return_value=[bound_active_job]):
            bound = self.client.get("/api/v1/bootstrap")
        self.assertEqual(bound.status_code, 200, bound.text)
        self.assertEqual(bound.json()["job"]["id"], job_id)

    def test_expected_missing_path_and_no_fit_errors_are_typed(self) -> None:
        missing = self.client.post(
            "/api/v1/profile",
            json={"dataset_path": str(self.root / "missing.jsonl")},
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"], "path_not_found")

        payload = self.plan_payload()
        payload["model"] = {
            **payload["model"],
            "parameters_b": 70,
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "layers": 80,
        }
        payload["hardware"] = {
            **payload["hardware"],
            "vram_gib": 4,
            "free_vram_gib": 4,
            "reserve_gib": 1,
            "host_ram_gib": 512,
            "host_ram_free_gib": 512,
            "disk_free_gib": 2000,
        }
        no_fit = self.client.post("/api/v1/plan", json=payload)
        self.assertEqual(no_fit.status_code, 422, no_fit.text)
        failure = no_fit.json()
        self.assertEqual(failure["error"], "no_feasible_plan")
        self.assertEqual(len(failure["candidates"]), 12)
        self.assertEqual(failure["model_policy_decision_source"], "user-attested")
        self.assertIsNone(failure["inspection_receipt"])
        self.assertEqual(failure["model"]["model_id"], payload["model"]["model_id"])
        self.assertEqual(failure["model"]["revision"], payload["model"]["revision"])
        self.assertTrue(
            all(
                candidate["model_policy_decision_id"]
                == failure["model_policy_decision"]["decision_id"]
                for candidate in failure["candidates"]
            )
        )
        required_execution_fields = {
            "method",
            "distribution",
            "status",
            "feasible",
            "rejection_reasons",
            "target_modules",
            "runtime_contract",
        }
        self.assertTrue(
            all(
                required_execution_fields.issubset(candidate)
                and candidate["feasible"] is False
                and candidate["status"] in {"infeasible", "unsupported"}
                and candidate["rejection_reasons"]
                and candidate["runtime_contract"]["schema_version"]
                == "aptus.runtime-contract.v1"
                for candidate in failure["candidates"]
            )
        )

    def test_no_fit_response_preserves_provider_inspection_receipt(self) -> None:
        payload = self.plan_payload()
        receipt = self.inspection_receipt(payload)
        payload["inspection_receipt"] = receipt
        payload["hardware"] = {
            **payload["hardware"],
            "host_ram_gib": 1,
            "host_ram_free_gib": 1,
            "disk_free_gib": 0.1,
        }

        response = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(response.status_code, 422, response.text)
        failure = response.json()
        self.assertEqual(failure["error"], "no_feasible_plan")
        self.assertEqual(failure["model_policy_decision_source"], "provider-inspection")
        self.assertEqual(
            failure["inspection_receipt"]["receipt_id"], receipt["receipt_id"]
        )
        self.assertEqual(
            failure["inspection_receipt"]["decision"]["decision_id"],
            failure["model_policy_decision"]["decision_id"],
        )
        self.assertEqual(failure["model"]["model_id"], payload["model"]["model_id"])
        self.assertEqual(failure["model"]["revision"], payload["model"]["revision"])
        self.assertTrue(
            all(
                candidate["model_policy_decision_id"]
                == failure["model_policy_decision"]["decision_id"]
                for candidate in failure["candidates"]
            )
        )

    def test_qwen2_no_fit_response_preserves_dense_policy_projection(self) -> None:
        payload = self.plan_payload()
        payload["model"] = {
            **payload["model"],
            "model_id": QWEN2_5_ACCEPTANCE_MODEL_ID,
            "revision": QWEN2_5_ACCEPTANCE_REVISION,
            "family": "qwen",
            "parameters_b": 0.494,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "layers": 24,
            "context_length": 32768,
            "model_type": "qwen2",
            "architecture": "Qwen2ForCausalLM",
            "quantization_bits": 4,
            "quantization_layout": {
                "default_bits": 4,
                "default_group_size": 64,
                "module_overrides": [],
            },
        }
        payload["hardware"] = {
            **payload["hardware"],
            "backend": "mps",
            "gpu_count": 1,
            "vram_gib": 64,
            "free_vram_gib": 64,
            "supports_bf16": False,
            "supports_8bit": False,
            "supports_4bit": False,
            "host_ram_gib": 64,
            "host_ram_free_gib": 64,
            "reserve_gib": 8,
            "disk_free_gib": 0.1,
        }
        payload["target"] = {
            **payload["target"],
            "method_preference": "qlora",
            "training_runtime": "mlx-lm",
            "effective_batch_size": 1,
        }

        response = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(response.status_code, 422, response.text)
        failure = response.json()
        self.assertEqual(failure["error"], "no_feasible_plan")
        self.assertEqual(failure["model_policy_decision_source"], "user-attested")
        decision = failure["model_policy_decision"]
        self.assertEqual(decision["policy_id"], "model.qwen2-24l.mlx-qlora")
        self.assertEqual(
            decision["reason_codes"],
            ["reviewed-runtime-path", "pilot-not-yet-proven"],
        )
        self.assertEqual(len(decision["paths"]), 1)
        self.assertEqual(
            decision["paths"][0]["adapter_profile_id"],
            "dense-causal-lm.v1",
        )
        bound = [
            candidate
            for candidate in failure["candidates"]
            if candidate["policy_binding"] is not None
        ]
        self.assertEqual(len(bound), 1)
        candidate = bound[0]
        self.assertFalse(candidate["feasible"])
        self.assertEqual(candidate["status"], "infeasible")
        self.assertEqual(
            candidate["policy_binding"]["path_id"],
            "mlx-lm.qlora.single.dense-causal-lm.v1",
        )
        self.assertEqual(
            candidate["target_modules"],
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    def test_hardware_probe_runtime_failure_returns_manual_fallback(self) -> None:
        with patch(
            "aptus.api.probe_local_hardware",
            side_effect=RuntimeError("driver unavailable"),
        ):
            response = self.client.get("/api/v1/hardware")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertTrue(response.json()["manual_facts_supported"])

    def test_darwin_local_scan_enforces_unified_memory_reserve(self) -> None:
        payload = self.plan_payload()
        payload["hardware"] = {
            **payload["hardware"],
            "discovery": "local-scan",
            "reserve_gib": 2,
        }
        apple_hardware = build_hardware_spec(
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
        with (
            patch("aptus.api.sys.platform", "darwin"),
            patch(
                "aptus.api.probe_local_hardware",
                return_value=apple_hardware,
            ) as probe,
        ):
            response = self.client.post("/api/v1/plan", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        probe.assert_called_once_with(reserve_gib=8.0)
        self.assertEqual(
            response.json()["hardware"]["reserve_per_device_bytes"],
            8 * 1024**3,
        )

    def test_untrusted_host_header_is_rejected(self) -> None:
        response = self.client.get(
            "/api/v1/health", headers={"host": "attacker.example"}
        )
        self.assertEqual(response.status_code, 400, response.text)

    def test_job_submission_conflict_is_typed_as_http_409(self) -> None:
        bundle = self.root / "bundle"
        identity = self.owned_bundle_request(bundle)
        with patch.object(
            self.client.app.state.aptus.jobs,
            "submit",
            side_effect=ActiveJobError("one local GPU job is already active"),
        ):
            response = self.client.post(
                "/api/v1/jobs",
                json={"bundle_dir": str(bundle), "action": "pilot", **identity},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "active_job_conflict")

    def test_stale_policy_job_submission_requires_replanning(self) -> None:
        bundle = self.root / "bundle"
        identity = self.owned_bundle_request(bundle)
        plan_id = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))[
            "plan_id"
        ]
        (bundle / "validation-report.json").write_text(
            json.dumps(
                {
                    "schema_version": "aptus.validation.v2",
                    "state": "pilot-pass",
                }
            ),
            encoding="utf-8",
        )
        changed_snapshot = copy.deepcopy(current_model_policy_snapshot())
        changed_snapshot["dense_families"] = sorted(
            [*changed_snapshot["dense_families"], "future-dense-family"]
        )

        with patch(
            "aptus.model_compatibility.current_model_policy_snapshot",
            return_value=changed_snapshot,
        ):
            response = self.client.post(
                "/api/v1/jobs",
                json={
                    "bundle_dir": str(bundle),
                    "action": "train",
                    "confirm_full_train": True,
                    **identity,
                },
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "replan_required")
        self.assertEqual(response.json()["plan_id"], plan_id)
        self.assertEqual(response.json()["found_schema"], "aptus.training-plan.v6")
        self.assertEqual(response.json()["required_schema"], "aptus.training-plan.v6")
        self.assertEqual(response.json()["project_id"], identity["project_id"])
        self.assertEqual(
            response.json()["project_revision_id"],
            identity["expected_project_revision_id"],
        )
        self.assertEqual(self.client.app.state.aptus.jobs.list(), [])

    def test_job_prerequisite_failure_is_typed_as_http_409(self) -> None:
        bundle = self.root / "bundle"
        identity = self.owned_bundle_request(bundle)
        with patch.object(
            self.client.app.state.aptus.jobs,
            "submit",
            side_effect=JobPrerequisiteError(
                action="pilot",
                required_state="measured-preflight-pass",
                current_state="model-data-pass",
                reason="insufficient_state",
            ),
        ):
            response = self.client.post(
                "/api/v1/jobs",
                json={"bundle_dir": str(bundle), "action": "pilot", **identity},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "job_prerequisite_not_met")
        self.assertEqual(response.json()["action"], "pilot")
        self.assertEqual(response.json()["required_state"], "measured-preflight-pass")
        self.assertEqual(response.json()["current_state"], "model-data-pass")
        self.assertEqual(response.json()["reason"], "insufficient_state")

    def test_post_disposition_on_completed_train_returns_job(self) -> None:
        job_id = self.seed_job(job_id="job_" + "a" * 32)
        response = self.client.post(
            f"/api/v1/jobs/{job_id}/disposition",
            json={"kind": "use"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["id"], job_id)
        disposition = body["run_disposition"]
        self.assertEqual(disposition["schema_version"], "aptus.run-disposition.v1")
        self.assertEqual(disposition["kind"], "use")
        self.assertEqual(disposition["source"], "operator-attested")
        self.assertEqual(disposition["job_id"], job_id)
        self.assertEqual(disposition["plan_id"], "plan_abc")
        self.assertIsNone(disposition["previous_kind"])
        self.assertEqual(disposition["operator_next_step"]["action"], "load-adapter")
        self.assertEqual(disposition["evidence"]["evaluation_decision"], "omitted")

    def test_post_disposition_on_pilot_is_typed_as_http_409(self) -> None:
        job_id = self.seed_job(
            job_id="job_" + "b" * 32, action="pilot", state="completed"
        )
        response = self.client.post(
            f"/api/v1/jobs/{job_id}/disposition",
            json={"kind": "use"},
        )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"], "job_disposition_refused")
        self.assertTrue(response.json()["message"])

    def test_post_disposition_missing_job_is_typed_as_http_404(self) -> None:
        job_id = "job_" + "c" * 32
        response = self.client.post(
            f"/api/v1/jobs/{job_id}/disposition",
            json={"kind": "done"},
        )
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(response.json()["error"], "job_not_found")
        self.assertEqual(response.json()["job_id"], job_id)

    def test_get_job_includes_run_disposition_after_post(self) -> None:
        job_id = self.seed_job(job_id="job_" + "d" * 32)
        posted = self.client.post(
            f"/api/v1/jobs/{job_id}/disposition",
            json={"kind": "done"},
        )
        self.assertEqual(posted.status_code, 200, posted.text)
        response = self.client.get(f"/api/v1/jobs/{job_id}")
        self.assertEqual(response.status_code, 200, response.text)
        disposition = response.json()["run_disposition"]
        self.assertEqual(disposition["kind"], "done")
        self.assertEqual(disposition["source"], "operator-attested")
        self.assertEqual(disposition["operator_next_step"]["action"], "none")

    def test_get_job_omits_run_disposition_without_sibling(self) -> None:
        job_id = self.seed_job(job_id="job_" + "e" * 32)
        response = self.client.get(f"/api/v1/jobs/{job_id}")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIsNone(body.get("run_disposition"))
        self.assertNotEqual((body.get("run_disposition") or {}).get("kind"), "use")

    def test_validate_response_defers_deep_pilot_authorization_to_submit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            static = root / "web"
            static.mkdir()
            (static / "index.html").write_text("ok", encoding="utf-8")
            with patch("aptus.validation.validate_bundle") as validate_mock:
                app = create_app(
                    state_dir=root / "state",
                    static_dir=static,
                    allow_unauthenticated=True,
                )
            with patch.object(
                app.state.aptus.jobs,
                "pilot_authorization",
            ) as authorization:
                with TestClient(app) as client:
                    plan_response = client.post(
                        "/api/v1/plan", json=self.plan_payload()
                    )
                    self.assertEqual(plan_response.status_code, 200, plan_response.text)
                    plan = plan_response.json()
                    compile_response = client.post(
                        "/api/v1/compile",
                        json={
                            "plan_id": plan["plan_id"],
                            "output_dir": str(root / "bundle"),
                            "project_id": plan["project_id"],
                            "expected_project_revision_id": plan["project_revision_id"],
                        },
                    )
                    self.assertEqual(
                        compile_response.status_code, 200, compile_response.text
                    )
                    compiled = compile_response.json()
                    validate_mock.return_value = ValidationReport(
                        state=ValidationState.PILOT_PASS,
                        findings=(),
                        checked_files=(),
                        artifact_fingerprint=compiled["report"]["artifact_fingerprint"],
                        validation_level="pilot",
                    )
                    response = client.post(
                        "/api/v1/validate",
                        json={
                            "bundle_dir": str(root / "bundle"),
                            "project_id": plan["project_id"],
                            "expected_project_revision_id": compiled[
                                "project_revision_id"
                            ],
                            "level": "static",
                            "run": False,
                        },
                    )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["authorization_status"], "deferred")
        self.assertFalse(response.json()["authorization_current"])
        self.assertIsNone(response.json()["prelaunch_capacity_check"])
        self.assertIn(
            "performed atomically when full training is submitted",
            response.json()["authorization_error"],
        )
        authorization.assert_not_called()


@unittest.skipIf(
    TestClient is None,
    "Install the server and test extras for endpoint integration tests.",
)
class AppleRuntimeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.client = TestClient(
            create_app(
                state_dir=Path(self.temporary.name) / "state",
                allow_unauthenticated=True,
            )
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_platform_and_runtime_inventory_are_explicit_endpoints(self) -> None:
        platform_profile = types.SimpleNamespace(
            to_dict=lambda: {
                "chip_name": "Apple M5 Pro",
                "metal_gpu_core_count": 20,
                "unified_memory_bytes": 64,
            }
        )
        with (
            patch("aptus.api.probe_apple_platform", return_value=platform_profile),
            patch(
                "aptus.api.runtime_inventory",
                return_value={
                    "schema_version": "aptus.runtime-inventory.v1",
                    "interpreters": [],
                    "available": {"mlx-lm": ["/managed/python"]},
                    "compatible": {"mlx-lm": ["/managed/python"]},
                    "configuration": {},
                },
            ),
        ):
            self.client.app.state.aptus.runtime_paths = {"mlx-lm": "/managed/python"}
            platform_response = self.client.get("/api/v1/platform")
            runtime_response = self.client.get("/api/v1/runtimes")

        self.assertEqual(platform_response.status_code, 200)
        self.assertEqual(
            platform_response.json()["platform"]["chip_name"], "Apple M5 Pro"
        )
        self.assertEqual(
            platform_response.json()["platform"]["metal_gpu_core_count"], 20
        )
        self.assertEqual(
            runtime_response.json()["available"]["mlx-lm"], ["/managed/python"]
        )
        self.assertEqual(
            runtime_response.json()["selected"],
            {"mlx-lm": "/managed/python"},
        )

    def test_inference_models_reject_non_loopback_endpoints(self) -> None:
        response = self.client.post(
            "/api/v1/inference/models",
            json={
                "service": "lm-studio",
                "endpoint": "http://example.com:1234",
            },
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"]["code"], "non_loopback_endpoint")

    def test_runtime_configuration_returns_the_canonical_interpreter(self) -> None:
        configured = {
            "status": "ok",
            "runtime_id": "mlx-lm",
            "interpreter_path": "/managed/python",
            "interpreter": {"path": "/managed/python"},
            "persisted": True,
        }
        with patch.object(
            self.client.app.state.aptus,
            "configure_runtime",
            return_value=configured,
        ) as configure:
            response = self.client.post(
                "/api/v1/runtimes/configure",
                json={
                    "runtime_id": "mlx-lm",
                    "interpreter_path": "/selected/python",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["interpreter_path"], "/managed/python")
        configure.assert_called_once_with("mlx-lm", Path("/selected/python"))

    def test_inference_generation_uses_the_selected_local_adapter(self) -> None:
        client = types.SimpleNamespace(
            generate=lambda **values: {
                "status": "ok",
                "service": "omlx",
                "endpoint": "http://127.0.0.1:8080/v1",
                "model": "local/model",
                "content": values["messages"][0]["content"],
                "usage": None,
                "response_id": "response-test",
                "payload": {},
            }
        )
        with patch("aptus.api.OMLXClient", return_value=client) as constructor:
            response = self.client.post(
                "/api/v1/inference/generate",
                json={
                    "service": "omlx",
                    "model": "local/model",
                    "messages": [{"role": "user", "content": "test prompt"}],
                    "max_tokens": 32,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["content"], "test prompt")
        constructor.assert_called_once_with(endpoint=None, timeout=5.0)


if __name__ == "__main__":
    unittest.main()
