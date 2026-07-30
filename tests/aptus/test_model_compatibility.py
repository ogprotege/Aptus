from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from aptus.domain import (
    AdapterProfile,
    Backend,
    Distribution,
    Method,
    ModelCompatibilitySubject,
    ModelPolicyDecisionKind,
    QuantizationLayout,
    TrainingRuntime,
    to_primitive,
)
from aptus.model_compatibility import (
    INVALID_COMPATIBILITY_FACTS_REASON,
    QWEN3_MOE_FOUR_BIT_REASON,
    QWEN3_MOE_IDENTITY_REASON,
    QWEN3_MOE_LAYOUT_REASON,
    QWEN3_MOE_SHARED_EXPERT_REASON,
    QWEN3_MOE_TOPOLOGY_REASON,
    UNREVIEWED_SPARSE_MODEL_REASON,
    adapter_target_modules,
    compatibility_response_v1,
    evaluate_model_compatibility,
    subject_from_model,
    validate_execution_path_selection,
)
from tests.aptus.helpers import make_plan, make_qwen3_moe_plan


class ModelCompatibilityPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cls.plan = make_qwen3_moe_plan(Path(temporary))
        cls.subject = subject_from_model(cls.plan.model)

    def test_exact_qwen_policy_emits_one_registry_bound_path(self) -> None:
        decision = evaluate_model_compatibility(self.subject)

        self.assertEqual(decision.schema_version, "aptus.model-compatibility.v2")
        self.assertEqual(decision.policy_id, "model.qwen3-moe.mlx-qlora")
        self.assertEqual(decision.policy_version, "1.0.0")
        self.assertTrue(decision.decision_id.startswith("compat_"))
        self.assertEqual(len(decision.subject_facts_sha256), 64)
        self.assertEqual(decision.kind, ModelPolicyDecisionKind.PATH_MATCHED)
        self.assertEqual(decision.family, "qwen3_moe")
        self.assertEqual(len(decision.paths), 1)
        path = decision.paths[0]
        self.assertEqual(path.path_id, "mlx-lm.qlora.single.attention-qkvo.v1")
        self.assertEqual(
            path.required_validation_levels,
            ("model-data", "measured-preflight", "pilot"),
        )
        self.assertTrue(path.evidence_ids)
        self.assertEqual(path.method, Method.QLORA)
        self.assertEqual(path.distribution, Distribution.SINGLE)
        self.assertEqual(
            path.adapter_profile_id,
            AdapterProfile.ATTENTION_QKVO_V1,
        )
        self.assertEqual(
            path.target_modules,
            ("q_proj", "k_proj", "v_proj", "o_proj"),
        )
        self.assertEqual(path.runtime_contract.training_runtime, TrainingRuntime.MLX_LM)
        self.assertEqual(path.runtime_contract.compute_backend, Backend.MPS)
        self.assertEqual(path.runtime_contract.compiler_id, "mlx-lm.qlora.v1")
        self.assertEqual(path.runtime_contract.estimator_id, "aptus-memory-mlx-v2")
        self.assertEqual(path.runtime_contract.export_kind, "mlx-lm-adapter")
        self.assertEqual(
            to_primitive(decision)["kind"],
            "path-matched",
        )

    def test_qwen_policy_mutations_fail_at_the_first_predicate(self) -> None:
        shared_expert = replace(
            self.subject.moe,
            shared_expert_intermediate_size=1024,
        )
        no_sparse_layer = replace(
            self.subject.moe,
            decoder_sparse_step=self.subject.layers + 1,
        )
        cases = {
            "family": (
                replace(self.subject, family="qwen"),
                QWEN3_MOE_IDENTITY_REASON,
            ),
            "model type": (
                replace(self.subject, model_type="qwen3"),
                QWEN3_MOE_IDENTITY_REASON,
            ),
            "architecture": (
                replace(self.subject, architecture="Qwen3MoeModel"),
                QWEN3_MOE_IDENTITY_REASON,
            ),
            "layout": (
                replace(
                    self.subject,
                    quantization_layout=QuantizationLayout(4, 64),
                ),
                QWEN3_MOE_LAYOUT_REASON,
            ),
            "topology": (
                replace(self.subject, moe=None),
                QWEN3_MOE_TOPOLOGY_REASON,
            ),
            "sparse layers": (
                replace(self.subject, moe=no_sparse_layer),
                QWEN3_MOE_TOPOLOGY_REASON,
            ),
            "shared expert": (
                replace(self.subject, moe=shared_expert),
                QWEN3_MOE_SHARED_EXPERT_REASON,
            ),
            "quantization bits": (
                replace(self.subject, quantization_bits=8),
                QWEN3_MOE_FOUR_BIT_REASON,
            ),
        }
        for name, (subject, reason) in cases.items():
            with self.subTest(name=name):
                decision = evaluate_model_compatibility(subject)
                self.assertEqual(decision.kind, ModelPolicyDecisionKind.BLOCKED)
                self.assertEqual(decision.reason, reason)
                self.assertEqual(decision.paths, ())

    def test_compatibility_fact_errors_block_an_exact_policy_match(self) -> None:
        for fact_error in (
            "moe: conflicting provider facts",
            "quantization: conflicting provider facts",
        ):
            with self.subTest(fact_error=fact_error):
                decision = evaluate_model_compatibility(
                    replace(self.subject, fact_errors=(fact_error,))
                )
                self.assertEqual(decision.kind, ModelPolicyDecisionKind.BLOCKED)
                self.assertEqual(
                    decision.reason,
                    INVALID_COMPATIBILITY_FACTS_REASON,
                )
                self.assertEqual(decision.paths, ())

    def test_fact_errors_do_not_upgrade_a_qwen_near_match_to_exact_identity(
        self,
    ) -> None:
        subject = replace(
            self.subject,
            model_type="not_qwen",
            architecture="NotQwen",
            fact_errors=("quantization: conflicting provider facts",),
        )

        decision = evaluate_model_compatibility(subject)
        response = compatibility_response_v1(decision)

        self.assertEqual(decision.kind, ModelPolicyDecisionKind.BLOCKED)
        self.assertEqual(decision.reason, QWEN3_MOE_IDENTITY_REASON)
        self.assertEqual(response["status"], "unsupported")
        self.assertEqual(
            response["reason"],
            "No exact Aptus model-family compatibility policy matches this "
            "provider model type and architecture.",
        )

    def test_dense_unknown_and_unreviewed_sparse_states_remain_distinct(self) -> None:
        dense = ModelCompatibilitySubject(
            family="llama",
            model_type="llama",
            architecture="LlamaForCausalLM",
            layers=32,
            quantization_bits=None,
            quantization_layout=None,
            moe=None,
        )
        unknown = replace(dense, family="custom", model_type="custom")
        sparse = replace(
            unknown,
            family="mixtral",
            model_type="mixtral",
            architecture="MixtralForCausalLM",
            moe=self.subject.moe,
        )

        self.assertEqual(
            evaluate_model_compatibility(dense).kind,
            ModelPolicyDecisionKind.FAMILY_RECOGNIZED,
        )
        self.assertEqual(
            evaluate_model_compatibility(unknown).kind,
            ModelPolicyDecisionKind.UNKNOWN,
        )
        self.assertEqual(
            evaluate_model_compatibility(sparse).kind,
            ModelPolicyDecisionKind.BLOCKED,
        )

    def test_sparse_identity_markers_block_when_topology_is_missing(self) -> None:
        base = ModelCompatibilitySubject(
            family="qwen",
            model_type="qwen2",
            architecture="Qwen2MoeForCausalLM",
            layers=32,
            quantization_bits=None,
            quantization_layout=None,
            moe=None,
        )
        subjects = (
            base,
            replace(
                base,
                family="mistral",
                model_type="mistral",
                architecture="MixtralForCausalLM",
            ),
        )

        for subject in subjects:
            with self.subTest(subject=subject):
                decision = evaluate_model_compatibility(subject)
                self.assertEqual(decision.kind, ModelPolicyDecisionKind.BLOCKED)
                self.assertEqual(decision.reason, UNREVIEWED_SPARSE_MODEL_REASON)

    def test_execution_path_validation_delegates_to_the_method_registry(self) -> None:
        contract = validate_execution_path_selection(
            method=Method.QLORA,
            training_runtime=TrainingRuntime.MLX_LM,
            compute_backend=Backend.MPS,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.ATTENTION_QKVO_V1,
        )
        self.assertEqual(contract.compiler_id, "mlx-lm.qlora.v1")
        self.assertEqual(contract.estimator_id, "aptus-memory-mlx-v2")
        self.assertEqual(contract.export_kind, "mlx-lm-adapter")

        invalid = (
            {
                "method": Method.FULL,
                "training_runtime": TrainingRuntime.TRANSFORMERS_PEFT_CUDA,
                "compute_backend": Backend.CUDA,
                "distribution": Distribution.SINGLE,
                "adapter_profile_id": AdapterProfile.ATTENTION_QKVO_V1,
            },
            {
                "method": Method.QLORA,
                "training_runtime": TrainingRuntime.MLX_LM,
                "compute_backend": Backend.CUDA,
                "distribution": Distribution.SINGLE,
                "adapter_profile_id": AdapterProfile.ATTENTION_QKVO_V1,
            },
            {
                "method": Method.QLORA,
                "training_runtime": TrainingRuntime.MLX_LM,
                "compute_backend": Backend.MPS,
                "distribution": Distribution.DDP,
                "adapter_profile_id": AdapterProfile.ATTENTION_QKVO_V1,
            },
            {
                "method": Method.QLORA,
                "training_runtime": TrainingRuntime.MLX_LM,
                "compute_backend": Backend.MPS,
                "distribution": Distribution.SINGLE,
                "adapter_profile_id": None,
            },
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_execution_path_selection(**values)

    def test_domain_decision_and_path_invariants_fail_closed(self) -> None:
        decision = evaluate_model_compatibility(self.subject)
        path = decision.paths[0]
        with self.assertRaises(ValueError):
            replace(path, method=Method.FULL)
        with self.assertRaises(ValueError):
            replace(path, target_modules=("q_proj", "q_proj"))
        with self.assertRaises(ValueError):
            replace(decision, paths=())
        with self.assertRaises(ValueError):
            replace(decision, family=None)
        with self.assertRaises(ValueError):
            replace(
                decision,
                kind=ModelPolicyDecisionKind.FAMILY_RECOGNIZED,
                family="qwen",
                paths=(path,),
            )
        with self.assertRaises(ValueError):
            replace(decision, paths=(path, path))

    def test_v1_projection_rejects_heterogeneous_policy_paths(self) -> None:
        qwen_decision = evaluate_model_compatibility(self.subject)
        qwen_path = qwen_decision.paths[0]
        cuda_contract = validate_execution_path_selection(
            method=Method.LORA,
            training_runtime=TrainingRuntime.TRANSFORMERS_PEFT_CUDA,
            compute_backend=Backend.CUDA,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.ATTENTION_QKVO_V1,
        )
        cuda_path = replace(
            qwen_path,
            path_id="test.cuda.lora.single.attention-qkvo.v1",
            method=Method.LORA,
            distribution=Distribution.SINGLE,
            target_modules=adapter_target_modules(AdapterProfile.ATTENTION_QKVO_V1),
            runtime_contract=cuda_contract,
        )
        decision = replace(
            qwen_decision,
            paths=(qwen_path, cuda_path),
        )

        with patch("aptus.model_compatibility.validate_model_policy_path"):
            with self.assertRaisesRegex(ValueError, "heterogeneous paths"):
                compatibility_response_v1(decision)

    def test_v1_projection_rejects_a_forged_policy_path(self) -> None:
        base_decision = evaluate_model_compatibility(self.subject)
        path = base_decision.paths[0]
        forged_paths = (
            replace(path, target_modules=("q_proj", "k_proj")),
            replace(
                path,
                runtime_contract=replace(
                    path.runtime_contract,
                    compiler_id="forged.compiler",
                ),
            ),
        )

        for forged_path in forged_paths:
            with self.subTest(path=forged_path):
                decision = replace(
                    base_decision,
                    paths=(forged_path,),
                )
                with self.assertRaises(ValueError):
                    compatibility_response_v1(decision)

    def test_policy_registry_fails_import_when_catalog_targets_drift(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = "src:."
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import aptus.catalog as catalog; "
                    "catalog.TARGET_MODULES['qwen3_moe'] = ('q_proj',); "
                    "import aptus.model_compatibility"
                ),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "adapter targets differ from the family catalog",
            completed.stderr,
        )

    def test_qwen_v4_identities_are_deterministic_and_policy_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repeated = make_qwen3_moe_plan(Path(temporary))

        self.assertEqual(self.plan.plan_id, repeated.plan_id)
        self.assertEqual(
            tuple(item.candidate_id for item in self.plan.candidates),
            tuple(item.candidate_id for item in repeated.candidates),
        )
        self.assertEqual(len({item.candidate_id for item in self.plan.candidates}), 12)
        self.assertTrue(
            all(
                item.model_policy_decision_id
                == self.plan.model_policy_decision.decision_id
                for item in self.plan.candidates
            )
        )
        bound = [
            item for item in self.plan.candidates if item.policy_binding is not None
        ]
        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0].candidate_id, self.plan.recommended.candidate_id)

    def test_dense_cuda_v4_identities_are_deterministic_and_decision_bound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary))
            repeated = make_plan(Path(temporary))

        self.assertEqual(plan.plan_id, repeated.plan_id)
        self.assertEqual(
            tuple(item.candidate_id for item in plan.candidates),
            tuple(item.candidate_id for item in repeated.candidates),
        )
        self.assertTrue(all(item.policy_binding is None for item in plan.candidates))
        self.assertTrue(
            all(
                item.model_policy_decision_id == plan.model_policy_decision.decision_id
                for item in plan.candidates
            )
        )

    def test_host_policy_import_order_has_no_cycle(self) -> None:
        orders = (
            ("aptus.model_compatibility", "aptus.methods", "aptus.inspection"),
            ("aptus.inspection", "aptus.planning", "aptus.model_compatibility"),
            ("aptus.methods", "aptus.model_compatibility", "aptus.api_contracts"),
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = "src:."
        for modules in orders:
            with self.subTest(modules=modules):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "; ".join(f"import {module}" for module in modules),
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
