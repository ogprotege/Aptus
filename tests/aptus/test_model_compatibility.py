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
    ModelPolicyDecision,
    ModelPolicyDecisionKind,
    ModelPolicyPath,
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

        self.assertEqual(decision.kind, ModelPolicyDecisionKind.PATH_MATCHED)
        self.assertEqual(decision.family, "qwen3_moe")
        self.assertEqual(len(decision.paths), 1)
        path = decision.paths[0]
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
        path = evaluate_model_compatibility(self.subject).paths[0]
        with self.assertRaises(ValueError):
            ModelPolicyPath(
                method=Method.FULL,
                distribution=path.distribution,
                adapter_profile_id=path.adapter_profile_id,
                target_modules=path.target_modules,
                runtime_contract=path.runtime_contract,
            )
        with self.assertRaises(ValueError):
            ModelPolicyPath(
                method=path.method,
                distribution=path.distribution,
                adapter_profile_id=path.adapter_profile_id,
                target_modules=("q_proj", "q_proj"),
                runtime_contract=path.runtime_contract,
            )
        with self.assertRaises(ValueError):
            ModelPolicyDecision(
                kind=ModelPolicyDecisionKind.PATH_MATCHED,
                family="qwen3_moe",
                paths=(),
                reason="Matched.",
            )
        with self.assertRaises(ValueError):
            ModelPolicyDecision(
                kind=ModelPolicyDecisionKind.PATH_MATCHED,
                family=None,
                paths=(path,),
                reason="Matched.",
            )
        with self.assertRaises(ValueError):
            ModelPolicyDecision(
                kind=ModelPolicyDecisionKind.FAMILY_RECOGNIZED,
                family="qwen",
                paths=(path,),
                reason="Recognized.",
            )
        with self.assertRaises(ValueError):
            ModelPolicyDecision(
                kind=ModelPolicyDecisionKind.PATH_MATCHED,
                family="qwen3_moe",
                paths=(path, path),
                reason="Matched.",
            )

    def test_v1_projection_rejects_heterogeneous_policy_paths(self) -> None:
        qwen_path = evaluate_model_compatibility(self.subject).paths[0]
        cuda_contract = validate_execution_path_selection(
            method=Method.LORA,
            training_runtime=TrainingRuntime.TRANSFORMERS_PEFT_CUDA,
            compute_backend=Backend.CUDA,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.ATTENTION_QKVO_V1,
        )
        cuda_path = ModelPolicyPath(
            method=Method.LORA,
            distribution=Distribution.SINGLE,
            adapter_profile_id=AdapterProfile.ATTENTION_QKVO_V1,
            target_modules=adapter_target_modules(AdapterProfile.ATTENTION_QKVO_V1),
            runtime_contract=cuda_contract,
        )
        decision = ModelPolicyDecision(
            kind=ModelPolicyDecisionKind.PATH_MATCHED,
            family="qwen3_moe",
            paths=(qwen_path, cuda_path),
            reason="Matched.",
        )

        with patch("aptus.model_compatibility.validate_model_policy_path"):
            with self.assertRaisesRegex(ValueError, "heterogeneous paths"):
                compatibility_response_v1(decision)

    def test_v1_projection_rejects_a_forged_policy_path(self) -> None:
        path = evaluate_model_compatibility(self.subject).paths[0]
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
                decision = ModelPolicyDecision(
                    kind=ModelPolicyDecisionKind.PATH_MATCHED,
                    family="qwen3_moe",
                    paths=(forged_path,),
                    reason="Matched.",
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

    def test_qwen_v3_plan_and_candidate_identities_do_not_change(self) -> None:
        self.assertEqual(self.plan.plan_id, "plan_6cf48d2656249b62457e")
        self.assertEqual(
            self.plan.recommended.candidate_id,
            "cand_f41093c9a9aaf5294138",
        )
        expected = {
            ("full", "single"): "cand_5a67adaa9f9df795bde1",
            ("full", "ddp"): "cand_1f94483ffa3951a0476f",
            ("full", "fsdp"): "cand_5df8d5aaa561184dc1f5",
            ("lora", "single"): "cand_7671ea9e9278c8c23a49",
            ("lora", "ddp"): "cand_ecb820e9cbe2907fe7a7",
            ("lora", "fsdp"): "cand_7c013a184afc2c42e2b6",
            ("int8-lora", "single"): "cand_d58e772674b58bde3aca",
            ("int8-lora", "ddp"): "cand_3547790cd48c1edb5481",
            ("int8-lora", "fsdp"): "cand_cf3afcedc7be1f6ac9d8",
            ("qlora", "single"): "cand_f41093c9a9aaf5294138",
            ("qlora", "ddp"): "cand_a135a981bc7548521d24",
            ("qlora", "fsdp"): "cand_66ef650a0a8dcd6d522c",
        }
        actual = {
            (candidate.method.value, candidate.distribution.value): (
                candidate.candidate_id
            )
            for candidate in self.plan.candidates
        }
        self.assertEqual(actual, expected)

    def test_dense_cuda_v3_plan_and_candidate_identities_do_not_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = make_plan(Path(temporary))

        self.assertEqual(plan.plan_id, "plan_67777e0dd66328c9ab5a")
        expected = {
            ("full", "single"): "cand_b0f860595649f2ba6394",
            ("full", "ddp"): "cand_811d09debf94b91fe1f1",
            ("full", "fsdp"): "cand_1cb6a1b1dc53e7d2791e",
            ("lora", "single"): "cand_8a5af924ce3ca398b700",
            ("lora", "ddp"): "cand_9bdf56c1b4b36a947b2f",
            ("lora", "fsdp"): "cand_178c0f3988dd6557db9e",
            ("int8-lora", "single"): "cand_cc3fe360eeabd2bd0561",
            ("int8-lora", "ddp"): "cand_730b9bc51be5dd60a17a",
            ("int8-lora", "fsdp"): "cand_7332d35546f7714f9af9",
            ("qlora", "single"): "cand_b94008cb7e9d98b7c748",
            ("qlora", "ddp"): "cand_ed9a14b544302fd1170d",
            ("qlora", "fsdp"): "cand_3ab87df7672e3725e914",
        }
        actual = {
            (candidate.method.value, candidate.distribution.value): (
                candidate.candidate_id
            )
            for candidate in plan.candidates
        }
        self.assertEqual(actual, expected)

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
