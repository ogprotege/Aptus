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
    GEMMA4_POLICY_ID,
    INVALID_COMPATIBILITY_FACTS_REASON,
    QWEN3_MOE_FOUR_BIT_REASON,
    QWEN3_MOE_IDENTITY_REASON,
    QWEN3_MOE_LAYOUT_REASON,
    QWEN3_MOE_SHARED_EXPERT_REASON,
    QWEN3_MOE_TOPOLOGY_REASON,
    UNREVIEWED_SPARSE_MODEL_REASON,
    adapter_target_modules,
    compatibility_response_v1,
    current_model_policy_snapshot,
    evaluate_model_compatibility,
    subject_from_model,
    validate_execution_path_selection,
)
from aptus.policy_snapshot import evaluate_model_policy_snapshot
from tests.aptus.helpers import (
    QWEN2_5_ACCEPTANCE_MODEL_ID,
    QWEN2_5_ACCEPTANCE_REVISION,
    make_plan,
    make_qwen2_runtime_footprint_plan,
    make_qwen3_moe_plan,
)


class ModelCompatibilityPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cls.plan = make_qwen3_moe_plan(Path(temporary))
            cls.qwen2_plan = make_qwen2_runtime_footprint_plan(Path(temporary))
        cls.subject = subject_from_model(cls.plan.model)
        cls.qwen2_subject = subject_from_model(cls.qwen2_plan.model)

    def test_qwen2_reviewed_runtime_footprint_emits_one_dense_mlx_path(
        self,
    ) -> None:
        decision = evaluate_model_compatibility(self.qwen2_subject)
        response = compatibility_response_v1(decision)

        self.assertEqual(decision.kind, ModelPolicyDecisionKind.PATH_MATCHED)
        self.assertEqual(decision.policy_id, "model.qwen2-24l.mlx-qlora")
        self.assertEqual(decision.policy_version, "1.0.0")
        self.assertEqual(decision.family, "qwen")
        self.assertEqual(
            [item.value for item in decision.reason_codes],
            ["reviewed-runtime-path", "pilot-not-yet-proven"],
        )
        self.assertEqual(len(decision.paths), 1)
        path = decision.paths[0]
        self.assertEqual(
            path.path_id,
            "mlx-lm.qlora.single.dense-causal-lm.v1",
        )
        self.assertEqual(path.adapter_profile_id.value, "dense-causal-lm.v1")
        self.assertEqual(path.method, Method.QLORA)
        self.assertEqual(path.distribution, Distribution.SINGLE)
        self.assertEqual(
            path.target_modules,
            (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        )
        self.assertEqual(
            path.required_validation_levels,
            ("model-data", "measured-preflight", "pilot"),
        )
        self.assertEqual(path.runtime_contract.training_runtime, TrainingRuntime.MLX_LM)
        self.assertEqual(path.runtime_contract.compute_backend, Backend.MPS)
        self.assertEqual(
            response,
            {
                "status": "conditional",
                "family": "qwen",
                "supported_runtime": "mlx-lm",
                "supported_methods": ["qlora"],
                "compute_backend": "mps",
                "distribution": "single",
                "evidence_requirement": "pilot-required",
                "adapter_profile_id": "dense-causal-lm.v1",
                "reason": decision.reason,
            },
        )

        subject_payload = to_primitive(self.qwen2_subject)
        self.assertNotIn("model_id", subject_payload)
        self.assertNotIn("revision", subject_payload)
        artifact_evidence = [
            record
            for record in self.qwen2_plan.evidence_records
            if record.revision == QWEN2_5_ACCEPTANCE_REVISION
        ]
        self.assertEqual(len(artifact_evidence), 1)
        self.assertIn(QWEN2_5_ACCEPTANCE_MODEL_ID, artifact_evidence[0].scope)

    def test_gemma4_dense_family_matches_qlora_and_lora_paths(self) -> None:
        for bits, layers in ((4, 60), (8, 60), (6, 35), (1, 35)):
            with self.subTest(bits=bits, layers=layers):
                subject = ModelCompatibilitySubject(
                    family="gemma4",
                    model_type="gemma4_text",
                    architecture="Gemma4ForConditionalGeneration",
                    layers=layers,
                    quantization_bits=bits,
                    quantization_layout=QuantizationLayout(bits, 64),
                    moe=None,
                )
                decision = evaluate_model_compatibility(subject)
                response = compatibility_response_v1(decision)

                self.assertEqual(decision.kind, ModelPolicyDecisionKind.PATH_MATCHED)
                self.assertEqual(decision.policy_id, GEMMA4_POLICY_ID)
                self.assertEqual(decision.family, "gemma4")
                self.assertEqual(
                    {path.path_id for path in decision.paths},
                    {
                        "mlx-lm.qlora.single.gemma4-dense.v1",
                        "mlx-lm.lora.single.gemma4-dense.v1",
                    },
                )
                self.assertEqual(response["status"], "conditional")
                self.assertEqual(response["family"], "gemma4")
                self.assertEqual(response["supported_methods"], ["qlora", "lora"])
                self.assertEqual(response["supported_runtime"], "mlx-lm")

    def test_gemma4_moe_topology_stays_blocked(self) -> None:
        subject = ModelCompatibilitySubject(
            family="gemma4",
            model_type="gemma4_text",
            architecture="Gemma4ForConditionalGeneration",
            layers=60,
            quantization_bits=4,
            quantization_layout=QuantizationLayout(4, 64),
            moe=self.subject.moe,
        )
        decision = evaluate_model_compatibility(subject)
        self.assertEqual(decision.kind, ModelPolicyDecisionKind.BLOCKED)
        self.assertEqual(decision.policy_id, GEMMA4_POLICY_ID)
        self.assertEqual(
            [item.value for item in decision.reason_codes],
            ["dense-topology-required"],
        )

    def test_qwen2_layer_mismatch_stays_blocked_without_operator_confirm(
        self,
    ) -> None:
        subject = replace(self.qwen2_subject, layers=28)
        decision = evaluate_model_compatibility(subject)
        self.assertEqual(decision.kind, ModelPolicyDecisionKind.BLOCKED)
        self.assertEqual(
            [item.value for item in decision.reason_codes],
            ["layer-count-mismatch"],
        )

    def test_qwen2_layer_mismatch_becomes_unreviewed_path_when_confirmed(
        self,
    ) -> None:
        subject = replace(self.qwen2_subject, layers=28)
        decision = evaluate_model_compatibility(
            subject,
            confirm_unreviewed_runtime=True,
        )
        self.assertEqual(decision.kind, ModelPolicyDecisionKind.PATH_MATCHED)
        self.assertEqual(
            [item.value for item in decision.reason_codes],
            [
                "unreviewed-runtime-operator-attested",
                "pilot-not-yet-proven",
            ],
        )
        self.assertNotIn(
            "reviewed-runtime-path",
            [item.value for item in decision.reason_codes],
        )
        self.assertIn("not the reviewed 24-layer", decision.reason)
        self.assertEqual(len(decision.paths), 1)
        self.assertEqual(decision.paths[0].method, Method.QLORA)

    def test_confirm_does_not_open_layout_or_moe_mismatches(self) -> None:
        layout = evaluate_model_compatibility(
            replace(
                self.qwen2_subject,
                layers=28,
                quantization_layout=QuantizationLayout(4, 128),
            ),
            confirm_unreviewed_runtime=True,
        )
        self.assertEqual(layout.kind, ModelPolicyDecisionKind.BLOCKED)
        moe = evaluate_model_compatibility(
            replace(self.qwen2_subject, layers=28, moe=self.subject.moe),
            confirm_unreviewed_runtime=True,
        )
        self.assertEqual(moe.kind, ModelPolicyDecisionKind.BLOCKED)

    def test_qwen2_host_and_portable_policy_have_exact_mutation_parity(
        self,
    ) -> None:
        assert self.subject.moe is not None
        cases = {
            "exact reviewed runtime footprint": (
                self.qwen2_subject,
                ModelPolicyDecisionKind.PATH_MATCHED,
                "model.qwen2-24l.mlx-qlora",
                ("reviewed-runtime-path", "pilot-not-yet-proven"),
            ),
            "family": (
                replace(self.qwen2_subject, family="llama"),
                ModelPolicyDecisionKind.BLOCKED,
                "model.qwen2-24l.mlx-qlora",
                ("identity-mismatch",),
            ),
            "model type": (
                replace(self.qwen2_subject, model_type="qwen3"),
                ModelPolicyDecisionKind.BLOCKED,
                "model.qwen2-24l.mlx-qlora",
                ("identity-mismatch",),
            ),
            "architecture": (
                replace(
                    self.qwen2_subject,
                    architecture="Qwen3ForCausalLM",
                ),
                ModelPolicyDecisionKind.BLOCKED,
                "model.qwen2-24l.mlx-qlora",
                ("identity-mismatch",),
            ),
            "layers": (
                replace(self.qwen2_subject, layers=25),
                ModelPolicyDecisionKind.BLOCKED,
                "model.qwen2-24l.mlx-qlora",
                ("layer-count-mismatch",),
            ),
            "quantization layout": (
                replace(
                    self.qwen2_subject,
                    quantization_layout=QuantizationLayout(4, 128),
                ),
                ModelPolicyDecisionKind.BLOCKED,
                "model.qwen2-24l.mlx-qlora",
                ("quantization-layout-mismatch",),
            ),
            "quantization bits": (
                replace(self.qwen2_subject, quantization_bits=8),
                ModelPolicyDecisionKind.BLOCKED,
                "model.qwen2-24l.mlx-qlora",
                ("four-bit-required",),
            ),
            "dense topology": (
                replace(self.qwen2_subject, moe=self.subject.moe),
                ModelPolicyDecisionKind.BLOCKED,
                "model.qwen2-24l.mlx-qlora",
                ("dense-topology-required",),
            ),
            "invalid facts": (
                replace(
                    self.qwen2_subject,
                    fact_errors=("quantization: conflicting provider facts",),
                ),
                ModelPolicyDecisionKind.BLOCKED,
                "model.qwen2-24l.mlx-qlora",
                ("invalid-compatibility-facts",),
            ),
            "unclaimed dense qwen variant": (
                replace(
                    self.qwen2_subject,
                    model_type="qwen3",
                    architecture="Qwen3ForCausalLM",
                ),
                ModelPolicyDecisionKind.FAMILY_RECOGNIZED,
                None,
                ("family-recognized",),
            ),
            "unclaimed sparse qwen variant": (
                replace(
                    self.qwen2_subject,
                    architecture="Qwen2MoeForCausalLM",
                    layers=32,
                    quantization_bits=None,
                    quantization_layout=None,
                ),
                ModelPolicyDecisionKind.BLOCKED,
                None,
                ("unreviewed-sparse-model",),
            ),
        }
        snapshot = current_model_policy_snapshot()
        for name, (
            subject,
            expected_kind,
            expected_policy_id,
            expected_reason_codes,
        ) in cases.items():
            with self.subTest(name=name):
                portable = evaluate_model_policy_snapshot(
                    snapshot,
                    to_primitive(subject),
                )
                host = to_primitive(evaluate_model_compatibility(subject))
                self.assertEqual(portable, host)
                self.assertEqual(host["kind"], expected_kind.value)
                self.assertEqual(host["policy_id"], expected_policy_id)
                self.assertEqual(host["reason_codes"], list(expected_reason_codes))
                if expected_kind == ModelPolicyDecisionKind.BLOCKED:
                    self.assertEqual(host["paths"], [])

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

    def test_host_and_snapshot_evaluators_have_exact_decision_parity(self) -> None:
        assert self.subject.moe is not None
        assert self.subject.quantization_layout is not None
        cases = (
            self.subject,
            replace(self.subject, architecture="OtherForCausalLM"),
            replace(self.subject, quantization_layout=QuantizationLayout(4, 64)),
            replace(self.subject, moe=None),
            replace(
                self.subject,
                moe=replace(
                    self.subject.moe,
                    shared_expert_intermediate_size=1024,
                ),
            ),
            replace(self.subject, quantization_bits=8),
            replace(
                self.subject,
                family="llama",
                model_type="llama",
                architecture="LlamaForCausalLM",
                quantization_layout=None,
                moe=None,
            ),
            replace(
                self.subject,
                family="custom",
                model_type="custom",
                architecture="CustomForCausalLM",
                quantization_layout=None,
                moe=None,
            ),
            replace(
                self.subject,
                family="mixtral_custom",
                model_type="custom",
                architecture="CustomForCausalLM",
                quantization_layout=None,
                moe=None,
            ),
        )
        snapshot = current_model_policy_snapshot()
        for subject in cases:
            with self.subTest(family=subject.family, architecture=subject.architecture):
                self.assertEqual(
                    evaluate_model_policy_snapshot(snapshot, to_primitive(subject)),
                    to_primitive(evaluate_model_compatibility(subject)),
                )

    def test_host_and_snapshot_evaluators_have_exact_fact_error_parity(
        self,
    ) -> None:
        dense = replace(
            self.subject,
            family="llama",
            model_type="llama",
            architecture="LlamaForCausalLM",
            quantization_layout=None,
            moe=None,
        )
        unknown = replace(
            dense,
            family="custom",
            model_type="custom",
            architecture="CustomForCausalLM",
        )
        sparse = replace(
            unknown,
            family="mixtral_custom",
            model_type="mixtral",
            architecture="MixtralForCausalLM",
        )
        cases = {
            "exact Qwen": (
                replace(
                    self.subject,
                    fact_errors=("quantization: contradictory",),
                ),
                ["invalid-compatibility-facts"],
            ),
            "Qwen identity near-match": (
                replace(
                    self.subject,
                    model_type="not_qwen",
                    architecture="NotQwen",
                    fact_errors=("quantization: contradictory",),
                ),
                ["identity-mismatch"],
            ),
            "dense": (
                replace(
                    dense,
                    fact_errors=("quantization: contradictory",),
                ),
                ["invalid-compatibility-facts"],
            ),
            "sparse": (
                replace(
                    sparse,
                    fact_errors=("quantization: contradictory",),
                ),
                ["invalid-compatibility-facts"],
            ),
            "unknown": (
                replace(
                    unknown,
                    fact_errors=("quantization: contradictory",),
                ),
                ["invalid-compatibility-facts"],
            ),
            "unsorted multi-error": (
                replace(
                    self.subject,
                    fact_errors=(
                        "quantization: conflicting provider facts",
                        "moe: conflicting provider facts",
                    ),
                ),
                ["invalid-compatibility-facts"],
            ),
        }
        snapshot = current_model_policy_snapshot()
        for name, (subject, expected_reason_codes) in cases.items():
            with self.subTest(name=name):
                portable = evaluate_model_policy_snapshot(
                    snapshot, to_primitive(subject)
                )
                self.assertEqual(
                    portable,
                    to_primitive(evaluate_model_compatibility(subject)),
                )
                self.assertEqual(portable["reason_codes"], expected_reason_codes)

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
        for family in ("qwen3_moe", "qwen"):
            with self.subTest(family=family):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import aptus.catalog as catalog; "
                            f"catalog.TARGET_MODULES[{family!r}] = ('q_proj',); "
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

    def test_qwen_v5_identities_are_deterministic_and_policy_bound(self) -> None:
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

    def test_dense_cuda_v5_identities_are_deterministic_and_decision_bound(
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
