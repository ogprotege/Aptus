import json
import unittest
from urllib.error import URLError

from aptus.inspection import (
    _compatibility_subject,
    inspect_huggingface_model,
)
from aptus.model_compatibility import (
    compatibility_response_v1,
    evaluate_model_compatibility,
)
from tests.aptus.helpers import (
    QWEN2_5_ACCEPTANCE_MODEL_ID,
    QWEN2_5_ACCEPTANCE_REVISION,
    QWEN3_MOE_MODEL_ID,
    QWEN3_MOE_REVISION,
    qwen3_moe_quantization_config,
)


class FakeResponse:
    def __init__(self, value, headers=None):
        self.payload = json.dumps(value).encode("utf-8")
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit):
        return self.payload[:limit]


class SequenceTransport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request.full_url, timeout))
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value


class ModelInspectionTests(unittest.TestCase):
    def test_resolves_immutable_provider_facts_without_guessing_size_or_permission(
        self,
    ) -> None:
        commit = "b" * 40
        transport = SequenceTransport(
            [
                FakeResponse(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "hidden_size": 4096,
                        "intermediate_size": 11008,
                        "num_hidden_layers": 32,
                        "max_position_embeddings": 4096,
                        "num_attention_heads": 32,
                    },
                    {"X-Repo-Commit": commit},
                ),
                FakeResponse(
                    {"cardData": {"license": "apache-2.0"}}, {"X-Repo-Commit": commit}
                ),
            ]
        )
        result = inspect_huggingface_model("org/model", "main", transport=transport)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["resolved_revision"], commit)
        self.assertEqual(result["facts"]["architecture"], "LlamaForCausalLM")
        self.assertEqual(result["facts"]["architectures"], ["LlamaForCausalLM"])
        self.assertEqual(result["facts"]["model_type"], "llama")
        self.assertEqual(result["facts"]["family"], "llama")
        self.assertEqual(result["facts"]["layers"], 32)
        self.assertEqual(result["facts"]["license_name"], "apache-2.0")
        self.assertIsNone(result["facts"]["parameters"])
        self.assertIsNone(result["facts"]["training_allowed"])
        self.assertEqual(
            result["explicit_user_facts_required"], ["parameters", "training_allowed"]
        )
        self.assertEqual(result["compatibility"]["status"], "recognized")
        self.assertEqual(
            transport.requests,
            [
                (
                    "https://huggingface.co/org/model/resolve/main/config.json",
                    10.0,
                ),
                (
                    f"https://huggingface.co/api/models/org/model/revision/{commit}",
                    10.0,
                ),
            ],
        )
        self.assertEqual(
            result["provenance"]["license_name"]["source"],
            f"https://huggingface.co/api/models/org/model/revision/{commit}",
        )
        self.assertEqual(result["provenance"]["family"]["kind"], "inferred")

        receipt = result["inspection_receipt"]
        self.assertEqual(receipt["schema_version"], "aptus.model-inspection-receipt.v1")
        self.assertTrue(receipt["receipt_id"].startswith("receipt_"))
        self.assertEqual(receipt["model_id"], "org/model")
        self.assertEqual(receipt["resolved_revision"], commit)
        self.assertEqual(len(receipt["decision"]["subject_facts_sha256"]), 64)
        receipt_fields = {item["field"]: item for item in receipt["provenance_summary"]}
        self.assertEqual(
            set(receipt_fields),
            {
                "architecture",
                "context_length",
                "family",
                "hidden_size",
                "intermediate_size",
                "layers",
                "license_name",
                "model_type",
            },
        )
        self.assertEqual(receipt_fields["family"]["kind"], "inferred")
        self.assertEqual(
            receipt_fields["license_name"]["source"],
            f"https://huggingface.co/api/models/org/model/revision/{commit}",
        )
        for excluded in (
            "architectures",
            "attention_heads",
            "key_value_heads",
            "parameters",
            "training_allowed",
            "vocab_size",
        ):
            self.assertNotIn(excluded, receipt_fields)

    def test_fallback_architecture_and_family_are_inferred_and_license_cites_config(
        self,
    ) -> None:
        commit = "a" * 40
        config_url = f"https://huggingface.co/org/model/resolve/{commit}/config.json"
        transport = SequenceTransport(
            [
                FakeResponse(
                    {
                        "model_type": "llama",
                        "license": "apache-2.0",
                        "num_hidden_layers": 32,
                    },
                    {"X-Repo-Commit": commit},
                ),
                FakeResponse({}, {"X-Repo-Commit": commit}),
            ]
        )

        result = inspect_huggingface_model("org/model", "main", transport=transport)

        self.assertEqual(result["facts"]["architecture"], "llama")
        self.assertEqual(result["provenance"]["architecture"]["kind"], "inferred")
        self.assertIn(
            "architecture fallback",
            result["provenance"]["architecture"]["source"],
        )
        self.assertEqual(result["provenance"]["family"]["kind"], "inferred")
        self.assertEqual(result["provenance"]["license_name"]["source"], config_url)
        receipt_provenance = {
            item["field"]: item
            for item in result["inspection_receipt"]["provenance_summary"]
        }
        self.assertEqual(receipt_provenance["architecture"]["kind"], "inferred")
        self.assertEqual(receipt_provenance["family"]["kind"], "inferred")
        self.assertEqual(receipt_provenance["license_name"]["source"], config_url)

    def test_normalizes_only_exact_dense_aliases_and_keeps_raw_evidence(self) -> None:
        cases = (
            ("qwen2", "Qwen2ForCausalLM", "qwen"),
            ("qwen3", "Qwen3ForCausalLM", "qwen"),
            ("gemma2", "Gemma2ForCausalLM", "gemma"),
            ("gemma3", "Gemma3ForCausalLM", "gemma"),
            ("gemma3_text", "Gemma3ForCausalLM", "gemma"),
        )
        for model_type, architecture, expected_family in cases:
            with self.subTest(model_type=model_type):
                commit = "c" * 40
                transport = SequenceTransport(
                    [
                        FakeResponse(
                            {
                                "model_type": model_type,
                                "architectures": [architecture],
                            },
                            {"X-Repo-Commit": commit},
                        ),
                        FakeResponse({}, {"X-Repo-Commit": commit}),
                    ]
                )

                result = inspect_huggingface_model(
                    "org/model", "main", transport=transport
                )

                self.assertEqual(result["facts"]["family"], expected_family)
                self.assertEqual(result["facts"]["model_type"], model_type)
                self.assertEqual(result["facts"]["architecture"], architecture)
                self.assertEqual(result["facts"]["architectures"], [architecture])
                self.assertEqual(
                    result["provenance"]["model_type"]["kind"], "provider-declared"
                )
                self.assertEqual(
                    result["provenance"]["architectures"]["kind"],
                    "provider-declared",
                )
                self.assertEqual(result["provenance"]["family"]["kind"], "inferred")
                self.assertTrue(
                    any("was normalized" in warning for warning in result["warnings"])
                )

    def test_does_not_prefix_map_moe_or_multimodal_provider_types(self) -> None:
        cases = (
            ("qwen2_moe", "Qwen2MoeForCausalLM"),
            ("qwen3_moe", "Qwen3MoeModel"),
            ("gemma3", "Gemma3Model"),
            ("gemma3", "Gemma3ForConditionalGeneration"),
        )
        for model_type, architecture in cases:
            with self.subTest(model_type=model_type, architecture=architecture):
                commit = "d" * 40
                transport = SequenceTransport(
                    [
                        FakeResponse(
                            {
                                "model_type": model_type,
                                "architectures": [architecture],
                            },
                            {"X-Repo-Commit": commit},
                        ),
                        FakeResponse({}, {"X-Repo-Commit": commit}),
                    ]
                )

                result = inspect_huggingface_model(
                    "org/model", "main", transport=transport
                )

                self.assertEqual(result["facts"]["family"], model_type)
                self.assertEqual(result["facts"]["model_type"], model_type)
                self.assertEqual(result["compatibility"]["status"], "unsupported")
                self.assertIsNone(result["compatibility"]["supported_runtime"])

    def test_exact_four_bit_qwen3_moe_is_conditionally_supported(self) -> None:
        commit = QWEN3_MOE_REVISION
        transport = SequenceTransport(
            [
                FakeResponse(
                    {
                        "model_type": "qwen3_moe",
                        "architectures": ["Qwen3MoeForCausalLM"],
                        "hidden_size": 2048,
                        "intermediate_size": 6144,
                        "num_hidden_layers": 48,
                        "max_position_embeddings": 262144,
                        "num_experts": 128,
                        "num_experts_per_tok": 8,
                        "moe_intermediate_size": 768,
                        "decoder_sparse_step": 1,
                        "mlp_only_layers": [],
                        "quantization": qwen3_moe_quantization_config(),
                    },
                    {"X-Repo-Commit": commit},
                ),
                FakeResponse(
                    {"cardData": {"license": "apache-2.0"}},
                    {"X-Repo-Commit": commit},
                ),
            ]
        )

        result = inspect_huggingface_model(
            QWEN3_MOE_MODEL_ID, "main", transport=transport
        )

        self.assertEqual(result["facts"]["family"], "qwen3_moe")
        self.assertEqual(result["facts"]["quantization_bits"], 4)
        self.assertEqual(
            len(result["facts"]["quantization_layout"]["module_overrides"]),
            48,
        )
        self.assertEqual(
            result["facts"]["moe"],
            {
                "expert_count": 128,
                "experts_per_token": 8,
                "expert_intermediate_size": 768,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "shared_expert_intermediate_size": None,
            },
        )
        self.assertEqual(
            result["compatibility"],
            {
                "status": "conditional",
                "family": "qwen3_moe",
                "supported_runtime": "mlx-lm",
                "supported_methods": ["qlora"],
                "compute_backend": "mps",
                "distribution": "single",
                "evidence_requirement": "pilot-required",
                "adapter_profile_id": "attention-qkvo.v1",
                "reason": (
                    "The model identity, mixed-precision layout, routed-expert "
                    "topology, and attention-only q/k/v/o target policy match the "
                    "reviewed Qwen3 MoE slice. "
                    "Measured preflight and a real-model pilot remain mandatory."
                ),
            },
        )
        for field in (
            "supported_runtime",
            "compute_backend",
            "distribution",
            "adapter_profile_id",
        ):
            self.assertIs(type(result["compatibility"][field]), str)
        self.assertTrue(
            all(
                type(method) is str
                for method in result["compatibility"]["supported_methods"]
            )
        )
        subject = _compatibility_subject(
            family=result["facts"]["family"],
            model_type=result["facts"]["model_type"],
            architecture=result["facts"]["architecture"],
            layers=result["facts"]["layers"],
            quantization_bits=result["facts"]["quantization_bits"],
            quantization_layout=result["facts"]["quantization_layout"],
            quantization_error=None,
            moe=result["facts"]["moe"],
            moe_error=None,
        )
        self.assertEqual(
            compatibility_response_v1(evaluate_model_compatibility(subject)),
            result["compatibility"],
        )

    def test_reviewed_qwen2_runtime_footprint_is_conditionally_supported(
        self,
    ) -> None:
        transport = SequenceTransport(
            [
                FakeResponse(
                    {
                        "model_type": "qwen2",
                        "architectures": ["Qwen2ForCausalLM"],
                        "hidden_size": 896,
                        "intermediate_size": 4864,
                        "num_hidden_layers": 24,
                        "max_position_embeddings": 32768,
                        "num_attention_heads": 14,
                        "num_key_value_heads": 2,
                        "vocab_size": 151936,
                        "quantization": {"bits": 4, "group_size": 64},
                    },
                    {"X-Repo-Commit": QWEN2_5_ACCEPTANCE_REVISION},
                ),
                FakeResponse(
                    {"cardData": {"license": "apache-2.0"}},
                    {"X-Repo-Commit": QWEN2_5_ACCEPTANCE_REVISION},
                ),
            ]
        )

        result = inspect_huggingface_model(
            QWEN2_5_ACCEPTANCE_MODEL_ID,
            "main",
            transport=transport,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["facts"]["family"], "qwen")
        self.assertEqual(
            result["facts"]["quantization_layout"],
            {
                "default_bits": 4,
                "default_group_size": 64,
                "module_overrides": [],
            },
        )
        self.assertEqual(
            result["compatibility"]["adapter_profile_id"],
            "dense-causal-lm.v1",
        )
        self.assertEqual(result["compatibility"]["supported_methods"], ["qlora"])
        receipt = result["inspection_receipt"]
        self.assertEqual(
            receipt["decision"]["policy_id"],
            "model.qwen2-24l.mlx-qlora",
        )
        self.assertTrue(receipt["provenance_requirement_met"])
        receipt_fields = {item["field"] for item in receipt["provenance_summary"]}
        self.assertNotIn("moe", receipt_fields)
        self.assertTrue(
            {
                "architecture",
                "layers",
                "model_type",
                "quantization_bits",
                "quantization_layout",
            }.issubset(receipt_fields)
        )

    def test_dense_family_with_sparse_topology_does_not_bypass_policy(self) -> None:
        commit = "f" * 40
        transport = SequenceTransport(
            [
                FakeResponse(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "num_hidden_layers": 2,
                        "num_experts": 8,
                        "num_experts_per_tok": 2,
                        "moe_intermediate_size": 256,
                        "decoder_sparse_step": 1,
                        "mlp_only_layers": [],
                    },
                    {"X-Repo-Commit": commit},
                ),
                FakeResponse({}, {"X-Repo-Commit": commit}),
            ]
        )

        result = inspect_huggingface_model(
            "org/model",
            "main",
            transport=transport,
        )

        self.assertEqual(result["facts"]["family"], "llama")
        self.assertEqual(result["compatibility"]["status"], "unsupported")

    def test_sparse_identity_without_topology_does_not_bypass_policy(self) -> None:
        cases = (
            ("qwen2", "Qwen2MoeForCausalLM", "qwen"),
            ("mistral", "MixtralForCausalLM", "mistral"),
        )

        for model_type, architecture, expected_family in cases:
            with self.subTest(architecture=architecture):
                commit = "f" * 40
                transport = SequenceTransport(
                    [
                        FakeResponse(
                            {
                                "model_type": model_type,
                                "architectures": [architecture],
                                "num_hidden_layers": 32,
                            },
                            {"X-Repo-Commit": commit},
                        ),
                        FakeResponse({}, {"X-Repo-Commit": commit}),
                    ]
                )

                result = inspect_huggingface_model(
                    "org/model",
                    "main",
                    transport=transport,
                )

                self.assertEqual(result["facts"]["family"], expected_family)
                self.assertEqual(
                    result["compatibility"]["status"],
                    "unsupported",
                )

    def test_qwen_near_identity_with_fact_error_is_not_labeled_exact(self) -> None:
        commit = "f" * 40
        transport = SequenceTransport(
            [
                FakeResponse(
                    {
                        "model_type": "qwen3_moe",
                        "architectures": ["NotQwen"],
                        "num_hidden_layers": 32,
                        "quantization": {"bits": "four", "group_size": 64},
                    },
                    {"X-Repo-Commit": commit},
                ),
                FakeResponse({}, {"X-Repo-Commit": commit}),
            ]
        )

        result = inspect_huggingface_model(
            "org/model",
            "main",
            transport=transport,
        )

        self.assertEqual(result["compatibility"]["status"], "unsupported")
        self.assertEqual(
            result["compatibility"]["reason"],
            "No exact Aptus model-family compatibility policy matches this "
            "provider model type and architecture.",
        )
        self.assertNotIn(
            "exact Qwen3 MoE identity was recognized",
            result["compatibility"]["reason"],
        )

    def test_qwen3_moe_inspection_rejects_malformed_topology(self) -> None:
        config = {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 48,
            "max_position_embeddings": 262144,
            "num_experts": 8,
            "num_experts_per_tok": 9,
            "moe_intermediate_size": 768,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "quantization": qwen3_moe_quantization_config(),
        }
        transport = SequenceTransport(
            [
                FakeResponse(config, {"X-Repo-Commit": QWEN3_MOE_REVISION}),
                FakeResponse({}, {"X-Repo-Commit": QWEN3_MOE_REVISION}),
            ]
        )

        result = inspect_huggingface_model(
            QWEN3_MOE_MODEL_ID, "main", transport=transport
        )

        self.assertEqual(result["compatibility"]["status"], "unsupported")
        self.assertTrue(
            any("cannot exceed" in item for item in result["warnings"]),
            result["warnings"],
        )

    def test_qwen3_moe_inspection_rejects_layout_override_drift(self) -> None:
        quantization = qwen3_moe_quantization_config()
        quantization.pop("model.layers.47.mlp.gate")
        config = {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 48,
            "max_position_embeddings": 262144,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "quantization": quantization,
        }
        transport = SequenceTransport(
            [
                FakeResponse(config, {"X-Repo-Commit": QWEN3_MOE_REVISION}),
                FakeResponse({}, {"X-Repo-Commit": QWEN3_MOE_REVISION}),
            ]
        )

        result = inspect_huggingface_model(
            QWEN3_MOE_MODEL_ID, "main", transport=transport
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["compatibility"]["status"], "unsupported")

    def test_alias_normalization_fails_closed_if_catalog_policy_changes(self) -> None:
        from unittest.mock import patch

        commit = "e" * 40
        transport = SequenceTransport(
            [
                FakeResponse(
                    {
                        "model_type": "qwen2",
                        "architectures": ["Qwen2ForCausalLM"],
                    },
                    {"X-Repo-Commit": commit},
                ),
                FakeResponse({}, {"X-Repo-Commit": commit}),
            ]
        )
        with patch("aptus.inspection.TARGET_MODULES", {"qwen": ("q_proj",)}):
            result = inspect_huggingface_model("org/model", "main", transport=transport)

        self.assertEqual(result["facts"]["family"], "qwen2")
        self.assertTrue(
            any("catalog policy has changed" in item for item in result["warnings"])
        )

    def test_missing_immutable_commit_fails_closed(self) -> None:
        transport = SequenceTransport([FakeResponse({"model_type": "llama"})])
        result = inspect_huggingface_model("org/model", "main", transport=transport)
        self.assertEqual(result["status"], "unsupported")

    def test_mutable_ref_rejects_body_asserted_commit_hash(self) -> None:
        transport = SequenceTransport(
            [FakeResponse({"model_type": "llama", "_commit_hash": "a" * 40})]
        )

        result = inspect_huggingface_model("org/model", "main", transport=transport)

        self.assertEqual(result["status"], "unsupported")
        self.assertIn("immutable commit", result["error"])

    def test_headerless_immutable_requested_revision_remains_bound(self) -> None:
        commit = "a" * 40
        transport = SequenceTransport(
            [
                FakeResponse(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "num_hidden_layers": 32,
                    }
                ),
                FakeResponse({}),
            ]
        )

        result = inspect_huggingface_model("org/model", commit, transport=transport)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["resolved_revision"], commit)
        self.assertTrue(
            all(
                item["source"].endswith(f"/resolve/{commit}/config.json")
                or item["source"].startswith("Aptus ")
                for item in result["provenance"].values()
            )
        )

    def test_network_failure_returns_typed_unavailable_result(self) -> None:
        result = inspect_huggingface_model(
            "org/model",
            "main",
            transport=SequenceTransport([URLError("offline")]),
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("offline", result["error"])


if __name__ == "__main__":
    unittest.main()
