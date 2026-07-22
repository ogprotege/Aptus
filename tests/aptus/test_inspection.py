import json
import unittest
from urllib.error import URLError

from aptus.inspection import inspect_huggingface_model


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
            ("qwen3_moe", "Qwen3MoeForCausalLM"),
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
