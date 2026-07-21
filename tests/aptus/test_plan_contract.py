import copy
import unittest

from aptus.plan_contract import validate_plan_payload


def valid_payload() -> dict:
    memory = {
        "base_weights_bytes": 1,
        "quantization_metadata_bytes": 0,
        "adapter_weights_bytes": 1,
        "adapter_gradients_bytes": 1,
        "optimizer_states_bytes": 1,
        "activations_bytes": 1,
        "temporary_overhead_bytes": 1,
        "safety_margin_bytes": 1,
        "estimated_peak_bytes": 7,
    }
    candidate = {
        "method": "lora",
        "feasible": True,
        "rejection_reasons": [],
        "precision": "bf16",
        "quantization": None,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 8,
        "rank": 16,
        "alpha": 32,
        "learning_rate": 0.0002,
        "target_modules": ["q_proj"],
        "memory": memory,
        "preference_score": 100.0,
        "confidence": "low-until-calibrated",
        "assumptions": ["heuristic"],
        "evidence": ["source"],
    }
    return {
        "schema_version": "aptus.training-plan.v1",
        "model": {
            "model_id": "example/model",
            "revision": "a" * 40,
            "family": "llama",
            "parameters": 1_000_000_000,
            "hidden_size": 1024,
            "layers": 16,
            "context_length": 2048,
            "license_name": "apache-2.0",
            "training_allowed": True,
        },
        "dataset": {
            "source_path": "/tmp/data.jsonl",
            "source_sha256": "b" * 64,
            "source_format": "jsonl",
            "schema_name": "text",
            "example_count": 10,
            "total_estimated_tokens": 100,
            "sequence_p50": 8,
            "sequence_p95": 12,
            "sequence_max": 16,
            "measurement": "estimated",
            "warnings": [],
        },
        "hardware": {
            "devices": [
                {
                    "name": "GPU 0",
                    "backend": "cuda",
                    "total_vram_bytes": 24 * 1024**3,
                    "supports_bf16": True,
                    "supports_4bit": True,
                }
            ],
            "host_ram_bytes": 64 * 1024**3,
            "reserve_per_device_bytes": 2 * 1024**3,
        },
        "target": {
            "objective": "quality",
            "sequence_length": 128,
            "effective_batch_size": 8,
            "max_epochs": 3,
            "method_preference": None,
        },
        "recommended": candidate,
        "candidates": [candidate],
        "warnings": ["uncalibrated"],
        "recommendation_rationale": ["LoRA fits and quality prefers LoRA."],
    }


class PlanContractTests(unittest.TestCase):
    def test_rejects_falsey_and_non_object_payloads(self) -> None:
        for value in ({}, None, [], ""):
            with self.subTest(value=value):
                self.assertTrue(validate_plan_payload(value, verify_dataset=False))

    def test_accepts_complete_valid_payload(self) -> None:
        self.assertEqual(
            validate_plan_payload(valid_payload(), verify_dataset=False),
            (),
        )

    def test_rejects_unsafe_cross_field_values(self) -> None:
        mutations = [
            ("training permission", ("model", "training_allowed"), False),
            ("immutable revision", ("model", "revision"), "main"),
            ("license", ("model", "license_name"), ""),
            ("devices", ("hardware", "devices"), []),
            ("epochs", ("target", "max_epochs"), 0),
            ("micro batch", ("recommended", "micro_batch_size"), -1),
            (
                "effective batch",
                ("recommended", "effective_batch_size"),
                16,
            ),
        ]
        for expected, path, value in mutations:
            with self.subTest(path=path):
                payload = copy.deepcopy(valid_payload())
                payload[path[0]][path[1]] = value
                errors = validate_plan_payload(payload, verify_dataset=False)
                self.assertTrue(
                    any(expected in error.lower() for error in errors),
                    errors,
                )

    def test_rejects_candidate_that_exceeds_per_device_vram(self) -> None:
        payload = valid_payload()
        memory = payload["recommended"]["memory"]
        memory["base_weights_bytes"] = 30 * 1024**3
        memory["estimated_peak_bytes"] = sum(
            value
            for key, value in memory.items()
            if key.endswith("_bytes") and key != "estimated_peak_bytes"
        )

        errors = validate_plan_payload(payload, verify_dataset=False)

        self.assertTrue(
            any("usable per-device vram" in error.lower() for error in errors),
            errors,
        )

    def test_rejects_unsupported_precision_quantization_and_modules(self) -> None:
        cases = []

        no_four_bit = valid_payload()
        no_four_bit["recommended"]["method"] = "qlora"
        no_four_bit["recommended"]["quantization"] = "nf4-double-quant"
        no_four_bit["hardware"]["devices"][0]["supports_4bit"] = False
        cases.append(("4-bit", no_four_bit))

        no_bf16 = valid_payload()
        no_bf16["hardware"]["devices"][0]["supports_bf16"] = False
        cases.append(("bf16", no_bf16))

        wrong_module = valid_payload()
        wrong_module["recommended"]["target_modules"] = ["not_a_module"]
        cases.append(("target module", wrong_module))

        for expected, payload in cases:
            with self.subTest(expected=expected):
                errors = validate_plan_payload(payload, verify_dataset=False)
                self.assertTrue(
                    any(expected in error.lower() for error in errors),
                    errors,
                )


if __name__ == "__main__":
    unittest.main()
