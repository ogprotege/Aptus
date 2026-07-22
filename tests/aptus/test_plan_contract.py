import copy
import tempfile
import unittest
from pathlib import Path

from aptus.domain import (
    Backend,
    Objective,
    TrainingRuntime,
    TrainingTarget,
    to_primitive,
)
from aptus.methods import selectable_method_descriptors
from aptus.plan_contract import (
    RUNTIME_BINDING_IDENTITIES,
    candidate_id_for_payload,
    validate_plan_payload,
)
from aptus.planning import plan_training
from aptus.profiling import (
    build_hardware_spec,
    build_model_spec,
    profile_dataset,
)

from tests.aptus.helpers import make_plan


class PlanContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.payload = to_primitive(make_plan(self.root))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_real_v2_plan_is_valid(self) -> None:
        self.assertEqual(validate_plan_payload(self.payload, verify_dataset=True), ())

    def test_embedded_runtime_identities_match_the_registered_bindings(self) -> None:
        registered = {
            (
                descriptor.method_id,
                binding.training_runtime,
                binding.compute_backend,
            ): (
                binding.compiler_id,
                binding.estimator_id,
                binding.export_kind,
                binding.evidence_requirement,
            )
            for descriptor in selectable_method_descriptors()
            for binding in descriptor.runtime_bindings
        }
        self.assertEqual(RUNTIME_BINDING_IDENTITIES, registered)

    def test_rejects_falsey_and_non_object_payloads(self) -> None:
        for value in ({}, None, [], ""):
            self.assertTrue(validate_plan_payload(value, verify_dataset=False))

    def test_rejects_formula_or_candidate_identity_tampering(self) -> None:
        for mutate in (
            lambda value: value.update(formula_version="hidden-multiplier-v1"),
            lambda value: value["candidates"][0].update(candidate_id="cand_tampered"),
        ):
            value = copy.deepcopy(self.payload)
            mutate(value)
            self.assertTrue(validate_plan_payload(value, verify_dataset=False))

    def test_rejects_plan_id_tampering(self) -> None:
        value = copy.deepcopy(self.payload)
        value["plan_id"] = "plan_" + "0" * 20
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("Plan immutable ID" in item for item in errors))

    def test_candidate_identity_binds_learning_rate(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        candidate["learning_rate"] *= 2
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("normalized execution contract" in item for item in errors))

    def test_candidate_identity_binds_target_modules(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = next(item for item in value["candidates"] if item["target_modules"])
        candidate["target_modules"] = [*candidate["target_modules"], "lm_head"]
        if candidate["candidate_id"] == value["recommended"]["candidate_id"]:
            value["recommended"] = copy.deepcopy(candidate)
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("normalized execution contract" in item for item in errors))

    def test_plan_identity_catches_consistent_memory_recomputation(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        original_candidate_id = candidate["candidate_id"]
        delta = 4096
        memory = candidate["memory"]
        memory["base_weights_bytes"] += delta
        memory["point_estimate_bytes"] += delta
        memory["estimated_peak_bytes"] += delta
        memory["component_upper_bounds"]["base_weights_bytes"] += delta
        memory["upper_estimate_bytes"] += delta
        candidate["candidate_id"] = candidate_id_for_payload(
            candidate,
            model=value["model"],
            dataset=value["dataset"],
            hardware=value["hardware"],
            target=value["target"],
        )
        if value["recommended"]["candidate_id"] == original_candidate_id:
            value["recommended"] = copy.deepcopy(candidate)

        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertFalse(
            any(
                "point memory" in item or "component upper bounds" in item
                for item in errors
            )
        )
        self.assertFalse(any("candidate ID" in item for item in errors))
        self.assertTrue(any("Plan immutable ID" in item for item in errors))

    def test_candidate_identity_binds_normalized_input_facts(self) -> None:
        value = copy.deepcopy(self.payload)
        value["target"]["checkpoint_steps"] += 1
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("normalized execution contract" in item for item in errors))
        self.assertTrue(any("Plan immutable ID" in item for item in errors))

    def test_rejects_global_batch_tampering(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        candidate["effective_batch_size"] += 1
        if candidate["candidate_id"] == value["recommended"]["candidate_id"]:
            value["recommended"] = copy.deepcopy(candidate)
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("batch arithmetic" in item for item in errors))

    def test_rejects_hidden_or_inconsistent_upper_bound(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        candidate["memory"]["upper_estimate_bytes"] += 1
        if candidate["candidate_id"] == value["recommended"]["candidate_id"]:
            value["recommended"] = copy.deepcopy(candidate)
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("component upper bounds" in item for item in errors))

    def test_rejects_dataset_hash_mismatch(self) -> None:
        self.payload["dataset"]["source_sha256"] = "0" * 64
        errors = validate_plan_payload(self.payload, verify_dataset=True)
        self.assertTrue(any("hash" in item.lower() for item in errors))

    def test_rejects_invalid_precision_learning_rate_and_nonfinite_numbers(
        self,
    ) -> None:
        value = copy.deepcopy(self.payload)
        value["candidates"][0]["precision"] = "banana"
        value["candidates"][1]["learning_rate"] = float("nan")
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("precision" in item for item in errors))
        self.assertTrue(any("learning_rate" in item for item in errors))
        self.assertTrue(any("finite JSON" in item for item in errors))

    def test_rejects_invalid_target_and_hardware_ranges(self) -> None:
        value = copy.deepcopy(self.payload)
        value["target"]["objective"] = "guess"
        value["target"]["method_preference"] = "invented"
        value["hardware"]["devices"][0]["free_vram_bytes"] = (
            value["hardware"]["devices"][0]["total_vram_bytes"] + 1
        )
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("objective" in item for item in errors))
        self.assertTrue(any("method_preference" in item for item in errors))
        self.assertTrue(any("free_vram_bytes" in item for item in errors))

    def test_rejects_unenforced_wall_time_contract(self) -> None:
        value = copy.deepcopy(self.payload)
        value["target"]["max_wall_time_minutes"] = 30
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("must be null in Aptus v0.2" in item for item in errors))

    def test_rejects_malformed_duplicate_and_missing_evidence(self) -> None:
        value = copy.deepcopy(self.payload)
        duplicate = copy.deepcopy(value["evidence_records"][0])
        duplicate["claim"] = ""
        value["evidence_records"].append(duplicate)
        value["candidates"][0]["evidence"] = ["missing.evidence"]
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("Duplicate evidence ID" in item for item in errors))
        self.assertTrue(any("non-empty string claim" in item for item in errors))
        self.assertTrue(
            any("references missing evidence ID" in item for item in errors)
        )

    def test_rejects_non_list_evidence_shapes_without_crashing(self) -> None:
        value = copy.deepcopy(self.payload)
        value["evidence_records"] = "not-a-list"
        value["candidates"][0]["evidence"] = "method.lora.paper"
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(
            any("evidence_records must be a list" in item for item in errors)
        )
        self.assertTrue(any("evidence must be a list" in item for item in errors))

    def test_runtime_contract_is_identity_bound_and_backend_checked(self) -> None:
        value = copy.deepcopy(self.payload)
        candidate = value["candidates"][0]
        candidate["runtime_contract"]["compute_backend"] = "mps"
        if candidate["candidate_id"] == value["recommended"]["candidate_id"]:
            value["recommended"] = copy.deepcopy(candidate)
        errors = validate_plan_payload(value, verify_dataset=False)
        self.assertTrue(any("runtime and compute backend" in item for item in errors))
        self.assertTrue(any("normalized execution contract" in item for item in errors))

    def test_rejects_invented_runtime_binding_identities(self) -> None:
        candidate_index = next(
            index
            for index, item in enumerate(self.payload["candidates"])
            if item["status"] in {"feasible", "conditional"}
        )
        mutations = (
            ("compiler_id", "invented.compiler.v99"),
            ("estimator_id", "invented-estimator-v99"),
            ("export_kind", "invented-export"),
            ("evidence_requirement", "implementation-required"),
            ("training_runtime", "invented-runtime"),
            ("compute_backend", "rocm"),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):
                value = copy.deepcopy(self.payload)
                value["candidates"][candidate_index]["runtime_contract"][field] = (
                    replacement
                )
                errors = validate_plan_payload(value, verify_dataset=False)
                self.assertTrue(
                    any(
                        "registered compiler, estimator, export, and evidence identity"
                        in item
                        or "exact unavailable identity" in item
                        for item in errors
                    ),
                    errors,
                )

    def test_mps_static_fit_uses_live_host_memory_headroom(self) -> None:
        dataset_path = self.root / "apple-source.jsonl"
        dataset_path.write_text('{"text":"apple example"}\n', encoding="utf-8")
        dataset = profile_dataset(
            dataset_path,
            sample_limit=64,
            sequence_length=128,
        )
        model = build_model_spec(
            model_id="example/apple-model-1b",
            revision="b" * 40,
            family="llama",
            parameters_b=1,
            hidden_size=2048,
            intermediate_size=8192,
            layers=24,
            context_length=4096,
            license_name="apache-2.0",
            training_allowed=True,
        )
        hardware = build_hardware_spec(
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
        target = TrainingTarget(
            objective=Objective.MEMORY,
            sequence_length=128,
            effective_batch_size=8,
            max_epochs=1,
            task="sft",
            checkpoint_steps=10,
            training_runtime=TrainingRuntime.MLX_LM,
        )
        value = to_primitive(
            plan_training(
                model=model,
                dataset=dataset,
                hardware=hardware,
                target=target,
            )
        )
        self.assertEqual(validate_plan_payload(value, verify_dataset=False), ())

        value["hardware"]["host_ram_free_bytes"] = int(8.25 * 1024**3)
        errors = validate_plan_payload(value, verify_dataset=False)

        self.assertTrue(
            any("exceeds usable per-device memory" in item for item in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
