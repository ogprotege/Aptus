import unittest
from pathlib import Path

from aptus.domain import (
    Backend,
    DatasetProfile,
    DeviceSpec,
    HardwareSpec,
    MeasurementKind,
    Method,
    ModelSpec,
    Objective,
    TrainingTarget,
    gibibytes,
)
from aptus.planning import (
    NoFeasiblePlanError,
    estimate_candidate,
    plan_training,
)


def model() -> ModelSpec:
    return ModelSpec(
        model_id="meta-llama/example-7b",
        revision="a" * 40,
        family="llama",
        parameters=7_000_000_000,
        hidden_size=4096,
        layers=32,
        context_length=4096,
        license_name="example",
        training_allowed=True,
    )


def dataset() -> DatasetProfile:
    return DatasetProfile(
        source_path=Path("/tmp/data.jsonl"),
        source_sha256="a" * 64,
        source_format="jsonl",
        schema_name="text",
        example_count=1_000,
        total_estimated_tokens=512_000,
        sequence_p50=400,
        sequence_p95=512,
        sequence_max=700,
        measurement=MeasurementKind.ESTIMATED,
    )


def hardware(vram_gb: int, *, four_bit: bool = True) -> HardwareSpec:
    return HardwareSpec(
        devices=(
            DeviceSpec(
                name="CUDA GPU 0",
                backend=Backend.CUDA,
                total_vram_bytes=gibibytes(vram_gb),
                supports_bf16=True,
                supports_4bit=four_bit,
            ),
        ),
        host_ram_bytes=gibibytes(64),
        reserve_per_device_bytes=gibibytes(2),
    )


def target(objective: Objective) -> TrainingTarget:
    return TrainingTarget(
        objective=objective,
        sequence_length=512,
        effective_batch_size=16,
        max_epochs=3,
    )


class PlannerTests(unittest.TestCase):
    def test_quality_prefers_lora_when_both_methods_fit(self) -> None:
        plan = plan_training(
            model=model(),
            dataset=dataset(),
            hardware=hardware(24),
            target=target(Objective.QUALITY),
        )

        self.assertEqual(plan.recommended.method, Method.LORA)
        self.assertTrue(all(candidate.feasible for candidate in plan.candidates))
        self.assertTrue(
            any("heuristic" in warning.lower() for warning in plan.warnings)
        )

    def test_constrained_memory_selects_qlora_when_lora_does_not_fit(self) -> None:
        plan = plan_training(
            model=model(),
            dataset=dataset(),
            hardware=hardware(12),
            target=target(Objective.QUALITY),
        )
        by_method = {candidate.method: candidate for candidate in plan.candidates}

        self.assertFalse(by_method[Method.LORA].feasible)
        self.assertTrue(by_method[Method.QLORA].feasible)
        self.assertEqual(plan.recommended.method, Method.QLORA)

    def test_memory_objective_prefers_lower_peak_estimate(self) -> None:
        plan = plan_training(
            model=model(),
            dataset=dataset(),
            hardware=hardware(24),
            target=target(Objective.MEMORY),
        )

        self.assertEqual(plan.recommended.method, Method.QLORA)
        self.assertLess(
            plan.recommended.memory.estimated_peak_bytes,
            next(
                candidate.memory.estimated_peak_bytes
                for candidate in plan.candidates
                if candidate.method == Method.LORA
            ),
        )

    def test_method_preference_cannot_reverse_memory_objective(self) -> None:
        memory_target = TrainingTarget(
            objective=Objective.MEMORY,
            sequence_length=512,
            effective_batch_size=16,
            max_epochs=3,
            method_preference=Method.LORA,
        )

        plan = plan_training(
            model=model(),
            dataset=dataset(),
            hardware=hardware(24),
            target=memory_target,
        )

        self.assertEqual(plan.recommended.method, Method.QLORA)

    def test_effective_batch_size_is_preserved_exactly(self) -> None:
        batch_target = TrainingTarget(
            objective=Objective.QUALITY,
            sequence_length=512,
            effective_batch_size=10,
            max_epochs=3,
        )

        plan = plan_training(
            model=model(),
            dataset=dataset(),
            hardware=hardware(24),
            target=batch_target,
        )

        for candidate in plan.candidates:
            self.assertEqual(candidate.effective_batch_size, 10)
            self.assertEqual(
                candidate.micro_batch_size
                * candidate.gradient_accumulation_steps,
                10,
            )

    def test_plan_explains_why_recommendation_won(self) -> None:
        plan = plan_training(
            model=model(),
            dataset=dataset(),
            hardware=hardware(24),
            target=target(Objective.QUALITY),
        )

        self.assertTrue(plan.recommendation_rationale)
        self.assertTrue(
            any(
                plan.recommended.method.value in reason.lower()
                for reason in plan.recommendation_rationale
            )
        )

    def test_estimate_grows_with_sequence_length(self) -> None:
        short = estimate_candidate(
            method=Method.QLORA,
            model=model(),
            dataset=dataset(),
            hardware=hardware(24),
            target=TrainingTarget(
                objective=Objective.MEMORY,
                sequence_length=256,
                effective_batch_size=16,
                max_epochs=3,
            ),
        )
        long = estimate_candidate(
            method=Method.QLORA,
            model=model(),
            dataset=dataset(),
            hardware=hardware(24),
            target=TrainingTarget(
                objective=Objective.MEMORY,
                sequence_length=2048,
                effective_batch_size=16,
                max_epochs=3,
            ),
        )

        self.assertGreater(
            long.memory.activations_bytes,
            short.memory.activations_bytes,
        )

    def test_unsupported_backend_fails_with_explicit_reasons(self) -> None:
        mps = HardwareSpec(
            devices=(
                DeviceSpec(
                    name="Apple GPU",
                    backend=Backend.MPS,
                    total_vram_bytes=gibibytes(24),
                    supports_bf16=False,
                    supports_4bit=False,
                ),
            ),
            host_ram_bytes=gibibytes(32),
            reserve_per_device_bytes=gibibytes(2),
        )

        with self.assertRaises(NoFeasiblePlanError) as error:
            plan_training(
                model=model(),
                dataset=dataset(),
                hardware=mps,
                target=target(Objective.QUALITY),
            )

        self.assertTrue(error.exception.candidates)
        self.assertTrue(
            all(candidate.rejection_reasons for candidate in error.exception.candidates)
        )


if __name__ == "__main__":
    unittest.main()
