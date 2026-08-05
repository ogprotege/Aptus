from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from aptus.domain import (
    Backend,
    Method,
    Objective,
    TrainingRuntime,
    TrainingTarget,
    ValidationState,
)
from aptus.generation import generate_bundle
from aptus.methods import selectable_method_descriptors
from aptus.planning import plan_training
from aptus.profiling import build_hardware_spec, build_model_spec, profile_dataset

from tests.aptus.helpers import make_dataset, make_qwen2_runtime_footprint_plan


RuntimeRow = tuple[str, str, str, str]


@dataclass(frozen=True)
class CudaPlanCase:
    vram_gib: float
    effective_batch_size: int
    objective: Objective
    method_preference: Method | None = None
    supports_4bit: bool = False
    supports_8bit: bool = False


CUDA_CASES: dict[tuple[str, str], CudaPlanCase] = {
    ("full", "single"): CudaPlanCase(4, 2, Objective.QUALITY),
    ("full", "ddp"): CudaPlanCase(4, 64, Objective.SPEED),
    ("lora", "single"): CudaPlanCase(2, 2, Objective.SPEED),
    ("lora", "ddp"): CudaPlanCase(2, 64, Objective.SPEED),
    ("lora", "fsdp"): CudaPlanCase(2, 2, Objective.QUALITY),
    ("int8-lora", "single"): CudaPlanCase(
        2,
        2,
        Objective.SPEED,
        method_preference=Method.INT8_LORA,
        supports_8bit=True,
    ),
    ("int8-lora", "ddp"): CudaPlanCase(
        2,
        2,
        Objective.MEMORY,
        supports_8bit=True,
    ),
    ("qlora", "single"): CudaPlanCase(
        2,
        2,
        Objective.QUALITY,
        supports_4bit=True,
    ),
    ("qlora", "ddp"): CudaPlanCase(
        2,
        2,
        Objective.MEMORY,
        supports_4bit=True,
    ),
}


def _registered_runtime_rows() -> set[RuntimeRow]:
    return {
        (
            binding.training_runtime,
            binding.compute_backend,
            descriptor.method_id,
            distribution,
        )
        for descriptor in selectable_method_descriptors()
        for binding in descriptor.runtime_bindings
        for distribution in binding.supported_distributions
    }


def _fixture_runtime_rows() -> set[RuntimeRow]:
    return {
        (
            TrainingRuntime.TRANSFORMERS_PEFT_CUDA.value,
            Backend.CUDA.value,
            method,
            distribution,
        )
        for method, distribution in CUDA_CASES
    } | {
        (TrainingRuntime.MLX_LM.value, Backend.MPS.value, "lora", "single"),
        (TrainingRuntime.MLX_LM.value, Backend.MPS.value, "qlora", "single"),
    }


def _cuda_plan(root: Path, case: CudaPlanCase):
    dataset = profile_dataset(make_dataset(root), sample_limit=64, sequence_length=128)
    model = build_model_spec(
        model_id="example/operator-docs-model",
        revision="a" * 40,
        family="llama",
        parameters_b=0.1,
        hidden_size=512,
        intermediate_size=2048,
        layers=8,
        context_length=4096,
        license_name="apache-2.0",
        training_allowed=True,
    )
    hardware = build_hardware_spec(
        backend=Backend.CUDA,
        gpu_count=2,
        vram_gib=case.vram_gib,
        supports_bf16=True,
        supports_4bit=case.supports_4bit,
        supports_8bit=case.supports_8bit,
        host_ram_gib=128,
        reserve_gib=0.1,
        disk_free_gib=500,
    )
    target = TrainingTarget(
        objective=case.objective,
        sequence_length=128,
        effective_batch_size=case.effective_batch_size,
        max_epochs=1,
        method_preference=case.method_preference,
        task="sft",
        checkpoint_steps=10,
        training_runtime=TrainingRuntime.TRANSFORMERS_PEFT_CUDA,
    )
    return plan_training(
        model=model,
        dataset=dataset,
        hardware=hardware,
        target=target,
    )


def _mlx_lora_plan(root: Path):
    dataset = profile_dataset(make_dataset(root), sample_limit=64, sequence_length=128)
    model = build_model_spec(
        model_id="example/operator-docs-mlx-model",
        revision="a" * 40,
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
        supports_bf16=False,
        supports_4bit=False,
        host_ram_gib=64,
        host_ram_free_gib=48,
        reserve_gib=8,
        disk_free_gib=500,
    )
    target = TrainingTarget(
        objective=Objective.SPEED,
        sequence_length=128,
        effective_batch_size=8,
        max_epochs=1,
        method_preference=Method.LORA,
        task="sft",
        checkpoint_steps=10,
        training_runtime=TrainingRuntime.MLX_LM,
    )
    return plan_training(
        model=model,
        dataset=dataset,
        hardware=hardware,
        target=target,
    )


class GeneratedOperatorDocumentationTests(unittest.TestCase):
    def assert_in_order(self, text: str, *values: str) -> None:
        cursor = -1
        for value in values:
            next_cursor = text.find(value, cursor + 1)
            self.assertGreater(
                next_cursor,
                cursor,
                f"Expected {value!r} after offset {cursor}.",
            )
            cursor = next_cursor

    def test_fixture_matrix_matches_every_registered_executable_row(self) -> None:
        self.assertEqual(_fixture_runtime_rows(), _registered_runtime_rows())

    def test_every_executable_row_emits_complete_operator_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for runtime, backend, method, distribution in sorted(
                _fixture_runtime_rows()
            ):
                row = (runtime, backend, method, distribution)
                with self.subTest(row=row):
                    case_root = root / "-".join(row)
                    case_root.mkdir(parents=True)
                    if runtime == TrainingRuntime.TRANSFORMERS_PEFT_CUDA.value:
                        plan = _cuda_plan(case_root, CUDA_CASES[(method, distribution)])
                    elif method == "lora":
                        plan = _mlx_lora_plan(case_root)
                    else:
                        plan = make_qwen2_runtime_footprint_plan(case_root)

                    self.assertEqual(plan.recommended.method.value, method)
                    self.assertEqual(
                        plan.recommended.distribution.value,
                        distribution,
                    )
                    self.assertIsNotNone(plan.recommended.runtime_contract)
                    contract = plan.recommended.runtime_contract
                    assert contract is not None
                    self.assertEqual(contract.training_runtime.value, runtime)
                    self.assertEqual(contract.compute_backend.value, backend)

                    output = case_root / "bundle"
                    report = generate_bundle(plan, output)
                    self.assertEqual(report.state, ValidationState.STATIC_PASS)

                    payload = json.loads(
                        (output / "plan.json").read_text(encoding="utf-8")
                    )
                    manifest = json.loads(
                        (output / "bundle-manifest.json").read_text(encoding="utf-8")
                    )
                    readme = (output / "README.md").read_text(encoding="utf-8")
                    decision = (output / "decision-report.md").read_text(
                        encoding="utf-8"
                    )
                    runbook = (output / "runbook.md").read_text(encoding="utf-8")
                    normalized_readme = " ".join(readme.split())

                    manifested = {item["path"] for item in manifest["files"]}
                    self.assertTrue(
                        {"README.md", "decision-report.md", "runbook.md"} <= manifested
                    )
                    self.assertEqual(payload["recommended"]["method"], method)
                    self.assertEqual(
                        payload["recommended"]["distribution"], distribution
                    )

                    self.assert_in_order(
                        readme,
                        "`decision-report.md`",
                        "`plan.json`",
                        "`evidence.jsonl`",
                        "`requirements.txt`",
                        "`runbook.md`",
                    )
                    self.assert_in_order(
                        readme,
                        "python validate.py --level static",
                        "python validate.py --level dependency",
                        "python validate.py --level model-data",
                        "python validate.py --level measured-preflight",
                        "python validate.py --level pilot",
                        "python run.py --confirm-full-train",
                    )
                    self.assertIn("`pilot-pass`", readme)
                    self.assertIn("`measured-run-pass`", readme)
                    self.assertIn("does not prove model quality", normalized_readme)
                    self.assertIn("`policy/model-policy-snapshot.v1.json`", readme)
                    self.assertIn("Installed Aptus checks its current registry", readme)

                    world_size = 1 if distribution == "single" else 2
                    self.assertIn(f"- Method: `{method}`", decision)
                    self.assertIn(
                        f"- Distribution and world size: `{distribution}`, `{world_size}`",
                        decision,
                    )
                    self.assertIn(f"- Training runtime: `{runtime}`", decision)
                    self.assertIn(f"- Compute backend: `{backend}`", decision)
                    self.assertIn(f"`{contract.compiler_id}`", decision)
                    self.assertIn(f"`{contract.estimator_id}`", decision)
                    self.assertIn(f"`{contract.export_kind}`", decision)
                    self.assertIn(
                        "does not claim measured throughput or model quality",
                        decision,
                    )

                    self.assert_in_order(
                        runbook,
                        "python -m pip install -r requirements.txt",
                        "python validate.py --level dependency",
                        "python validate.py --level model-data",
                        "python validate.py --level measured-preflight",
                        "python validate.py --level pilot",
                        "python run.py --confirm-full-train",
                    )
                    self.assertIn("`validation-report.json`", runbook)
                    self.assertIn("`pilot-pass`", runbook)
                    self.assertIn("`measured-run-pass`", runbook)
                    self.assertIn("quality claim", runbook)

                    if backend == Backend.CUDA.value:
                        self.assertIn("small synthetic CUDA model", runbook)
                        self.assertIn("recorded CUDA peaks", runbook)
                        self.assertNotIn("Apple silicon", readme)
                    else:
                        self.assertIn("Apple silicon and MLX-LM", readme)
                        self.assertIn("unified-memory headroom", runbook)
                        self.assertIn("never substitutes bitsandbytes", runbook)
                        self.assertNotIn("synthetic CUDA model", runbook)

                    if distribution == "single":
                        self.assertIn("Single-device placement binds", decision)
                    else:
                        self.assertIn("Distributed placement binds", decision)
                        self.assertIn(
                            f"{distribution.upper()} host staging budgets", decision
                        )
                    if distribution == "fsdp":
                        self.assertIn("use_orig_params=true", decision)


if __name__ == "__main__":
    unittest.main()
