from __future__ import annotations

from pathlib import Path

from aptus.domain import Backend, Objective, TrainingTarget
from aptus.planning import plan_training
from aptus.profiling import build_hardware_spec, build_model_spec, profile_dataset


def make_dataset(root: Path, content: str | None = None) -> Path:
    path = root / "source.jsonl"
    path.write_text(
        content
        or '{"prompt":"Question one?","completion":"Answer one."}\n'
        '{"messages":[{"role":"user","content":"Question two?"},{"role":"assistant","content":"Answer two."}]}\n',
        encoding="utf-8",
    )
    return path


def make_plan(
    root: Path,
    *,
    gpu_count: int = 2,
    vram_gib: float = 24,
    host_ram_gib: float = 64,
    disk_free_gib: float | None = 500,
    objective: Objective = Objective.MEMORY,
    effective_batch: int = 8,
    task: str = "sft",
    packing: bool = False,
):
    dataset_path = make_dataset(root)
    dataset = profile_dataset(dataset_path, sample_limit=64, sequence_length=128)
    model = build_model_spec(
        model_id="example/model-1b",
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
        backend=Backend.CUDA,
        gpu_count=gpu_count,
        vram_gib=vram_gib,
        supports_bf16=True,
        supports_4bit=True,
        host_ram_gib=host_ram_gib,
        reserve_gib=2,
        disk_free_gib=disk_free_gib,
    )
    target = TrainingTarget(
        objective=objective,
        sequence_length=128,
        effective_batch_size=effective_batch,
        max_epochs=1,
        task=task,
        packing=packing,
        checkpoint_steps=10,
    )
    return plan_training(model=model, dataset=dataset, hardware=hardware, target=target)
