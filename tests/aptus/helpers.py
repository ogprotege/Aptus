from __future__ import annotations

from pathlib import Path

from aptus.catalog import (
    reviewed_qwen3_moe_quantization_layout,
)
from aptus.domain import Backend, Objective, TrainingRuntime, TrainingTarget
from aptus.planning import plan_training
from aptus.profiling import build_hardware_spec, build_model_spec, profile_dataset


QWEN3_MOE_MODEL_ID = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
QWEN3_MOE_REVISION = "e9675aa3ca5f900ccef55267914466d55ab325fa"


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


def make_qwen3_moe_plan(root: Path):
    dataset_path = make_dataset(root)
    dataset = profile_dataset(dataset_path, sample_limit=64, sequence_length=128)
    model = build_model_spec(
        model_id=QWEN3_MOE_MODEL_ID,
        revision=QWEN3_MOE_REVISION,
        family="qwen3_moe",
        parameters_b=30.5,
        hidden_size=2048,
        intermediate_size=6144,
        layers=48,
        context_length=262144,
        license_name="apache-2.0",
        training_allowed=True,
        architecture="Qwen3MoeForCausalLM",
        model_type="qwen3_moe",
        quantization_bits=4,
        quantization_layout=reviewed_qwen3_moe_quantization_layout(48),
        moe={
            "expert_count": 128,
            "experts_per_token": 8,
            "expert_intermediate_size": 768,
            "decoder_sparse_step": 1,
            "mlp_only_layers": (),
            "shared_expert_intermediate_size": None,
        },
    )
    hardware = build_hardware_spec(
        backend=Backend.MPS,
        gpu_count=1,
        vram_gib=64,
        supports_bf16=False,
        supports_4bit=False,
        host_ram_gib=64,
        host_ram_free_gib=56,
        reserve_gib=8,
        disk_free_gib=500,
    )
    target = TrainingTarget(
        objective=Objective.MEMORY,
        sequence_length=128,
        effective_batch_size=8,
        max_epochs=1,
        method_preference=None,
        task="sft",
        checkpoint_steps=10,
        training_runtime=TrainingRuntime.MLX_LM,
    )
    return plan_training(model=model, dataset=dataset, hardware=hardware, target=target)


def qwen3_moe_quantization_config() -> dict[str, object]:
    """Return the provider config shape equivalent to the canonical plan layout."""

    layout = reviewed_qwen3_moe_quantization_layout(48)
    return {
        "bits": layout.default_bits,
        "group_size": layout.default_group_size,
        **{
            item.module_path: {
                "bits": item.bits,
                "group_size": item.group_size,
            }
            for item in layout.module_overrides
        },
    }
