from __future__ import annotations

from pathlib import Path

from aptus.catalog import (
    reviewed_qwen3_moe_quantization_layout,
)
from aptus.domain import (
    Backend,
    Objective,
    QuantizationLayout,
    TrainingRuntime,
    TrainingTarget,
)
from aptus.planning import plan_training
from aptus.profiling import build_hardware_spec, build_model_spec, profile_dataset


QWEN3_MOE_MODEL_ID = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
QWEN3_MOE_REVISION = "e9675aa3ca5f900ccef55267914466d55ab325fa"
QWEN2_5_ACCEPTANCE_MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
QWEN2_5_ACCEPTANCE_REVISION = "53a32aee5e9447773fd2b85988395066aef3700a"


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
    optimizer_steps: int | None = None,
    split_seed: int = 424242,
    training_seed: int = 17,
    data_order_seed: int = 1000017,
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
        free_vram_gib=vram_gib,
        supports_bf16=True,
        supports_4bit=True,
        host_ram_gib=host_ram_gib,
        host_ram_free_gib=host_ram_gib,
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
        optimizer_steps=optimizer_steps,
        split_seed=split_seed,
        training_seed=training_seed,
        data_order_seed=data_order_seed,
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


def make_qwen2_runtime_footprint_plan(root: Path):
    """Return the reviewed Qwen2 runtime footprint using its evidence artifact.

    The artifact identity and immutable revision scope the acceptance evidence. The
    compatibility subject remains the architecture/configuration footprint only.
    """

    dataset_path = make_dataset(root)
    dataset = profile_dataset(dataset_path, sample_limit=64, sequence_length=128)
    model = build_model_spec(
        model_id=QWEN2_5_ACCEPTANCE_MODEL_ID,
        revision=QWEN2_5_ACCEPTANCE_REVISION,
        family="qwen",
        parameters_b=0.494,
        hidden_size=896,
        intermediate_size=4864,
        layers=24,
        context_length=32768,
        license_name="apache-2.0",
        training_allowed=True,
        architecture="Qwen2ForCausalLM",
        model_type="qwen2",
        quantization_bits=4,
        quantization_layout=QuantizationLayout(
            default_bits=4,
            default_group_size=64,
        ),
        moe=None,
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
