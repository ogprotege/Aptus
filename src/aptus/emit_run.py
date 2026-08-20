"""Probe this host, fill omitted hardware facts, and emit runnable scripts.

``aptus emit-run`` does not train. Full training still requires
``--confirm-full-train`` on ``aptus run``. Rank, alpha, and learning rate
remain labeled priors, not optima.
"""

from __future__ import annotations

import shlex
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from .domain import HardwareSpec, to_primitive
from .profiling import probe_local_hardware


EMIT_RUN_SCHEMA_VERSION = "aptus.emit-run.v1"
RANK_OBJECTIVE_NOTE = (
    "objective memory selects rank 8; quality or speed select rank 16 below "
    "1e6 estimated tokens, else 32. Rank is a labeled prior, not an optimum."
)
NON_CLAIMS = (
    "emit-run does not train and does not skip --confirm-full-train.",
    "A written spec-plan.sh is not a measured-run-pass.",
    "Exact-match on gold is not general model quality.",
)


def bytes_to_gib(value: int) -> float:
    return value / 1024**3


def _format_number(value: int | float) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected a numeric flag value.")
    if isinstance(value, int):
        return str(value)
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def fill_namespace_from_hardware(
    arguments: Namespace, hardware: HardwareSpec
) -> list[str]:
    """Fill omitted hardware flags from a this-host probe. Return operator notes."""

    notes: list[str] = []
    if not hardware.devices:
        raise ValueError("Hardware probe listed no devices.")
    if hardware.host_ram_free_bytes is None:
        raise ValueError("Hardware probe omitted free host RAM; Aptus fail-closes.")
    if hardware.disk_free_bytes is None:
        raise ValueError("Hardware probe omitted free disk; Aptus fail-closes.")
    backends = {device.backend.value for device in hardware.devices}
    if len(backends) != 1:
        raise ValueError("emit-run requires homogeneous probed devices.")
    probed = next(iter(backends))
    training_runtime = getattr(arguments, "training_runtime", None)
    if probed == "mps":
        if training_runtime == "transformers-peft-cuda":
            raise ValueError(
                "emit-run probed mps devices on this machine; a CUDA training "
                "runtime needs spec-plan with manual CUDA facts for that host."
            )
        if arguments.backend not in {"cuda", "mps", None}:
            raise ValueError(
                f"emit-run probed mps devices; --backend {arguments.backend} "
                "is not this machine."
            )
        if arguments.backend == "cuda" and training_runtime is None:
            notes.append(
                "Hardware probe is Apple Silicon; emit-run selected "
                "--backend mps --training-runtime mlx-lm for this machine. "
                "Use spec-plan with manual CUDA facts for a different host."
            )
        arguments.backend = "mps"
        if training_runtime is None:
            arguments.training_runtime = "mlx-lm"
        arguments.reserve_gib = max(float(arguments.reserve_gib), 8.0)
    elif arguments.backend == "mps":
        raise ValueError(
            f"emit-run probed {probed} devices; --backend mps is this-host only."
        )
    elif probed == "cuda":
        arguments.backend = "cuda"

    if arguments.gpu_count is None:
        arguments.gpu_count = hardware.gpu_count
    if arguments.vram_gib is None:
        arguments.vram_gib = bytes_to_gib(
            min(item.total_vram_bytes for item in hardware.devices)
        )
    if arguments.free_vram_gib is None:
        free_values = [item.free_vram_bytes for item in hardware.devices]
        if all(value is not None for value in free_values):
            arguments.free_vram_gib = bytes_to_gib(
                min(int(value) for value in free_values)
            )
    if arguments.host_ram_gib is None:
        arguments.host_ram_gib = bytes_to_gib(hardware.host_ram_bytes)
    if arguments.host_ram_free_gib is None:
        arguments.host_ram_free_gib = bytes_to_gib(hardware.host_ram_free_bytes)
    if arguments.disk_free_gib is None:
        arguments.disk_free_gib = bytes_to_gib(hardware.disk_free_bytes)
    if any(item.supports_bf16 for item in hardware.devices):
        arguments.bf16 = True
    if any(item.supports_4bit for item in hardware.devices):
        arguments.four_bit = True
    if any(item.supports_8bit for item in hardware.devices):
        arguments.eight_bit = True
    notes.append(
        "Omitted hardware facts were filled from probe_local_hardware on this "
        "machine. They are measured inventory, not a remote-host profile."
    )
    notes.append(RANK_OBJECTIVE_NOTE)
    return notes


def spec_plan_argv(arguments: Namespace, plan_output: Path) -> list[str]:
    """Build a complete ``python -m aptus spec-plan`` command for this host."""

    argv: list[str] = [sys.executable, "-m", "aptus", "spec-plan"]

    def add(flag: str, value: object) -> None:
        if value is None or value is False:
            return
        if value is True:
            argv.append(flag)
            return
        argv.extend([flag, str(value)])

    add("--model-id", arguments.model_id)
    add("--revision", arguments.revision)
    add("--family", arguments.family)
    add("--parameters-b", _format_number(arguments.parameters_b))
    add("--model-type", arguments.model_type)
    add("--architecture", arguments.architecture)
    add(
        "--quantization-bits",
        None
        if arguments.quantization_bits is None
        else str(arguments.quantization_bits),
    )
    add("--quantization-layout-profile", arguments.quantization_layout_profile)
    add(
        "--quantization-group-size",
        None
        if arguments.quantization_group_size is None
        else str(arguments.quantization_group_size),
    )
    add("--hidden-size", str(arguments.hidden_size))
    add(
        "--intermediate-size",
        None
        if arguments.intermediate_size is None
        else str(arguments.intermediate_size),
    )
    add(
        "--moe-expert-count",
        None if arguments.moe_expert_count is None else str(arguments.moe_expert_count),
    )
    add(
        "--moe-experts-per-token",
        None
        if arguments.moe_experts_per_token is None
        else str(arguments.moe_experts_per_token),
    )
    add(
        "--moe-expert-intermediate-size",
        None
        if arguments.moe_expert_intermediate_size is None
        else str(arguments.moe_expert_intermediate_size),
    )
    add(
        "--moe-decoder-sparse-step",
        None
        if arguments.moe_decoder_sparse_step is None
        else str(arguments.moe_decoder_sparse_step),
    )
    for layer in arguments.moe_mlp_only_layer or ():
        argv.extend(["--moe-mlp-only-layer", str(layer)])
    add(
        "--moe-shared-expert-intermediate-size",
        None
        if arguments.moe_shared_expert_intermediate_size is None
        else str(arguments.moe_shared_expert_intermediate_size),
    )
    add("--layers", str(arguments.layers))
    add("--context-length", str(arguments.context_length))
    add("--license", arguments.license)
    add("--confirm-training-allowed", arguments.confirm_training_allowed)
    add("--confirm-unreviewed-runtime", arguments.confirm_unreviewed_runtime)
    if arguments.inspection_receipt is not None:
        add("--inspection-receipt", str(Path(arguments.inspection_receipt).resolve()))
    add("--dataset", str(Path(arguments.dataset).resolve()))
    add("--sample-limit", str(arguments.sample_limit))
    add("--backend", arguments.backend)
    add("--training-runtime", arguments.training_runtime)
    add("--gpu-count", str(arguments.gpu_count))
    add("--vram-gib", _format_number(float(arguments.vram_gib)))
    add(
        "--free-vram-gib",
        None
        if arguments.free_vram_gib is None
        else _format_number(float(arguments.free_vram_gib)),
    )
    add("--bf16", arguments.bf16)
    add("--four-bit", arguments.four_bit)
    add("--eight-bit", arguments.eight_bit)
    add("--host-ram-gib", _format_number(float(arguments.host_ram_gib)))
    add(
        "--host-ram-free-gib",
        None
        if arguments.host_ram_free_gib is None
        else _format_number(float(arguments.host_ram_free_gib)),
    )
    add("--reserve-gib", _format_number(float(arguments.reserve_gib)))
    add(
        "--disk-free-gib",
        None
        if arguments.disk_free_gib is None
        else _format_number(float(arguments.disk_free_gib)),
    )
    add("--objective", arguments.objective)
    add("--sequence-length", str(arguments.sequence_length))
    add("--effective-batch-size", str(arguments.effective_batch_size))
    add("--epochs", str(arguments.epochs))
    add("--prefer-method", arguments.prefer_method)
    add("--evaluation-fraction", _format_number(float(arguments.evaluation_fraction)))
    add("--checkpoint-steps", str(arguments.checkpoint_steps))
    add(
        "--optimizer-steps",
        None if arguments.optimizer_steps is None else str(arguments.optimizer_steps),
    )
    add("--split-seed", str(arguments.split_seed))
    add("--training-seed", str(arguments.training_seed))
    add("--data-order-seed", str(arguments.data_order_seed))
    add(
        "--micro-batch-size",
        None if arguments.micro_batch_size is None else str(arguments.micro_batch_size),
    )
    add(
        "--gradient-accumulation-steps",
        None
        if arguments.gradient_accumulation_steps is None
        else str(arguments.gradient_accumulation_steps),
    )
    add("--packing", arguments.packing)
    add("--output", str(plan_output.resolve()))
    return argv


def write_executable(path: Path, body: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def write_spec_plan_script(path: Path, argv: list[str]) -> None:
    quoted = " ".join(shlex.quote(part) for part in argv)
    write_executable(path, f"#!/bin/sh\nset -eu\nexec {quoted}\n")


def write_ladder_script(path: Path, *, python: str, bundle: Path, state: Path) -> None:
    bundle_s = shlex.quote(str(bundle.resolve()))
    state_s = shlex.quote(str(state.resolve()))
    py = shlex.quote(python)
    write_executable(
        path,
        (
            "#!/bin/sh\n"
            "set -eu\n"
            f"BUNDLE={bundle_s}\n"
            f"STATE={state_s}\n"
            f"PYTHON={py}\n"
            'mkdir -p "$STATE"\n'
            '"$PYTHON" -m aptus run "$BUNDLE" --action dependency --state-dir "$STATE"\n'
            '"$PYTHON" -m aptus run "$BUNDLE" --action model-data --state-dir "$STATE"\n'
            '"$PYTHON" -m aptus run "$BUNDLE" --action preflight --state-dir "$STATE"\n'
            '"$PYTHON" -m aptus run "$BUNDLE" --action pilot --state-dir "$STATE"\n'
            'echo "Pilot done. Full train still requires an explicit confirm:"\n'
            'echo "$PYTHON -m aptus run \\"$BUNDLE\\" --action train '
            '--confirm-full-train --state-dir \\"$STATE\\""\n'
        ),
    )


def write_eval_script(
    path: Path,
    *,
    aptus_python: str,
    bundle: Path,
    gold: Path,
    eval_dir: Path,
) -> None:
    aptus = shlex.quote(aptus_python)
    bundle_s = shlex.quote(str(bundle.resolve()))
    gold_s = shlex.quote(str(gold.resolve()))
    eval_s = shlex.quote(str(eval_dir.resolve()))
    write_executable(
        path,
        (
            "#!/bin/sh\n"
            "set -eu\n"
            "MLX_PYTHON=${APTUS_MLX_PYTHON:?Set APTUS_MLX_PYTHON to the mlx-lm interpreter}\n"
            f"APTUS_PYTHON={aptus}\n"
            f"BUNDLE={bundle_s}\n"
            f"GOLD={gold_s}\n"
            f"EVAL={eval_s}\n"
            "ADAPTER=${1:?adapter directory from a finished run, for example runs/run_<id>/final}\n"
            'mkdir -p "$EVAL"\n'
            'PRED="$EVAL/predictions.jsonl"\n'
            '"$MLX_PYTHON" "$BUNDLE/eval.py" --gold "$GOLD" --adapter "$ADAPTER" '
            '--output "$PRED"\n'
            'echo "Predictions: $PRED"\n'
            'echo "Score with a threshold you choose; exact-match is not quality:"\n'
            'echo "$APTUS_PYTHON -m aptus eval-contract --dataset $GOLD '
            '--claim \\"Exact-match of operator gold completions. Not general model quality.\\" '
            '--threshold THRESHOLD --output $EVAL/contract.json"\n'
            'echo "$APTUS_PYTHON -m aptus eval --contract $EVAL/contract.json '
            '--gold $GOLD --predictions $PRED --output $EVAL/result.json"\n'
        ),
    )


def probe_this_host(*, reserve_gib: float) -> HardwareSpec:
    return probe_local_hardware(reserve_gib=reserve_gib)


def emit_run_report(
    *,
    workdir: Path,
    scripts: list[str],
    notes: list[str],
    hardware: HardwareSpec,
    plan: str | None,
    bundle: str | None,
    prepared_train: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": EMIT_RUN_SCHEMA_VERSION,
        "workdir": str(workdir.resolve()),
        "scripts": scripts,
        "plan": plan,
        "bundle": bundle,
        "prepared_train": prepared_train,
        "trained": False,
        "hardware": to_primitive(hardware),
        "notes": notes,
        "non_claims": list(NON_CLAIMS),
    }
