"""Emit an Aptus spec-plan command from live hardware + a measured model pin.

Rank is not a flag. ``--objective quality`` selects 16 for this token volume.
``--objective memory`` (CLI default) selects 8 and did not recit this corpus.

This script does not train. It writes the plan JSON and a ladder shell that
still requires ``--confirm-full-train``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from aptus.domain import to_primitive
from aptus.profiling import probe_local_hardware

# Measured pin for Journey B / mix on this host. Not Path Alpha.
QWEN25_7B_MLX_4BIT = {
    "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "revision": "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
    "family": "qwen",
    "parameters_b": "7.62",
    "model_type": "qwen2",
    "architecture": "Qwen2ForCausalLM",
    "quantization_bits": "4",
    "quantization_group_size": "64",
    "hidden_size": "3584",
    "intermediate_size": "18944",
    "layers": "28",
    "context_length": "32768",
    "license": "apache-2.0",
}

# 5 epochs: train loss ~0.05–0.12, gold recitation 38/62 when gold is in train.
# 10 epochs: loss exploded ~0.05 → 7. Do not emit 10.
MEASURED_EPOCHS = 5
MEASURED_OBJECTIVE = "quality"
MEASURED_SEQUENCE = 1024
APPLE_RESERVE_GIB = 8.0


def _gib(num: int | float | None) -> str:
    if num is None:
        raise ValueError("Required measured byte count is missing.")
    return f"{float(num) / 1024**3:.2f}"


def spec_plan_argv(
    *,
    dataset: Path,
    plan_output: Path,
    epochs: int = MEASURED_EPOCHS,
    objective: str = MEASURED_OBJECTIVE,
) -> list[str]:
    hardware = probe_local_hardware(reserve_gib=APPLE_RESERVE_GIB)
    if hardware.host_ram_free_bytes is None or hardware.disk_free_bytes is None:
        raise ValueError("Hardware probe omitted free RAM or disk; Aptus fail-closes.")
    if not hardware.devices:
        raise ValueError("Hardware probe listed no devices.")
    device = hardware.devices[0]
    pin = QWEN25_7B_MLX_4BIT
    argv = [
        sys.executable,
        "-m",
        "aptus",
        "spec-plan",
        "--model-id",
        pin["model_id"],
        "--revision",
        pin["revision"],
        "--family",
        pin["family"],
        "--parameters-b",
        pin["parameters_b"],
        "--model-type",
        pin["model_type"],
        "--architecture",
        pin["architecture"],
        "--quantization-bits",
        pin["quantization_bits"],
        "--quantization-group-size",
        pin["quantization_group_size"],
        "--hidden-size",
        pin["hidden_size"],
        "--intermediate-size",
        pin["intermediate_size"],
        "--layers",
        pin["layers"],
        "--context-length",
        pin["context_length"],
        "--license",
        pin["license"],
        "--confirm-training-allowed",
        "--confirm-unreviewed-runtime",
        "--dataset",
        str(dataset),
        "--sample-limit",
        "512",
        "--backend",
        "mps",
        "--training-runtime",
        "mlx-lm",
        "--gpu-count",
        "1",
        "--vram-gib",
        _gib(device.total_vram_bytes),
        "--host-ram-gib",
        _gib(hardware.host_ram_bytes),
        "--host-ram-free-gib",
        _gib(hardware.host_ram_free_bytes),
        "--reserve-gib",
        str(APPLE_RESERVE_GIB),
        "--disk-free-gib",
        _gib(hardware.disk_free_bytes),
        "--objective",
        objective,
        "--sequence-length",
        str(MEASURED_SEQUENCE),
        "--effective-batch-size",
        "1",
        "--epochs",
        str(epochs),
        "--prefer-method",
        "qlora",
        "--output",
        str(plan_output),
    ]
    return argv


def write_ladder_script(path: Path, bundle: Path, state: Path) -> None:
    body = f"""#!/bin/zsh
set -euo pipefail
cd {Path.cwd()}
source .venv/bin/activate
export PYTHONPATH=src:.
export APTUS_MLX_PYTHON="$(pwd)/.venv/bin/python"
BUNDLE={bundle}
STATE={state}
mkdir -p "$STATE"
python -m aptus run "$BUNDLE" --action dependency --state-dir "$STATE"
python -m aptus run "$BUNDLE" --action model-data --state-dir "$STATE"
python -m aptus run "$BUNDLE" --action preflight --state-dir "$STATE"
python -m aptus run "$BUNDLE" --action pilot --state-dir "$STATE"
echo "Pilot done. Full train still requires an explicit confirm:"
echo "python -m aptus run \\"$BUNDLE\\" --action train --confirm-full-train --state-dir \\"$STATE\\""
"""
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe hardware and write spec-plan argv for the measured 7B MLX QLoRA pin."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=MEASURED_EPOCHS)
    parser.add_argument(
        "--objective",
        default=MEASURED_OBJECTIVE,
        choices=("quality", "memory", "speed"),
    )
    parser.add_argument(
        "--run-plan",
        action="store_true",
        help="Execute spec-plan after writing the command (still no train).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the plan into workdir/bundle after spec-plan.",
    )
    args = parser.parse_args()
    if args.epochs >= 10:
        raise SystemExit(
            "Measured 10-epoch run exploded (train loss ~0.05 → 7). "
            "Refuse to emit epochs>=10 for this pin."
        )
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    plan_output = workdir / "plan.json"
    argv = spec_plan_argv(
        dataset=args.dataset.resolve(),
        plan_output=plan_output,
        epochs=args.epochs,
        objective=args.objective,
    )
    cmd_path = workdir / "spec-plan.sh"
    if cmd_path.exists() or (args.run_plan and plan_output.exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing plan artifacts in {workdir}"
        )
    cmd_path.write_text(
        "#!/bin/zsh\nset -euo pipefail\ncd "
        + str(Path.cwd())
        + "\nsource .venv/bin/activate\nexport PYTHONPATH=src:.\n"
        + " ".join(shlex.quote(part) for part in argv)
        + "\n",
        encoding="utf-8",
    )
    cmd_path.chmod(cmd_path.stat().st_mode | 0o111)
    hardware = to_primitive(probe_local_hardware(reserve_gib=APPLE_RESERVE_GIB))
    (workdir / "hardware.json").write_text(
        json.dumps(hardware, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("wrote", cmd_path)
    print("objective", args.objective, "epochs", args.epochs)
    print("note: quality → rank 16; memory → rank 8 (did not recit this corpus)")
    if args.run_plan:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src:." + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        subprocess.run(argv, check=True, cwd=Path.cwd(), env=env)
        print("wrote", plan_output)
    if args.compile:
        if not plan_output.exists():
            raise SystemExit("Compile requires --run-plan (plan.json missing).")
        bundle = workdir / "bundle"
        compile_argv = [
            sys.executable,
            "-m",
            "aptus",
            "compile",
            "--plan",
            str(plan_output),
            "--output",
            str(bundle),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = "src:." + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        subprocess.run(compile_argv, check=True, cwd=Path.cwd(), env=env)
        write_ladder_script(workdir / "ladder.sh", bundle, workdir / "state")
        print("wrote", bundle)
        print("wrote", workdir / "ladder.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
