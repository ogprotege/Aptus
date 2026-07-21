from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .domain import Backend, Method, Objective, TrainingTarget, ValidationState
from .generation import generate_bundle
from .planning import plan_training
from .profiling import build_hardware_spec, build_model_spec, profile_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aptus",
        description="Plan and validate a fine-tuning training bundle.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Create a validated training bundle.")

    plan.add_argument("--model-id", required=True)
    plan.add_argument("--revision", required=True)
    plan.add_argument("--family", required=True)
    plan.add_argument("--parameters-b", required=True, type=float)
    plan.add_argument("--hidden-size", required=True, type=int)
    plan.add_argument("--layers", required=True, type=int)
    plan.add_argument("--context-length", required=True, type=int)
    plan.add_argument("--license", required=True)
    plan.add_argument("--confirm-training-allowed", action="store_true")

    plan.add_argument("--dataset", required=True, type=Path)
    plan.add_argument("--sample-limit", type=int)

    plan.add_argument(
        "--backend",
        required=True,
        choices=[backend.value for backend in Backend],
    )
    plan.add_argument("--gpu-count", required=True, type=int)
    plan.add_argument("--vram-gib", required=True, type=float)
    plan.add_argument("--bf16", action="store_true")
    plan.add_argument("--four-bit", action="store_true")
    plan.add_argument("--host-ram-gib", required=True, type=float)
    plan.add_argument("--reserve-gib", required=True, type=float)

    plan.add_argument(
        "--objective",
        required=True,
        choices=[objective.value for objective in Objective],
    )
    plan.add_argument("--sequence-length", type=int)
    plan.add_argument("--effective-batch-size", type=int, default=16)
    plan.add_argument("--epochs", type=int, default=3)
    plan.add_argument(
        "--prefer-method",
        choices=[method.value for method in Method],
    )
    plan.add_argument("--output", required=True, type=Path)
    return parser


def _default_sequence_length(p95: int, context_length: int) -> int:
    power_of_two = 1 << max(0, p95 - 1).bit_length()
    return min(max(128, power_of_two), context_length)


def _plan(arguments: argparse.Namespace) -> int:
    if not arguments.confirm_training_allowed:
        print(
            "Aptus requires explicit model training permission confirmation.",
            file=sys.stderr,
        )
        return 2

    try:
        dataset = profile_dataset(
            arguments.dataset,
            sample_limit=arguments.sample_limit,
        )
        model = build_model_spec(
            model_id=arguments.model_id,
            revision=arguments.revision,
            family=arguments.family,
            parameters_b=arguments.parameters_b,
            hidden_size=arguments.hidden_size,
            layers=arguments.layers,
            context_length=arguments.context_length,
            license_name=arguments.license,
            training_allowed=True,
        )
        hardware = build_hardware_spec(
            backend=Backend(arguments.backend),
            gpu_count=arguments.gpu_count,
            vram_gib=arguments.vram_gib,
            supports_bf16=arguments.bf16,
            supports_4bit=arguments.four_bit,
            host_ram_gib=arguments.host_ram_gib,
            reserve_gib=arguments.reserve_gib,
        )
        target = TrainingTarget(
            objective=Objective(arguments.objective),
            sequence_length=(
                arguments.sequence_length
                if arguments.sequence_length is not None
                else _default_sequence_length(
                    dataset.sequence_p95,
                    model.context_length,
                )
            ),
            effective_batch_size=arguments.effective_batch_size,
            max_epochs=arguments.epochs,
            method_preference=(
                Method(arguments.prefer_method)
                if arguments.prefer_method
                else None
            ),
        )
        plan = plan_training(
            model=model,
            dataset=dataset,
            hardware=hardware,
            target=target,
        )
        report = generate_bundle(plan, arguments.output)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"Aptus could not create a plan: {error}", file=sys.stderr)
        return 2

    if report.state != ValidationState.STATIC_PASS:
        print(
            "Aptus generated a bundle, but static validation failed. "
            f"Review {arguments.output / 'validation-report.json'}.",
            file=sys.stderr,
        )
        return 1

    candidate = plan.recommended
    peak_gib = candidate.memory.estimated_peak_bytes / 1024**3
    print("Aptus plan and static validation complete.")
    print(f"Method: {candidate.method.value}")
    print(f"Precision: {candidate.precision}")
    print(f"Estimated peak: {peak_gib:.2f} GiB")
    print(f"Bundle: {arguments.output.resolve()}")
    print("Confidence: low-until-calibrated")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "plan":
        return _plan(arguments)
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
