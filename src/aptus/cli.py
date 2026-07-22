from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .domain import (
    Backend,
    Method,
    Objective,
    TrainingTarget,
    ValidationState,
    to_primitive,
    training_plan_from_primitive,
)
from .generation import create_bundle_archive, generate_bundle
from .planning import plan_training
from .profiling import (
    build_hardware_spec,
    build_model_spec,
    probe_local_hardware,
    profile_dataset,
)


def _write_json(value: Any, output: Path | None) -> None:
    text = (
        json.dumps(to_primitive(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def _add_fact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--parameters-b", required=True, type=float)
    parser.add_argument("--hidden-size", required=True, type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--layers", required=True, type=int)
    parser.add_argument("--context-length", required=True, type=int)
    parser.add_argument("--license", required=True)
    parser.add_argument("--confirm-training-allowed", action="store_true")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--sample-limit", type=int, default=512)
    parser.add_argument(
        "--backend", default="cuda", choices=[item.value for item in Backend]
    )
    parser.add_argument("--gpu-count", required=True, type=int)
    parser.add_argument("--vram-gib", required=True, type=float)
    parser.add_argument("--free-vram-gib", type=float)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--four-bit", action="store_true")
    parser.add_argument("--eight-bit", action="store_true")
    parser.add_argument("--host-ram-gib", required=True, type=float)
    parser.add_argument("--host-ram-free-gib", type=float)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--disk-free-gib", type=float)
    parser.add_argument(
        "--objective", default="memory", choices=[item.value for item in Objective]
    )
    parser.add_argument("--sequence-length", required=True, type=int)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--prefer-method", choices=[item.value for item in Method])
    parser.add_argument("--evaluation-fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--packing", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aptus",
        description="Evidence-backed fine-tuning planner and artifact compiler.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="Profile a local training dataset.")
    profile.add_argument("--dataset", required=True, type=Path)
    profile.add_argument("--sample-limit", type=int, default=512)
    profile.add_argument("--sequence-length", type=int)
    profile.add_argument("--output", type=Path)

    for name, help_text in (
        ("spec-plan", "Write a persisted v2 plan JSON without compiling."),
        ("plan", "Compatibility flow: plan, compile, validate, and archive."),
        ("build", "Plan, compile, validate, and archive."),
    ):
        command = commands.add_parser(name, help=help_text)
        _add_fact_arguments(command)
        command.add_argument("--output", required=True, type=Path)
        if name in {"plan", "build"}:
            command.add_argument("--plan-output", type=Path)

    compile_command = commands.add_parser(
        "compile", help="Compile a persisted plan JSON into a portable bundle."
    )
    compile_command.add_argument("--plan", required=True, type=Path)
    compile_command.add_argument("--output", required=True, type=Path)
    compile_command.add_argument("--archive", type=Path)

    validate = commands.add_parser(
        "validate", help="Validate a bundle at one explicit evidence level."
    )
    validate.add_argument("bundle", type=Path)
    validate.add_argument(
        "--level",
        choices=(
            "contract",
            "static",
            "dependency",
            "model-data",
            "measured-preflight",
            "pilot",
        ),
        default="static",
    )
    validate.add_argument(
        "--run",
        action="store_true",
        help="Execute checks required above static validation.",
    )
    validate.add_argument("--state-dir", type=Path, default=Path(".aptus-state"))

    run = commands.add_parser(
        "run", help="Start a persisted local preflight, pilot, or training job."
    )
    run.add_argument("bundle", type=Path)
    run.add_argument(
        "--action",
        choices=("dependency", "model-data", "preflight", "pilot", "train"),
        default="preflight",
    )
    run.add_argument("--confirm-full-train", action="store_true")
    run.add_argument("--state-dir", type=Path, default=Path(".aptus-state"))

    jobs = commands.add_parser("jobs", help="List or inspect persisted local jobs.")
    jobs.add_argument("--state-dir", type=Path, default=Path(".aptus-state"))
    jobs.add_argument("--id")

    serve = commands.add_parser(
        "serve", help="Serve the local API and built React app from one origin."
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--state-dir", type=Path, default=Path(".aptus-state"))
    serve.add_argument("--web-dist", type=Path)
    serve.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Acknowledge that the trusted-user jobs API is unsafe to expose without an external security boundary.",
    )

    commands.add_parser(
        "hardware", help="Inspect CUDA hardware on this local Aptus host."
    )
    inspect = commands.add_parser(
        "inspect", help="Inspect local hardware or bounded provider model facts."
    )
    inspect_commands = inspect.add_subparsers(dest="inspect_command", required=True)
    inspect_commands.add_parser("hardware")
    model = inspect_commands.add_parser("model")
    model.add_argument("--model-id", required=True)
    model.add_argument("--revision", required=True)
    model.add_argument("--timeout", type=float, default=10.0)
    return parser


def _make_plan(arguments: argparse.Namespace) -> Any:
    if not arguments.confirm_training_allowed:
        raise ValueError("Model training permission must be explicitly confirmed.")
    dataset = profile_dataset(
        arguments.dataset,
        sample_limit=arguments.sample_limit,
        sequence_length=arguments.sequence_length,
    )
    model = build_model_spec(
        model_id=arguments.model_id,
        revision=arguments.revision,
        family=arguments.family,
        parameters_b=arguments.parameters_b,
        hidden_size=arguments.hidden_size,
        intermediate_size=arguments.intermediate_size,
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
        supports_8bit=arguments.eight_bit,
        free_vram_gib=arguments.free_vram_gib,
        host_ram_gib=arguments.host_ram_gib,
        host_ram_free_gib=arguments.host_ram_free_gib,
        reserve_gib=arguments.reserve_gib,
        disk_free_gib=arguments.disk_free_gib,
    )
    target = TrainingTarget(
        objective=Objective(arguments.objective),
        sequence_length=arguments.sequence_length,
        effective_batch_size=arguments.effective_batch_size,
        max_epochs=arguments.epochs,
        method_preference=Method(arguments.prefer_method)
        if arguments.prefer_method
        else None,
        task="sft",
        evaluation_fraction=arguments.evaluation_fraction,
        packing=arguments.packing,
        checkpoint_steps=arguments.checkpoint_steps,
    )
    return plan_training(model=model, dataset=dataset, hardware=hardware, target=target)


def _compile(plan: Any, output: Path, archive: Path | None = None) -> dict[str, Any]:
    output_target = output.resolve()
    archive_target = (archive or output_target.with_suffix(".zip")).resolve()
    if archive_target == output_target or output_target in archive_target.parents:
        raise ValueError(
            "Bundle archives must be written outside the bundle directory."
        )
    if archive_target.exists():
        raise FileExistsError(f"Archive output already exists: {archive_target}")
    report = generate_bundle(plan, output)
    archive_path = create_bundle_archive(output, archive_target)
    return {
        "bundle_dir": str(output.resolve()),
        "archive_path": str(archive_path),
        "report": to_primitive(report),
    }


def _wait_for_job(service: Any, job: dict[str, Any]) -> int:
    try:
        while job["state"] in {"queued", "running", "cancelling"}:
            time.sleep(0.25)
            job = service.get(job["id"])
    except KeyboardInterrupt:
        try:
            job = service.cancel(job["id"])
        except ValueError as error:
            print(
                f"Aptus could not cancel interrupted job {job['id']}: {error}",
                file=sys.stderr,
            )
        else:
            _write_json(job, None)
        return 130
    _write_json(job, None)
    return 0 if job["state"] == "completed" else 1


def _run(arguments: argparse.Namespace) -> int:
    if arguments.command == "profile":
        _write_json(
            profile_dataset(
                arguments.dataset,
                sample_limit=arguments.sample_limit,
                sequence_length=arguments.sequence_length,
            ),
            arguments.output,
        )
        return 0
    if arguments.command in {"spec-plan", "plan", "build"}:
        plan = _make_plan(arguments)
        if arguments.command == "spec-plan":
            _write_json(plan, arguments.output)
        else:
            if arguments.plan_output:
                _write_json(plan, arguments.plan_output)
            _write_json(_compile(plan, arguments.output), None)
        return 0
    if arguments.command == "compile":
        value = json.loads(arguments.plan.read_text(encoding="utf-8"))
        plan = training_plan_from_primitive(value)
        _write_json(_compile(plan, arguments.output, arguments.archive), None)
        return 0
    if arguments.command == "validate":
        if arguments.run and arguments.level not in {"contract", "static"}:
            from .execution import JobService

            action = {
                "dependency": "dependency",
                "model-data": "model-data",
                "measured-preflight": "preflight",
                "pilot": "pilot",
            }[arguments.level]
            service = JobService(arguments.state_dir / "jobs")
            job = service.submit(arguments.bundle, action=action)
            return _wait_for_job(service, job)
        from .validation import validate_bundle

        report = validate_bundle(
            arguments.bundle, level=arguments.level, run=arguments.run
        )
        _write_json(report, None)
        return 1 if report.state == ValidationState.INVALID else 0
    if arguments.command == "run":
        from .execution import JobService

        service = JobService(arguments.state_dir / "jobs")
        job = service.submit(
            arguments.bundle,
            action=arguments.action,
            confirm_full_train=arguments.confirm_full_train,
        )
        return _wait_for_job(service, job)
    if arguments.command == "jobs":
        from .execution import JobService

        service = JobService(arguments.state_dir / "jobs")
        _write_json(service.get(arguments.id) if arguments.id else service.list(), None)
        return 0
    if arguments.command == "serve":
        if arguments.host not in {"127.0.0.1", "localhost", "::1"}:
            if not arguments.allow_non_loopback:
                raise ValueError(
                    "Non-loopback serving is blocked by default. Add --allow-non-loopback only behind authentication, "
                    "approved bundle roots, and worker isolation."
                )
            print(
                "Aptus warning: non-loopback serving exposes a trusted-user execution API. "
                "Provide an external security boundary before accepting traffic.",
                file=sys.stderr,
            )
        try:
            import uvicorn
        except ImportError as error:
            raise ValueError(
                "Install Aptus with the server extra to use `aptus serve`."
            ) from error
        from .api import create_app

        allowed_hosts = ("*",) if arguments.allow_non_loopback else (arguments.host,)
        uvicorn.run(
            create_app(
                state_dir=arguments.state_dir,
                static_dir=arguments.web_dist,
                allowed_hosts=allowed_hosts,
            ),
            host=arguments.host,
            port=arguments.port,
        )
        return 0
    if arguments.command == "hardware" or (
        arguments.command == "inspect" and arguments.inspect_command == "hardware"
    ):
        try:
            value = {
                "status": "ok",
                "scope": "server-local",
                "hardware": to_primitive(probe_local_hardware()),
            }
        except ValueError as error:
            value = {
                "status": "unavailable",
                "scope": "server-local",
                "error": str(error),
                "manual_facts_supported": True,
            }
        _write_json(value, None)
        return 0 if value["status"] == "ok" else 2
    if arguments.command == "inspect" and arguments.inspect_command == "model":
        from .inspection import inspect_huggingface_model

        value = inspect_huggingface_model(
            arguments.model_id, arguments.revision, timeout=arguments.timeout
        )
        _write_json(value, None)
        return 0 if value["status"] == "ok" else 2
    raise AssertionError(f"Unhandled command: {arguments.command}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except (
        OSError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"Aptus error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
