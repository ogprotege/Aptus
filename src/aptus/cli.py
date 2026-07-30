from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import (
    QWEN3_MOE_QUANTIZATION_PROFILE,
    reviewed_qwen3_moe_quantization_layout,
)
from .domain import (
    Backend,
    Method,
    MoETopology,
    Objective,
    SCHEMA_VERSION,
    TrainingRuntime,
    TrainingTarget,
    ValidationState,
    model_inspection_receipt_from_primitive,
    to_primitive,
    training_plan_from_primitive,
)
from .generation import create_bundle_archive, generate_bundle
from .plan_contract import require_current_model_policy, validate_plan_payload
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
    parser.add_argument(
        "--model-id", required=True, help="Provider repository ID, such as org/model."
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="Immutable 40-to-64-character hexadecimal provider commit.",
    )
    parser.add_argument(
        "--family",
        required=True,
        help="Aptus architecture family from the current target-module catalog.",
    )
    parser.add_argument(
        "--parameters-b",
        required=True,
        type=float,
        help="Total resident model parameter count in billions.",
    )
    parser.add_argument(
        "--model-type",
        help="Exact provider model_type. Required by allowlisted MoE contracts.",
    )
    parser.add_argument(
        "--architecture",
        help="Exact provider architecture class. Required by allowlisted MoE contracts.",
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        choices=range(1, 17),
        metavar="BITS",
        help="Pinned checkpoint weight precision from 1 through 16 bits.",
    )
    parser.add_argument(
        "--quantization-layout-profile",
        choices=(QWEN3_MOE_QUANTIZATION_PROFILE,),
        help=(
            "Exact reviewed provider quantization map. The Qwen3 MoE profile "
            "binds four-bit group-64 defaults and one eight-bit router gate per layer."
        ),
    )
    parser.add_argument(
        "--hidden-size", required=True, type=int, help="Model hidden width."
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        help="Optional MLP intermediate width; planner fallback is 4x hidden size.",
    )
    parser.add_argument(
        "--moe-expert-count",
        type=int,
        help="Total routed experts in an exact MoE topology.",
    )
    parser.add_argument(
        "--moe-experts-per-token",
        type=int,
        help="Routed experts selected for each token.",
    )
    parser.add_argument(
        "--moe-expert-intermediate-size",
        type=int,
        help="Intermediate width of each routed expert.",
    )
    parser.add_argument(
        "--moe-decoder-sparse-step",
        type=int,
        help="Layer cadence for sparse expert blocks.",
    )
    parser.add_argument(
        "--moe-mlp-only-layer",
        action="append",
        default=[],
        type=int,
        help="Dense MLP-only layer index. Repeat for each declared layer.",
    )
    parser.add_argument(
        "--moe-shared-expert-intermediate-size",
        type=int,
        help="Optional intermediate width of the shared expert path.",
    )
    parser.add_argument("--layers", required=True, type=int, help="Model layer count.")
    parser.add_argument(
        "--context-length",
        required=True,
        type=int,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--license", required=True, help="Operator-reviewed model license label."
    )
    parser.add_argument(
        "--confirm-training-allowed",
        action="store_true",
        help="Attest that the intended model training is permitted.",
    )
    parser.add_argument(
        "--inspection-receipt",
        type=Path,
        help=(
            "Optional inspection_receipt JSON from `aptus inspect model`; "
            "all bound provider facts are revalidated before planning."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Local JSON, JSONL, CSV, or text data.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=512,
        help="Rows used for deterministic length statistics (default: 512).",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=[item.value for item in Backend],
        help="Planned compute backend (default: cuda).",
    )
    parser.add_argument(
        "--training-runtime",
        choices=[item.value for item in TrainingRuntime],
        help="Explicit training runtime; must match the planned backend (default: inferred).",
    )
    parser.add_argument(
        "--gpu-count",
        required=True,
        type=int,
        help="Number of repeated manual device profiles.",
    )
    parser.add_argument(
        "--vram-gib",
        required=True,
        type=float,
        help="Total memory per declared device in GiB.",
    )
    parser.add_argument(
        "--free-vram-gib",
        type=float,
        help="Optional current free memory per device in GiB.",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Declare BF16 support on every device."
    )
    parser.add_argument(
        "--four-bit",
        action="store_true",
        help="Declare the four-bit base-load path supported.",
    )
    parser.add_argument(
        "--eight-bit",
        action="store_true",
        help="Declare the eight-bit base-load path supported.",
    )
    parser.add_argument(
        "--host-ram-gib", required=True, type=float, help="Total host memory in GiB."
    )
    parser.add_argument(
        "--host-ram-free-gib",
        type=float,
        help="Optional current free host memory in GiB.",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=2.0,
        help="Memory excluded from the fit budget (default: 2; Apple unified memory minimum: 8).",
    )
    parser.add_argument(
        "--disk-free-gib",
        type=float,
        help="Optional current free staging and output disk in GiB.",
    )
    parser.add_argument(
        "--objective",
        default="memory",
        choices=[item.value for item in Objective],
        help="Deterministic ranking policy (default: memory).",
    )
    parser.add_argument(
        "--sequence-length",
        required=True,
        type=int,
        help="Compiled maximum token sequence length.",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=16,
        help="Required exact global effective batch (default: 16).",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Maximum training epochs (default: 3)."
    )
    parser.add_argument(
        "--prefer-method",
        choices=[item.value for item in Method],
        help="Secondary method preference; cannot override feasibility.",
    )
    parser.add_argument(
        "--evaluation-fraction",
        type=float,
        default=0.1,
        help="Requested full-run evaluation fraction (default: 0.1).",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=100,
        help="Checkpoint interval in optimizer steps (default: 100).",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Request sequence packing; unsupported and fail-closed in Aptus 0.2.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aptus",
        description="Evidence-backed fine-tuning planner and artifact compiler.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="Profile a local training dataset.")
    profile.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Local JSON, JSONL, CSV, or text data.",
    )
    profile.add_argument(
        "--sample-limit",
        type=int,
        default=512,
        help="Rows used for deterministic length statistics (default: 512).",
    )
    profile.add_argument(
        "--sequence-length", type=int, help="Optional truncation-analysis token limit."
    )
    profile.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of standard output.",
    )

    for name, help_text in (
        ("spec-plan", "Write a persisted v4 plan JSON without compiling."),
        ("plan", "Compatibility flow: plan, compile, validate, and archive."),
        ("build", "Plan, compile, validate, and archive."),
    ):
        command = commands.add_parser(name, help=help_text)
        _add_fact_arguments(command)
        command.add_argument(
            "--output",
            required=True,
            type=Path,
            help="Plan JSON for spec-plan; no-clobber bundle directory otherwise.",
        )
        if name in {"plan", "build"}:
            command.add_argument(
                "--plan-output",
                type=Path,
                help="Optional path for the standalone plan JSON.",
            )

    compile_command = commands.add_parser(
        "compile", help="Compile a persisted plan JSON into a portable bundle."
    )
    compile_command.add_argument(
        "--plan", required=True, type=Path, help="Persisted Aptus v4 plan JSON."
    )
    compile_command.add_argument(
        "--output", required=True, type=Path, help="New or empty bundle directory."
    )
    compile_command.add_argument(
        "--archive", type=Path, help="Optional no-clobber ZIP path outside the bundle."
    )

    validate = commands.add_parser(
        "validate", help="Validate a bundle at one explicit evidence level."
    )
    validate.add_argument("bundle", type=Path, help="Compiled bundle directory.")
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
        help="Required evidence level (default: static).",
    )
    validate.add_argument(
        "--run",
        action="store_true",
        help="Execute checks required above static validation.",
    )
    validate.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".aptus-state"),
        help="Managed job state root for --run (default: .aptus-state).",
    )

    run = commands.add_parser(
        "run",
        help="Start one ordered dependency, model-data, preflight, pilot, or training job.",
    )
    run.add_argument("bundle", type=Path, help="Compiled bundle directory.")
    run.add_argument(
        "--action",
        choices=("dependency", "model-data", "preflight", "pilot", "train"),
        default="preflight",
        help="Ordered runtime action (default: preflight).",
    )
    run.add_argument(
        "--confirm-full-train",
        action="store_true",
        help="Required explicit confirmation for the train action.",
    )
    run.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".aptus-state"),
        help="Managed job state root (default: .aptus-state).",
    )

    jobs = commands.add_parser("jobs", help="List or inspect persisted local jobs.")
    jobs.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".aptus-state"),
        help="Managed job state root (default: .aptus-state).",
    )
    jobs.add_argument("--id", help="Return one reconciled job instead of the job list.")

    doctor = commands.add_parser(
        "doctor", help="Inspect local training-runtime readiness without changing it."
    )
    doctor.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".aptus-state"),
        help="Local state root to summarize (default: .aptus-state).",
    )
    doctor.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of standard output.",
    )

    diagnostics = commands.add_parser(
        "diagnostics", help="Create a privacy-bounded support archive."
    )
    diagnostics.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".aptus-state"),
        help="Local state root to summarize (default: .aptus-state).",
    )
    diagnostics.add_argument(
        "--output", required=True, type=Path, help="New diagnostic ZIP path."
    )

    serve = commands.add_parser(
        "serve", help="Serve the local API and built React app from one origin."
    )
    serve.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)."
    )
    serve.add_argument(
        "--port", type=int, default=8787, help="Bind port (default: 8787)."
    )
    serve.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".aptus-state"),
        help="Plans and managed job state root (default: .aptus-state).",
    )
    serve.add_argument(
        "--web-dist", type=Path, help="Optional workbench build directory override."
    )
    serve.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Acknowledge that the trusted-user jobs API is unsafe to expose without an external security boundary.",
    )

    commands.add_parser(
        "hardware",
        help="Inspect local CUDA hardware or fail-closed Apple Silicon inventory.",
    )
    inspect = commands.add_parser(
        "inspect", help="Inspect local hardware or bounded provider model facts."
    )
    inspect_commands = inspect.add_subparsers(dest="inspect_command", required=True)
    inspect_commands.add_parser("hardware")
    model = inspect_commands.add_parser("model")
    model.add_argument(
        "--model-id", required=True, help="Provider repository ID, such as org/model."
    )
    model.add_argument(
        "--revision", required=True, help="Immutable provider commit to inspect."
    )
    model.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: 10).",
    )
    return parser


def _make_plan(arguments: argparse.Namespace) -> Any:
    if not arguments.confirm_training_allowed:
        raise ValueError("Model training permission must be explicitly confirmed.")
    backend = Backend(arguments.backend)
    training_runtime = (
        TrainingRuntime(arguments.training_runtime)
        if arguments.training_runtime is not None
        else None
    )
    if training_runtime is not None:
        expected_backend = {
            TrainingRuntime.TRANSFORMERS_PEFT_CUDA: Backend.CUDA,
            TrainingRuntime.MLX_LM: Backend.MPS,
            TrainingRuntime.PYTORCH_MPS: Backend.MPS,
        }[training_runtime]
        if backend != expected_backend:
            raise ValueError(
                f"Training runtime {training_runtime.value} requires "
                f"--backend {expected_backend.value}."
            )
    dataset = profile_dataset(
        arguments.dataset,
        sample_limit=arguments.sample_limit,
        sequence_length=arguments.sequence_length,
    )
    moe_required = {
        "--moe-expert-count": arguments.moe_expert_count,
        "--moe-experts-per-token": arguments.moe_experts_per_token,
        "--moe-expert-intermediate-size": arguments.moe_expert_intermediate_size,
        "--moe-decoder-sparse-step": arguments.moe_decoder_sparse_step,
    }
    moe_requested = any(
        value is not None
        for value in (
            *moe_required.values(),
            arguments.moe_shared_expert_intermediate_size,
            *arguments.moe_mlp_only_layer,
        )
    )
    missing_moe = [name for name, value in moe_required.items() if value is None]
    if moe_requested and missing_moe:
        raise ValueError("MoE topology requires " + ", ".join(missing_moe) + ".")
    moe = (
        MoETopology(
            expert_count=arguments.moe_expert_count,
            experts_per_token=arguments.moe_experts_per_token,
            expert_intermediate_size=arguments.moe_expert_intermediate_size,
            decoder_sparse_step=arguments.moe_decoder_sparse_step,
            mlp_only_layers=tuple(arguments.moe_mlp_only_layer),
            shared_expert_intermediate_size=(
                arguments.moe_shared_expert_intermediate_size
            ),
        )
        if moe_requested
        else None
    )
    model_arguments: dict[str, Any] = {
        "model_id": arguments.model_id,
        "revision": arguments.revision,
        "family": arguments.family,
        "parameters_b": arguments.parameters_b,
        "hidden_size": arguments.hidden_size,
        "intermediate_size": arguments.intermediate_size,
        "layers": arguments.layers,
        "context_length": arguments.context_length,
        "license_name": arguments.license,
        "training_allowed": True,
    }
    if arguments.model_type is not None:
        model_arguments["model_type"] = arguments.model_type
    if arguments.architecture is not None:
        model_arguments["architecture"] = arguments.architecture
    if arguments.quantization_bits is not None:
        model_arguments["quantization_bits"] = arguments.quantization_bits
    if arguments.quantization_layout_profile is not None:
        layout = reviewed_qwen3_moe_quantization_layout(arguments.layers)
        model_arguments.setdefault("quantization_bits", layout.default_bits)
        model_arguments["quantization_layout"] = layout
    if moe is not None:
        model_arguments["moe"] = moe
    model = build_model_spec(
        **model_arguments,
    )
    reserve_gib = arguments.reserve_gib
    if backend == Backend.MPS:
        reserve_gib = max(reserve_gib, 8.0)
    hardware = build_hardware_spec(
        backend=backend,
        gpu_count=arguments.gpu_count,
        vram_gib=arguments.vram_gib,
        supports_bf16=arguments.bf16,
        supports_4bit=arguments.four_bit,
        supports_8bit=arguments.eight_bit,
        free_vram_gib=arguments.free_vram_gib,
        host_ram_gib=arguments.host_ram_gib,
        host_ram_free_gib=arguments.host_ram_free_gib,
        reserve_gib=reserve_gib,
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
        training_runtime=training_runtime,
    )
    inspection_receipt = None
    if arguments.inspection_receipt is not None:
        receipt_value = json.loads(
            arguments.inspection_receipt.read_text(encoding="utf-8")
        )
        if isinstance(receipt_value, Mapping) and "status" in receipt_value:
            if receipt_value.get("status") != "ok":
                raise ValueError(
                    "Only successful model inspection output can supply a receipt."
                )
            nested_receipt = receipt_value.get("inspection_receipt")
            if not isinstance(nested_receipt, Mapping):
                raise ValueError(
                    "Inspection output does not contain a usable inspection_receipt."
                )
            receipt_value = nested_receipt
        inspection_receipt = model_inspection_receipt_from_primitive(receipt_value)
    return plan_training(
        model=model,
        dataset=dataset,
        hardware=hardware,
        target=target,
        inspection_receipt=inspection_receipt,
    )


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
        if isinstance(value, Mapping) and value.get("schema_version") == SCHEMA_VERSION:
            require_current_model_policy(value)
            validation_errors = validate_plan_payload(value)
            if validation_errors:
                raise ValueError(
                    "Persisted plan failed the current executable contract: "
                    + " ".join(validation_errors)
                )
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
    if arguments.command == "doctor":
        from .diagnostics import build_doctor_report

        report = build_doctor_report(arguments.state_dir)
        _write_json(report, arguments.output)
        return 0 if report["status"] == "ready" else 2
    if arguments.command == "diagnostics":
        from .diagnostics import create_diagnostic_archive

        archive = create_diagnostic_archive(arguments.state_dir, arguments.output)
        _write_json({"archive_path": str(archive)}, None)
        return 0
    if arguments.command == "serve":
        if arguments.host not in {"127.0.0.1", "localhost", "::1"}:
            if not arguments.allow_non_loopback:
                raise ValueError(
                    "Non-loopback serving is blocked by default. Add --allow-non-loopback only behind authentication, "
                    "approved bundle roots, and worker isolation."
                )
            print(
                "Aptus warning: non-loopback serving sends the session token over plain HTTP. "
                "Provide an approved TLS and network boundary before accepting traffic.",
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
        session_token = secrets.token_urlsafe(32)
        application = create_app(
            state_dir=arguments.state_dir,
            static_dir=arguments.web_dist,
            allowed_hosts=allowed_hosts,
            session_token=session_token,
        )
        display_host = (
            f"[{arguments.host}]" if ":" in arguments.host else arguments.host
        )
        workbench_url = (
            f"http://{display_host}:{arguments.port}/"
            f"?aptus_session_token={session_token}"
        )
        print(f"Aptus workbench: {workbench_url}", file=sys.stderr)
        print(f"Aptus API bearer token: {session_token}", file=sys.stderr)
        uvicorn.run(
            application,
            host=arguments.host,
            port=arguments.port,
            access_log=False,
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
