#!/usr/bin/env python3
"""Execute the MLX-LM compiler slice selected by an Aptus plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
from plan_contract import validate_bundle_manifest, validate_plan_payload


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    errors = validate_plan_payload(plan, root=ROOT, verify_dataset=True)
    errors += validate_bundle_manifest(ROOT)
    if errors:
        raise RuntimeError("Invalid Aptus bundle: " + " | ".join(errors))
    candidate = plan["recommended"]
    runtime = candidate.get("runtime_contract")
    if (
        not isinstance(runtime, dict)
        or runtime.get("training_runtime") != "mlx-lm"
        or runtime.get("compute_backend") != "mps"
        or candidate.get("distribution") != "single"
        or candidate.get("method") not in {"lora", "qlora"}
    ):
        raise RuntimeError("The selected candidate is not an executable MLX-LM contract.")
    return plan, candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_output(path: Path) -> Path:
    unresolved = path if path.is_absolute() else ROOT / path
    resolved = unresolved.resolve()
    allowed = ((ROOT / "runs").resolve(), (ROOT / "pilot-output").resolve())
    if not any(parent == resolved or parent in resolved.parents for parent in allowed):
        raise RuntimeError("MLX adapter output must remain under runs/ or pilot-output/.")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def require_data(path: Path) -> Path:
    unresolved = path if path.is_absolute() else ROOT / path
    resolved = unresolved.resolve(strict=True)
    expected = (ROOT / "data" / "mlx").resolve(strict=True)
    if resolved != expected:
        raise RuntimeError("The MLX data argument must match the compiler-bound data/mlx directory.")
    for name in ("train.jsonl", "valid.jsonl", "split-contract.json"):
        if not (resolved / name).is_file():
            raise RuntimeError(f"MLX dataset is missing {name}.")
    return resolved


def download_pinned_model(plan: dict[str, Any], requested_model: str) -> Path:
    model = plan["model"]
    if requested_model != model["model_id"]:
        raise RuntimeError("The model argument must equal the plan-bound provider model ID.")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=requested_model,
            revision=model["revision"],
        )
    ).resolve(strict=True)


def require_method_model(candidate: dict[str, Any], model_path: Path) -> None:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_file"):
        raise RuntimeError(
            "MLX-LM custom model_file code is unsupported; Aptus only executes pinned built-in MLX model implementations."
        )
    quantization = config.get("quantization") or config.get("quantization_config")
    text_config = config.get("text_config")
    if not quantization and isinstance(text_config, dict):
        quantization = text_config.get("quantization_config")
    bits = quantization.get("bits") if isinstance(quantization, dict) else None
    if candidate["method"] == "qlora" and bits != 4:
        raise RuntimeError(
            "MLX-LM QLoRA requires a pinned model revision with explicit four-bit MLX quantization metadata. Aptus will not substitute bitsandbytes or quantize an unbound model during training."
        )
    if candidate["method"] == "lora" and quantization:
        raise RuntimeError(
            "MLX-LM LoRA requires an unquantized pinned base model. A quantized base "
            "would execute QLoRA semantics under the wrong planned method."
        )


def current_available_unified_memory_bytes() -> int:
    try:
        completed = subprocess.run(
            ["/usr/bin/vm_stat"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Current Apple unified-memory admission probe failed.") from error
    if completed.returncode:
        raise RuntimeError("Current Apple unified-memory admission probe failed.")
    page_match = re.search(r"page size of\s+(\d+) bytes", completed.stdout)
    if page_match is None:
        raise RuntimeError("vm_stat did not report its page size.")
    counts: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        match = re.fullmatch(r"([^:]+):\s*([0-9]+)\.", line.strip())
        if match is not None:
            counts[match.group(1)] = int(match.group(2))
    names = ("Pages free", "Pages inactive", "Pages speculative")
    if not any(name in counts for name in names):
        raise RuntimeError("vm_stat did not report available-memory page classes.")
    available = sum(counts.get(name, 0) for name in names) * int(page_match.group(1))
    if available <= 0:
        raise RuntimeError("Current available Apple unified memory is zero or unknown.")
    return available


def require_unified_memory_admission(plan: dict[str, Any]) -> dict[str, Any]:
    candidate = plan["recommended"]
    memory = candidate["memory"]
    point = int(memory["point_estimate_bytes"])
    upper = int(memory["upper_estimate_bytes"])
    reserve = max(
        int(plan["hardware"].get("reserve_per_device_bytes", 0)),
        8 * 1024**3,
    )
    available = current_available_unified_memory_bytes()
    required = max(point, upper) + reserve
    if available < required:
        raise RuntimeError(
            "Current available Apple unified memory is below the candidate upper "
            "estimate plus the required 8 GiB Aptus reserve."
        )
    return {
        "schema_version": "aptus.mlx-unified-memory-admission.v1",
        "available_unified_memory_bytes": available,
        "point_estimate_bytes": point,
        "upper_estimate_bytes": upper,
        "reserve_bytes": reserve,
        "required_available_bytes": required,
    }


def resolve_lora_keys(model: Any, candidate: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    planned = candidate.get("target_modules")
    if (
        not isinstance(planned, list)
        or not planned
        or any(not isinstance(target, str) or not target for target in planned)
        or len(set(planned)) != len(planned)
    ):
        raise RuntimeError("The MLX-LM candidate requires unique planned target modules.")
    layers = tuple(getattr(model, "layers", ()))
    if not layers:
        raise RuntimeError("The loaded MLX-LM model exposes no transformer layers.")
    resolved: dict[str, str] = {}
    for target in planned:
        observed: list[str] = []
        for layer_index, layer in enumerate(layers):
            matches = sorted(
                name
                for name, _module in layer.named_modules()
                if name == target or name.endswith("." + target)
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"Planned MLX target {target!r} matched {len(matches)} modules "
                    f"in transformer layer {layer_index}; exactly one is required."
                )
            observed.append(matches[0])
        if len(set(observed)) != 1:
            raise RuntimeError(
                f"Planned MLX target {target!r} does not resolve to one stable layer-relative key."
            )
        resolved[target] = observed[0]
    resolved_keys = [resolved[target] for target in planned]
    if len(set(resolved_keys)) != len(resolved_keys):
        raise RuntimeError("Distinct planned MLX targets resolve to the same runtime key.")
    binding = {
        "schema_version": "aptus.mlx-trainable-target-binding.v1",
        "planned_target_modules": planned,
        "resolved_layer_keys": resolved_keys,
        "transformer_layer_count": len(layers),
        "expected_adapter_target_instance_count": len(layers) * len(planned),
    }
    return resolved_keys, binding


def require_trainable_binding(
    names: list[str], binding: dict[str, Any]
) -> dict[str, Any]:
    planned = binding["planned_target_modules"]
    pairs: dict[str, set[str]] = {}
    target_counts = {target: 0 for target in planned}
    for name in names:
        suffix = next(
            (suffix for suffix in (".lora_a", ".lora_b") if name.endswith(suffix)),
            None,
        )
        if suffix is None:
            raise RuntimeError(
                "MLX-LM left a non-LoRA parameter trainable in the bounded smoke."
            )
        base = name[: -len(suffix)]
        matches = [
            target
            for target in planned
            if base == target or base.endswith("." + target)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "A trainable MLX adapter parameter does not bind exactly one planned target."
            )
        pairs.setdefault(base, set()).add(suffix.removeprefix("."))
    if not pairs or any(kinds != {"lora_a", "lora_b"} for kinds in pairs.values()):
        raise RuntimeError("Every planned MLX adapter instance requires one LoRA A/B pair.")
    for base in pairs:
        target = next(
            target
            for target in planned
            if base == target or base.endswith("." + target)
        )
        target_counts[target] += 1
    layer_count = binding["transformer_layer_count"]
    if (
        len(pairs) != binding["expected_adapter_target_instance_count"]
        or any(count != layer_count for count in target_counts.values())
    ):
        raise RuntimeError(
            "The MLX trainable adapter set does not cover every planned target in every layer."
        )
    descriptor = {
        **binding,
        "adapter_target_instance_count": len(pairs),
        "trainable_tensor_count": len(names),
        "target_instance_counts": target_counts,
    }
    descriptor["descriptor_sha256"] = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return descriptor


def derive_iterations(
    *,
    action: str,
    requested_iterations: int,
    candidate: dict[str, Any],
    plan: dict[str, Any],
    train_examples: int,
) -> int:
    accumulation = int(candidate["gradient_accumulation_steps"])
    if accumulation <= 0 or requested_iterations <= 0 or train_examples <= 0:
        raise RuntimeError("MLX-LM iteration inputs must be positive.")
    if action == "bounded-smoke":
        iterations = max(requested_iterations, accumulation)
        if iterations > 8:
            raise RuntimeError(
                "The planned gradient accumulation exceeds the eight-iteration measured-preflight bound."
            )
        return iterations
    if action == "pilot":
        # Keep the pilot bounded and deterministic while proving two complete updates.
        return 2 * accumulation
    if action == "full":
        micro_batch = int(candidate["micro_batch_size"])
        max_epochs = int(plan["target"]["max_epochs"])
        if micro_batch <= 0 or max_epochs <= 0:
            raise RuntimeError("MLX-LM full-run batch and epoch values must be positive.")
        if train_examples < micro_batch:
            raise RuntimeError("MLX-LM full training has no complete micro-batch.")
        batches_per_epoch = train_examples // micro_batch
        epoch_iterations = batches_per_epoch * max_epochs
        return math.ceil(epoch_iterations / accumulation) * accumulation
    raise RuntimeError("Unknown MLX-LM training action.")


def load_pinned_local_model(
    loader: Any,
    requested_model: str,
    *args: Any,
    model_path: Path,
    plan: dict[str, Any],
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    try:
        requested_path = Path(requested_model).resolve(strict=True)
        expected_path = model_path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("MLX-LM attempted to load a missing model path.") from error
    tokenizer_config = kwargs.get("tokenizer_config")
    if (
        requested_path != expected_path
        or args
        or set(kwargs) != {"tokenizer_config"}
        or not isinstance(tokenizer_config, dict)
        or tokenizer_config != {"trust_remote_code": True}
    ):
        raise RuntimeError(
            "Pinned MLX-LM model loading changed shape; Aptus refuses unbound loader arguments."
        )
    binding = {
        "schema_version": "aptus.mlx-model-load-binding.v1",
        "model_id": plan["model"]["model_id"],
        "model_revision": plan["model"]["revision"],
        "resolved_local_snapshot": True,
        "trust_remote_code": False,
    }
    loaded = loader(str(expected_path), tokenizer_config={"trust_remote_code": False})
    return loaded, binding


def run_smoke(arguments: argparse.Namespace) -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX-LM execution requires Apple silicon macOS.")
    plan, candidate = load_contract()
    actions = {
        "bounded-smoke": arguments.bounded_smoke,
        "pilot": arguments.pilot,
        "full": arguments.confirm_full_train,
    }
    selected = [name for name, enabled in actions.items() if enabled]
    if len(selected) != 1:
        raise RuntimeError("Choose exactly one MLX-LM training action.")
    action = selected[0]
    if arguments.resume_from is not None:
        raise RuntimeError(
            "MLX-LM resume is unsupported. Aptus runs this path uninterrupted from scratch."
        )
    data_path = require_data(arguments.data)
    adapter_path = require_output(arguments.adapter_path)
    model_path = download_pinned_model(plan, arguments.model)
    require_method_model(candidate, model_path)

    import mlx.core as mx
    from mlx_lm import lora
    from mlx.utils import tree_flatten

    mx.reset_peak_memory()
    configured_keys = candidate.get("target_modules")
    expected_scale = float(candidate["alpha"]) / int(candidate["rank"])
    evidence: dict[str, Any] = {"train_losses": [], "validation_losses": []}
    original_load = lora.load
    original_linear_to_lora_layers = lora.linear_to_lora_layers
    original_train = lora.train
    original_get_reporting_callbacks = lora.get_reporting_callbacks

    class EvidenceCallback:
        def __init__(self) -> None:
            self.delegate = None

        def on_train_loss_report(self, info: dict[str, Any]) -> None:
            loss = float(info.get("train_loss", float("nan")))
            evidence["train_losses"].append(loss)
            if self.delegate is not None:
                self.delegate.on_train_loss_report(info)

        def on_val_loss_report(self, info: dict[str, Any]) -> None:
            loss = float(info.get("val_loss", float("nan")))
            evidence["validation_losses"].append(loss)
            if self.delegate is not None:
                self.delegate.on_val_loss_report(info)

    callback = EvidenceCallback()

    def pinned_local_load(requested_model: str, *args: Any, **kwargs: Any) -> Any:
        loaded, binding = load_pinned_local_model(
            original_load,
            requested_model,
            *args,
            model_path=model_path,
            plan=plan,
            **kwargs,
        )
        evidence["model_load_binding"] = binding
        return loaded

    def reporting_callbacks(*args: Any, **kwargs: Any) -> EvidenceCallback:
        callback.delegate = original_get_reporting_callbacks(*args, **kwargs)
        return callback

    def linear_to_lora_layers(
        model: Any,
        num_layers: int,
        config: dict[str, Any],
        use_dora: bool = False,
    ) -> None:
        if config.get("keys") != configured_keys:
            raise RuntimeError(
                "MLX-LM LoRA keys do not equal the plan-bound target modules."
            )
        if (
            config.get("rank") != candidate["rank"]
            or float(config.get("scale", float("nan"))) != expected_scale
        ):
            raise RuntimeError("MLX-LM LoRA rank or alpha/r scale violates the plan.")
        resolved_keys, binding = resolve_lora_keys(model, candidate)
        config["keys"] = resolved_keys
        evidence["resolved_binding"] = binding
        original_linear_to_lora_layers(
            model, num_layers, config, use_dora=use_dora
        )

    def instrumented_train(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        training_args = kwargs.get("args")
        if model is None or training_args is None:
            raise RuntimeError("Pinned MLX-LM train invocation changed shape.")
        update_opportunities = (
            int(training_args.iters) // int(training_args.grad_accumulation_steps)
        )
        if update_opportunities < 1:
            raise RuntimeError(
                "The bounded MLX smoke schedules no optimizer update after gradient accumulation."
            )
        before = dict(tree_flatten(model.trainable_parameters()))
        binding = require_trainable_binding(
            sorted(before), evidence.get("resolved_binding", {})
        )
        mx.eval(*before.values())
        optimizer = kwargs.get("optimizer")
        if optimizer is None or not hasattr(optimizer, "step"):
            raise RuntimeError("Pinned MLX-LM optimizer exposes no step counter.")
        mx.eval(optimizer.step)
        optimizer_step_before = int(optimizer.step.item())
        kwargs["training_callback"] = callback
        result = original_train(*args, **kwargs)
        after = dict(tree_flatten(model.trainable_parameters()))
        if set(after) != set(before):
            raise RuntimeError("The MLX trainable parameter set changed during training.")
        mx.eval(*after.values())
        mx.eval(optimizer.step)
        completed_optimizer_updates = int(optimizer.step.item()) - optimizer_step_before
        if completed_optimizer_updates != update_opportunities:
            raise RuntimeError(
                "MLX-LM optimizer step count does not equal the scheduled update count."
            )
        deltas = []
        for name in sorted(before):
            delta = float(mx.sum(mx.abs(after[name] - before[name])).item())
            if not math.isfinite(delta) or delta < 0:
                raise RuntimeError("MLX-LM produced a non-finite adapter delta.")
            deltas.append(delta)
        delta_l1 = sum(deltas)
        if not math.isfinite(delta_l1) or delta_l1 <= 0:
            raise RuntimeError(
                "MLX-LM produced no nonzero adapter delta; an optimizer update is unproven."
            )
        evidence.update(
            trainable_target_binding=binding,
            optimizer_update_opportunities=update_opportunities,
            completed_optimizer_updates=completed_optimizer_updates,
            optimizer_update_observed=(completed_optimizer_updates > 0),
            adapter_delta_l1=delta_l1,
            changed_adapter_tensor_count=sum(delta > 0 for delta in deltas),
            trainable_parameter_names=sorted(after),
        )
        return result

    lora.load = pinned_local_load
    lora.linear_to_lora_layers = linear_to_lora_layers
    lora.train = instrumented_train
    lora.get_reporting_callbacks = reporting_callbacks
    previous_argv = sys.argv
    accumulation = int(candidate["gradient_accumulation_steps"])
    train_examples = sum(
        1
        for line in (data_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    valid_examples = sum(
        1
        for line in (data_path / "valid.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    split_contract = json.loads(
        (data_path / "split-contract.json").read_text(encoding="utf-8")
    )
    split_values = split_contract.get("splits", {})
    if (
        split_contract.get("schema_version") != "aptus.mlx-split.v1"
        or split_contract.get("micro_batch_size") != candidate["micro_batch_size"]
        or split_values.get("train", {}).get("compiled_row_count") != train_examples
        or split_values.get("valid", {}).get("compiled_row_count") != valid_examples
        or train_examples < int(candidate["micro_batch_size"])
        or valid_examples < int(candidate["micro_batch_size"])
        or train_examples % int(candidate["micro_batch_size"])
        or valid_examples % int(candidate["micro_batch_size"])
    ):
        raise RuntimeError("Compiled MLX split counts do not match their bound contract.")
    source_train_examples = split_values["train"]["source_row_count"]
    source_validation_examples = split_values["valid"]["source_row_count"]
    required_iterations = derive_iterations(
        action=action,
        requested_iterations=int(arguments.iters),
        candidate=candidate,
        plan=plan,
        train_examples=train_examples,
    )
    if required_iterations <= 0:
        raise RuntimeError("MLX-LM derived no training iterations from the compiled data.")
    sys.argv = [
        "mlx_lm.lora",
        "--config",
        str(ROOT / "config" / "mlx-lm.yaml"),
        "--model",
        str(model_path),
        "--data",
        str(data_path),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(required_iterations),
        "--save-every",
        str(required_iterations + 1),
        "--train",
    ]
    memory_admission = require_unified_memory_admission(plan)
    try:
        lora.main()
    finally:
        sys.argv = previous_argv
        lora.load = original_load
        lora.linear_to_lora_layers = original_linear_to_lora_layers
        lora.train = original_train
        lora.get_reporting_callbacks = original_get_reporting_callbacks
    adapter_file = adapter_path / "adapters.safetensors"
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_file.is_file() or not adapter_config.is_file():
        raise RuntimeError("MLX-LM did not emit the required adapter artifact pair.")
    losses = evidence.get("train_losses")
    if (
        not isinstance(losses, list)
        or not losses
        or any(not math.isfinite(loss) for loss in losses)
    ):
        raise RuntimeError("MLX-LM did not report a finite measured training loss.")
    if evidence.get("optimizer_update_observed") is not True:
        raise RuntimeError("MLX-LM did not prove a non-skipped optimizer update.")
    if action == "pilot" and evidence.get("completed_optimizer_updates", 0) < 2:
        raise RuntimeError("MLX-LM pilot requires at least two completed optimizer updates.")
    validation_losses = evidence.get("validation_losses")
    if (
        valid_examples > 0
        and (
            not isinstance(validation_losses, list)
            or not validation_losses
            or any(not math.isfinite(loss) for loss in validation_losses)
        )
    ):
        raise RuntimeError("MLX-LM did not report finite validation loss evidence.")
    saved_parameters = mx.load(str(adapter_file))
    if sorted(saved_parameters) != evidence.get("trainable_parameter_names"):
        raise RuntimeError(
            "The saved MLX adapter does not exactly match the proven trainable set."
        )
    emitted_config = json.loads(adapter_config.read_text(encoding="utf-8"))
    emitted_lora = emitted_config.get("lora_parameters")
    expected_resolved = evidence["trainable_target_binding"]["resolved_layer_keys"]
    if (
        not isinstance(emitted_lora, dict)
        or emitted_lora.get("keys") != expected_resolved
        or emitted_lora.get("rank") != candidate["rank"]
        or float(emitted_lora.get("scale", float("nan"))) != expected_scale
    ):
        raise RuntimeError("The emitted MLX adapter config is not plan-bound.")
    manifest = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in (adapter_config, adapter_file)
    ]
    scope = {
        "bounded-smoke": "bounded-compiler-smoke-not-pilot-evidence",
        "pilot": "uninterrupted-pilot",
        "full": "uninterrupted-full-train",
    }[action]
    metrics = {
        "schema_version": "aptus.runtime-metrics.v1",
        "plan_id": plan["plan_id"],
        "candidate_id": candidate["candidate_id"],
        "model_revision": plan["model"]["revision"],
        "dataset_sha256": plan["dataset"]["source_sha256"],
        "method": candidate["method"],
        "training_runtime": "mlx-lm",
        "compute_backend": "mps",
        "compiler_id": candidate["runtime_contract"]["compiler_id"],
        "scope": scope,
        "action": action,
        "execution_semantics": "uninterrupted",
        "resume_supported": False,
        "micro_iterations": required_iterations,
        "global_step": required_iterations,
        "gradient_accumulation_steps": accumulation,
        "optimizer_update_opportunities": evidence["optimizer_update_opportunities"],
        "completed_optimizer_updates": evidence["completed_optimizer_updates"],
        "train_examples": train_examples,
        "validation_examples": valid_examples,
        "source_train_examples": source_train_examples,
        "source_validation_examples": source_validation_examples,
        "max_epochs": int(plan["target"]["max_epochs"]),
        "distribution": "single",
        "actual_world_size": 1,
        "measured_peak_bytes": int(mx.get_peak_memory()),
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "memory_metric_backend": "mlx",
        "model_load_binding": evidence["model_load_binding"],
        "unified_memory_admission": memory_admission,
        "finite_train_loss": True,
        "train_loss_observations": losses,
        "finite_validation_loss": bool(validation_losses) if valid_examples else True,
        "validation_loss_observations": validation_losses,
        "optimizer_update_observed": True,
        "trainable_target_binding": evidence["trainable_target_binding"],
        "adapter_delta_l1": evidence["adapter_delta_l1"],
        "changed_adapter_tensor_count": evidence["changed_adapter_tensor_count"],
        "adapter_path": str(adapter_path.relative_to(ROOT)),
        "adapter_manifest": manifest,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = adapter_path.parent / "training-metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"MLX-LM {action} training completed: {metrics_path}")
    return 0


def main() -> int:
    plan, _candidate = load_contract()
    parser = argparse.ArgumentParser(description="Run the Aptus MLX-LM compiler slice.")
    parser.add_argument("--model", default=plan["model"]["model_id"])
    parser.add_argument("--data", type=Path, default=Path("data/mlx"))
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--bounded-smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--confirm-full-train", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    arguments = parser.parse_args()
    if arguments.iters <= 0:
        parser.error("--iters must be positive.")
    if arguments.resume_from is not None:
        parser.error(
            "--resume-from is unsupported for MLX-LM; runs are uninterrupted from scratch."
        )
    return run_smoke(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
