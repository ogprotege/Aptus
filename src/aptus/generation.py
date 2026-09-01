from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import STACK_VERSIONS, bundle_requirements
from .domain import (
    Provenance,
    TrainingPlan,
    TrainingRuntime,
    ValidationReport,
    to_primitive,
)
from .methods import method_descriptor
from .model_compatibility import (
    current_model_policy_snapshot_bytes,
    current_model_policy_snapshot_sha256,
)
from .plan_contract import validate_bundle_manifest
from .profiling import canonical_training_rows, pilot_sample_rows


_BUNDLE_PROGRAMS = {
    "cuda": (
        "campaign_events.py",
        "train.py",
        "run.py",
        "preflight.py",
        "validate.py",
    ),
    "mlx": (
        "train.py",
        "run.py",
        "reload.py",
        "eval.py",
        "preflight.py",
        "validate.py",
    ),
}


def _bundle_program_bytes(runtime: str, name: str) -> bytes:
    """Read one generated entrypoint from the installed Aptus resources."""

    available = _BUNDLE_PROGRAMS.get(runtime)
    if available is None or name not in available:
        raise ValueError(f"Unknown Aptus bundle program: {runtime}/{name}")
    resource = resources.files("aptus").joinpath("_bundle_programs", runtime, name)
    try:
        return resource.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"Aptus bundle program resource is unavailable: {runtime}/{name}"
        ) from error


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _portable_plan(plan: TrainingPlan, relative_dataset: str) -> TrainingPlan:
    provenance = plan.dataset.provenance
    portable_provenance = None
    if provenance is not None:
        portable_provenance = Provenance(
            kind=provenance.kind,
            source=f"bundle:{relative_dataset}",
            observed_at=provenance.observed_at,
            digest=provenance.digest,
            detail=provenance.detail,
        )
    dataset = replace(
        plan.dataset,
        source_path=Path(relative_dataset),
        bundle_path=relative_dataset,
        provenance=portable_provenance,
    )
    return replace(plan, dataset=dataset)


def _accelerate_config(plan: TrainingPlan) -> str:
    distribution = plan.recommended.distribution.value
    distributed_type = {"single": "NO", "ddp": "MULTI_GPU", "fsdp": "FSDP"}[
        distribution
    ]
    lines = [
        "compute_environment: LOCAL_MACHINE",
        f"distributed_type: {distributed_type}",
        f"mixed_precision: {plan.recommended.precision}",
        f"num_processes: {plan.recommended.world_size}",
        "num_machines: 1",
        "machine_rank: 0",
        "same_network: true",
        "use_cpu: false",
    ]
    if distribution == "fsdp":
        lines.extend(
            (
                "fsdp_config:",
                "  fsdp_version: 1",
                "  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP",
                "  fsdp_backward_prefetch: BACKWARD_PRE",
                "  fsdp_offload_params: false",
                "  fsdp_sharding_strategy: FULL_SHARD",
                "  fsdp_state_dict_type: SHARDED_STATE_DICT",
                "  fsdp_sync_module_states: true",
                "  fsdp_use_orig_params: true",
                "  fsdp_cpu_ram_efficient_loading: true",
            )
        )
    return "\n".join(lines) + "\n"


def _trainer_config(plan: TrainingPlan) -> dict[str, Any]:
    candidate = plan.recommended
    target = plan.target
    descriptor = method_descriptor(candidate.method)
    runtime = candidate.runtime_contract
    is_mlx = bool(runtime and runtime.training_runtime == TrainingRuntime.MLX_LM)
    return {
        "schema_version": "aptus.trainer-config.v3",
        "compiler_id": runtime.compiler_id if runtime else descriptor.compiler_id,
        "export_kind": runtime.export_kind if runtime else descriptor.export_kind,
        "training_runtime": (
            runtime.training_runtime.value
            if runtime
            else TrainingRuntime.TRANSFORMERS_PEFT_CUDA.value
        ),
        "compute_backend": runtime.compute_backend.value if runtime else "cuda",
        "task": target.task,
        "sequence_length": target.sequence_length,
        "packing": target.packing,
        "evaluation_fraction": target.evaluation_fraction,
        "max_epochs": target.max_epochs,
        "per_device_train_batch_size": candidate.micro_batch_size,
        "per_device_eval_batch_size": candidate.micro_batch_size,
        "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        "effective_global_batch_size": candidate.effective_batch_size,
        "world_size": candidate.world_size,
        "device_indices": list(candidate.device_indices),
        "learning_rate": candidate.learning_rate,
        "optimizer": "adamw" if is_mlx else "adamw_torch",
        "lr_scheduler_type": None if is_mlx else "linear",
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "max_grad_norm": None if is_mlx else 1.0,
        "precision": candidate.precision,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": (
            not is_mlx
            and candidate.method.value == "qlora"
            and candidate.distribution.value == "single"
        ),
        "checkpoint_steps": target.checkpoint_steps,
        "logging_steps": min(10, target.checkpoint_steps),
        "save_total_limit": 3,
        "checkpoint_retention_bytes": candidate.checkpoint_retention_bytes,
        "final_export_bytes": candidate.final_export_bytes,
        "report_to": [],
        "remove_unused_columns": False,
        "optimizer_steps": target.optimizer_steps,
        "split_seed": target.split_seed,
        "training_seed": target.training_seed,
        "data_order_seed": target.data_order_seed,
        "counter_contract": {
            "schema_version": "aptus.training-counters.v1",
            "separate_train_and_evaluation": True,
            "fields": [
                "micro_iterations",
                "completed_non_skipped_optimizer_steps",
                "examples_consumed",
                "padded_input_elements",
                "non_padding_tokens",
                "supervised_tokens",
            ],
            "duration_clock": "time.monotonic_ns",
        },
        "pilot_row_limit": max(32, target.effective_batch_size * 2),
        "pilot_dataset_path": "data/pilot-sample.jsonl",
        "training_dataset_path": "data/training.jsonl",
        "truncation_policy": "completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision",
    }


def _mlx_config(plan: TrainingPlan) -> str:
    candidate = plan.recommended
    values = {
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "seed": plan.target.training_seed,
        "num_layers": -1,
        "batch_size": candidate.micro_batch_size,
        "iters": plan.target.optimizer_steps or 2,
        "val_batches": 1,
        "learning_rate": candidate.learning_rate,
        "steps_per_report": 1,
        "steps_per_eval": 2,
        "grad_accumulation_steps": candidate.gradient_accumulation_steps,
        "save_every": 1,
        "max_seq_length": plan.target.sequence_length,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "lora_parameters": {
            "rank": candidate.rank,
            "dropout": 0.0,
            "scale": float(candidate.alpha) / candidate.rank,
            "keys": list(candidate.target_modules),
        },
    }
    lines: list[str] = []
    if plan.model.moe is not None:
        lines.extend(
            (
                "# Aptus MoE policy: attention projections only.",
                "# Expert and router weights remain frozen in this executable profile.",
            )
        )
    for key, value in values.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for nested_key, nested_value in value.items():
                lines.append(f"  {nested_key}: {json.dumps(nested_value)}")
        else:
            lines.append(f"{key}: {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def _mlx_training_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize an Aptus structured row to MLX-LM's masked chat contract."""

    text = row.get("text")
    if isinstance(text, str):
        raise ValueError(
            "MLX-LM compilation refuses text rows because pinned MLX-LM 0.31.3 "
            "cannot combine full-text supervision with the bundle's required prompt masking."
        )
    prompt, completion = row.get("prompt"), row.get("completion")
    if isinstance(prompt, str) or isinstance(completion, str):
        if (
            not isinstance(prompt, str)
            or not isinstance(completion, str)
            or not completion.strip()
        ):
            raise ValueError(
                "MLX-LM prompt/completion rows require a non-empty completion."
            )
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
        }
    instruction, output = row.get("instruction"), row.get("output")
    if isinstance(instruction, str) or isinstance(output, str):
        if (
            not isinstance(instruction, str)
            or not isinstance(output, str)
            or not output.strip()
        ):
            raise ValueError(
                "MLX-LM instruction/output rows require a non-empty output."
            )
        prompt_parts = ["### Instruction:\n" + instruction.strip()]
        input_value = row.get("input")
        if isinstance(input_value, str) and input_value.strip():
            prompt_parts.append("### Input:\n" + input_value.strip())
        prompt_parts.append("### Response:\n")
        return {
            "messages": [
                {"role": "user", "content": "\n".join(prompt_parts)},
                {"role": "assistant", "content": output},
            ]
        }
    messages = row.get("messages")
    if isinstance(messages, list):
        normalized = {"messages": messages}
        if "tools" in row:
            normalized["tools"] = row["tools"]
        return normalized
    content = row.get("content")
    if isinstance(content, str):
        raise ValueError(
            "MLX-LM compilation refuses content-only rows because they have no "
            "separable prompt and completion for masking."
        )
    raise ValueError("MLX-LM compilation encountered an unsupported dataset row.")


def _decision_report(plan: TrainingPlan) -> str:
    runtime = plan.recommended.runtime_contract
    optimizer_policy = (
        "MLX-LM AdamW; the pinned compiler config does not declare a separate learning-rate scheduler or CUDA gradient-scaler policy"
        if runtime and runtime.training_runtime == TrainingRuntime.MLX_LM
        else "adamw_torch, linear scheduler, weight decay 0.0, warmup steps 0, max grad norm 1.0"
    )
    rows = [
        "| Candidate | Method | Distribution | Status | Point GiB | Upper GiB | Batch | Frontier |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in plan.candidates:
        rows.append(
            "| {id} | {method} | {distribution} | {status} | {point:.2f} | {upper:.2f} | {batch} | {frontier} |".format(
                id=candidate.candidate_id,
                method=candidate.method.value,
                distribution=candidate.distribution.value,
                status=candidate.status.value,
                point=candidate.memory.point_estimate_bytes / 1024**3,
                upper=candidate.memory.upper_bytes / 1024**3,
                batch=candidate.effective_batch_size,
                frontier="yes" if candidate.pareto_frontier else "no",
            )
        )
    rationale = "\n".join(f"- {item}" for item in plan.recommendation_rationale)
    warnings = "\n".join(f"- {item}" for item in plan.warnings)
    assumptions = "\n".join(f"- {item}" for item in plan.recommended.assumptions)
    target_modules = ", ".join(plan.recommended.target_modules) or "full model"
    moe_policy = (
        "\n- MoE adapter policy: attention-only QLoRA; expert and router weights remain frozen; all base weights remain resident in unified memory"
        if plan.model.moe is not None
        else ""
    )
    return f"""# Aptus decision report

Selected candidate: `{plan.recommended.candidate_id}`.

Formula: `{plan.formula_version}`. Point estimates sum named point components. Upper estimates sum `component_upper_bounds`, including a separate uncertainty term. The user reserve is not counted as usage. It reduces usable VRAM.

## Decision rationale

{rationale}

## Selected execution contract

- Candidate status: `{plan.recommended.status.value}`
- Training runtime: `{runtime.training_runtime.value if runtime else "transformers-peft-cuda"}`
- Compute backend: `{runtime.compute_backend.value if runtime else "cuda"}`
- Compiler and estimator: `{runtime.compiler_id if runtime else "legacy"}`, `{runtime.estimator_id if runtime else plan.formula_version}`
- Evidence requirement and export: `{runtime.evidence_requirement.value if runtime else "pilot-required"}`, `{runtime.export_kind if runtime else "legacy"}`
- Method: `{plan.recommended.method.value}`
- Distribution and world size: `{plan.recommended.distribution.value}`, `{plan.recommended.world_size}`
- Planned visible device indices: `{", ".join(str(item) for item in plan.recommended.device_indices)}`
- Precision and quantization: `{plan.recommended.precision}`, `{plan.recommended.quantization or "none"}`
- Adapter rank and alpha: `{plan.recommended.rank}`, `{plan.recommended.alpha}`
- Learning rate: `{plan.recommended.learning_rate:g}`
- Optimizer policy: `{optimizer_policy}`
- Truncation policy: completion first, then keep the prompt suffix that fits; refuse rows with no supervised tokens
- Target modules: `{target_modules}`{moe_policy}
- Per-device micro-batch and accumulation: `{plan.recommended.micro_batch_size}`, `{plan.recommended.gradient_accumulation_steps}`
- Required host RAM: `{plan.recommended.required_host_ram_bytes}` bytes
- Required staging and output disk: `{plan.recommended.required_disk_bytes}` bytes
- Checkpoint-retention estimate: `{plan.recommended.checkpoint_retention_bytes}` bytes for up to three retained checkpoints
- Final-export estimate: `{plan.recommended.final_export_bytes}` bytes
- Confidence label: `{plan.recommended.confidence}`
- Pareto frontier: `{str(plan.recommended.pareto_frontier).lower()}`

## Assumptions

{assumptions}

## Candidate comparison

{chr(10).join(rows)}

This ranking does not claim measured throughput or model quality. Those require execution evidence.

## Warnings

{warnings}
"""


def _readme(plan: TrainingPlan) -> str:
    policy_boundary = f"""
## Model-policy snapshot

This `aptus.bundle.v3` bundle carries the canonical
`policy/model-policy-snapshot.v1.json` used by its
`aptus.training-plan.v6` plan. The canonical snapshot bytes hash to digest
`{plan.model_policy_snapshot_sha256}`; the plan, manifest, and manifested file
entry bind that digest. The copied
`policy_snapshot.py` evaluator and `validate.py` can check canonical bytes,
digest and path integrity, and saved decision parity without importing Aptus.

That package-free result proves only the integrity of the frozen policy bundled
with this artifact. It cannot determine whether the policy is current on a
later host. Installed Aptus checks its current registry during managed
admission, pilot authorization, worker launch, and the completion verification
and promotion transaction. If submission reports `replan_required`, or a later
current-policy check fails, preserve this bundle and create a new plan and
bundle; do not edit the snapshot, plan, manifest, or digests.
"""
    if (
        plan.recommended.runtime_contract
        and plan.recommended.runtime_contract.training_runtime == TrainingRuntime.MLX_LM
    ):
        moe_policy = (
            "\nFor this MoE plan, Aptus adapts only `q_proj`, `k_proj`, `v_proj`, and `o_proj`. Expert and router weights remain frozen. The full quantized base still resides in unified memory. Active-parameter counts describe per-token routing, not model residency. Model-data validation records the exact architecture, expert topology, logical total parameters, and logical active parameters before adapter conversion.\n"
            if plan.model.moe is not None
            else ""
        )
        return f"""# Aptus MLX-LM training bundle

This portable bundle contains candidate `{plan.recommended.candidate_id}` from
plan `{plan.plan_id}`. It is compiled for Apple silicon and MLX-LM.
{policy_boundary}

## Before execution

1. Review `decision-report.md`, `plan.json`, and `evidence.jsonl`.
2. Confirm the model revision, data rights, target facts, and selected Apple
   silicon host.
3. Create the Python environment outside this directory. An in-bundle virtual
   environment is an unexpected path and invalidates the manifest.
4. Install the exact direct pins from `requirements.txt`.
5. Follow `runbook.md` in order.

The candidate is conditional and pilot-required. The generated wrapper runs a
bounded compiler smoke, an uninterrupted pilot, or a confirmed uninterrupted
full train from the pinned base revision. Each action rechecks live Apple
unified-memory headroom before loading the model.
{moe_policy}

## Portable commands

```bash
python validate.py --level static
python validate.py --level dependency
python validate.py --level model-data
python validate.py --level measured-preflight
python validate.py --level pilot
python run.py --confirm-full-train --output-dir runs/run_<new-name>
python eval.py --gold GOLD.jsonl --adapter runs/run_<id>/final --output predictions.jsonl
```

`eval.py` writes prediction-only JSONL for `aptus eval`. It is not a
training gate and not a quality claim.

The pilot proves at least two completed optimizer updates, finite train and
validation loss, exact target-module coverage, a positive adapter delta, and a
fresh-process adapter reload with bounded generation. Full training derives
its iteration count from the compiled train split, epoch count, micro-batch,
and gradient accumulation. Successful full runs publish an immutable adapter
manifest and `final-export.json`, then atomically promote `measured-run-pass`.
Failed or cancelled runs never promote.

MLX-LM crash resume is unsupported in this bundle. `--resume-from` always
fails closed. Pilot and full actions start from the pinned base revision and
must run uninterrupted.

## Evidence boundary

`pilot-pass` authorizes a later full-train admission check. It does not
guarantee that current resources still suffice. `measured-run-pass` proves the
bound run, metrics, trainable census, dataset split, immutable adapter export,
and fresh-process reload passed parent verification. It does not prove model
quality, safety, or deployment fitness.
"""
    return f"""# Aptus training bundle

This portable bundle contains candidate `{plan.recommended.candidate_id}` from
plan `{plan.plan_id}`.
{policy_boundary}

## Before execution

1. Review `decision-report.md`, `plan.json`, and `evidence.jsonl`.
2. Confirm the model revision, data rights, target facts, and selected hardware.
3. Create the Python environment outside this directory. An in-bundle virtual
   environment is an unexpected path and invalidates the manifest.
4. Install the exact direct pins from `requirements.txt`.
5. Follow `runbook.md` in order.

The analytic point estimate and heuristic upper envelope are not calibration
results. The selected method must pass dependency, model-data, measured
preflight, and bounded real-model pilot gates before full training.

## Portable commands

```bash
python validate.py --level static
python validate.py --level dependency
python validate.py --level model-data
python validate.py --level measured-preflight
python validate.py --level pilot
python run.py --confirm-full-train
```

Do not launch `train.py` directly. `run.py` owns the full-run lease, aggregate
exit handling, artifact verification, and completion promotion.

## Evidence boundary

`pilot-pass` authorizes a later deep train-admission check. It does not guarantee
that current resources still suffice. `measured-run-pass` proves the bound run,
metrics, trainable census, dataset split, and structural export file tree passed
parent verification. It does not prove model quality, safety, or deployment
fitness.
"""


def _runbook(plan: TrainingPlan) -> str:
    if (
        plan.recommended.runtime_contract
        and plan.recommended.runtime_contract.training_runtime == TrainingRuntime.MLX_LM
    ):
        moe_policy = (
            "\nFor this MoE profile, Aptus uses attention-only QLoRA. Only `q_proj`, `k_proj`, `v_proj`, and `o_proj` receive adapters. Expert and router weights remain frozen. The full quantized base remains resident in unified memory even though each token activates only part of the expert graph. The model-data gate verifies the pinned architecture, all sparse layers, expert count, experts per token, logical total parameters, and logical active parameters before training.\n"
            if plan.model.moe is not None
            else ""
        )
        return f"""# MLX-LM runbook

## 1. Create an external environment

```bash
python -m venv ../aptus-mlx-env
source ../aptus-mlx-env/bin/activate
python -m pip install -r requirements.txt
```

## 2. Validate dependencies

```bash
python validate.py --level dependency
```

## 3. Validate model and data

```bash
python validate.py --level model-data
```

This resolves only the plan-pinned model revision, disables remote model code,
checks the architecture, parameter census, packed checkpoint, target modules,
unified-memory admission, compiled split, tokenization, and prompt masking. It
constructs no optimizer and takes no training step.

## 4. Exercise the compiler slice

```bash
python validate.py --level measured-preflight
```

The validator launches the owned bounded-smoke wrapper, downloads only the
plan-pinned model revision, disables remote model code, binds `data/mlx`, runs
at most eight iterations, and records the exact completed artifact tree plus
runtime-neutral MLX memory metrics in the validation report.

For QLoRA, the pinned model must contain declared MLX quantization metadata
whose bits match the plan (1 through 16). Aptus never substitutes bitsandbytes
and never quantizes an unbound model during training.
{moe_policy}

## 5. Pilot gate

```bash
python validate.py --level pilot
```

The validator launches one owned, uninterrupted pilot from the pinned base.
It promotes `pilot-pass` only after two completed optimizer updates, finite
train and validation loss, exact adapter-target census, positive adapter
change, live headroom, and fresh-process adapter reload plus bounded generation.

Review `pilot-output/metrics.json`, the adapter and reload evidence, and
`validation-report.json`. A pilot failure blocks full training.

## 6. Confirm full training

```bash
python run.py --confirm-full-train --output-dir runs/run_<new-name>
```

Before creating the run directory, the wrapper re-verifies the canonical pilot
report and artifact tree, live unified-memory headroom against the measured
pilot peak plus reserve, and evidence-derived disk headroom. The full action
derives its iteration count from the compiled train split and
`max_epochs`. It writes `metrics.json` last, after the adapter manifest, fresh
reload evidence, and `final-export.json` have passed verification. It then
re-verifies the owned tree and atomically promotes `measured-run-pass`. A failed
or cancelled process leaves an unpromoted owned directory.

## 7. Interpret the result

Read `validation-report.json`, the selected run `metrics.json`, and
`final-export.json`. `measured-run-pass` is operational and structural
evidence. Use a separate evaluation contract before making a quality claim.

## 8. Optional exact-match generate

```bash
python eval.py --gold GOLD.jsonl --adapter runs/run_<id>/final --output predictions.jsonl
```

This greedy CompletionsDataset generate writes only a `prediction` field.
Score it with host `aptus eval`. It is not general quality, safety, or
`measured-run-pass`.

## Resume boundary

MLX-LM optimizer, scheduler, and random-state crash continuation is not
supported. Do not pass `--resume-from`; the wrapper rejects it. Every pilot and
full run starts from the pinned base revision and runs uninterrupted.
"""
    return """# Runbook

## 1. Protect the bundle

Keep the bundle unchanged. Create the isolated environment beside it, never
inside it. For example, while your shell is in the bundle directory:

```bash
python -m venv ../aptus-runtime-env
source ../aptus-runtime-env/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` contains exact direct pins, not a transitive lock. Preserve
the installed-environment binding written by validation.

## 2. Validate dependencies

```bash
python validate.py --level dependency
```

This cumulatively checks the contract and static levels, then verifies the exact
direct requirements in the active environment.

## 3. Validate model and data

```bash
python validate.py --level model-data
```

This loads the pinned model and tokenizer, checks plan-driving architecture
facts and target modules, prepares the selected method, enables its compiled
checkpointing path, enforces the trainable-parameter census, and transforms
every canonical row. It constructs no optimizer and takes no training step.

## 4. Run measured preflight

```bash
python validate.py --level measured-preflight
```

This runs the selected broad method on a small synthetic CUDA model. It performs
a forward pass, backward pass, optimizer step, finite-loss check, census check,
and peak-memory measurement. It is method and kernel evidence, not planned-model
fit evidence.

## 5. Run the real-model pilot

```bash
python validate.py --level pilot
```

Each pilot phase reads only the compiler-produced pressure set. The compiler
supplies at least two effective batches and repeats real rows when needed. Phase
one writes a checkpoint. A fresh process continues from it and completes phase
two. Both phases must report the same plan-bound trainable census.

Review `pilot-output/metrics.json`, the checkpoint and export manifests, the
recorded CUDA peaks, and `validation-report.json`. `pilot-pass` is required
before full training; a pilot failure blocks it.

## 6. Start a unique full run

```bash
python run.py --confirm-full-train
```

Do not launch `train.py` directly. `run.py` rechecks admission, holds the shared
lease, chooses the single or distributed command, waits for aggregate exit,
verifies pending metrics and the structural export, and promotes completion.

New outputs appear under a unique `runs/run_*` directory. The trainer binds the
full canonical dataset, deterministic train and evaluation assignment, rank
evidence, optimizer membership, trainable census, metrics, and export tree.

## 7. Interpret the result

Read `validation-report.json`, the selected run metrics, and the final export
manifest. `measured-run-pass` is operational and structural evidence. Use a
separate evaluation contract before making a quality claim.

## Recovery boundary

Full-training resume is fail-closed in Aptus v0.2. Checkpoints are emitted, but
Aptus will not resume one until a future contract binds complete model,
optimizer, scheduler, scaler, RNG, environment, plan, data progress, and
distributed state. Preserve a failed run for diagnosis and start a new run ID.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(root: Path, plan: TrainingPlan) -> None:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"bundle-manifest.json", "validation-report.json"}:
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    entrypoints = {
        "run": "run.py",
        "train": "train.py",
        "preflight": "preflight.py",
        "validate": "validate.py",
    }
    if (
        plan.recommended.runtime_contract
        and plan.recommended.runtime_contract.training_runtime == TrainingRuntime.MLX_LM
    ):
        entrypoints["reload"] = "reload.py"
        entrypoints["eval"] = "eval.py"
    _write_json(
        root / "bundle-manifest.json",
        {
            "schema_version": "aptus.bundle.v3",
            "compiler": {"name": "aptus", "version": __version__},
            "stack_versions": STACK_VERSIONS,
            "plan_id": plan.plan_id,
            "plan_sha256": _sha256(root / "plan.json"),
            "policy_snapshot_path": "policy/model-policy-snapshot.v1.json",
            "policy_snapshot_sha256": current_model_policy_snapshot_sha256(),
            "candidate_id": plan.recommended.candidate_id,
            "formula_version": plan.formula_version,
            "entrypoints": entrypoints,
            "validation": {
                "levels": [
                    "contract",
                    "static",
                    "dependency",
                    "model-data",
                    "measured-preflight",
                    "pilot",
                ],
                "required_before_full_training": "pilot-pass",
            },
            "files": entries,
        },
    )


def _compile_into(plan: TrainingPlan, root: Path) -> TrainingPlan:
    suffix = plan.dataset.source_path.suffix.lower()
    if suffix not in {".jsonl", ".json", ".csv", ".txt"}:
        raise ValueError(f"Unsupported portable dataset suffix: {suffix or '<none>'}")
    relative_dataset = f"data/dataset{suffix}"
    portable = _portable_plan(plan, relative_dataset)
    policy_snapshot_sha256 = current_model_policy_snapshot_sha256()
    if portable.model_policy_snapshot_sha256 != policy_snapshot_sha256:
        raise ValueError(
            "Plan model policy snapshot digest does not match the current host policy."
        )
    dataset_destination = root / relative_dataset
    dataset_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plan.dataset.source_path, dataset_destination)
    if _sha256(dataset_destination) != plan.dataset.source_sha256:
        raise ValueError(
            "Dataset changed after profiling; re-profile before compiling."
        )
    pilot_row_count = max(32, plan.target.effective_batch_size * 2)
    pressure_rows = pilot_sample_rows(
        replace(plan.dataset, source_path=dataset_destination), limit=pilot_row_count
    )
    pilot_rows = tuple(
        pressure_rows[index % len(pressure_rows)] for index in range(pilot_row_count)
    )
    (root / "data" / "pilot-sample.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in pilot_rows
        ),
        encoding="utf-8",
    )
    training_rows = canonical_training_rows(
        replace(plan.dataset, source_path=dataset_destination)
    )
    is_mlx = bool(
        portable.recommended.runtime_contract
        and portable.recommended.runtime_contract.training_runtime
        == TrainingRuntime.MLX_LM
    )
    mlx_train = mlx_valid = None
    if is_mlx:
        if portable.target.evaluation_fraction == 0:
            raise ValueError(
                "MLX-LM compilation refuses evaluation_fraction=0; it does not invent a validation hold-out."
            )
        if portable.dataset.example_count < 2:
            raise ValueError(
                "MLX-LM compilation requires at least two usable rows for disjoint train and validation files."
            )
        mlx_root = root / "data" / "mlx"
        mlx_root.mkdir(parents=True, exist_ok=True)
        mlx_train = (mlx_root / "train.jsonl").open("w", encoding="utf-8")
        mlx_valid = (mlx_root / "valid.jsonl").open("w", encoding="utf-8")
        valid_count = max(
            1,
            round(portable.dataset.example_count * portable.target.evaluation_fraction),
        )
        valid_start = portable.dataset.example_count - min(
            valid_count, portable.dataset.example_count - 1
        )
    try:
        with (root / "data" / "training.jsonl").open("w", encoding="utf-8") as output:
            for index, row in enumerate(training_rows):
                line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                output.write(line)
                if mlx_train is not None and mlx_valid is not None:
                    try:
                        mlx_row = _mlx_training_row(row)
                    except ValueError as error:
                        raise ValueError(
                            f"MLX-LM dataset row {index + 1}: {error}"
                        ) from error
                    mlx_line = (
                        json.dumps(mlx_row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    (mlx_valid if index >= valid_start else mlx_train).write(mlx_line)
    finally:
        if mlx_train is not None:
            mlx_train.close()
        if mlx_valid is not None:
            mlx_valid.close()
    if is_mlx:
        micro_batch = portable.recommended.micro_batch_size
        split_contract = {
            "schema_version": "aptus.mlx-split.v1",
            "micro_batch_size": micro_batch,
            "padding_policy": "repeat-within-disjoint-split-to-complete-final-batch",
            "splits": {},
        }
        for name in ("train", "valid"):
            path = root / "data" / "mlx" / f"{name}.jsonl"
            rows = [
                line for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
            if not rows:
                raise ValueError(f"MLX-LM {name} split must not be empty.")
            original_count = len(rows)
            padded_count = math.ceil(original_count / micro_batch) * micro_batch
            padded = [rows[index % original_count] for index in range(padded_count)]
            path.write_text("".join(line + "\n" for line in padded), encoding="utf-8")
            split_contract["splits"][name] = {
                "source_row_count": original_count,
                "compiled_row_count": padded_count,
            }
        _write_json(root / "data" / "mlx" / "split-contract.json", split_contract)

    payload = to_primitive(portable)
    _write_json(root / "plan.json", payload)
    policy_snapshot_path = root / "policy" / "model-policy-snapshot.v1.json"
    policy_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    policy_snapshot_path.write_bytes(current_model_policy_snapshot_bytes())
    _write_json(root / "profiles" / "model.json", payload["model"])
    _write_json(root / "profiles" / "dataset.json", payload["dataset"])
    _write_json(root / "profiles" / "hardware.json", payload["hardware"])
    _write_json(root / "candidates.json", payload["candidates"])
    evidence_lines = "".join(
        json.dumps(item, sort_keys=True) + "\n" for item in payload["evidence_records"]
    )
    (root / "evidence.jsonl").write_text(evidence_lines, encoding="utf-8")
    (root / "decision-report.md").write_text(
        _decision_report(portable), encoding="utf-8"
    )
    (root / "README.md").write_text(_readme(portable), encoding="utf-8")
    (root / "runbook.md").write_text(_runbook(portable), encoding="utf-8")
    training_runtime = (
        portable.recommended.runtime_contract.training_runtime
        if portable.recommended.runtime_contract
        else TrainingRuntime.TRANSFORMERS_PEFT_CUDA
    )
    requirements = (
        "\n".join(
            bundle_requirements(
                portable.recommended.method,
                training_runtime=training_runtime,
            )
        )
        + "\n"
    )
    (root / "requirements.txt").write_text(requirements, encoding="utf-8")
    (root / "config").mkdir(parents=True, exist_ok=True)
    accelerate = _accelerate_config(portable)
    (root / "config" / "accelerate.yaml").write_text(accelerate, encoding="utf-8")
    (root / "accelerate_config.yaml").write_text(accelerate, encoding="utf-8")
    _write_json(root / "config" / "trainer.json", _trainer_config(portable))
    if is_mlx:
        (root / "config" / "mlx-lm.yaml").write_text(
            _mlx_config(portable), encoding="utf-8"
        )
    contract_source = (
        resources.files("aptus")
        .joinpath("plan_contract.py")
        .read_text(encoding="utf-8")
    )
    (root / "plan_contract.py").write_text(contract_source, encoding="utf-8")
    policy_snapshot_source = (
        resources.files("aptus")
        .joinpath("policy_snapshot.py")
        .read_text(encoding="utf-8")
    )
    (root / "policy_snapshot.py").write_text(policy_snapshot_source, encoding="utf-8")
    runtime_lease_source = (
        resources.files("aptus")
        .joinpath("runtime_lease.py")
        .read_text(encoding="utf-8")
    )
    (root / "runtime_lease.py").write_text(runtime_lease_source, encoding="utf-8")
    program_runtime = "mlx" if is_mlx else "cuda"
    for name in _BUNDLE_PROGRAMS[program_runtime]:
        (root / name).write_bytes(_bundle_program_bytes(program_runtime, name))
    _write_manifest(root, portable)
    return portable


def generate_bundle(plan: TrainingPlan, output_dir: Path) -> ValidationReport:
    """Compile and statically validate a bundle, publishing it atomically."""

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Bundle output is not empty: {output_dir}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        _compile_into(plan, temporary)
        from .validation import validate_bundle

        report = validate_bundle(temporary, level="static", run=False)
        if report.state.value == "invalid":
            raise ValueError("Generated bundle failed static validation.")
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(temporary, output_dir)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


_ARCHIVE_EXCLUDED_NAMES = frozenset(
    {
        ".validation-report.lock",
        "model-data-evidence.json",
        "preflight-metrics.json",
        "validation-report.json",
    }
)


def _archive_source_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _bundle_archive_sources(
    bundle_dir: Path,
) -> list[tuple[Path, str, tuple[int, ...]]]:
    sources: list[tuple[Path, str, tuple[int, ...]]] = []
    for path in sorted(bundle_dir.rglob("*")):
        relative_path = path.relative_to(bundle_dir)
        relative = relative_path.as_posix()
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ValueError("Bundle archive source inventory is unstable.") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("Bundle archives cannot contain or traverse symlinks.")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(
                "Bundle archive sources must be regular non-hardlinked files."
            )
        if (
            path.name in _ARCHIVE_EXCLUDED_NAMES
            or "pilot-output" in relative_path.parts
            or "runs" in relative_path.parts
        ):
            continue
        sources.append((path, relative, _archive_source_fingerprint(metadata)))
    return sources


def _write_pinned_archive_member(
    archive: zipfile.ZipFile,
    path: Path,
    relative: str,
    expected: tuple[int, ...],
) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if _archive_source_fingerprint(os.fstat(descriptor)) != expected:
            raise ValueError("Bundle archive source changed while opening.")
        info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        with archive.open(info, "w") as destination:
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                destination.write(block)
        if _archive_source_fingerprint(os.fstat(descriptor)) != expected:
            raise ValueError("Bundle archive source changed while reading.")
    finally:
        os.close(descriptor)
    if _archive_source_fingerprint(path.lstat()) != expected:
        raise ValueError("Bundle archive source path changed while reading.")


def _write_bundle_archive(bundle_dir: Path, destination: Path | io.BytesIO) -> None:
    sources = _bundle_archive_sources(bundle_dir)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path, relative, fingerprint in sources:
            _write_pinned_archive_member(archive, path, relative, fingerprint)
    if _bundle_archive_sources(bundle_dir) != sources:
        raise ValueError("Bundle archive source inventory changed during construction.")


def _files_equal(left: Path, right: Path) -> bool:
    left_metadata = left.lstat()
    right_metadata = right.lstat()
    if (
        not stat.S_ISREG(left_metadata.st_mode)
        or stat.S_ISLNK(left_metadata.st_mode)
        or left_metadata.st_nlink != 1
        or not stat.S_ISREG(right_metadata.st_mode)
        or stat.S_ISLNK(right_metadata.st_mode)
        or right_metadata.st_nlink != 1
        or left_metadata.st_size != right_metadata.st_size
    ):
        return False
    left_fingerprint = _archive_source_fingerprint(left_metadata)
    right_fingerprint = _archive_source_fingerprint(right_metadata)
    left_descriptor = os.open(left, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    right_descriptor = os.open(right, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    equal = True
    try:
        if (
            _archive_source_fingerprint(os.fstat(left_descriptor)) != left_fingerprint
            or _archive_source_fingerprint(os.fstat(right_descriptor))
            != right_fingerprint
        ):
            return False
        while True:
            left_block = os.read(left_descriptor, 1024 * 1024)
            right_block = os.read(right_descriptor, 1024 * 1024)
            if left_block != right_block:
                equal = False
                break
            if not left_block:
                break
        if (
            _archive_source_fingerprint(os.fstat(left_descriptor)) != left_fingerprint
            or _archive_source_fingerprint(os.fstat(right_descriptor))
            != right_fingerprint
        ):
            equal = False
    finally:
        os.close(left_descriptor)
        os.close(right_descriptor)
    return bool(
        equal
        and _archive_source_fingerprint(left.lstat()) == left_fingerprint
        and _archive_source_fingerprint(right.lstat()) == right_fingerprint
    )


def verify_bundle_archive(bundle_dir: Path, archive_path: Path) -> bool:
    """Recompute and byte-compare the Aptus archive without loading it in memory."""

    bundle = bundle_dir.resolve(strict=True)
    archive = archive_path.absolute()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".aptus-archive-verify-")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_bundle_archive(bundle, temporary)
        return _files_equal(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)


_MAX_BUNDLE_ARCHIVE_BYTES = 256 * 1024**2
_MAX_BUNDLE_ARCHIVE_MEMBERS = 4096
_MAX_BUNDLE_ARCHIVE_MEMBER_BYTES = 128 * 1024**2
_MAX_BUNDLE_ARCHIVE_TOTAL_BYTES = 512 * 1024**2


def validate_bundle_archive_file(path: Path) -> dict[str, dict[str, Any]]:
    """Stream-validate exact Aptus ZIP bytes and return the member inventory."""

    archive_path = path.absolute()
    try:
        before = archive_path.lstat()
    except OSError as error:
        raise ValueError("Bundle archive is unavailable.") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > _MAX_BUNDLE_ARCHIVE_BYTES
    ):
        raise ValueError("Bundle archive file identity or size is invalid.")
    fingerprint = _archive_source_fingerprint(before)
    descriptor = os.open(archive_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    expected_descriptor, expected_name = tempfile.mkstemp(
        prefix=".aptus-archive-canonical-"
    )
    os.close(expected_descriptor)
    expected_path = Path(expected_name)
    inventory: dict[str, dict[str, Any]] = {}
    validation_succeeded = False
    try:
        if _archive_source_fingerprint(os.fstat(descriptor)) != fingerprint:
            raise ValueError("Bundle archive changed during open.")
        with os.fdopen(os.dup(descriptor), "rb") as source_file:
            with zipfile.ZipFile(source_file, "r") as source:
                infos = source.infolist()
                names = [info.filename for info in infos]
                if (
                    not names
                    or len(names) > _MAX_BUNDLE_ARCHIVE_MEMBERS
                    or names != sorted(names)
                    or len(names) != len(set(names))
                    or source.comment
                ):
                    raise ValueError(
                        "Bundle archive member inventory is not canonical."
                    )
                declared_total = 0
                for info in infos:
                    if (
                        info.file_size < 0
                        or info.file_size > _MAX_BUNDLE_ARCHIVE_MEMBER_BYTES
                    ):
                        raise ValueError("Bundle archive member is unbounded.")
                    declared_total += info.file_size
                    if declared_total > _MAX_BUNDLE_ARCHIVE_TOTAL_BYTES:
                        raise ValueError("Bundle archive expansion is unbounded.")
                with zipfile.ZipFile(
                    expected_path,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                ) as expected:
                    total_size = 0
                    for info in infos:
                        member_path = Path(info.filename)
                        if (
                            info.is_dir()
                            or member_path.is_absolute()
                            or ".." in member_path.parts
                            or "\\" in info.filename
                            or member_path.name in _ARCHIVE_EXCLUDED_NAMES
                            or "pilot-output" in member_path.parts
                            or "runs" in member_path.parts
                        ):
                            raise ValueError("Bundle archive member path is invalid.")
                        canonical_info = zipfile.ZipInfo(
                            info.filename, date_time=(1980, 1, 1, 0, 0, 0)
                        )
                        canonical_info.compress_type = zipfile.ZIP_DEFLATED
                        canonical_info.external_attr = 0o644 << 16
                        digest = hashlib.sha256()
                        size = 0
                        with (
                            source.open(info, "r") as member,
                            expected.open(canonical_info, "w") as destination,
                        ):
                            while True:
                                block = member.read(1024 * 1024)
                                if not block:
                                    break
                                size += len(block)
                                total_size += len(block)
                                if (
                                    size > _MAX_BUNDLE_ARCHIVE_MEMBER_BYTES
                                    or total_size > _MAX_BUNDLE_ARCHIVE_TOTAL_BYTES
                                ):
                                    raise ValueError(
                                        "Bundle archive expansion is unbounded."
                                    )
                                digest.update(block)
                                destination.write(block)
                        if size != info.file_size:
                            raise ValueError(
                                "Bundle archive member size is inconsistent."
                            )
                        inventory[info.filename] = {
                            "sha256": digest.hexdigest(),
                            "size_bytes": size,
                        }
                with tempfile.TemporaryDirectory(
                    prefix=".aptus-archive-contract-"
                ) as bundle_name:
                    bundle_root = Path(bundle_name)
                    for info in infos:
                        destination = bundle_root / Path(info.filename)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with (
                            source.open(info, "r") as member,
                            destination.open("xb") as extracted,
                        ):
                            while True:
                                block = member.read(1024 * 1024)
                                if not block:
                                    break
                                extracted.write(block)
                    contract_errors = validate_bundle_manifest(bundle_root)
                    if contract_errors:
                        raise ValueError(
                            "Archive does not satisfy the canonical Aptus bundle contract: "
                            + "; ".join(contract_errors[:3])
                        )
                    manifest = json.loads(
                        (bundle_root / "bundle-manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    entries = manifest.get("files")
                    if not isinstance(entries, list):
                        raise ValueError(
                            "Archive does not satisfy the canonical Aptus bundle contract."
                        )
                    manifested_names = {
                        item.get("path") for item in entries if isinstance(item, dict)
                    }
                    if None in manifested_names or set(names) != manifested_names | {
                        "bundle-manifest.json"
                    }:
                        raise ValueError(
                            "Archive member inventory differs from the canonical Aptus bundle manifest."
                        )
                    from .validation import validate_bundle

                    static_report = validate_bundle(
                        bundle_root, level="static", run=False
                    )
                    if static_report.state.value != "static-pass":
                        raise ValueError(
                            "Archive does not satisfy the canonical Aptus static bundle contract."
                        )
        if _archive_source_fingerprint(os.fstat(descriptor)) != fingerprint:
            raise ValueError("Bundle archive changed during validation.")
        validation_succeeded = True
    except (EOFError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError("Bundle archive is not a valid Aptus ZIP.") from error
    finally:
        os.close(descriptor)
        if not validation_succeeded:
            expected_path.unlink(missing_ok=True)
    try:
        if _archive_source_fingerprint(
            archive_path.lstat()
        ) != fingerprint or not _files_equal(expected_path, archive_path):
            raise ValueError("Bundle archive bytes are not canonical Aptus output.")
        return inventory
    finally:
        expected_path.unlink(missing_ok=True)


def validate_bundle_archive_bytes(payload: bytes) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper over the bounded streaming file validator."""

    if type(payload) is not bytes or not payload:
        raise ValueError("Bundle archive payload must be nonempty bytes.")
    descriptor, name = tempfile.mkstemp(prefix=".aptus-archive-bytes-")
    path = Path(name)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Bundle archive temporary write made no progress.")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        return validate_bundle_archive_file(path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def create_bundle_archive(bundle_dir: Path, archive_path: Path | None = None) -> Path:
    """Create a byte-deterministic ZIP from a compiled bundle."""

    bundle_dir = bundle_dir.resolve(strict=True)
    archive_path = (archive_path or bundle_dir.with_suffix(".zip")).resolve()
    if archive_path == bundle_dir or bundle_dir in archive_path.parents:
        raise ValueError(
            "Bundle archives must be written outside the bundle directory."
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(f"Archive output already exists: {archive_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.tmp-", dir=archive_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_bundle_archive(bundle_dir, temporary)
        try:
            os.link(temporary, archive_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Archive output already exists: {archive_path}"
            ) from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return archive_path


def bundle_files(bundle_dir: Path) -> list[str]:
    return sorted(
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file()
    )
