# Configuration Defaults

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Operators, API clients, automation authors, and compiler maintainers |
| Authority | Normative v0.2 reference for declared defaults, planner priors, and emitted runtime settings |
| Last reviewed | 2026-07-22 |
| Next review | 2026-10-22, or sooner when CLI, API, planning, catalog, or generation code changes |

Aptus has three kinds of configuration value:

1. **Input defaults** fill omitted CLI or API fields.
2. **Planner priors** derive candidate settings from supplied facts.
3. **Compiler-fixed settings** are written into the selected bundle.

These values are reproducible product policy. They are not universal
fine-tuning recommendations or claims that a setting is optimal. A generated
plan records the selected values and assumptions. Runtime gates must still
verify the exact model, data, hardware, and pinned dependency stack.

## Precedence and persistence

| Source | When it applies | Where the result is recorded |
| --- | --- | --- |
| Explicit user value | When the CLI option or API field is supplied | Profile, plan, or job request |
| Interface default | When an optional field is omitted | Resulting profile, plan, validation request, or job |
| Planner prior | While all candidate configurations are enumerated | Candidate fields and assumptions in `plan.json` |
| Compiler-fixed setting | When the recommended candidate is compiled | `config/trainer.json`, `config/accelerate.yaml`, or `requirements.txt` |
| Pilot-only override | When `train.py --pilot` receives an allowed override | Pilot evidence; it does not change the full-run contract |

Recompiling a persisted plan does not rerun planning. It preserves the plan's
recommended candidate and emits configuration from that candidate.

## CLI input defaults

### Dataset profiling

| Option | Default | Meaning |
| --- | --- | --- |
| `--sample-limit` | `512` | Maximum deterministic sample used for length statistics |
| `--sequence-length` | `null` | No target-length pressure calculation unless supplied |
| `--output` | `null` | Print JSON to standard output |

The sample limit does not bypass whole-file schema validation and does not
limit rows copied into a compiled bundle.

### Planning, build, and compatibility flow

| Option | Default | Meaning |
| --- | --- | --- |
| `--sample-limit` | `512` | Profiling-statistics sample bound |
| `--backend` | `cuda` | Declared device backend |
| `--free-vram-gib` | `null` | Planner uses total device memory as the available-memory fact |
| `--bf16` | False | BF16 capability is not assumed |
| `--four-bit` | False | Four-bit kernel capability is not assumed |
| `--eight-bit` | False | Eight-bit kernel capability is not assumed |
| `--host-ram-free-gib` | `null` | Planner uses total host RAM as the available-memory fact |
| `--reserve-gib` | `2.0` | GiB excluded from the fit budget on each device |
| `--disk-free-gib` | `null` | Analytic disk rejection is skipped; runtime disk checks still apply |
| `--objective` | `memory` | Rank viable candidates by the memory policy |
| `--effective-batch-size` | `16` | Required exact global batch |
| `--epochs` | `3` | Maximum full-run epochs |
| `--prefer-method` | `null` | No secondary tie-break preference |
| `--evaluation-fraction` | `0.1` | Requested full-run evaluation fraction |
| `--checkpoint-steps` | `100` | Optimizer steps between checkpoints |
| `--packing` | False | Packing remains disabled and fails closed if requested |
| `--plan-output` | `null` | Do not write a separate host plan beside a build |

Model identity, model shape, dataset path, device count and memory, total host
RAM, and sequence length have no CLI defaults. Training permission must be
confirmed explicitly.

### Validation, jobs, inspection, and serving

| Command or option | Default | Meaning |
| --- | --- | --- |
| `validate --level` | `static` | Run cumulative contract and static checks |
| `validate --run` | False | Do not execute checks above static directly |
| `validate --state-dir` | `.aptus-state` | Managed state root when `--run` delegates to a job |
| `run --action` | `preflight` | Submit cumulative dependency, model-data, and measured-preflight work |
| `run --confirm-full-train` | False | Full training is not authorized |
| `run --state-dir` | `.aptus-state` | Job state root |
| `jobs --state-dir` | `.aptus-state` | Job state root |
| `jobs --id` | `null` | List jobs instead of inspecting one |
| `serve --host` | `127.0.0.1` | Bind to loopback |
| `serve --port` | `8787` | HTTP port |
| `serve --state-dir` | `.aptus-state` | API plans, current bundle reference, and job state |
| `serve --web-dist` | Auto-detected | Use the packaged workbench if present |
| `serve --allow-non-loopback` | False | Reject non-loopback binding |
| `inspect model --timeout` | `10.0` seconds | Provider inspection timeout |

The shared model-inspection function requires a positive timeout no greater
than 30 seconds for both CLI and API calls. The API request model enforces the
same range before calling it. The CLI exposes the 10-second default and relies
on the shared function for the upper bound.

## API request defaults

API request models reject unknown fields.

### Hardware request

| Field | Default |
| --- | --- |
| `discovery` | `manual` |
| `backend` | `cuda` |
| `supports_bf16` | False |
| `supports_4bit` | False |
| `supports_8bit` | False |
| `free_vram_gib` | `null` |
| `host_ram_free_gib` | `null` |
| `reserve_gib` | `2.0` |
| `disk_free_gib` | `null` |

`gpu_count`, `vram_gib`, and `host_ram_gib` are required even when
`discovery` is `local-scan`, because the current request schema validates them
before the server replaces manual hardware facts with a local probe. During a
local scan, only `reserve_gib` is carried into the probe. The other manual
hardware values are ignored.

### Target, profile, and plan requests

| Field | Default | Applies to |
| --- | --- | --- |
| `effective_batch_size` | `16` | Target |
| `max_epochs` | `3` | Target |
| `method_preference` | `null` | Target |
| `task` | `sft` | Target |
| `evaluation_fraction` | `0.1` | Target |
| `packing` | False | Target |
| `checkpoint_steps` | `100` | Target |
| `sample_limit` | `512` | Profile and plan |
| `sequence_length` | `null` | Profile only |

The plan target still requires `objective` and `sequence_length`. The plan
request requires model, dataset, hardware, and target facts.

### Validation, job, and inspection requests

| Field | Default |
| --- | --- |
| Validation `level` | `static` |
| Validation `run` | False |
| Job `action` | `preflight` |
| Job `confirm_full_train` | False |
| Model inspection `timeout_seconds` | `10.0`, allowed range greater than 0 through 30 |

## Planner-derived candidate settings

The planner evaluates every selectable method against `single`, `ddp`, and
`fsdp`, for 12 candidate records before support and feasibility filtering.

### Adapter rank and alpha

| Condition | Rank | Alpha |
| --- | ---: | ---: |
| Full fine-tuning | `0` | `0` |
| Any adapter method with objective `memory` | `8` | `16` |
| Adapter method, non-memory objective, at least 1,000,000 estimated tokens | `32` | `64` |
| Adapter method, non-memory objective, fewer than 1,000,000 estimated tokens | `16` | `32` |

For adapter methods, alpha always follows `2 * rank`. These are method-class
priors. They are not tuned against the supplied corpus.

### Precision, quantization, and learning rate

| Method | Precision rule | Quantization | Learning rate |
| --- | --- | --- | ---: |
| `full` | BF16 only; the FP16 path is unsupported | `null` | `0.00002` |
| `lora` | BF16 if every bound device declares support, otherwise FP16 | `null` | `0.0002` |
| `int8-lora` | BF16 if every bound device declares support, otherwise FP16 | `int8-bitsandbytes` | `0.0002` |
| `qlora` | BF16 if every bound device declares support, otherwise FP16 | `nf4-double-quant` | `0.0002` |

Eight-bit and four-bit candidates also require the corresponding capability on
every participating device. Capability flags never manufacture support.

### Batch derivation

The planner sets world size to 1 for `single`. For `ddp` and `fsdp`, world size
is the number of bound devices. The requested effective batch must be divisible
by world size.

Eligible micro-batches are exact divisors searched from
`min(32, effective_batch_size)` down to 1. Aptus selects the first micro-batch
whose memory upper envelope fits. If none does, it retains the first point-fit
choice when available. Gradient accumulation is:

```text
ceil(effective_batch_size / (micro_batch_size * world_size))
```

The candidate is infeasible unless the resulting global batch equals the
requested batch exactly.

### Objective ranking

All policies rank a fully feasible candidate before a conditional candidate.
The method preference is a secondary key and cannot reverse feasibility.

| Objective | Remaining keys, in order |
| --- | --- |
| `memory` | Smaller upper memory estimate, preferred method, fewer accumulation steps |
| `speed` | Fewer accumulation steps, preferred method, higher method fidelity |
| `quality` | Higher method fidelity, preferred method, smaller upper memory estimate |

Method fidelity order is `full`, `lora`, `int8-lora`, then `qlora`. The policy
does not predict measured quality or throughput.

## Compiler-fixed trainer settings

The compiler writes `config/trainer.json` with schema
`aptus.trainer-config.v2`.

| Setting | Emitted value or source |
| --- | --- |
| `task` | Target value, currently `sft` |
| `sequence_length` | Target value |
| `packing` | Target value, currently required to be False |
| `evaluation_fraction` | Target value |
| `max_epochs` | Target value |
| Train and evaluation micro-batch | Recommended candidate micro-batch |
| `gradient_accumulation_steps` | Recommended candidate value |
| `effective_global_batch_size` | Recommended candidate value |
| `world_size` and `device_indices` | Recommended candidate placement |
| `learning_rate` | Recommended candidate prior |
| `optimizer` | `adamw_torch` |
| `lr_scheduler_type` | `linear` |
| `weight_decay` | `0.0` |
| `warmup_steps` | `0` |
| `max_grad_norm` | `1.0` |
| `gradient_checkpointing` | True |
| `gradient_checkpointing_use_reentrant` | True only for single-device QLoRA; False otherwise |
| `checkpoint_steps` | Target value |
| `logging_steps` | `min(10, checkpoint_steps)` |
| `save_total_limit` | `3` |
| `report_to` | Empty list |
| `remove_unused_columns` | False |
| `seed` | `17` |
| `pilot_row_limit` | `max(32, 2 * effective_batch_size)` |
| `pilot_dataset_path` | `data/pilot-sample.jsonl` |
| `training_dataset_path` | `data/training.jsonl` |
| `truncation_policy` | `completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision` |

The full run uses the compiled target and seed. `--max-steps` and `--seed` are
pilot-only overrides. Full-training resume remains fail-closed.

## Accelerate configuration

`config/accelerate.yaml` always sets:

| Setting | Value |
| --- | --- |
| `compute_environment` | `LOCAL_MACHINE` |
| `distributed_type` | `NO`, `MULTI_GPU`, or `FSDP` from the selected distribution |
| `mixed_precision` | Recommended precision |
| `num_processes` | Recommended world size |
| `num_machines` | `1` |
| `machine_rank` | `0` |
| `same_network` | True |
| `use_cpu` | False |

FSDP additionally emits version 1, transformer-based wrapping,
`BACKWARD_PRE`, no parameter offload, `FULL_SHARD`, `SHARDED_STATE_DICT`,
synchronized module state, original parameters, and CPU-RAM-efficient loading.

## Pinned runtime versions

The compiler selects only the packages required by the method and writes exact
direct pins to `requirements.txt`. This file is not a transitive lock.

| Package | Version | Included for |
| --- | --- | --- |
| `torch` | `2.13.0` | Every method |
| `transformers` | `5.14.1` | Every method |
| `accelerate` | `1.14.0` | Every method |
| `safetensors` | `0.8.0` | Every method |
| `peft` | `0.19.1` | Adapter methods |
| `bitsandbytes` | `0.49.2` | `int8-lora` and `qlora` |

## Development web defaults

The Vite development server binds to `127.0.0.1:4173`. Its `/api` proxy uses
`APTUS_API_ORIGIN` when set and otherwise targets `http://127.0.0.1:8787`.
These are development settings, not generated training-bundle settings.

## Related documentation

- [CLI reference](cli.md)
- [API reference](api.md)
- [Plan schema](plan-schema.md)
- [Bundle manifest](bundle-manifest.md)
- [Method registry](method-registry.md)
- [Validation states](validation-states.md)
- [Ranking and uncertainty](../methodology/ranking-uncertainty.md)
- [Precision and quantization](../methodology/precision-quantization.md)
