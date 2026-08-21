# Configuration Defaults

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Operators, API clients, automation authors, and compiler maintainers |
| Authority | Normative v0.2 reference for declared defaults, planner priors, and emitted runtime settings |
| Last reviewed | 2026-08-16 |
| Next review | 2026-10-27, or sooner when CLI, API, planning, catalog, or generation code changes |

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
| CUDA pilot-only override | When CUDA `train.py --pilot` receives an allowed override | Pilot evidence; it does not change the full-run contract |

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
| `--training-runtime` | `null` | Infer the runtime from backend and method |
| `--free-vram-gib` | `null` | Omitted CUDA free memory is infeasible; Aptus will not treat total as free |
| `--bf16` | False | BF16 capability is not assumed |
| `--four-bit` | False | Four-bit kernel capability is not assumed |
| `--eight-bit` | False | Eight-bit kernel capability is not assumed |
| `--host-ram-free-gib` | `null` | Omitted free host RAM is infeasible; Apple Silicon uses this as unified headroom |
| `--reserve-gib` | `2.0` | GiB excluded from the fit budget on each device |
| `--disk-free-gib` | `null` | Omitted free disk is infeasible; Aptus will not assume staging space |
| `--objective` | `memory` | Rank viable candidates by the memory policy |
| `--effective-batch-size` | `16` | Required exact global batch |
| `--epochs` | `3` | Maximum full-run epochs |
| `--prefer-method` | `null` | No secondary tie-break preference |
| `--evaluation-fraction` | `0.1` | Requested full-run evaluation fraction |
| `--checkpoint-steps` | `100` | CUDA optimizer steps between checkpoints; MLX retains the target fact but writes non-resumable weight snapshots |
| `--packing` | False | Packing remains disabled and fails closed if requested |
| `--plan-output` | `null` | Do not write a separate host plan beside a build |

Model identity, model shape, dataset path, device count and memory, total host
RAM, and sequence length have no CLI defaults. Training permission must be
confirmed explicitly.

The `2.0` GiB reserve is the CLI's syntactic default, not the effective Apple
unified-memory default. When the resolved backend is `mps`, the CLI raises the
effective reserve to `max(--reserve-gib, 8.0)` before it constructs hardware
facts. Supplying `--reserve-gib 0` or `2` therefore cannot reduce the MPS
reserve below 8 GiB. CUDA retains the submitted non-negative value.

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
| `serve` session token | Fresh random value per launch | Authenticate protected API requests |
| `serve` workbench URL | Printed origin on standard error | Does not include the session token |
| `serve` API credential | Printed bearer token on standard error | Use `Authorization: Bearer TOKEN` for programmatic calls |
| `serve` optional query handoff | Operator-appended `aptus_session_token` | Exchange the token for an HttpOnly, SameSite Strict cookie and redirect with `303` |
| `serve` access logging | False | Keep request lines out of Uvicorn access logs |
| `inspect model --timeout` | `10.0` seconds | Provider inspection timeout |

Only health and static workbench assets are public under `aptus serve`. Every
other API route and the generated API documentation require the session cookie
or bearer token. Non-loopback mode retains this authentication, but the built-in
server remains plain HTTP. The operator must provide approved TLS and network
controls.

The shared model-inspection function requires a positive timeout no greater
than 30 seconds for both CLI and API calls. The API request model enforces the
same range before calling it. The CLI exposes the 10-second default and relies
on the shared function for the upper bound.

## API request defaults

API request models reject unknown fields.

### Bootstrap defaults

`GET /api/v1/bootstrap` chooses host-aware workbench defaults. On Darwin it
returns `backend: mps`, `training_runtime: mlx-lm`, `reserve_gib: 8.0`, and
`supported_execution_backend: mps`. On other platforms it returns
`backend: cuda`, `training_runtime: transformers-peft-cuda`,
`reserve_gib: 2.0`, and `supported_execution_backend: cuda`. These values seed
the interface. They do not bypass runtime inventory, planning, or validation.

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

The `2.0` GiB request default is also syntactic. Before either manual hardware
construction or a local probe, the API raises the effective reserve to at least
`8.0` GiB when the request uses backend `mps`, selects training runtime
`mlx-lm` or `pytorch-mps`, or requests local discovery from a Darwin server.
Consequently, a caller cannot use a lower submitted reserve to bypass the
unified-memory floor. CUDA requests outside those conditions retain the
submitted non-negative value.

### Target, profile, and plan requests

| Field | Default | Applies to |
| --- | --- | --- |
| `effective_batch_size` | `16` | Target |
| `max_epochs` | `3` | Target |
| `method_preference` | `null` | Target |
| `training_runtime` | `null` | Target; infer from backend and method |
| `task` | `sft` | Target |
| `evaluation_fraction` | `0.1` | Target |
| `packing` | False | Target |
| `checkpoint_steps` | `100` | Target |
| `sample_limit` | `512` | Profile and plan |
| `sequence_length` | `null` | Profile only |

The plan target still requires `objective` and `sequence_length`. The plan
request requires model, dataset, hardware, and target facts.

### Validation, job, inspection, and inference requests

| Field | Default |
| --- | --- |
| Validation `level` | `static` |
| Validation `run` | False |
| Job `action` | `preflight` |
| Job `confirm_full_train` | False |
| Model inspection `timeout_seconds` | `10.0`, allowed range greater than 0 through 30 |
| Inference model-list `timeout_seconds` | `5.0`, allowed range greater than 0 through 30 |
| Inference generation `timeout_seconds` | `5.0`, allowed range greater than 0 through 30 |
| Inference generation `max_tokens` | `256`, allowed range 1 through 32,768 |
| Inference generation `temperature` | `0.0`, allowed range 0 through 2 |

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

| Runtime and method | Precision rule | Quantization | Learning rate |
| --- | --- | --- | ---: |
| CUDA `full` | BF16 only; the FP16 path is unsupported | `null` | `0.00002` |
| CUDA `lora` | BF16 if every bound device declares support, otherwise FP16 | `null` | `0.0002` |
| CUDA `int8-lora` | BF16 if every bound device declares support, otherwise FP16 | `int8-bitsandbytes` | `0.0002` |
| CUDA `qlora` | BF16 if every bound device declares support, otherwise FP16 | `nf4-double-quant` | `0.0002` |
| MLX-LM `lora` | BF16 if declared, otherwise FP16; discovered MPS currently records FP16 | `null` | `0.0002` |
| MLX-LM `qlora` | BF16 if declared, otherwise FP16; discovered MPS currently records FP16 | `mlx-{bits}bit-groupwise` from the pin | `0.0002` |

CUDA eight-bit and four-bit candidates require the corresponding capability on
every participating device. MLX QLoRA ignores the CUDA-style device flag and
requires declared MLX quantization bits (1 through 16) in the pinned model
during model-data validation. A 4-bit pin still emits `mlx-4bit-groupwise`.
Capability flags never manufacture support.

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
| `optimizer` | `adamw_torch` for CUDA; `adamw` for MLX-LM |
| `lr_scheduler_type` | `linear` for CUDA; `null` for MLX-LM |
| `weight_decay` | `0.0` |
| `warmup_steps` | `0` |
| `max_grad_norm` | `1.0` for CUDA; `null` for MLX-LM |
| `gradient_checkpointing` | True |
| `gradient_checkpointing_use_reentrant` | True only for single-device CUDA QLoRA; False otherwise |
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

Compare and CLI name rank, alpha, learning rate, completions-mask, epochs, and
dataset size as Aptus v0.2 method-class, compiler, or instruction-SFT priors,
not optima. Instruction-SFT defaults: supervision prior of 100 rows; epoch-cap
prior of 3 (Aptus will not rewrite the requested epoch count); small-corpus
band 100–299 rows with `max_epochs` ≥ 10 matches the parrot/sycophancy
over-training prior. Those presentation knobs do not change the emitted values
above, including `weight_decay` `0.0` and `warmup_steps` `0`.

The CUDA full run uses the compiled target and seed. `--max-steps` and `--seed`
are CUDA pilot-only overrides. Full-training resume remains fail-closed. An
MLX-LM bundle also writes `config/mlx-lm.yaml` with `fine_tune_type: lora`,
AdamW, seed `17`, gradient checkpointing, the candidate batch and accumulation
values, two default smoke iterations, an eight-iteration measured-preflight
ceiling, and periodic adapter weight saves. The MLX pilot overrides duration to
exactly two optimizer updates. Full training derives an uninterrupted duration
from compiled train rows, batch, accumulation, and epochs. These periodic files
are weight snapshots, not resumable checkpoints.

## Accelerate configuration

`config/accelerate.yaml` is emitted for bundle compatibility. The CUDA compiler
uses it with these settings:

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
| `torch` | `2.13.0` | Every CUDA method |
| `transformers` | `5.14.1` | Every CUDA method |
| `accelerate` | `1.14.0` | Every CUDA method |
| `safetensors` | `0.8.0` | Every CUDA method |
| `peft` | `0.19.1` | CUDA adapter methods |
| `bitsandbytes` | `0.49.2` | CUDA `int8-lora` and `qlora` |
| `mlx` | `0.31.2` | MLX-LM LoRA and QLoRA bundles |
| `mlx-lm` | `0.31.3` | MLX-LM LoRA and QLoRA bundles |

The Transformers, PEFT, Accelerate, Torch, safetensors, and bitsandbytes rows
apply to CUDA bundles as selected by method. MLX bundles contain only the two
MLX pins above. Those pins do not establish training resume. Aptus proves its
MLX pilot through uninterrupted optimizer work and proves adapter reload through
a separate bounded-generation process. Every resume argument remains rejected.

## Local runtime and inference defaults

The desktop runtime configuration has no guessed training interpreter. A user
selects an exact compatible Python, and Aptus persists that absolute command path in the
private state directory. LM Studio defaults to `http://127.0.0.1:1234` and oMLX
defaults to `http://127.0.0.1:8000` when their integrations are requested.
Origins must remain explicit loopback hosts with ports. Both services are
inference-only and never satisfy a training-runtime dependency.

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
