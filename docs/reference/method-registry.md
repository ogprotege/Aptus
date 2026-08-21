# Fine-Tuning Method Registry

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Users comparing methods, planner and compiler maintainers, and research reviewers |
| Authority | Normative v0.2 reference for method identity, lifecycle, selection, compilation, and release gates |
| Last reviewed | 2026-08-04 |
| Next review | 2026-10-22, or sooner when `src/aptus/methods/` changes |

The method registry is Aptus's product boundary between knowing about a
fine-tuning method and being able to execute it. A descriptor can document a
method without making it selectable. Only a gated executable descriptor with
a registered compiler, export contract, backend, and distribution can enter
the planner's candidate matrix.

The v0.2 registry contains 11 descriptors. Four are selectable. Seven remain
visible with explicit blockers and proof requirements.

## Lifecycle model

| Lifecycle | Selectable | Meaning |
| --- | ---: | --- |
| `gated-executable` | Yes | Aptus has a compiler and export contract, but runtime validation and pilot gates still apply |
| `experimental` | No | The method has a defined research identity, but an Aptus execution contract is incomplete |
| `research-only` | No | Aptus tracks the method for comparison and future work, without an executable product path |

`gated-executable` does not mean universally feasible. The planner can still
mark a candidate unsupported or infeasible because of model family, precision,
hardware capability, distribution, memory, host RAM, disk, task, or packing.

## Descriptor schema

Every API descriptor uses schema `aptus.method-descriptor.v1` and contains:

| Field | Meaning |
| --- | --- |
| `method_id` | Canonical lowercase product identifier |
| `display_name` | Human-readable method name |
| `summary` | Bounded description of the trainable object and update pattern |
| `lifecycle` | `gated-executable`, `experimental`, or `research-only` |
| `selectable` | Whether the method can enter planning |
| `parameter_scope` | What parameters or representations can change |
| `parameterization` | Registry label for the update construction |
| `base_storage` | How the frozen or trainable base is stored |
| `compiler_id` | Versioned compiler dispatch key, or `null` |
| `export_kind` | Verified artifact contract, or `null` |
| `supported_backends` | Backends admitted by the descriptor |
| `supported_distributions` | Distribution modes admitted by the descriptor |
| `runtime_bindings` | Per-runtime contracts that determine executability. Each binds training runtime, compute backend, compiler, estimator, export kind, and supported distributions |
| `evidence_ids` | Evidence registry records supporting the method description |
| `pilot_requirement` | Minimum implementation evidence required for release or execution |
| `blocker` | Required reason a non-selectable method cannot execute |
| `aliases` | Reserved descriptive aliases |
| `schema_version` | `aptus.method-descriptor.v1` |

Aliases are metadata in v0.2. They are not accepted by CLI choices, API target
validation, `Method(...)`, or registry lookup. Use the canonical `method_id` in
all plan requests and persisted artifacts.

## Selectable method matrix

| Method | Parameter scope | Base storage | Compiler | Export | Backend | Distribution |
| --- | --- | --- | --- | --- | --- | --- |
| `full` | All parameters | Unquantized | `transformers.full.v2` | `full-model-safetensors` | CUDA | `single`, `ddp` |
| `lora` | Frozen base plus adapter | Unquantized | `transformers.peft-lora.v2` | `peft-adapter-safetensors` | CUDA, MPS | `single`, `ddp`, `fsdp` |
| `int8-lora` | Frozen base plus adapter | Bitsandbytes INT8 | `transformers.peft-int8-lora.v2` | `peft-adapter-safetensors` | CUDA | `single`, `ddp` |
| `qlora` | Frozen base plus adapter | Runtime-native four-bit | `transformers.peft-qlora.v2` | `peft-adapter-safetensors` | CUDA, MPS | `single`, `ddp` |

The Compiler, Export, and Distribution columns describe the CUDA runtime
binding. `lora` and `qlora` each carry a second `mlx-lm` binding on the `mps`
backend, with compilers `mlx-lm.lora.v1` and `mlx-lm.qlora.v1`, estimator
`aptus-memory-mlx-v2`, export kind `mlx-lm-adapter`, and `single` placement only.

### Full fine-tuning

Full fine-tuning updates every model parameter and exports a complete
Safetensors model.

Additional v0.2 gates:

- every selected candidate must use CUDA;
- BF16 support is required because the generated full-parameter FP16 path does
  not retain a verified FP32 trainable master-weight contract;
- full-parameter FSDP remains fail-closed because the pinned runtime's FP32
  upcast and full-state export transient is not calibrated; and
- exact model-data validation, measured preflight, and a bounded real-model
  pilot are mandatory.

Registered alias: `full-parameter`.

### LoRA

LoRA freezes the base model and trains low-rank adapters on model-family target
modules. The current target catalog covers `llama`, `mistral`, `gemma`, and
`qwen` projection names, each using the dense causal-LM target tuple. It also
registers a fifth family, `qwen3_moe`, whose targets are attention-only
(`q_proj`, `k_proj`, `v_proj`, `o_proj`) so that adapters never attach to routed
expert or router-gate modules.

Additional v0.2 gates:

- real-model inspection must find the compiled target module suffixes;
- trainable parameter names and counts must match the expected adapter census;
- FSDP uses original parameters to support interspersed frozen and trainable
  parameters; and
- measured preflight and a bounded real-model pilot remain mandatory.

LoRA has no registered alias.

### Eight-bit LoRA

Eight-bit LoRA trains LoRA adapters over a frozen bitsandbytes INT8 base.

Additional v0.2 gates:

- every participating device must explicitly support the eight-bit path;
- FSDP is outside the verified compiler matrix;
- exact kernel capability and target inspection are required; and
- measured preflight and a bounded pilot are mandatory.

Registered alias: `8bit-lora`.

### QLoRA

QLoRA trains LoRA adapters through a frozen runtime-native quantized base: an
NF4 double-quantized four-bit base on `transformers-peft-cuda`, and a declared
MLX groupwise base (bits 1 through 16) on `mlx-lm`.

Additional v0.2 gates:

- every participating CUDA device must explicitly support the four-bit path.
  This gate does not apply to `mlx-lm`, where eligibility comes from declared
  MLX quantization metadata in the pinned model revision, checked at
  model-data validation;
- FSDP is outside the verified compiler matrix;
- single-device CUDA QLoRA uses reentrant gradient checkpointing;
- distributed CUDA QLoRA uses non-reentrant gradient checkpointing. Gradient
  checkpointing selection applies to `transformers-peft-cuda` only; and
- exact kernel capability, target inspection, measured preflight, and a
  bounded pilot are mandatory.

Registered alias: `4bit-lora`.

## Non-selectable method matrix

| Method | Lifecycle | Parameterization | Blocker summary |
| --- | --- | --- | --- |
| `dora` | `experimental` | `dora` | No Aptus compiler, calibrated estimator, or verified export and reload contract |
| `bitfit` | `experimental` | `bias-only` | Eligible biases can be absent, and Aptus has no bias-delta export contract |
| `adalora` | `experimental` | `adaptive-budget-lora` | Changing trainable budget, schedule, importance state, and restart are not modeled |
| `sharelora` | `experimental` | `shared-factor-lora` | No shared-module serializer or distributed ownership contract |
| `loreft` | `research-only` | `low-rank-reft` | Current trainer, collator, checkpoint, and export contracts do not represent interventions |
| `aflora` | `research-only` | `adaptive-freezing-lora` | No compatible compiler or checkpoint contract has passed dynamic-freezing checks |
| `bilora` | `research-only` | `pseudo-svd-lora` | Generic Trainer cannot express the required inner and outer optimization contract |

### DoRA

DoRA separates weight magnitude from direction and applies a low-rank update
to the directional component. Release requires a pinned PEFT `use_dora` path
that passes target-type, trainable-state, save, reload, and bounded-pilot
checks.

### BitFit

BitFit freezes the model except for explicitly enumerated existing bias
tensors. Many decoder architectures, including default Llama configurations,
can expose no eligible attention or MLP biases. Release requires a non-empty
bias set on the exact pinned architecture and agreement among selected-name
digest, optimizer membership, export, reload, and pilot evidence.

Registered alias: `bias-only`.

### AdaLoRA

AdaLoRA allocates a changing rank budget across pseudo-SVD parameter groups.
Release requires a pinned implementation that binds initial rank, final budget,
schedule, importance state, optimizer membership, checkpoint continuation,
export, and reload.

### ShareLoRA

ShareLoRA shares one or both low-rank factors across shape-compatible layers.
Release requires proof of shape grouping, unique versus logical parameter
accounting, serialization, reload, and distributed synchronization.

### LoReFT

LoReFT learns low-rank interventions on hidden representations at explicit
layers and token positions. It is not a PEFT weight-adapter alias. Release
requires a pinned intervention runtime that binds component paths, layers,
positions, gradients, checkpoints, export, and reload.

Registered alias: `low-rank-reft`.

### AFLoRA

AFLoRA dynamically scores and freezes low-rank parameter groups during
training. Release requires a run that crosses a freeze event and proves
deterministic scores, optimizer membership, restart equivalence, export, and
reload.

### BiLoRA

BiLoRA uses bilevel optimization over disjoint data partitions and a pseudo-SVD
low-rank update. Release requires a dedicated two-optimizer loop that binds the
two partitions, both optimizer states, restart semantics, export, and reload.

## Registry invariants

Registry validation runs at import time. It enforces:

- selectable IDs exactly equal the executable `Method` enum:
  `full`, `lora`, `int8-lora`, and `qlora`;
- every selectable method is `gated-executable`;
- every selectable method declares a compiler, export kind, backend, and
  distribution;
- selectable compiler IDs are unique;
- supported backend and distribution values belong to the domain enums;
- all canonical IDs and aliases are unique after lowercase normalization;
- every method ID is non-empty, lowercase, and trimmed; and
- every non-selectable descriptor has an explicit blocker.

The invariant checks do not prove that kernels are available on a host or that
a pilot will pass. Those are later gates.

## Planner behavior

The planner iterates only selectable descriptors. It enumerates each one over
`single`, `ddp`, and `fsdp`, then records unsupported combinations rather than
hiding them. This produces a stable 12-record comparison matrix.

The descriptor support matrix is only the first filter. Candidate evaluation
also applies:

- exact MoE model identity, architecture, and expert-topology policy, plus host
  model-policy decision and registered-path matching for the reviewed dense
  Qwen2 footprint;
- canonical quantization-layout equality for the reviewed Qwen3 MoE slice;
- the same canonical layout equality for the reviewed dense Qwen2 path;
- CUDA backend and per-device capability rules;
- sequence length, task, and packing rules;
- world-size and exact global-batch arithmetic;
- precision and quantization constraints;
- point and upper memory envelopes;
- host model-loading budget; and
- disk staging, pilot, checkpoint, and export budget.

A method preference is a secondary ranking key. It never causes an unsupported
or infeasible method to become recommended.

## Compiler dispatch and exports

The compiler resolves the recommended method descriptor and writes both its
`compiler_id` and `export_kind` into `config/trainer.json`. Runtime validation
uses those values to select and verify the method-specific preparation and
artifact contract.

| Export kind | Required meaning |
| --- | --- |
| `full-model-safetensors` | Complete model weights and reloadable model metadata |
| `peft-adapter-safetensors` | Adapter weights and reloadable PEFT adapter metadata over the pinned base revision |
| `mlx-lm-adapter` | MLX-LM adapter weights and reloadable adapter metadata over the pinned MLX base revision. Written for every `mlx-lm` candidate |

An export directory existing is not enough. Pilot and full-run checks parse the
expected configuration or adapter metadata and verify the artifact type,
structural safetensors file tree, tensor keys, manifest, sizes, digests, and
binding to the current plan and candidate. They do not reload the trained model
for bounded inference or establish semantic inference parity.

## API and UI visibility

`GET /api/v1/bootstrap` publishes two views:

| Field | Contents |
| --- | --- |
| `capabilities.methods` | The four canonical selectable IDs |
| `capabilities.method_catalog` | All 11 descriptors, including blockers and pilot requirements |

The workbench can explain unavailable methods without presenting them as
executable. Clients must use `selectable` and lifecycle fields. They must not
infer executability from the presence of an evidence citation or a display
name.

## Adding or promoting a method

A registry edit alone is insufficient. Before a method becomes selectable, its
implementation must supply and test:

1. a versioned compiler dispatch path;
2. deterministic planner and memory-estimator behavior;
3. supported backend and distribution boundaries;
4. exact trainable-object and optimizer-membership checks;
5. dependency and kernel capability checks;
6. checkpoint, restart, export, and reload contracts;
7. runtime-specific bounded measured-preflight evidence;
8. exact model-and-data pilot evidence;
9. manifest and validation bindings; and
10. documentation, fixtures, and regression tests for failure as well as
    success.

Promotion must preserve fail-closed behavior. Unsupported combinations should
remain visible with precise reasons.

## Related documentation

- [Capability matrix](capability-matrix.md)
- [Configuration defaults](configuration-defaults.md)
- [Evidence records](evidence-records.md)
- [Plan schema](plan-schema.md)
- [Validation states](validation-states.md)
- [Choose a method](../guides/choose-a-method.md)
- [Method taxonomy](../methodology/method-taxonomy.md)
- [Candidate enumeration](../methodology/candidate-enumeration.md)
- [Reviewed corpus contract](reviewed-corpus-contract.md)
