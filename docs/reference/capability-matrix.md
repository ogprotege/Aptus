# Capability Matrix

| Metadata | Value |
| --- | --- |
| Status | Active, unreleased engineering preview |
| Audience | Operators, product owners, method authors, and release reviewers |
| Authority | Normative v0.2 support boundary |
| Last reviewed | 2026-07-27 |
| Next review | 2026-10-27, or sooner when the method registry, planner, or compiler changes |

This matrix distinguishes a planner path from target-host proof. A planner row
marked supported can become viable when all facts and analytic gates pass. It
still requires runtime evidence for the exact bundle and host. CUDA training
requires static, dependency, model-data, measured-preflight, and pilot evidence.
MLX-LM uses the same state ladder with a runtime-specific uninterrupted pilot.
A current `pilot-pass` can authorize explicit full-duration adapter training.

Two clean Apple Silicon MLX-LM workflows reached `measured-run-pass` in the
[2026-07-27 acceptance record](../operations/evidence/2026-07-27-mlx-lm-acceptance/README.md).
No real CUDA target-host pilot has been recorded. Aptus v0.2 remains unreleased.

## CUDA method and placement matrix

| Method | Single | DDP | FSDP | Export contract |
| --- | --- | --- | --- | --- |
| Full | Planner-supported with BF16 | Planner-supported with BF16 and at least 2 GPUs | Unsupported | Full-model safetensors |
| LoRA | Planner-supported | Planner-supported with at least 2 GPUs | Conditional with at least 2 GPUs | PEFT adapter safetensors |
| int8-LoRA | Planner-supported with eight-bit capability | Planner-supported with shared eight-bit capability and at least 2 GPUs | Unsupported | PEFT adapter safetensors |
| QLoRA | Planner-supported with four-bit capability | Planner-supported with shared four-bit capability and at least 2 GPUs | Unsupported | PEFT adapter safetensors |

For each selected training runtime, all 12 method and placement pairs remain in
the candidate matrix. Unsupported and infeasible rows are evidence, not hidden
branches.

## MLX-LM method and placement matrix

| Method | Single | DDP | FSDP | Export contract |
| --- | --- | --- | --- | --- |
| Full | No compiler | Unsupported | Unsupported | None |
| LoRA | Conditional through uninterrupted pilot and full-duration adapter training | Unsupported | Unsupported | MLX-LM adapter |
| int8-LoRA | No compiler | Unsupported | Unsupported | None |
| QLoRA | Conditional through uninterrupted pilot and full-duration adapter training, with explicit four-bit capability facts and MLX model metadata | Unsupported | Unsupported | MLX-LM adapter |

MLX-LM uses the `mps` compute backend and `aptus-memory-mlx-v1` estimator. Its
LoRA and QLoRA candidates always remain conditional and pilot-required. Its
pilot is one uninterrupted run from the pinned base with at least two optimizer
updates, finite losses, exact target binding, positive memory and adapter-delta
evidence, live headroom, immutable artifacts, and fresh-process adapter reload
with one to four generated tokens. The reload is inference proof, not training
resume. A pass can admit an uninterrupted full-duration adapter run.

### CUDA method-specific hard boundaries

- Full training in FP16 is unsupported because the generated path does not
  retain a verified FP32 trainable master-weight contract.
- Full FSDP is unsupported because the pinned transient and export path is not
  calibrated safely.
- int8-LoRA FSDP and QLoRA FSDP are outside the verified compiler matrix.
- LoRA FSDP uses `use_orig_params=true` and remains conditional even when the
  analytic envelope fits.
- CUDA adapter methods require every target module in the family catalog to
  exist on the loaded revision.

## Precision and quantization

| Path | Planner rule | Runtime proof still required |
| --- | --- | --- |
| BF16 | Selected only when every participating device declares BF16 | Actual device, stack, method, and pilot behavior |
| FP16 full | Always unsupported | No launch |
| FP16 adapters | Selected when participating devices do not all declare BF16 | AMP behavior and exact pilot |
| Unquantized base | Full and LoRA | Model load and measured peak |
| INT8 bitsandbytes base | int8-LoRA only | Exact bitsandbytes load and kernel path |
| NF4 double-quantized base | QLoRA only | Exact bitsandbytes load and kernel path |
| MLX unquantized base | MLX-LM LoRA | Exact MLX model load, measured preflight, uninterrupted pilot, and adapter reload |
| MLX four-bit groupwise base | MLX-LM QLoRA only | Explicit quantization metadata in the pinned MLX model, measured preflight, uninterrupted pilot, and adapter reload |
| FP32 compute | Not enumerated | Future contract |
| FP8 | Not enumerated | Future contract |

CUDA probe fallback derives four-bit eligibility at CUDA compute capability
6.0 or newer and eight-bit eligibility at 7.5 or newer. Manual planning still
requires explicit capability flags. Runtime model-data, synthetic, and pilot
checks remain authoritative for the pinned software stack.

MLX-LM QLoRA does not use bitsandbytes or NF4 assumptions. Aptus does not
quantize an unbound model during training. The pinned model revision must
already declare its MLX four-bit quantization metadata.

## Backend matrix

| Backend | Accepted fact value | Local discovery | Planner execution rows | Compiler and runtime |
| --- | ---: | ---: | ---: | ---: |
| CUDA | Yes | Yes | Yes | Yes, subject to gates |
| ROCm | Yes | No supported probe result | Explicitly unsupported | No |
| MPS | Yes | Apple shared-memory and Metal inventory | MLX-LM LoRA and QLoRA single-device rows | MLX-LM uninterrupted adapter training; PyTorch MPS has no compiler |
| CPU | Yes | No supported accelerator result | Explicitly unsupported | No |
| MLX | Not a backend enum | Separate runtime probe | Runtime selected as `mlx-lm` over `mps` | Separate MLX-LM compiler |

On Apple Silicon, discovery reports one `mps` compatibility device backed by the
Metal working-set advisory when available, otherwise the measured unified-memory
capacity. It does not infer BF16, four-bit, eight-bit, or free VRAM. Apple
platform probing reports current host memory headroom and an optional Metal GPU
core count separately. MLX planning caps usable unified memory by the measured
free host RAM when available. The CUDA compiler never silently routes through
MPS or MLX.

### Training-runtime matrix

| Runtime | Discovery and configuration | Current compiler | Highest reachable evidence |
| --- | --- | --- | --- |
| `transformers-peft-cuda` | Exact active CUDA Python environment | Full, LoRA, int8-LoRA, QLoRA | `measured-run-pass`, subject to every gate |
| `mlx-lm` | Exact external Python executable, including persisted Mac selection | Single-device LoRA and QLoRA | `measured-run-pass`, subject to uninterrupted pilot and full-run gates |
| `pytorch-mps` | Discoverable and configurable exact external Python | None | No compiled runtime evidence |

LM Studio and oMLX are not training runtimes. They are loopback inference-only
services for model listing and text generation.

## Distribution behavior

MLX-LM supports only `single`. The table below defines the CUDA placement
behavior.

| Distribution | World size | Device binding | Memory rule |
| --- | ---: | --- | --- |
| `single` | 1 | Compatible device with greatest usable memory, stable index as tie-break | Selected device free-or-total minus reserve |
| `ddp` | All planned devices | Every planned index in order | Least per-device usable memory; state replicated |
| `fsdp` | All planned devices | Every planned index in order | Simplified sharding prior; LoRA only and conditional |

The requested global batch must divide exactly by world size. Aptus tests
per-device micro-batches from the largest exact divisor at or below 32 and
selects the largest whose upper envelope fits. DDP never sums VRAM across
devices.

## Model support

### Adapter target-module families

| Family | Target modules |
| --- | --- |
| `llama` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `mistral` | Same seven dense projection names |
| `gemma` | Same seven dense projection names |
| `qwen` | Same seven dense projection names |

Full training does not need adapter target modules. Provider inspection performs
only exact alias normalization:

- `qwen2` and `qwen3` to `qwen`;
- `gemma2`, `gemma3`, and `gemma3_text` to `gemma`; and
- `gemma3` only for explicitly accepted text architectures.

MoE, multimodal, prefix-matched, or unknown architectures are not silently
mapped. CUDA model-data validation checks the loaded parameter count, hidden
size, optional intermediate size, layers, context length, and adapter targets.
MLX-LM model-data validation loads the exact revision, validates its QLoRA
quantization metadata when applicable, and tokenizes every bound row.

## Dataset support

| Capability | V0.2 behavior |
| --- | --- |
| File formats | `.jsonl`, `.json`, `.csv`, `.txt` |
| Row schemas | Text, content alias, prompt-completion, instruction-output, messages |
| Mixed schema file | Supported when every row independently validates |
| Empty supported row | Ignored by profiling and canonical compilation |
| Malformed structured row | Rejected with row context |
| Canonical compilation | Every valid row, deterministic sorted-key JSONL |
| Tokenization | Exact pinned tokenizer at model-data and training time |
| Loss mask | Full text for text/content; completion only for structured rows |
| Sequence packing | Unsupported |
| Split grouping | Optional `split_group` or `metadata.split_group` |
| Evaluation fraction | Deterministic, exact for ungrouped rows when possible; closest atomic grouped result otherwise |

The source, canonical dataset, and split assignments are digest-bound. Full
training checks the canonical file across three split passes and during lazy
consumption. Distributed ranks must agree on digest, assignments, and counts.

## Target support

| Target dimension | Supported values |
| --- | --- |
| Task | `sft` only |
| Objective | `quality`, `memory`, `speed` ranking policies |
| Effective batch | Positive exact global batch |
| Sequence length | Positive and within model context |
| Evaluation fraction | `[0, 1)` |
| Checkpoint interval | Positive steps |
| Packing | False only |
| Maximum wall time | No enforced value |
| Quality metric or threshold | Not implemented |

The `quality` objective is a deterministic method-fidelity ordering. It does not
predict downstream model quality.

## Method registry visibility

The API and workbench expose 11 typed descriptors:

- four selectable `gated-executable` methods: Full, LoRA, int8-LoRA, QLoRA;
- four nonselectable `experimental` methods: DoRA, BitFit, AdaLoRA, ShareLoRA;
  and
- three nonselectable `research-only` methods: LoReFT, AFLoRA, BiLoRA.

Only selectable descriptors enter the 12 planner rows. The other descriptors
have no compiler ID, export kind, backend, or distribution contract.

## Compiler and bundle support

Supported now:

- atomic no-clobber directory publication;
- deterministic no-clobber ZIP publication;
- exact direct package pins by selected method and runtime;
- a versioned candidate runtime contract with compute, runtime, compiler,
  estimator, evidence, and export identities;
- single, DDP, and conditional LoRA FSDP Accelerate configuration;
- portable CUDA contract, validation, preflight, training-child, and run-parent
  code;
- separate MLX-LM validation, bounded preflight, adapter, data, and run-wrapper
  artifacts;
- structural CUDA full-model or adapter safetensors export;
- MLX-LM adapter output at measured preflight; and
- cleartext source, canonical, and pilot data copies.

Not implemented:

- transitive dependency lock generation;
- encrypted bundle data;
- provider provisioning or cloud infrastructure;
- merged or deployment-specific exporter plugins;
- general or CUDA semantic export inference validation beyond the bounded MLX
  adapter reload check; and
- arbitrary user overrides of generated training source.

## Execution support

Supported now:

- five ordered managed actions;
- exact external Python runtime probing, selection, and private persisted
  configuration;
- persisted local jobs and logs;
- POSIX process-group cancellation;
- one per-user host-global Aptus lease across state roots;
- runtime-specific current train capacity admission;
- unique full-run output paths;
- parent-owned completion verification and recovery; and
- structural output integrity attestation.

Runtime-specific limit:

- CUDA can proceed through all five actions when every prerequisite passes.
- MLX-LM can proceed through all five actions for conditional single-device LoRA
  and QLoRA. Pilot and full training run uninterrupted from the pinned base, and
  crash resume remains unsupported.
- PyTorch MPS has no executable compiler path.
- LM Studio and oMLX remain inference-only and never enter managed training.

Explicitly unsupported:

- a secure multi-user or public jobs service;
- remote scheduler semantics;
- coordination with non-Aptus CUDA programs;
- full-training resume;
- direct portable child execution on Windows; and
- quality, safety, or deployment approval from run completion.

## Future seams

MLX-LM crash resume, full-parameter MLX, DoRA, a PyTorch MPS compiler, ROCm, CPU
training, cloud runners, provider provisioning, automated cost selection,
evaluation policies, exporter plugins, experiment-tracker integration, and MCP
adapters are outside the current support contract.

## Related documentation

- [Method registry](method-registry.md)
- [Dataset schemas](dataset-schemas.md)
- [Configuration defaults](configuration-defaults.md)
- [Plan schema](plan-schema.md)
- [Validation states](validation-states.md)
- [Current capabilities](../product/current-capabilities.md)
- [Release gates](../operations/release-gates.md)
