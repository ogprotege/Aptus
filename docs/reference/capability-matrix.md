# Capability Matrix

| Metadata | Value |
| --- | --- |
| Status | Active, unreleased engineering preview |
| Audience | Operators, product owners, method authors, and release reviewers |
| Authority | Normative v0.2 support boundary |
| Last reviewed | 2026-07-22 |
| Next review | 2026-10-22, or sooner when the method registry, planner, or compiler changes |

This matrix distinguishes a planner path from target-host proof. A planner row
marked supported can become viable when all facts and analytic gates pass. It
still requires static, dependency, model-data, measured-preflight, and pilot
evidence for the exact bundle and host.

No real CUDA pilot has been completed on the current development Mac. Aptus
v0.2 is not release-ready.

## Executable method and placement matrix

| Method | Single | DDP | FSDP | Export contract |
| --- | --- | --- | --- | --- |
| Full | Planner-supported with BF16 | Planner-supported with BF16 and at least 2 GPUs | Unsupported | Full-model safetensors |
| LoRA | Planner-supported | Planner-supported with at least 2 GPUs | Conditional with at least 2 GPUs | PEFT adapter safetensors |
| int8-LoRA | Planner-supported with eight-bit capability | Planner-supported with shared eight-bit capability and at least 2 GPUs | Unsupported | PEFT adapter safetensors |
| QLoRA | Planner-supported with four-bit capability | Planner-supported with shared four-bit capability and at least 2 GPUs | Unsupported | PEFT adapter safetensors |

All 12 method and placement pairs remain in the candidate matrix. Unsupported
and infeasible rows are evidence, not hidden branches.

### Method-specific hard boundaries

- Full training in FP16 is unsupported because the generated path does not
  retain a verified FP32 trainable master-weight contract.
- Full FSDP is unsupported because the pinned transient and export path is not
  calibrated safely.
- int8-LoRA FSDP and QLoRA FSDP are outside the verified compiler matrix.
- LoRA FSDP uses `use_orig_params=true` and remains conditional even when the
  analytic envelope fits.
- Adapter methods require every target module in the family catalog to exist on
  the loaded revision.

## Precision and quantization

| Path | Planner rule | Runtime proof still required |
| --- | --- | --- |
| BF16 | Selected only when every participating device declares BF16 | Actual device, stack, method, and pilot behavior |
| FP16 full | Always unsupported | No launch |
| FP16 adapters | Selected when participating devices do not all declare BF16 | AMP behavior and exact pilot |
| Unquantized base | Full and LoRA | Model load and measured peak |
| INT8 bitsandbytes base | int8-LoRA only | Exact bitsandbytes load and kernel path |
| NF4 double-quantized base | QLoRA only | Exact bitsandbytes load and kernel path |
| FP32 compute | Not enumerated | Future contract |
| FP8 | Not enumerated | Future contract |

Local probe fallback derives four-bit eligibility at CUDA compute capability
6.0 or newer and eight-bit eligibility at 7.5 or newer. Manual planning still
requires explicit capability flags. Runtime model-data, synthetic, and pilot
checks remain authoritative for the pinned software stack.

## Backend matrix

| Backend | Accepted fact value | Local discovery | Planner execution rows | Compiler and runtime |
| --- | ---: | ---: | ---: | ---: |
| CUDA | Yes | Yes | Yes | Yes, subject to gates |
| ROCm | Yes | No supported probe result | Explicitly unsupported | No |
| MPS | Yes | Apple shared-memory inventory only | Explicitly unsupported | No |
| CPU | Yes | No supported accelerator result | Explicitly unsupported | No |
| MLX | Not a backend enum | No | No | No |

On Darwin arm64 without CUDA, discovery reports one `mps` device backed by the
measured shared unified-memory pool. It does not infer BF16, four-bit, eight-bit,
or current free memory when unavailable. The CUDA compiler never silently routes
through MPS or MLX.

## Distribution behavior

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
mapped. Model-data validation checks the loaded parameter count, hidden size,
optional intermediate size, layers, context length, and adapter targets.

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
- exact direct package pins by selected method;
- single, DDP, and conditional LoRA FSDP Accelerate configuration;
- portable contract, validation, preflight, training-child, and run-parent code;
- structural full-model or adapter safetensors export; and
- cleartext source, canonical, and pilot data copies.

Not implemented:

- transitive dependency lock generation;
- encrypted bundle data;
- provider provisioning or cloud infrastructure;
- merged or deployment-specific exporter plugins;
- semantic export inference validation; and
- arbitrary user overrides of generated training source.

## Execution support

Supported now:

- five ordered managed actions;
- persisted local jobs and logs;
- POSIX process-group cancellation;
- one per-user host-global Aptus lease across state roots;
- current train capacity admission;
- unique full-run output paths;
- parent-owned completion verification and recovery; and
- structural output integrity attestation.

Explicitly unsupported:

- a secure multi-user or public jobs service;
- remote scheduler semantics;
- coordination with non-Aptus CUDA programs;
- full-training resume;
- direct portable child execution on Windows; and
- quality, safety, or deployment approval from run completion.

## Future seams

ROCm, MPS or MLX execution, cloud runners, provider provisioning, automated
cost selection, evaluation policies, exporter plugins, experiment-tracker
integration, and MCP adapters are outside the current support contract.

## Related documentation

- [Method registry](method-registry.md)
- [Dataset schemas](dataset-schemas.md)
- [Configuration defaults](configuration-defaults.md)
- [Plan schema](plan-schema.md)
- [Validation states](validation-states.md)
- [Current capabilities](../product/current-capabilities.md)
- [Release gates](../operations/release-gates.md)
