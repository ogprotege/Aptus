# Apple Silicon runtime and pilot matrix

> **Status:** Active | **Authority:** Measured acceptance record and proposed experiment plan | **Applies to:** Measured 64 GB M5 Pro host | **Audience:** Local experiment operators | **Last reviewed:** 2026-07-27 | **Review by:** 2026-10-27 or before any additional model download

This page records the completed small-model QLoRA acceptance and the proposed
next experiments for the measured 64 GB M5 Pro host. The accepted result proves
only its exact model, revision, synthetic dataset, runtime, plan, bundle, and
actions. The larger-model and LoRA rows remain proposals.

## Measured host

The intended machine is an Apple M5 Pro with an 18-core CPU, 20-core GPU,
16-core Neural Engine, and 64 GB unified memory. Apple lists 307 GB/s memory
bandwidth for this M5 Pro configuration. The CPU and GPU share the same memory
pool. A 64 GB specification is not equivalent to a CUDA GPU with 64 GB of
dedicated VRAM and a separate host-memory pool.

The native Aptus design is macOS 26 first and retains a macOS 15 semantic
material fallback. Both paths use the same private loopback service and runtime
contracts.

Sources:

- [Apple M5 Pro MacBook Pro specifications](https://support.apple.com/en-us/126318)
- [MLX, Apple's array framework for Apple silicon](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX-LM fine-tuning guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

The local Aptus probe records Apple Silicon as `mps` and labels its memory as
shared unified memory. It records live available host memory separately from the
compatibility device. MLX planning uses the lesser of live availability and the
Metal compatibility capacity, then subtracts the reserve.

The current compiler binds single-device LoRA and QLoRA to `mlx-lm` through
`aptus.runtime-contract.v1`. It uses the `aptus-memory-mlx-v2` estimator, exact
pins `mlx==0.31.2` and `mlx-lm==0.31.3`, MLX train and validation JSONL, an
MLX-native adapter export, and runtime-neutral metrics. These candidates are
always conditional. PyTorch MPS has no compiler. CUDA remains an external-host
handoff from this Mac.

## Current method boundary

The Aptus MLX compiler accepts only LoRA and QLoRA. MLX QLoRA requires an
already four-bit pinned model with explicit MLX quantization metadata. Aptus
does not consult the CUDA four-bit device flag, invoke bitsandbytes, or quantize
an unbound model during execution.

The pinned MLX-LM library exposes other features, but library presence is not an
Aptus compiler contract. DoRA, full-parameter MLX fine-tuning, BitFit through
PyTorch MPS, adapter fusion, and crash resume remain outside the current
executable path. Aptus does support uninterrupted full-duration LoRA and QLoRA
adapter runs after `pilot-pass`.

## Completed QLoRA acceptance

On 2026-07-27, two clean workflows reached `measured-run-pass` with:

- model `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at immutable revision
  `53a32aee5e9447773fd2b85988395066aef3700a`;
- the four-row synthetic `examples/support-sft.jsonl` dataset;
- Python 3.12.13, `mlx==0.31.2`, and `mlx-lm==0.31.3`; and
- generated compiler contract `mlx-lm.qlora.v1`.

Each clean workflow completed dependency, model-data, measured-preflight,
uninterrupted pilot, fresh-process adapter reload, explicitly confirmed full
training, final export, a second fresh-process reload, and parent verification.
The full runs completed three optimizer updates, changed 336 adapter tensors,
and generated four tokens after reload. Their highest recorded full-run MLX peak
was 555.1 MiB.

The [immutable acceptance record](evidence/2026-07-27-mlx-lm-acceptance/README.md)
contains the run IDs, timing, hashes, metrics, admission evidence, and retained
logs. This result proves runtime and artifact correctness for the recorded
configuration. It does not establish production throughput, model quality,
usefulness, safety, or broader Apple Silicon fit.

Evidence should now progress in this order:

1. Completed: QLoRA on a small quantized model proved the complete local
   workflow.
2. Unquantized LoRA on a 7B model.
3. QLoRA on a 14B model after the 7B run establishes memory and throughput.
4. The exact Qwen3 MoE compatibility row on a reviewed mixed-layout checkpoint.
5. A tightly bounded 70B QLoRA stress test only after the smaller runs pass.

I would not begin with AFLoRA, BiLoRA, or LoReFT. Each needs a custom training,
state, and inference contract. Their supplied CUDA papers do not establish MLX
behavior.

## Acceptance and proposed runs

Gate 0 is complete. The remaining rows are staged experiment envelopes. Every
future row first needs the standard dependency, model-data, measured-preflight,
and two-update uninterrupted pilot. Only a passing pilot can authorize its
full-duration envelope. Rank and target modules remain compiler-selected plan
facts. The current product does not expose selected-layer or q/v-only MLX
controls.

| Gate | Status | Model | Method | Starting full-run envelope or result | Purpose |
|---|---|---|---|---|---|
| 0 | Accepted | [`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/53a32aee5e9447773fd2b85988395066aef3700a) | QLoRA | Two clean `measured-run-pass` results on four synthetic rows | Prove the complete gated workflow, immutable adapter export, and bounded fresh-process generation |
| 1A | Proposed | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | LoRA | Batch 1, 1,024 tokens, one epoch | Establish a larger unquantized runtime baseline on an Apache-2.0 model |
| 2 | Proposed | [`mlx-community/Qwen2.5-14B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-4bit) | QLoRA | Batch 1, 1,024 tokens, one epoch | Test whether a materially larger model improves the target task within a safe memory reserve |
| 2M | Admission blocked | Revision-pinned Qwen3 30B-A3B MoE checkpoint with four-bit group-64 defaults and eight-bit group-64 router gates | QLoRA | Dependency passed; model-data refused before model load with an 18.932 GiB live-memory shortfall | Accept the exact `qwen3_moe` compatibility row and measure routed-expert memory and throughput |
| 3 | Proposed | [`mlx-community/Llama-3.3-70B-Instruct-4bit`](https://huggingface.co/mlx-community/Llama-3.3-70B-Instruct-4bit) | QLoRA stress test | Batch 1, 512 tokens, pilot only before any full run | Measure bounded pilot fit only. Do not claim useful tuning or quality from this run |

Every proposed model must be replaced by one reviewed immutable revision before
Aptus can compile the run. The accepted 0.5B run is plumbing and runtime
evidence, not a target-quality model. The 7B LoRA run is the first larger
unquantized experiment. The 14B run is a later quality and efficiency candidate.
The MoE row requires one router-gate override per layer, a complete
no-shared-expert topology, attention-only adapters, and the `aptus.facts.v3`
model-facts contract.
Its active parameter count cannot replace
the total resident parameter count. The 70B run is intentionally last.
The [Qwen3 MoE admission record](evidence/2026-07-28-qwen3-moe-admission/README.md)
binds the exact first attempt, packed checkpoint measurement, safe refusal, and
separate synthetic MLX timings. It records no 30B training or generation speed.

Four-bit weights for a 70B model start near 35 GB before quantization metadata,
activations, adapter state, optimizer state, caches, temporary buffers, the
runtime, and macOS use the shared pool. That arithmetic makes a 70B pilot
plausible enough to measure, but not safe enough to promise. Start with at
least 16 GB reserved for the operating system and runtime. Abort on sustained
memory pressure, rapid swap growth, a non-finite metric, or an incomplete
action-owned artifact set.

## Why BitFit was a good idea but the wrong Llama path

BitFit is a real parameter-efficient method. It updates existing bias tensors.
The supplied evidence is strongest on BERT and RoBERTa-style encoder tasks.
Default Llama attention and MLP projections do not add biases, and RMSNorm has
no bias. The historical scripts froze every parameter and then selected names
containing `bias`, but they never required a positive count. A Llama 3.3 70B
run could therefore select few or zero parameters.

Current Aptus executable methods reject an empty, non-finite, or
method-scope-invalid trainable set and bind a digest of trainable names, shapes,
and dtypes. BitFit remains nonselectable. It should become selectable only after
model inspection proves a meaningful existing bias set and a bias-delta save
and reload contract passes. Adding new biases would be a different method.

## Dataset and evaluation contract

Do not train directly from raw captured conversations. Export reviewed records
under the [reviewed corpus contract](../reference/reviewed-corpus-contract.md).
Every chunk or paraphrase from the same work, document, conversation, or seed
must share one `split_group` in the reviewed source record.

The current generated MLX bundle requires at least two usable rows and writes
disjoint `data/mlx/train.jsonl` and `data/mlx/valid.jsonl` files. It pads each
split within itself to a complete micro-batch and records source and compiled
counts in `aptus.mlx-split.v1`. Pilot and full-duration training use those bound
files. Unlike the CUDA full-run splitter, the current MLX compiler does not
claim group-aware subset selection or an exact requested evaluation fraction.
Review the emitted split before training.

For quality experiments, retain three immutable logical sets before compilation:

- `train.jsonl` for approved training rows;
- `valid.jsonl` for model selection and stopping; and
- `test.jsonl` for a final comparison that never affects training choices.

Bind the source dataset digest, all three set digests, the grouping rule, and
the row IDs assigned to each set outside Aptus. Keep the test set outside the
training source. Prompt masking must be enabled for chat or completion rows so
the loss applies to the reviewed assistant target rather than the prompt.

## Required pass criteria

The completed Gate 0 run satisfied these criteria. Every future exact bundle
must satisfy them again before the next larger run starts:

1. immutable model repository and revision;
2. complete environment and package-version record;
3. source dataset digest and bound `aptus.mlx-split.v1` counts;
4. at least two completed optimizer updates and finite train and validation
   losses;
5. exact target coverage with one LoRA A/B pair per planned target and layer,
   and no other trainable tensor;
6. positive adapter delta and positive measured MLX peak;
7. passing live unified-memory admission with the required reserve;
8. immutable action-owned marker, metrics, adapter, and artifact manifests;
9. fresh-process reload of the pinned base plus adapter with one to four
   generated tokens; and
10. explicit `execution_semantics: uninterrupted` and
    `resume_supported: false` evidence.

After `pilot-pass`, an explicitly confirmed full-duration run must start again
from the pinned base. It must complete the plan-derived epoch schedule without
interruption with at least one optimizer update, retain finite loss and exact
target evidence, pass another fresh-process adapter reload, emit
`aptus.mlx-final-export.v1`, and survive parent verification before
`measured-run-pass`. Periodic MLX files are weight snapshots. They are not
resumable checkpoints, and every resume argument fails.

One finite loss is operational evidence only. A quality claim requires a named
task metric, baseline, direction, threshold, test set, and repeated seeds. For
a theological corpus, the rubric should separately score source fidelity,
quotation accuracy, doctrinal classification, unsupported attribution, and
answer usefulness. Human review remains necessary.

## Authorization boundary

This document records only the completed acceptance configuration linked above.
It does not authorize another model or package download or another fine-tuning
run. Future model downloads range from several gigabytes to tens of gigabytes,
and training creates new artifacts. Start a future run only after the immutable
model revision, corpus revision, method, disk budget, output directory, and
compatible external MLX Python are explicitly chosen. LM Studio and oMLX are
loopback inference integrations only and cannot supply that training
environment.

## Related documentation

- [Current capabilities](../product/current-capabilities.md)
- [2026-07-27 MLX-LM acceptance evidence](evidence/2026-07-27-mlx-lm-acceptance/README.md)
- [Method selection guide](../guides/choose-a-method.md)
- [Reviewed corpus contract](../reference/reviewed-corpus-contract.md)
- [Release evidence template](release-evidence-template.md)
