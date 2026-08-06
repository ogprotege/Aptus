# Apple Silicon runtime and pilot matrix

> **Status:** Active | **Authority:** Measured acceptance record and proposed experiment plan | **Applies to:** Measured 64 GB M5 Pro host | **Audience:** Local experiment operators | **Last reviewed:** 2026-08-06 | **Review by:** 2026-10-27 or before any additional model download

This page records the completed small-model QLoRA acceptance and the proposed
next experiments for the measured 64 GB M5 Pro host. The
[2026-08-05 exact-source acceptance](evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
closed the current-source Phase 6 MLX-LM runtime gate twice with fresh,
independent workflows using an
`aptus.training-plan.v5` plan and `aptus.bundle.v3` bundle at source commit
`719255153e3fc7e38e83b5ff826d587e5e58bf80`, tree
`be99f5664ccb580f2600471f1ae3241a294b1a7e`, and bundle fingerprint
`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`.

That result proves only its exact model, immutable revision, synthetic dataset,
host, runtime, policy snapshot, plan, bundle, and ordered actions. It does not
authorize every artifact with the same `model.qwen2-24l.mlx-qlora`
configuration footprint. The July 27 v2/v2 acceptance remains valid historical
evidence for its tested source and contracts; it is not the source of the
current v5/v3 claim. The larger-model and LoRA rows remain proposals. A
separate [2026-08-06 CUDA LoRA single-device
record](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
qualifies one exact external-host workflow only; it is not an Apple matrix
result, repeatability evidence, or support for any other CUDA scope.

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

## Reviewed Qwen2 configuration footprint

Policy `model.qwen2-24l.mlx-qlora` version `1.0.0` recognizes one portable
configuration footprint:

- family `qwen`, provider model type `qwen2`, and architecture
  `Qwen2ForCausalLM`;
- exactly 24 transformer layers and no MoE topology;
- uniform four-bit, group-size-64 quantization with no module overrides; and
- single-device MLX-LM QLoRA through path
  `mlx-lm.qlora.single.dense-causal-lm.v1`.

The path uses the seven dense targets `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj`. Its policy and runtime evidence IDs are
`policy.qwen2-24l.mlx-qlora.v1` and
`runtime.qwen2-0.5b.mlx-qlora.2026-07-27`.

The first ID records implementation review of the configuration-to-path rule.
The second records the July measurements only for
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` at revision
`53a32aee5e9447773fd2b85988395066aef3700a`. The policy does not make those
measurements transferable to another matching artifact. The August 5
exact-source refresh separately proves two fresh current v5/v3 runs for that
same exact artifact and its newly bound source, tree, environment, and bundle
fingerprint; it also does not transfer to a different matching artifact.

## Current Phase 6 QLoRA acceptance

On 2026-08-05, two fresh, clean detached-checkout workflows at exact source
`719255153e3fc7e38e83b5ff826d587e5e58bf80` reached `measured-run-pass` for
the exact pinned Qwen2.5 0.5B artifact. Both independently completed dependency,
model-data, measured-preflight, uninterrupted pilot, confirmed full training,
immutable adapter export, fresh-process reload, and parent-owned completion
promotion under the current v5 plan, v3 bundle, and installed-host policy
snapshot contract.

The two fresh bundles were byte-identical with fingerprint
`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`, and
both full runs produced the same
learned `adapters.safetensors` SHA-256,
`4717543bb38f084573a6f1ea2fa0638d71c1a1a38b1b2103545951e052d5f31b`.
The [exact-source acceptance record](evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
binds the source, host, runtime, model, dataset, policy snapshot, bundle, jobs,
receipts, metrics, and retained sanitized evidence. It establishes runtime and
artifact correctness only for that exact scope, not safety, model quality,
general Qwen2 compatibility, CUDA acceptance, performance, production
throughput, production readiness, or release readiness. Relative to the
unchanged [original Phase 6 baseline](evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md),
only manifested operator `README.md` and `runbook.md` changed; runtime programs
and requirements remained byte-identical.

## Historical completed QLoRA acceptance

On 2026-07-27, two clean workflows reached `measured-run-pass` with:

- model `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at immutable revision
  `53a32aee5e9447773fd2b85988395066aef3700a`;
- the four-row synthetic `examples/support-sft.jsonl` dataset;
- Python 3.12.13, `mlx==0.31.2`, and `mlx-lm==0.31.3`; and
- `aptus.training-plan.v2` and `aptus.bundle.v2`, without the later portable
  model-policy snapshot binding; and
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
It also does not establish current `aptus.training-plan.v5` and
`aptus.bundle.v3` acceptance. The historical
`runtime.qwen2-0.5b.mlx-qlora.2026-07-27` record remains bound to the exact
artifact and contracts that produced it.

With current-source Gate 0 complete for the exact accepted artifact, evidence
should now progress in this order:

1. Unquantized LoRA on a 7B model.
2. QLoRA on a 14B model after the 7B run establishes memory and throughput.
3. The exact Qwen3 MoE compatibility row on a reviewed mixed-layout checkpoint.
4. A tightly bounded 70B QLoRA stress test only after the smaller runs pass.

I would not begin with AFLoRA, BiLoRA, or LoReFT. Each needs a custom training,
state, and inference contract. Their supplied CUDA papers do not establish MLX
behavior.

## Acceptance and proposed runs

Gate 0 is complete under the current v5/v3 contracts only for the exact artifact
and environment in the August 5 record. The remaining rows are staged
experiment envelopes. Every future row, including a different artifact that
matches the reviewed Qwen2 footprint, first needs the standard dependency,
model-data, measured-preflight, and two-update uninterrupted pilot. Only its
current passing pilot can authorize its full-duration envelope. Rank and target
modules remain compiler-selected plan facts. The current product does not
expose selected-layer or q/v-only MLX controls.

| Gate | Status | Model | Method | Starting full-run envelope or result | Purpose |
|---|---|---|---|---|---|
| 0 | Current v5/v3 acceptance passed twice on 2026-08-05 | [`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/53a32aee5e9447773fd2b85988395066aef3700a) | QLoRA | Two clean `measured-run-pass` results on four synthetic rows | Prove the current gated workflow, parent-owned promotion, immutable adapter export, and bounded fresh-process generation for the exact artifact |
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

The current Gate 0 repetitions satisfied these criteria under the v5/v3
contract. Every future exact bundle must satisfy them again before the next
larger run starts. Its v5 plan, embedded canonical snapshot, v3 manifest, and
installed host must also agree on current model policy; a package-free
frozen-snapshot pass alone is not current-host authorization:

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
environment. Current installed Aptus must accept the plan and snapshot under its
host registry before managed admission; `replan_required` means generate a new
plan and bundle rather than modifying the historical artifact.

## Related documentation

- [Current capabilities](../product/current-capabilities.md)
- [2026-08-05 Phase 6 MLX-LM exact-source acceptance evidence](evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
- [2026-08-05 original Phase 6 MLX-LM acceptance baseline](evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
- [2026-07-27 MLX-LM acceptance evidence](evidence/2026-07-27-mlx-lm-acceptance/README.md)
- [Method selection guide](../guides/choose-a-method.md)
- [Reviewed corpus contract](../reference/reviewed-corpus-contract.md)
- [Release evidence template](release-evidence-template.md)
