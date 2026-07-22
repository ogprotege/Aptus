# Apple Silicon fine-tuning pilot matrix

> **Status:** Experimental | **Authority:** Proposed experiment plan | **Applies to:** Measured 64 GB M5 Pro host | **Audience:** Local experiment operators | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or before any model download

Status: machine-specific recommendation for the measured 64 GB M5 Pro host.
These are proposed experiments, not completed Aptus validation runs.

## Measured host

The intended machine is an Apple M5 Pro with an 18-core CPU, 20-core GPU,
16-core Neural Engine, and 64 GB unified memory. Apple lists 307 GB/s memory
bandwidth for this M5 Pro configuration. The CPU and GPU share the same memory
pool. A 64 GB specification is not equivalent to a CUDA GPU with 64 GB of
dedicated VRAM and a separate host-memory pool.

Sources:

- [Apple M5 Pro MacBook Pro specifications](https://support.apple.com/en-us/126318)
- [MLX, Apple's array framework for Apple silicon](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX-LM fine-tuning guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

The local Aptus probe now records Apple Silicon as `mps` and labels its memory
as shared unified memory. The current v0.2 compiler still fails closed because
it emits a CUDA Transformers, PEFT, Accelerate, and bitsandbytes bundle. It does
not silently route that bundle through MPS. A separate MLX compiler needs its
own estimator, dependency lock, trainer, checkpoint verifier, and export
contract.

## Methods I would test

MLX-LM currently documents `lora`, `dora`, and `full` fine-tuning. It treats
LoRA over a quantized model as QLoRA. It also supports local JSONL train,
validation, and test files, prompt masking, adapter resume, evaluation, and
adapter fusion. Those library capabilities still require an Aptus pilot before
they become planner support.

I would test methods in this order:

1. QLoRA on a small quantized model to prove the complete local workflow.
2. Unquantized LoRA and DoRA on the same 7B model for a controlled method
   comparison.
3. QLoRA on a 14B model after the 7B run establishes memory and throughput.
4. A tightly bounded 70B QLoRA stress test only after the smaller runs pass.
5. A separate BitFit architecture diagnostic on a bias-bearing small model.

I would not begin with AFLoRA, BiLoRA, or LoReFT. Each needs a custom training,
checkpoint, and inference contract. Their supplied CUDA papers do not establish
MLX behavior.

## Proposed runs

| Gate | Model | Method | Starting envelope | Purpose |
|---|---|---|---|---|
| 0 | [`mlx-community/Llama-3.2-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit) | QLoRA | Rank 8, last 4 layers, q/v targets, batch 1, 512 tokens, 50 iterations | Prove data loading, masked loss, nonzero trainable census, memory reporting, save, reload, and deterministic generation |
| 1A | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | LoRA | Rank 8, last 8 layers, q/k/v/o targets, batch 1, 1,024 tokens, 200 iterations | Establish the real corpus baseline on an Apache-2.0 model |
| 1B | Same immutable Qwen revision | DoRA | Same data, split, layers, rank, seed, sequence length, and iteration budget as 1A | Measure DoRA's actual memory, speed, and held-out quality against LoRA |
| 2 | [`mlx-community/Qwen2.5-14B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-4bit) | QLoRA | Rank 8, last 8 layers, batch 1, 1,024 tokens, gradient checkpointing, 50-iteration pilot before any longer run | Test whether a materially larger model improves the target task within a safe memory reserve |
| 3 | [`mlx-community/Llama-3.3-70B-Instruct-4bit`](https://huggingface.co/mlx-community/Llama-3.3-70B-Instruct-4bit) | QLoRA stress test | Rank 8, last 4 layers, q/v targets, batch 1, 512 tokens, gradient checkpointing, 10 iterations | Measure fit only. Do not claim useful tuning or quality from this run |
| B | [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) | BitFit diagnostic through PyTorch MPS | Existing bias tensors only, exact name-shape-dtype digest, short causal-LM pilot | Prove the bias-only compiler on an architecture that actually has biases |

The 3B run is plumbing evidence, not a target-quality model. The 7B LoRA and
DoRA pair is the first meaningful experiment. The 14B run is the likely local
quality and efficiency candidate. The 70B run is intentionally last.

Four-bit weights for a 70B model start near 35 GB before quantization metadata,
activations, adapter state, optimizer state, caches, temporary buffers, the
runtime, and macOS use the shared pool. That arithmetic makes a 70B pilot
plausible enough to measure, but not safe enough to promise. Start with at
least 16 GB reserved for the operating system and runtime. Abort on sustained
memory pressure, rapid swap growth, a non-finite metric, or an incomplete
checkpoint.

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
must share one `split_group`. Aptus keeps declared groups on one side of the
train and evaluation boundary. The requested evaluation fraction can move when
a group is indivisible, so review the recorded target, realized fraction, and
row error.

For the manual MLX experiments, materialize three immutable files before the
run:

- `train.jsonl` for approved training rows;
- `valid.jsonl` for model selection and stopping; and
- `test.jsonl` for a final comparison that never affects training choices.

Bind the source dataset digest, all three output digests, the grouping rule,
and the row IDs assigned to each split. Use the same split for LoRA and DoRA.
Prompt masking must be enabled for chat or completion rows so the loss applies
to the reviewed assistant target rather than the prompt.

## Required pass criteria

Each run must produce all of these before the next larger run starts:

1. immutable model repository and revision;
2. complete environment and package-version record;
3. source and split dataset digests with zero declared-group overlap;
4. positive trainable tensor and parameter counts with a stable descriptor
   digest;
5. finite initial, training, and validation losses;
6. measured peak unified memory, tokens per second, and iteration time;
7. a checkpoint that resumes for at least one additional optimizer step;
8. a final adapter whose configuration and weights pass structural checks;
9. a fresh-process adapter reload and bounded generation test; and
10. comparison against the untouched base model on the same held-out rows.

One finite loss is operational evidence only. A quality claim requires a named
task metric, baseline, direction, threshold, test set, and repeated seeds. For
a theological corpus, the rubric should separately score source fidelity,
quotation accuracy, doctrinal classification, unsupported attribution, and
answer usefulness. Human review remains necessary.

## Authorization boundary

No model or package download and no fine-tuning run is part of this repository
change. Model downloads range from several gigabytes to tens of gigabytes, and
training creates new artifacts. Start only after the model, corpus revision,
method, disk budget, and output directory are explicitly chosen.

## Related documentation

- [Current capabilities](../product/current-capabilities.md)
- [Method selection guide](../guides/choose-a-method.md)
- [Reviewed corpus contract](../reference/reviewed-corpus-contract.md)
- [Release evidence template](release-evidence-template.md)
