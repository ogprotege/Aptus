# Fine-tuning method taxonomy

> **Status:** Active | **Authority:** Normative taxonomy | **Applies to:** Aptus 0.2 and the research backlog | **Audience:** Practitioners, researchers, and contributors | **Last reviewed:** 2026-08-03 | **Review by:** 2027-01-22 or when a method lifecycle changes

Aptus models a training plan as a composition of independent choices. It does
not treat every named paper, optimizer, precision, and distributed system as a
peer "method."

The normalized profile is:

```text
facts
+ objective and data contract
+ parameter scope
+ parameterization
+ recipes and modifiers
+ optimizer and schedule
+ compute precision and storage quantization
+ distribution
+ evaluation contract
+ export contract
= one identity-bound candidate
```

Changing any term can change feasibility, behavior, dependencies, checkpoints,
or artifacts. Candidate identity therefore binds the normalized model, dataset,
hardware, target, strategy, resource, and memory-policy facts used to compile
the artifact. Plan identity binds its schema and formula versions, normalized
facts, the semantic policy decision and source,
`model_policy_snapshot_sha256`, the optional inspection receipt with its nested
explanatory decision reason excluded, canonical evidence records, ordered
candidate IDs, and recommendation. Runtime reports separately bind the
installed dependency environment, measured hardware, and validation evidence
to that plan and candidate identity.

This taxonomy incorporates the useful distinctions in
[`Reference/top-50-llm-training-methods.pplx.md`](../../Reference/top-50-llm-training-methods.pplx.md).
It does not import that file's editorial ranking as an Aptus quality score. The
complete source disposition is recorded in the
[reconciliation ledger](../research/reference-and-to-review-reconciliation.md).
The documentation-only [machine-readable catalog](method-catalog.json) mirrors
the current 12-row planner matrix and indexes a wider research backlog. Runtime
authority for method identity and readiness belongs to the 11 descriptors in
`src/aptus/methods/registry.py`. Four are selectable and
`gated-executable`. DoRA, BitFit, AdaLoRA, and ShareLoRA are `experimental`.
LoReFT, AFLoRA, and BiLoRA are `research-only`. The seven nonselectable
descriptors have no compiler or export contract.

## Registry lifecycle

The lifecycle value and selectable flag answer different questions:

- `gated-executable` means the descriptor is selectable and names a compiler,
  export contract, supported backend, and supported placement. Every plan still
  needs the normal feasibility and runtime evidence gates.
- `experimental` means Aptus has accepted the method identity and a concrete
  path to implementation. It remains nonselectable, carries an explicit
  blocker and pilot requirement, and cannot enter a plan.
- `research-only` means Aptus tracks the primary mechanism and missing proof,
  but has not accepted an implementation path. It is also nonselectable.
- A name present only in the documentation research index has no runtime
  descriptor or product lifecycle.

The bootstrap API returns all runtime descriptors for explanation. Only its
separate selectable method list can populate planner preferences.

Selectable method identity does not make every runtime executable. The
`transformers-peft-cuda` runtime binds all four selectable methods to CUDA.
`mlx-lm` binds single-device LoRA and QLoRA to MPS as conditional,
pilot-required paths. `pytorch-mps` is a known runtime without a compiler.

## Axis 1: objective and data contract

The objective defines the loss and the data needed to compute it.

| Objective family | Required data | Current Aptus status |
|---|---|---|
| Supervised fine-tuning (`sft`) | Text or supervised completion/chat rows with non-empty target tokens | Executable, subject to all gates |
| Instruction tuning | Multitask instruction and response rows plus task/template policy | Representable as SFT data, but no separate multitask quality contract |
| Continued pretraining, DAPT, TAPT | Packed raw corpus and causal-LM continuation policy | Research only |
| Offline paired preference optimization: DPO, IPO, ORPO, SimPO | Prompt, chosen, and rejected response contract, with objective-specific variants | Research only |
| Unpaired feedback: KTO, BCO | Prompt, completion, and binary desirability contract | Research only |
| Online RL: PPO, GRPO, DAPO, GSPO, Dr. GRPO, RLOO, REINFORCE++, CISPO | Prompt sampler, reward contract, reference policy rules, rollout system, and online evaluation | Research only |
| Distillation: KD, GKD, MiniLLM | Student inputs plus a versioned teacher and divergence contract | Research only |
| Pretraining from scratch | Architecture definition, corpus, tokenizer, and corpus-scale compute policy | Out of scope for v0.2 |

V0.2 accepts `task="sft"` only. Sequence packing is fail-closed because the
current masking compiler does not implement it. A non-SFT label must never be
routed through the SFT loss merely because a library exposes a trainer class.

## Axis 2: parameter scope

Parameter scope answers which existing parameters may change.

| Scope | Meaning | Current Aptus status |
|---|---|---|
| All parameters | Every base parameter is trainable | `full` path |
| Frozen base plus injected parameters | Base parameters remain frozen; added adapter parameters train | LoRA-based paths |
| Selected existing biases | Existing bias tensors train while all other model parameters remain frozen | BitFit is experimental and nonselectable |
| Other selected existing parameters | Selected layers, tokens, or another sparse subset train | Research index only |
| Representation intervention | Base remains frozen; learned operations alter hidden representations | LoReFT is research-only and nonselectable |
| Student parameters | A student trains against a teacher | Research only |

Scope is distinct from objective. SFT can train all parameters or a LoRA
adapter. A future DPO objective could do the same. The objective determines what
is optimized. Scope determines where updates are allowed.

## Axis 3: parameterization

Parameterization defines the trainable mathematical object.

| Parameterization | Scope | Current Aptus status |
|---|---|---|
| Dense full weights | All parameters | Executable as `full` |
| LoRA low-rank update | Frozen base plus injected low-rank matrices | Executable as `lora`, `int8-lora`, and the LoRA part of `qlora` |
| DoRA magnitude and direction | Frozen base plus magnitude and low-rank directional update | Experimental, nonselectable |
| AdaLoRA adaptive budget | Changing low-rank budget and importance state | Experimental, nonselectable |
| ShareLoRA shared factors | Low-rank factors shared across compatible layers | Experimental, nonselectable |
| BitFit | Selected existing bias tensors | Experimental, nonselectable |
| LoReFT | Hidden-representation interventions | Research-only, nonselectable |
| AFLoRA dynamic freezing | Low-rank parameter groups freeze under a scored schedule | Research-only, nonselectable |
| BiLoRA bilevel update | Disjoint data partitions drive inner and outer optimization | Research-only, nonselectable |
| PiSSA, rsLoRA, LoRA+, prompt, prefix, P-Tuning v2, adapters, IA3, VeRA, BOFT/OFT, MiSS, and LISA | Variant-specific | Documentation research index only |

DoRA means Weight-Decomposed Low-Rank Adaptation
([primary paper](https://arxiv.org/abs/2402.09353)). ReFT means Representation
Finetuning, and LoReFT is a low-rank linear subspace representation
intervention ([primary paper](https://arxiv.org/abs/2404.03592)). The conflicting
expansions in `Reference/Fine-Tuning_Methods.md` are rejected.

## Axis 4: recipes, pipelines, and modifiers

A recipe is a named composition. A pipeline is an outer loop around one or more
training objectives. A modifier alters another training objective.

| Class | Examples | Aptus treatment |
|---|---|---|
| Recipe | QLoRA, LoftQ, QPiSSA | Expand into atomic choices and validate each one |
| Pipeline | RLHF, RLAIF, RAFT, STaR, Self-Instruct, SPIN | Model as a versioned state machine with evidence between stages |
| Modifier | NEFTune, EWC, replay, LwF | Attach to an objective with explicit compatibility and evaluation rules |
| Infrastructure modifier | Gradient checkpointing, sequence packing, offload, FlashAttention | Record as execution policy, not as a fine-tuning objective |

QLoRA is not a new loss. It combines a frozen four-bit base with LoRA adapters
and a memory-aware training recipe
([Dettmers et al.](https://arxiv.org/abs/2305.14314)). The CUDA compiler uses
NF4 and double quantization through bitsandbytes. The MLX-LM compiler requires
an already four-bit MLX model revision with explicit pinned metadata. Its
optimizer choice remains an explicit Aptus policy, not an implied paper default.

## Axis 5: optimizer and schedule

An optimizer changes the update rule. It does not by itself define the
objective or parameter scope.

The CUDA compiler uses these explicit defaults:

- PyTorch AdamW;
- linear scheduler;
- zero weight decay;
- zero warmup steps;
- maximum gradient norm of 1.0;
- method-class learning-rate priors, not tuned optima.

The MLX-LM compiler uses MLX-LM AdamW and gradient checkpointing. It does not
declare a separate learning-rate scheduler, maximum gradient norm, or CUDA
gradient-scaler policy.

Sophia, Adam-mini, SAM, and PagedAdamW8bit are not current Aptus optimizer
choices. The numeric suggestions in `Reference/hparam_methods_reference.md`
remain unverified priors.

GaLore, LOMO, MeZO, and LISA require more than changing an optimizer string.
They change gradient representation, update storage, gradient estimation, or
active parameter scope. They need dedicated compilers, memory models,
checkpoint contracts, and pilots before they can become executable.

## Axis 6: compute precision and storage quantization

Compute precision and base-weight storage are separate facts.

| Runtime and method | Base storage | Compute record | Current rule |
|---|---|---|---|
| CUDA `full` | Unquantized | BF16 | Full FP16 is fail-closed because the generated path does not retain verified FP32 trainable master weights |
| CUDA `lora` | Unquantized frozen base | BF16 or FP16 | Requires target-module inspection and pilot |
| CUDA `int8-lora` | Bitsandbytes eight-bit frozen base | BF16 or FP16 | Every participating GPU must support LLM.int8 and compute capability 7.5 or newer |
| CUDA `qlora` | NF4 four-bit frozen base with double quantization | BF16 or FP16 | Every participating GPU must support the four-bit path and compute capability 6.0 or newer |
| MLX-LM `lora` | Unquantized frozen MLX base | Candidate precision field | Conditional single-device path through uninterrupted adapter training |
| MLX-LM `qlora` | MLX groupwise four-bit frozen base | Candidate precision field | Pinned model metadata must declare four-bit quantization; no CUDA flag is used |

The planner does not assume that nominal total VRAM proves a quantized kernel
works. Declared capability permits planning. Dependency validation, model-data
inspection, measured preflight, and a bounded pilot provide progressively
stronger evidence.

## Axis 7: distribution

Distribution determines placement and communication. It is infrastructure, not
a fine-tuning method.

| Distribution | Meaning | Current rule |
|---|---|---|
| Single | One selected CUDA GPU or Apple unified-memory device | Eligible when a registered runtime binding and resource gates pass |
| DDP | One complete replica per participating GPU | Requires at least two CUDA GPUs, per-device fit, and exact global-batch divisibility |
| FSDP | Selected state is sharded across ranks | Only LoRA is a conditional v0.2 candidate; it uses `use_orig_params=true`, and exact behavior requires the real-model pilot |

DDP never adds device VRAM into one pool. Every replica must fit. Host staging
budgets one model materialization per rank.

Full-parameter FSDP is fail-closed in v0.2. The pinned runtime can upcast
trainable shards and full-state export to FP32, while Aptus has not calibrated
that transient path. Eight-bit LoRA and QLoRA with FSDP are also unsupported.
These rows remain visible so the planner explains the failed rule.

## Axis 8: evaluation

Evaluation must name a target, dataset, metric, baseline, threshold, direction,
sample policy, and uncertainty rule. Operational evidence and quality evidence
are different.

The CUDA runtime can verify this operational behavior through pilot:

- the exact model and tokenizer resolve at the pinned revision;
- supported rows transform without empty supervision;
- target modules exist;
- a bounded pilot performs training work on the compiler-produced pressure set;
- losses are finite and training steps are positive;
- checkpoint continuation is observed across the two pilot phases;
- measured resources and output sizes are recorded.

The MLX-LM runtime uses a different pilot proof. Measured preflight exercises a
bounded real-input adapter update. Pilot starts from the pinned base and runs the
exact model and data without interruption for at least two optimizer updates.
It requires finite losses, exact target binding, positive MLX memory and adapter
delta, live headroom, immutable artifacts, and fresh-process adapter reload with
one to four generated tokens. That reload does not preserve or resume training
state.

This does not prove that a model meets a user quality target. MMLU, HellaSwag,
GSM8K, TruthfulQA, custom task metrics, safety evaluations, regression
baselines, and contamination checks are future evaluation contracts. The
benchmark names in the removed staging sketch are interface ideas only; their
disposition is preserved in the
[reconciliation ledger](../research/reference-and-to-review-reconciliation.md).

## Axis 9: export

Export is an output contract, not a successful-training synonym.

The CUDA compiler emits one of two structural forms:

- full fine-tuning: pinned model configuration, tokenizer material, and
  safetensors weight files;
- LoRA-based training: pinned adapter configuration, tokenizer material, and
  safetensors adapter weights that retain base-model provenance.

MLX measured preflight emits `adapter_config.json` and
`adapters.safetensors` under one owned evidence directory with a path, size, and
SHA-256 manifest. That bounded adapter is preflight evidence, not a pilot or
full-run export. Pilot emits its own immutable adapter tree and fresh reload
evidence. Full training emits `aptus.mlx-final-export.v1` after the same bounded
reload check.

The verifier opens safetensors files, checks non-empty tensor keys, checks index
mappings when present, and binds the file tree and metrics to the run. This is
structural file-tree verification. It is not an inference-parity, merged-model,
serving-latency, GGUF, ONNX, or deployment proof.

GGUF, ONNX, PyTorch pickle export, adapter merge, quantized serving conversion,
model registry publication, and provider deployment are future exporter
contracts. Historical declarations are recorded in the
[reconciliation ledger](../research/reference-and-to-review-reconciliation.md)
and do not make those features current.

## Current executable matrix

The planner enumerates four Aptus method labels across three distribution
choices, producing 12 visible candidates. Runtime binding determines which
cells have a guarded compiler path. It does not mean every model, dataset,
machine, or pinned library combination has passed its required evidence.

| CUDA SFT parameter and storage path | Single | DDP | FSDP |
|---|---|---|---|
| Full, unquantized | Eligible only with BF16 and all resource gates | Eligible only with BF16, two or more GPUs, per-replica fit, and exact batch arithmetic | **Unsupported, fail-closed** |
| LoRA, unquantized base | Eligible with target-module and resource gates | Eligible with two or more GPUs, per-replica fit, and exact batch arithmetic | **Conditional** under an uncalibrated sharding prior; real pilot required |
| Eight-bit LoRA | Eligible with explicit eight-bit capability and resource gates | Eligible with explicit capability on every rank, per-replica fit, and exact batch arithmetic | **Unsupported** |
| QLoRA, NF4 plus double quantization | Eligible with explicit four-bit capability and resource gates | Eligible with explicit capability on every rank, per-replica fit, and exact batch arithmetic | **Unsupported** |

MLX-LM adds two single-device MPS paths:

| MLX-LM path | Single | DDP | FSDP |
|---|---|---|---|
| LoRA, unquantized base | **Conditional**, executable through uninterrupted pilot and full-duration adapter training | Unsupported | Unsupported |
| QLoRA, pinned MLX four-bit base | **Conditional**, executable through uninterrupted pilot and full-duration adapter training after metadata verification | Unsupported | Unsupported |
| Full or eight-bit LoRA | Unsupported | Unsupported | Unsupported |

Every guarded row still requires:

1. a matching runtime contract, compute backend, and supported model family;
2. an immutable provider model commit and explicit training permission;
3. a supported SFT dataset schema and content-bound digest;
4. sequence length within the model context;
5. point and upper memory checks, host RAM, disk, and user reserve;
6. dependency and environment validation;
7. model-data inspection and measured preflight;
8. the required bounded pilot and current full-training admission before any
   full run;
9. explicit full-run confirmation;
10. aggregate process success and artifact verification.

No row bypasses these gates because a paper, README, or user preference calls a
method efficient. MLX-LM satisfies item 8 only through its uninterrupted pilot.
Its fresh-process adapter generation is not CUDA-style checkpoint continuation
and does not authorize crash resume.

## Nonselectable and future method families

The runtime registry exposes seven nonselectable descriptors so the API and
workbench can explain their evidence and blockers:

- Experimental: DoRA, BitFit, AdaLoRA, and ShareLoRA.
- Research-only: LoReFT, AFLoRA, and BiLoRA.

None has a compiler ID, export contract, supported backend, or supported
placement. The documentation research index also tracks the following broader
backlog. Those names have no runtime descriptor unless listed above.

- Objectives: continued pretraining, DAPT, TAPT, instruction-specific
  multitask tuning, DPO, IPO, ORPO, SimPO, KTO, BCO, PPO, GRPO, DAPO, GSPO,
  Dr. GRPO, RLOO, REINFORCE++, CISPO, KD, GKD, MiniLLM, and from-scratch
  causal-LM pretraining.
- Additional parameterizations: Prompt Tuning, Prefix-Tuning, P-Tuning v2,
  PiSSA, Houlsby adapters, IA3, LoRA+, rsLoRA, VeRA, BOFT, OFT, MiSS, LoHa,
  LoKr, X-LoRA, Poly, HRA, RandLoRA, SHiRA, RoAd, C3A, FourierFT, MoRA, and
  Trainable Tokens.
- Recipes and pipelines: LoftQ, QPiSSA, RLHF, RLAIF, RAFT, STaR,
  Self-Instruct, and SPIN.
- Modifiers and update strategies: NEFTune, EWC, Experience Replay, LwF,
  GaLore, LISA, LOMO, MeZO, SAM, Sophia, and Adam-mini.
- Infrastructure: ZeRO, tensor parallelism, pipeline parallelism, CPU or NVMe
  offload, FlashAttention selection, and sequence packing.
- Backends and runners: ROCm execution, a PyTorch MPS compiler, CPU training,
  managed cloud runners, and provider execution connectors. MLX-LM LoRA and
  QLoRA uninterrupted adapter training is current. MLX crash resume remains
  unsupported.
- Evaluation and export: named benchmark suites, custom target thresholds,
  inference-parity checks, GGUF, ONNX, adapter merge, publication, and
  deployment.

Names found only in the rejected unsourced list, including DoReFT, FishDip,
FAR, CIAT, KODA, MerA, PHA, and PaFi, are not in the research catalog. They
first require a primary source and a distinct mechanism.

## Admission rule for a new executable method

A nonselectable descriptor or documentation-only research item becomes
executable only when all of these exist:

1. primary-source identity and a stable, distinct definition;
2. a pinned dependency and model-family compatibility rule;
3. objective-specific data and evaluation schemas;
4. typed configuration with bounded values and explicit defaults;
5. deterministic compiler output and a portable entry point;
6. transparent VRAM, host RAM, disk, checkpoint, and export estimates;
7. static, dependency, environment, model-data, and measured preflight gates;
8. a bounded real-model pilot with continuation evidence;
9. cancellation, lease, restart, unique no-clobber run directories, and completion attestation;
10. an artifact verifier and negative tests for every unsupported combination;
11. a gated runtime descriptor with a compiler ID, export contract, supported
    backend and placement, and `selectable=true`; and
12. real target-runtime evidence for the pinned stack before a release support
    claim.

## Related documentation

- [Method selection guide](../guides/choose-a-method.md)
- [Method registry](../reference/method-registry.md)
- [Research index](../research/index.md)
- [Adding a method](../contributing/adding-a-method.md)
