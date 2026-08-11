# CUDA Campaign Phase 1 Protocol

| Metadata | Value |
| --- | --- |
| Status | Frozen Phase 1 protocol; implementation pending |
| Audience | Campaign operators, Aptus maintainers, evidence reviewers, and release reviewers |
| Authority | Normative experiment protocol for the bounded RTX 3050 campaign; non-normative for current Aptus capability |
| Machine companion | [`cuda-campaign-protocol.v1.json`](cuda-campaign-protocol.v1.json) |
| Operational plan | [RTX 3050 CUDA empirical evidence campaign](../operations/cuda-empirical-campaign.md) |
| Owner | CUDA runtime and release evidence |
| Last reviewed | 2026-08-08 |
| Next review | Before Phase 2 implementation, before any protocol amendment, or by 2026-09-08 |

This page freezes the human-readable Phase 1 decisions for Aptus's bounded
single-device CUDA evidence campaign. The operational campaign plan owns the
phase order. The machine companion owns the exact versioned literals and
record shapes. A divergence between this page, its machine companion, a
normative release gate, or executable code is a blocking protocol defect; do
not choose a convenient interpretation and continue.

Freezing a protocol does not implement it and does not establish a runtime
result. Aptus at the Phase 1 source still lacks parts of the candidate-selection,
seed, duration, token-counting, progress, capture, sanitizer, telemetry, and
watchdog contracts described here. Only later reviewed code and dated evidence
may establish those capabilities or measured results.

## Required phase order

The order is fail-closed:

1. Phase 0 recovers and protects the August 6 raw evidence without changing the
   Ubuntu source, package environment, Aptus state, or runtime.
2. Phase 1 freezes this protocol and its machine companion.
3. Phase 2 implements the capture, telemetry, watchdog, sealing, retrieval, and
   constructive sanitizer infrastructure. It then publishes and independently
   reviews the sanitized August 6 recovery supplement.
4. The reviewed Phase 2 recovery supplement must merge before Phase 3 changes
   the planner, compiler, or runtime.
5. Phase 3 implements explicit candidate selection and the frozen measurement
   controls.
6. Only after the applicable Phase 0, 2, and 3 gates pass may Phase 4 mutate the
   target host to install the frozen campaign source and dependencies.

Phase 0 recovery copies are the only host writes authorized before that point.
Neither a complete Phase 0 inventory nor this Phase 1 document independently
authorizes installation, cleanup, model download, or Aptus execution.
Where this protocol explicitly permits a measurement control to be deferred,
the dependent metric or later phase remains forbidden; deferral never permits a
weaker substitute.

## Claim and evidence boundary

This campaign characterizes one bound Ubuntu host with one RTX 3050. It may
support exact-host facts about workflow completion, elapsed time, memory,
utilization, temperature, sampled power, estimated GPU energy, artifact size,
bounded fit, cancellation, lease behavior, and recovery. It does not by itself
establish model quality, safety, production throughput, universal fit, cost,
statistical estimator calibration, or a universal method ranking.

Every value retains its evidence class:

- **Declared** values come from a configuration or operator and were not
  observed by the run.
- **Inferred** values are deterministic consequences of bound declared or
  measured facts.
- **Estimated** values come from a model or numerical approximation. In
  particular, sampled-power integration is estimated GPU energy, not measured
  host energy.
- **Measured** values were observed by the bound runtime or capture process on
  the exact host during the exact run.

The slot, native-outcome, and evidence axes stay independent:

| Axis | Allowed values | Meaning |
| --- | --- | --- |
| Slot status | `started`, `planned-not-started` | Whether the harness activated the frozen slot |
| Native outcome | `passed`, `refused`, `failed`, `cancelled`, `timed-out`, `guard-blocked`, `unknown` | What execution did after a slot started |
| Evidence status | `protocol-valid`, `capture-invalid`, `not-started` | Whether the protocol completely captured a started slot |

A refusal is negative evidence, never a pass. A capture problem changes the
evidence status without rewriting the native outcome. Only the intersection of
native `passed` and evidence `protocol-valid` may support a passing result.
Every frozen slot remains in its denominator, and no failed, refused,
cancelled, timed-out, guard-blocked, unknown, capture-invalid, or unstarted slot
is replaced.

## Versioned records and identifiers

The machine companion uses schema `aptus.cuda-campaign-protocol.v1`. Canonical
JSON uses sorted keys, two-space indentation, UTF-8, no non-finite numbers, and
one trailing newline. SHA-256 values are lowercase 64-character hexadecimal
digests over exact bytes.

Content-addressed identifiers use the compact, sorted, UTF-8 JSON identity
payload defined by the companion, serialized with `ensure_ascii=false` and
`allow_nan=false`. They contain a 20-hexadecimal digest suffix:

| Prefix | Identity |
| --- | --- |
| `campaign_` | The complete bounded campaign |
| `cohort_` | One frozen comparison question, controls, schedule, and decision rule |
| `cell_` | One model, method, placement, stable configuration, and seed policy, excluding actual repetition seeds |
| `slot_` | One predeclared role, block, ordinal, order position, and scheduled seed |
| `exec_` | Every behavior-affecting value for one started attempt, including `split_seed`, `training_seed`, and `data_order_seed` |

Runtime and protected-storage identifiers use cryptographically random
32-hexadecimal suffixes:

| Prefix | Identity |
| --- | --- |
| `xrun_` | One actual experiment invocation |
| `artifact_` | One protected sealed artifact |
| `copy_` | One protected recovery or evidence copy |
| `host_` | One public-safe opaque host identity |
| `domain_` | One independently failing storage domain |

A slot that never starts has no execution-configuration or experiment-run ID.
Changing a held comparison factor creates a new cell. Changing a behavior
value, including any actual seed, creates a new execution configuration.
Fresh paths and timestamps create a new experiment run without changing its
cell. Aptus has no generic `bundle_id`; records bind the real plan ID,
candidate ID, manifest digest, archive digest, and bundle fingerprint.

## Frozen deterministic fixtures

### Contract fixture

`examples/support-sft.jsonl` remains the four-row synthetic contract smoke
fixture with SHA-256
`bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44`.
It exercises bounded workflow contracts and is not a performance or quality
benchmark.

### Campaign fixture

The measured campaign uses
[`examples/cuda-campaign-sft-v1.jsonl`](../../examples/cuda-campaign-sft-v1.jsonl),
generated by
[`tools/generate_cuda_campaign_fixture.py`](../../tools/generate_cuda_campaign_fixture.py).

| Field | Frozen value |
| --- | --- |
| Fixture ID | `aptus.cuda-campaign-sft.v1` |
| Generator | `aptus.cuda-campaign-fixture-generator.v1` |
| Generator seed | `20260808` |
| Generator SHA-256 | `e2ca1d29f3eeebb3d1c2d07916086748de1d698b383c858e16cdaf3ece09e230` |
| Rows | 512 |
| Split groups | 128 groups of exactly 4 rows |
| Byte size | 1,635,765 |
| SHA-256 | `6d90599e949bf2698b940e0c159e1fa24f3dc0c162005546bd270fc761aac7f2` |
| Schema | `row_id`, `target_content_words`, `prompt`, `completion`, `split_group` |
| Rights | Project-generated synthetic text; no external corpus |

Every `row_id`, prompt, and group identity is stable and unique as applicable.
The content-word bands are generator controls, not claims about a model
tokenizer:

| Target content words | Row count |
| ---: | ---: |
| 128 | 256 |
| 256 | 128 |
| 512 | 64 |
| 1,024 | 32 |
| 2,048 | 32 |

The frozen split seed is `424242`, with evaluation fraction `0.125`. The
current split contract produces exactly 448 training rows in 112 groups and 64
evaluation rows in 16 groups. Its assignment SHA-256 is
`7e9e747a6e69868d2d542137468cd1baf3d81d7aaac1de29ed14e4dd83b428ed`.

| Target content words | Training rows | Evaluation rows |
| ---: | ---: | ---: |
| 128 | 232 | 24 |
| 256 | 112 | 16 |
| 512 | 56 | 8 |
| 1,024 | 24 | 8 |
| 2,048 | 24 | 8 |

Phase 4 must reproduce the group, row, band, and assignment counts and the
assignment digest before a measured slot starts. Training seeds never change
this split.

Phase 1 also performed a nonqualifying, read-only tokenizer preview against the
tokenizer recovered from the exact 135M anchor revision. This preview is a
protocol-design check, not sealed campaign evidence:

| Tokenizer binding | Frozen preview value |
| --- | --- |
| Runtime class | `GPT2Tokenizer` |
| Maximum length | 8,192 |
| `tokenizer.json` byte size | 3,522,871 |
| `tokenizer.json` SHA-256 | `bf346d64f6f0fbcefb4c1b6928a98241467dff36c6fbae5fe1785c4ff90667f4` |
| `tokenizer_config.json` byte size | 452 |
| `tokenizer_config.json` SHA-256 | `9b6f7008bcd69b60572d2e15b28caa540d605df1c08149553296574f66545e53` |
| Canonical per-row token-count manifest byte size | 63,234 |
| Canonical per-row token-count manifest SHA-256 | `fa8a4c9223e47fa95cb163db871c35159978b92c4ea559b95e8719697c7be9f6` |

The recovered public behavior binding uses BOS `<|im_start|>`, EOS and pad
`<|im_end|>`, unknown token `<|endoftext|>`, and
`clean_up_tokenization_spaces=false`. For each fixture row, encode the prompt
with special tokens enabled, encode the completion with special tokens disabled,
append EOS only when it is configured and the sequence is not already terminal,
and count the total before truncation. `completion_tokens` is the completion-ID
count after that conditional EOS append. In fixture order, the canonical
manifest is one JSON array whose row objects contain exactly `row_id`,
`target_content_words`, `completion_tokens`, and
`total_tokens_before_truncation`. Render it with sorted keys, UTF-8,
`ensure_ascii=false`, `allow_nan=false`, compact comma/colon separators, and
exactly one trailing LF.

Its pre-truncation total-token bands were:

| Target content words | Minimum tokens | Median tokens | Maximum tokens |
| ---: | ---: | ---: | ---: |
| 128 | 148 | 149 | 152 |
| 256 | 276 | 279 | 286 |
| 512 | 534 | 540 | 546 |
| 1,024 | 1,052 | 1,060 | 1,071 |
| 2,048 | 2,089 | 2,099 | 2,110 |

The recovered Phase 1 preview inventory contains the two tokenizer files above;
it is not permission to assume a fresh provider inventory contains only those
files. Phase 4 must resolve, bind, hash, and seal the entire fresh tokenizer
inventory, reproduce these counts from it, and seal the exact canonical per-row
manifest before any campaign activation. A mismatch blocks activation; it is
not waived or repaired by adopting the Phase 1 preview. No frozen fixture row
reaches 4,096 tokens, so the Phase 8 sequence frontier stops at 2,048 unless a
separately frozen longer fixture is added through a reviewed protocol amendment
before any result is observed.

The fixture measures runtime behavior only. Its synthetic completions have no
external task-validity ground truth and cannot support a model-quality claim.

## Frozen model and method scope

The same-family core staircase uses these immutable revisions:

| Cell label | Model repository | Immutable revision |
| --- | --- | --- |
| 135M anchor | `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| 360M | `HuggingFaceTB/SmolLM2-360M-Instruct` | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| 1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` |

Each must still pass exact provider inspection, license review, model-file
hashing, policy evaluation, target-module inspection, static validation,
dependency validation, model-data validation, measured preflight, and pilot.
An immutable revision is necessary but does not manufacture a viable
candidate.

The four single-device method cells are:

| Method | Frozen path boundary |
| --- | --- |
| `full` | BF16, unquantized, `single`; only when planning and live admission pass |
| `lora` | Supported precision, unquantized adapter, `single` |
| `int8-lora` | Bitsandbytes INT8 base, `single`; exact kernel and environment gates required |
| `qlora` | Bitsandbytes NF4 double-quantized base, `single`; exact kernel and environment gates required |

No DDP or FSDP runtime cell is part of this one-GPU campaign. Qwen2, MoE, and
multimodal models are excluded. Phase 4 may freeze at most one provider-inspected
dense text artifact from each of these breadth repositories before any breadth
outcome is observed:

- `Qwen/Qwen3-0.6B`;
- `google/gemma-3-1b-it`, after the operator accepts its license; and
- `mistralai/Mistral-7B-v0.3`.

Their immutable revisions and admitted methods must be added through a reviewed
protocol amendment before their attempt ledgers are sealed. No favorable or
unfavorable result may cause an artifact substitution.

The independently reviewed [Phase 7 architecture-breadth
amendment](../operations/evidence/2026-08-11-cuda-phase7-breadth-amendment/README.md)
froze the following exact dispositions. Its first cohort then stopped during
model-data validation because the amendment used serialized tensor elements as
the runtime parameter declaration. The independently reviewed [parameter-
semantics correction](../operations/evidence/2026-08-11-cuda-phase7-breadth-parameter-correction/README.md)
kept the model and method disposition but required a fresh reviewed cohort. A
second cohort stopped before optimizer work when Linux admission excluded
reclaimable page cache. After that probe was corrected, a third independently
reviewed [breadth cohort](../operations/evidence/2026-08-11-cuda-phase7-breadth-stability/README.md)
passed conditioning, all three exploratory slots, and the common stability
contract without replacement.

| Repository | Immutable revision | Amendment disposition |
| --- | --- | --- |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Admit single-device BF16 `lora` only; three exploratory slots with the frozen Phase 7 seeds |
| `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | Exclude because the operator has not accepted the manual provider license gate; no planner cell or informal retry |
| `mistralai/Mistral-7B-v0.3` | `caa1feb0e54d415e2df31207e5f4e273e33509b1` | Exclude because no single-device method is planner-admitted on the frozen host; no replacement |

For the admitted Qwen artifact, the amendment binds all seven execution-file
digests, 1,519,182,365 exact artifact bytes, 751,632,384 serialized state-
dictionary tensor elements, all seven LoRA target modules across 28 layers,
and a 512-row tokenizer manifest. The correction binds 596,049,920 unique
loaded model parameters. Qwen ties its 155,582,464-element input embedding and
output head, which exactly explains the difference between those two counts.
The other two repositories retain their exact provider-declared artifact
inventories as negative admission evidence; their model bytes are not implied
to have been downloaded or validated locally. Any change to these revisions,
licenses, artifact selections, hardware capability facts, or admitted methods
requires another reviewed amendment before a replacement or additional ledger
exists.

## Environment and cache controls

Phase 4 freezes clean, separately identified environments for unquantized and
bitsandbytes paths when their installed closures differ. Direct pins and the
complete installed distribution closure are part of each cell. Full and LoRA
may share an unquantized environment only when its exact closure is frozen;
int8-LoRA and QLoRA may share a bitsandbytes environment under the same rule.

Installation and model download are setup records, not training performance.
Publish their duration only when capture started before the exact setup command.
Otherwise exclude them rather than reconstructing a time. Before measured
slots, all model files are resolved, hashed, and available under the frozen
warm-cache policy. Measured actions use local immutable files and perform no
network download.

The campaign intentionally uses:

- a warm provider/model and normal operating-system page cache;
- no `drop_caches`, cache eviction, overclock, power-limit change, or persistence
  mutation;
- a fresh worker process, CUDA context, state root, bundle, run directory,
  output directory, and capture directory for every attempt; and
- exactly one nonqualifying managed conditioning attempt for each new
  model/method/environment cell, followed by the complete cooldown rule.

The conditioning attempt uses `split_seed=424242`, `training_seed=17`, and
`data_order_seed=1000017`, the same cell configuration, `local_files_only`, and
fresh worker, CUDA context, state, bundle, run, output, and capture paths. It
performs complete validation and the bounded pilot only; it does not start
confirmed full training or export. A non-passing or capture-invalid conditioning
attempt blocks that cell pending diagnosis. Conditioning attempts are published
as rehearsals, never replaced, and never enter measured aggregates. The first
eight optimizer steps of a measured run remain in its end-to-end duration and
resource peaks. They are excluded only from the separately labeled steady-state
step-rate calculation.

## Anchor execution configuration

Phases 5 and 6 hold these controls constant:

| Control | Frozen value |
| --- | --- |
| Model | 135M anchor revision above |
| Dataset | `aptus.cuda-campaign-sft.v1` |
| Placement | `single`, world size 1 |
| Compute precision | BF16 for every method; unsupported or inadmissible is `planned-not-started`, with no fallback |
| Sequence length | 256 |
| Effective batch | 8 |
| Per-device micro-batch | 4 |
| Gradient accumulation | 2 |
| Optimizer-step target | 128 |
| Checkpoint cadence | 64 optimizer steps |
| Evaluation fraction | 0.125 |
| Split seed | 424242 |
| Training seed | The slot's scheduled seed |
| Data-order seed | `1000000 + training_seed` |
| Packing | false |
| Gradient checkpointing | true |
| Optimizer | AdamW Torch |
| Scheduler | linear |
| Optimizer warmup | 0 steps |
| Weight decay | 0.0 |
| Maximum gradient norm | 1.0 |

Full uses the frozen method prior learning rate `0.00002`. Adapter methods use
`0.0002`. All adapter methods use rank 16, alpha 32, and the exact ordered target
tuple `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`. Phase 4 must provider-inspect and rebind that tuple to the exact
135M candidate; a mismatch blocks activation. Full has no adapter tuple. These
method-specific parameter scopes and learning rates are disclosed confounds.
Resource measurements under them are not a quality comparison.

The Phase 5 and 6 safety watchdog deadline is 30 minutes per started attempt.
It is an emergency ceiling, not the training-duration controller. The reviewed
optimizer-step contract must end normal training at exactly 128 completed,
non-skipped updates.

## Attempt schedules

Every slot materializes three independent seed fields:
`split_seed=424242`, `training_seed=scheduled_seed`, and
`data_order_seed=1000000+scheduled_seed`. The runtime must use separate random
number generator objects or streams for training behavior and data order. Each
field independently mutates execution identity. Methods paired within a block
therefore receive the same explicit training seed and the same explicit data
order.

### Phase 5 repeatability anchor

After its conditioning attempt, LoRA has exactly five measured slots:

| Ordinal | Training seed |
| ---: | ---: |
| 1 | 101 |
| 2 | 211 |
| 3 | 307 |
| 4 | 401 |
| 5 | 503 |

There are no additional Phase 5 warm-up slots and no replacement slots.
The August 6 acceptance is historical evidence and is not one of these slots.

### Phase 6 exploratory blocks

All four methods receive three predeclared exploratory slots. Admission may
leave a slot `planned-not-started`, but it does not remove it from the ledger.

| Block | Paired training seed | Master order |
| --- | ---: | --- |
| E1 | 601 | `full`, `lora`, `int8-lora`, `qlora` |
| E2 | 701 | `qlora`, `int8-lora`, `lora`, `full` |
| E3 | 809 | `lora`, `full`, `qlora`, `int8-lora` |

### Phase 6 confirmatory blocks

Only methods that satisfy the frozen promotion rule enter confirmatory
execution. Filter each master order to those promoted methods without
rerandomizing it.

| Block | Paired training seed | Master order |
| --- | ---: | --- |
| C1 | 1009 | `full`, `lora`, `int8-lora`, `qlora` |
| C2 | 2003 | `lora`, `qlora`, `full`, `int8-lora` |
| C3 | 3001 | `int8-lora`, `full`, `qlora`, `lora` |
| C4 | 4001 | `qlora`, `int8-lora`, `lora`, `full` |
| C5 | 5003 | `int8-lora`, `qlora`, `lora`, `full` |

Exploratory results remain public but never enter confirmatory aggregates.
Phase 5 LoRA runs also remain distinct from the Phase 6 confirmatory cohort.

### Phase 7 scale and breadth

Run the 135M, 360M, and 1.7B cells in ascending order. At each size, filter the
master method order `lora`, `full`, `int8-lora`, `qlora` to methods with valid
exact candidates without rerandomizing it. LoRA must be admitted before that
size proceeds, and its three slots run first. Then run all three slots for each
remaining admitted method in master order. Every admitted cell receives exactly
three exploratory slots with paired seeds `6101`, `6203`, and `6301`.

The deadlines are 30 minutes for 135M, 45 minutes for 360M, and 90 minutes for
1.7B. A later reviewed larger or breadth artifact has a 120-minute ceiling.
These remain safety ceilings, never post-hoc duration targets.

LoRA must be native `passed` and `protocol-valid` in all three slots and satisfy
the common three-run stability thresholds before the next model size activates.
Every other method must meet that same rule to retain its cells at larger sizes.
A refusal, failure, capture invalidity, safety or integrity event, or unstable
batch blocks the affected progression without replacement. Architecture breadth
begins only after the same-family ladder is reviewed and uses three exploratory
slots per admitted cell. Apply the same filtered master method order and finish
all three seeds for a method before advancing to the next method. No Phase 7
batch is called confirmatory repeatability.

### Phase 8 guarded frontier

Choose the largest stable Phase 7 cell whose every qualifying input run meets
the Phase 9 headroom rule. Ties prefer `lora`, then `qlora`, `int8-lora`, and
`full`. Freeze that cell identity before the first frontier point. Create a
separate cohort for each axis and recompile every point.

| Axis | Frozen increasing ladder | Controls held |
| --- | --- | --- |
| Sequence length | 128, 256, 512, 1,024, 2,048 | Selected stable-cell effective batch, micro-batch, and accumulation |
| Effective batch | 1, 2, 4, 8, 16, 32, 64 | Selected stable-cell sequence length; freeze every point's reviewed micro-batch/accumulation realization before the axis starts |
| Micro-batch at effective batch 16 | `(1,16)`, `(2,8)`, `(4,4)`, `(8,2)`, `(16,1)` as micro-batch/accumulation | Effective batch 16 and selected stable-cell sequence length |

Every comparable pilot uses `training_seed=8009` and
`data_order_seed=1008009`. Each point gets at most one bounded pilot slot and a
30-minute emergency-watchdog ceiling. Stop an axis at the first admission
refusal or any bounded-pilot nonpass, including a capacity failure. Never start
confirmed full training to seek an OOM. If every frozen point through 2,048
passes, report the upper
endpoint as right-censored rather than claiming an unobserved failure boundary.
Extending the sequence ladder requires a separately frozen longer fixture and
reviewed protocol amendment before any result is observed.

### Phase 9 endurance

Select a point below the frontier only when every qualifying input run has:

- maximum GPU temperature no greater than 78°C;
- minimum free VRAM at least 3 GiB;
- minimum available RAM at least 12 GiB;
- minimum free disk at least 48 GiB;
- swap activity below its warning threshold; and
- no safety warning event, thermal throttle, Xid event, foreign compute process,
  or qualifying telemetry gap.

Selection has no discretionary tie-break. For each Phase 8 axis, start with the
highest passing rung strictly below that axis's endpoint: the predecessor of the
first non-passing rung, or the predecessor of the right-censored top rung when
all rungs pass. Walk downward on that axis until a point satisfies every
headroom rule above. Rank the eligible axis candidates by greatest
`sequence_length * effective_batch_size`, then by greater micro-batch, then by
fixed axis order: sequence length, effective batch, micro-batch/accumulation.
Freeze the selected execution configuration before Phase 9 begins. If no point
is eligible, all Phase 9 slots are `planned-not-started`.

The frozen endurance seeds are `9101`, `9203`, and `9301`, with no replacement.
If Phase 3 implements a reviewed endurance controller, freeze before the first
slot either an exact graceful full-run deadline from 30 through 60 minutes or an
exact optimizer-step target of at least 300. The Phase 2 safety watchdog remains
an emergency stop, not that duration controller. Without one of those reviewed
Phase 3 contracts, Phase 9 uses the frozen epochs and dataset, publishes only
observed elapsed time and aggregate rates, and cannot claim an enforced
duration or step-time drift. Every attempt retains a 90-minute emergency
ceiling.

The endurance batch passes only if all three slots are native `passed` and
`protocol-valid`, with no safety warning or stop, no capture, artifact, or
integrity defect, complete required copy verification, and successful current
off-host retrieval. A non-passing, invalid, or unstarted slot is not replaced.
The Phase 5 and 6 cross-run duration stability ratios do not apply to endurance.
Any step-time drift remains a descriptive observation and requires the reviewed
Phase 3 progress contract.

## Promotion and batch decisions

The Phase 5 repeatability anchor passes only when all five slots:

1. are native `passed` and evidence `protocol-valid`;
2. complete exactly 128 non-skipped optimizer steps;
3. satisfy every capture, seal, copy-verification, off-host retrieval, artifact,
   completion, and method-integrity contract, with no safety warning or stop;
4. have at least 99% core-valid telemetry coverage and no gap above 2.5
   seconds;
5. have externally observed monotonic training-segment-duration
   `MAD / median <= 0.10` across the required slots;
6. have externally observed monotonic training-segment-duration
   `maximum / minimum <= 1.20` across the required slots; and
7. have a peak device-memory range no greater than the larger of 128 MiB or
   10% of the median.

An exploratory Phase 6 method promotes only when all three slots satisfy the
same conditions. A batch of passing but unstable runs is published as unstable
and does not promote. Refusal, capture invalidity, a safety warning or stop,
method-integrity discrepancy, or unknown ownership prevents promotion.

The duration ratios apply only to the externally observed `training` segment
under the exact 128-step contract. Trainer-reported runtime and steady-state
rate are not substitutes. Publish other valid metrics, but do not silently make
them promotion variables.

A confirmatory pair requires five globally complete blocks in which every
promoted method is native `passed` and `protocol-valid`; do not construct a
post-hoc complete subset for two methods. Fewer than five complete pairs require
an abstention from the confirmatory pairwise conclusion; individual values and
denominators remain published.

## Idle admission and cooldown

Phase 4 captures and binds a 10-minute idle baseline as 600 consecutive
one-hertz samples. It binds the GPU-temperature median and type-7 p95, free-VRAM
median, supported-power type-7 p95, utilization, process inventory, and thermal,
Xid, reset, lost-device, and hardware-error state. Before every started slot,
the host must satisfy admission and then meet all of these conditions over 120
consecutive one-hertz samples:

- GPU temperature is no greater than 50°C, no greater than the idle median plus
  5°C, and no greater than the idle p95 plus 3°C;
- the GPU-temperature slope is strictly less than 0.1°C per minute, calculated
  by ordinary least squares over all 120 valid
  `(monotonic_seconds, gpu_temperature_c)` samples and compared without prior
  rounding;
- GPU utilization is exactly 0% in at least 110 of the 120 samples;
- free VRAM remains within 128 MiB of its idle median and on the safe side of
  its warning threshold;
- when power is supported, GPU power is no greater than the idle p95 plus 10 W;
- no unrelated CUDA compute process, residual Aptus host-global lease, thermal
  throttle, Xid, reset, lost-device, or applicable hardware-error signal exists;
- available RAM, free disk, and swap activity stay on the safe side of their
  warning thresholds; and
- every required telemetry and watchdog heartbeat channel is healthy.

Zero variance in the cooldown timestamps or any invalid required sample fails
the cooldown window.

Cooldown is measured and retained but excluded from performance durations. The
harness has at most 30 minutes to obtain a complete qualifying 120-sample
window. If it cannot, the next slot remains `planned-not-started` and the
campaign stops for diagnosis. Expiration never permits a slot to start.

## Frozen host safety thresholds

The Phase 1 host record binds these reported RTX 3050 references:

| Fact | Frozen value |
| --- | ---: |
| Driver target temperature | 83°C |
| Maximum operating temperature | 92°C |
| Slowdown temperature | 94°C |
| Shutdown temperature | 97°C |
| Power limit | 130 W |

The campaign never changes clocks or the power limit. Its conservative stops
are lower than the device limits:

| Channel | Warning or guard | Abort |
| --- | --- | --- |
| GPU temperature | 78°C sustained 30 seconds | 84°C sustained 5 seconds; 89°C on one sample is emergency abort |
| Free VRAM | Below 2.5 GiB for 10 seconds | Below 2 GiB for 5 seconds; CUDA OOM is immediate |
| Available RAM | Below 12 GiB for 30 seconds | Below 8 GiB for 5 seconds |
| Free disk | Below 48 GiB for 30 seconds | Below 32 GiB, or insufficient frozen remaining budget |
| Swap activity | At least 16 MiB/s over a rolling 10 seconds | At least 64 MiB/s over 10 seconds, or at least 16 MiB/s continuously for 60 seconds |
| Telemetry | Gap above 3 seconds or heartbeat stale 3 seconds | Gap above 5 seconds, collector death, or heartbeat not recovered by 5 seconds |

Any software or hardware thermal slowdown signal, NVIDIA Xid, driver reset,
lost device, applicable uncorrected hardware error, unrelated CUDA compute
process, non-finite loss or state, invalid trainable parameter census, invalid
split, invalid checkpoint/export/completion evidence, or uncertain job
ownership stops immediately.

Admission requires:

- the normal 2 GiB CUDA planning reserve, never reduced for the campaign;
- available RAM of at least the candidate's required host RAM plus 8 GiB; and
- free disk of at least the 32 GiB hard floor plus every remaining frozen
  download, output, and vault budget.

Materialize those disk budgets in bytes before activation:

```text
download = ceil(1.25 * exact_artifact_bytes) + 2 GiB
output   = ceil(1.25 * max(plan_required_disk_bytes,
                           4 * largest_pilot_checkpoint_bytes + final_export_bytes))
vault    = ceil(1.10 * expected_copied_outputs_logs_telemetry_bytes)
```

Every expected copied output, log, and telemetry byte is included in the vault
input. A missing budget input blocks the slot; it is never treated as zero.

If the slowdown limit is unavailable during the initial Phase 4 freeze but GPU
temperature remains readable, the only permitted fallback is 75°C for 30
seconds, 82°C for 5 seconds, and 85°C for one sample. If a limit or sensor that
was supported at freeze later becomes unreadable, the attempt is invalid; the
fallback cannot silently replace it. CPU and NVMe temperature channels may be
declared unavailable during Phase 4. A declared-required or previously
supported channel becoming unreadable invalidates the attempt.

## Watchdog and cancellation semantics

A pre-action violation is `guard-blocked`. A safety trigger during an active
managed job requests owned cancellation and produces native `cancelled`. A
frozen deadline produces native `timed-out`. Capture failure independently
produces evidence `capture-invalid`. Uncertain ownership, termination, or lease
reconciliation produces native `unknown` and blocks every later submission.

The watchdog operates through the owning `JobService` and exact job ID. It
never signals a stale numeric PID directly. The frozen service levels are:

| Transition | Maximum elapsed time |
| --- | ---: |
| Trigger detection to persisted cancellation request | 2 seconds |
| Cancellation request to confirmed process-group termination | 10 seconds |
| Termination to global-lease reconciliation | 2 seconds |
| Any unreconciled cancellation sequence | 15 seconds before `unknown` and campaign block |

Every transition timestamp enters the event ledger. A safety stop seals the
incomplete attempt or its capture-failure receipt before diagnosis.

## Measurement contract

### Time and event ledger

All durations derive from monotonic nanoseconds. The ledger also records a UTC
mapping for review. Its JSON Lines sequence is contiguous and covers harness,
telemetry, command, managed-job transition, pilot phase, training, export,
parent verification, sealing, retrieval, and cooldown boundaries. Events state
whether their timestamp was emitted by the runtime, observed by the harness, or
derived. `seal.started` is the final normal ledger event; the seal records its
own `sealed_at` fact without rewriting the ledger.

Keep these time channels separate:

- each of dependency, model-data, preflight, pilot phase 1, pilot phase 2,
  training, export, and parent verification;
- trainer-reported runtime and externally observed child-process runtime;
- managed-job queued-to-terminal duration;
- end-to-end five-action duration, excluding setup and cooldown; and
- separately captured installation and model-download setup durations.

### Telemetry

Core telemetry is scheduled at 1 Hz. Qualifying evidence requires at least 99%
core-valid scheduled-slot coverage and a maximum core gap of 2.5 seconds. The
warning and abort thresholds above remain additional safety controls; a stream
can therefore be capture-invalid before its safety abort threshold is reached.

Core GPU channels are protected-only GPU UUID; directly reported memory used,
free, reserved, and total with source units retained and values normalized to integer
bytes; utilization; temperature; power draw and limit; graphics and memory
clocks; performance state; throttle reasons; Xid projection; and unrelated
compute processes. Safety binds the directly reported free-memory channel,
never a derived value. Convert each reported used, free, reserved, and total
value from its retained source unit to integer bytes exactly. Reconcile
`total_bytes` against `used_bytes + free_bytes + reserved_bytes`, allowing only
the sum of one-half of each retained source value's display resolution to account
for independent `nvidia-smi` rounding. An inexact or invalid conversion or a
reconciliation mismatch outside that source-resolution bound is
`capture-invalid` and blocks qualification pending diagnosis. Core host
channels are `MemAvailable`, swap used and I/O, load, filesystem free bytes,
managed-process RSS/CPU/I/O counters, disk growth, collector health, and
watchdog heartbeat.

Unsupported optional fields remain explicit rather than disappearing. Preserve
runtime allocated and reserved CUDA peaks separately from device-level memory.
For each supported scalar channel publish sample count, coverage, maximum, and
median. Publish type-7 p95 for used or increasing resource channels and type-7
p05 plus minimum for free-resource channels.

For ordered observations `x[0] ... x[n-1]`, type-7 quantiles use
`h=(n-1)*p`, the observations at `floor(h)` and `ceil(h)`, and linear
interpolation by the fractional part of `h`. No other interpolation may be
chosen after results exist.

For a marked segment, expected slots equal
`floor((stop_monotonic_ns - start_monotonic_ns) / 1000000000) + 1`; slot `k` is
scheduled at `start_monotonic_ns + k * 1000000000`. Coverage is at most one
core-valid sample per slot divided by that expected count. There is no catch-up
sampling, and boundary gaps count. Report missing slots and the largest
monotonic gap. No missing telemetry is imputed.

Estimated GPU energy uses trapezoidal integration only between adjacent power
samples separated by no more than two scheduled intervals. Publish the covered
duration, scheduled-window duration, scheduled-slot coverage, and separately
labeled clipped time-support coverage with the estimate. Do not publish an
energy estimate when core coverage is below 99% or the qualifying gap exceeds
2.5 seconds. Never label it host or whole-machine energy.

### Training counters

The Phase 3 counter schema separates training and evaluation and defines:

- padded input elements presented to the model;
- non-padding tokens where the attention mask is active;
- supervised tokens where the label is not `-100`;
- micro-iterations and completed non-skipped optimizer steps; and
- examples consumed, including any repeated epoch traversal.

Micro-iterations, completed non-skipped optimizer steps, and examples consumed
are required before Phase 5. Exact padded, non-padding, and supervised token
counters may be deferred only when every token-throughput field and claim stays
absent. Token throughput is forbidden until those exact counters and their
marked monotonic duration are implemented and bound. Steady-state rates use
steps 9 through 128. Full end-to-end and resource metrics still include every
step, model load, evaluation, export, and verification segment as labeled.

Loss, evaluation loss, and structural export are diagnostic runtime evidence.
They are not task-quality measurements. CUDA structural adapter verification
also does not establish semantic reload or generation parity.

## Aggregation and uncertainty

Publish every individual slot before aggregates. No observation is deleted as
an outlier. A predeclared diagnostic outlier flag may be shown, but flagged
values remain in every applicable calculation and denominator.

For a comparison cell publish:

- planned, started, and planned-not-started counts;
- every native-outcome and evidence-status count;
- median, minimum, maximum, and median absolute deviation for each scalar
  metric; and
- the complete individual values and execution-configuration IDs.

Median absolute deviation is `median(abs(x - median(x)))`. With only three or
five attempts, do not publish a population confidence level or significance
claim. Telemetry p95 is a within-run sample quantile, not uncertainty across
runs.

Pairwise method summaries use only the five predeclared complete blocks and
publish each paired difference and ratio, their median, minimum, maximum, and
MAD; publish a ratio only for a positive finite denominator. A bounded label of
**consistent exact-host difference** additionally
requires all five pairs, at least four paired directions agreeing, and a median
ratio outside `[0.95, 1.05]`. The label is descriptive, not statistical
significance or a universal ranking. Otherwise the pairwise conclusion is
inconclusive under this protocol.

Ledger validation must prove:

```text
Planned = Started + Planned-not-started
Started = sum(native outcomes)
Started = Protocol-valid + Capture-invalid
```

Only native `passed` intersected with `protocol-valid` enters a passing
aggregate. Missing blocks remain visible and never trigger imputation or a
replacement run.

## Raw sealing, copies, retrieval, and retention

The raw vault is outside Git. POSIX vault directories use mode `0700`; files
use mode `0600`. Maintain two checksum-verified copies in independent failure
domains, including one off the Ubuntu host, with encryption in transit and at
rest, a key custodian, and a recovery procedure.

A canonical raw manifest inventories every raw file by relative path, media
type, byte size, and SHA-256. It excludes itself and `SEALED.json`. Sealing
writes the canonical manifest to a fresh no-clobber directory, flushes content,
and atomically writes `SEALED.json` binding the raw-manifest SHA-256, manifest
byte size, protected artifact ID, and sealing time. Later content mutation must
fail verification; the same experiment-run ID is never resealed.

Copy verification, retrieval, retention, renewal, claim suspension,
restoration, and withdrawal are typed append-only receipts. They reference the
immutable raw-manifest digest and never modify a sealed run.

Retention policy `cuda-v02-public-claim-evidence-24m-v1` sets the provisional
deadline to seal or recovery time plus 24 calendar months. Publication replaces
it with an effective receipt whose deadline is packet merge time plus 24
calendar months. Add calendar months in UTC, clamping the day only when the
target month is shorter. Deletion is permitted only after every applicable
deadline has passed **and** every dependent claim has been withdrawn or
superseded.

Verify both copies initially, before publication, every 90 days, after any
storage, key, or custodian change, and no later than 90 days before removal
eligibility. Perform a full off-host retrieval into a fresh destination
initially, before publication, every 180 days, after those changes, and no later
than 90 days before removal eligibility. A lost copy or failed retrieval
suspends the claim until redundancy and retrieval are restored and independently
reviewed. If consent, license, security, or another controlling requirement
forces earlier deletion, publish claim withdrawal first whenever permitted.

## Constructive sanitization and recovery supplement

The public packet is constructed from an explicit field allowlist; it is not a
redacted raw copy. Every public field has a JSON-Pointer trace entry to its
sealed raw source or a documented deterministic derivation. Deny-pattern scans
are supplemental defense, not the primary sanitizer. Public stable reasons are
at most 240 Unicode code points. Byte-exact exceptions, usernames, hostnames,
network identifiers, GPU UUIDs, home paths, tokens, raw logs, job state, model weights,
checkpoints, and adapters remain protected.

The August 6 recovery supplement is bound to the
[`aptus.raw-artifact-digests.v1` expected-digest manifest](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/raw-artifact-digests.json),
whose SHA-256 is
`db6c4845846dcc1bdd2cdb54992210d31b4eba489a514197f33a127ccb37da7a`.
It represents 40 logical expected rows covering 39 unique expected digests.
Duplicate logical rows may legitimately reference the same protected artifact.
Each row has exactly one disposition:

- `recovered-matching`;
- `recovered-mismatched`; or
- `not-found`.

The bounded recovery search also looks for the original byte-exact Python test
transcript. If it never existed as a file, record `not-found` with the bounded
search-scope reason codes. Never synthesize it from the historical 550-test
summary.

The supplement may claim only recovery integrity: what was found, its digest
comparison, copy verification, retrieval, protected opaque locator, and
retention binding. It cannot retroactively establish timing, telemetry,
performance, repeatability, the missing original test transcript, a test gate,
or release readiness. The original August 6 packet remains immutable.

## Implementation gap and exit gates

Phase 1 freezes requirements; it does not assert these implementation gaps are
closed.

| Contract | Phase responsible | Exit condition |
| --- | --- | --- |
| Raw capture, event ledger, telemetry, watchdog, sealing, copies, retrieval, receipts, sanitizer | Phase 2 | Fake-command and failure-path tests pass; recovery supplement merges after independent traceability and privacy review |
| Explicit complete-candidate selection and new plan identity | Phase 3 before Phase 4 | Stale, rejected, nonselectable, and mutated candidates fail closed across API/domain/CLI/compiler/runtime |
| Separate split, training, and data-order seeds | Phase 3 before Phase 5 | Full-run paired seeds preserve the exact frozen split assignment |
| Exact 128-step normal-training controller | Phase 3 before Phase 5 | Plan, candidate, bundle, execution identity, runtime completion, checkpoint state, and sanitizer bind exactly 128 completed non-skipped updates; epochs-only substitution is forbidden through the Phase 7 common stability contract |
| Training progress and consumption counters | Phase 3 before Phase 5 | Training and evaluation remain separate; micro-iterations, completed non-skipped optimizer steps, and examples consumed bind to the plan, runtime metrics, and sanitizer |
| Micro-batch/accumulation controls | Phase 3 before Phase 8 | Plan, candidate, compiler, validator, and runtime bind explicit values and effective-batch arithmetic without generated-file edits |
| Graceful deadline or endurance update target | Phase 3 only before an enforced Phase 9 duration or update-count claim | Reviewed controller is bound into planning and identity, or Phase 9 uses frozen epochs/data and reports observed elapsed time only |
| Exact token counters | Phase 3 before any token-rate claim; may otherwise be deferred | Mutation, counter, and resume-boundary tests pass, or every token-rate field and claim stays absent |
| Per-step progress timestamps | Phase 3 before any Phase 9 drift claim | Fixed-window and resume-boundary tests pass, or Phase 9 reports only aggregate observed rates without drift claims |
| Semantic CUDA export reload or generation | Outside this campaign's implemented scope | Remains an explicit open release gate |

Before Phase 4, run the complete repository gates through the reviewed capture
harness, retain their byte-exact transcript bindings, reproduce the fixture and
split digests, recover and verify the exact tokenizer artifacts, reproduce and
seal the canonical per-row token-count manifest, verify both raw copies and a
fresh off-host retrieval, and freeze one clean source commit, tree, host profile,
environment closure, model-file inventory, complete slot ledger, and every
remaining budget. Any protocol, schema, source, environment, fixture, tokenizer,
sanitizer, or stop-rule change after that freeze creates a new protocol version
and invalidates the rehearsal.

## Related documentation

- [RTX 3050 CUDA empirical evidence campaign](../operations/cuda-empirical-campaign.md)
- [Machine-readable CUDA campaign protocol](cuda-campaign-protocol.v1.json)
- [Release gates](../operations/release-gates.md)
- [Release evidence template](../operations/release-evidence-template.md)
- [Operator checklist](../operations/operator-checklist.md)
- [State, storage, and retention](../operations/state-storage-retention.md)
- [Preflight and calibration](../methodology/preflight-calibration.md)
- [Method registry](method-registry.md)
- [Configuration defaults](configuration-defaults.md)
- [Model-policy snapshot](model-policy-snapshot.md)
- [Design an evaluation](../guides/design-an-evaluation.md)
