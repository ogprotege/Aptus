# Glossary

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | All Aptus users and contributors |
| Authority | Canonical terminology for current v0.2 documentation |
| Last reviewed | 2026-07-22 |
| Next review | 2026-10-22, or sooner when public vocabulary changes |

## Analytic point estimate

The sum of named planner memory components before the separate uncertainty
envelope. It is a heuristic calculation, not a measurement.

## Apple platform profile

A measured Darwin arm64 record containing macOS version and build, chip name,
logical CPU count, unified-memory capacity and headroom, memory pressure, swap,
Metal working-set guidance, optional Metal GPU core count, and separate MLX,
MLX-LM, and PyTorch MPS capabilities. It uses no chip allowlist.

## Artifact fingerprint

The SHA-256 of `bundle-manifest.json` when the manifest exists. Validation uses
it to bind a report to one compiler-managed bundle state.

## Assignment digest

The SHA-256 evidence over each canonical row offset, split-unit digest, and
train or evaluation assignment. It binds the exact full-run split without
publishing declared group names.

## Candidate

One method and placement row with bound precision, quantization, batch,
resources, status, assumptions, and evidence. V0.2 enumerates exactly 12.

## Candidate status

One of `feasible`, `conditional`, `infeasible`, or `unsupported`. The legacy
`feasible` boolean is true for both feasible and conditional rows, so consumers
must read the status and reasons.

## Canonical row

A valid, non-empty source object serialized with sorted JSON keys to
`data/training.jsonl`. Compilation preserves supported metadata and schema.
Tokenizer-specific transformation happens later.

## Completion attestation

The managed parent's record that it verified one full run's marker, metrics,
rank evidence, dataset split, trainable census, and structural export before
marking the job completed.

## Compute backend

The accelerator family bound to a candidate, such as `cuda` or `mps`. It is
separate from the training runtime. For example, MLX-LM and PyTorch MPS both use
`mps`, but only MLX-LM has a current Aptus compiler.

## Conditional candidate

A planner row that can proceed only with unresolved runtime evidence. A common
case is a point estimate that fits while the heuristic upper envelope does not,
or LoRA FSDP whose transient behavior requires a pilot.

## DDP

Distributed Data Parallel. Each rank holds a model replica and processes its
part of the exact global batch. Fit uses the least usable participating GPU.

## Direct pins

The exact top-level versions written to `requirements.txt`. They constrain
selected packages but do not enumerate the resolved transitive environment.

## Environment binding

A SHA-256 over Python version, platform, direct package versions, and the
reachable installed distribution closure. Runtime reports use it to detect
environment drift.

## Evidence record

A versioned claim, source, scope, and confidence object cited by candidates and
embedded in the plan. Research evidence defines or motivates a method. It does
not prove Aptus execution support or task quality.

## FSDP

Fully Sharded Data Parallel. V0.2 permits only conditional LoRA FSDP planning.
It rejects full and quantized FSDP paths.

## Gated executable

A method lifecycle that can become selectable only with a compiler ID, export
contract, explicit backends, explicit distributions, and required runtime
proof. It does not mean every model and host combination is ready.

## Host-global Aptus lease

A per-user local coordination record that permits one Aptus accelerator action across
managed state roots and POSIX portable bundle launches. It does not coordinate
unrelated programs.

## Immutable revision

A 40 to 64 character hexadecimal provider commit identifier. Mutable branches
and tags are outside the model planning contract.

## Local scan

A planning mode that discards submitted manual capacity facts, re-probes the API
host, and retains the submitted per-device reserve. It is blocked while Aptus
owns an active job.

## Manifested file

A compiler-created file listed in `bundle-manifest.json` with exact relative
path, byte size, and SHA-256. Changing it invalidates the bundle.

## Measured preflight

A runtime-specific measured gate before pilot. CUDA runs a selected-method
synthetic optimizer step and binds its trainable census. MLX-LM runs a bounded
real model and data adapter smoke, writes an adapter, and records a positive
runtime-neutral memory peak. Neither result is pilot evidence.

## Measured run pass

The report state reached only after parent verification of one full run and its
structural export. It does not state that the trained model is useful, safe, or
better than a baseline.

## Method descriptor

A typed `aptus.method-descriptor.v1` record containing research identity,
lifecycle, selectability, parameter scope, compiler and export contracts,
support declarations, evidence, required pilot, aliases, and blocker.

## Model-data validation

The runtime gate that loads the pinned model and tokenizer and transforms every
canonical row. CUDA also verifies structural facts, prepares the selected
method, and enforces trainable scope. MLX-LM validates the QLoRA metadata when
applicable and tokenizes the bound train and validation rows. It performs no
optimizer step.

## No-clobber

A publication rule that refuses an existing archive, non-empty bundle, or run
output instead of replacing it. Some standalone JSON output options are not
no-clobber and can replace an existing file.

## Parent promotion

The completion transaction in which managed `JobService` or portable `run.py`
verifies pending full-run evidence before changing the report to
`measured-run-pass`.

## Pilot

The runtime-specific exact-model gate after measured preflight. The CUDA pilot
runs real data in two fresh processes, saves step one, and restores complete
state to reach at least step two. The MLX-LM pilot instead runs from the pinned
base without interruption for at least two optimizer updates, then reloads the
emitted adapter in a fresh process and generates one to four tokens. The MLX
reload proves adapter inference, not training continuation.

## Planner parity

The host static check that reconstructs planning from bound facts and requires
the resulting candidates, recommendation, and plan ID to match the bundle.

## Portable bundle

A compiler-produced directory with relative data paths, direct pins,
configuration, self-contained validators, training code, and an immutable
manifest. It can move to its runtime-compatible target host as a reviewed unit.

## Provenance

The origin of a fact: `measured`, `provider-declared`, `user-attested`,
`inferred`, or `unknown`. Provenance is distinct from a research evidence
record.

## Recommendation

The highest-ranked feasible or conditional candidate inside the fixed v0.2
matrix. It is not a universal optimum or a model-quality prediction.

## Runtime action

One managed operation: `dependency`, `model-data`, `preflight`, `pilot`, or
`train`. Actions have ordered report-state prerequisites.

## Runtime configuration

The private `aptus.runtime-config.v1` mapping from a training-runtime ID to one
probed canonical Python executable. Aptus persists it with mode 0600. A
Finder-launched Mac app uses this record instead of relying on shell variables.

## Runtime contract

An `aptus.runtime-contract.v1` record binding one candidate to its compute
backend, training runtime, compiler, estimator, evidence requirement, and export
kind. Runtime presence alone does not create a compiler contract.

## Runtime exclusion

An unmanifested mutable path explicitly permitted by the bundle validator:
the report lock, validation report, preflight metrics, `pilot-output/`, or
`runs/`. Any other unmanifested file invalidates the bundle.

## Split group

An optional non-empty string at top level or `metadata.split_group`. All rows
with the same value remain in one CUDA full-run partition. The current MLX
compiler uses a separate precompiled split and does not claim group-aware
assignment. Aptus does not decide whether the declared value is the correct
leakage unit.

## Split unit

One atomic item assigned by the full-run splitter. A declared group is one unit,
regardless of row count. Each ungrouped row is its own unit.

## Structural export

A method-specific config plus safetensors files whose tensor keys, optional
index, provenance, paths, sizes, and hashes pass verification. It does not prove
that inference produces correct behavior.

## Train admission

The runtime-specific atomic transaction before a queued train record is written.
CUDA rechecks the current bundle, pilot, environment, hardware, VRAM, host RAM,
disk, checkpoint, and export evidence. MLX re-verifies its owned uninterrupted
pilot, then checks current unified-memory headroom and disk.

## Trainable-parameter census

An `aptus.trainable-parameter-census.v1` object containing trainable and frozen
tensor and parameter counts, finite-value status, adapter target-pair counters,
unexpected-trainable count, and a SHA-256 over sorted trainable names, shapes,
and dtypes. MLX uses `aptus.mlx-trainable-target-binding.v1` instead. It binds
one LoRA A/B pair to every planned target instance and rejects other trainables.

## Training authorization

The current result of train admission. A historical `pilot-pass` or cached UI
message is not authorization for a new run.

## Training runtime

The concrete software path that compiles and executes a candidate.
`transformers-peft-cuda` and `mlx-lm` have current compiler bindings.
`pytorch-mps` is known and discoverable but lacks a compiler.

## Unified memory

The physical memory pool shared by CPU and GPU on Apple Silicon. Aptus records
it separately from dedicated VRAM and retains free VRAM as unknown. The MLX-LM
estimator uses current free host RAM as the live headroom cap when available.

## Upper envelope

The sum of named component upper bounds, including a separate uncertainty term.
It remains heuristic and uncalibrated until compared with target-host evidence.

## Validation level

Requested validation work such as `model-data` or `pilot`. Higher levels execute
all lower portable checks cumulatively.

## Validation state

The strongest evidence recorded in `validation-report.json`, such as
`static-pass`, `pilot-pass`, or `measured-run-pass`.

## Viable candidate

A candidate whose status is `feasible` or `conditional`. Ranking ignores
infeasible and unsupported rows.

## Weight snapshot

A periodic MLX adapter-weight save. It does not contain the complete optimizer,
scheduler, random, and data-position state required to resume training. Aptus
rejects every MLX resume argument and never calls these files checkpoints.

## Related documentation

- [Plan schema](plan-schema.md)
- [Method registry](method-registry.md)
- [Evidence records](evidence-records.md)
- [Validation states](validation-states.md)
- [Run states](run-states.md)
- [Bundle manifest](bundle-manifest.md)
