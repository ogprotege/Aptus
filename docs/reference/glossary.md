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

A per-user local coordination record that permits one Aptus GPU action across
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

A selected-method synthetic CUDA optimizer step. It binds a trainable census
and positive peak without loading the exact planned model or training corpus.

## Measured run pass

The report state reached only after parent verification of one full run and its
structural export. It does not state that the trained model is useful, safe, or
better than a baseline.

## Method descriptor

A typed `aptus.method-descriptor.v1` record containing research identity,
lifecycle, selectability, parameter scope, compiler and export contracts,
support declarations, evidence, required pilot, aliases, and blocker.

## Model-data validation

The runtime gate that loads the pinned model and tokenizer, verifies structural
facts, prepares the selected method, enforces trainable scope, and transforms
every canonical row. It performs no optimizer step.

## No-clobber

A publication rule that refuses an existing archive, non-empty bundle, or run
output instead of replacing it. Some standalone JSON output options are not
no-clobber and can replace an existing file.

## Parent promotion

The completion transaction in which managed `JobService` or portable `run.py`
verifies pending full-run evidence before changing the report to
`measured-run-pass`.

## Pilot

A bounded exact-model and real-data pressure run in two fresh processes. Phase
one saves step one. Phase two resumes it and reaches at least step two.

## Planner parity

The host static check that reconstructs planning from bound facts and requires
the resulting candidates, recommendation, and plan ID to match the bundle.

## Portable bundle

A compiler-produced directory with relative data paths, direct pins,
configuration, self-contained validators, training code, and an immutable
manifest. It can move to the target CUDA host as a reviewed unit.

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

## Runtime exclusion

An unmanifested mutable path explicitly permitted by the bundle validator:
the report lock, validation report, preflight metrics, `pilot-output/`, or
`runs/`. Any other unmanifested file invalidates the bundle.

## Split group

An optional non-empty string at top level or `metadata.split_group`. All rows
with the same value remain in one full-run partition. Aptus does not decide
whether the declared value is the correct leakage unit.

## Split unit

One atomic item assigned by the full-run splitter. A declared group is one unit,
regardless of row count. Each ungrouped row is its own unit.

## Structural export

A method-specific config plus safetensors files whose tensor keys, optional
index, provenance, paths, sizes, and hashes pass verification. It does not prove
that inference produces correct behavior.

## Train admission

The atomic transaction that rechecks current bundle, pilot, environment,
hardware, VRAM, host RAM, disk, checkpoint, and export evidence before a queued
train record is written.

## Trainable-parameter census

An `aptus.trainable-parameter-census.v1` object containing trainable and frozen
tensor and parameter counts, finite-value status, adapter target-pair counters,
unexpected-trainable count, and a SHA-256 over sorted trainable names, shapes,
and dtypes.

## Training authorization

The current result of train admission. A historical `pilot-pass` or cached UI
message is not authorization for a new run.

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

## Related documentation

- [Plan schema](plan-schema.md)
- [Method registry](method-registry.md)
- [Evidence records](evidence-records.md)
- [Validation states](validation-states.md)
- [Run states](run-states.md)
- [Bundle manifest](bundle-manifest.md)
