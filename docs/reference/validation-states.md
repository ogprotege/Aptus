# Validation Levels and States

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Operators, validator maintainers, UI developers, and auditors |
| Authority | Normative evidence-ladder reference for Aptus v0.2 |
| Last reviewed | 2026-07-28 |
| Next review | 2026-10-22, or sooner when validation or generated runtime code changes |

A validation level is work requested from a validator. A validation state is
the strongest evidence recorded in `validation-report.json`. Runtime levels are
cumulative. Requesting `pilot` executes contract, static, dependency,
model-data, measured preflight, and pilot checks in that order.

## Public levels

```text
contract
static
dependency
model-data
measured-preflight
pilot
```

`execution-approved` and `measured-run-pass` are lifecycle states created by the
full-run transaction. They are not standalone validation levels.

## State ladder

| Rank | State | Evidence established | Does not establish |
| ---: | --- | --- | --- |
| 0 | `invalid` | One or more required checks failed | Any positive gate |
| 0 | `unsupported` | The requested contract is outside an accepted path | Runtime viability |
| 1 | `contract-pass` | Plan and bundle integrity passed at the producing validator's contract scope | Generated source execution or dependencies |
| 2 | `static-pass` | Generated Python parses and static bundle rules pass | Package installation, model load, or accelerator fit |
| 3 | `dependency-pass` | Exact direct package versions are installed and the environment closure is bound | Model or data compatibility |
| 4 | `model-data-pass` | Runtime-specific exact model, tokenizer, and compiled-data compatibility pass | An optimizer step or measured memory fit |
| 5 | `measured-preflight-pass` | Runtime-specific bounded optimizer work and memory evidence pass | Pilot-pass or full-run fit |
| 6 | `pilot-pass` | Runtime-specific exact-model pilot passes: CUDA checkpoint continuation, or uninterrupted MLX adapter training plus fresh-process adapter reload | Full-run completion, general crash resume, or task quality |
| 7 | `execution-approved` | A full run was admitted or is represented as pending against current evidence | Successful completion |
| 8 | `measured-run-pass` | Parent verified one completed run and structural export tree | Task quality, safety, or deployment fitness |

The `unsupported` value remains part of the report vocabulary. Current host
bundle validation normally reports either `invalid` or the achieved pass state.
Candidate support is a separate planner status.

## Validator identities

A report can be produced by:

- host validator `aptus-validator-v2`, used by the repository CLI and API;
- generated CUDA validator `aptus-portable-validator-v2`; or
- generated MLX validator `aptus-validator-mlx-v1`.

Inspect `validator_version`, `validation_level`, `checked_files`, `findings`, and
`runtime_evidence`. The host contract pass includes required-file checks,
content identities, deterministic replanning parity, manifest checks, selected
dependency-set parity, source-value separation, and trainer-plan parity. The
portable contract pass uses the self-contained plan and manifest contracts. The
portable static level then parses every generated Python program.

## Level details

### Contract

Core contract checks bind:

- `aptus.training-plan.v3`, every `aptus.runtime-contract.v1`, and the selected
  memory estimator identity;
- candidate and plan content IDs;
- normalized model, dataset, hardware, and target facts;
- the copied dataset digest when dataset verification is enabled;
- `aptus.bundle.v2`, `plan_sha256`, file paths, sizes, and hashes;
- absence of symlinks and unauthorized unmanifested files; and
- method-specific direct requirements and trainer configuration on the host.

Contract validation does not import the training stack or allocate accelerator
memory.

### Static

Static validation adds AST parsing of:

- `plan_contract.py`;
- `preflight.py`;
- `run.py`;
- `runtime_lease.py`;
- `train.py`; and
- `validate.py`.

The host validator also rejects unresolved `{{`, `}}`, and `TODO` markers in
generated code and operator documents. Static validation does not prove that
imports or pinned APIs work.

### Dependency

Dependency validation requires every exact direct pin in `requirements.txt` to
be installed at the named version. The resulting environment binding hashes:

- Python version;
- platform identity;
- direct constraint versions; and
- the installed transitive distribution closure reachable from those direct
  packages.

`requirements.txt` is still not a complete lock. The environment binding is the
runtime record of what was actually resolved.

### Model-data

For `transformers-peft-cuda`, model-data validation runs
`train.py --preflight-model-data`. It:

1. requires planned CUDA visibility and device parity;
2. loads the pinned model configuration, tokenizer, and weights;
3. verifies parameter count and structural facts;
4. disables model cache and enables gradient checkpointing;
5. performs k-bit preparation for quantized methods;
6. injects the compiled PEFT LoRA configuration for adapter methods;
7. verifies target modules and the exact method-scope trainable census; and
8. tokenizer-transforms every row in `data/training.jsonl`.

It does not construct the training optimizer, execute a batch, backpropagate a
loss, save a checkpoint, or claim measured fit. It can change the in-memory
model structure while preparing the method, but it performs no training update.

The model-data action enforces the census but does not persist a separate census
object in the validation report. Later measured levels persist it in metrics.

For `mlx-lm`, model-data validation first scans the pinned snapshot's safetensors
shards and compares live available unified memory against the packed-checkpoint-
adjusted candidate estimate. It adds any observed packed size in excess of the
planned resident bytes to both the point and upper estimates, and requires
available unified memory of at least `max(adjusted point, adjusted upper)` plus
`max(plan reserve, 8 GiB)`. If that requirement is not met it fails closed before
any weights load, reporting exact required, available, and shortfall byte counts.
A passing measurement records `aptus.mlx-unified-memory-admission.v2` inside the
immutable `model-data-evidence.json` artifact.

Only after admission passes does it load the exact pinned revision through MLX-LM
and tokenize every row in the compiler-produced `data/mlx/train.jsonl` and
`data/mlx/valid.jsonl` files. MLX QLoRA also requires explicit four-bit MLX
quantization metadata in the pinned model configuration. It does not use a CUDA
device capability flag or bitsandbytes.

### Measured preflight

CUDA measured preflight runs `train.py --synthetic-preflight`. It creates a
bounded synthetic model for the selected method path, initializes the required
quantization or adapter stack, performs one optimizer step, validates finite
loss and trainable scope, and records a positive CUDA peak.

`preflight-metrics.json` must use `aptus.preflight-metrics.v1` and bind:

- candidate ID;
- method;
- precision and quantization;
- distribution and world size;
- scope `synthetic-method-preflight-not-model-data-pilot`;
- positive measured CUDA peak; and
- a valid `aptus.trainable-parameter-census.v1` object.

This is selected-method and kernel evidence, not exact model-data training
evidence.

MLX measured preflight instead invokes the bounded MLX-LM compiler slice against
the exact pinned model and compiled MLX data. It permits at most eight
iterations, completes at least one optimizer update, writes an adapter pair, and
emits `aptus.runtime-metrics.v1` with:

- candidate, method, runtime, backend, and compiler bindings;
- scope `bounded-compiler-smoke-not-pilot-evidence`;
- positive `measured_peak_bytes` from MLX;
- active and cache memory measurements; and
- exact planned-target binding plus positive adapter delta; and
- a path, size, and SHA-256 adapter manifest.

The validator copies this evidence into `preflight-metrics.json` and may promote
`measured-preflight-pass`. The real inputs make this stronger than the CUDA
synthetic check in one respect, but it still proves no pilot result, restart
continuation, or full-run fit.

### Pilot

The CUDA pilot is the first bounded optimizer-step gate using the exact pinned
model and compiler-produced real-data pressure set. It launches two fresh
processes:

1. phase one trains one step and saves `checkpoint-1`;
2. phase two resumes that checkpoint and reaches at least step two.

The pilot requires:

- identical valid trainable censuses in both phases;
- finite losses and positive finite-guard counts;
- optimizer membership equal to the validated trainable set;
- one CUDA peak per rank and positive aggregate reserved peak;
- method-specific final exports for both phases;
- complete checkpoint manifests for steps one and two;
- phase two bound to the phase-one checkpoint; and
- `checkpoint_continuation_observed: true`.

CUDA pilot evaluation is disabled. Pilot rows are repeated as needed and padded
to the target sequence length to create a bounded pressure test.

The MLX-LM pilot uses different proof semantics. It starts from the pinned base
and runs the exact model and compiled train and validation files without
interruption. It requires:

- exactly the bounded two-update schedule and at least two completed optimizer
  updates;
- finite training and validation losses;
- one LoRA A/B pair for every planned target in every planned layer, with no
  other trainable tensor;
- positive MLX peak memory and positive adapter delta;
- a passing live unified-memory admission;
- a fresh action-owned output root, immutable marker, adapter pair, metrics,
  and artifact manifest; and
- a fresh child process that loads the pinned base plus emitted adapter,
  generates one to four tokens, and records a positive MLX peak.

The adapter reload proves inference from the saved adapter. It does not restore
the training optimizer, scheduler, random state, or data position. MLX periodic
saves are weight snapshots, not resumable checkpoints. A passing pilot can
promote `pilot-pass` and permit an explicitly confirmed full-duration adapter
run from the pinned base.

### Execution approved

Train admission always rechecks the manifest, plan, report, source, model
revision, and bound pilot artifacts. CUDA additionally rechecks the environment,
runtime hardware identity, checkpoint and export trees, free VRAM, host RAM,
and disk. MLX re-verifies the owned uninterrupted pilot, then requires current
available unified memory above the measured pilot peak plus reserve and enough
disk for the plan and measured adapter artifacts. A parent may record active or
pending run evidence as `execution-approved` until completion verification
succeeds.

The state is not durable permission to start any future run. Admission is
repeated for each submission.

### Measured run pass

The CUDA parent promotes to `measured-run-pass` only after it verifies:

- aggregate process success and a positive completed step count;
- plan, candidate, run, distribution, world-size, and rank bindings;
- finite raw, backward, gradient, and trainable-parameter checks;
- exact optimizer membership and method-scope census;
- one valid measured CUDA peak per rank;
- deterministic dataset-split schema, strategy, counts, and digests;
- full-run metrics and marker identity;
- method-specific safetensors files and index mapping; and
- recursive final-export path, size, and SHA-256 coverage.

The MLX parent promotes only after it verifies one uninterrupted full-duration
run from the pinned base. It checks at least one completed optimizer update,
finite train and validation losses, exact target binding, positive MLX peak and
adapter delta, live headroom evidence, the owned run marker, immutable adapter
and artifact manifests, fresh-process one-to-four-token generation, and the
`aptus.mlx-final-export.v1` adapter tree. The full run does not become resumable.

The state proves this execution and structural artifact contract. Aptus v0.2
has no task metric, baseline, quality threshold, or deployment gate.

## Report object

| Field | Meaning |
| --- | --- |
| `state` | Strongest recorded validation or lifecycle state |
| `findings` | Code, message, severity, and optional path entries |
| `checked_files` | Host-validated compiler files |
| `artifact_fingerprint` | Bundle-manifest digest or fallback tree digest |
| `smoke_command` | Suggested measured-preflight command in host reports |
| `runtime_evidence` | Invoked command, return code, and bounded output evidence |
| `validation_level` | Level actually achieved |
| `bindings` | Bundle, data, environment, hardware, plan, candidate, and model identities |
| `validator_version` | Producing host or portable validator |
| `validated_at` | UTC validation timestamp |
| `preflight_metrics` | Bound runtime-specific metrics after measured preflight; CUDA is synthetic, while MLX uses the pinned model and compiled data |
| `pilot_metrics` | Bound aggregate pilot metrics after pilot |
| `final_export` | Parent-verified final export summary after full completion |
| `measured_run` | Parent-verified full-run summary |
| `measured_run_completed_at` | Completion-promotion timestamp |
| `latest_recheck` | Lower-level recheck preserved beside stronger current evidence |

## Bindings

All pass reports bind the bundle fingerprint, dataset digest, validator, and
validation time at their producing validator's scope. They also bind plan ID,
candidate ID, model revision, environment, and hardware when that validator
records those fields.

CUDA runtime reports bind the actual environment and visible device identities.
MLX jobs execute with the selected interpreter. Generated MLX reports bind the
bundle, data, plan, candidate, model revision, and action metrics. Pilot adds the
immutable pilot-metrics digest and owned output identity. Full completion adds
the run ID, output directory, metrics digest, immutable adapter export, reload
evidence digest, and completion timestamp. CUDA full completion also adds ranks
and its runtime-specific final export.

Changing a compiler-managed file invalidates the bundle. Changing the installed
environment or target hardware can invalidate a historical runtime state even
when the report file has not been edited.

## Stronger-state preservation

A lower-level recheck does not automatically erase stronger evidence. The
validator preserves the stronger report only when relevant bundle, dataset,
plan, candidate, model, environment, hardware, preflight, pilot, and completed
run bindings remain current. For a preserved completed run, the lower recheck
is recorded under `latest_recheck`.

If those checks fail, the new lower state replaces the stale stronger report.

## Direct versus managed execution

The host API refuses synchronous runtime validation with `run=true`. Use a
managed job for serialization and cancellation. The CLI can either submit that
job or run contract and static validation directly.

Portable `validate.py` is supported on POSIX under the shared host lease. On
Windows, use managed execution for child work.

## Related documentation

- [Run states](run-states.md)
- [Bundle manifest](bundle-manifest.md)
- [Evidence records](evidence-records.md)
- [Error and finding codes](error-codes.md)
- [Capability matrix](capability-matrix.md)
- [Compile, validate, and run](../guides/compile-validate-run.md)
