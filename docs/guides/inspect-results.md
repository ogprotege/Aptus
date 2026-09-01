# Inspect Results

> **Status:** Active | **Audience:** Fine-tuning practitioners and operators | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Runtime | **Last reviewed:** 2026-08-31 | **Review by:** 2026-10-22

An Aptus result is a chain of bound records, not a single success message.
Inspect the plan, validation report, managed job, run metrics, and final export
together before making any claim.

## Five places to inspect

| Record | Typical location | Question it answers |
|---|---|---|
| Decision report | `bundle/decision-report.md` | Why did the planner select this candidate? |
| Policy snapshot | `bundle/policy/model-policy-snapshot.v1.json` | Which frozen model-policy contract produced the saved compatibility decision? |
| Validation report | `bundle/validation-report.json` | Which evidence state has this exact bundle reached? |
| Managed job and log | `.aptus-state/jobs/job_*.json` and `.log` by default | What action ran, where, and how did the process end? |
| Run output | `bundle/runs/run_*/` | What did this unique full run measure and export? |

The CLI can show managed state without reading files directly:

```bash
aptus jobs
aptus jobs --id JOB_ID
```

Use the `bundle_dir`, `log`, `run_id`, and `run_output_dir` values from the job
record. Do not infer paths from timestamps or directory order.

An optional evaluation contract and result live outside that ladder:

```bash
aptus eval-generate --bundle BUNDLE --gold GOLD.jsonl \
  --adapter ADAPTER --output PRED.jsonl
aptus eval-contract --dataset GOLD.jsonl --claim "..." --threshold 1 --output contract.json
aptus eval --contract contract.json --gold GOLD.jsonl --predictions PRED.jsonl --output result.json
```

`decision: pass` is only exact-match against the bound gold digest. It is not
train-loss success, `measured-run-pass`, general quality, or release readiness.

## 1. Re-read the selected decision

Confirm that `plan.json`, `decision-report.md`, and the job all name the same
plan ID and candidate ID. Review the selected:

- method, distribution, world size, and device indices;
- precision and quantization;
- batch arithmetic;
- rank, alpha, learning-rate prior, and target modules;
- point and upper memory estimates;
- host RAM, disk, CUDA checkpoint or MLX weight-snapshot estimates, and export
  estimates;
- assumptions, warnings, evidence IDs, and conditional reasons.

Also inspect the saved `model_policy_decision`, its source and decision ID, any
exact policy/path binding, and `model_policy_snapshot_sha256`. The canonical
snapshot bytes, the plan field, the manifest `policy_snapshot_sha256`, and the
manifest file entry must agree. That proves the bundle's frozen policy integrity;
only installed-host validation can compare it with the current registry.

The recommendation is highest-ranked only within the enumerated viable catalog.
Do not reinterpret it as a universal optimum.

## 2. Read validation as an evidence ladder

The important states are ordered:

1. `contract-pass`
2. `static-pass`
3. `dependency-pass`
4. `model-data-pass`
5. `measured-preflight-pass`
6. `pilot-pass`
7. `execution-approved`
8. `measured-run-pass`

Read every finding even when the state passed. Warnings preserve uncertainty and
assumptions that a simple state label cannot express. A historical `pilot-pass`
does not guarantee current capacity. Train admission rechecks runtime-specific
capacity, bundle, plan, pilot artifacts, and current host model policy under the
lease. A package-free frozen-snapshot pass is not current-host authorization.

## 3. Inspect the managed job

For a nonterminal job, check `state` and `phase` together. `verifying` is a
phase within `running`, not a separate persisted state. It means the child has
exited and the parent is checking metrics and artifacts. Do not cancel or mark
the job complete by editing its JSON file.

For a terminal job, inspect:

- action and command;
- start and finish timestamps;
- return code and error text;
- process and process-group identities;
- log path;
- run ID and output path for training;
- prelaunch capacity evidence;
- the host-authorized model-policy snapshot digest;
- completion attestation and artifact-integrity status.

A zero process exit is necessary but not sufficient for a completed training
job. Missing, stale, non-finite, or misbound evidence causes parent verification
to fail.

## 4. Inspect measured resource and scope evidence

CUDA measured preflight, both CUDA pilot phases, and a completed CUDA full run
carry a trainable-parameter census. Verify that it has:

- positive integer tensor and parameter counts;
- finite initial values;
- the selected method's expected scope;
- a stable digest over sorted parameter names, shapes, and dtypes;
- identical pilot-phase records;
- exact optimizer membership during real training.

For LoRA-based methods, the runtime also proves one complete LoRA A/B pair for
each inspected target-module instance and rejects other trainable tensors. For
full training, every model tensor must remain trainable.

Compare the measured CUDA peak with the analytic point and upper estimates.
Treat the difference as calibration evidence for that exact configuration, not
as a universal correction factor.

For MLX measured preflight, pilot, and full runs, inspect the
`aptus.mlx-trainable-target-binding.v1` record. It must bind one LoRA A/B pair
for every present planned-target instance, reject other trainables, and carry a
stable descriptor digest. q_proj, o_proj, gate_proj, up_proj, and down_proj still
cover every transformer layer. Only family `gemma4` may omit k_proj and v_proj
together on KV-shared layers; those two counts must match. Lane 8 is on main
for families `gemma4` and `gemma4_moe`: omit-`v_proj` is allowed when
`v_count` does not exceed `k_count`. Inspect persists provider `attention_k_eq_v`
and `num_kv_shared_layers` when declared. Loaded `k_proj` / `v_proj` instance
counts require a bound loader; Gemma 4 unified is recognized and unsupported
by the current compiler contract, so that census is not a loaded fact yet.
Also inspect completed optimizer updates, finite train
and validation losses, positive adapter delta, positive MLX peak, and live
unified-memory admission. Pilot requires at least two updates. Full requires at
least one.

## 5. Inspect dataset-split evidence

CUDA full-run metrics must bind:

- split strategy identifier;
- canonical training JSONL digest;
- assignment digest;
- total, train, and evaluation row counts;
- declared-group and split-unit counts;
- requested evaluation fraction and target row count;
- realized evaluation fraction and row error.

An indivisible declared group can make the realized fraction differ from the
request. That is not a split failure when the evidence is internally consistent
and no group crosses sides. A digest mismatch, count mismatch, cross-rank
disagreement, or data mutation is a failure.

MLX uses the compiler-bound `aptus.mlx-split.v1` train and validation files. Its
metrics bind source and padded compiled row counts and require finite validation
loss. The current MLX split does not claim group-aware subset selection or an
exact evaluation fraction.

## 6. Inspect the final export

The unique run directory must contain run-bound metrics and
`final-export.json`. Parent verification checks the runtime-specific expected
safetensors form, provenance, paths, sizes, and digests.

For full training, expect complete model configuration, tokenizer material, and
model safetensors on CUDA. For CUDA LoRA-based training, expect adapter
configuration, tokenizer material, adapter safetensors, and base-model
provenance. For MLX, expect `adapter_config.json`, `adapters.safetensors`,
`aptus.mlx-final-export.v1`, and bound fresh-process adapter reload evidence.

`measured-run-pass` means the structural run and export contracts passed. It
does not mean:

- the adapted model beats the base model;
- the model is safe or factually reliable;
- the export reloads in an untested serving stack;
- inference parity, latency, or deployment fitness passed;
- the run met a task-quality threshold.

Use a separate, predefined evaluation for those claims.

## 7. Inspect training-signal correction (optional)

When a completed train job has a readable `metrics.json` with recorded
`train_loss_observations` (and optionally `validation_loss_observations`),
job GET and `aptus jobs --id JOB_ID` may attach a presentation-only
`aptus.run-correction.v1` object (`run_correction`). The Run UI labels it
**Training-signal correction (not quality).**

This is a regularization heuristic for the next plan. It is not model quality,
not an `aptus.evaluation-result.v1` decision, and not required for
`measured-run-pass`. Aptus does not auto-replan, auto-stop, or change weight
decay from this signal.

Kinds (first match): `eval-rose`, `loss-collapsed`, `loss-flat`, or `none`.
CLI stderr prints a block titled like the training-knobs presentation:

```text
Aptus training-signal correction (presentation only; not quality):
```

## Preserve a failed result

Keep the job JSON, complete log, validation report, run directory, plan,
manifest, embedded policy snapshot, installed-environment record, and relevant
hardware evidence. Do not reuse or overwrite the failed run directory. Correct
the cause, refresh every invalidated validation level, and submit a new run with
a new ID. If installed policy is no longer current for the saved v5 plan, retain
that chain as historical evidence and create a new plan and bundle.

Full-training resume is unsupported. CUDA pilot checkpoint continuation does not
authorize resuming an interrupted full run. MLX pilot and full training are
uninterrupted, and its fresh adapter reload does not restore training state.

## Related documentation

- [Validation states](../reference/validation-states.md)
- [Run states](../reference/run-states.md)
- [Compile, validate, and run](compile-validate-run.md)
- [Recovery and the resume boundary](resume-recover.md)
- [Design an evaluation](design-an-evaluation.md)
