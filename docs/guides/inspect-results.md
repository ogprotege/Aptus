# Inspect Results

> **Status:** Active | **Audience:** Fine-tuning practitioners and operators | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Runtime | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22

An Aptus result is a chain of bound records, not a single success message.
Inspect the plan, validation report, managed job, run metrics, and final export
together before making any claim.

## Four places to inspect

| Record | Typical location | Question it answers |
|---|---|---|
| Decision report | `bundle/decision-report.md` | Why did the planner select this candidate? |
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

## 1. Re-read the selected decision

Confirm that `plan.json`, `decision-report.md`, and the job all name the same
plan ID and candidate ID. Review the selected:

- method, distribution, world size, and device indices;
- precision and quantization;
- batch arithmetic;
- rank, alpha, learning-rate prior, and target modules;
- point and upper memory estimates;
- host RAM, disk, checkpoint, and export estimates;
- assumptions, warnings, evidence IDs, and conditional reasons.

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
does not guarantee current capacity. Train admission rechecks the environment,
hardware, host RAM, disk, bundle, plan, and pilot artifacts under the lease.

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
- completion attestation and artifact-integrity status.

A zero process exit is necessary but not sufficient for a completed training
job. Missing, stale, non-finite, or misbound evidence causes parent verification
to fail.

## 4. Inspect measured resource and scope evidence

Measured preflight, both pilot phases, and a completed full run carry a
trainable-parameter census. Verify that it has:

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

## 5. Inspect dataset-split evidence

Full-run metrics must bind:

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

## 6. Inspect the final export

The unique run directory must contain the run-bound metrics and
`final-export.json`. Parent verification checks the expected safetensors form,
non-empty tensor keys, index mappings when present, model or adapter provenance,
and recursive path, size, and digest coverage.

For full training, expect complete model configuration, tokenizer material, and
model safetensors. For LoRA-based training, expect adapter configuration,
tokenizer material, adapter safetensors, and base-model provenance.

`measured-run-pass` means the structural run and export contracts passed. It
does not mean:

- the adapted model beats the base model;
- the model is safe or factually reliable;
- the export reloads in an untested serving stack;
- inference parity, latency, or deployment fitness passed;
- the run met a task-quality threshold.

Use a separate, predefined evaluation for those claims.

## Preserve a failed result

Keep the job JSON, complete log, validation report, run directory, plan,
manifest, installed-environment record, and relevant hardware evidence. Do not
reuse or overwrite the failed run directory. Correct the cause, refresh every
invalidated validation level, and submit a new run with a new ID.

Full-training resume is unsupported. A pilot's bounded checkpoint continuation
does not authorize resuming an interrupted full run.

## Related documentation

- [Validation states](../reference/validation-states.md)
- [Run states](../reference/run-states.md)
- [Compile, validate, and run](compile-validate-run.md)
- [Recovery and the resume boundary](resume-recover.md)
- [Design an evaluation](design-an-evaluation.md)
