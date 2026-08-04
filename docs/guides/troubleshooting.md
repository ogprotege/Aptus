# Troubleshooting

> **Status:** Active | **Authority:** Operational troubleshooting guide | **Applies to:** Aptus 0.2 | **Audience:** Users and operators | **Last reviewed:** 2026-08-04 | **Review by:** 2026-10-27 or after a new failure class

Begin with the read-only report:

```bash
aptus doctor --state-dir .aptus-state
```

For a shareable support packet, create a new no-clobber archive and inspect its
JSON before sending it:

```bash
aptus diagnostics --state-dir .aptus-state --output aptus-diagnostics.zip
```

## No feasible candidate

Read every candidate's unsupported reason. Common causes are missing BF16,
unsupported backend, insufficient upper-envelope VRAM, insufficient host RAM, an
unsupported distribution, or an invalid pinned model fact. Aptus will not
silently change sequence length, effective batch size, method, or hardware.

## Hardware scan unavailable

On Darwin arm64 without CUDA, a successful scan returns an `mps` discovery
record for shared unified memory. It is not dedicated VRAM. Aptus records live
available host memory separately and constrains MLX planning by the lesser of
that measurement and the Metal compatibility capacity, minus the reserve. If
availability is unknown, the plan cannot claim current headroom.

Use `GET /api/v1/platform` to inspect MLX, MLX-LM, PyTorch MPS, swap, memory
pressure, and Metal facts. Use `GET /api/v1/runtimes` to inspect the exact
training interpreters Aptus can use. MLX-LM LoRA and QLoRA require an external
compatible Python selected through the private runtime configuration endpoint.
LM Studio and oMLX are inference services only. Aptus never treats either
service as a training interpreter.

## Static validation fails

Do not patch a compiled bundle in place. The manifest binds compiler-managed
files. Correct source facts or generator code, then compile to a new empty path.

For model-policy snapshots, use the exact finding to classify the failure:

- `POLICY_SNAPSHOT_MISSING` and `POLICY_SNAPSHOT_JSON_ERROR` mean the snapshot
  file is absent or unreadable JSON;
- `POLICY_SNAPSHOT_CONTRACT` and `POLICY_SNAPSHOT_NONCANONICAL` mean the parsed
  snapshot violates its exact schema or canonical byte encoding; and
- `POLICY_SNAPSHOT_DIGEST` and `POLICY_SNAPSHOT_PATH` mean one of the snapshot,
  plan, manifest, or current-host digest bindings differs or the manifest names
  the wrong path.

The digest finding names invalid and differing bindings. A snapshot, plan, or
manifest disagreement is bundle-integrity failure: restore from a trusted source
or recompile. A valid `host` binding difference is policy currency and requires
replanning, as described below. Never make the digests agree by editing the
bundle.

Installed-host validation requires plan, trainer, manifest, and snapshot JSON
roots to be objects. Package-free validation enforces that boundary for the
plan, manifest, and snapshot. On covered readers, JSON `null`, arrays, scalars,
excessive nesting, and oversized integers are controlled invalid-input results,
not repairable bundle state. The trainer configuration is compiler-managed
runtime input; recompile from trusted inputs rather than editing it. If a
covered input escapes as an unhandled `TypeError`, `AttributeError`, `KeyError`,
`StopIteration`, or `RecursionError`, retain the smallest redacted reproduction
and report it as a validator defect.

## `replan_required` or host rejection after a portable pass

Package-free `validate.py` checks the frozen snapshot embedded in its bundle. It
cannot know whether an installed host's current model-policy registry has
advanced. Installed Aptus performs that separate currency check before managed
admission, at pilot authorization and worker launch, and during the completion
verification and promotion transaction.

For a coherent v5 plan whose saved decision or snapshot digest is no longer
current, preserve the old plan and bundle, recreate the plan from its source
facts, and compile to a new empty path. Saved-plan load, compile, project
recovery, and managed job submission APIs report HTTP `409 replan_required`;
host static validation instead records `POLICY_SNAPSHOT_DIGEST` in its invalid
report. Do not relabel the old schema, copy a new digest into the old plan, or
replace only the embedded snapshot. A malformed or tampered v5 decision remains
invalid input rather than a replanning case.

## Dependency validation fails

Use an isolated environment and install `requirements.txt`. The file contains
exact direct pins, not a transitive lock. Check Python version, package index
access, and the resolved installed-environment report. CUDA bundles also require
the matching CUDA driver. MLX bundles require Apple silicon macOS, `mlx==0.31.2`,
and `mlx-lm==0.31.3` in the configured interpreter.

## Model-data validation fails

Confirm network or cache access, repository ID, immutable revision, tokenizer,
model family, parameter count, gated-model credentials, and every canonical
training row. Provider inspection does not guarantee that the training runtime
can load the revision. Also inspect the trainable census. Full training rejects
any frozen model tensor, while LoRA-based paths reject trainable tensors outside
the compiled LoRA parameter scope.

On `mlx-lm`, model-data can also refuse on memory before it loads the model at
all. See the next section before re-pulling the model or re-checking credentials.

## MLX model-data refuses before the model loads

MLX-LM model-data validation measures the pinned snapshot's safetensors shards
and live available unified memory *before* any weight loads. It adds any packed
size in excess of the planned resident bytes to both the point and upper
estimates, then requires

```text
available >= max(adjusted point, adjusted upper) + max(plan reserve, 8 GiB)
```

If that fails, the action stops with `required=`, `available=`, and `shortfall=`
byte counts in the job log, and writes no evidence. This is a fail-closed refusal,
not a crash: no weights were loaded and nothing was overwritten.

Read the three byte counts from the log, then either free unified memory and
re-run model-data, or replan a smaller candidate. Increasing `--reserve-gib` does
not help, because the reserve is added to the requirement. A passing action binds
`aptus.mlx-unified-memory-admission.v2` inside `model-data-evidence.json`.

The recorded 2026-07-28 Qwen3 30B-A3B attempt stopped here, 18.932 GiB short. See
[Qwen3 MoE admission evidence](../operations/evidence/2026-07-28-qwen3-moe-admission/README.md).

## Preflight or pilot runs out of memory

Free VRAM or Apple unified-memory headroom can change after planning. Stop
unrelated accelerator work or choose different explicit facts and replan. On
Apple silicon, also inspect memory pressure and swap growth. Do not treat the
analytic estimate as a measured ceiling. A pilot failure blocks full training.

## MLX pilot fails

The MLX-LM pilot should proceed after `measured-preflight-pass`. It runs the
exact model and compiled data from the pinned base without interruption. Inspect
the job log and owned `pilot-output/pilot_*` directory for the failing contract:

- fewer than two completed optimizer updates;
- non-finite train or validation loss;
- incomplete target coverage or an unexpected trainable tensor;
- zero adapter delta or MLX peak;
- insufficient live unified-memory headroom;
- changed, missing, or escaped action-owned artifacts; or
- fresh-process adapter reload that did not generate one to four tokens.

Do not use a resume argument. The reload proves adapter inference, not optimizer
or training-state continuation. Preserve failed output and start a new pilot
after correcting the cause.

## Dataset split is rejected

Top-level and `metadata.split_group` values must be non-empty strings and must
agree when both are present. Aptus rejects a canonical dataset that changes
while its split is computed or consumed. In distributed runs, every rank must
observe the same canonical digest, assignment digest, and row counts. Large
declared groups may prevent an exact evaluation fraction. Use the recorded
target, realized fraction, and row error to review the result.

## Full FSDP is rejected

This is expected in v0.2. Full-parameter FSDP is unsupported. LoRA FSDP is
conditional. Quantized FSDP is unsupported.

## Train submission is rejected after pilot

Admission deeply rechecks current environment, hardware, host RAM, disk, bundle,
pilot artifacts, and installed-host model-policy currency. Review the returned
reason. A displayed pilot pass can be historical while current authorization
fails. For `replan_required`, create and compile a new current plan rather than
rerunning the old pilot unchanged.

## Active job conflict

Aptus runs one managed or portable accelerator action at a time for the same
user across state roots. Wait, cancel through the API or workbench, or inspect
the recorded owner. The global lease coordinates Aptus only and can also reveal
another Aptus service instance.

## Job remains cancelling or verifying

`cancelling` means process-group termination is still being reconciled.
`verifying` means the child exited and parent-owned artifact verification is in
progress. Neither state is safe to relabel or overwrite.

## A project or job record was quarantined

Aptus moves a malformed, symlinked, or unsupported persistent record into the
state root's private `quarantine/` tree and writes a reason receipt. Other
healthy state remains usable. Preserve the quarantined file, inspect the reason,
and recover from a known-good project revision or backup. Do not rename the file
back into the active tree without correcting and validating its contract.

## The Mac app refuses to quit or restart

The native host refuses application termination when its backend process tree
still has an active survivor. This is intentional containment. Wait for the
process to exit, then request Quit again so the controller can retry cleanup.
The backend log contains an `aptus-shutdown-timeout` diagnostic with process
identity, observed state, and signal attempts. Do not delete the session
directory or launch a second copy to bypass the failed shutdown.

## A run exited zero but failed

Process exit is only one completion condition. Missing, stale, non-finite, or
misbound metrics and export files cause parent verification to fail.

## I need to resume full training

Full-run resume is fail-closed. Preserve a CUDA checkpoint or MLX weight
snapshot for investigation, then start a new immutable run after correcting the
cause. Every MLX resume argument is rejected. See
[recovery and the resume boundary](resume-recover.md).

## Related documentation

- [Error and finding codes](../reference/error-codes.md)
- [Validation states](../reference/validation-states.md)
- [Operator checklist](../operations/operator-checklist.md)
- [Compile, validate, and run](compile-validate-run.md)
