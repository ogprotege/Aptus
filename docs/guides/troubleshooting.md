# Troubleshooting

> **Status:** Active | **Authority:** Operational troubleshooting guide | **Applies to:** Aptus 0.2 | **Audience:** Users and operators | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or after a new failure class

## No feasible candidate

Read every candidate's unsupported reason. Common causes are missing BF16,
unsupported backend, insufficient upper-envelope VRAM, insufficient host RAM, an
unsupported distribution, or an invalid pinned model fact. Aptus will not
silently change sequence length, effective batch size, method, or hardware.

## Hardware scan unavailable

On Darwin arm64 without CUDA, a successful scan returns an `mps` discovery
record for measured shared unified memory. It can inform hardware inventory, but
it cannot authorize MPS or MLX execution. Current memory availability may remain
unknown. If no supported probe can measure the host, the API returns a
manual-facts option. Manual facts can support planning, but preflight and pilot
must run on the actual CUDA host before training. The current development Mac
cannot provide CUDA execution evidence.

## Static validation fails

Do not patch a compiled bundle in place. The manifest binds compiler-managed
files. Correct source facts or generator code, then compile to a new empty path.

## Dependency validation fails

Use an isolated environment and install `requirements.txt`. The file contains
exact direct pins, not a transitive lock. Check Python version, CUDA driver,
package index access, and the resolved installed-environment report.

## Model-data validation fails

Confirm network or cache access, repository ID, immutable revision, tokenizer,
model family, parameter count, gated-model credentials, and every canonical
training row. Provider inspection does not guarantee that the training runtime
can load the revision. Also inspect the trainable census. Full training rejects
any frozen model tensor, while LoRA-based paths reject trainable tensors outside
the compiled LoRA parameter scope.

## Preflight or pilot runs out of memory

Free VRAM can change after planning. Stop unrelated GPU work or choose different
explicit facts and replan. Do not treat the analytic estimate as a measured
ceiling. A pilot failure blocks full training.

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
and pilot artifacts. Review the returned reason. A displayed pilot pass can be
historical while current authorization fails.

## Active job conflict

Aptus runs one managed or portable GPU action at a time for the same user across state roots. Wait,
cancel through the API or workbench, or inspect the recorded owner. The global
lease coordinates Aptus only and can also reveal another Aptus service instance.

## Job remains cancelling or verifying

`cancelling` means process-group termination is still being reconciled.
`verifying` means the child exited and parent-owned artifact verification is in
progress. Neither state is safe to relabel or overwrite.

## A run exited zero but failed

Process exit is only one completion condition. Missing, stale, non-finite, or
misbound metrics and export files cause parent verification to fail.

## I need to resume full training

Full-run resume is fail-closed. Preserve the checkpoint for investigation and
start a new immutable run after correcting the cause. See
[recovery and the resume boundary](resume-recover.md).

## Related documentation

- [Error and finding codes](../reference/error-codes.md)
- [Validation states](../reference/validation-states.md)
- [Operator checklist](../operations/operator-checklist.md)
- [Compile, validate, and run](compile-validate-run.md)
