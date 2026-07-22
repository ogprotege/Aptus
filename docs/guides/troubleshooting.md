# Troubleshooting

## No feasible candidate

Read every candidate's unsupported reason. Common causes are missing BF16,
unsupported backend, insufficient upper-envelope VRAM, insufficient host RAM, an
unsupported distribution, or an invalid pinned model fact. Aptus will not
silently change sequence length, effective batch size, method, or hardware.

## Hardware scan unavailable

The API returns a manual-facts option when CUDA inspection is unavailable. Enter
facts for planning, but run preflight and pilot on the actual CUDA host before
training. The current development Mac cannot provide CUDA evidence.

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
can load the revision.

## Preflight or pilot runs out of memory

Free VRAM can change after planning. Stop unrelated GPU work or choose different
explicit facts and replan. Do not treat the analytic estimate as a measured
ceiling. A pilot failure blocks full training.

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
