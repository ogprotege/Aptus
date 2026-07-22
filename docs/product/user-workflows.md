# User Workflows

> **Status:** Active | **Authority:** Explanatory workflow guide | **Applies to:** Aptus 0.2 | **Audience:** Users and operators | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when a workflow changes

## Plan for a known host

1. Pin the model repository to an immutable commit.
2. Verify license and training permission.
3. Profile the local dataset.
4. Scan the target host or enter measured hardware facts.
5. Select MLX-LM for Apple Silicon or Transformers and PEFT for CUDA.
6. State sequence length, effective batch, epochs, objective, and checkpoint
   interval.
7. Compare all candidate statuses, runtime contracts, and assumptions.
8. Compile the selected plan to a new path.

## Prove a bundle before training

For a CUDA bundle, run and review these actions in order:

1. Dependency validation.
2. Model-data validation across every canonical row.
3. Selected-method measured preflight.
4. Two-phase real-model and data pilot.
5. Full training with explicit confirmation.

Each action can invalidate an earlier analytic expectation. Stop and replan when
the evidence changes a bound fact. The service rejects a skipped action, and a
higher validation job cumulatively rechecks the lower validation levels.

For an MLX-LM bundle, run the same five managed actions. Its pilot is not the
CUDA two-process continuation test. It starts from the pinned base, completes at
least two optimizer updates without interruption, verifies finite losses and
exact target coverage, then reloads the saved adapter in a fresh process for
one to four generated tokens. `pilot-pass` permits explicitly confirmed
full-duration adapter training from the same pinned base.

MLX-LM does not support crash resume. Do not pass a resume argument or describe
periodic weight snapshots as checkpoints. If pilot or full training is
interrupted, preserve its unique output and start a new run after correction.

## Configure Apple training and inference separately

In Aptus for Mac, open Models and choose the exact external Python executable
that imports the pinned MLX and MLX-LM versions. Aptus probes and persists that
canonical path. Finder-launched apps do not inherit your shell environment.

LM Studio and oMLX are separate loopback inference services. Their model lists
and generated text cannot satisfy a training runtime or evidence gate. PyTorch
MPS is discoverable and configurable but has no compiler.

## Plan for a different host

Manual hardware facts can compare candidates before access to the target host.
Label them as user-attested. Transfer a CUDA bundle only after reviewing
its cleartext dataset copies. On the target host, create an isolated environment
outside the bundle directory and repeat the entire runtime sequence. An
in-bundle virtual environment invalidates the manifest.

## Inspect a provider model

Use the pinned ID and revision with model inspection. Copy only reviewed
provider-declared architecture fields. Supply license and permission separately.
Model-data validation remains mandatory because metadata inspection is not a
runtime load test.

## Respond to a failed CUDA pilot

Preserve metrics and logs. Identify whether the failure came from dependency,
model structure, data transformation, CUDA capacity, distribution, checkpoint,
or export. Correct the source fact or implementation, then create a new bundle
when compiler-managed content changes. Do not authorize training from the failed
bundle.

For MLX-LM, inspect optimizer-update counts, loss evidence, target census,
adapter delta, unified-memory admission, immutable artifact manifests, and the
fresh-process adapter reload. Do not reinterpret a failed fresh reload as proof
that training can resume.

## Recover an interrupted full run

Preserve the unique failed run directory. Full-run resume is unsupported. Fix
the cause, refresh any invalidated validation action, and submit a new train job
with a new run ID.

## Interpret completion

`measured-run-pass` means the exact run's metrics and structural export tree
passed parent verification. Review output under the recorded `run_output_dir`.
Use a separate, explicit evaluation process before making quality or deployment
claims.

## Related documentation

- [Choose your path](../getting-started/choose-your-path.md)
- [Operator checklist](../operations/operator-checklist.md)
- [Inspect results](../guides/inspect-results.md)
- [Recovery and resume boundary](../guides/resume-recover.md)
