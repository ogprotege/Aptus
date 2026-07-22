# User Workflows

## Plan for a known CUDA host

1. Pin the model repository to an immutable commit.
2. Verify license and training permission.
3. Profile the local dataset.
4. Scan the target host or enter measured hardware facts.
5. State sequence length, effective batch, epochs, objective, and checkpoint
   interval.
6. Compare all candidate statuses and assumptions.
7. Compile the selected plan to a new path.

## Prove a bundle before training

Run and review these actions in order:

1. Dependency validation.
2. Model-data validation across every canonical row.
3. Selected-method measured preflight.
4. Two-phase real-model and data pilot.
5. Full training with explicit confirmation.

Each action can invalidate an earlier analytic expectation. Stop and replan when
the evidence changes a bound fact. The service rejects a skipped action, and a
higher validation job cumulatively rechecks the lower validation levels.

## Plan for a different host

Manual hardware facts can compare candidates before access to the target host.
Label them as user-attested. Transfer the compiled bundle only after reviewing
its cleartext dataset copies. On the target host, create an isolated environment
and repeat the entire runtime sequence.

## Inspect a provider model

Use the pinned ID and revision with model inspection. Copy only reviewed
provider-declared architecture fields. Supply license and permission separately.
Model-data validation remains mandatory because metadata inspection is not a
runtime load test.

## Respond to a failed pilot

Preserve metrics and logs. Identify whether the failure came from dependency,
model structure, data transformation, CUDA capacity, distribution, checkpoint,
or export. Correct the source fact or implementation, then create a new bundle
when compiler-managed content changes. Do not authorize training from the failed
bundle.

## Recover an interrupted full run

Preserve the unique failed run directory. Full-run resume is unsupported. Fix
the cause, refresh any invalidated validation action, and submit a new train job
with a new run ID.

## Interpret completion

`measured-run-pass` means the exact run's metrics and structural export tree
passed parent verification. Review output under the recorded `run_output_dir`.
Use a separate, explicit evaluation process before making quality or deployment
claims.
