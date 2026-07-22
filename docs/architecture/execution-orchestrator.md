# Execution Orchestrator

> **Status:** Active | **Authority:** Normative architecture | **Applies to:** Aptus 0.2 | **Audience:** Contributors and operators | **Last reviewed:** 2026-07-22 | **Review by:** 2027-01-22 or when job semantics change

The orchestrator turns runtime validation and training into persisted,
cancellable local jobs. It is a single-user local process manager, not a remote
scheduler.

## Job actions

The accepted actions are ordered:

1. `dependency`
2. `model-data`
3. `preflight`
4. `pilot`
5. `train`

Dependency, model-data, and preflight invoke the corresponding portable
validation level. Pilot invokes the two-phase bounded pilot. The service rejects
an action until the preceding validation state is recorded. Higher validation
levels also rerun lower levels inside the submitted job as defense-in-depth.
Earlier passed actions remain available for an explicit recheck; only forward
skips are rejected.
Train launches the selected single-device or interpreter-bound Accelerate
command.

Full training requires `confirm_full_train=true`. `resume_from` is rejected.

## Persistence

Each job record contains identity, action, bundle path, command, log path,
timestamps, process identity, process-group identity, return code, current
phase, run ID and output path when applicable, capacity evidence, error text,
and completion attestation when available.

Atomic JSON replacement prevents partially written records. Startup and reads
reconcile stale owners and processes against recorded process identities.

## Serialization

The service holds:

- an in-process lock;
- a state-root record lock;
- a per-user host-global Aptus lease.

Together they permit one Aptus GPU action across state roots. Validation that
can mutate runtime reports is serialized against job submission. The global
lease coordinates managed jobs and direct portable entrypoints. Unrelated CUDA
programs do not participate.

## Admission

Job submission first validates the bundle and required preceding report state.
Train submission also performs deep pilot authorization while holding the lease
and record locks. It checks current
free CUDA memory, free host RAM, disk, environment, bundle, plan, pilot metrics,
checkpoint contracts, and pilot export contracts before persisting a queued
train job.

Public polling can use cached completion evidence and cheap presence checks.
The deep admission transaction decides whether a train job may start.

## Launch and cancellation

On POSIX systems, jobs launch in their own process group. Cancellation records
`cancelling`, sends termination to the recorded group, and escalates if required.
The service validates process identity to reduce PID-reuse risk. Cancellation is
not accepted while parent-owned completion verification is committing.

Windows uses the supported direct-process termination path and has a narrower
group-control contract. Direct portable child execution is fail-closed on
Windows in v0.2 and must use `JobService`.

## Full-run transaction

Train jobs receive a unique `run_*` output path. The child cannot overwrite a
prior run. On aggregate exit code zero, the parent changes phase to `verifying`
and deeply checks the job-specific marker, metrics, report bindings, and final
export manifest.

Verified pending evidence is stored in the job record before report promotion.
Promotion to `measured-run-pass` is idempotent. Startup reconciliation can finish
that transaction after a parent crash when the persisted evidence still passes.

Terminal job state does not imply model quality. It means the execution and
artifact contracts passed or failed as recorded.

## Related documentation

- [Run states](../reference/run-states.md)
- [Validation states](../reference/validation-states.md)
- [Operator checklist](../operations/operator-checklist.md)
- [Recovery and resume boundary](../guides/resume-recover.md)
