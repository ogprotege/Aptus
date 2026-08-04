# Execution Orchestrator

> **Status:** Active | **Authority:** Normative architecture | **Applies to:** Aptus 0.2 | **Audience:** Contributors and operators | **Last reviewed:** 2026-08-04 | **Review by:** 2027-01-27 or when job semantics change

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
validation level. Pilot invokes the runtime-specific bounded pilot. CUDA uses
two fresh training processes and checkpoint continuation. MLX uses one
uninterrupted training process and a separate fresh adapter-reload process. The
service rejects an action until the preceding validation state is recorded. Higher validation
levels also rerun lower levels inside the submitted job as defense-in-depth.
Earlier passed actions remain available for an explicit recheck; only forward
skips are rejected. Train has three launch shapes: MLX train launches `run.py`;
single-distribution CUDA train launches `train.py` directly under the resolved
interpreter; and distributed CUDA train launches `train.py` through the
interpreter-bound `accelerate.commands.accelerate_cli launch --config_file
config/accelerate.yaml`. For managed CUDA jobs the parent that owns completion
verification is `JobService`, not `run.py`.

Full training requires `confirm_full_train=true`. `resume_from` is rejected.

## Persistence

Each `aptus.job-record.v1` record contains identity, action, bundle path, command, log path,
timestamps, process identity, process-group identity, return code, current
phase, run ID and output path when applicable, capacity evidence, error text,
the submission-bound artifact fingerprint and authorized model-policy snapshot
digest, and completion attestation when available.

Atomic mode-0600 JSON replacement prevents partially written records. The job
root is user-private. Startup and reads reconcile stale owners and processes
against recorded process identities. Legacy records migrate with durable
authorization cleared. Corrupt, symlinked, and unsupported records move to a
private quarantine with a reason receipt, while healthy records remain usable.

## Serialization

The service holds:

- an in-process lock;
- a state-root record lock;
- a per-user host-global Aptus lease.

Together they permit one Aptus accelerator action across state roots. Validation that
can mutate runtime reports is serialized against job submission. The global
lease coordinates managed jobs and direct portable entrypoints. Unrelated CUDA
or Metal programs do not participate.

## Admission

Job submission first validates the bundle and required preceding report state.
Installed Aptus separately requires the coherent v5 decision and embedded
snapshot digest to match the current host registry. It records the approved
digest as `authorized_model_policy_snapshot_sha256`; a stale plan requires
replanning before a job or lease is created. Package-free validation of the
bundle's frozen snapshot cannot establish this currency.
API submission also requires the exact current project revision. It compares
the resolved bundle path, saved plan snapshot, plan ID, selected candidate, and
recorded bundle-manifest fingerprint. A same-content plan in another project
does not authorize the bundle. The accepted job record carries that fingerprint
through launch.
Train submission also performs deep pilot authorization while holding the lease
and record locks. CUDA checks current free CUDA memory, host RAM, disk,
environment, bundle, plan, pilot metrics, checkpoint contracts, and pilot export
contracts. MLX verifies the owned uninterrupted pilot, then checks current
unified-memory headroom against measured peak plus reserve and current disk
against plan and measured adapter artifacts.

The public pilot-authorization query performs the same current-policy check. A
previously passing pilot reports non-current after the installed registry
changes and cannot authorize a full run.

Public polling can use cached completion evidence and cheap presence checks.
The deep admission transaction decides whether a train job may start.

## Launch and cancellation

On POSIX systems, jobs launch in their own process group. Cancellation records
`cancelling`, sends termination to the recorded group, and escalates if required.
The service validates process identity to reduce PID-reuse risk. Cancellation is
not accepted while parent-owned completion verification is committing.

The worker starts a permit launcher with the verified bundle as its working
directory. Manifest entries remain relative to that directory. After the parent
records process identity and binds the global lease, it rechecks the project
artifact binding and current host policy, then releases the permit. The launch
specification and child environment carry the approved digest as
`APTUS_AUTHORIZED_MODEL_POLICY_SNAPSHOT_SHA256`. Generated entrypoints compare
it with the plan and manifest before using plan state. The launcher then rereads the manifest,
verifies its recorded fingerprint, rejects relative-path escapes and symlinks,
and rehashes every manifested file immediately before `exec`. This closes the
submission-to-launch mutation interval without trusting an absolute path from
the launch specification.

Windows uses the supported direct-process termination path and has a narrower
group-control contract. Direct portable child execution is fail-closed on
Windows in v0.2 and must use `JobService`.

## Full-run transaction

Train jobs receive a unique `run_*` output path. The child cannot overwrite a
prior run. On aggregate exit code zero, the parent changes phase to `verifying`
and deeply checks the job-specific marker, metrics, report bindings, and final
export manifest.

An MLX train job starts again from the pinned base and runs without interruption
for its plan-derived duration. Its parent also verifies exact target binding,
finite train and validation losses, positive memory and adapter delta, immutable
artifacts, and fresh-process one-to-four-token adapter generation. MLX weight
snapshots are not resumable checkpoints.

Verified pending evidence is stored in the job record before report promotion.
Before a new promotion, the parent requires the submission-bound artifact and
current host policy, computes the runtime-specific promotion, and repeats that
check immediately before writing the report. Promotion to `measured-run-pass`
is idempotent. Startup reconciliation can finish that transaction after a
parent crash only when the persisted evidence and current policy still pass. A
policy change leaves pending evidence unpromoted and fails reconciliation.

Terminal job state does not imply model quality. It means the execution and
artifact contracts passed or failed as recorded.

## Related documentation

- [Run states](../reference/run-states.md)
- [Validation states](../reference/validation-states.md)
- [Model-policy snapshot](../reference/model-policy-snapshot.md)
- [Operator checklist](../operations/operator-checklist.md)
- [Recovery and resume boundary](../guides/resume-recover.md)
