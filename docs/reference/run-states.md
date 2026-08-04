# Managed Run States

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Operators, UI developers, and job-service integrators |
| Authority | Normative lifecycle reference for persisted Aptus jobs |
| Last reviewed | 2026-08-04 |
| Next review | 2026-11-01, or sooner when `src/aptus/execution.py` changes |

Aptus runtime validation and training use persisted local jobs. A job state
describes process lifecycle. It is not the same as validation evidence state.

## State machine

```text
queued -> running -> completed
   |         |  \
   |         |   -> failed
   |         -> cancelling -> cancelled
   -> failed
   -> cancelling -> cancelled
```

Startup or polling reconciliation can also convert an unattached active record
to `failed` when Aptus cannot prove a valid live owner or child.

| State | Terminal | Meaning |
| --- | ---: | --- |
| `queued` | No | Record and global lease exist; child launch has not completed |
| `running` | No | Runtime child or parent-owned verification is active |
| `cancelling` | No | Termination was requested and is being reconciled |
| `completed` | Yes | Action-specific completion checks passed |
| `failed` | Yes | Admission-adjacent persistence, launch, child execution, ownership, or verification failed |
| `cancelled` | Yes | Owned process termination completed without success promotion |

`completed` is action-specific. A completed dependency job proves the dependency
gate. A completed train job additionally requires parent verification and report
promotion. Neither result proves task quality.

## Actions and prerequisites

| Action | Required report state or later | Command behavior |
| --- | --- | --- |
| `dependency` | `static-pass` | Runs portable dependency validation |
| `model-data` | `dependency-pass` | Reruns lower levels, then exact model and data validation |
| `preflight` | `model-data-pass` | Reruns lower levels, then runtime-specific measured preflight |
| `pilot` | `measured-preflight-pass` | Reruns lower levels, then the runtime-specific exact-model pilot |
| `train` | `pilot-pass` | Performs deep admission, then launches full training |

The prerequisite check accepts a later valid state, so operators can explicitly
recheck an earlier action. It rejects forward skips. Full training also requires
`confirm_full_train=true` and a current deep pilot authorization.

Every managed submission first requires the bundle's coherent v5 decision and
snapshot digest to match the installed host registry. A stale same-schema plan
requires replanning before Aptus creates a job record or lease. Pilot
authorization returns non-current after a policy change even when its historical
report still says `pilot-pass`.

## Job record fields

### Persisted submission fields

| Field | Meaning |
| --- | --- |
| `id`, `job_id` | Same `job_` plus UUID-hex identity |
| `state` | Persisted lifecycle state |
| `action` | One of the five managed actions |
| `bundle_dir` | Resolved bundle root |
| `command` | Exact argument vector, never a shell command string |
| `log` | Persisted combined stdout and stderr path |
| `return_code` | Child exit code when known |
| `resume_from` | Always null through public v0.2 APIs |
| `run_id` | `run_` identity for train, otherwise null |
| `run_output_dir` | Unique full-run path for train, otherwise null |
| `created_at`, `started_at`, `finished_at` | UTC lifecycle timestamps |
| `error` | Current terminal or cancellation error text |
| `owner_pid`, `owner_process_identity` | Submitting service identity |
| `process_pid`, `process_identity` | Managed launcher identity |
| `process_group_id` | POSIX group identity when available |
| `prelaunch_capacity_check` | Deep current train-admission evidence |
| `authorized_model_policy_snapshot_sha256` | Snapshot digest approved from the current host registry at submission |

Launch can add `launch_protocol`, `cancel_requested_at`, and completion
transaction fields. Train verification can add `verified_pending_evidence`,
`completion_attestation`, `artifact_integrity_status`, and verification
timestamps.

### Computed read fields

`JobService.get()` adds or refreshes:

| Field | Meaning |
| --- | --- |
| `phase` | Usually the lifecycle state; `verifying` during parent completion checks |
| `cancellable` | True only when this live service owns active work and is not verifying |
| `owner_status` | `owning-service`, `external-service`, `orphan-child`, `unavailable`, or `terminal` |
| `cancellation_note` | Human-readable ownership and cancellation guidance |
| `log_tail` | Last 16,000 bytes of the combined log |
| `validation_report` | Current bundle report on single-job reads |
| `validation_report_error` | Reason the report could not be attached |
| `artifact_integrity` | Cheap post-completion presence status for completed train jobs |

Job list responses omit the attached validation report but still reconcile
records and include the log tail.

## Queued and launch protocol

Submission writes the queued record, creates the host-global lease, and then
starts a local worker thread. The worker writes a launch specification, starts a
permit-file launcher in a new process group where supported, records process
identity, changes the job to `running`, binds the global lease to the child, and
only then releases the launch permit.

The launcher uses the verified bundle as its working directory and resolves
manifest entries only as relative children of that directory. After the permit
appears, it rereads the manifest, compares its SHA-256 with the project-bound
artifact fingerprint, rejects escapes and symlinks, and rehashes each manifested
file before replacing itself with the job command. The parent also rechecks the
same binding immediately before it writes the permit.

The managed launch specification and child environment also carry the
submission-approved snapshot digest as
`APTUS_AUTHORIZED_MODEL_POLICY_SNAPSHOT_SHA256`. The parent rechecks current
host currency before releasing the permit, and generated entrypoints compare
the environment binding with the plan and manifest before using plan state.
Direct package-free execution has no host-authorized binding and continues to
verify only its embedded frozen snapshot.

This sequence narrows the interval in which a child could start without a
durable identity. Worker-start or lease-persistence failure records `failed` and
releases the lease where possible.

## `verifying` phase

After the child tree exits, the record receives `return_code` and a completion
verification timestamp while state remains active. Reads expose
`phase: verifying` and `cancellable: false`.

For train, the parent then:

1. verifies the runtime-specific run marker, metrics, finite guards, trainable
   scope, data evidence, and export;
2. persists verified pending evidence in the job record;
3. rechecks the bundle fingerprint and current host policy before and
   immediately after computing a new promotion;
4. promotes the bundle report to `measured-run-pass`; and
5. writes a completion attestation.

A zero child exit with failed parent verification becomes `failed`, not
`completed`. A policy change while evidence is pending leaves the report
unpromoted and fails the job rather than accepting evidence under a registry
that did not authorize it.

## Cancellation

On POSIX, Aptus starts the launcher in a process group. Cancellation records
`cancelling`, sends termination to the recorded group, and escalates when
required. It validates process identity to reduce PID-reuse risk.

Cancellation rules:

- terminal records are returned unchanged;
- only the live owning `JobService` can cancel its active worker;
- external owners must receive cancellation through their own service;
- `verifying` is non-cancellable;
- an exited child without an available verifier is marked failed; and
- Aptus never infers success from a dead process alone.

Windows uses the managed direct-process termination path. Portable direct child
execution is fail-closed on Windows.

## Concurrency and ownership

The service combines an in-process lock, a state-root records lock, and a
per-user host-global lease. Managed jobs across different state roots and POSIX
portable bundle entrypoints participate in the same lease.

The lease coordinates Aptus only. It does not reserve CUDA or Apple unified
memory against unrelated software. A foreign live Aptus service retains
ownership of its record and cannot be cancelled by another service instance.

## Reconciliation after interruption

Startup and reads compare recorded owner and process identities with live
processes. Outcomes include:

- preserve a valid live external owner;
- report an orphan child and require operating-system intervention;
- mark an unattached active record failed;
- preserve `cancelling` while a recorded child group remains live; or
- complete a pending train promotion when verified evidence was already
  persisted and still passes, including current host policy.

Reconciliation is intentionally fail-closed. It does not relabel uncertain work
as successful.

## Artifact status after completion

For a completed train job, polling checks only for `.aptus-run.json`,
`final-export.json`, `metrics.json`, and `final/`. It reports either
`verified-at-completion-not-rehashed` or `missing-since-completion`.

This is historical integrity plus current presence. It is not a fresh recursive
hash. The deep file-tree check occurred before completion promotion.

## Resume boundary

There is no supported full-run resume request or state. Every train submission
gets a new run ID and output directory. CUDA pilot phase two is a bounded
checkpoint-continuation test, not a general resume feature. MLX pilot and full
training start from the pinned base and run uninterrupted. Fresh-process MLX
adapter generation does not restore training state, and periodic MLX files are
weight snapshots rather than checkpoints.

## Related documentation

- [Validation states](validation-states.md)
- [API reference](api.md)
- [CLI reference](cli.md)
- [Bundle manifest](bundle-manifest.md)
- [Model-policy snapshot](model-policy-snapshot.md)
- [Error and finding codes](error-codes.md)
- [Execution orchestrator](../architecture/execution-orchestrator.md)
