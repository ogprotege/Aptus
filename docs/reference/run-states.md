# Run States

Managed jobs persist one of these states:

| State | Meaning |
|---|---|
| `queued` | Record and global lease exist; child launch has not completed |
| `running` | Runtime action or parent-owned work is active |
| `cancelling` | Termination was requested and is being reconciled |
| `completed` | Action-specific checks passed; train completion includes parent verification |
| `failed` | Launch, child execution, admission-adjacent work, or verification failed |
| `cancelled` | Termination completed without success promotion |

The record can include a more specific phase such as `verifying` while state is
still nonterminal. Consumers should display `phase` when present and use `state`
for lifecycle logic.

## Job record

Common fields include:

- `id` and `job_id`;
- `action` and current `state`;
- `phase` when available;
- `bundle_dir`, `command`, and `log`;
- owner, process, and process-group identities;
- creation, start, and finish timestamps;
- `return_code` and `error`;
- `run_id` and `run_output_dir` for train;
- `prelaunch_capacity_check` for admitted train jobs;
- completion attestation and artifact-integrity status for completed train jobs.

## Cancellation

POSIX cancellation targets the recorded process group and can escalate after a
grace period. The service verifies process identity to reduce PID-reuse risk.
Cancellation does not convert a dead child into a successful job. A train job
cannot be cancelled while its parent is committing verified completion evidence.

## Concurrency

One Aptus GPU action runs for the same user and host across state roots. Managed
jobs and POSIX portable entrypoints share the host-global lease. The lease does
not include unrelated CUDA programs.

## Resume

There is no supported full-run resume state or request field in v0.2. Every new
train job receives a new immutable run ID.
