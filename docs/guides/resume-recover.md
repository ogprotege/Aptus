# Recovery and the Resume Boundary

> **Status:** Active | **Authority:** Operational recovery guide | **Applies to:** Aptus 0.2 | **Audience:** Operators | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when recovery changes

## Full-training resume is unsupported

Aptus v0.2 rejects `resume_from` for full training. The CLI and API do not expose
a supported full-resume operation. Generated MLX entrypoints also reject every
resume argument. Do not edit generated code to bypass this boundary.

A safe general resume contract must bind the exact model or adapter, optimizer,
scheduler, scaler, RNG state per rank, dataloader progress, environment,
distributed topology, plan, candidate, and checkpoint file tree. V0.2 does not
yet attest all of that for arbitrary full-run checkpoints.

## What each pilot proves

CUDA pilot validation runs two bounded phases in fresh processes. Phase one
writes a checkpoint contract. Phase two continues from it. Aptus records
`checkpoint_continuation_observed` when the expected step transition and bound
artifacts pass.

That evidence tests the selected stack's bounded checkpoint continuation. It is
not permission to resume an interrupted full run.

MLX pilot validation starts from the pinned base and trains without interruption
for at least two optimizer updates. A separate fresh process loads the pinned
base plus saved adapter and generates one to four tokens. This tests adapter
reload and inference. It does not restore optimizer, scheduler, random state, or
data position. MLX periodic files are weight snapshots, not checkpoints.

## Recover a managed job

Inspect persisted state:

```bash
aptus jobs
aptus jobs --id JOB_ID
```

Persisted states `queued`, `running`, and `cancelling` are not complete.
`verifying` is a derived phase within `running`, not a separate persisted state.
If an owner process dies, the service reconciles the persisted process identity
and lease before admitting new work. It may complete a crash-interrupted
promotion only when verified pending evidence was already persisted and still
passes the completion transaction.

Never mark a job complete by editing its JSON record.

## Retry after failure

1. Preserve the failed job log and run directory for diagnosis.
2. Correct the underlying environment, capacity, data, or code issue.
3. Re-run the required ordered validation action if its binding changed.
4. Submit a new train job.

A new train job receives a new run ID. Failed and cancelled run directories are
not reused. An interrupted MLX run also starts over from the pinned base.

## Artifact changes

Changing compiler-managed bundle files invalidates the manifest and pilot
binding. Recompile to a new bundle path, then repeat dependency, model-data,
preflight, and pilot actions.

Historical completion records retain a completion-time attestation and perform
only cheap presence checks during polling. V0.2 has no explicit command to
deep-rehash a historical full run. That command is roadmap work.

## Related documentation

- [Execution orchestrator](../architecture/execution-orchestrator.md)
- [Run states](../reference/run-states.md)
- [Operator checklist](../operations/operator-checklist.md)
- [Troubleshooting](troubleshooting.md)
