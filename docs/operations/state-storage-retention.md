# State, Storage, and Retention

> **Status:** Active | **Authority:** Operational storage guide | **Applies to:** Aptus 0.2 | **Audience:** Operators and security reviewers | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when a persistent path changes

Aptus writes plans, bundles, data copies, runtime evidence, logs, CUDA
checkpoints, MLX adapter weight snapshots, and exports. It does not currently
provide an automated retention or cleanup command. Operators must preserve
active state and remove old material only after resolving exact paths and
retention requirements.

## Persistent locations

| Location | Contents | Mutability | Sensitivity |
| --- | --- | --- | --- |
| Selected state root, default `.aptus-state/` | Persisted plans, current-bundle pointer, jobs, logs, and locks | Mutable control-plane state | Paths, errors, commands, and runtime evidence |
| Compiled bundle | Cleartext source copy, canonical data, pilot data, plan, generated code, pins, manifest, and reports | Compiler inputs immutable; named runtime paths mutable | Training data and operational metadata |
| Bundle ZIP | Second copy of all compiler-managed bundle material | Immutable archive | Same sensitivity as bundle inputs |
| `pilot-output/` | Pilot metrics, CUDA continuation checkpoints, MLX adapter artifacts and reload evidence, export evidence, and run contracts | Runtime output | Model, data, and hardware evidence |
| `runs/run_*/` | Unique full-run metrics, CUDA checkpoints or MLX adapter weight snapshots, final export, and manifests | Runtime output; never reused | Model or adapter weights and training evidence |
| Model and package caches | Provider artifacts and resolved dependencies | Managed by external stacks | Model weights, tokens in external config, and supply-chain state |
| Host-global lease root | Per-user lease and lock under the operating-system temporary directory | Ephemeral coordination state | Process and state-root identities |

The API state root contains:

```text
.aptus-state/
  plans/plan_*.json
  current-bundle.json
  jobs/
    .jobs.lock
    job_*.json
    job_*.log
```

On POSIX, the shared lease root is
`/tmp/aptus-gpu-lease-<user-identity-hash>/`. Do not delete an active lease to
bypass a job conflict. Let `JobService` reconcile stale ownership.

## Bundle mutability boundary

The compiler manifest rejects unexpected files. Keep virtual environments,
notes, evaluation outputs, and downloaded artifacts outside the bundle unless a
runtime path is explicitly part of the contract.

The allowed mutable roots are:

- `.validation-report.lock`;
- `validation-report.json`;
- `preflight-metrics.json`;
- `pilot-output/`; and
- `runs/`.

Do not edit compiler-managed data, configuration, generated source, or manifest
entries in place. Recompile to a new bundle path.

## Before removing anything

1. Confirm no job is `queued`, `running`, or `cancelling`.
2. Record the exact bundle, state root, job ID, run ID, and resolved path.
3. Decide whether legal, research, incident, reproducibility, or release rules
   require retention.
4. Preserve the plan, manifest, validation report, job record, log, run metrics,
   environment binding, and export manifest needed to interpret any retained
   artifact.
5. Back up material that must survive and verify the backup.
6. Remove only the resolved inactive target. Do not use an unresolved variable,
   broad glob, home directory, repository root, or filesystem root.

Deleting a runtime output can make a cheap historical presence check fail even
though a completion-time attestation remains in the job record. Aptus 0.2 has
no deep historical re-verification or automatic garbage-collection command.
MLX weight snapshots are not resumable checkpoints. Preserve them as evidence
or adapter artifacts, not as restart state.

## Recommended retention classes

| Class | Keep at minimum | Suggested policy owner |
| --- | --- | --- |
| Failed planning or static bundle | Input facts, plan or finding, and source digest | Project owner |
| Failed runtime action | Bundle identity, validation report, job record, complete log, and relevant runtime outputs | Operator |
| Completed experiment | All bindings, metrics, environment, test protocol, export manifest, and approved final artifact | Experiment owner |
| Release evidence | Immutable evidence packet required by the release gates | Release maintainer |
| Sensitive source and intermediate copies | Only as long as consent, license, security, and reproducibility policy permits | Data owner |

Document the actual retention duration for each project. This page does not set
a universal legal or organizational policy.

## Disk-pressure response

Do not remove files from an active bundle or run. Stop new submissions, inspect
job state, identify completed inactive paths, preserve required evidence, and
then remove the narrowest approved target. If current free disk falls below the
admitted requirement, train submission must fail until capacity is restored.

## Related documentation

- [Operator checklist](operator-checklist.md)
- [Security policy](../../SECURITY.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Run states](../reference/run-states.md)
- [Recovery and resume boundary](../guides/resume-recover.md)
