# State, Storage, and Retention

> **Status:** Active | **Authority:** Operational storage guide | **Applies to:** Aptus 0.2 | **Audience:** Operators and security reviewers | **Last reviewed:** 2026-08-07 | **Review by:** 2026-10-27 or when a persistent path changes

Aptus writes plans, bundles, data copies, runtime evidence, logs, CUDA
checkpoints, MLX adapter weight snapshots, and exports. It does not currently
provide an automated retention or cleanup command. Operators must preserve
active state and remove old material only after resolving exact paths and
retention requirements.

## Persistent locations

| Location | Contents | Mutability | Sensitivity |
| --- | --- | --- | --- |
| Selected state root, default `.aptus-state/` | Named projects and revisions, plans, current pointers, runtime choice, jobs, logs, quarantine, and locks | Mutable control-plane state with immutable revisions | Paths, facts, errors, commands, and runtime evidence |
| Compiled bundle | Cleartext source copy, canonical data, pilot data, v5 plan, frozen model-policy snapshot and copied evaluator, generated code, pins, v3 manifest, and reports | Compiler inputs immutable; named runtime paths mutable | Training data, policy metadata, and operational metadata |
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
  current-project.json
  runtime-config.json
  legacy-project-import.json
  projects/
    project_*/
      project.json
      revisions/revision_*.json
  jobs/
    .jobs.lock
    job_*.json
    job_*.log
  quarantine/
    jobs/
    projects/
```

On POSIX, state directories use mode 0700 and JSON records use mode 0600.
Writers use flush, `fsync`, and atomic replacement. Loaders reject symlinks and
wrong-owner state roots. Project revisions are immutable and content hashed.

Schema-less legacy jobs migrate to `aptus.job-record.v1` with durable
authorization cleared. Legacy plans, the current bundle pointer, and matching
jobs import once into named project history without deleting their source
records. The versioned import receipt makes this idempotent.

Malformed or unsupported job and project state moves into private quarantine
with an `aptus.quarantine-receipt.v1` reason file. Project manifests can repair
their head to the latest safe immutable revision. Quarantine is intended for
inspection and recovery. Do not delete it until the cause and retention needs
are understood.

On POSIX, the shared lease root is
`/tmp/aptus-gpu-lease-<user-identity-hash>/`. Do not delete an active lease to
bypass a job conflict. Let `JobService` reconcile stale ownership.

## Bundle mutability boundary

The compiler manifest rejects unexpected files. Keep virtual environments,
notes, evaluation outputs, and downloaded artifacts outside the bundle unless a
runtime path is explicitly part of the contract.

The allowed mutable roots are:

- `.validation-report.lock`;
- `model-data-evidence.json`, written by the MLX model-data gate and absent for
  CUDA bundles;
- `validation-report.json`;
- `preflight-metrics.json`;
- `pilot-output/`; and
- `runs/`.

Do not edit compiler-managed data, configuration, generated source, or manifest
entries in place. Recompile to a new bundle path.

That immutable set includes `policy_snapshot.py` and the canonical
`policy/model-policy-snapshot.v1.json`. The snapshot bytes, plan digest field,
manifest digest field, and manifested file entry form one frozen integrity
chain. An installed host separately checks current registry currency. If that
check returns `replan_required`, retain the old chain as historical evidence and
create a new plan and bundle; never replace only the snapshot or its digests.

## Protected experiment evidence vault

Repeated target-host experiments need a private evidence layer beyond the
mutable Aptus state root. Select a no-clobber vault outside both Git and the
compiled bundle before the first run. Maintain two checksum-verified copies in
separate failure domains, with at least one off the experiment host, and test
retrieval from the second copy before mutating the only source. Record the
custodian, backup boundary, protected opaque identifier, retention-policy ID,
and provisional retention not-before date. On POSIX, use mode `0700` for vault
and run directories and `0600` for raw files. Any copy leaving the trusted local
filesystem requires equivalent access control, encryption in transit and at
rest, and a recorded encryption-key custodian and recovery procedure.

For each terminal attempt, copy and seal the exact command record, complete
stdout/stderr, Aptus job JSON and log, reports, runtime metrics, manifests,
telemetry, and required artifact bindings. The vault manifest records relative
paths, schemas or media types, SHA-256 digests, byte sizes, timestamps, and the
campaign, comparison-cohort, comparison-cell, attempt-slot,
execution-configuration, experiment-run, Aptus job, and Aptus run identities.
Do not use an API log tail, sanitized summary, mutable source path, or an
experiment tracker as the raw source of truth.

If normal capture or sealing fails, retain an immutable capture-failure receipt
instead. It binds the started slot and run, stable failure code, available-file
inventory, missing-field list, SHA-256, byte size, and recoverable locator. A
cohort with a started slot that has neither a canonical raw manifest nor this
receipt cannot support a qualifying result.

Sealing means writing a versioned canonical manifest in a fresh directory,
flushing all content, atomically creating a no-clobber completion marker, and
anchoring the completed manifest digest in an independent copy or public
packet. Post-seal mutation must fail verification; create a new run identity
instead of replacing or resealing content.

Retention, retrieval, copy-verification, renewal, and claim-withdrawal events
are append-only receipts that reference the immutable raw-manifest digest; do
not place mutable renewal dates or later retrieval results inside the sealed
manifest. At public-packet merge, issue an effective retention receipt that
satisfies the campaign minimum. Verify required copies at the frozen cadence
and after storage, key, or custodian changes. Failed redundancy or retrieval
makes the dependent claim nonqualifying until restored and reviewed. If a
controlling consent, license, or security requirement requires earlier removal,
withdraw the claim before deletion whenever permitted; retention never
overrides that requirement.

Before relying on a public result, restore or retrieve the sealed raw record by
its protected identifier and verify its manifest. Store the retrieval date and
result. A public Git packet may contain sanitized summaries, opaque locators,
retention-policy and receipt bindings, canonical raw-manifest digest and byte
size, two-copy verification, retrieval date/result, and raw-to-public digest
mappings, but never raw logs, raw job state, private identifiers, source data,
weights, checkpoints, adapters, credentials, or byte-exact exception text.
Missing raw retrieval leaves a release-evidence gate incomplete.

The [RTX 3050 CUDA empirical evidence campaign](cuda-empirical-campaign.md)
defines the first planned use of this vault boundary and its campaign-specific
24-month minimum plus pre-expiry renewal or claim-withdrawal review. It does not
create an automated retention or cleanup feature in Aptus or impose that
campaign duration on unrelated projects.

## Before removing anything

1. Confirm no job is `queued`, `running`, or `cancelling`.
2. Record the exact bundle, state root, job ID, run ID, and resolved path.
3. Decide whether legal, research, incident, reproducibility, or release rules
   require retention.
4. Preserve the plan, embedded policy snapshot, manifest, validation report, job
   record, log, run metrics, environment binding, and export manifest needed to
   interpret any retained artifact.
5. Back up material that must survive and verify the backup.
6. Remove only the resolved inactive target. Do not use an unresolved variable,
   broad glob, home directory, repository root, or filesystem root.

Deleting a runtime output can make a cheap historical presence check fail even
though a completion-time attestation remains in the job record. Aptus 0.2 has
no deep historical re-verification or automatic garbage-collection command.
MLX weight snapshots are not resumable checkpoints. Preserve them as evidence
or adapter artifacts, not as restart state.

Use `aptus diagnostics --output aptus-diagnostics.zip` for a bounded support
packet. It reports counts and runtime facts, not logs, project names, dataset or
model content, environment values, or unredacted home paths. Review it before
sharing.

## Recommended retention classes

| Class | Keep at minimum | Suggested policy owner |
| --- | --- | --- |
| Failed planning or static bundle | Input facts, plan or finding, embedded policy snapshot and digest bindings, and source digest | Project owner |
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

- [RTX 3050 CUDA empirical evidence campaign](cuda-empirical-campaign.md)
- [Operator checklist](operator-checklist.md)
- [Security policy](../../SECURITY.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Run states](../reference/run-states.md)
- [Recovery and resume boundary](../guides/resume-recover.md)
