# Data and Identity Flow

> **Status:** Active | **Audience:** Contributors, operators, and security reviewers | **Authority:** Explanatory | **Applies to:** Aptus 0.2 | **Owner:** Architecture | **Last reviewed:** 2026-07-29 | **Review by:** 2027-01-27

Aptus binds decisions and runtime evidence to exact content. It uses separate
identities for projects, revisions, source data, candidates, plans, bundles,
environments, hardware, splits, trainable parameters, jobs, runs, and exports. No single digest stands
in for all of them.

## End-to-end flow

```mermaid
flowchart TD
  S["Local source dataset"] -->|"SHA-256 + profile"| DP["Dataset profile"]
  PM["Provider metadata at immutable revision"] --> IR["Inspection receipt"]
  MF["Model facts"] --> P["Planner"]
  IR --> P
  HF["Hardware facts"] --> P
  TF["Target facts"] --> P
  DP --> P
  P -->|"one policy decision + canonical IDs"| C["12 candidate records"]
  C --> TP["Identity-bound training plan"]
  TP -->|"verify source digest"| AC["Atomic compiler"]
  S --> AC
  AC --> DS["Copied source"]
  AC --> CJ["Canonical training JSONL"]
  AC --> PS["Bounded pilot sample"]
  AC --> BM["Bundle manifest and fingerprint"]
  BM --> V["Ordered runtime validation"]
  CJ --> V
  PS --> V
  V -->|"environment + hardware + census"| PP["Pilot-pass"]
  PP --> TA["Deep train admission"]
  TA --> R["Unique run ID"]
  CJ -->|"canonical + assignment digests"| R
  R --> FE["Final export manifest"]
  FE --> PV["Parent verification"]
  PV --> MR["Measured-run-pass"]
```

## Fact provenance

Each important fact carries one of five provenance kinds:

- `measured`: observed from a local file, device, or runtime check;
- `provider-declared`: returned by a bounded repository endpoint;
- `user-attested`: entered and affirmed by the operator;
- `inferred`: produced by a versioned Aptus rule;
- `unknown`: no defensible value is available.

Provider-declared model fields do not become permission facts. Manual hardware
values do not become target-host measurements. An inferred model family does
not replace the raw provider model type or architecture evidence.

For a sparse model, the v4 model payload binds exact provider type,
architecture, checkpoint precision, expert count, experts per token, expert
width, sparse cadence, dense-only layer indices, and optional shared-expert
width. It also binds backend-derived active parameters and sparse-layer count.
The user-attested total parameter count remains a separate resident-weight fact.
Changing any of these values changes candidate and plan identity.

A provider inspection resolves one immutable revision and can emit an
`aptus.model-inspection-receipt.v1`. Its `subject_facts_sha256` binds only the
facts evaluated by `aptus.model-compatibility.v2`. Its separate
`observed_facts_sha256` binds every provider-declared or inferred model fact
actually carried into planning. Parameter count and training permission remain
user-attested and are excluded from the receipt. A plan without a receipt says
`user-attested`; a plan with a valid receipt says `provider-inspection`.
Receipt entries use only those two inspection kinds. They cover every non-null
compatibility subject field and include at least one provider-declared subject
observation. This prevents unrelated or wholly user-attested facts from
establishing provider-inspection provenance.

## Source dataset identity

Profiling resolves the source path and records its SHA-256, size, format,
schemas, counts, sequence statistics, sample indices, warnings, and provenance.
The sample limit bounds length statistics. It does not define the compiled
training set.

At compilation, Aptus copies the source to `data/dataset.<suffix>` and hashes
the copy. A mismatch with the profiled digest aborts compilation. It then
validates every supported source row and writes deterministic, key-sorted JSONL
to `data/training.jsonl`. It separately writes a bounded pilot pressure set to
`data/pilot-sample.jsonl`.

The portable plan refers to the copied source path inside the bundle while
retaining the original content digest. This makes the bundle relocatable
without weakening data identity.

## Candidate and plan identities

A candidate ID begins with `cand_` and hashes a canonical semantic payload. The
payload includes:

- normalized model, dataset, hardware, and target facts;
- method, distribution, precision, quantization, and device placement;
- exact batch arithmetic;
- adapter and target-module settings;
- exact sparse topology, total resident parameters, and derived routed activity
  when the model is MoE;
- status and feasibility;
- the shared policy decision ID and, only for an exact path match, the
  `aptus.model-policy-binding.v1` object;
- memory components, upper bounds, formula version, host RAM, disk, checkpoint,
  export, and reserve terms.

A plan ID begins with `plan_`. It binds the plan and formula schema versions,
normalized facts, the complete policy decision and source, the optional
inspection receipt, the sorted canonical evidence records, the ordered
candidate IDs, and the recommended candidate ID. IDs are content identities,
not editable labels. Changing a bound semantic fact without recomputing
identity makes validation fail. Known evidence records must also match their
code-owned canonical contents.

All candidates link to the same decision, including candidates with no
registered execution path. Only the candidate whose method, placement, target
modules, and runtime contract match the emitted path has a non-null policy
binding. Loading, compilation, recovery, and validation also compare a v4
decision with the current registry. An obsolete policy version returns
`replan_required` only after the complete saved identity chain validates. V3,
v2, and schema-less plans receive the same fail-closed
result and remain unchanged on disk.

The historical coherence pass uses the persisted decision and one internally
consistent adapter-target set, not the current mutable family catalog. The
ordinary current-plan pass still requires exact current catalog targets. An
unsupported adapter for an unregistered family carries no targets and can
truthfully record zero checkpoint retention because it has no trainable adapter
parameters.

## Project and revision identities

A named project uses schema `aptus.project.v1` and keeps an ordered list of
immutable revision IDs. Each `aptus.project-revision.v1` record contains its
project and parent identities, ordinal, reason, available fact and plan
snapshots, selected candidate, bundle reference, durable validation summary,
and job IDs. `content_sha256` binds the revision payload.

Project reads and writes share an in-process lock and an operating-system file
lock. Revision publication uses a durable transaction receipt, then writes the
content-hashed revision, advances the manifest, updates the selected-project
pointer, and removes the receipt. Each write uses atomic replacement and a
directory sync. After interruption, Aptus can finish a receipt-bound revision
or adopt a unique orphan chain that extends the indexed parent and ordinal. It
rejects and quarantines corrupt, ambiguous, or previously rejected orphans.
They never become history merely because a revision-shaped file exists.

The current-project pointer records selection intent separately from project
recency. Recovery may advance the selected project's revision, but it cannot
replace a later explicit selection of another project with an older interrupted
transaction. This ordering prevents crash repair from changing which project
the operator chose after that transaction began.

Recovery does not restore a snapshot in place. It validates any referenced plan
or bundle, then appends a new revision whose reason identifies the source. Both
ordinary persistence and recovery store `training_authorization.current` as
false. Current capacity and deep authorization remain train-submission facts.

At first startup after this change, Aptus imports legacy plans, the current
bundle pointer, and matching jobs into named project history. The import receipt
is versioned and idempotent. Source records remain in place.

## Bundle integrity

`bundle-manifest.json` uses schema `aptus.bundle.v3`. It binds the plan digest,
plan ID, selected candidate ID, formula version, compiler version, direct stack
versions, entrypoints, validation levels, and every compiler-managed file by
relative path, size, and SHA-256. It also names
`policy/model-policy-snapshot.v1.json` and binds its digest. That digest must
equal `model_policy_snapshot_sha256` in the v5 plan and the snapshot file's
manifested SHA-256.

The host registry serializes `aptus.model-policy-snapshot.v1` canonically and
deterministically. The bundle carries those exact bytes plus a generic evaluator
that has no installed-Aptus dependency. Host and portable evaluation must
produce the same decision for the same compatibility subject.

The bundle fingerprint is the SHA-256 of that manifest when it exists. Bundle
validation rejects symlink roots, symlink entries, unsafe paths, missing files,
changed files, duplicate entries, and unexpected unmanifested inputs.

A compiled project revision stores that manifest fingerprint as a first-class
artifact identity. If a ZIP exists, the revision also stores its SHA-256 and
exact byte size. Recovery verifies the saved plan snapshot, selected candidate,
bundle path, manifest fingerprint, and ZIP identity before it appends a new
revision. Validation, job submission, and bootstrap require the current
revision's exact plan, candidate, bundle path, and manifest fingerprint. A
matching path or matching plan ID alone is insufficient.

These mutable paths are intentionally outside the compiler file list:

- `.validation-report.lock`;
- `model-data-evidence.json`;
- `validation-report.json`;
- `preflight-metrics.json`;
- `pilot-output/`;
- `runs/`.

Their evidence is bound by runtime schemas, IDs, paths, sizes, and hashes rather
than by pretending runtime outputs are immutable compiler inputs.

## Environment and hardware bindings

Dependency validation records Python, platform, exact direct constraint
versions, and the installed runtime distribution closure. `requirements.txt`
is an exact direct constraint set, not a complete transitive lock.

Measured validation records runtime-specific device and memory evidence. CUDA
binds participating device identities. MLX binds live unified-memory admission.
Train admission compares current capacity with pilot-bound evidence. A
historical pass cannot reserve current resources.

## Trainable-parameter identity

The trainable census binds the selected method to positive, finite tensor and
parameter counts plus a digest over sorted names, shapes, and dtypes. The digest
does not expose parameter values.

Full training requires every model tensor to be trainable. LoRA-based methods
require one complete A/B pair for every inspected target-module instance and no
other trainable tensor. Optimizer parameter identities must exactly equal the
validated trainable identities. Both CUDA pilot phases must report the same
census. MLX binds one A/B pair to every planned target in every layer and proves
a positive adapter delta in its uninterrupted run.

## Split identity and mutation detection

The CUDA full trainer computes one deterministic split over the complete canonical
JSONL. It records:

- the canonical JSONL digest;
- the split strategy;
- an assignment digest;
- total, train, evaluation, group, and split-unit counts;
- target and realized evaluation sizes and row error.

Rows sharing a declared `split_group` remain atomic. Ungrouped rows are
independent split units. The grouped solver reaches the exact target when the
group sizes and available ungrouped rows permit it, then chooses the closest
feasible size otherwise.

The trainer hashes the canonical file during split passes, checks file identity
before and after each lazy read, and requires distributed ranks to agree on the
canonical and assignment digests and counts. A mutation aborts the run.

MLX compilation instead creates disjoint train and validation files, pads only
within each split, and binds source and compiled counts in
`aptus.mlx-split.v1`. The current MLX contract does not claim group-aware subset
selection or an exact requested evaluation fraction.

## Job, run, and export identities

Managed actions receive persisted `job_` identities. Full training also
receives a unique `run_` identity and a new `bundle/runs/run_*/` directory. A
child cannot reuse an existing run path.

The child writes metrics and final-export evidence. The parent verifies
runtime-specific process success, plan/candidate/run bindings, finite metrics,
positive updates, trainable scope, data evidence, and immutable safetensors.
CUDA also binds distribution ranks and its split contract. MLX also binds fresh
adapter generation and `resume_supported: false`. Only then does the parent
promote the report to `measured-run-pass`.

Persisted jobs carry `aptus.job-record.v1`. A record without a schema version
migrates to that shape, with authorization cleared. Unsupported, malformed, or
symlinked records are quarantined instead of silently accepted.

This attests the exact run and structural file tree. It does not establish
quality, safety, inference parity, or deployment fitness.

## Data-copy and trust boundary

The source copy, canonical JSONL, pilot sample, MLX split files, ZIP, model
cache, validation reports, job logs, CUDA checkpoints, MLX weight snapshots,
metrics, tokenizer files, and exports can all contain sensitive material.
Backup and synchronization software can create more
copies. Bundle integrity detects changes. It does not encrypt or govern access
to those files.

Receipt, decision, candidate, plan, and bundle hashes are tamper-evident content
bindings. They are not authenticated signatures and do not prove who created an
artifact. Aptus therefore trusts its local process and client boundary while
still rejecting mismatched content.

Keep the API on loopback. Treat anyone who can call it as able to use the Aptus
process user's file and compute permissions.

## Related documentation

- [Facts and provenance](../methodology/facts-and-provenance.md)
- [Plan schema](../reference/plan-schema.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Validation states](../reference/validation-states.md)
- [Security boundaries](security-boundaries.md)
