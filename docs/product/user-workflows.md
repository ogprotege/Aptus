# User Workflows

> **Status:** Active | **Authority:** Explanatory workflow guide | **Applies to:** Aptus 0.2 | **Audience:** Users and operators | **Last reviewed:** 2026-08-04 | **Review by:** 2026-10-27 or when a workflow changes

## Plan for a known host

1. Pin the model repository to an immutable commit.
2. Verify license and training permission.
3. Profile the local dataset.
4. Scan the target host or enter measured hardware facts.
5. Select MLX-LM for Apple Silicon or Transformers and PEFT for CUDA.
6. State sequence length, effective batch, epochs, objective, and checkpoint
   interval.
7. Compare all candidate statuses, runtime contracts, assumptions, and the
   three separate model-policy records for artifact match, selected candidate
   path, and evidence readiness.
8. Compile the selected plan to a new path.

The current result is an `aptus.training-plan.v6` and an `aptus.bundle.v3`.
Their identities cross-bind the compatibility decision and canonical
`aptus.model-policy-snapshot.v1` digest.

If no candidate passes every hard gate, the workbench still shows the rejected
rows from the typed HTTP 422 response together with the server decision, source,
and nullable inspection receipt. Its required model subject must match the
submitted model ID and immutable revision, followed by the expected source and
receipt, and every row must be rejected with a complete execution and policy
tuple. This is a comparison-only partial view; it cannot be compiled. A null
binding means no emitted path matches the tuple; an exact path match without its
binding is invalid. Provider path-matched receipts require provider-declared
provenance rather than inferred-only observations. A successful recommendation
must structurally equal its complete listed candidate record.

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

Package-free validation checks the bundle's frozen policy snapshot and decision
parity. It cannot establish current host-policy currency. Installed Aptus checks
the current registry during host static validation and managed admission, then
repeats that check at pilot authorization, worker launch, and the completion
verification and promotion transaction. A coherent non-current snapshot
requires replanning; API job submission reports that condition as HTTP 409
`replan_required`.

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
absolute command path without resolving away a virtual-environment symlink.
Finder-launched apps do not inherit your shell environment.
The environment doctor shows probe evidence before selection and gives the exact
external-environment recipe when no interpreter passes. It changes no package.

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

Review the model-policy match separately from the MoE topology. After planning,
select a candidate to inspect only that candidate's explicit path binding. The
evidence-readiness record advances only from a validation report bound to the
same plan ID, candidate ID, and model revision. It distinguishes incomplete or
complete validation evidence from launch admission, and that same exact tuple
gates stage completion and validation or run actions. The optional
`authorization_status` is exactly `current`, `deferred`, or `blocked`: current
requires `authorization_current: true` with no error; deferred or blocked
requires false with a non-empty diagnostic. If the tuple has no non-null member,
admission is not checked. The browser does not infer status from that prose diagnostic or
change the report after a generic training-request failure. A non-current status
is not by itself a stale-policy or replan result. Replan only when Aptus returns
the separate `replan_required` lifecycle error.

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

## Recover a project revision

Project history is different from runtime resume. Aptus records an immutable
revision after planning, compilation, validation, and job submission. Inspect an
older revision, then choose **Recover as new revision** only when its referenced
plan is a current, contract-valid v5 plan. Aptus verifies any referenced local
plan or bundle and creates a new head revision. It does not rewrite history and
always records training authorization as false. Revalidate current evidence and
confirm training again.

Every v4, v3, v2, or schema-less plan, and every coherent v5 plan whose policy
semantics or snapshot digest is stale, remains preserved but cannot be loaded,
compiled, or recovered as executable state. Those operations return HTTP 409
`replan_required`, create no new revision, and leave the saved bytes unchanged.
Create a new deterministic v5 plan from the preserved facts; do not edit the old
schema, decision, snapshot, or digest. Malformed or tampered state is a separate
invalid-input condition, not a replanning condition.

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
