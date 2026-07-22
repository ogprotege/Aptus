# Release Gates

> **Status:** Active | **Authority:** Normative release checklist | **Applies to:** Aptus 0.2 | **Audience:** Maintainers and release reviewers | **Last reviewed:** 2026-07-22 | **Review by:** Every release candidate

Version 0.2 remains unreleased until a dated evidence record proves every
applicable gate. Passing repository tests on a non-CUDA development Mac is not
enough.

## 1. Source and packaging

- Clean checkout installs on every supported Python version.
- Wheel and source distribution contain the packaged workbench and required
  runtime modules, including the typed method registry.
- Installed-wheel CLI, API, static assets, plan, compile, and static validation
  smoke tests pass outside the source tree.
- Generated bundles install from `requirements.txt` in clean environments.
- Resolved transitive distributions are captured in the environment binding.

## 2. Planner and compiler

- The registry's selectable IDs exactly equal the executable `Method` enum.
  Every selectable descriptor is `gated-executable` and has a unique compiler
  ID, an export contract, and supported backend and placement values.
- Every nonselectable descriptor has no compiler or export contract and carries
  an explicit blocker, pilot requirement, and resolvable evidence identity.
- Candidate enumeration covers the fixed 12 planner rows formed from four
  selectable methods and three placements, with an exact unsupported reason for
  every rejected row. Documentation-only research entries never become
  candidates.
- Plan and candidate identity mutation tests pass.
- Memory component arithmetic and point versus upper separation pass.
- Full FP16, full FSDP, quantized FSDP, packing, non-SFT, and wall-time targets
  fail closed as documented.
- Compilation and archive creation remain atomic, deterministic, and no-clobber.
- Every supported source-schema training row is validated and deterministically
  serialized with supported metadata intact, then transformed with the pinned
  tokenizer at runtime.
- Compiled trainer configuration binds the selected descriptor's compiler and
  export identifiers.
- Manifest and path-tamper tests pass.
- Darwin arm64 discovery without CUDA reports one `mps` shared unified-memory
  inventory record, leaves unavailable memory unknown, and produces no
  executable candidate. Discovery must not route through MPS or MLX execution.

## 3. Runtime sequence on CUDA

For each claimed executable method and placement:

- Dependency action passes in a clean isolated environment.
- Model-data action loads the pinned revision, checks parameter count, hidden
  size, layers, context length, supplied intermediate size, and target modules,
  prepares the selected method, enforces its trainable scope, and transforms
  every canonical row.
- The trainable census requires unique names, positive tensor and parameter
  counts, finite initial values, and a stable digest over sorted names, shapes,
  and dtypes. Full training rejects any frozen model tensor. LoRA-based methods
  reject any trainable tensor outside the compiled LoRA scope.
- Measured preflight records a positive peak with exact precision,
  quantization, distribution, world-size, and method-scope census bindings.
- Both pilot phases pass in fresh processes.
- Both pilot phases carry identical trainable census records.
- Adapter censuses contain exactly one LoRA A/B pair for every inspected target
  instance, no unplanned trainable tensor, and strict integer counters. Optimizer
  membership equals the validated trainable identities.
- Pilot checkpoint path, size, hash, optimizer, scheduler, RNG, scaler where
  applicable, and distributed state contracts pass.
- `checkpoint_continuation_observed` is true.
- Current VRAM, host RAM, and disk admission passes under measured pressure.

## 4. Full-run transaction

- Train submission rejects stale bundle, model, data, environment, hardware,
  pilot, checkpoint, and export bindings.
- Each train job receives a unique run ID and refuses an existing output.
- An ungrouped dataset uses `deterministic-exact-row-count-sha256`. A dataset
  with declared groups uses `deterministic-size-aware-group-sha256`; no declared
  group crosses the train and evaluation boundary.
- Split tests cover conflicting or empty group declarations, one indivisible
  group, imbalanced and adversarial group sizes, exact attainable subset sums,
  and ungrouped rows. Evidence records requested and realized evaluation size
  plus the row error instead of claiming that an indivisible grouped dataset
  always reaches the requested fraction.
- The canonical JSONL digest must remain stable across split passes and lazy
  consumption. Distributed ranks must agree on the canonical digest, assignment
  digest, and row counts before training proceeds.
- Single and distributed aggregate exit handling is correct.
- Non-finite loss, zero steps, missing ranks, missing marker, wrong run binding,
  and incomplete export each fail completion.
- Structural safetensors and index checks pass for valid full and adapter output.
- Full-run metrics contain a valid method-scope trainable census and internally
  consistent dataset-split counts, strategy, canonical digest, assignment
  digest, target size, realized fraction, and row error.
- Parent promotion is idempotent.
- A crash after verified pending evidence can reconcile safely.
- Historical reads clearly distinguish completion-time verification from current
  cheap presence status.

## 5. Job control

- One managed job is enforced across state roots for the same user and host.
- Global lease ownership, permissions, stale-owner detection, and PID identity
  checks pass.
- Cancellation terminates the recorded process group and never relabels a dead
  process as successful.
- `cancelling` and parent `verifying` behavior pass interruption tests.
- POSIX portable and managed actions share the per-user host lease, including
  child process-group liveness and restart recovery.
- Windows direct portable child execution fails closed and uses the managed
  service path instead.

## 6. API and workbench

- Strict request schemas reject unknown and resume fields.
- Bootstrap exposes all 11 method descriptors with exact lifecycle and
  selectable values, while its selectable list contains only the four guarded
  executable methods.
- Runtime validation must use cancellable jobs.
- Hardware and local-scan operations respect active-job guards. The workbench
  labels Apple Silicon discovery as inventory, not execution readiness.
- Train admission, not cached UI state, performs deep authorization.
- The UI exposes five ordered runtime actions, current phase, run output,
  completion attestation, and artifact integrity.
- The method preference control cannot select experimental or research-only
  descriptors. The readiness board displays their blocker and required proof.
- Keyboard, focus, live-region, contrast, and narrow-viewport checks pass against
  the packaged build.
- Example mode is visibly non-executed on every stage.

## 7. Security and data handling

- Secret scan covers source and generated fixtures.
- Cleartext source, canonical, pilot, archive, cache, checkpoint, and export
  copies are documented and tested for expected placement.
- Loopback is the default and non-loopback requires explicit acknowledgment.
- Provider data is treated as untrusted and cannot set training permission.
- Generated source and dependency changes receive manual review.

## 8. Documentation consistency

- README, API, CLI, bundle, capability, validation, run-state, security, and
  recovery documents match executable behavior.
- No page describes `requirements.txt` as a transitive lock.
- No page offers full-training resume.
- No page claims full FSDP support.
- No page claims guaranteed fit, quality, or universal optimality.
- Method lifecycle wording matches the runtime registry, and research-catalog
  presence is never described as execution support.
- Dataset-split documentation uses the current strategy identifiers and states
  that declared groups can prevent an exact requested fraction.
- Hardware pages distinguish CUDA execution from Apple Silicon discovery.
- Future evaluation, exporter, provider, cloud, and MCP seams are labeled as
  future work.

## Current result

Not passed. No real CUDA pilot or full training evidence has been completed on
the current development Mac. Aptus v0.2 is not release-ready.

## Related documentation

- [Release evidence template](release-evidence-template.md)
- [Current capabilities](../product/current-capabilities.md)
- [Operator checklist](operator-checklist.md)
- [Documentation health](../maintenance/documentation-health.md)
