# Release Gates

Version 0.2 remains unreleased until a dated evidence record proves every
applicable gate. Passing repository tests on a non-CUDA development Mac is not
enough.

## 1. Source and packaging

- Clean checkout installs on every supported Python version.
- Wheel and source distribution contain the packaged workbench and required
  runtime modules.
- Installed-wheel CLI, API, static assets, plan, compile, and static validation
  smoke tests pass outside the source tree.
- Generated bundles install from `requirements.txt` in clean environments.
- Resolved transitive distributions are captured in the environment binding.

## 2. Planner and compiler

- Candidate enumeration covers every catalog row and exact unsupported reason.
- Plan and candidate identity mutation tests pass.
- Memory component arithmetic and point versus upper separation pass.
- Full FP16, full FSDP, quantized FSDP, packing, non-SFT, and wall-time targets
  fail closed as documented.
- Compilation and archive creation remain atomic, deterministic, and no-clobber.
- Every supported source-schema training row is validated and deterministically
  serialized, then transformed with the pinned tokenizer at runtime.
- Manifest and path-tamper tests pass.

## 3. Runtime sequence on CUDA

For each claimed executable method and placement:

- Dependency action passes in a clean isolated environment.
- Model-data action loads the pinned revision, checks parameter count, hidden
  size, layers, context length, supplied intermediate size, and target modules,
  and transforms every canonical row.
- Measured preflight records a positive peak with exact precision,
  quantization, distribution, and world-size bindings.
- Both pilot phases pass in fresh processes.
- Pilot checkpoint path, size, hash, optimizer, scheduler, RNG, scaler where
  applicable, and distributed state contracts pass.
- `checkpoint_continuation_observed` is true.
- Current VRAM, host RAM, and disk admission passes under measured pressure.

## 4. Full-run transaction

- Train submission rejects stale bundle, model, data, environment, hardware,
  pilot, checkpoint, and export bindings.
- Each train job receives a unique run ID and refuses an existing output.
- Single and distributed aggregate exit handling is correct.
- Non-finite loss, zero steps, missing ranks, missing marker, wrong run binding,
  and incomplete export each fail completion.
- Structural safetensors and index checks pass for valid full and adapter output.
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
- Runtime validation must use cancellable jobs.
- Hardware and local-scan operations respect active-job guards.
- Train admission, not cached UI state, performs deep authorization.
- The UI exposes five ordered runtime actions, current phase, run output,
  completion attestation, and artifact integrity.
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
- Future evaluation, exporter, provider, cloud, and MCP seams are labeled as
  future work.

## Current result

Not passed. No real CUDA pilot or full training evidence has been completed on
the current development Mac. Aptus v0.2 is not release-ready.
