# Release Gates

> **Status:** Active | **Authority:** Normative release checklist | **Applies to:** Aptus 0.2 | **Audience:** Maintainers and release reviewers | **Last reviewed:** 2026-07-27 | **Review by:** Every release candidate

Version 0.2 remains unreleased until a dated evidence record proves every
applicable gate. Passing repository tests is not target-runtime evidence.

## 1. Source and packaging

- Clean checkout installs on every supported Python version.
- Wheel and source distribution contain the packaged workbench and required
  runtime modules, including the typed method registry and CUDA and MLX program
  resources. Source, wheel, and frozen-sidecar compilation emit identical
  program bytes and manifest hashes.
- Installed-wheel CLI, API, static assets, plan, compile, and static validation
  smoke tests pass outside the source tree.
- The arm64 macOS build embeds the current tested workbench and Python runtime,
  starts outside the repository without Homebrew or `.venv`, enforces its
  per-launch cookie, passes native bridge tests, and verifies every nested and
  outer code signature.
- The native Mac interface uses the macOS 26 treatment when available and the
  tested semantic-material fallback on macOS 15.
- The DMG installs and launches on a clean supported Mac. A public artifact is
  Developer ID signed and notarized. An ad-hoc signature is local evidence only.
- `Aptus.app`, `Aptus.app.zip`, DMG, `SHA256SUMS`, and `COMMIT` identify one
  build. `APTUS_REQUIRE_CLEAN_CHECKOUT=1` rejects dirty release evidence.
- A public build uses `APTUS_CODESIGN_IDENTITY`, `APTUS_NOTARY_PROFILE`, and
  `APTUS_REQUIRE_NOTARIZATION=1`; the app and DMG pass submission, stapling,
  validation, and Gatekeeper assessment.
- Generated bundles install from `requirements.txt` in clean environments.
- Resolved transitive distributions are captured in the environment binding.
- `tools/generate_openapi.py --check` proves the checked API schema is current,
  and `tools/verify_versions.py` proves `aptus.__version__`, the web package and
  lock, `Info.plist`, and OpenAPI report the same version. The desktop build runs
  both commands inside its isolated Python gate.

Run the repeated desktop gate only from a clean checkout:

```bash
tools/repeat_desktop_release_gate.zsh
```

The default is ten consecutive complete `desktop/macos/build.sh` runs. An
explicit positive argument changes the repetition count for diagnosis, but it
does not satisfy the ten-run release requirement. Every iteration enables
`APTUS_REQUIRE_CLEAN_CHECKOUT=1`, captures its full log, verifies the app
signature, verifies the DMG, and records both artifact hashes and timing. A
passing ten-run gate writes:

```text
desktop/macos/dist/RELEASE-GATE.tsv
desktop/macos/dist/release-gate-logs.zip
```

Their hashes are appended to `SHA256SUMS`. The script uses the build's default
ad-hoc signature unless the caller supplies a real `APTUS_CODESIGN_IDENTITY`
and `APTUS_NOTARY_PROFILE`. Ad-hoc repetitions prove build stability only. They
do not prove public notarization.

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
  compatibility record and never presents it as dedicated VRAM. Live available
  host memory constrains MLX planning separately from that record.
- MLX-LM single-device LoRA and QLoRA candidates bind
  `aptus.runtime-contract.v1`, compile with `aptus-memory-mlx-v1`, and remain
  conditional. MLX QLoRA eligibility comes from pinned-model four-bit metadata,
  not a CUDA capability flag.
- PyTorch MPS has no compiler and produces no executable candidate.

## 3. Runtime sequence by training runtime

For each claimed CUDA method and placement:

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

For each claimed MLX-LM LoRA or QLoRA path:

- Dependency validation proves `mlx==0.31.2` and `mlx-lm==0.31.3` in the exact
  configured Apple Silicon interpreter.
- Model-data validation loads the pinned model and tokenizer, tokenizes every
  compiled train and validation row, and verifies MLX four-bit metadata for
  QLoRA.
- Measured preflight runs the bounded real-input compiler slice, records a
  positive MLX peak and adapter delta, completes at least one optimizer update,
  proves exact target binding, and verifies its adapter manifest.
- Pilot runs once without interruption from the pinned base against the bound
  MLX train and validation files. It completes at least two optimizer updates
  and records finite train and validation losses.
- Pilot target binding contains exactly one LoRA A/B pair for every planned
  target in every layer and no other trainable tensor.
- Pilot live unified-memory admission passes, and measured MLX peak plus reserve
  remains within current available headroom.
- Pilot output uses a fresh owned directory. Its marker, metrics, adapter pair,
  and artifact manifest bind exact plan, candidate, model, dataset, action,
  paths, sizes, and hashes.
- A fresh child process loads the pinned base plus emitted adapter and generates
  one to four tokens with a positive MLX peak.
- `pilot-pass` binds the immutable owned pilot metrics and can authorize an
  explicitly confirmed full-duration adapter run.
- Every MLX action reports uninterrupted semantics and
  `resume_supported: false`. Resume arguments fail, and periodic MLX files are
  called weight snapshots rather than resumable checkpoints.

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
- An MLX full run starts from the pinned base, derives its duration from the
  compiled train rows, micro-batch, accumulation, and maximum epochs, and runs
  without interruption.
- MLX full completion verifies finite train and validation losses, at least one
  completed optimizer update, exact target binding, positive memory and adapter
  delta, live headroom, immutable artifacts, fresh-process one-to-four-token
  generation, and `aptus.mlx-final-export.v1` before `measured-run-pass`.

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
- `aptus.job-record.v1` migration, private file modes, symlink rejection, and
  recoverable quarantine pass without one bad record hiding healthy jobs.
- Native sidecar shutdown returns typed success or failure, retains ownership
  for survivors, blocks queued restart, refuses app termination on failure, and
  passes late-fork, zombie, PID-reuse, explicit-retry, and repeated-poll tests.

## 6. API and workbench

- Strict request schemas reject unknown and resume fields.
- Every success route has an explicit response model. Generated
  `docs/reference/openapi.v1.json` matches the application and reports
  `aptus.api.v1`.
- Bootstrap exposes all 11 method descriptors with exact lifecycle and
  selectable values, while its selectable list contains only the four guarded
  executable methods.
- Runtime validation must use cancellable jobs.
- Hardware and local-scan operations respect active-job guards. The workbench
  labels Apple memory as shared, shows live headroom separately, and does not
  present a conditional MLX plan as already proven.
- Train admission, not cached UI state, performs deep authorization.
- The UI exposes five ordered runtime actions, current phase, run output,
  completion attestation, and artifact integrity.
- The method preference control cannot select experimental or research-only
  descriptors. The readiness board displays their blocker and required proof.
- Keyboard, focus, live-region, contrast, and narrow-viewport checks pass against
  the packaged build.
- Example mode is visibly non-executed on every stage.
- The macOS app supplies native dataset and output pickers, reveals generated
  artifacts in Finder, configures an exact MLX Python, runs eligible MLX actions
  locally, and keeps CUDA as an external target-host handoff.
- LM Studio and oMLX integrations remain inference-only and never satisfy a
  training runtime gate.
- Named project revisions are immutable and content hashed. Recovery appends a
  new revision, never restores authorization, and requires fresh validation and
  confirmation.
- The native environment doctor reports measured interpreter evidence and
  performs no installation. `aptus diagnostics` excludes logs, project names,
  dataset or model content, environment values, and unredacted home paths.

## 7. Security and data handling

- Secret scan covers source and generated fixtures.
- Cleartext source, canonical, pilot, archive, cache, CUDA checkpoint, MLX
  weight-snapshot, and export copies are documented and tested for expected
  placement.
- Loopback is the default and non-loopback requires explicit acknowledgment.
- Desktop sessions use an ephemeral loopback port and a random token that is
  absent from URLs, readiness files, application state, JavaScript, and logs.
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
- Hardware pages distinguish CUDA dedicated VRAM from Apple shared unified
  memory and describe the bounded MLX execution path accurately.
- Future evaluation, exporter, provider, cloud, and MCP seams are labeled as
  future work.

## Current result

Partially passed. The
[2026-07-27 Apple Silicon record](evidence/2026-07-27-mlx-lm-acceptance/README.md)
proves two clean, independent MLX-LM workflows through measured preflight,
pilot, fresh-process adapter reload, confirmed full training, final export, and
`measured-run-pass`. No real CUDA pilot or full training evidence has completed
on an external CUDA host. A default ad-hoc Mac build is not public notarization
evidence. Aptus v0.2 remains unreleased until every claimed release gate passes.

## Related documentation

- [Release evidence template](release-evidence-template.md)
- [Current capabilities](../product/current-capabilities.md)
- [Operator checklist](operator-checklist.md)
- [Documentation health](../maintenance/documentation-health.md)
