# Release Gates

> **Status:** Active | **Authority:** Normative release checklist | **Applies to:** Aptus 0.2 | **Audience:** Maintainers and release reviewers | **Last reviewed:** 2026-08-11 | **Review by:** Every release candidate

Version 0.2 remains unreleased until a dated evidence record proves every
applicable gate. Passing repository tests is not target-runtime evidence.

## 1. Source and packaging

- Clean checkout installs on every supported Python version.
- Wheel and source distribution contain the packaged workbench and required
  runtime modules, including the typed method registry and CUDA and MLX program
  resources, `policy_snapshot.py`, and the model-policy snapshot generator.
  Source, wheel, and frozen-sidecar compilation emit identical program bytes,
  canonical snapshot bytes, and manifest hashes.
- Installed-wheel CLI, API, static assets, plan, compile, and static validation
  smoke tests pass outside the source tree.
- A generated bundle runs package-free with copied `plan_contract.py` and
  `policy_snapshot.py`; it validates its embedded frozen snapshot without
  importing installed Aptus.
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
- `tools/generate_openapi.py --check` proves the checked API schema is current.
  `npm run openapi:check` proves the generated TypeScript schema and path map is
  current. `tools/check_client_contracts.py` checks the maintained Swift and
  covered client boundary against OpenAPI. `tools/verify_versions.py` proves
  `aptus.__version__`, the web package and lock, `Info.plist`, and OpenAPI report
  the same version. The desktop build runs every check.
- `npm audit --omit=dev` has no production advisory. Any development-tool
  advisory is recorded with its transitive path, exposure, available fix, and
  release disposition rather than hidden by the production result.
- Every test-suite result used to pass a release gate records the exact command,
  source commit and tree, interpreter or tool version, exit code,
  passed/failed/skipped counts, duration, and captured-output binding. Record
  the SHA-256 and byte size of one byte-exact combined stdout/stderr transcript,
  or of each stream separately, together with a protected non-Git artifact
  identifier or immutable CI URL, retention-policy ID, and append-only effective
  retention receipt and date. Raw transcripts, raw job state, and per-job logs
  remain outside Git; commit only a bounded sanitized summary and its digests.
  Missing transcript capture, digest, or retention record leaves the gate
  unpassed.

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
- Plans use `aptus.training-plan.v6`, bundles use `aptus.bundle.v3`, and every
  bundle carries canonical `aptus.model-policy-snapshot.v1` bytes at
  `policy/model-policy-snapshot.v1.json`.
- Snapshot, plan, manifest, and current-host digest bindings must be lowercase
  64-character hexadecimal text. The snapshot, plan, manifest, and manifested
  file entry agree; installed-host validation separately compares the current
  registry digest.
- Host and portable snapshot evaluators return identical complete decisions for
  exact, near-match, dense, sparse, unknown, and unsorted multi-error subjects.
  Snapshot generation fails closed on a host rule the generic evaluator cannot
  express.
- Host validation covers all six typed snapshot findings:
  `POLICY_SNAPSHOT_MISSING`, `POLICY_SNAPSHOT_JSON_ERROR`,
  `POLICY_SNAPSHOT_CONTRACT`, `POLICY_SNAPSHOT_NONCANONICAL`,
  `POLICY_SNAPSHOT_DIGEST`, and `POLICY_SNAPSHOT_PATH`. Missing, malformed,
  noncanonical, path-tampered, digest-tampered, JSON-null, deeply nested, and
  oversized-integer inputs return controlled invalid results rather than parser
  or primitive-shape exceptions.
- Package-free validation proves frozen-snapshot integrity and saved-decision
  parity but cannot claim host policy currency. Installed-host submission, pilot
  authorization, worker launch, recovery, and the completion verification and
  promotion transaction enforce the current registry. Coherent stale v5 state
  requires replanning; malformed or tampered v5 state remains invalid.
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
  `aptus.runtime-contract.v1`, compile with `aptus-memory-mlx-v2`, and remain
  conditional. MLX QLoRA eligibility comes from pinned-model four-bit metadata,
  not a CUDA capability flag.
- The first MoE row accepts only `qwen3_moe`, `Qwen3MoeForCausalLM`, four-bit
  group-64 defaults, exactly one eight-bit group-64 router-gate override per
  layer, a complete no-shared-expert topology, MLX-LM, QLoRA, `single`, and
  attention-only adapters. Every near match stays unsupported.
- The second policy, `model.qwen2-24l.mlx-qlora`, is a configuration-footprint
  rule rather than an artifact allowlist. It requires the `qwen`, `qwen2`, and
  `Qwen2ForCausalLM` identity, exactly 24 layers, dense topology, and uniform
  four-bit group-size-64 quantization with no module overrides. Its single-device
  MLX-LM QLoRA path targets exactly `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`.
- `policy.qwen2-24l.mlx-qlora.v1` records implementation review of that
  configuration-to-path rule.
  `runtime.qwen2-0.5b.mlx-qlora.2026-07-27` records only the exact pinned July
  27 artifact under training-plan v2 and bundle v2. The August 5
  current-contract evidence at exact source records two fresh v5/v3
  `measured-run-pass` repetitions
  for the same exact pinned artifact at
  `719255153e3fc7e38e83b5ff826d587e5e58bf80`; the original August 5 packet
  remains its historical Phase 6 baseline. None of these records supplies runtime
  evidence for every artifact that matches the configuration footprint.
- Registry-path and compiler eligibility satisfy an implementation boundary,
  not a target-runtime gate. Only an exact bound runtime record can qualify the
  execution it observed.
- The plan and portable validator recompute sparse-layer count and active
  parameters. Base-weight, metadata, staging, and disk terms use the total
  resident parameter count.
- PyTorch MPS has no compiler and produces no executable candidate.

## 3. Runtime sequence by training runtime

When a release claim uses repeated CUDA time, resource, observed completion, or
performance results, every planned slot records a campaign, comparison-cohort,
comparison-cell, and attempt-slot ID plus its role, block, and ordinal. A
started slot additionally records its execution-configuration and
experiment-run IDs and every exact Aptus ID created inside it; a
planned-not-started slot has none of those execution identities. Each started
slot binds either a sealed canonical raw manifest covering complete job state
and logs, stdout/stderr, monotonic event ledger, and telemetry, or an immutable
capture-failure receipt that records its stable failure code, available-file
inventory, missing fields, digest, byte size, and recoverable locator. A missing
normal locator or retrieval result is permitted only when that failure receipt
records why. If neither record can be sealed, the cohort cannot support a
qualifying result.

The public packet keeps three independent axes: slot status (`started` or
`planned-not-started`), native outcome for a started slot (`passed`, `refused`,
`failed`, `cancelled`, `timed-out`, `guard-blocked`, or `unknown`), and evidence
status (`protocol-valid`, `capture-invalid`, or `not-started`). Each row carries
the last action, an allowlisted stable reason code and bounded sanitized reason,
raw-manifest and retrieval bindings when started, and an individual sanitized
evidence-record location. Byte-exact exception text remains protected and is
bound only by digest. Slot IDs are unique, every frozen slot appears exactly
once, and the packet proves `Planned = Started + Planned-not-started`, `Started`
equals the sum of native outcomes, and independently proves
`Started = Protocol-valid + Capture-invalid`. Only native `passed` plus
`protocol-valid` evidence may support a pass. It also reports sanitization
mapping, sampling interval and coverage, and the predeclared aggregation rule.
The [RTX 3050 CUDA empirical
campaign](cuda-empirical-campaign.md) schedules the first such cohort but does
not change these gates or establish a pass by itself.

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
- A claimed Qwen3 MoE path additionally proves exact provider type,
  architecture, canonical quantization layout and digest, complete expert
  topology, derived sparse facts, and attention-only trainable targets at
  model-data, preflight, pilot, reload, and completion boundaries.
- A claimed `model.qwen2-24l.mlx-qlora` path additionally proves the reviewed
  identity, 24-layer dense topology, uniform four-bit group-size-64 layout, and
  exact seven-target census at model-data, preflight, pilot, reload, and
  completion boundaries. The August 5 exact-source refresh satisfies these
  gates for its exact artifact and revision, source and tree, host, runtime,
  dataset, plan, policy snapshot, bundle, and fingerprint. It cannot satisfy
  them for a different artifact or later source state.

## 4. Full-run transaction

- Train submission rejects stale bundle, model, data, environment, hardware,
  pilot, checkpoint, export, and current model-policy bindings.
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
- Pending completion evidence is not promoted after host policy changes. A job
  record binds the host-authorized snapshot digest, and worker launch rechecks
  both that binding and current registry currency.
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
- The maintained browser normalizer rejects v5 plan responses with a missing,
  non-string, uppercase, short, or non-hexadecimal
  `model_policy_snapshot_sha256`.
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
- Saved-plan load, compile, project recovery, and managed job submission map v4,
  v3, v2, schema-less, and coherent stale-policy or stale-snapshot v5 state to
  structured HTTP `409 replan_required`, not generic `400 invalid_request`;
  source bytes are preserved. Host static validation records a typed invalid
  finding instead.
- The UI exposes five ordered runtime actions, current phase, run output,
  completion attestation, and artifact integrity.
- The method preference control cannot select experimental or research-only
  descriptors. The readiness board displays their blocker and required proof.
- An inspected MoE topology renders experts selected per token, total experts,
  sparse layers, checkpoint precision, total resident parameters, active
  parameters, runtime scope, and pilot status. Editing an inspection-derived
  model fact clears stale topology.
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
- Current documents name all six model-policy snapshot findings, bind the
  snapshot digest wherever plan identity is enumerated, and distinguish
  package-free frozen-snapshot integrity from installed-host registry currency.
- Current documents distinguish the reviewed Qwen2 configuration-footprint
  policy, the July historical runtime record, the original August Phase 6
  baseline, and the August current-contract evidence at exact source. They do not
  transfer the current v5/v3 result to a different matching artifact.
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

Partially passed.

| Gate family | Current status | Boundary |
| --- | --- | --- |
| Source, planner, compiler, and package checks | Passed at their recorded commits | Must pass again for the exact release commit |
| Exact Qwen2.5 MLX-LM runtime | Passed twice | Exact recorded artifact, source, M5 Pro host, runtime, data, policy, plan, and bundle only |
| Bounded RTX 3050 CUDA campaign | Complete through Phase 10 | Six exact stable cells, guarded frontier, and endurance/job-control scope only |
| CUDA semantic adapter reload | Open | Structural export does not prove fresh-process semantic reload |
| Model quality and production safety | Open | Execution completion and loss are not quality or safety evidence |
| DDP and conditional LoRA FSDP | Open | Require a separately reviewed multi-GPU campaign |
| Public macOS distribution | Open | Requires Developer ID signing, notarization, stapling, and Gatekeeper assessment |
| Aptus 0.2 release readiness | Open | All claims selected for release must have exact-commit evidence |

There is no Phase 11 in the completed CUDA campaign. Open release gates and any
future CUDA expansion are separately authorized work.

The
[2026-08-05 Apple Silicon current-contract record at exact
source](evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
supplies Phase 6 MLX-LM runtime evidence for the exact pinned Qwen2.5 0.5B
artifact at acceptance source
`719255153e3fc7e38e83b5ff826d587e5e58bf80`, tree
`be99f5664ccb580f2600471f1ae3241a294b1a7e`, and bundle fingerprint
`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`.
Two fresh, clean, independent workflows
used the recorded `aptus.training-plan.v5` and `aptus.bundle.v3` contracts and completed dependency,
model-data, measured preflight, uninterrupted pilot, confirmed full training,
fresh-process reload, final export, parent-owned promotion, and
`measured-run-pass`. Relative to the unchanged [original Phase 6 acceptance
baseline](evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md), only manifested
operator `README.md` and `runbook.md` changed; runtime programs and requirements
remained byte-identical. The record is bound to its exact source and tree,
artifact, host, runtime, dataset, policy snapshot, plan, bundle, and fingerprint.
It does not transfer to another artifact matching
`model.qwen2-24l.mlx-qlora`, qualify CUDA, establish safety, model quality or
performance, promise production throughput, or establish production or release
readiness.

The [2026-08-06 CUDA LoRA single-device
record](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
separately binds one fresh five-job SmolLM2 workflow through
`measured-run-pass` to source
`c12c4d8db0037a2c278a2ad95a0a2cbda4387eed`, the recorded Ubuntu/RTX 3050
host, runtime closure, immutable model revision, synthetic dataset, v5 plan,
v3 bundle, and policy snapshot. It does not establish repeatability or qualify
another CUDA method, placement, host, model, dataset, or environment.

That packet records the source-suite count and duration but no Python
test-transcript digest or retained location. The summary does not satisfy the
transcript-retention requirement above and cannot independently close a
source-test gate; this limitation does not alter the packet's separately bound
one-execution runtime result. See the [complete committed packet and retention
boundary](index.md#complete-ubuntu-cuda-acceptance-packet).

The separate [2026-08-10 Phase 5 repeatability anchor
packet](evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md) records
five of five new predeclared LoRA single-device slots passing the frozen common
stability and integrity contract, including off-host copies and fresh
retrieval. It closes the repeatability gate only for that exact source, host,
environment, model revision, dataset, and configuration. Method-matrix,
remaining target-host, quality, safety, distribution, and release gates remain
open.

The independently reviewed [2026-08-11 Phase 10 campaign
certification](evidence/2026-08-11-cuda-phase10-certification/README.md)
reconciles all 149 frozen measured, exploratory, frontier, and endurance slots:
58 started, 91 planned-not-started, and 47 native-pass plus protocol-valid,
with no replacement runs. It verifies the six listed stable cells, the Phase 8
probe-only frontier, the three-slot 900-update Phase 9 aggregate, and eight
bounded job-control exercises. This closes the campaign evidence workflow but
does not close semantic CUDA adapter reload, DDP, LoRA FSDP, model quality,
production safety, public notarization, or Aptus 0.2 release readiness.

The [2026-07-27 MLX-LM record](evidence/2026-07-27-mlx-lm-acceptance/README.md)
remains historical v2/v2 evidence for the same pinned artifact. The
[2026-07-28 Qwen3 MoE admission record](evidence/2026-07-28-qwen3-moe-admission/README.md)
passed static and dependency gates, then blocked before model loading because
live unified memory was 18.932 GiB below the exact packed-checkpoint-adjusted
requirement. It is safe refusal evidence, not MoE acceptance.

The [2026-07-27 desktop record](evidence/2026-07-27-desktop-release/README.md)
proves 10 of 10 clean local engineering builds at implementation commit
`1038ecdd13103418ef1135e1ced634c10370a961`, including 327 Python, 61 web, and 78
native tests per iteration, packaged launch, signature checks, and DMG
verification. That historical record does not bind a later source head. The
submitted pull request must pass the repeated local gate after its documentation
commit and the GitHub packaging workflow for the exact synthetic merge commit.

The completed CUDA campaign establishes only the exact cells and bounded
outcomes listed in its Phase 10 certification. Remaining CUDA
method/placement coverage and the gates named above are open. The local Mac packages are ad-hoc signed,
not Developer ID signed and notarized public artifacts. Aptus v0.2 remains
unreleased until every claimed release gate passes.

## Related documentation

- [RTX 3050 CUDA empirical evidence campaign](cuda-empirical-campaign.md)
- [CUDA Phase 10 campaign certification](evidence/2026-08-11-cuda-phase10-certification/README.md)
- [Release evidence template](release-evidence-template.md)
- [SmolLM2 CUDA LoRA single-device acceptance](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
- [Phase 6 Qwen2 MLX-LM current-contract evidence at exact source](evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
- [Original Phase 6 Qwen2 MLX-LM acceptance baseline](evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
- [Desktop engineering acceptance](evidence/2026-07-27-desktop-release/README.md)
- [Qwen3 MoE admission evidence](evidence/2026-07-28-qwen3-moe-admission/README.md)
- [Current capabilities](../product/current-capabilities.md)
- [Operator checklist](operator-checklist.md)
- [Documentation health](../maintenance/documentation-health.md)
