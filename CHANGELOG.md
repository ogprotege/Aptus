# Changelog

> **Status:** Active | **Authority:** Release record | **Applies to:** Aptus 0.2 | **Audience:** Users and maintainers | **Last reviewed:** 2026-08-11 | **Review by:** Every release

All notable changes are recorded here.

## 0.2.0 - Unreleased

### Added

- A standing mission-sustain checklist on every pull request (M9). It does not
  add a method or a measured ladder. Path Beta “current HEAD” wording is
  replaced with the recorded M4 source.
- An optional exact-match evaluation contract (`aptus.evaluation-contract.v1`)
  and result (`aptus.evaluation-result.v1`). Operators attach a gold JSONL
  digest, threshold, and optional export digest, then score supplied
  predictions with `aptus eval-contract` / `aptus eval` or
  `POST /api/v1/evaluations*`. Training finished, train loss, and
  `measured-run-pass` are not an evaluation pass. This is not general quality,
  safety, or human preference.
- A second CUDA model on the Path Beta runtime:
  `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc151…` LoRA single reached
  `measured-run-pass` on Sherminator. Evidence:
  `docs/operations/evidence/2026-08-13-path-beta-360m-lora-m7a/`.
- A Path Beta fresh-process CUDA PEFT adapter reload (1–4 tokens) bound to
  compile `36bef48d6ca3c0b11bf39da823ae4bc24f4c94fb` and fingerprint
  `cf7858e5…`. Evidence:
  `docs/operations/evidence/2026-08-13-path-beta-cuda-reload-m7c/`. CUDA
  `measured-run-pass` still does not require that gate.
- A Developer ID signed, notarized, and stapled arm64 Mac app and DMG bound to
  source `edc6cfdec48daeb17af8cae7dbb9fde0d8112a81`. Gatekeeper assessed both
  artifacts as Notarized Developer ID. The identity, hashes, and notary IDs
  live in
  `docs/operations/evidence/2026-08-13-desktop-public-release/`. Default and CI
  builds remain ad-hoc. This is not Aptus 0.2 product release.
- A reviewed CUDA Phase 10 certification that closes the bounded RTX 3050
  campaign without new training, replacement runs, or external-resource
  acquisition. It reconciles 149 frozen slots to 58 started, 91
  planned-not-started, and 47 native-pass plus protocol-valid results;
  recomputes the six listed stable-cell summaries, Phase 8 probe-only frontier,
  and Phase 9 endurance aggregate; and independently verifies 13 prior public
  packets and 68 selected protected artifacts. Release readiness, model
  quality, semantic CUDA adapter reload, production safety, distributed CUDA,
  remain open. Public notarization of one later arm64 desktop identity is
  recorded separately in the 2026-08-13 packet.
- A reviewed CUDA Phase 6 Full confirmatory-stability packet at exact merged
  source `2bc4d9a38f88cb0be1087b6e35a329587d1942bf`: all five
  predeclared Full slots passed with protocol-valid evidence, exactly 128
  optimizer steps, complete telemetry, verified off-host copies, and verified
  fresh retrievals. Duration and peak-device-memory stability passed the frozen
  thresholds, establishing one stable exact-host Full cell and authorizing the
  bounded Phase 7 campaign procedure. The earlier remediation matrix and the
  intervening nonqualifying source-defect diagnostic remain immutable and do
  not enter the successful cohort aggregate.
- A reviewed CUDA Phase 6 remediation packet at source
  `af91225e1a2ba601a0b6dacd2366619e550babda`: Full passed all three
  exploratory attempts and was promoted, then produced one pass and four
  unrelated-GPU-activity safety cancellations across its five confirmatory
  attempts. All 12 conditioning and matrix artifacts were sealed, copied off
  host, and restored with verification. That cohort established no stable
  method. It remains immutable historical evidence and is superseded only for
  current operational status by the separate successful Full cohort above.
- A reviewed CUDA Phase 5 repeatability packet: five of five predeclared
  SmolLM2 LoRA single-device slots passed the frozen stability and integrity
  contract at source `3bfec547d4cffedbaf049426d9713f1ccc25b5a2`, with verified
  off-host copies and fresh retrieval. The result establishes only the exact
  frozen anchor and Phase 6 eligibility, not broad CUDA support or release
  readiness.
- Aptus for Mac, a native AppKit and WebKit application with automatic bundled
  backend lifecycle, private session authentication, native path pickers,
  Finder actions, startup recovery, app packaging, and CUDA-host handoff.
- `aptus.facts.v3`, `aptus.training-plan.v6`, and `aptus.bundle.v3`, plus the
  retained versioned runtime, validation, and job contracts.
- Persisted `aptus.model-compatibility.v2` decisions, explicit
  `provider-inspection` or `user-attested` decision sources,
  `aptus.model-inspection-receipt.v1` provenance, and exact-path
  `aptus.model-policy-binding.v1` records. Provider receipts bind compatibility
  facts separately from the broader observed planning facts.
- Exact Qwen3 MoE compatibility for inspected four-bit `qwen3_moe` checkpoints
  using `Qwen3MoeForCausalLM`. The implemented conditional planner slice is
  single-device MLX-LM QLoRA with attention-only adapters and mandatory pilot
  evidence.
- A second registry-driven policy for the reviewed 24-layer dense Qwen2
  four-bit group-64 configuration footprint. It permits only single-device
  MLX-LM QLoRA with q/k/v/o and gate/up/down adapters. Two fresh exact
  pinned-artifact repetitions reached `measured-run-pass` at source
  `719255153e3fc7e38e83b5ff826d587e5e58bf80` and bundle fingerprint
  `ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`.
  Relative to the unchanged original acceptance baseline, only manifested
  `README.md` and `runbook.md` changed; runtime programs and requirements stayed
  byte-identical. The refreshed record does not admit every matching artifact
  or establish CUDA, safety, quality, performance, production, or release
  readiness claims.
- Provider-declared MoE topology facts, backend-derived active-parameter and
  sparse-layer counts, strict API and CLI inputs, and a workbench expert-routing
  rail. Resident-weight estimates always use the user-attested total parameter
  count.
- Exact required, available, and shortfall bytes in MLX unified-memory
  admission failures, plus a bound Qwen3 30B-A3B admission and performance
  evidence record. The recorded 30B attempt stopped before model loading.
- Full, LoRA, int8-LoRA, and QLoRA candidate enumeration with explicit
  distribution feasibility.
- A typed 11-descriptor method registry. Four gated executable methods are
  selectable. DoRA, BitFit, AdaLoRA, and ShareLoRA are experimental. LoReFT,
  AFLoRA, and BiLoRA are research-only.
- Bootstrap API and workbench readiness metadata for method lifecycle, evidence,
  blockers, required pilots, compiler contracts, and export contracts.
- Point and upper memory estimates with evidence records.
- Atomic no-clobber bundle compilation and deterministic archives.
- Portable dependency, model-data, measured preflight, pilot, and full-run
  entrypoints.
- Deterministic canonical `aptus.model-policy-snapshot.v1` artifacts bound into
  every v6 plan and v3 bundle, plus a generic package-free evaluator that
  reproduces host compatibility decisions from the frozen snapshot.
- Local FastAPI service, React workbench, CLI, persisted jobs, cancellation, and
  a per-user host-global Aptus execution lease.
- Immutable full-run output IDs and parent-owned completion verification.
- Runtime-specific pilot and measured-capacity admission contracts. CUDA proves
  checkpoint continuation. MLX proves exact target binding, at least two
  optimizer updates with finite losses, changed adapter weights, positive peak
  and delta measurements, live unified-memory headroom, immutable artifacts,
  and a fresh-process adapter reload that generates one to four tokens.
- Opt-in CUDA campaign Phase 2A source tooling for canonical evidence records,
  separately sealed Phase 4 authority, exact admission and post-gate identity
  activation, all seven native outcomes, exact-argv and managed-sequence
  capture, Linux/NVIDIA telemetry, watchdog-owned cancellation, runtime
  journals, retained activation provenance, semantic no-clobber sealing,
  custody receipts, allowlisted recovery sanitization, read-only eligibility,
  and two-pass inode-pinned publication with verified rollback. The [Phase 2A
  tooling contract](docs/operations/cuda-campaign-phase2-tooling.md) records the
  closed adversarial findings and Phase 2B preconditions now satisfied by the
  dated supplement below. This is source and
  contract evidence, not a product-capability claim or target-runtime result.
- A dated, independently reviewed [Phase 2B sanitized recovery
  supplement](docs/operations/evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md)
  produced only from protected Phase 0 copies at merged Phase 2A source
  `f6a58612263ccd1b7284ffa9f5460631ba64c2e1`. All 40 frozen logical rows have
  dispositions: 39 are recovered and digest-matching, while the raw model-file
  manifest remains `not-found`; the separate original Python test transcript
  also remains `not-found`. Two-copy verification, full off-host retrieval,
  retention, traceability and privacy review, finalized-byte verification,
  fail-closed eligibility, and two-pass publication passed. No Linux
  connection, Ubuntu mutation, model workload, or new empirical result
  occurred. At that Phase 2B publication boundary, Phase 3 controls were pending.
- Phase 3 explicit complete-candidate selection across domain, API, CLI, and
  workbench, producing a new plan identity and rejecting stale, mutated,
  rejected, unknown, or already-selected candidates. V6 plans now bind exact
  optimizer-step, split/training/data-order seed, micro-batch, and accumulation
  controls. CUDA bundles bind the same values through trainer configuration,
  checkpoints, runtime completion evidence, separate training/evaluation
  consumption and exact token counters, and per-step monotonic progress timing.
  This implementation changed no Ubuntu host and produced no empirical result.
- Uninterrupted full-duration MLX LoRA and QLoRA execution from the pinned base
  model after `pilot-pass`, with duration derived from compiled training rows,
  batch size, accumulation, and maximum epochs.
- Method-specific trainable-parameter census checks before optimizer creation,
  with strict typed counts, finite values, one LoRA A/B pair per inspected target
  instance, exact optimizer membership, and a stable name-shape-dtype descriptor
  digest in measured evidence.
- Deterministic full-dataset splitting with optional `split_group` isolation,
  exact subset selection when a grouped target is attainable, closest feasible
  grouped selection otherwise, canonical and assignment digests, target and
  realized evaluation sizes, cross-rank agreement, and mutation detection during
  split and consumption.
- Fail-closed Apple Silicon discovery that records measured shared unified
  memory, plus generated MLX-LM validation, pilot, full-run, and adapter-reload
  entrypoints for supported LoRA and QLoRA plans.
- Structural safetensors export file-tree verification and environment bindings.
- A governed reviewed-corpus contract, Apple Silicon pilot matrix, and complete
  reconciliation ledgers for the retained Reference packet, removed TO-REVIEW
  staging files, and reviewed then removed EXAMPLE intake.
- A task-oriented documentation system with section indexes, first-run and
  method-selection guides, code-derived reference contracts, operator and
  contributor procedures, lifecycle metadata, maintenance policy, debt and
  health records, support guidance, and repository workflow templates.
- Documentation checks for links and anchors, navigation reachability, review
  metadata, CLI and API surface coverage, method-catalog overlap, stale
  contracts, and sealed-bundle environment safety.
- Native arm64 macOS 26 CI packaging that uploads the verified Aptus DMG,
  permissions-preserving application ZIP, and SHA-256 checksums for every pull
  request and push to `main`.
- Named local projects with immutable, content-hashed revisions for facts,
  plans, compiled bundles, validations, and jobs. Recovery creates a new
  revision and never restores training authorization.
- Versioned private job and project storage, legacy-state import, recoverable
  quarantine for corrupt or unsupported records, and atomic mode-0600 JSON
  writes under mode-0700 directories.
- Explicit Pydantic response contracts, API contract identity `aptus.api.v1`,
  and a checked generated OpenAPI artifact.
- Read-only `aptus doctor` runtime readiness reports and privacy-bounded
  `aptus diagnostics` support archives.
- A native MLX environment doctor that reports exact interpreter evidence and
  never installs or changes packages.
- Developer ID and notarization support in the Mac build, including app and DMG
  submission, stapling, assessment, source markers, and checksums when the
  required identity and keychain profile are supplied.

### Changed

- Local-door hardening: the audit scanner reports Hugging Face `hf_` tokens;
  provider inspection accepts only repository model IDs, disables HTTP proxies,
  and stays on `https://huggingface.co`; `aptus serve` prints the workbench
  origin without a session token; the GPU lease lives under `XDG_RUNTIME_DIR`
  or `~/.aptus/run` instead of world-writable `/tmp`; and `create_app()`
  requires `session_token` unless tests or schema generation pass
  `allow_unauthenticated=True`. Bundles compiled before this change still
  embed the old `/tmp` lease path and must be recompiled to share the new
  host lease.
- The Phase 5 workbench now presents the server-owned v2 decision as separate
  artifact-match, selected-path, and evidence-readiness records. Strict ingress
  correlates the model subject in successful plans and typed 422 no-feasible
  responses with the submitted artifact, then verifies source, receipt, complete
  candidate tuple, exact path binding, provider-declared provenance for a
  provider path match, and full structural equality between the recommendation
  and its listed candidate. Validation evidence, stage completion, and runtime
  actions use a report only when plan, candidate, and model revision match.
  Validation completeness remains separate from the optional typed
  `authorization_status` vocabulary `current`, `deferred`, or `blocked`; the
  companion boolean and diagnostic must agree, and a tuple with no non-null
  member means not checked. The browser neither derives status from diagnostic prose
  nor rewrites a report after a generic training-request failure. The unused
  legacy browser compatibility projection was removed.
- Provider inspection, sparse candidate admission, and API execution-path
  validation now consume one host-side model compatibility registry. Runtime
  contracts remain derived from the method registry. The API remains
  `aptus.api.v1`, facts remain `aptus.facts.v3`, and candidate runtime contracts
  remain `aptus.runtime-contract.v1`; current plans are v6 and bundles are v3.
- Package-free bundle programs validate frozen-snapshot integrity and decision
  parity. Installed-host validation, job admission, pilot authorization, worker
  launch, and the completion verification and promotion transaction separately
  enforce the current registry. A coherent stale-policy plan is preserved and
  returns `replan_required`; API load, compile, recovery, and job submission map
  that condition to HTTP 409.
- Contract readers now normalize their covered non-object and resource-hostile
  inputs as controlled invalid input instead of leaking parser or traversal
  exceptions. Installed-host validation covers plan, manifest, trainer, and
  policy-snapshot documents; package-free validation covers plan, manifest, and
  policy-snapshot documents. CUDA entrypoints validate the plan before binding
  devices.
- Conditional API claims must match a path registered for their model family,
  and sparse model-type or architecture markers cannot fall through as a dense
  family when provider topology is missing.
- Model-inspection compatibility now binds known runtime, compute-backend,
  method, distribution, and adapter-profile IDs, rejects unregistered method
  tuples and non-adapter/profile contradictions, and describes a matching
  artifact as eligible for the reviewed pilot path rather than claiming runtime
  support.
- The workflow rail announces completed and failed-run stage states to
  assistive technology, with dedicated accessibility tests.
- Workbench micro-interactions use shared 150 ms motion tokens on buttons,
  inputs, and selectable controls, neutralized under `prefers-reduced-motion`.
- The Aptus mark is now a single graphite letterform whose crossbar is the
  teal fit line, in the web workbench, the desktop resource, and the
  build-rendered Dock icon.
- Generated dependencies are now in `requirements.txt` as exact direct pins.
  This file is not described as a transitive lock.
- Model-data validation transforms and checks every canonical training row.
- Method readiness is now separate from selectability. Documentation-only
  research entries do not enter the planner.
- Runtime execution follows five ordered actions: dependency, model-data,
  preflight, pilot, and train.
- Full training uses `python run.py --confirm-full-train` for portable bundles.
- Full-parameter FSDP is fail-closed. LoRA FSDP is conditional.
- A successful process exit is no longer enough to mark a training job complete.
  The parent verifies and promotes pending evidence.
- Profiling and generated training code now use the same deterministic dataset
  schema precedence for rows that contain fields from more than one supported
  shape.
- Installed CLI help and generated bundle reports now explain defaults,
  side effects, evidence boundaries, ordered validation, external environment
  setup, and the fail-closed recovery contract.
- The CLI reference now exposes a structured parser contract checked against
  every live command, argument, choice, and non-suppressed default. Generated
  operator-document tests compile every executable runtime, method, and
  placement row and verify ordered commands, filenames, platform notes,
  successor links, and evidence boundaries.
- The web lockfile refresh removes the OpenAPI generator development advisories
  without changing declared dependency ranges or generated OpenAPI output.
- The native and web interfaces now form one product surface. Native navigation
  owns Home, Workbench, Machine, and Models. The inline React workbench owns the
  Facts, Compare, Compile, Validate, and Run workflow plus project history.
- Generated CUDA and MLX runtime programs moved from large string constants to
  packaged resources. Source, wheel, and frozen builds emit the same bytes and
  manifest identities.
- Native backend shutdown now retains ownership after a timeout, blocks restart
  and application termination while descendants survive, rejects PID reuse, and
  permits an explicit cleanup retry.
- MLX memory estimation now distinguishes MoE resident weights from routed
  per-token computation under `aptus-memory-mlx-v2`.
- Managed MLX completion now leaves terminal promotion to the parent, which
  re-verifies the active run and source report before committing an
  `aptus.parent-promotion.v1` receipt.

### Removed

- The retired handwritten bundle-policy decision helper. Generated bundles now
  use the generic snapshot evaluator, with exact parity tests against the host
  registry.
- The 75-file local `EXAMPLE` intake after accepted findings were integrated
  and unsafe, duplicate, stale, or copyrighted source copies were discarded.
- Pass-through full-training resume. CUDA arbitrary-checkpoint resume remains
  fail-closed, every MLX resume argument is rejected, and MLX weight snapshots
  are not described as resumable checkpoints.
- Claims of automatic FP16 fallback.
- Claims that analysis alone proves fit, speed, cost, or quality.
- Reviewed TO-REVIEW implementation sketches after their exact snapshot and
  dispositions were preserved in the reconciliation ledger.

### Evidence status

Repository tests and static checks are necessary but not sufficient for release.
The dated Apple Silicon record under
`docs/operations/evidence/2026-07-27-mlx-lm-acceptance/` proves two clean,
independent MLX-LM workflows through `measured-run-pass`. The dated desktop
record under `docs/operations/evidence/2026-07-27-desktop-release/` proves 10 of
10 clean local engineering builds at implementation commit
`1038ecdd13103418ef1135e1ced634c10370a961`. It does not bind a later source
head. Pull-request CI rebuilds and packages GitHub's exact tested merge commit,
then records that identity in `COMMIT`. The July MLX-LM record predates Phase 4
and does not bind the current source head; no current-head MLX or CUDA
target-runtime pilot was collected for the Phase 4 closeout. The later
`docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/`
packet records two fresh MLX-LM `measured-run-pass` workflows at exact source
`719255153e3fc7e38e83b5ff826d587e5e58bf80`; the original August 5 Phase 6
acceptance packet remains its historical baseline. The later
`docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/`
packet records one exact five-job SmolLM2 CUDA LoRA single-device workflow
through `measured-run-pass` at source
`c12c4d8db0037a2c278a2ad95a0a2cbda4387eed`. It does not establish
repeatability or qualify other CUDA methods, placements, devices, artifacts,
or environments. The default Mac artifacts are
ad-hoc signed, not a Developer ID signed and notarized public distribution.
Version 0.2.0 remains unreleased.

The integrated Phase 2A CUDA campaign source gates passed on the stable
development tree: all 302 campaign tests passed in 22.833 seconds, with Ruff
lint, formatting, Python compilation, and diff-integrity checks also passing.
The encompassing closeout passed all 888 Python tests in 45.777 seconds, all
130 React tests, generated-contract and version checks, the installed-wheel
smoke, and the native app and DMG build.
All Phase 2A implementation and review work
occurred on the development Mac without connecting to or mutating the intended
Ubuntu host. It produced no new CUDA timing, resource, thermal, model, or
empirical result. Any future RTX 3050 campaign boundary is exact-host local
evidence, not an Aptus-wide cloud or multi-GPU ceiling.

## Related documentation

- [Current capabilities](docs/product/current-capabilities.md)
- [Release gates](docs/operations/release-gates.md)
- [Roadmap](ROADMAP.md)
