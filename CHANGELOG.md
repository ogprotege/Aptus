# Changelog

> **Status:** Active | **Authority:** Release record | **Applies to:** Aptus 0.2 | **Audience:** Users and maintainers | **Last reviewed:** 2026-07-29 | **Review by:** Every release

All notable changes are recorded here.

## 0.2.0 - Unreleased

### Added

- Aptus for Mac, a native AppKit and WebKit application with automatic bundled
  backend lifecycle, private session authentication, native path pickers,
  Finder actions, startup recovery, app packaging, and CUDA-host handoff.
- V3 fact and training-plan contracts, plus the retained versioned candidate,
  bundle, validation, and job contracts.
- Exact Qwen3 MoE compatibility for inspected four-bit `qwen3_moe` checkpoints
  using `Qwen3MoeForCausalLM`. The implemented conditional planner slice is
  single-device MLX-LM QLoRA with attention-only adapters and mandatory pilot
  evidence.
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
- Local FastAPI service, React workbench, CLI, persisted jobs, cancellation, and
  a per-user host-global Aptus execution lease.
- Immutable full-run output IDs and parent-owned completion verification.
- Runtime-specific pilot and measured-capacity admission contracts. CUDA proves
  checkpoint continuation. MLX proves exact target binding, at least two
  optimizer updates with finite losses, changed adapter weights, positive peak
  and delta measurements, live unified-memory headroom, immutable artifacts,
  and a fresh-process adapter reload that generates one to four tokens.
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

### Removed

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
then records that identity in `COMMIT`. No
qualifying CUDA target-host pilot or full run has been recorded. The default Mac
artifacts are ad-hoc signed, not a Developer ID signed and notarized public
distribution. Version 0.2.0 remains unreleased.

## Related documentation

- [Current capabilities](docs/product/current-capabilities.md)
- [Release gates](docs/operations/release-gates.md)
- [Roadmap](ROADMAP.md)
