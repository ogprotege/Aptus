# Roadmap

> **Status:** Active | **Authority:** Product planning | **Applies to:** Work after Aptus 0.2 | **Audience:** Users and contributors | **Last reviewed:** 2026-07-27 | **Review by:** Every release-planning cycle

The roadmap separates the executable v0.2 contract from future work. An item on
this page is not a supported capability until code, tests, documentation, and
target-host evidence all agree.

## v0.2 stabilization

Completed evidence:

- Two clean, independent MLX-LM QLoRA workflows reached
  `measured-run-pass` with the exact recorded model, revision, dataset, host,
  runtime, plan, and generated bundle.
- Every MLX resume argument fails closed. Pilot and full-run evidence explicitly
  records uninterrupted execution and no resume support.
- The local desktop stability gate completed 10 of 10 clean engineering builds
  at implementation commit `1038ecdd13103418ef1135e1ced634c10370a961`.
  This is historical evidence for that commit, not later source heads.
- The pull-request workflow rebuilds, verifies, and uploads the ad-hoc-signed
  Aptus.app ZIP and DMG with checksums and a source marker for the exact pushed
  commit on `main` or GitHub's exact synthetic merge commit for a pull request.
  Its result becomes evidence only after that workflow run passes.
- Browser accessibility checks cover the packaged React workbench. Native tests
  cover lifecycle, session, shutdown, navigation, and packaging contracts.

Remaining release work:

- Complete a real CUDA run for every claimed executable method and placement.
- Extend MLX-LM acceptance beyond the exact recorded M5 Pro, Qwen QLoRA, and
  synthetic-dataset configuration before making broader Apple Silicon claims.
- Record clean-environment dependency installation on every claimed CUDA path.
- Prove managed cancellation, stale-owner recovery, global-lease behavior, and
  crash-safe completion promotion on every claimed target operating system.
- Extend semantic export checks beyond the MLX pilot and full-run fresh-process
  adapter reload with bounded generation.
- Run native accessibility and appearance checks on macOS 26 and the macOS 15
  fallback.
- Obtain a Developer ID Application identity, notarize and staple the app and
  DMG, and pass Gatekeeper assessment before public distribution.
- Publish a release evidence record for the exact released commit and artifacts.

Full-parameter FSDP remains unsupported during v0.2. LoRA FSDP remains
conditional until its runtime gate is complete.

## First post-v0.2 milestone: MoE model support

Mixture-of-Experts models are not currently supported. Aptus does not silently
map an MoE, multimodal, prefix-matched, or unknown architecture onto a dense
family contract.

Delivery is split so useful, honest MoE planning reaches users before every
runtime is complete:

1. **Inspection and fit planning.** Detect one exact, allowlisted MoE family,
   report total and active parameters separately, census experts and routers,
   and produce sparse memory, storage, and throughput estimates. Training stays
   disabled and visibly marked unsupported in this slice.
2. **Apple Silicon QLoRA execution.** Add compile, pilot, train, export, and
   fresh-process reload support for that family through the maintained MLX-LM
   runtime. Release it only after real target-host acceptance.
3. **CUDA execution and family expansion.** Add each runtime, placement, and
   architecture independently. A passing Apple path never implies CUDA support,
   and one passing MoE family never admits prefix-matched variants.
4. **Comparable evaluation.** Compare dense and MoE candidates against the same
   immutable data, quality metrics, thresholds, and compute envelope.

Each executable MoE slice requires these gates:

1. an exact architecture and revision allowlist with explicit rejection for
   unknown or unreviewed variants;
2. separate total-parameter and active-parameter facts, with expert count,
   experts selected per token, router structure, and shared-expert facts;
3. an expert, router, and adapter-target census that proves the intended
   trainable set and rejects accidental dense or zero-parameter selection;
4. a sparse memory, storage, communication, and throughput estimator that does
   not substitute active parameters for resident weights;
5. runtime-specific compile, pilot, train, artifact, export, and fresh-process
   reload contracts for each supported method and placement;
6. dense-versus-MoE evaluation on the same immutable data, metrics, thresholds,
   and compute envelope; and
7. real target-host acceptance for every claimed runtime and placement.

Inspection and fit-planning visibility may precede execution only as an explicit
non-executable state. Training becomes selectable one allowlisted family and
runtime at a time, after that exact executable contract and its evidence gates
agree.

## Planner depth

- Evolve the versioned method descriptor registry into method-dispatched
  estimators, compilers, checkpoint contracts, and export verifiers.
- Add bounded, Optuna-style search only after feasibility filtering. Persist
  every trial's facts and provenance, prune without hiding failed trials, and
  optimize against validation-only objectives rather than test results.
- Separate training objective, parameterization, recipe modifiers, optimizer,
  precision, quantization, and distribution as explicit planning axes.
- Add calibrated device and model-family priors without replacing measured
  pilot evidence.
- Add target wall-time and budget constraints with honest abstention.
- Add richer dataset quality, contamination, and task-shape diagnostics.
- Implement DoRA first through the pinned maintained runtime, then BitFit with
  architecture-gated bias selection and a bias-delta artifact. Keep AdaLoRA,
  LoReFT, AFLoRA, BiLoRA, and ShareLoRA behind their documented gates.

## Evaluation and export contracts

- Define evaluation datasets, metrics, thresholds, and baseline comparisons as
  first-class target facts.
- Bind evaluation results to the exact run and exported artifact.
- Define exporter interfaces for adapters, merged models, and deployment
  packages.
- Add semantic load and inference checks beyond the current structural file-tree
  verification.

## Execution and recovery

- Design a full checkpoint manifest that binds model, optimizer, scheduler,
  scaler, RNG, dataloader progress, environment, plan, and distributed topology.
- Enable full-run resume only after that manifest survives interruption tests.
- Treat MLX-LM periodic saves as weight snapshots until a separately versioned
  optimizer, scheduler, random-state, and data-position continuation contract
  survives interruption tests.
- Add an explicit deep re-verification command for historical artifacts.
- Add retention and cleanup policies for unique no-clobber runs and caches.

## Additional platforms and integrations

- Harden the separate MLX-LM LoRA and QLoRA pilot and full-duration path across
  supported Apple Silicon systems. Keep unified-memory accounting separate from
  CUDA VRAM formulas.
- Add a PyTorch MPS compiler only after it has its own estimator, export
  contract, generated bundle, and measured evidence. Runtime discovery alone
  is not compiler support.
- Expand LM Studio and oMLX integrations only as bounded inference and
  evaluation adapters. Do not treat either service as a training runtime.
- Evaluate ROCm after its own runtime and hardware contracts are defined.
- Add cloud runner and provider adapters behind explicit credentials and cost
  boundaries.
- Add MCP and external automation adapters after the local authorization model
  is defined.
- Add experiment trackers as optional sinks, never as the source of truth for
  local completion.

## Corpus governance

- Build an append-only capture and review service with immutable interaction,
  turn, correction, reviewer, consent, license, redaction, and provenance
  records.
- Export only approved rows into versioned SFT or preference schemas.
- Bind deduplication, `split_group`, train, validation, and test identities so
  related chunks and paraphrases cannot leak across evaluation boundaries.
- Support revocation and trace which historical dataset digests contained a
  record.

## Non-goals

Aptus will not claim universal strategy optimality, guaranteed fit, guaranteed
quality, or automatic permission to train a model or dataset.

## Related documentation

- [Current capabilities](docs/product/current-capabilities.md)
- [Method selection guide](docs/guides/choose-a-method.md)
- [Documentation debt](docs/maintenance/documentation-debt.md)
