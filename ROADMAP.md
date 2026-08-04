# Roadmap

> **Status:** Active | **Authority:** Product planning | **Applies to:** Work after Aptus 0.2 | **Audience:** Users and contributors | **Last reviewed:** 2026-08-04 | **Review by:** Every release-planning cycle

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
- Phase 4's repository, installed-wheel, and desktop package gates closed the
  portable policy-snapshot source and contract review. They did not renew
  target-runtime acceptance: the July MLX-LM record predates the current v5
  plan, v3 bundle, and Phase 6 registry expansion, and no current-head MLX or
  CUDA target-runtime pilot was collected.

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

## First post-v0.2 milestone: exact MoE compatibility

The first implementation slice recognizes only an exact `qwen3_moe` checkpoint
whose architecture is `Qwen3MoeForCausalLM`, whose reviewed MLX layout uses
four-bit group-64 defaults with one eight-bit group-64 router-gate override per
layer, and which declares no shared expert. It carries the provider topology
into `aptus.training-plan.v5`, derives active parameters and sparse-layer count,
and permits only single-device MLX-LM QLoRA with attention `q_proj`, `k_proj`,
`v_proj`, and `o_proj` adapters. Every candidate remains pilot-required. This
slice is not released until a real target-host model run passes the gates below.

Aptus still rejects unreviewed MoE families, prefix matches, multimodal models,
shared-expert variants, non-four-bit Qwen3 MoE checkpoints, CUDA MoE execution,
distributed MoE placement, and other MoE training methods.

Next work proceeds one explicit compatibility row at a time:

1. **Real-model acceptance.** Compile and run the exact Qwen3 MoE MLX-LM path
   through dependency, model-data, measured preflight, pilot, adapter reload,
   confirmed training, and final export gates.
2. **Measured performance record.** Record tokens per second, wall time, peak
   unified memory, adapter size, and dataset facts for the exact accepted run.
   Do not generalize those measurements to another host or checkpoint.
3. **CUDA execution and family expansion.** Add each runtime, placement, and
   architecture independently. A passing Apple path never implies CUDA support,
   and one passing family never admits another variant.
4. **Comparable evaluation.** Compare dense and MoE candidates against the same
   immutable data, quality metrics, thresholds, and compute envelope.

Each executable MoE slice requires these gates:

1. an exact architecture contract and immutable revision binding. An arbitrary
   revision is eligible only when every structural and runtime gate matches,
   and acceptance evidence applies only to that exact revision;
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

Compatibility code may precede release evidence, but the UI and plan must retain
the conditional state. Support expands one exact family and runtime at a time,
after that executable contract and its evidence gates agree.

Phase 3 binds the host decision into persisted plans. It adds stable policy and
path IDs, a semantic policy version, a versioned inspection receipt, separate
compatibility-subject and observed-planning-facts digests, and an exact binding
only on a candidate that matches the registered path. Every candidate still
links to the plan decision. Plans created from direct facts remain explicitly
`user-attested`; provider-backed plans require a valid receipt. Parameter count
and training permission remain outside that receipt.

Phase 4 is complete. The host registry now emits deterministic canonical
`aptus.model-policy-snapshot.v1` bytes. `aptus.training-plan.v5` binds their
SHA-256, and every `aptus.bundle.v3` contains the snapshot, the same digest,
and a generic portable evaluator. Package-free programs establish frozen
snapshot integrity and decision parity but cannot determine current registry
currency. Installed Aptus enforces current-host currency during static
validation, managed admission, pilot authorization, worker launch, and the
completion verification and promotion transaction. Saved v4, v3, v2, and
schema-less plans, plus coherent stale-policy v5 plans, require replanning;
saved-plan load, compile, recovery, and managed job submission surface
`replan_required`, while host static validation records the typed snapshot
finding. Aptus preserves saved bytes and requires deterministic replanning under
v5. Contract boundaries also reject non-object or resource-hostile JSON as
controlled invalid input instead of leaking parser or traversal exceptions.

Phase 5 is complete. The browser no longer reconstructs model-policy predicates
from topology, family, candidate, or presentation fields. Its maintained ingress
strictly decodes the server-produced `aptus.model-compatibility.v2` decision,
nested paths, optional `aptus.model-inspection-receipt.v1`, and each candidate's
explicit nullable `aptus.model-policy-binding.v1`. A plan success or typed HTTP
422 `no_feasible_plan` response must preserve that complete decision, source,
receipt, candidate-link, and binding chain or the client rejects it. Planning
responses must also carry a required model subject matching the submitted model
ID and immutable revision, then match the expected policy source and receipt
identity. No-feasible rows must be rejected and carry the complete method,
distribution, status, feasibility, rejection, target, runtime, and policy tuple.
A provider path-matched receipt must satisfy its provider-declared provenance
requirement with at least one provider-declared observation, not inferred-only
provenance. On success, the recommendation must structurally equal the complete
listed candidate record with the same ID.

The Facts and Compare stages present model-policy match, selected candidate
path, and evidence readiness as three separate records. The selected record
uses only that candidate's explicit binding. An execution tuple that exactly
matches an emitted path cannot silently carry null; a genuinely unbound or
rejected row receives no browser-invented validation ladder or impossible
action. Readiness uses a validation report only when its plan ID, selected
candidate ID, and immutable model revision all match. The same exact tuple gates
workflow completion and validation or run actions.

Validation evidence and launch admission remain separate. The UI distinguishes
incomplete from complete required evidence, then consumes only the optional
typed `authorization_status` values `current`, `deferred`, and `blocked`.
`current` must pair with `authorization_current: true` and no error;
`deferred` or `blocked` must pair with false and a non-empty diagnostic. If the
tuple has no non-null member, admission is not checked. The browser never
derives a status from diagnostic prose or mutates the last report when a generic
training request fails. A non-current status does not by itself mean stale policy or
require replanning; a genuine `replan_required` lifecycle response is the
separate authority for that action.
The MoE topology rail remains separate: it explains routed activity and total
resident weight memory without making a policy claim or reducing residency by
active parameters.

Phase 6 is implemented at the registry, planner, compiler, portable-contract,
and test boundaries, but it is not complete. The host registry and canonical
snapshot now carry a second data-driven policy,
`model.qwen2-24l.mlx-qlora`, for the exact `qwen`, `qwen2`, and
`Qwen2ForCausalLM` identity with 24 layers, dense topology, and a uniform
four-bit group-64 layout with no overrides. It emits only
`mlx-lm.qlora.single.dense-causal-lm.v1`, whose adapter profile targets
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`.

This policy is a reviewed configuration footprint, not an artifact allowlist.
Its historical runtime evidence remains scoped to the exact
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` revision, recorded host, runtime,
and dataset. That v2-plan and v2-bundle acceptance record supports the
implementation decision but does not establish current contract acceptance for
another artifact or for the current source head.

Phase 6 therefore remains runtime-evidence-open until a fresh current-source
`aptus.training-plan.v5` and `aptus.bundle.v3` ladder passes dependency,
model-data, measured preflight, an uninterrupted multi-update pilot,
fresh-process reload, confirmed full training, final reload and export, and
`measured-run-pass`. Until then the new row remains conditional and
pilot-required. Phase 5's browser-authority history remains unchanged.
`aptus.api.v1`, `aptus.facts.v3`, `aptus.model-policy-snapshot.v1`, and
`aptus.runtime-contract.v1` remain unchanged, while the added registry row
changes the canonical snapshot digest and requires older v5 plans to replan.

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
