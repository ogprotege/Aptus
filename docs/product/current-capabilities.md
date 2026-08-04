# Current Capabilities

> **Status:** Active | **Authority:** Normative product boundary | **Applies to:** Aptus 0.2 | **Audience:** Users, operators, and integrators | **Last reviewed:** 2026-08-04 | **Review by:** 2026-10-27 and every release

This page is the normative v0.2 product boundary. Aptus v0.2 is unreleased.
Apple Silicon MLX-LM acceptance reached `measured-run-pass` twice in a clean
isolated checkout, but that July evidence predates Phases 4 and 5 and does not
bind the current source head. No current-head CUDA or MLX target-runtime pilot
was collected for the Phase 5 closeout. A separate local desktop gate completed
10 of 10 clean engineering builds at implementation commit
`1038ecdd13103418ef1135e1ced634c10370a961`. Pull-request CI rebuilds and
packages GitHub's exact tested merge commit and records it in `COMMIT`. CUDA
target-host and public Developer ID signed and notarized desktop-distribution
gates remain open.
The exact Qwen3 30B-A3B MoE attempt passed dependency validation but stopped
before model loading because live unified memory was 18.932 GiB below the
required envelope. Full MoE model-data, preflight, pilot, reload, and training
acceptance remain open.

## Available now

- Local profiling for JSON, JSONL, CSV, and text supervised data.
- Validation and deterministic JSONL serialization of every training row during
  compilation, followed by tokenizer-specific transformation in the selected
  runtime gates.
- Bounded provider model-metadata inspection at an immutable revision.
- One host-side model compatibility registry shared by provider inspection,
  sparse candidate admission, and API execution-path validation. It derives
  compiler, estimator, export, and evidence-requirement identities from the
  method registry instead of copying them.
- Exact policy matching for `qwen3_moe` checkpoints with
  `Qwen3MoeForCausalLM`, four-bit group-64 defaults, one eight-bit group-64
  router-gate override per layer, a complete reviewed routed-expert topology,
  and no shared expert. Inspection returns eligibility for one reviewed pilot
  path or an unsupported result. The eligible tuple binds `mlx-lm`, `mps`,
  QLoRA, `single`, and adapter profile `attention-qkvo.v1`. It never relies on
  a family prefix or treats inspection as passing runtime evidence. Sparse
  near-matches are blocked before dense-family recognition, including sparse
  model-type or architecture markers whose topology is missing. Conditional API
  claims must match a registered path for the stated model family.
- Persisted `aptus.training-plan.v5` compatibility provenance and the digest of
  one canonical `aptus.model-policy-snapshot.v1`. One
  `aptus.model-compatibility.v2` decision records stable reason and evidence
  IDs, policy ID `model.qwen3-moe.mlx-qlora`, policy version `1.0.0`, and path ID
  `mlx-lm.qlora.single.attention-qkvo.v1` when the exact row matches. Every
  candidate links to the decision. Only the exact matching candidate receives
  an `aptus.model-policy-binding.v1` path binding.
- Versioned `aptus.model-inspection-receipt.v1` output from successful provider
  inspection. The receipt separately binds compatibility-subject facts and all
  provider-declared or inferred planning facts carried into the plan. Parameters
  and training permission remain user-attested and are excluded. Direct facts
  without a receipt use the explicit `user-attested` decision source. A supplied
  invalid receipt fails instead of silently degrading to that source.
- Phase 5's server-authoritative workbench policy boundary. Model-inspection,
  bootstrap, successful planning, and typed HTTP 422 `no_feasible_plan` payloads
  are decoded at client ingress as exact `aptus.model-compatibility.v2`
  decisions, nested paths, optional receipts, candidate decision links, and
  explicit nullable bindings. Missing keys, extra keys, unknown contract
  versions or closed vocabulary, malformed identities, and cross-record drift
  fail before UI hydration. Successful and no-feasible planning responses must
  match the submitted model ID, immutable revision, policy source, and receipt
  identity. Candidate ingestion requires method, distribution, status,
  feasibility, rejection reasons, targets, runtime contract, decision link, and
  binding; every no-feasible row must be rejected. Generated TypeScript types do
  not replace these runtime checks. The closed failure also requires a `model`
  subject whose ID and immutable revision match the submitted artifact, so
  user-attested failures remain request-bound without a receipt. The browser
  contains no family-specific policy predicate or legacy flattened compatibility
  projection. Provider path-matched receipts require provider-declared
  provenance rather than inferred-only observations. A successful response's
  decoded recommendation must structurally equal its complete listed candidate
  record.
- Three separate model-policy UI records: model-policy match, selected candidate
  path, and evidence readiness. The path record uses only the selected
  candidate's explicit binding and never promotes an unbound execution contract
  into a policy claim. An exact emitted-path tuple cannot carry a null binding.
  Truly unbound and rejected candidates receive no synthesized policy gate
  ladder or impossible validation action. Readiness uses a validation report
  only when its `plan_id`, `candidate_id`, and `model_revision` bind the current
  selection. That same exact tuple gates stage completion and validation or run
  actions.
- Separate validation-evidence and launch-admission states. Required evidence is
  either incomplete or complete. The optional typed `authorization_status` is
  exactly `current`, `deferred`, or `blocked`. Current requires
  `authorization_current: true` with no error; deferred or blocked requires false
  with a non-empty diagnostic. A tuple with no non-null member means not checked,
  and partial or incoherent claims are rejected. The browser never derives
  status from the diagnostic or mutates the prior report after a generic training-request
  failure. Non-current authorization is not itself a stale-policy or replan
  result. HTTP 409 `replan_required` remains the distinct lifecycle authority.
  The separate MoE topology rail describes routing, total resident weights, and
  active-per-token computation without deciding compatibility or subtracting
  inactive experts from resident memory.
- A closed typed HTTP 422 `no_feasible_plan` response that preserves rejected
  candidates together with the server decision, decision source, and nullable
  inspection receipt. Provider-inspection failures require the matching receipt;
  user-attested failures forbid one; all candidate links and bindings must agree
  with the same chain and the submitted request. The resulting comparison is a
  non-compilable partial view.
- Strict replanning for v4, v3, v2, schema-less, and stale-policy or
  stale-snapshot v5 plans. Aptus preserves old saved bytes and does not relabel
  them. The HTTP API remains `aptus.api.v1`; facts remain `aptus.facts.v3`;
  candidate runtime contracts remain `aptus.runtime-contract.v1`. Phase 4
  changed bundles to `aptus.bundle.v3`, which carries the deterministic frozen
  snapshot at `policy/model-policy-snapshot.v1.json` plus a generic evaluator
  independent of installed Aptus.
- Package-free portable validation checks frozen-snapshot integrity and
  decision parity. Without an installed host and its current registry, it
  cannot determine policy currency. Installed Aptus separately enforces current
  registry currency for validation, host-managed admission, pilot authorization,
  worker launch, and the completion verification and promotion transaction.
- Installed-host JSON boundaries require object roots for plans, manifests,
  trainer configurations, and snapshots. Package-free validation enforces the
  plan, manifest, and snapshot boundaries. Covered malformed nested fields,
  JSON `null`, oversized integers, excessive nesting, and malformed snapshot
  operands become controlled contract errors or INVALID findings rather than
  escaped parser or traversal exceptions. CUDA validation and execution reject
  the plan before device binding; the trainer configuration remains
  compiler-managed runtime input.
- Local CUDA hardware inspection and explicit manual hardware facts.
- Apple Silicon platform inspection for macOS version and build, chip name,
  logical CPU count, unified-memory capacity and current headroom, memory
  pressure, swap, Metal working-set guidance, optional Metal GPU core count,
  and separate MLX, MLX-LM, and PyTorch MPS capability facts. Discovery uses
  measured capabilities, not a chip-name allowlist.
- A compatibility `mps` hardware record that represents shared unified memory,
  never dedicated VRAM. Host free RAM is not copied into `free_vram_bytes`.
  The MLX estimator instead caps usable unified memory by current
  `host_ram_free_bytes` when that live measurement exists.
- Full, LoRA, int8-LoRA, and QLoRA candidate enumeration.
- An 11-descriptor typed method registry. The four candidate methods are
  `gated-executable` and selectable. DoRA, BitFit, AdaLoRA, and ShareLoRA are
  `experimental`; LoReFT, AFLoRA, and BiLoRA are `research-only`. All seven are
  visible but nonselectable and lack compiler and export contracts.
- Deterministic CUDA full-run splitting. Ungrouped data uses an exact-row-count
  strategy. Declared `split_group` values use exact subset selection when the
  target is attainable, then the closest feasible size otherwise, while every
  group remains atomic. Metrics record canonical and assignment digests, target
  and realized evaluation sizes, and row error.
- Canonical-dataset mutation checks across split passes and lazy consumption,
  plus collective digest and count agreement for distributed runs.
- For CUDA, a positive, finite trainable-parameter census before optimizer construction.
  Full tuning requires every model tensor to remain trainable. LoRA-based paths
  require one complete LoRA A/B pair per inspected target instance, reject every
  other trainable tensor, and prove exact optimizer membership. Measured
  preflight, pilot, and full-run evidence carry a stable digest over sorted names,
  shapes, and dtypes, and both pilot phases must agree.
- For MLX-LM, an exact target binding that requires one LoRA A/B pair for every
  planned target in every transformer layer, rejects other trainables, proves
  completed optimizer updates, and records a positive adapter delta.
- Single-device and DDP candidates where method, capability, batch, host RAM,
  disk, and memory rules pass.
- Conditional LoRA FSDP candidates.
- Explicit unsupported records for full FSDP and quantized FSDP.
- Transparent point and upper memory calculations with evidence records.
- Deterministic ranking within the enumerated catalog.
- A versioned runtime contract on every candidate. It binds compute backend,
  training runtime, compiler, estimator, evidence requirement, and export kind.
- A separate MLX-LM unified-memory estimator and compiler for single-device
  LoRA and QLoRA. MLX-LM QLoRA requires four-bit quantization metadata in the
  pinned MLX model revision, verified at model-data validation rather than from
  a device capability flag. It never substitutes bitsandbytes.
- A narrow Qwen3 MoE MLX-LM QLoRA path. The plan records expert count, experts
  per token, expert width, sparse cadence, dense-only layers, total resident
  parameters, backend-derived active parameters, and sparse-layer count.
  Its canonical mixed quantization layout is identity-bound. Adapter targets
  are limited to attention `q_proj`, `k_proj`, `v_proj`, and `o_proj` modules.
- Atomic no-clobber bundle compilation and deterministic ZIP creation.
- Portable CUDA direct pins, validation, preflight, pilot, training child, and
  full-run parent programs, plus separate bounded MLX-LM runtime programs.
- Five ordered runtime actions: dependency, model-data, preflight, pilot, train.
- Persisted managed jobs, logs, cancellation, stale-owner reconciliation, and a
  per-user host-global Aptus lease.
- Versioned `aptus.job-record.v1` persistence. Legacy records migrate in place,
  while corrupt, symlinked, or unsupported records move to recoverable private
  quarantine without blocking healthy jobs.
- Runtime-specific deep train admission using current pilot and capacity
  evidence. MLX admission rechecks live Apple unified-memory headroom.
- Unique run-ID output directories, parent-owned completion promotion, and
  runtime-specific immutable export verification.
- Local same-origin API and React workbench.
- Exact external Python runtime discovery and configuration. Aptus probes the
  selected executable, persists its absolute command path in a private mode-0600
  configuration file, and launches MLX-LM work with that interpreter.
- Local LM Studio and oMLX adapters for bounded model listing and generation on
  explicit loopback origins. Both are inference-only.
- Named local projects with content-hashed `aptus.project-revision.v1`
  snapshots of facts, plans, bundles, validations, and job identities. Recovery
  creates a new revision and never restores training authorization.
- Native macOS application with an AppKit lifecycle and SwiftUI Home,
  Workbench, Machine, and Models shell. The authenticated React workbench is
  inline and owns the complete Facts, Compare, Compile, Validate, and Run flow.
- A clean-checkout desktop engineering gate that builds and launches the native
  app, verifies the packaged backend and workbench, checks the app signature and
  DMG, and can require ten consecutive passes. GitHub Actions performs the same
  authoritative build for each pull request's synthetic merge commit and
  uploads artifacts bound to that workflow commit.
- A read-only native MLX environment doctor that shows each likely Python path,
  source, version, import result, and exact-pin compatibility. Runtime selection
  rechecks the backend contract. Aptus performs no silent installation.
- Explicit API response models under `aptus.api.v1`, a checked OpenAPI JSON
  artifact, read-only `aptus doctor`, and privacy-bounded diagnostic archives.
- Local MLX-LM managed actions through pilot and explicitly confirmed
  full-duration adapter training. CUDA bundles remain explicit target-host
  handoffs from the Mac app.

## Conditional behavior

- Every viable MLX-LM candidate is conditional and pilot-required. Dependency
  validation verifies the pinned MLX and MLX-LM versions. Model-data validation
  loads the pinned revision and tokenizes every bound train and validation row.
  Measured preflight runs a bounded real MLX adapter smoke and records
  runtime-neutral memory metrics.
- The Qwen3 MoE row is conditional only when model type, architecture,
  four-bit group-64 defaults, one eight-bit group-64 router-gate override per
  layer, topology, runtime, compute backend, method, placement, and adapter
  profile all match the exact contract. All checkpoint weights remain resident.
  Active parameters describe per-token computation and never replace total
  parameters in the base-weight memory budget.
- The MLX-LM pilot is one uninterrupted exact-model and exact-data run from the
  pinned base. It requires at least two completed optimizer updates, finite
  train and validation losses, exact target coverage, positive MLX peak and
  adapter delta, live headroom admission, and immutable action-owned artifacts.
  A fresh child process then loads the pinned base plus saved adapter and
  generates one to four tokens. A passing result can promote `pilot-pass` and
  permit explicit full-duration adapter training.
- MLX full training starts from the pinned base and runs uninterrupted for the
  duration derived from compiled train rows, microbatch, accumulation, and
  maximum epochs. It completes at least one optimizer update. Its parent
  verifies the adapter tree, metrics, fresh reload evidence, and final export
  before `measured-run-pass`.
- MLX periodic saves are weight snapshots, not resumable checkpoints. Pilot
  reload proves adapter inference only. Every MLX resume argument fails closed.
- Adapter methods can use FP16 when participating devices do not declare BF16.
  The exact generated path must pass the selected pilot.
- LoRA FSDP depends on exact model structure, pinned runtime behavior, and a
  passing multi-rank pilot.
- Provider model inspection depends on network access, repository availability,
  and bounded metadata endpoints.
- Gated models depend on credentials supplied through the underlying model
  stack, not through Aptus plan fields.

## Explicitly unsupported

- MLX-LM crash resume and continuation from any weight snapshot.
- Full-parameter and DoRA training through MLX-LM. The current MLX compiler
  implements only LoRA and QLoRA adapters.
- PyTorch MPS compilation. The runtime is known, discoverable, and configurable,
  but it has no estimator, compiler, or export contract.
- CPU and ROCm training.
- CUDA execution on macOS. A manually entered CUDA profile describes another
  host and never turns the Mac into a CUDA target.
- Full-parameter FP16 training.
- Full-parameter FSDP.
- int8-LoRA FSDP and QLoRA FSDP.
- Sequence packing.
- Tasks other than supervised fine-tuning.
- Enforced maximum wall-time targets.
- Full-training resume.
- Multi-user or remotely exposed job service without an external boundary.
- MoE architectures other than the exact Qwen3 row above, Qwen3 MoE checkpoints
  with a shared expert or any other quantization layout, CUDA or distributed
  MoE execution, and MoE methods other than single-device MLX-LM QLoRA.

## Not implemented

- First-class evaluation datasets, metrics, thresholds, or baseline gates.
- General quality evaluation and CUDA semantic export load validation. MLX pilot
  and full runs perform only a bounded fresh-process adapter generation check.
- Exporter plugin contracts for merged or deployment-specific artifacts.
- Cloud runners, provider provisioning, or cost selection.
- MCP adapters or external automation authorization.
- Experiment-tracker ownership of completion state.
- Automated model-card or data-governance decisions.
- Chat capture, human-review workflow, consent enforcement, PII adjudication,
  or automatic feedback-to-training export.

## Evidence status

Static and local tests confirm contracts and platform-independent behavior. The
[2026-07-27 MLX-LM acceptance record](../operations/evidence/2026-07-27-mlx-lm-acceptance/README.md)
binds two clean runs through measured preflight, pilot, fresh-process adapter
reload, confirmed full training, final export, and `measured-run-pass`. The
[desktop engineering record](../operations/evidence/2026-07-27-desktop-release/README.md)
binds a 10-of-10 clean local stability result to implementation commit
`1038ecdd13103418ef1135e1ced634c10370a961`. It does not prove a later source
head. Pull-request CI must rebuild GitHub's exact tested merge commit and record
that identity. The July MLX-LM acceptance also predates Phases 4 and 5 and does
not bind the current source head. No current-head MLX or CUDA target-runtime
pilot was collected for the Phase 5 closeout, and no real CUDA pilot has run on
a CUDA target for this release. The default Mac artifact is ad-hoc signed, not a
Developer ID signed and notarized public distribution.
The [2026-07-28 Qwen3 MoE admission record](../operations/evidence/2026-07-28-qwen3-moe-admission/README.md)
proves exact plan, compile, dependency, packed-checkpoint, and live-memory
admission behavior. It does not prove 30B model loading or training speed.

## Related documentation

- [Capability matrix](../reference/capability-matrix.md)
- [Model-policy snapshot](../reference/model-policy-snapshot.md)
- [Method registry](../reference/method-registry.md)
- [Release gates](../operations/release-gates.md)
- [Claim language](claim-language.md)
