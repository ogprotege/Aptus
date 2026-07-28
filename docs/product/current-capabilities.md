# Current Capabilities

> **Status:** Active | **Authority:** Normative product boundary | **Applies to:** Aptus 0.2 | **Audience:** Users, operators, and integrators | **Last reviewed:** 2026-07-27 | **Review by:** 2026-10-27 and every release

This page is the normative v0.2 product boundary. Aptus v0.2 is unreleased.
Apple Silicon MLX-LM acceptance reached `measured-run-pass` twice in a clean
isolated checkout. CUDA target-host and public desktop distribution gates remain
open.

## Available now

- Local profiling for JSON, JSONL, CSV, and text supervised data.
- Validation and deterministic JSONL serialization of every training row during
  compilation, followed by tokenizer-specific transformation in the selected
  runtime gates.
- Bounded provider model-metadata inspection at an immutable revision.
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
  LoRA and QLoRA. MLX-LM QLoRA requires explicit four-bit capability facts and
  pinned MLX model metadata. It never substitutes bitsandbytes.
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
reload, confirmed full training, final export, and `measured-run-pass`. No real
CUDA pilot has run on a CUDA target for this release. The default Mac artifact
is ad-hoc signed, not a notarized public distribution.

## Related documentation

- [Capability matrix](../reference/capability-matrix.md)
- [Method registry](../reference/method-registry.md)
- [Release gates](../operations/release-gates.md)
- [Claim language](claim-language.md)
