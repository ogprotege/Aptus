# Current Capabilities

> **Status:** Active | **Authority:** Normative product boundary | **Applies to:** Aptus 0.2 | **Audience:** Users, operators, and integrators | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 and every release

This page is the normative v0.2 product boundary. Aptus v0.2 is unreleased and
still lacks target CUDA release evidence.

## Available now

- Local profiling for JSON, JSONL, CSV, and text supervised data.
- Validation and deterministic JSONL serialization of every training row during
  compilation, followed by tokenizer-specific transformation at model-data and
  training time.
- Bounded provider model-metadata inspection at an immutable revision.
- Local CUDA hardware inspection, explicit manual hardware facts, and
  fail-closed Darwin arm64 discovery that records an `mps` shared
  unified-memory inventory without treating it as executable or inventing free
  memory.
- Full, LoRA, int8-LoRA, and QLoRA candidate enumeration.
- An 11-descriptor typed method registry. The four candidate methods are
  `gated-executable` and selectable. DoRA, BitFit, AdaLoRA, and ShareLoRA are
  `experimental`; LoReFT, AFLoRA, and BiLoRA are `research-only`. All seven are
  visible but nonselectable and lack compiler and export contracts.
- Deterministic full-run splitting. Ungrouped data uses an exact-row-count
  strategy. Declared `split_group` values use exact subset selection when the
  target is attainable, then the closest feasible size otherwise, while every
  group remains atomic. Metrics record canonical and assignment digests, target
  and realized evaluation sizes, and row error.
- Canonical-dataset mutation checks across split passes and lazy consumption,
  plus collective digest and count agreement for distributed runs.
- A positive, finite trainable-parameter census before optimizer construction.
  Full tuning requires every model tensor to remain trainable. LoRA-based paths
  require one complete LoRA A/B pair per inspected target instance, reject every
  other trainable tensor, and prove exact optimizer membership. Measured
  preflight, pilot, and full-run evidence carry a stable digest over sorted names,
  shapes, and dtypes, and both pilot phases must agree.
- Single-device and DDP candidates where method, capability, batch, host RAM,
  disk, and memory rules pass.
- Conditional LoRA FSDP candidates.
- Explicit unsupported records for full FSDP and quantized FSDP.
- Transparent point and upper memory calculations with evidence records.
- Deterministic ranking within the enumerated catalog.
- Atomic no-clobber bundle compilation and deterministic ZIP creation.
- Portable direct pins, validation, preflight, pilot, training child, and
  full-run parent programs.
- Five ordered runtime actions: dependency, model-data, preflight, pilot, train.
- Persisted managed jobs, logs, cancellation, stale-owner reconciliation, and a
  per-user host-global Aptus lease.
- Deep train admission using current pilot and capacity evidence.
- Unique run-ID output directories, parent-owned completion promotion, and
  structural safetensors file-tree verification.
- Local same-origin API and React workbench.
- Native macOS application host with automatic private backend startup, native
  dataset and output selection, Finder reveal actions, persisted local state,
  and explicit CUDA-host handoff.

## Conditional behavior

- Adapter methods can use FP16 when participating devices do not declare BF16.
  The exact generated path must pass the selected pilot.
- LoRA FSDP depends on exact model structure, pinned runtime behavior, and a
  passing multi-rank pilot.
- Provider model inspection depends on network access, repository availability,
  and bounded metadata endpoints.
- Gated models depend on credentials supplied through the underlying model
  stack, not through Aptus plan fields.

## Explicitly unsupported

- CPU, MPS, MLX, and ROCm execution. Apple Silicon total shared memory can be
  discovered, but the current compiler does not execute there and does not
  claim current free unified memory when the host cannot provide it.
- CUDA execution inside the macOS application. A manually entered CUDA profile
  describes another host and never enables local Mac run controls.
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
- Semantic export load and inference validation.
- Exporter plugin contracts for merged or deployment-specific artifacts.
- Cloud runners, provider provisioning, or cost selection.
- MCP adapters or external automation authorization.
- Experiment-tracker ownership of completion state.
- Automated model-card or data-governance decisions.
- Chat capture, human-review workflow, consent enforcement, PII adjudication,
  or automatic feedback-to-training export.

## Evidence status

Static and local tests can confirm contracts and platform-independent behavior.
No real CUDA pilot has been run on the current development Mac. Aptus must not be
described as release-ready until the release record passes every applicable gate.

## Related documentation

- [Capability matrix](../reference/capability-matrix.md)
- [Method registry](../reference/method-registry.md)
- [Release gates](../operations/release-gates.md)
- [Claim language](claim-language.md)
