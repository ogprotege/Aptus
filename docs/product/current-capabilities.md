# Current Capabilities

This page is the normative v0.2 product boundary. Aptus v0.2 is unreleased and
still lacks target CUDA release evidence.

## Available now

- Local profiling for JSON, JSONL, CSV, and text supervised data.
- Validation and deterministic JSONL serialization of every training row during
  compilation, followed by tokenizer-specific transformation at model-data and
  training time.
- Bounded provider model-metadata inspection at an immutable revision.
- Local CUDA hardware inspection or explicit manual hardware facts.
- Full, LoRA, int8-LoRA, and QLoRA candidate enumeration.
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

- CPU, MPS, and ROCm execution.
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

## Evidence status

Static and local tests can confirm contracts and platform-independent behavior.
No real CUDA pilot has been run on the current development Mac. Aptus must not be
described as release-ready until the release record passes every applicable gate.
