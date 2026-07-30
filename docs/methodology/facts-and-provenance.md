# Facts and Provenance

> **Status:** Active | **Authority:** Normative methodology | **Applies to:** Aptus 0.2 | **Audience:** Practitioners and contributors | **Last reviewed:** 2026-07-29 | **Review by:** 2027-01-22 or when fact provenance changes

Fact contract version: `aptus.facts.v3`.

V0.2 accepts explicit facts and preserves the provenance that the current
profilers can support. It does not inspect every model or runtime fact.

## Provenance kinds

The serialized plan may use:

- `measured`: Aptus observed the value from a local file or device interface;
- `provider-declared`: a provider-owned source declared the value;
- `user-attested`: the user supplied and accepted responsibility for the value;
- `inferred`: a versioned rule derived the value;
- `unknown`: no defensible value is available.

Model and manually entered hardware fields currently share a
`user-attested` provenance record with source `cli-or-api`. Dataset provenance
records the resolved local path, file modification time, and SHA-256 digest.
A local hardware probe records `measured` provenance when it succeeds.

## Model facts

V0.2 requires explicit:

- model ID;
- immutable 40-to-64-character hexadecimal revision;
- supported family name;
- parameter count in billions;
- hidden size, layer count, and context length;
- optional intermediate size;
- license identifier;
- affirmative training-permission attestation.

V4 can also bind exact provider `model_type`, architecture, checkpoint
precision, and a complete MoE topology. The topology contains expert count,
experts selected per token, expert width, sparse cadence, dense-only layer
indices, and optional shared-expert width. Aptus derives active parameters and
sparse-layer count from the complete model contract. The total parameter count
remains user-attested and remains the resident-weight basis.

These values are contract-checked but not independently inspected during
planning. The model-data gate resolves the pinned config and tokenizer, loads
the exact model weights, checks the parameter count and plan-driving structural
config fields, and verifies that every catalog-derived target module exists.
The same gate prepares the selected method and rejects zero trainable
parameters, non-finite trainable values, or a trainable set outside the method
scope. Full tuning permits no frozen model tensor. Current LoRA-based paths
permit only compiled LoRA tensors. The CUDA gate computes positive tensor and
parameter counts plus a SHA-256 descriptor digest over sorted trainable names,
shapes, and dtypes. The digest discloses no parameter names or values. CUDA
measured preflight, pilot, and full-run metrics persist this census, and both
pilot phases must agree. MLX metrics instead bind every planned target instance
to one LoRA A/B pair, reject other trainables, and record a descriptor digest.
Its pilot applies the selected method to compiled real data in one uninterrupted
run, then verifies adapter reload in a fresh process without resuming training.
The exact Qwen3 MoE row also requires the pinned config to reproduce every
identity, quantization, and topology fact in the plan. Its compiler scope binds
attention adapters only.

## Inspection receipt and policy source

Successful model inspection produces an
`aptus.model-inspection-receipt.v1` at one resolved immutable revision. The
receipt contains the complete `aptus.model-compatibility.v2` decision and two
deliberately different SHA-256 digests:

- `subject_facts_sha256` covers only compatibility inputs used by the policy;
- `observed_facts_sha256` covers every provider-declared or inferred planning
  fact actually carried from inspection.

The broader digest can cover architecture, context length, family, hidden and
intermediate sizes, layer count, license label, raw model type, MoE topology,
and quantization precision and layout. It covers only fields present in the
receipt's sorted provenance summary. Omitted provider fields stay
user-attested. Parameter count and training permission are always excluded.

A receipt can contain only `provider-declared` and `inferred` entries. It must
cover every non-null compatibility subject field and include at least one
provider-declared subject observation. Registered policies can impose a
stricter field rule. These constraints prevent a caller from relabeling wholly
user-attested or unrelated facts as provider inspection.

Passing a valid receipt to planning produces source `provider-inspection`.
Planning without one produces source `user-attested`. A present receipt is
recomputed against the submitted model ID, revision, carried facts, current
policy, provenance requirements, and receipt ID. Invalid input is rejected and
never downgraded to user-attested.

These content hashes are tamper-evident, not authenticated. They establish
content agreement inside the trusted local process and client boundary. They do
not prove the identity of the provider or caller.

## Dataset profile

The profiler reads JSON, JSONL, CSV, or text. It records:

- resolved path, format, size, and SHA-256 digest;
- resolved row-schema counts;
- valid, empty, and normalized-duplicate counts;
- total estimated tokens;
- sampled indices and sample count;
- p50, p95, and maximum sequence lengths;
- truncation count and rate when a sequence length is supplied;
- warnings and measurement kind.

Without an injected tokenizer, token counts use the explicit
four-characters-per-token estimate. API profiling uses this estimate. A
deterministic digest-seeded reservoir supplies sampled length statistics when a
sample limit is set. The model-data gate later transforms every canonical row
with the pinned tokenizer.

For full training, an optional top-level or `metadata.split_group` declares rows
that must remain together. Ungrouped data uses
`deterministic-exact-row-count-sha256`; data with declared groups uses
`deterministic-size-aware-group-sha256`. The generated trainer records target
and realized evaluation size, row error, train and evaluation row counts,
declared-group counts, split-unit counts, the canonical dataset digest, and the
assignment digest. It checks canonical data across split passes and lazy
consumption, and distributed ranks must agree on the binding. It does not expose
group names. An indivisible group can prevent an exact requested fraction.
V0.2 still does not record p99 length, padding efficiency, a data-rights
attestation, or a separate loss-mask fact. The compiled transformation
implements schema-specific masking.

## Hardware facts

Manual hardware input requires backend, GPU count, total per-device VRAM, total
host RAM, and a user reserve. It accepts separate BF16, 8-bit, and 4-bit
capability flags. Current free VRAM, available host RAM, and free disk are
optional. Each declared device receives the same entered capacity and
capabilities.

The local probe reads CUDA device name, total and free VRAM, compute capability,
BF16 support, total and available host RAM, and free disk. It derives NF4/FP4
and LLM.int8 hardware eligibility from the documented compute-capability
thresholds. On Darwin arm64 without CUDA, it instead records a single `mps`
device representing measured shared unified memory. Current availability stays
unknown when the host interface does not provide it. This fallback does not
assert MPS, MLX, BF16, or bitsandbytes execution support. When current host RAM
availability is measurable, the MLX estimator treats it as live unified-memory
headroom after the user reserve. It still leaves `free_vram_bytes` empty because
Apple Silicon has no separate free-VRAM pool. The Apple platform probe also
records the macOS build, chip, CPU count, optional built-in Metal GPU core
count, pressure, swap, working-set advisory, and interpreter-local runtime
facts. The exact external-runtime inventory separately proves whether MLX-LM or
PyTorch MPS is usable. MLX-LM QLoRA verifies four-bit metadata on the pinned
model revision. CUDA dependency validation verifies bitsandbytes where that
runtime requires it. The hardware profile does not yet record device UUID,
driver version, CUDA version, interconnect, or package versions.

When an API plan request uses `discovery: local-scan`, the planning endpoint
probes the Aptus host again. It preserves the requested reserve but replaces
submitted capacity and capability values with that measurement. The resulting
hardware and device provenance stays `measured`. A prior UI scan is a preview,
so facts can change between preview and planning.

## Target facts

The target records objective, sequence length, effective batch, epoch count,
optional method preference, SFT task, evaluation fraction, packing flag, and
checkpoint interval. V0.2 rejects tasks other than `sft` and rejects enabled
packing.

## Changes and conflicts

Changing an input requires a new v4 plan. V0.2 does not serialize a field-level
override actor, reason, or conflict-resolution history. A future fact ledger
must retain original and replacement values instead of overwriting evidence.

## Related documentation

- [Model, dataset, and hardware facts](../guides/model-dataset-hardware.md)
- [Data and identity flow](../architecture/data-and-identity-flow.md)
- [Evidence records](../reference/evidence-records.md)
- [Plan schema](../reference/plan-schema.md)
