# Facts and Provenance

Fact contract version: `aptus.facts.v2`.

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

These values are contract-checked but not independently inspected during
planning. The model-data gate resolves the pinned config and tokenizer, loads
the exact model weights, checks the parameter count and plan-driving structural
config fields, and verifies that every catalog-derived target module exists. The
later pilot applies and exercises the
selected method with the compiled real data and checkpoint-continuation
contract.

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

V0.2 does not record p99 length, padding efficiency, split identities, a data
rights attestation, or a separate loss-mask fact. The compiled transformation
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
thresholds. The later dependency gate verifies the installed bitsandbytes
package. The hardware profile does not yet record device UUID, driver version,
CUDA version, interconnect, or package versions.

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

Changing an input requires a new plan. V0.2 does not serialize a field-level
override actor, reason, or conflict-resolution history. A future fact ledger
must retain original and replacement values instead of overwriting evidence.
