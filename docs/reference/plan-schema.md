# Plan Schema

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Planner consumers, compiler authors, reviewers, and integrators |
| Authority | Normative field reference for `aptus.training-plan.v2` |
| Last reviewed | 2026-07-22 |
| Next review | 2026-10-22, or sooner when domain or plan-contract code changes |

An Aptus plan is a canonical semantic record, not a loose set of launch flags.
The current schema identifier is `aptus.training-plan.v2`. Numbers must be
finite JSON values. The self-contained bundle validator recomputes candidate and
plan identities and rejects semantic mutation.

## Top-level object

```json
{
  "schema_version": "aptus.training-plan.v2",
  "plan_id": "plan_0123456789abcdef0123",
  "formula_version": "aptus-memory-v2",
  "model": {},
  "dataset": {},
  "hardware": {},
  "target": {},
  "recommended": {},
  "candidates": [],
  "warnings": [],
  "recommendation_rationale": [],
  "evidence_records": []
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Exact plan schema identifier |
| `plan_id` | string | `plan_` plus a 20-hex content identity |
| `formula_version` | string | Memory equation contract used by all candidates |
| `model` | object | Explicit model identity, structure, permission, and provenance |
| `dataset` | object | Source identity and profile |
| `hardware` | object | Planned devices, host capacity, and provenance |
| `target` | object | Requested training policy |
| `recommended` | candidate object | Highest-ranked viable candidate |
| `candidates` | array | Fixed 12-row method and placement matrix |
| `warnings` | string array | Plan-wide limitations and inferred assumptions |
| `recommendation_rationale` | string array | Human-readable ranking explanation |
| `evidence_records` | array | Resolved evidence objects cited by candidates |

## Model object

| Field | Type | Contract |
| --- | --- | --- |
| `model_id` | string | Provider repository ID, not a local path |
| `revision` | string | Immutable 40 to 64 character hexadecimal commit |
| `family` | string | Planner family, currently one of `gemma`, `llama`, `mistral`, `qwen` for adapter paths |
| `parameters` | integer | Positive exact user-attested parameter count |
| `hidden_size` | integer | Positive hidden width |
| `intermediate_size` | integer or null | Optional positive MLP width |
| `layers` | integer | Positive layer count |
| `context_length` | integer | Positive model context limit |
| `license_name` | string | Non-empty user-supplied license label |
| `training_allowed` | boolean | Must be true |
| `architecture` | string | Defaults to `causal-lm` in the domain contract |
| `tokenizer_id` | string or null | Optional tokenizer override; current builders leave it null |
| `provenance` | object | Field name to provenance object mapping |

Provider inspection can supply architecture facts, but it cannot supply
`parameters` or set `training_allowed`.

## Dataset object

| Field | Type | Meaning |
| --- | --- | --- |
| `source_path` | string | Resolved source path before compilation, bundle-relative path after compilation |
| `bundle_path` | string or null | Copied dataset path inside a portable bundle |
| `source_sha256` | 64-hex string | Source bytes at profiling and compilation |
| `source_format` | string | `jsonl`, `json`, `csv`, or `txt` |
| `schema_name` | string | One canonical schema or `mixed` |
| `schema_counts` | object | Canonical schema name to valid-row count |
| `example_count` | integer | Positive non-empty supported row count |
| `total_estimated_tokens` | integer | Sum from the selected measurement mode |
| `sequence_p50` | integer | Nearest-rank median over sampled lengths |
| `sequence_p95` | integer | Nearest-rank 95th percentile over sampled lengths |
| `sequence_max` | integer | Maximum sampled length |
| `measurement` | string | `estimated` or `tokenizer-measured` |
| `sampled_examples` | integer | Number of rows used for percentile statistics |
| `sample_indices` | integer array | Deterministically sampled source-row indices |
| `duplicate_count` | integer | Repeated normalized-text count |
| `empty_count` | integer | Ignored empty-row count |
| `truncation_count` | integer | Rows longer than requested sequence length |
| `truncation_rate` | number | `truncation_count / example_count` |
| `source_size_bytes` | integer | Source file size |
| `canonical_size_bytes` | integer | Predicted deterministic JSONL bytes for valid rows |
| `max_canonical_row_bytes` | integer | Largest deterministic canonical row |
| `warnings` | string array | Sampling, estimate, duplicate, empty, and truncation notices |
| `provenance` | object or null | Source observation and digest |

The four-characters-per-token fallback is labeled `estimated`. Compilation
still validates and serializes every supported row. See
[dataset schemas](dataset-schemas.md) for exact accepted shapes.

## Hardware object

| Field | Type | Meaning |
| --- | --- | --- |
| `devices` | array | Ordered device records |
| `host_ram_bytes` | integer | Positive total host RAM |
| `host_ram_free_bytes` | integer or null | Current available host RAM when measured or declared |
| `reserve_per_device_bytes` | integer | Non-negative user reserve |
| `disk_free_bytes` | integer or null | Current free disk when measured or declared |
| `cuda_version` | string or null | Optional discovery field |
| `interconnect` | string or null | Optional discovery field |
| `provenance` | object or null | Hardware-source provenance |

Each device contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Non-empty display identity |
| `backend` | string | `cuda`, `rocm`, `mps`, or `cpu` |
| `total_vram_bytes` | integer | Positive device or discovered shared-memory total |
| `free_vram_bytes` | integer or null | Current free capacity when known |
| `supports_bf16` | boolean | Declared or measured capability |
| `supports_4bit` | boolean | Declared or derived capability |
| `supports_8bit` | boolean | Declared or derived capability |
| `compute_capability` | string or null | CUDA capability when known |
| `driver_version` | string or null | Optional driver fact |
| `provenance` | object or null | Device-source provenance |

For a distributed row, fit uses the least usable participating device. For a
single row, the candidate records the selected method-compatible device index.
When free VRAM is null, planning uses total memory, then subtracts the reserve.
Runtime admission later requires a current measurement.

## Target object

| Field | Type | Contract |
| --- | --- | --- |
| `objective` | string | `quality`, `memory`, or `speed` |
| `sequence_length` | integer | Positive training limit, no greater than model context |
| `effective_batch_size` | integer | Positive requested global batch |
| `max_epochs` | integer | Positive full-run epoch count |
| `method_preference` | string or null | Optional executable-method preference |
| `task` | string | Only `sft` is supported |
| `evaluation_fraction` | number | In `[0, 1)` |
| `packing` | boolean | Must remain false in v0.2 |
| `checkpoint_steps` | integer | Positive save and evaluation interval |
| `max_wall_time_minutes` | integer or null | Positive when present, but any value is fail-closed in v0.2 |

## Candidate object

The plan contains four methods crossed with three distributions, for 12 rows.
Every row remains visible even when unsupported.

### Identity, status, and placement

| Field | Type | Meaning |
| --- | --- | --- |
| `candidate_id` | string | `cand_` plus a 20-hex content identity |
| `method` | string | `full`, `lora`, `int8-lora`, or `qlora` |
| `status` | string | `feasible`, `conditional`, `infeasible`, or `unsupported` |
| `feasible` | boolean | True for both `feasible` and `conditional` |
| `rejection_reasons` | string array | Unsupported, infeasible, and conditional reasons in evaluation order |
| `distribution` | string | `single`, `ddp`, or `fsdp` |
| `world_size` | integer | One for single; participating rank count otherwise |
| `device_indices` | integer array | Bound planned devices in rank order |
| `user_reserve_bytes` | integer | Reserve excluded from usable capacity |

Consumers must use `status`, not only `feasible`. A conditional row can become
the recommendation, but unresolved reasons remain binding warnings.

### Precision and training configuration

| Field | Type | Meaning |
| --- | --- | --- |
| `precision` | string | `bf16` when every participating device declares it, otherwise `fp16` |
| `quantization` | string or null | `int8-bitsandbytes`, `nf4-double-quant`, or null |
| `micro_batch_size` | integer | Per-device batch selected from exact divisors up to 32 |
| `gradient_accumulation_steps` | integer | Exact global-batch accumulation |
| `effective_batch_size` | integer | `micro * accumulation * world_size` |
| `rank` | integer | Zero for full training; adapter prior otherwise |
| `alpha` | integer | Zero for full training; `2 * rank` for adapters |
| `learning_rate` | number | Method-class prior |
| `target_modules` | string array | Empty for full; family catalog modules for adapters |

### Resource and decision fields

| Field | Type | Meaning |
| --- | --- | --- |
| `memory` | object | Point components, upper components, uncertainty, and assumptions |
| `required_host_ram_bytes` | integer | Host model-loading prior for the planned rank count |
| `required_disk_bytes` | integer | Staging, pilot, retention, and export prior |
| `checkpoint_retention_bytes` | integer | Three retained checkpoint units |
| `final_export_bytes` | integer | Minimum predicted final export size |
| `preference_score` | number | Negative deterministic rank, or a large negative sentinel for rejected rows |
| `pareto_frontier` | boolean | Nondominated viable row under memory, fidelity, and accumulation criteria |
| `ranking_basis` | string array | Honest statement of the ranking policy |
| `confidence` | string | Currently `uncalibrated-pilot-required` |
| `assumptions` | string array | Formula and policy assumptions |
| `evidence` | string array | Evidence IDs resolved at plan level |

## Memory object

Stored point components are:

- `base_weights_bytes`;
- `quantization_metadata_bytes`;
- `adapter_weights_bytes`;
- `adapter_gradients_bytes`;
- `optimizer_states_bytes`;
- `activations_bytes`;
- `temporary_overhead_bytes`;
- `communication_bytes`;
- `workspace_bytes`;
- `allocator_bytes`; and
- `load_transient_bytes`.

The object also stores `component_upper_bounds`, `safety_margin_bytes`,
`formula_version`, and `assumptions`. Serialization adds calculated
`point_estimate_bytes`, compatibility alias `estimated_peak_bytes`,
`upper_estimate_bytes`, and `uncertainty_bytes`. The user reserve is not a
memory-use component.

## Provenance object

```json
{
  "kind": "user-attested",
  "source": "cli-or-api",
  "observed_at": null,
  "digest": null,
  "detail": null
}
```

`kind` is one of `measured`, `provider-declared`, `user-attested`, `inferred`,
or `unknown`. Provenance describes where a fact came from. It is distinct from
the research and methodology evidence records cited by candidates.

## Evidence record object

Each entry contains `evidence_id`, `claim`, `source`, `source_kind`, `scope`,
`confidence`, and optional `revision`. The plan includes the sorted union of
evidence IDs cited by every candidate, not only the recommendation.

## Content identity

Candidate identity binds normalized model, dataset, hardware, and target facts,
plus the execution strategy, resource fields, status, target modules, and memory
object. Plan identity binds schema and formula versions, normalized facts, all
candidate IDs in order, and the recommended candidate ID.

Evidence-record content, warnings, and rationale are serialized and later bound
by the bundle manifest, but they are not direct `plan_id` inputs in v0.2.

Narrative warnings and rationale do not replace content identity. A payload must
also pass deterministic replanning parity during host static validation.

Compilation rewrites the dataset path and provenance source to bundle-relative
values while retaining the same semantic dataset digest and plan identity.

## Related documentation

- [Dataset schemas](dataset-schemas.md)
- [Evidence records](evidence-records.md)
- [Method registry](method-registry.md)
- [Capability matrix](capability-matrix.md)
- [Bundle manifest](bundle-manifest.md)
- [Validation states](validation-states.md)
