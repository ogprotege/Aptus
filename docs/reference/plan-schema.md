# Plan Schema

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Planner consumers, compiler authors, reviewers, and integrators |
| Authority | Normative field reference for `aptus.training-plan.v3` |
| Last reviewed | 2026-07-27 |
| Next review | 2026-10-27, or sooner when domain or plan-contract code changes |

An Aptus plan is a canonical semantic record, not a loose set of launch flags.
The current schema identifier is `aptus.training-plan.v3`. Numbers must be
finite JSON values. The self-contained bundle validator recomputes candidate and
plan identities and rejects semantic mutation. Plans with
`aptus.training-plan.v2` or no schema identifier do not contain every fact
required by v3. Aptus preserves those saved bytes, but it does not reinterpret,
compile, or recover them. Create a deterministic v3 plan from the preserved
source facts. Do not relabel the old plan.

## Top-level object

```json
{
  "schema_version": "aptus.training-plan.v3",
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
| `formula_version` | string | Plan-level baseline formula identity; each candidate memory object carries its exact estimator version |
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
| `family` | string | Planner family. Dense adapter paths use `gemma`, `llama`, `mistral`, or `qwen`; the exact sparse row uses `qwen3_moe` |
| `parameters` | integer | Positive exact user-attested total parameter count. This is the resident-weight basis |
| `active_parameters` | integer | Backend-derived logical parameters used per token. This never replaces `parameters` in resident memory estimates |
| `hidden_size` | integer | Positive hidden width |
| `intermediate_size` | integer or null | Optional positive MLP width |
| `layers` | integer | Positive layer count |
| `context_length` | integer | Positive model context limit |
| `license_name` | string | Non-empty user-supplied license label |
| `training_allowed` | boolean | Must be true |
| `architecture` | string | Exact provider architecture when inspected; otherwise defaults to `causal-lm` |
| `model_type` | string or null | Exact provider model type when inspected |
| `quantization_bits` | integer or null | Pinned checkpoint precision from 1 through 16 bits |
| `quantization_layout` | object or null | Canonical MLX groupwise defaults and module overrides; required by the exact Qwen3 MoE row |
| `moe` | object or null | Exact routed-expert topology when present |
| `sparse_layer_count` | integer | Backend-derived sparse decoder-layer count; zero for dense models |
| `tokenizer_id` | string or null | Optional tokenizer override; current builders leave it null |
| `provenance` | object | Field name to provenance object mapping |

Provider inspection can supply identity, quantization, and MoE topology facts,
but it cannot supply `parameters` or set `training_allowed`.

When `quantization_layout` is present, it contains:

| Field | Type | Contract |
| --- | --- | --- |
| `default_bits` | integer | Default checkpoint precision from 1 through 16 bits; must equal `quantization_bits` |
| `default_group_size` | integer | Positive default MLX quantization group size |
| `module_overrides` | array | Canonical module exceptions, sorted by unique `module_path` |

Each module override contains a dotted `module_path`, `bits` from 1 through 16,
and a positive `group_size`. The exact Qwen3 MoE row requires
`default_bits: 4` and `default_group_size: 64`. It also requires exactly one
override for every model layer, and no others. The override for layer `N` is
`model.layers.N.mlp.gate` with `bits: 8` and `group_size: 64`. The array uses
canonical module-path order. Its complete content participates in candidate and
plan identity. Compilation also records the canonical layout SHA-256 and checks
the generated MLX configuration against it.

When `moe` is present, it contains:

| Field | Type | Contract |
| --- | --- | --- |
| `expert_count` | integer | Positive total routed-expert count |
| `experts_per_token` | integer | Positive routed experts selected per token, no greater than `expert_count` |
| `expert_intermediate_size` | integer | Positive intermediate width for each routed expert |
| `decoder_sparse_step` | integer | Positive sparse-block cadence |
| `mlp_only_layers` | integer array | Sorted, unique, zero-based dense-only layer indices within the model |
| `shared_expert_intermediate_size` | integer or null | Positive shared-expert width when declared |

The initial executable sparse row requires `model_type: qwen3_moe`,
`architecture: Qwen3MoeForCausalLM`, the exact mixed quantization layout above,
a complete topology, and no shared expert. It permits only single-device
MLX-LM QLoRA. A different override count, module path, precision, group size,
or ordering is unsupported even when the checkpoint is otherwise four-bit.

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

For a discovered Apple Silicon device, `total_vram_bytes` is a compatibility
capacity from the Metal recommended working set when measurable, otherwise the
unified-memory capacity. It is not dedicated VRAM. `free_vram_bytes` remains
null. When `host_ram_free_bytes` is present, MLX planning uses the lesser of
that live available-memory value and the compatibility capacity, then subtracts
the reserve.

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
| `checkpoint_steps` | integer | Positive CUDA checkpoint and evaluation interval; retained as a plan fact for MLX, whose generated runtime uses non-resumable adapter weight snapshots |
| `max_wall_time_minutes` | integer or null | Positive when present, but any value is fail-closed in v0.2 |
| `training_runtime` | string or null | Explicit `transformers-peft-cuda`, `mlx-lm`, or `pytorch-mps` binding; null requests backend-based inference |

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
| `runtime_contract` | object | Versioned compute, compiler, estimator, evidence, and export binding |

Consumers must use `status`, not only `feasible`. A conditional row can become
the recommendation, but unresolved reasons remain binding warnings.

### Precision and training configuration

| Field | Type | Meaning |
| --- | --- | --- |
| `precision` | string | `bf16` when every participating device declares it, otherwise `fp16` |
| `quantization` | string or null | `int8-bitsandbytes`, `nf4-double-quant`, `mlx-4bit-groupwise`, or null |
| `micro_batch_size` | integer | Per-device batch selected from exact divisors up to 32 |
| `gradient_accumulation_steps` | integer | Exact global-batch accumulation |
| `effective_batch_size` | integer | `micro * accumulation * world_size` |
| `rank` | integer | Zero for full training; adapter prior otherwise |
| `alpha` | integer | Zero for full training; `2 * rank` for adapters |
| `learning_rate` | number | Method-class prior |
| `target_modules` | string array | Empty for full; family catalog modules for adapters. The exact Qwen3 MoE row uses only attention `q_proj`, `k_proj`, `v_proj`, and `o_proj` |

### Resource and decision fields

| Field | Type | Meaning |
| --- | --- | --- |
| `memory` | object | Point components, upper components, uncertainty, and assumptions |
| `required_host_ram_bytes` | integer | Host model-loading prior for the planned rank count |
| `required_disk_bytes` | integer | Staging, pilot, retention, and export prior |
| `checkpoint_retention_bytes` | integer | Conservative retention estimate; it does not make MLX weight snapshots resumable checkpoints |
| `final_export_bytes` | integer | Minimum predicted final export size |
| `preference_score` | number | Negative deterministic rank, or a large negative sentinel for rejected rows |
| `pareto_frontier` | boolean | Nondominated viable row under memory, fidelity, and accumulation criteria |
| `ranking_basis` | string array | Honest statement of the ranking policy |
| `confidence` | string | Currently `uncalibrated-pilot-required` |
| `assumptions` | string array | Formula and policy assumptions |
| `evidence` | string array | Evidence IDs resolved at plan level |

### Runtime contract

Every current candidate contains an `aptus.runtime-contract.v1` object:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Exact runtime contract identity |
| `compute_backend` | string | `cuda` for Transformers/PEFT or `mps` for Apple runtimes |
| `training_runtime` | string | `transformers-peft-cuda`, `mlx-lm`, or `pytorch-mps` |
| `compiler_id` | string or null | Exact compiler identity; null means implementation is required |
| `estimator_id` | string | `aptus-memory-v2`, `aptus-memory-mlx-v2`, or `unavailable` |
| `evidence_requirement` | string | `pilot-required` or `implementation-required` |
| `export_kind` | string or null | Runtime-specific artifact contract; null when no compiler exists |

Runtime and backend pairs are strict. `transformers-peft-cuda` requires CUDA.
`mlx-lm` and `pytorch-mps` require MPS. MLX-LM currently has single-device LoRA
and QLoRA compilers. PyTorch MPS has no compiler and remains
`implementation-required`.

For MLX-LM, `pilot-required` means an uninterrupted pilot from the pinned base,
not CUDA-style checkpoint continuation. A passing pilot can authorize an
uninterrupted full-duration adapter run from the same pinned base. Both actions
record `resume_supported: false`, and every resume argument is rejected. The
MLX compiler has no full-parameter or DoRA path.

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
`formula_version`, and `assumptions`. CUDA candidates use `aptus-memory-v2`.
MLX-LM LoRA and QLoRA candidates use `aptus-memory-mlx-v2`. For MoE models,
base weights use the total resident parameter count while activation terms can
use backend-derived routed activity where the formula states it. Serialization
adds calculated `point_estimate_bytes`, compatibility alias
`estimated_peak_bytes`, `upper_estimate_bytes`, and `uncertainty_bytes`. The
user reserve is not a memory-use component.

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
