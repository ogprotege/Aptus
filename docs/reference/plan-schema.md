# Plan Schema

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Planner consumers, compiler authors, reviewers, and integrators |
| Authority | Normative field reference for `aptus.training-plan.v5` |
| Last reviewed | 2026-08-05 |
| Next review | 2026-11-01, or sooner when domain or plan-contract code changes |

An Aptus plan is a canonical semantic record, not a loose set of launch flags.
Its JSON root must be an object; JSON null, arrays, and scalar roots are invalid.
Parser resource failures such as oversized integers or excessive nesting are
reported as controlled invalid input rather than escaping from host or
generated entrypoints. The current schema identifier is
`aptus.training-plan.v5`. Numbers must be finite JSON values. The self-contained
bundle validator recomputes candidate and plan identities and rejects semantic
mutation. The plan binds the exact
canonical model-policy snapshot through `model_policy_snapshot_sha256`. Plans
with `aptus.training-plan.v4`, v3, v2, or no schema identifier lack this v5
binding. Aptus preserves those saved bytes, but it does not reinterpret,
compile, or recover them. Create a deterministic v5 plan from the preserved
source facts. Do not relabel the old plan. A coherent v5 plan also enters
`replan_required` when its decision or snapshot digest no longer matches the
current host registry. Malformed or tampered v5 policy state is invalid input,
not a stale-plan migration.

The current snapshot has two registered policy rows. Adding the dense Qwen2 row
did not change the serialized contract shapes: the plan remains
`aptus.training-plan.v5`, its embedded policy decision remains
`aptus.model-compatibility.v2`, the snapshot remains
`aptus.model-policy-snapshot.v1`, and compiled bundles remain
`aptus.bundle.v3`.

## Top-level object

```json
{
  "schema_version": "aptus.training-plan.v5",
  "plan_id": "plan_0123456789abcdef0123",
  "formula_version": "aptus-memory-v2",
  "model_policy_snapshot_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "model": {},
  "dataset": {},
  "hardware": {},
  "target": {},
  "model_policy_decision": {},
  "model_policy_decision_source": "user-attested",
  "inspection_receipt": null,
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
| `model_policy_snapshot_sha256` | string | Lowercase SHA-256 of the canonical `aptus.model-policy-snapshot.v1` used for the decision |
| `model` | object | Explicit model identity, structure, permission, and provenance |
| `dataset` | object | Source identity and profile |
| `hardware` | object | Planned devices, host capacity, and provenance |
| `target` | object | Requested training policy |
| `model_policy_decision` | object | Versioned compatibility result under `aptus.model-compatibility.v2` |
| `model_policy_decision_source` | string | `provider-inspection` when backed by a valid receipt, otherwise `user-attested` |
| `inspection_receipt` | object or null | Exact `aptus.model-inspection-receipt.v1` observation used for planning; required for `provider-inspection` and forbidden for `user-attested` |
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
| `family` | string | Canonical lowercase planner family. Dense adapter paths use `gemma`, `llama`, `mistral`, or `qwen`; the exact sparse row uses `qwen3_moe` |
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
| `quantization_layout` | object or null | Canonical MLX groupwise defaults and module overrides; required by both current registered Qwen policy rows |
| `moe` | object or null | Exact routed-expert topology when present |
| `sparse_layer_count` | integer | Backend-derived sparse decoder-layer count; zero for dense models |
| `tokenizer_id` | string or null | Optional tokenizer override; current builders leave it null |
| `provenance` | object | Field name to provenance object mapping |

Provider inspection can supply identity, quantization, and MoE topology facts,
but it cannot supply `parameters` or set `training_allowed`. Those two facts
remain user-attested and are excluded from the inspection receipt.

## Model policy decision

Every v5 plan contains one `aptus.model-compatibility.v2` decision. The planner
evaluates it once and links every candidate to its `decision_id`.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Exact `aptus.model-compatibility.v2` identifier |
| `decision_id` | string | `compat_` plus a 20-hex content identity |
| `subject_facts_sha256` | 64-hex string | Digest of compatibility-only model facts |
| `kind` | string | `path-matched`, `family-recognized`, `blocked`, or `unknown` |
| `family` | string or null | Normalized family when known |
| `policy_id` | string or null | Stable registered policy ID for matched or blocked registered-policy results |
| `policy_version` | string or null | Semantic version paired with `policy_id` |
| `paths` | array | Complete registered paths for a `path-matched` result; empty otherwise |
| `reason_codes` | string array | Stable machine-readable result reasons |
| `evidence_ids` | string array | Evidence records supporting the policy result |
| `reason` | string | Human-readable explanation; excluded from decision identity |

`subject_facts_sha256` binds family, raw model type, architecture, layer count,
quantization precision and layout, MoE topology, and compatibility fact errors.
The generic evaluator uses only those fixed compatibility fields, ignores
caller-only metadata, and sorts `fact_errors` before hashing. Any non-empty
error list is handled fail-closed before ordinary policy matching. It does not
stand in for all planning facts. The decision ID also binds the decision kind,
policy identity and version, complete path objects, reason codes, and evidence
IDs. See the [model-policy snapshot reference](model-policy-snapshot.md) for the
portable evaluation order and rule shapes.

The current registry contains two exact configuration rows:

- Qwen3 MoE uses policy `model.qwen3-moe.mlx-qlora` version `1.0.0`, path
  `mlx-lm.qlora.single.attention-qkvo.v1`, and adapter profile
  `attention-qkvo.v1`.
- Dense 24-layer Qwen2 uses policy `model.qwen2-24l.mlx-qlora` version
  `1.0.0`, path `mlx-lm.qlora.single.dense-causal-lm.v1`, and adapter profile
  `dense-causal-lm.v1`.

The Qwen2 row requires family `qwen`, model type `qwen2`, architecture
`Qwen2ForCausalLM`, exactly 24 layers, explicit four-bit metadata, a uniform
group-size-64 layout with no module overrides, and `moe: null`. It is a reviewed
runtime configuration footprint, not an artifact allowlist. The
[2026-08-05 Qwen2 MLX-LM exact-source acceptance](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
records two fresh, clean `measured-run-pass` repetitions under
`aptus.training-plan.v5` and `aptus.bundle.v3` for the exact pinned artifact,
source commit `719255153e3fc7e38e83b5ff826d587e5e58bf80`, source tree,
Apple M5 Pro host, Python/MLX runtime, dataset, policy snapshot, and bundle
fingerprint `ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`.
That result closes the current-source Phase 6 runtime gate only for
that scope. A different matching artifact still requires its own model-data,
measured-preflight, and pilot gates; the result does not qualify CUDA or
establish safety, model quality, performance, production throughput,
production readiness, or release readiness. The [original Phase 6
packet](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
remains the unchanged historical baseline.

Each policy path binds method, distribution, adapter profile, target modules,
`aptus.runtime-contract.v1`, required `model-data`, `measured-preflight`, and
`pilot` levels, and evidence IDs. Policy and path IDs are stable. Any semantic
policy change requires a new policy version or path ID.

## Inspection receipt

A successful bounded provider inspection returns an
`aptus.model-inspection-receipt.v1` object:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Exact receipt contract identifier |
| `receipt_id` | string | `receipt_` plus a 20-hex content identity |
| `model_id` | string | Inspected provider repository |
| `resolved_revision` | string | Immutable provider revision used for every observation |
| `observed_facts_sha256` | 64-hex string | Digest of every inspected planning fact carried into the plan |
| `decision` | object | Complete `aptus.model-compatibility.v2` decision for the inspected facts |
| `provenance_summary` | array | Sorted per-field kind, source, observation time, and resolved revision |
| `provenance_requirement` | string or null | Required provenance kind for a registered policy match |
| `provenance_requirement_met` | boolean | Whether every policy-required field meets that provenance requirement |
| `evaluated_at` | string | Timezone-aware evaluation time |

The observed-facts digest is deliberately broader than
`subject_facts_sha256`. It covers each provider-declared or inferred planning
field actually carried from inspection: architecture, context length, family,
hidden size, intermediate size, layers, license label, raw model type, MoE
topology, quantization precision, and quantization layout when present. Omitted
provider fields remain user-attested. `parameters` and `training_allowed` never
enter the receipt.

Receipt provenance is intentionally narrower than general plan provenance.
Every receipt entry must be `provider-declared` or `inferred`. Every non-null
compatibility subject field must have one sorted receipt entry, and at least one
subject field must be provider-declared. A registered path can require a
stricter provider-declared field set. That set comes from the matched policy,
not from a global model-family rule:

- Qwen3 MoE requires provider-declared `architecture`, `layers`, `model_type`,
  `moe`, `quantization_bits`, and `quantization_layout`.
- Dense Qwen2 requires provider-declared `architecture`, `layers`,
  `model_type`, `quantization_bits`, and `quantization_layout`. It does not
  require a provider-declared `moe` entry; the policy constraint requires the
  normalized subject value to be null.

A supplied receipt is revalidated against the model ID, resolved revision,
observed-facts digest, current policy decision, provenance requirements, and
receipt content identity. A missing receipt selects the explicit
`user-attested` path. A present but malformed, stale, mismatched, or modified
receipt is rejected. Aptus never downgrades a bad receipt to user-attested.

These hashes are tamper-evident content bindings, not authenticated signatures.
They can expose accidental or untrusted mutation after creation. They do not
prove who produced a receipt, so the local client and process boundary remains
trusted.

When `quantization_layout` is present, it contains:

| Field | Type | Contract |
| --- | --- | --- |
| `default_bits` | integer | Default checkpoint precision from 1 through 16 bits; must equal `quantization_bits` |
| `default_group_size` | integer | Positive default MLX quantization group size |
| `module_overrides` | array | Canonical module exceptions, sorted by unique `module_path` |

Each module override contains a dotted `module_path`, `bits` from 1 through 16,
and a positive `group_size`. The dense Qwen2 row requires a uniform layout with
`default_bits: 4`, `default_group_size: 64`, and an empty `module_overrides`
array. Its MLX packed-storage arithmetic uses the declared defaults for every
logical parameter.

The exact Qwen3 MoE row uses the same four-bit group-64 defaults but requires
exactly one override for every model layer, and no others. The override for
layer `N` is `model.layers.N.mlp.gate` with `bits: 8` and `group_size: 64`.
The array uses canonical module-path order. Complete layout content participates
in candidate and plan identity. Compilation also records the canonical layout
SHA-256 and checks the generated MLX configuration against it.

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
| `model_policy_decision_id` | string | Required link to the plan's `compat_` decision for every candidate |
| `policy_binding` | object or null | Exact registered policy path binding; non-null only when this candidate matches an emitted path |

Consumers must use `status`, not only `feasible`. A conditional row can become
the recommendation, but unresolved reasons remain binding warnings.

Every candidate carries the plan decision link, including dense, blocked,
unknown, unsupported, and infeasible rows. Only a candidate whose method,
placement, target modules, and runtime contract exactly match a registered path
may carry a policy binding. Every other candidate serializes
`"policy_binding": null`.

An `aptus.model-policy-binding.v1` object contains `decision_id`,
`subject_facts_sha256`, `policy_id`, `policy_version`, `path_id`, `source`,
`inspection_receipt_id`, `reason_codes`, and `evidence_ids`. Its source must
equal the plan source. A provider-inspection binding requires the plan receipt
ID. A user-attested binding must use `inspection_receipt_id: null`.

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
| `target_modules` | string array | Empty for full and for unsupported adapters on an unregistered family; otherwise the exact family catalog modules. Qwen3 MoE uses attention `q_proj`, `k_proj`, `v_proj`, and `o_proj`; dense Qwen2 additionally uses `gate_proj`, `up_proj`, and `down_proj` |

### Resource and decision fields

| Field | Type | Meaning |
| --- | --- | --- |
| `memory` | object | Point components, upper components, uncertainty, and assumptions |
| `required_host_ram_bytes` | integer | Host model-loading prior for the planned rank count |
| `required_disk_bytes` | integer | Staging, pilot, retention, and export prior |
| `checkpoint_retention_bytes` | integer | Conservative retention estimate; zero only for an unsupported adapter with no targets or trainable parameters. It does not make MLX weight snapshots resumable checkpoints |
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

For a dense layout with no module overrides, MLX QLoRA storage is derived from
the bound defaults: weight bytes are `parameters * default_bits / 8`, and
affine scale-and-bias metadata bytes are
`parameters * 4 / default_group_size`, each rounded to an integer. Nonempty
module overrides still require the reviewed MoE topology needed to price their
parameter scope.

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
evidence records cited by every candidate, not only the recommendation. Each
known evidence ID must resolve to its exact code-owned canonical record.

## Content identity

Candidate identity binds normalized model, dataset, hardware, and target facts,
plus the execution strategy, resource fields, status, target modules, memory
object, policy decision link, and optional exact path binding. Plan identity
binds schema and formula versions, normalized facts, the semantic policy
decision excluding its explanatory `reason`, its provider-inspection or
user-attested source, the optional inspection receipt with that same nested-decision
exclusion, `model_policy_snapshot_sha256`, the complete sorted canonical
evidence records, all candidate IDs in order, and the recommended candidate ID.
Changing an evidence claim, source, source kind, scope, confidence, or revision
changes plan identity and fails canonical evidence-registry validation unless
the evidence ID changes with the code-owned record. Changing the snapshot digest
independently changes plan identity and must match the current host registry for
host-managed use.

Narrative warnings and rationale do not replace content identity. A payload must
also pass deterministic replanning parity and current-policy validation during
loading, compilation, recovery, and host-managed admission. Host static
validation reports a currency mismatch as an invalid
`POLICY_SNAPSHOT_DIGEST` finding. Every v4, v3, v2, or schema-less plan requires
replanning. A coherent v5 plan also requires replanning after a snapshot-digest
or policy-semantic change, including a policy-version change, policy addition
or removal, or changed registered path. For a same-schema v5 policy change,
loaders and host-managed workflows surface `replan_required` only after the
saved decision, receipt, candidate links and bindings, candidate IDs,
recommendation, evidence, and plan ID form a coherent historical chain. Broken
dependencies are malformed or tampered input, not legitimate stale policy
state.
Historical classification uses the persisted decision and its internally
consistent candidate targets. It does not reinterpret the old plan through a
newer family-target catalog. Current plans still validate against the current
catalog, so copied targets on an unknown family fail closed. Malformed JSON
scalar types return validation errors instead of escaping as runtime failures.

When a bundle is available, installed-host currency checks use its validated,
digest-bound historical snapshot to recheck the old policy-specific receipt
requirements before returning `replan_required`. A standalone plan does not
carry that snapshot payload. If its historical policy definition is no longer
in the installed registry, stale classification conservatively requires every
non-null compatibility field except the normalized family alias to remain
provider-declared. A downgraded inferred field therefore stays invalid instead
of being relabeled as legitimate stale state.

The package-free generated validator evaluates the plan against the canonical
snapshot embedded in its bundle. That proves frozen snapshot integrity and
decision parity, but portable validation cannot determine host policy currency
or know whether an installed host's current registry has advanced. Installed
Aptus supplies the current host snapshot during managed admission, pilot
authorization, worker launch, and the completion verification and promotion
transaction. It compares the current digest with the plan, manifest, and
embedded file; only that host context establishes policy currency.

Compilation rewrites the dataset path and provenance source to bundle-relative
values while retaining the same semantic dataset digest and plan identity.

## Related documentation

- [Dataset schemas](dataset-schemas.md)
- [Evidence records](evidence-records.md)
- [Model-policy snapshot](model-policy-snapshot.md)
- [Method registry](method-registry.md)
- [Capability matrix](capability-matrix.md)
- [Bundle manifest](bundle-manifest.md)
- [Validation states](validation-states.md)
