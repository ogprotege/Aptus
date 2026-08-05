# Model-Policy Snapshot

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Planner consumers, compiler and validator authors, operators, and security reviewers |
| Authority | Normative reference for `aptus.model-policy-snapshot.v1` |
| Last reviewed | 2026-08-05 |
| Next review | 2026-11-01, or sooner when model-policy registry, snapshot, or validation code changes |

An Aptus model-policy snapshot is the portable, data-only form of the host
model-compatibility registry. The compiler writes it to
`policy/model-policy-snapshot.v1.json` and copies the package-independent
`policy_snapshot.py` evaluator into every bundle. Together they let a
package-free validator reproduce the saved `aptus.model-compatibility.v2`
decision without importing Aptus.

The snapshot is an integrity and reproducibility contract. It is not durable
authorization that the policy remains current. A transferred package-free
bundle can validate only its embedded frozen snapshot. Installed Aptus performs
the separate current-host-registry comparison described below.

The current registry contains two ordered policy rows, but this additive
registry change does not alter the contract shape. The snapshot remains
`aptus.model-policy-snapshot.v1`, decisions remain
`aptus.model-compatibility.v2`, plans remain `aptus.training-plan.v5`, and
bundles remain `aptus.bundle.v3`.

## Canonical bytes and digest

The canonical snapshot bytes are UTF-8 JSON with:

- object keys sorted lexicographically;
- compact `,` and `:` separators;
- non-ASCII text retained as UTF-8;
- non-finite numbers rejected; and
- exactly one trailing line feed.

`model_policy_snapshot_sha256` and `policy_snapshot_sha256` are the lowercase
64-character hexadecimal SHA-256 of those exact bytes. Array order remains
digest-significant. The host generator therefore emits every registry list in
deterministic order; `dense_families` and `sparse_identity_markers` must also be
sorted and unique.

## Snapshot object

The current host-generated object contains these required fields:

| Field | Type | Contract |
| --- | --- | --- |
| `schema_version` | string | Exact value `aptus.model-policy-snapshot.v1` |
| `compatibility_schema_version` | string | Non-empty unpadded decision schema, currently `aptus.model-compatibility.v2` |
| `dense_families` | string array | Sorted unique canonical families that may return `family-recognized`; the array may be empty |
| `sparse_identity_markers` | string array | Non-empty sorted unique markers used for fail-closed sparse classification |
| `reasons` | object | Non-empty unpadded reason key to human-readable reason text |
| `policies` | array | Ordered registered policy objects |

`reasons` must define the evaluator keys `identity`, `layout`, `topology`,
`shared`, `four_bit`, `invalid`, `matched`, `dense`, `sparse`, and `unknown`.
Constraint and policy result reason keys must resolve through this object.

Every policy requires:

| Field | Type | Contract |
| --- | --- | --- |
| `policy_id` | string | Non-empty unpadded ID, unique within the snapshot |
| `policy_version` | string | Non-empty unpadded semantic policy version |
| `family` | string | Non-empty unpadded canonical family |
| `claims` | object | Exact `any_identity` claim shape described below |
| `constraints` | array | Non-empty ordered constraint list with exactly one `exact_identity` constraint |
| `paths` | array | Non-empty registered portable path list |
| `matched_reason` | string | Key present in `reasons` |
| `matched_reason_codes` | string array | Non-empty unique unpadded machine-readable codes |
| `evidence_ids` | string array | Non-empty unique unpadded evidence IDs |
| `required_provenance_fields` | string array, optional | Non-empty unique unpadded fields required from provider inspection |

`claims` contains only `any_identity`. That value is a non-empty mapping whose
keys are drawn from `family`, `model_type`, and `architecture`; each value is a
non-empty unique unpadded string list. Claims may intentionally name only a
subset of the three exact-identity fields. For every field that is claimed, the
corresponding exact-identity value must appear in its claim list. The
`exact_identity` constraint itself still supplies all three fields. This lets a
policy claim a distinguishing provider model type or architecture without
claiming every model in a broader normalized family.

## Constraint objects

Every constraint has a supported `kind`, a non-empty `reason` key defined by
the snapshot, and a non-empty `reason_code`. Its remaining fields are exact for
the selected kind:

| Kind | Additional fields | Contract |
| --- | --- | --- |
| `exact_identity` | `values` | Exact `family`, `model_type`, and `architecture` mapping with non-empty unpadded text values; exactly one per policy |
| `quantization_layout` | `default_bits`, `default_group_size`, `override_module_template`, `override_bits`, `override_group_size` | Every numeric operand is a positive non-boolean integer; the template contains exactly one plain `{layer}` placeholder and no other placeholder |
| `sparse_topology` | None | Requires a structurally usable sparse cadence and at least one sparse decoder layer |
| `no_shared_expert` | None | Requires an MoE mapping whose `shared_expert_intermediate_size` lookup is absent or JSON null |
| `field_equals` | `field`, `value` | `field` is one of the fixed compatibility-subject fields; its subject lookup must compare equal to `value` using the evaluator's ordinary equality operation |

Unknown kinds, missing or additional operand fields, malformed templates, and
undefined reasons invalidate the snapshot before evaluation.

For `sparse_topology`, `moe` must be a mapping and `layers` must be a positive
non-boolean Python integer. `decoder_sparse_step` must also be a positive
non-boolean integer, and `mlp_only_layers` must be a list whose entries pass the
evaluator's Python-integer and zero-through-`layers - 1` range checks. The
constraint matches only if at least one zero-based layer index satisfies
`(index + 1) % decoder_sparse_step == 0` and is absent from the set of
`mlp_only_layers`.

For `quantization_layout`, the subject layout must be a mapping and `layers`
must be a positive non-boolean Python integer. Its defaults compare equal to the
constraint's `default_bits` and `default_group_size`. Its `module_overrides`
must equal the complete expected list: one exact object with `module_path`,
`bits`, and `group_size` for every rendered layer, ordered by the decimal layer
index as text (`0`, `1`, `10`, `11`, ..., `2`, ...). Extra, missing, reordered,
or differently valued override objects do not match.

`field_equals` intentionally mirrors the Python evaluator's ordinary equality
semantics rather than a type-strict JSON comparison. For example, JSON `true`
and the number `1` compare equal. Policy authors must account for that behavior
when choosing a field and value.

## Current registry policies

The snapshot currently carries these two registry-driven policies. Both emit a
single-device MLX-LM QLoRA path and remain conditional on `model-data`,
`measured-preflight`, and `pilot` validation.

### Qwen3 MoE

- Policy: `model.qwen3-moe.mlx-qlora`, version `1.0.0`.
- Exact identity: family and model type `qwen3_moe`, architecture
  `Qwen3MoeForCausalLM`.
- Layout and topology: four-bit group-64 defaults, one eight-bit group-64
  `model.layers.N.mlp.gate` override per layer, a usable sparse topology, and no
  shared expert.
- Path: `mlx-lm.qlora.single.attention-qkvo.v1` with adapter profile
  `attention-qkvo.v1` and targets `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
- Provider-inspection requirement: `architecture`, `layers`, `model_type`,
  `moe`, `quantization_bits`, and `quantization_layout` must all be
  provider-declared at the resolved revision.

### Dense Qwen2, 24 layers

- Policy: `model.qwen2-24l.mlx-qlora`, version `1.0.0`.
- Claims: provider model type `qwen2` or architecture `Qwen2ForCausalLM`.
  Family `qwen` remains an exact-identity constraint but is deliberately not a
  broad claim.
- Exact configuration footprint: family `qwen`, model type `qwen2`,
  architecture `Qwen2ForCausalLM`, exactly 24 layers, explicit four-bit
  metadata, no MoE topology, and a uniform group-64 layout:

  ```json
  {
    "default_bits": 4,
    "default_group_size": 64,
    "module_overrides": []
  }
  ```

- Path: `mlx-lm.qlora.single.dense-causal-lm.v1` with adapter profile
  `dense-causal-lm.v1` and targets `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`.
- Provider-inspection requirement: `architecture`, `layers`, `model_type`,
  `quantization_bits`, and `quantization_layout` must all be provider-declared
  at the resolved revision. Dense topology is enforced by the `moe: null`
  constraint rather than by requiring a provider-declared `moe` field.

The Qwen2 row describes a reviewed configuration footprint, not an artifact
allowlist. The
[2026-08-05 Qwen2 MLX-LM acceptance](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
records two clean `measured-run-pass` repetitions under the current
`aptus.training-plan.v5` and `aptus.bundle.v3` contracts for the exact pinned
artifact, source commit, Apple M5 Pro host, Python/MLX runtime, dataset, and
policy snapshot. It closes Phase 6's current-source runtime gate only for that
scope. Another matching artifact still has to pass its own model-data,
measured-preflight, and pilot gates. The result does not qualify CUDA or
establish model quality or production throughput.

## Portable path objects

Each `paths` entry has exactly these fields:

| Field | Type | Contract |
| --- | --- | --- |
| `path_id` | string | Non-empty unpadded ID, unique across every policy in the snapshot |
| `method` | string | Non-empty unpadded Aptus method |
| `distribution` | string | Non-empty unpadded placement |
| `adapter_profile_id` | string or null | Bound adapter profile when the path uses one |
| `target_modules` | string array | Non-empty unique unpadded module names |
| `runtime_contract` | object | Exact portable runtime contract shape below |
| `required_validation_levels` | string array | Non-empty unique unpadded levels |
| `evidence_ids` | string array | Non-empty unique unpadded evidence IDs |

The nested runtime contract has exactly `compute_backend`, `training_runtime`,
`compiler_id`, `estimator_id`, `evidence_requirement`, `export_kind`, and
`schema_version`. Every value is non-empty unpadded text except that the
portable structural validator also permits JSON null for `compiler_id` and
`export_kind`. Current host-generated executable paths use non-null identifiers.

## Compatibility subject and identity

The generic evaluator constructs one normalized subject from exactly these
compatibility fields:

- `family`;
- `model_type`;
- `architecture`;
- `layers`;
- `quantization_bits`;
- `quantization_layout`;
- `moe`; and
- `fact_errors`, sorted before hashing.

Missing `family`, `model_type`, `architecture`, `layers`, `quantization_bits`,
`quantization_layout`, or `moe` fields become JSON null. An omitted
`fact_errors` field becomes an empty list; a supplied list is sorted before
hashing. Caller-only planning or metadata fields are ignored, so adding
unrelated data cannot change `subject_facts_sha256` or the decision. Reordering
an otherwise identical `fact_errors` list also cannot change the result.

Any non-empty `fact_errors` list is handled before ordinary policy matching and
fails closed. A claimed registered identity can retain its policy identity and
evidence while reporting either its exact-identity failure or
`invalid-compatibility-facts`; an unclaimed subject reports the generic invalid
result. Other constraints and paths are not evaluated as if malformed facts
were sound.

With no fact errors, policies are evaluated in snapshot order. The first policy
whose eligible identity claims match is checked in constraint order. A dense
policy claim is not eligible to capture a subject carrying sparse identity
markers unless that policy's own exact identity is also sparse. The first
failed constraint produces `blocked`; a complete match produces `path-matched`
with the policy's full portable path list. With no claimed policy, sparse
evidence produces a fail-closed `blocked` result, a listed dense family produces
`family-recognized`, and every other subject produces `unknown`.

The four decision outcomes are therefore:

| Kind | Meaning |
| --- | --- |
| `path-matched` | One reviewed policy and all its constraints matched; registered paths are emitted |
| `blocked` | Invalid compatibility facts, a claimed-policy constraint failure, or unreviewed sparse evidence failed closed |
| `family-recognized` | No registered path matched, but the family is in the snapshot's reviewed dense-family set |
| `unknown` | No registered policy, sparse marker, or reviewed dense family matched |

The decision identity includes the compatibility decision `schema_version`,
normalized subject digest, outcome kind, family, policy identity and version,
complete paths, reason codes, and evidence IDs. Aptus serializes that identity
as compact sorted-key UTF-8 JSON without a trailing line feed, hashes it with
SHA-256, and forms `decision_id` as `compat_` plus the first 20 lowercase hex
characters. The human-readable `reason` is explanatory and excluded from that
identity.

## Bundle integrity and host currency

`aptus.training-plan.v5` binds the canonical digest in
`model_policy_snapshot_sha256`. `aptus.bundle.v3` binds the same digest in
`policy_snapshot_sha256`, names the exact snapshot path, and manifests the
snapshot file by size and digest. Portable validation checks that chain,
canonical bytes, snapshot structure, and exact decision parity.

Installed host validation additionally compares a fourth `host` digest from
the current registry. The four named bindings are `snapshot`, `plan`,
`manifest`, and `host`; each must be lowercase 64-character hexadecimal text.
A host mismatch is a policy-currency failure even when the frozen bundle is
internally coherent.

Installed loading, compilation, recovery, job submission, pilot authorization,
worker launch, and the completion verification and promotion transaction
require the current host decision and digest. A coherent v5 mismatch requires
deterministic replanning. Malformed or tampered v5 state remains invalid input
rather than a migration case. A package-free generated validator has no current
registry and must not claim host currency from frozen-snapshot integrity.

## Failure behavior

The snapshot root must be a JSON object. Invalid UTF-8, invalid JSON, and parser
resource failures are snapshot JSON failures. JSON null or another non-object
root, malformed fields, invalid constraint operands, and semantic traversal
failures are snapshot-contract failures. Host validation records typed
`POLICY_SNAPSHOT_*` findings rather than allowing parser or evaluator exceptions
to escape. Generated validators also fail closed with controlled errors.

The finding inventory and the distinction between snapshot JSON, contract,
canonicalization, path, digest, and missing-file failures are defined in
[Error and finding codes](error-codes.md).

## Related documentation

- [Plan schema](plan-schema.md)
- [Bundle manifest](bundle-manifest.md)
- [Validation states](validation-states.md)
- [Error and finding codes](error-codes.md)
- [Data and identity flow](../architecture/data-and-identity-flow.md)
- [Changing contracts](../contributing/changing-contracts.md)
