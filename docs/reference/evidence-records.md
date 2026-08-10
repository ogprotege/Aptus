# Evidence Records and Attestations

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Plan reviewers, operators, auditors, and maintainers |
| Authority | Normative v0.2 reference for provenance, cited evidence, measurements, and runtime attestations |
| Last reviewed | 2026-08-10 |
| Next review | 2026-11-01, or sooner when domain, evidence, validation, or execution contracts change |

Aptus separates seven concepts that answer different questions:

| Concept | Question answered | Location |
| --- | --- | --- |
| Provenance | Where did this supplied or observed fact come from? | Model, dataset, hardware, and device records |
| Measurement kind | Was dataset token length measured with the pinned tokenizer or estimated? | Dataset profile |
| Policy decision | What did the current compatibility registry decide from the subject facts? | Plan-level `model_policy_decision` |
| Inspection receipt | Which provider and inferred planning facts were observed at one immutable revision? | Plan-level `inspection_receipt` when used |
| Policy binding | Which exact registered policy path does one candidate match? | Candidate `policy_binding` when matched |
| Evidence record | What source supports a method or estimator claim? | Plan-level `evidence_records` |
| Runtime attestation | What exact artifact and execution fact did Aptus verify? | `validation-report.json`, metrics files, and job completion records |

None of these is a model-quality guarantee. A research citation establishes a
method definition or reported result within its stated scope. It does not prove
that Aptus implements that method, that a candidate fits the supplied machine,
or that a trained model meets the user's target.

## Fact provenance

A provenance object has this shape:

```json
{
  "kind": "measured",
  "source": "local-hardware-probe",
  "observed_at": "2026-07-22T12:00:00+00:00",
  "digest": null,
  "detail": "Probe-specific context"
}
```

| Field | Required | Meaning |
| --- | ---: | --- |
| `kind` | Yes | Controlled origin classification |
| `source` | Yes | Tool, provider, interface, or bundle-relative source label |
| `observed_at` | No | ISO timestamp when the fact was observed |
| `digest` | No | Content or response digest when one is available |
| `detail` | No | Bounded explanatory context |

### Provenance kinds

| Kind | Meaning | Typical use |
| --- | --- | --- |
| `measured` | Aptus observed the fact on a concrete resource or host | Local hardware probe, local dataset bytes and digest |
| `provider-declared` | A provider response declared the fact | Resolved model metadata |
| `user-attested` | The user supplied or confirmed the fact | Manual model shape, license, training permission, hardware profile |
| `inferred` | Aptus derived the fact from other known values | Explicitly identified fallbacks or derived metadata |
| `unknown` | The origin cannot be established | Compatibility input without stronger provenance |

`user-attested` means the value was declared. It does not mean Aptus verified
the declaration. A local probe can replace manual hardware provenance only
when the intended training host is the machine being probed.

During compilation, dataset provenance becomes portable. Its source changes to
`bundle:data/dataset.<source-suffix>`, the copied source artifact. The original
observation timestamp, digest, and detail are retained when present.

## Policy provenance records

`aptus.model-compatibility.v2` separates one policy decision from candidate
feasibility. Its `subject_facts_sha256` covers only compatibility inputs. Stable
reason codes and evidence IDs identify the result. A matched or blocked
registered policy can also name a stable policy ID and semantic version. The
registry currently contains two reviewed version `1.0.0` paths:

- `model.qwen3-moe.mlx-qlora` binds the exact reviewed sparse identity and
  topology row to `mlx-lm.qlora.single.attention-qkvo.v1`.
- `model.qwen2-24l.mlx-qlora` binds a reviewed dense Qwen2 configuration
  footprint to `mlx-lm.qlora.single.dense-causal-lm.v1`. The footprint requires
  the `qwen`, `qwen2`, and `Qwen2ForCausalLM` identity, exactly 24 layers, no
  MoE topology, and a uniform four-bit group-size-64 layout with no module
  overrides. Its adapter targets are `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`.

The Qwen2 policy is a configuration-footprint decision, not an artifact
allowlist or a transferable runtime attestation. A different model artifact can
match the configuration while still requiring its own model-data, measured-
preflight, and pilot evidence.

`aptus.model-inspection-receipt.v1` separately records
`observed_facts_sha256`. That digest covers every provider-declared or inferred
planning fact carried from inspection, not only the compatibility subject. The
receipt lists each covered field's provenance kind, source, observation time,
and resolved revision. Parameters and training permission are excluded and
remain user-attested.

Receipt provenance accepts only `provider-declared` and `inferred`. It covers
every non-null compatibility subject field and includes at least one
provider-declared subject observation. A registered path can impose a stricter
provider-declared field set.

Every candidate in an `aptus.training-plan.v6` plan carries
`model_policy_decision_id`, including candidates that match no policy path.
Only the candidate whose method, distribution, target modules, and runtime
contract match a registered path carries an `aptus.model-policy-binding.v1`.
Its source is `provider-inspection` with a receipt ID or `user-attested` with no
receipt ID.

Receipt and decision hashes are tamper-evident content bindings, not
authenticated signatures. They detect mismatched content but do not prove the
identity of the producer. A present invalid receipt is rejected rather than
silently treated as user-attested.

## Dataset measurement kind

Dataset profiling reports one of two measurement kinds:

| Kind | Meaning |
| --- | --- |
| `tokenizer-measured` | Aptus resolved the pinned model tokenizer and measured sampled token lengths |
| `estimated` | Aptus used its deterministic text-length estimate because tokenizer measurement was not available |

The measurement kind applies to token-length statistics. File size, row count,
schema validation, and SHA-256 identity still come from the local dataset. An
estimated profile must not be described as tokenizer-measured evidence.

## Evidence record schema

Each `EvidenceRecord` contains:

| Field | Required | Meaning |
| --- | ---: | --- |
| `evidence_id` | Yes | Stable registry identifier referenced by candidate records |
| `claim` | Yes | Bounded proposition supported by the source |
| `source` | Yes | URL or Aptus methodology URI |
| `source_kind` | Yes | Source classification, such as `research-paper` |
| `scope` | Yes | Explicit boundary on what the source supports |
| `confidence` | Yes | Evidence-status label, not a numeric probability |
| `revision` | No | Paper or source revision identifier when recorded |

The current confidence labels are:

| Confidence | Interpretation |
| --- | --- |
| `documented` | An official integration or training document states the claim |
| `paper-reported` | A research paper defines or reports the claim within its scope |
| `uncalibrated` | Aptus records an analytic methodology that still requires empirical calibration |
| `implementation-reviewed` | An Aptus compatibility path has passed code and contract review but still requires runtime gates |
| `measured-blocked` | A recorded target-host attempt failed at a named measured admission gate |
| `measured-historical` | A dated runtime result passed for its exact historical artifact and contracts but does not establish current-contract acceptance |

Confidence does not cross scopes. For example, `paper-reported` on a LoRA
definition does not provide runtime evidence for an Aptus LoRA bundle.

## Evidence registry

The registry currently contains these records.

| Evidence ID | Source kind | Confidence | Supported claim and boundary |
| --- | --- | --- | --- |
| `method.full.transformers` | Official documentation | `documented` | Transformers causal language-model training updates model parameters |
| `method.lora.paper` | Research paper | `paper-reported` | LoRA freezes base weights and injects trainable low-rank matrices; no Aptus quality guarantee |
| `method.qlora.paper` | Research paper | `paper-reported` | QLoRA trains adapters through a frozen four-bit base; hardware support remains separate |
| `method.dora.paper` | Research paper | `paper-reported` | DoRA separates magnitude and direction; no Aptus compiler claim |
| `method.bitfit.paper` | Research paper | `paper-reported` | BitFit updates existing bias terms; applicability depends on actual model biases |
| `method.loreft.paper` | Research paper | `paper-reported` | LoReFT learns hidden-representation interventions; an intervention-aware runtime is required |
| `method.aflora.paper` | Research paper | `paper-reported` | AFLoRA dynamically scores and freezes low-rank groups; Aptus does not implement its dynamic state |
| `method.bilora.paper` | Research paper | `paper-reported` | BiLoRA uses bilevel optimization over disjoint partitions; a dedicated trainer is required |
| `method.adalora.paper` | Research paper | `paper-reported` | AdaLoRA changes rank allocation with a scheduled importance state; Aptus does not preserve that state |
| `method.sharelora.paper` | Research paper | `paper-reported` | ShareLoRA shares compatible low-rank factors; serialization and synchronization need separate proof |
| `method.bitsandbytes.int8` | Official documentation | `documented` | Bitsandbytes supports eight-bit loading in supported CUDA environments |
| `estimate.memory.v2` | Aptus methodology | `uncalibrated` | Aptus emits a point estimate and heuristic upper envelope; an exact pilot remains required |
| `policy.qwen3-moe.mlx-qlora.v1` | Aptus compatibility policy | `implementation-reviewed` | The exact reviewed Qwen3 MoE tuple maps to one MLX-LM QLoRA path; runtime validation remains mandatory |
| `admission.qwen3-30b-a3b.memory-blocked.2026-07-28` | Measured admission record | `measured-blocked` | One exact 30B attempt failed live unified-memory admission before model loading; it is refusal evidence, not a passing pilot |
| `policy.qwen2-24l.mlx-qlora.v1` | Aptus compatibility policy | `implementation-reviewed` | The reviewed 24-layer dense Qwen2 four-bit group-size-64 configuration maps to one single-device MLX-LM QLoRA path with seven dense targets; runtime gates remain mandatory |
| `runtime.qwen2-0.5b.mlx-qlora.2026-07-27` | Measured runtime record | `measured-historical` | Two clean runs passed for the exact pinned Qwen2.5 0.5B artifact under training-plan v2 and bundle v2; this is not current v5/v3 acceptance and does not transfer to other matching artifacts |

Canonical source URLs and revisions are serialized into each generated plan.
Consumers should read the record rather than reconstructing a URL from its ID.
The two Qwen2 records intentionally answer different questions: the policy
record supports the reviewed configuration-to-path mapping, while the runtime
record supports only the exact July 27 artifact, host, dataset, runtime, v2
plan, and v2 bundle named by its scope. The
[original 2026-08-05 Qwen2 MLX-LM acceptance](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
is the unchanged historical Phase 6 baseline. The separate
[current-contract evidence at exact source](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
records two fresh, clean `aptus.training-plan.v6` and
`aptus.bundle.v3` `measured-run-pass` repetitions at source
`719255153e3fc7e38e83b5ff826d587e5e58bf80`, tree
`be99f5664ccb580f2600471f1ae3241a294b1a7e`, and bundle fingerprint
`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`.
It supplies current-contract Phase 6 MLX-LM runtime evidence at that exact
source only for its pinned artifact and revision, source and tree, Apple M5 Pro
host, Python/MLX runtime, dataset, policy snapshot, plan, bundle, and
fingerprint. It does not broaden or relabel either canonical evidence record.

The separate [2026-08-06 SmolLM2 CUDA LoRA single-device acceptance
packet](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
is operational runtime-attestation evidence for one exact five-job execution.
It establishes neither repeatability nor another CUDA method, placement,
artifact, device, host, environment, or release claim. The packet is not a
code-owned `EvidenceRecord` entry and this document does not assign it a
canonical plan-level evidence ID. Do not insert a fabricated CUDA evidence ID
into `evidence_records`.

The later [2026-08-10 Phase 5 repeatability anchor
packet](../operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md)
is a separate operational cohort record: five predeclared, protocol-valid
SmolLM2 LoRA single-device slots passed the frozen stability and integrity
contract and established Phase 6 eligibility for that exact anchor scope. It
does not retroactively broaden the August 6 packet, create a plan-level
`EvidenceRecord`, or qualify another method, artifact, host, or environment.

## Candidate evidence mapping

Selectable candidates reference evidence IDs directly.

| Method | Candidate evidence IDs |
| --- | --- |
| `full` | `method.full.transformers`, `estimate.memory.v2` |
| `lora` | `method.lora.paper`, `estimate.memory.v2` |
| `int8-lora` | `method.lora.paper`, `method.bitsandbytes.int8`, `estimate.memory.v2` |
| `qlora` | `method.qlora.paper`, `estimate.memory.v2` |

A candidate that matches a registered policy path also cites that path's
policy-specific evidence. The Qwen2 path therefore adds
`policy.qwen2-24l.mlx-qlora.v1` and
`runtime.qwen2-0.5b.mlx-qlora.2026-07-27`. Carrying the historical runtime ID in
a current plan preserves its scope; it does not relabel the current plan or
bundle as runtime-tested. The separate exact-source refresh applies only when
its exact plan, bundle, artifact, source, host, runtime, dataset, policy
snapshot, and fingerprint bindings match. The policy remains a configuration footprint rather
than an artifact allowlist; other matching artifacts still require their own
model-data, measured-preflight, and pilot gates. The packet does not qualify
CUDA or establish model quality or production throughput.

Research-only and experimental method descriptors reference their own papers,
but they do not enter the v0.2 candidate matrix. Their citations make their
identity and blocker reviewable. They do not make those methods selectable.

The plan-level `evidence_records` array is the sorted union of evidence IDs
used by all candidate records, including rejected candidates. Plan validation
requires every candidate reference to resolve to one plan-level record and
rejects duplicate evidence IDs.

The compiler writes the same plan-level records to `evidence.jsonl`, one JSON
object per line. The bundle manifest binds that file by path, size, and digest.

## Evidence identity rules

Evidence-record content is a direct `plan_id` input. The v6 plan ID binds schema
and formula versions, normalized facts, the semantic policy decision and source,
the `model_policy_snapshot_sha256` binding, the optional inspection receipt with its
nested explanatory decision reason excluded, the sorted canonical evidence
records, ordered candidate IDs, and the recommended candidate ID. Changing a
claim, source, source kind, scope, confidence, revision, or snapshot digest
changes the plan ID. The portable validator also
requires each known evidence ID to resolve to its exact code-owned record and
requires the record set to equal the candidates' cited evidence union.

Within a bundle:

- `plan.json` is bound by the bundle manifest;
- `evidence.jsonl` is independently bound by the bundle manifest;
- candidate evidence references must resolve to exact canonical records inside
  `plan.json`; and
- validation reproduces the plan from its facts and compares candidate and
  recommendation parity.

The evidence registry is code-owned in v0.2. The CLI and API do not accept
arbitrary evidence records from a request.

## Runtime evidence

Runtime evidence is produced by execution, not by citation. The validation
report can contain:

| Field | Meaning |
| --- | --- |
| `runtime_evidence` | Executed command, return code, and bounded output tail from host-driven validation |
| `bindings` | Digests and identities that tie the report to the bundle, data, environment, hardware, plan, candidate, model revision, and metrics |
| `preflight_metrics` | Parsed runtime-specific measured-preflight evidence |
| `pilot_metrics` | Parsed exact model-and-data pilot evidence |
| `final_export` | Verified final export identity and manifest information |
| `measured_run` | Verified full-run metrics and output binding |
| `measured_run_completed_at` | Completion timestamp for a promoted full-run attestation |
| `latest_recheck` | A weaker later recheck retained beside a still-current stronger attestation |

### Standard bindings

Host validation writes these bindings when their source facts are available:

| Binding | Bound value |
| --- | --- |
| `bundle` | Artifact fingerprint of manifested immutable files |
| `dataset` | `plan.dataset.source_sha256`, the copied source-file digest |
| `environment` | Pinned requirement environment identity |
| `hardware` | Runtime hardware identity when measured, otherwise the planned hardware hash |
| `planned_hardware` | Hash of the plan's hardware object |
| `validator` | `aptus-validator-v2` |
| `validated_at` | Validation timestamp |
| `plan_id` | Deterministic plan identifier |
| `candidate_id` | Recommended candidate identifier |
| `model_revision` | Immutable model revision |
| `preflight_metrics` | SHA-256 of `preflight-metrics.json` after measured preflight |
| `pilot_metrics` | SHA-256 of `pilot-output/metrics.json` after a pilot |

Portable validation also verifies the cumulative level and writes a bound
report. Host validation imports those bindings only after the portable command
succeeds and the report is structurally valid.

## Attestation progression

Evidence strength increases through cumulative gates:

```text
contract
  -> static
  -> dependency
  -> model-data
  -> measured-preflight
  -> pilot
  -> execution-approved
  -> measured-run
```

Static success is structural evidence only. Model-data success proves that the
pinned model, tokenizer, canonical rows, target modules, and device path can be
instantiated. CUDA measured preflight adds a synthetic backward and optimizer
step. MLX measured preflight adds bounded real-input optimizer work, exact target
binding, adapter delta, memory, and adapter evidence. CUDA pilot proves bounded
checkpoint continuation across two fresh training processes. MLX pilot proves
one uninterrupted two-update run plus fresh-process adapter reload and bounded
generation. Full-run evidence is promoted only after the execution service
verifies the runtime-specific completed output tree.

The validator preserves a stronger report only when its bindings and artifacts
remain current. If a later weaker recheck runs, `latest_recheck` records that
result without falsely downgrading a still-valid stronger attestation.

## Review guidance

When reviewing a claim, ask the matching question:

- Is the fact measured, provider-declared, user-attested, inferred, or unknown?
- Were token statistics measured with the pinned tokenizer or estimated?
- Does the cited source support this exact claim within the recorded scope?
- Does the method have a selectable compiler and export contract?
- Does runtime evidence bind the current plan, candidate, dataset, hardware,
  environment, metrics, and output artifacts?

Do not substitute one evidence layer for another. A paper cannot replace a
pilot, and a successful pilot cannot establish a legal right to train on the
data.

## Related documentation

- [Plan schema](plan-schema.md)
- [Method registry](method-registry.md)
- [Validation states](validation-states.md)
- [Run states](run-states.md)
- [Bundle manifest](bundle-manifest.md)
- [Dataset schemas](dataset-schemas.md)
- [Facts and provenance methodology](../methodology/facts-and-provenance.md)
- [Memory estimation methodology](../methodology/memory-estimation.md)
- [SmolLM2 CUDA LoRA single-device acceptance](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
- [Qwen2 MLX-LM current-contract evidence at exact source](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
