# Evidence Records and Attestations

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Plan reviewers, operators, auditors, and maintainers |
| Authority | Normative v0.2 reference for provenance, cited evidence, measurements, and runtime attestations |
| Last reviewed | 2026-07-22 |
| Next review | 2026-10-22, or sooner when domain, evidence, validation, or execution contracts change |

Aptus separates four concepts that answer different questions:

| Concept | Question answered | Location |
| --- | --- | --- |
| Provenance | Where did this supplied or observed fact come from? | Model, dataset, hardware, and device records |
| Measurement kind | Was dataset token length measured with the pinned tokenizer or estimated? | Dataset profile |
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

Canonical source URLs and revisions are serialized into each generated plan.
Consumers should read the record rather than reconstructing a URL from its ID.

## Candidate evidence mapping

Selectable candidates reference evidence IDs directly.

| Method | Candidate evidence IDs |
| --- | --- |
| `full` | `method.full.transformers`, `estimate.memory.v2` |
| `lora` | `method.lora.paper`, `estimate.memory.v2` |
| `int8-lora` | `method.lora.paper`, `method.bitsandbytes.int8`, `estimate.memory.v2` |
| `qlora` | `method.qlora.paper`, `estimate.memory.v2` |

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

Evidence content is integrity-bound but is not an input to `plan_id` in v0.2.
That ID binds schema and formula versions, normalized facts, ordered candidate
IDs, and the recommended candidate ID. Changing an evidence claim, source,
scope, confidence label, or revision changes the serialized plan bytes and any
bundle manifest and artifact fingerprint that contains it, but it does not by
itself change `plan_id`.

Within a bundle:

- `plan.json` is bound by the bundle manifest;
- `evidence.jsonl` is independently bound by the bundle manifest;
- candidate evidence references must resolve inside `plan.json`; and
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
| `preflight_metrics` | Parsed synthetic method-preflight evidence |
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
pinned model, tokenizer, canonical rows, target modules, trainable census, and
device path can be instantiated. Measured preflight adds a synthetic backward
and optimizer step. Pilot adds a bounded real-model and real-data training run,
checkpoint and reload checks, metrics, and export evidence. Full-run evidence
is promoted only after the execution service verifies both pending attestations
against the completed output tree.

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
