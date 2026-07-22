# Plan Schema

The current schema identifier is `aptus.training-plan.v2`. A plan is a canonical
semantic record, not a loose set of CLI flags.

## Top-level shape

```json
{
  "schema_version": "aptus.training-plan.v2",
  "plan_id": "plan_<20 hex>",
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

## Bound facts

`model` contains provider ID, immutable revision, architecture facts, license
label, permission confirmation, tokenizer reference, and field provenance.

`dataset` contains the resolved local source, SHA-256, format, schema,
row and token statistics, canonical-size statistics, warnings, and provenance.

`hardware` contains device records, total and free VRAM, capability flags, host
RAM, free host RAM, reserve, free disk, and provenance.

`target` contains supervised task, objective, sequence length, effective batch,
epochs, optional method preference, evaluation fraction, packing flag,
checkpoint interval, and optional wall-time target. Packing and an enforced
wall-time target are fail-closed in v0.2.

## Candidate shape

Each candidate records:

- identity, method, status, distribution, world size, and device indices;
- precision and quantization;
- micro batch, accumulation, and exact effective batch;
- adapter rank, alpha, learning rate, and target modules;
- point and upper memory components;
- required host RAM, disk, checkpoint retention, and final export bytes;
- assumptions, evidence IDs, confidence, unsupported reasons, and ranking basis.

`feasible=true` covers both `feasible` and `conditional` status. Consumers must
read `status` and reasons rather than relying on the boolean alone.

## Identity

Plan and candidate IDs are recomputed from canonical semantic payloads. IDs are
not user labels. Editing a bound field without recomputing identity makes the
contract invalid.

The source dataset digest is also checked when compilation and runtime require
the source content.
