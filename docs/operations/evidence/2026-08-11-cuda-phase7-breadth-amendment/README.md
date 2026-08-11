# CUDA Phase 7 Architecture-Breadth Amendment

> **Status:** Complete and independently reviewed; effective only after merge; authorizes one three-slot Qwen3-0.6B LoRA exploratory cell and no other breadth cell | **Authority:** Sanitized exact-provider, artifact, license, planner-admission, and pre-execution review evidence; not runtime, stability, model-quality, production-safety, or Phase 8 evidence | **Applies to:** The Phase 7 architecture-breadth extension on the intended Ubuntu RTX 3050 host after the reviewed same-family cohort | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-11 | **Review by:** Before sealing a breadth attempt ledger, changing any frozen artifact or method, or activating Phase 8

## Decision

The reviewed amendment admits exactly one architecture-breadth cell:
`Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca` with single-device BF16 LoRA.
After this amendment merges, that cell receives exactly three Phase 7
exploratory slots with the already frozen seeds `6101`, `6203`, and `6301`, in
that order, and the unchanged 7,200-second safety ceiling.

No attempt ledger existed and no training began before independent review.
This packet therefore authorizes a later ledger; it does not report a runtime
result.

## Frozen provider candidates

| Repository | Exact revision | License disposition | Provider/Aptus disposition | Admitted methods |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 reviewed for repository-owned synthetic training | Public dense text artifact; exact inspection passed; local artifact and targets verified | `lora` |
| `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | Manual Gemma license not accepted | Provider metadata frozen; exact file inspection returned 401 and failed closed | None |
| `mistralai/Mistral-7B-v0.3` | `caa1feb0e54d415e2df31207e5f4e273e33509b1` | Apache-2.0 reviewed for repository-owned synthetic training | Public dense text artifact; exact inspection passed; no planner-admitted method on the frozen host | None |

These are the only three breadth repositories named by the frozen protocol,
and this amendment freezes at most one dense text artifact from each. Gemma is
excluded from this cohort rather than held as an informal retry. Mistral is
planner-ineligible rather than replaced. A later change to either disposition
requires another reviewed amendment before a new ledger exists.

## Exact Qwen artifact verification

The admitted Qwen execution artifact contains seven files and 1,519,182,365
bytes. Every provider digest matched the local immutable snapshot. The local
safetensors count is exactly 751,632,384 parameters. The seven LoRA targets
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`
each occur in all 28 transformer layers.

The exact tokenizer produced a sealed 512-row manifest with SHA-256
`5b1bc203cb312954967eb201d6d7136a6f607c8327184493fc9469dda26f6dfb`.
This is tokenizer and target-module evidence, not a model-data validation or
training pass.

## Method admission

The exact planner admitted Qwen LoRA with candidate
`cand_562511657f3a989c33fb` and an estimated upper device-memory envelope of
4,557,722,668 bytes. Qwen Full was infeasible because its point estimate
exceeded usable device memory. INT8-LoRA and QLoRA were not admitted because
the frozen host contract does not declare runtime-native eight-bit or four-bit
support. An installed bitsandbytes environment does not manufacture those
hardware facts.

All Mistral methods were either infeasible or unsupported. In particular,
Mistral LoRA's estimated upper device-memory envelope was 25,960,739,067 bytes,
well outside the frozen single-device budget. No model was substituted because
of those outcomes.

## Evidence and custody

[`amendment.json`](amendment.json) publishes the exact revisions, execution
file digests, license decisions, inspections, and planner dispositions.
[`sanitization-map.json`](sanitization-map.json) records the protected-to-public
projection. [`independent-review.json`](independent-review.json) binds and
reviews this public packet. [`SHA256SUMS`](SHA256SUMS) binds every public file.

Protected source records, the downloaded model snapshot, raw host paths,
machine and network identifiers, and operator scripts remain outside Git. The
sealed amendment input, artifact verification, tokenizer manifest, and
independent raw-record review have a verified off-host copy.

## Phase boundary

After merge, the next action is to seal and review a new three-slot Phase 7
Qwen LoRA ledger, then execute only those slots subject to the unchanged static,
dependency, model-data, measured-preflight, pilot, live-admission, capture,
stability, custody, and no-replacement rules. Phase 8 remains unauthorized.

## Claim boundary

This packet establishes a reviewed pre-execution breadth contract. It does not
establish that Qwen trains successfully, that its three runs are stable, that
Gemma or Mistral is supported, that any method is superior, or that Phase 8 may
begin.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [Machine-readable protocol](../../../reference/cuda-campaign-protocol.v1.json)
- [Reviewed Phase 7 same-family cohort](../2026-08-11-cuda-phase7-same-family-stability/README.md)
