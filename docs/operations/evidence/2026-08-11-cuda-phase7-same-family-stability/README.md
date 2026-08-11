# CUDA Phase 7 Same-Family Stability

> **Status:** Complete and independently reviewed; all three planner-admitted same-family cells passed the frozen three-slot stability contract; architecture breadth requires a separate reviewed protocol amendment and Phase 8 is not authorized | **Authority:** Sanitized exact-host Phase 7 planning, runtime, stability, custody, and review evidence; not model-quality, broad-performance, production-safety, or release-readiness evidence | **Applies to:** The new frozen Phase 7 same-family cohort on the intended Ubuntu RTX 3050 host at exact merged source `412095b` | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-11 | **Review by:** Before architecture-breadth activation, Phase 8 activation, protocol amendment, or host/cooling change

## Result

A new Phase 7 cohort completed every planner-admitted same-family cell. The
135M LoRA, 135M Full, and 360M LoRA cells each produced three of three
predeclared, protocol-valid, 128-step exploratory passes without replacement.
All three cells passed the frozen duration, peak-device-memory, telemetry,
optimizer-step, seal, copy, and retrieval stability checks.

The immutable ledger contains 18 exploratory slots. Nine started and passed;
the other nine remain `planned-not-started` because 360M Full and both 1.7B
methods were not admitted by the exact planner. Three conditioning artifacts
also passed. All 12 started artifacts passed deep seal verification, off-host
copy verification, and fresh retrieval verification.

This cohort did not resume or alter the earlier stopped Phase 7 cohort. It was
pre-reviewed before any conditioning or exploratory outcome, used the same
frozen models, methods, seeds, order, dataset, and no-replacement rule, and ran
at the exact merged admission-correction source.

## Stable cells

| Cell | Three observed training durations (ns) | Duration MAD / median | Max / min | Peak-device-memory range | Result |
| --- | --- | ---: | ---: | ---: | --- |
| 135M LoRA | 76,013,306,915; 75,948,138,933; 75,890,722,992 | 0.000755989 | 1.001615269 | 2,097,152 bytes | Stable |
| 135M Full | 55,094,512,614; 54,902,125,720; 54,904,845,458 | 0.0000495355 | 1.003504179 | 119,537,664 bytes | Stable |
| 360M LoRA | 125,225,359,258; 124,400,956,191; 124,156,800,328 | 0.001962653 | 1.008606528 | 2,097,152 bytes | Stable |

Every run completed exactly 128 non-skipped optimizer steps with telemetry
coverage 1.0. Maximum telemetry gaps ranged from 1.040169513 to 1.107246711
seconds, below the frozen 2.5-second limit.

## Admission correction and cohort boundary

The earlier cohort stopped when production admission evaluated only its first
120 thermal samples even though the frozen protocol allowed up to 1,800 seconds
to obtain a qualifying 120-sample window. A protected, training-free 600-sample
diagnosis showed that qualifying windows existed. The correction merged in
source `412095bd66618fee9d3e1936e79b90da12a4c61b`; it searches successive
120-sample windows within the unchanged acquisition deadline while preserving
the full acquisition clock and selected-window offset.

The correction did not retroactively activate the historical slot. The new
cohort has comparison ID `cohort_e997f7c26410b97fef03`, and its pre-execution
review explicitly verified that the historical ledger was not resumed and no
outcome existed before review.

## Planner ledger

The exact planner admitted 135M LoRA, 135M Full, and 360M LoRA. It rejected
360M Full because even its point estimate exceeded usable per-device memory.
It produced no feasible or conditional exact plan for 1.7B LoRA or Full on the
single 8 GiB device. Those three rejected cells account for all nine
`planned-not-started` exploratory slots.

The three immutable model revisions and their 512-row tokenizer manifest were
freshly verified. The common token manifest remained 63,234 bytes with SHA-256
`fa8a4c9223e47fa95cb163db871c35159978b92c4ea559b95e8719697c7be9f6`.

## Evidence and custody

[`phase7-outcome.json`](phase7-outcome.json) publishes the exact bindings,
planner dispositions, stable-cell measurements, and custody counts.
[`sanitization-map.json`](sanitization-map.json) records how protected records
project to this packet. The procedurally separate deterministic review is in
[`independent-review.json`](independent-review.json). [`SHA256SUMS`](SHA256SUMS)
binds the packet.

Raw logs, machine and network identifiers, working paths, job state, model and
checkpoint bytes, archives, and unredacted telemetry remain outside Git. The
protected Ubuntu record and its Mac failure-domain copy retain the raw seals,
manifests, receipts, fresh restorations, plans, contexts, freezes, and operator
records.

## Phase boundary

The same-family portion of Phase 7 is complete and reviewed. Architecture
breadth has not activated: the frozen protocol requires the exact artifact,
revision, license, admitted method, inspection, and digests to be fixed in a
separate reviewed amendment before any breadth ledger or result exists.
Phase 8 remains unauthorized until that Phase 7 boundary is resolved and
reviewed.

## Claim boundary

This packet establishes three stable exact-host same-family Phase 7 cells and
the no-replacement disposition of the full 18-slot ledger. It does not compare
model quality, establish method superiority, qualify 1.7B or another model
family, authorize Phase 8, generalize to another host or configuration, or
support a production-safety or release-readiness claim.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [Machine-readable protocol](../../../reference/cuda-campaign-protocol.v1.json)
- [Historical stopped Phase 7 cohort](../2026-08-10-cuda-phase7-scale-staircase/README.md)
- [Phase 6 confirmatory stability](../2026-08-10-cuda-phase6-confirmatory-stability/README.md)
