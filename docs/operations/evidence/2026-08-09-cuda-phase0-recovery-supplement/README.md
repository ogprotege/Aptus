# Phase 2B CUDA Phase 0 Recovery Supplement

> **Status:** Complete, independently reviewed, and publication-eligible | **Authority:** Sanitized recovery-integrity evidence for the protected August 6 CUDA records; not target-runtime, performance, repeatability, or release-readiness evidence | **Applies to:** The exact August 6 SmolLM2 CUDA LoRA single-device packet and protected Phase 0 recovery | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-09 | **Review by:** Before changing the dependent claim, after any custody or retention failure, or by 2026-11-07

## Result

Phase 2B completed from merged Phase 2A source
`f6a58612263ccd1b7284ffa9f5460631ba64c2e1` without connecting to or
mutating the Ubuntu host. The reviewed sanitizer reconstructed this public
packet only from the two immutable protected Phase 0 copies.

The recovery supplement accounts for all 40 logical rows in the frozen
`aptus.raw-artifact-digests.v1` manifest:

- 39 logical rows are `recovered-matching`;
- the original raw model-file manifest is `not-found` after the bounded Phase 0
  search; and
- the original byte-exact Python test transcript is separately `not-found` and
  was not reconstructed from its historical 550-test summary.

The 39 recovered logical rows resolve to 38 protected payload entries because
one expected digest legitimately appears in two logical rows. The protected
artifact was verified in two distinct physical failure domains, including an
AES-256-encrypted removable off-host copy, and a complete retrieval from that
copy verified all 40 sealed-artifact files. The active retention receipt binds
the evidence through at least 2028-08-09.

## Review and publication

The independent review passed all six required checks: strict public schema,
complete raw-to-public traceability, private-value absence, numeric
recomputation, claim-boundary correctness, and complete sorted checksums.
Finalized bytes were reverified before publication.

The publisher then performed two fresh live eligibility passes and committed
the exact reviewed bytes with no reason codes. The final decision is
`eligible: true` and is bound to:

- campaign `campaign_dd4ddde0cab20ac2f7e9`;
- candidate `candidate_69cc090b8061cc4a086571f2fb9b3f69`;
- decision `decision_55b4e5f4f12d497b4144272c2aa5ebe5`;
- protected recovery manifest
  `f35b3383fd58263e7964f301dcadd9369e7b19b1fa85a2ce5d09e2348058f8b7`;
  and
- publication decision artifact manifest
  `4edb3c58a19f93027ed9ab726eb8830edcbbaf991c88ee4a9d465a4432bceb66`.

[`published/PUBLICATION-SHA256SUMS`](published/PUBLICATION-SHA256SUMS) binds
the complete published inventory. [`published/SHA256SUMS`](published/SHA256SUMS)
separately binds the finalized seven-file candidate reviewed before publication.

## Claim boundary

This supplement establishes only recovery integrity for the protected bytes
bound to the original August 6 packet. It does not retroactively establish:

- timing, telemetry, performance, resource usage, or repeatability;
- the absent raw model-file manifest or Python test transcript;
- another CUDA method, model, host, environment, or placement;
- the Ubuntu source-test gate; or
- product, production, safety, quality, or release readiness.

The original [August 6 acceptance packet](../2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
remains immutable. Phase 3 candidate-selection and exact measurement controls
remain required before any new campaign execution. This Phase 2B record does
not authorize Ubuntu-host mutation by itself.

## Published records

| Record | Purpose |
| --- | --- |
| [`recovery-supplement.json`](published/recovery-supplement.json) | All 40 frozen dispositions, custody projections, retention binding, and the separate transcript-search result |
| [`claim-boundary.json`](published/claim-boundary.json) | Exact allowed and forbidden claim scope |
| [`sanitization-map.json`](published/sanitization-map.json) | Complete JSON-Pointer trace from each public leaf to sealed protected provenance or deterministic derivation |
| [`independent-review.json`](published/independent-review.json) | Distinct reviewer identity and six passing checks |
| [`review-bindings.json`](published/review-bindings.json) | Exact reviewed stage inventory and protected provenance digests |
| [`finalization.json`](published/finalization.json) | Separate finalizer identity and durable prior-review binding |
| [`publication-candidate.json`](published/publication-candidate.json) | Exact custody, external-evidence, sanitizer, review, and finalized-byte candidate |
| [`publication-decision.json`](published/publication-decision.json) | Fresh eligible publication decision with no reason codes |
| [`publication-decision-binding.json`](published/publication-decision-binding.json) | Sealed decision-artifact identity and manifest binding |
| [`SHA256SUMS`](published/SHA256SUMS) | Finalized-candidate checksums |
| [`PUBLICATION-SHA256SUMS`](published/PUBLICATION-SHA256SUMS) | Complete publication checksums |

## Related documentation

- [CUDA campaign Phase 2 tooling contract](../../cuda-campaign-phase2-tooling.md)
- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [State, storage, and retention](../../state-storage-retention.md)
