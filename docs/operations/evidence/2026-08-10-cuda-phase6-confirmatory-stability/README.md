# CUDA Phase 6 Full Confirmatory Stability

> **Status:** Complete; Full established confirmatory stability and Phase 7 is authorized | **Authority:** Sanitized exact-host Phase 6 replacement-cohort stability and custody evidence; not model-quality, broad-performance, production-safety, or release-readiness evidence | **Applies to:** The frozen SmolLM2 135M Full single-device cell on the intended Ubuntu RTX 3050 host at exact merged source `2bc4d9a` | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-10 | **Review by:** Before beginning Phase 7, changing the frozen cell, or changing the capture contract

## Result

The separate Phase 6 Full replacement cohort completed and established a
stable method. All five predeclared confirmatory slots started in their frozen
order, completed exactly 128 non-skipped optimizer steps, ended with native
outcome `passed`, retained `protocol-valid` evidence, and used no replacement
slots. The conditioning artifact and all five measured artifacts passed seal
verification, off-experiment-host copy verification, and fresh retrieval
verification.

The common stability and integrity contract passed:

- externally observed training durations were 55,570,364,091;
  54,761,500,666; 54,603,235,896; 55,147,615,041; and 54,884,619,502
  nanoseconds in slot order;
- duration MAD/median was 0.0047917894, below the 0.10 maximum;
- duration maximum/minimum was 1.0177119209, below the 1.20 maximum;
- runtime peak CUDA allocation was 1,758,568,960 bytes in every slot, so the
  observed range was zero bytes against a 175,856,896-byte maximum;
- telemetry coverage was 1.0 in every slot and the largest observed gap was
  1.091772559 seconds, satisfying the 0.99 and 2.5-second thresholds; and
- no slot recorded a safety warning, stop, capture reason, integrity
  discrepancy, or non-`NONE` terminal reason code.

Full is therefore stable for this exact frozen Phase 6 scope, and the
predeclared campaign gate authorizes Phase 7. This authorization permits the
reviewed Phase 7 procedure; it is not a product, model-quality, safety, or
release-readiness claim.

## Defect and cohort separation

The immutable [Phase 6 remediation matrix](../2026-08-10-cuda-phase6-remediation-matrix/README.md)
remains the historical 32-slot result at source `af91225`: Full promoted, but
one confirmatory slot passed and four were cancelled for observed unrelated GPU
activity. Investigation found that Aptus-owned CUDA processes could become
visible before the campaign registered them as managed.

Two source corrections closed that race. Merge `1014ff6` registered the action
child before its launch permit was released. A separate diagnostic r4 cohort at
that source then exposed the same ordering defect in the pre-submission CUDA
capacity probe. It completed a passing conditioner and one cancelled C1, was
preserved as nonqualifying evidence, and was not completed or included in this
aggregate. Merge `2bc4d9a` registered that probe before its launch permit was
released as well.

The successful r5 cohort was frozen only after the second fix. It has a new
comparison cohort, comparison cell, five new attempt identities, and a new
source freeze. Its five slots are the complete denominator; none was removed,
replaced, or imputed.

## Frozen scope

The cohort binds source commit
`2bc4d9a38f88cb0be1087b6e35a329587d1942bf`, source tree
`bf8a3c0f86e59061a17aa1d4077a157835ef4f53`, protocol digest
`5da86458bc665410a0dfb95d867dfbab99b8eeff7a24274818ea360fc106b8e0`,
and immutable SmolLM2 135M Instruct revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The held configuration was Full,
BF16, single placement, world size one, sequence length 256, micro-batch four,
gradient accumulation two, effective batch eight, and 128 optimizer steps.
Only the frozen training and data-order seeds varied.

## Evidence and custody

[`phase6-outcome.json`](phase6-outcome.json) publishes every measured slot,
individual stability values, aggregate calculations, exact source/model/data
bindings, and public custody receipt identifiers. The raw-to-public derivation
and redaction boundary are recorded in
[`sanitization-map.json`](sanitization-map.json). The procedurally separate
review pass is recorded in
[`independent-review.json`](independent-review.json). [`SHA256SUMS`](SHA256SUMS)
binds all four public packet files.

Protected paths, usernames, hostnames, network addresses, raw GPU identifiers,
raw machine identifiers, job identifiers, raw logs, and model bytes remain
outside Git. The published identifiers and digests are bounded evidence
projections, not disclosure of the protected vault.

## Claim boundary

This packet establishes exact-host Full confirmatory stability for one frozen
five-slot SmolLM2 135M single-device cohort and authorizes the bounded Phase 7
campaign procedure from that scope. It does not establish model quality,
method superiority, population calibration, statistical significance,
production throughput, another model or method, another placement or artifact,
another host or environment, production safety, or release readiness.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [Machine-readable protocol](../../../reference/cuda-campaign-protocol.v1.json)
- [Phase 5 repeatability anchor](../2026-08-10-cuda-phase5-repeatability-anchor/README.md)
- [Immutable Phase 6 remediation matrix](../2026-08-10-cuda-phase6-remediation-matrix/README.md)
