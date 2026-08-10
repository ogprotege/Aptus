# CUDA Phase 5 Repeatability Anchor

> **Status:** Complete; repeatability anchor established | **Authority:** Sanitized Phase 5 exact-host repeatability evidence; not model-quality, broad-performance, production-safety, or release-readiness evidence | **Applies to:** The frozen SmolLM2 135M LoRA single-device cohort on the intended Ubuntu RTX 3050 host | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-10 | **Review by:** Before changing the frozen anchor scope or beginning Phase 6

## Result

Phase 5 completed and established the repeatability anchor. All five
predeclared measured slots started in their frozen order, completed exactly 128
non-skipped optimizer steps, ended with native outcome `passed`, retained
`protocol-valid` evidence, and used no replacement slots. The five protected
artifacts passed seal verification, off-experiment-host copy verification, and
fresh retrieval verification.

The common stability and integrity contract passed:

- externally observed training durations were 75,931,015,115;
  77,001,524,308; 76,185,600,925; 76,371,491,533; and 75,799,436,644
  nanoseconds in slot order;
- duration MAD/median was 0.0033416526, below the 0.10 maximum;
- duration maximum/minimum was 1.0158587942, below the 1.20 maximum;
- runtime peak CUDA allocation was 1,012,389,888 bytes in every slot, so the
  observed range was zero bytes against a 134,217,728-byte maximum;
- telemetry coverage was 1.0 in every slot and the largest observed gap was
  1.086778720 seconds, satisfying the 0.99 and 2.5-second thresholds; and
- no slot recorded a safety warning, stop, capture reason, integrity
  discrepancy, or non-`NONE` terminal reason code.

The predeclared decision therefore authorizes Phase 6 for this exact frozen
anchor scope. It does not authorize an unreviewed configuration change.

## Frozen scope

The cohort binds source commit
`3bfec547d4cffedbaf049426d9713f1ccc25b5a2`, source tree
`6acaa096ad50b0e814e84e706d3dd12a3cc8cc33`, protocol digest
`5da86458bc665410a0dfb95d867dfbab99b8eeff7a24274818ea360fc106b8e0`,
and the immutable SmolLM2 135M Instruct revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The held configuration was
LoRA, BF16, single placement, world size one, sequence length 256, micro-batch
four, gradient accumulation two, effective batch eight, and 128 optimizer
steps. Only the frozen training and data-order seeds varied.

The earlier [August 9 stopped cohort](../2026-08-09-cuda-phase5-repeatability-anchor/README.md)
remains immutable failure history and is not included in this aggregate. The
successful cohort has a new comparison-cell ID, comparison-cohort ID, and five
new attempt identities. The nonqualifying conditioning work is also excluded
from the five-slot denominator.

## Evidence and custody

[`phase5-outcome.json`](phase5-outcome.json) publishes every measured slot,
individual stability values, aggregate calculations, exact source/model/data
bindings, and public custody receipt identifiers. The raw-to-public derivation
and redaction boundary are recorded in
[`sanitization-map.json`](sanitization-map.json). The procedurally separate
review pass is recorded in
[`independent-review.json`](independent-review.json). [`SHA256SUMS`](SHA256SUMS)
binds all four public packet files.

Protected paths, usernames, hostnames, network addresses, raw GPU identifiers,
raw machine identifiers, job identifiers, raw logs, and model or adapter bytes
remain outside Git. The published host facts and digests are bounded
projections, not disclosure of the protected evidence vault.

## Claim boundary

This packet establishes exact-host repeatability plus duration and
peak-device-memory stability for this one frozen five-slot SmolLM2 LoRA cohort.
It establishes Phase 6 eligibility only from that anchor. It does not establish
model quality, population calibration, statistical significance, production
throughput, another model or method, another placement or artifact, another
host or environment, production safety, or release readiness.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [Machine-readable protocol](../../../reference/cuda-campaign-protocol.v1.json)
- [August 6 historical acceptance](../2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
