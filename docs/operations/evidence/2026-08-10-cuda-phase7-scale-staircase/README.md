# CUDA Phase 7 Scale Staircase

> **Status:** Complete; stopped fail-closed at the second 135M LoRA slot’s live admission gate; no Phase 7 cell established stability and Phase 8 is not authorized | **Authority:** Sanitized exact-host Phase 7 planning, runtime, custody, and stop evidence; not model-quality, broad-performance, production-safety, or release-readiness evidence | **Applies to:** The frozen same-family Phase 7 staircase on the intended Ubuntu RTX 3050 host at exact merged source `2bc4d9a` | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-10 | **Review by:** Before any Phase 7 retry, Phase 8 activation, protocol amendment, or host/cooling change

## Result

Phase 7 reached its predeclared fail-closed stopping condition. The 135M LoRA
conditioner passed, and exploratory seed 6101 then passed all five managed
actions, completed exactly 128 non-skipped optimizer steps, retained
`protocol-valid` evidence, and passed seal, off-host copy, and fresh retrieval
verification. Its externally observed training segment was 75,580,574,815 ns,
runtime peak CUDA allocation was 1,012,389,888 bytes, telemetry coverage was
1.0, and the maximum telemetry gap was 1.0775679 seconds.

The next predeclared slot, seed 6203, did not activate. Its production live
admission gate returned `THERMAL_WARNING_SUSTAINED`, so its status remains
`planned-not-started`, native outcome remains unset, and evidence status remains
`not-started`. The protocol requires the campaign to stop for diagnosis when
the next slot cannot obtain the complete qualifying admission window. No retry
or replacement was created.

Consequently, seed 6301 and all later method/model slots remain
`planned-not-started`. No Phase 7 cell achieved the required 3-of-3 passing and
stable batch. Phase 8 is not authorized.

## Frozen model and planner ledger

Before any training outcome was observed, all three immutable core revisions
were provider-inspected, license-reviewed, downloaded, hashed, and checked
against the frozen 512-row tokenizer manifest. The reproduced canonical token
manifest matched the Phase 1 preview exactly at 63,234 bytes and SHA-256
`fa8a4c9223e47fa95cb163db871c35159978b92c4ea559b95e8719697c7be9f6`
for every model.

The complete provider repository inventories contained alternate ONNX and
training-history assets. The execution-inventory addendum therefore bound the
exact Transformers safetensors/config/tokenizer file set used by Aptus, without
silently treating unrelated provider formats as training artifact bytes. Exact
execution artifacts were 272,437,573 bytes for 135M, 727,051,918 bytes for
360M, and 3,426,155,020 bytes for 1.7B.

Exact planning admitted 135M LoRA, 135M Full, and 360M LoRA. It rejected 360M
Full because even its point estimate exceeded usable per-device memory, and it
could produce no feasible or conditional exact 1.7B LoRA or Full plan on this
8 GiB host. Those planning decisions were frozen before the first qualifying
outcome.

## Slot denominator and stopping decision

The immutable Phase 7 ledger contains 18 exploratory slots: three seeds for
LoRA and Full at each of 135M, 360M, and 1.7B. One slot started and passed; 17
remain planned-not-started. The second 135M LoRA slot is specifically bound to
the live admission refusal. The third LoRA slot, 135M Full slots, all larger
model slots, and architecture breadth were not activated after the stop.

This packet does not relabel the admission refusal as a runtime failure: the
harness never activated that slot. It also does not infer a three-run stability
statistic from one passing observation. Every slot remains in the denominator,
and no value is excluded, replaced, or imputed.

## Preparation history

Preparation initially encountered two setup-only typed-object mismatches before
any plan bundle or attempt activated. The sealed resume record documents the
boundary and reuse of unchanged frozen inputs; no attempt slot was consumed and
no replacement slot was created. Planner evaluation then completed, all admitted
bundles were materialized, and the production source freezes and contexts were
verified before conditioning began.

## Evidence and custody

[`phase7-outcome.json`](phase7-outcome.json) publishes the exact model freezes,
planner admissions, slot dispositions, runtime metrics, custody receipts, and
stop decision. [`sanitization-map.json`](sanitization-map.json) records how raw
protected records project to this public packet. The procedurally separate
deterministic review is in [`independent-review.json`](independent-review.json).
[`SHA256SUMS`](SHA256SUMS) binds the packet.

The conditioner and passing exploratory artifact each passed deep seal
verification, off-experiment-host copy verification, and fresh retrieval
verification. Raw logs, machine identifiers, network details, model bytes,
working paths, and unstarted-slot host state remain outside Git.

## Claim boundary

This packet establishes the exact planner ledger, one valid 135M LoRA result,
and the protocol-required thermal admission stop on this frozen host and source.
It does not establish a stable Phase 7 cell, authorize Phase 8, demonstrate
360M or 1.7B runtime behavior, establish architecture breadth, rank model
quality or methods, or support a production-safety or release-readiness claim.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [Machine-readable protocol](../../../reference/cuda-campaign-protocol.v1.json)
- [Phase 6 confirmatory stability](../2026-08-10-cuda-phase6-confirmatory-stability/README.md)
