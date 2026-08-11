# CUDA Phase 8 Guarded Configuration Frontier

> **Status:** Complete and independently reviewed; all three frozen one-axis frontier cohorts are closed, a Phase 9 candidate is selected but not authorized, and no full training was executed | **Authority:** Sanitized exact-host Phase 8 planning, bounded-pilot, telemetry, custody, headroom-selection, and review evidence; not model-quality, production-safety, Phase 9, cloud, multi-GPU, or release-readiness evidence | **Applies to:** The guarded Qwen3-0.6B LoRA frontier campaign on the intended Ubuntu RTX 3050 host at exact merged source `59993d7` | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-11 | **Review by:** Before Phase 9 authorization, protocol amendment, or host/cooling change

## Result

Phase 8 completed the frozen sequence-length, effective-batch, and
micro-batch/accumulation ladders with one bounded pilot per activated point,
no retry or replacement, and no confirmed-training action. Fourteen of sixteen
started slots were protocol-valid native passes. Two bounded pilots stopped
with `CUDA_OOM`; both retained healthy telemetry and sealed protocol-valid
evidence. The seventeenth predeclared slot, sequence length 2,048, remained
planned-not-started after the sequence-axis stopping rule fired.

| Axis | Observed endpoint | Highest passing point below endpoint | Disposition |
| --- | --- | --- | --- |
| Sequence length | 1,024: bounded-pilot `CUDA_OOM` | 512 | First bounded-pilot nonpass; 2,048 not activated |
| Effective batch | 64: pass | 32 | Right-censored at the frozen top rung; no unobserved failure claimed |
| Micro/accumulation at effective batch 16 | `(16,1)`: bounded-pilot `CUDA_OOM` | `(8,2)` | First bounded-pilot nonpass |

These OOM records are bounded-pilot capacity observations, not results from
running full training until failure. They do not prove that every intermediate,
larger, or differently configured point fails.

## Headroom selection

The frozen Phase 9 selection rule starts strictly below each endpoint and walks
down until every qualifying input run meets the numeric margins. Sequence 512
and micro `(8,2)` each fell below the 3 GiB minimum-free-VRAM margin, so their
axes walked down one rung. The resulting eligible candidates ranked as follows:

| Rank | Axis candidate | Sequence × effective batch | Qualifying inputs | Result |
| ---: | --- | ---: | ---: | --- |
| 1 | sequence 256, effective batch 32, micro 4, accumulation 8 | 8,192 | 1 | Selected, pending separate Phase 9 authorization |
| 2 | sequence 256, effective batch 16, micro 4, accumulation 4 | 4,096 | 2 | Eligible |
| 3 | sequence 256, effective batch 8, micro 4, accumulation 2 | 2,048 | 2 | Eligible |

For the selected batch-32 input, maximum GPU temperature was 55°C, minimum
free VRAM was 4,222,615,552 bytes, minimum available RAM was 62,125,764,608
bytes, and minimum free disk was 206,934,855,680 bytes. Those observations leave
23°C, 1,001,390,080 bytes, 49,240,862,720 bytes, and 155,395,248,128 bytes of
margin respectively. Telemetry coverage was 1.0, maximum gap was 1.059933312
seconds, observed rolling swap rate was zero, and no warning, thermal throttle,
Xid, hardware fault, ownership fault, or foreign compute process was present.

This deterministic selection does not activate Phase 9. It records the only
candidate that a later, separately authorized Phase 9 cohort may freeze.

## Source prerequisite and execution boundary

The first Phase 8 preparation exposed a source-contract gap before any managed
action ran: the qualifying harness did not yet accept the pilot-only
`phase8-frontier` role. That immutable root was not reused. The exact role and
four-action evidence profile were implemented and reviewed in
[PR #79](https://github.com/ogprotege/Aptus/pull/79), then merged before a fresh
base selection, cohort preparation, review, source freeze, and current-boot
baseline were created at commit
`59993d7d42d478dabdaf5f3b12f27bf0bd79ff11`, tree
`cf75905ed7082fdf26af259c163f433c2e53dc69`.

Every started slot retained role `phase8-frontier` and exactly four managed
actions: dependency, model-data, measured preflight, and bounded pilot. No
confirmed-training or export action was required, synthesized, or executed.

## Evidence and custody

[`phase8-outcome.json`](phase8-outcome.json) publishes the sanitized endpoints,
headroom observations, deterministic selection, and custody counts.
[`sanitization-map.json`](sanitization-map.json) records the protected-to-public
projection. [`independent-review.json`](independent-review.json) binds the
procedurally separate protected final review and recomputed public packet.
[`SHA256SUMS`](SHA256SUMS) binds every packet file.

All sixteen started artifacts passed deep seal verification, verified off-host
copy, and verified fresh retrieval. The protected evidence retains 604
manifested files and 42,133,048 bytes per custody layer. Raw logs, unredacted
telemetry, machine and network identifiers, paths, process metadata, model and
checkpoint bytes, archives, manifests, receipts, journals, and restored trees
remain outside Git.

## Phase boundary

Phase 8 is complete and independently reviewed. Phase 9 remains inactive and
unauthorized. Its next permissible action is a separately reviewed activation
that freezes the selected batch-32 execution configuration and its three
no-replacement endurance slots. Cloud resource acquisition, multi-GPU work,
and Phase 10 are outside this packet.

## Claim boundary

This packet establishes the three guarded frontiers and deterministic Phase 9
candidate selection only for the exact source, host, Qwen3-0.6B revision, LoRA
method, synthetic fixture, and frozen protocol. It does not establish model
quality, an absolute hardware limit, safety or performance on another host,
production readiness, release readiness, Phase 9 success, or authority to use
external GPU resources.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [Phase 7 architecture-breadth stability](../2026-08-11-cuda-phase7-breadth-stability/README.md)
