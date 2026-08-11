# CUDA Phase 9 Endurance and Job Control

> **Status:** Complete and independently reviewed; all three frozen 300-update endurance slots and every bounded job-control exercise passed, while Phase 10 remains unauthorized | **Authority:** Sanitized exact-host Phase 9 activation, full-training, telemetry, method-integrity, custody, job-control, aggregate, and independent-review evidence; not model-quality, semantic adapter-reload, production-safety, cloud, multi-GPU, Phase 10, or release-readiness evidence | **Applies to:** The frozen Qwen3-0.6B LoRA configuration on the intended Ubuntu RTX 3050 host at exact runtime source `59993d7` | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-11 | **Review by:** Before Phase 10 authorization, a protocol amendment, or host/cooling change

## Result

Phase 9 passed. Three predeclared, no-replacement slots independently ran the
same frozen configuration to exactly 300 completed non-skipped optimizer steps.
Every slot passed natively with protocol-valid evidence, healthy telemetry, a
valid LoRA parameter census, a valid cooldown, a deep-verifying seal, a verified
off-host Mac copy, and a verified fresh full retrieval.

| Slot | Training seed | Data-order seed | Steps | Counter-window duration | Aggregate step rate | Maximum GPU temperature | Minimum free VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9,101 | 1,009,101 | 300 | 1,628.894 s | 0.184174/s | 62°C | 4,218,421,248 B |
| 2 | 9,203 | 1,009,203 | 300 | 1,623.474 s | 0.184789/s | 63°C | 4,218,421,248 B |
| 3 | 9,301 | 1,009,301 | 300 | 1,616.950 s | 0.185534/s | 62°C | 4,218,421,248 B |

Across the three counter windows, the exact totals are 900 optimizer steps,
7,200 micro-iterations, 28,800 examples, 7,143,464 padded input elements,
5,747,174 non-padding tokens, and 2,471,612 supervised tokens over
4,869.318 seconds. The corresponding aggregate rates are 0.184831 optimizer
steps/s, 5.914586 examples/s, 1,467.035798 padded elements/s, 1,180.283123
non-padding tokens/s, and 507.588935 supervised tokens/s.

These are aggregate counter-window rates. This packet makes no cross-run
step-time drift claim and applies no Phase 5 or Phase 6 stability ratio.

## Frozen execution configuration

All slots used Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, LoRA, BF16, sequence length 256,
effective batch 32, micro-batch 4, accumulation 8, a 64-step checkpoint cadence,
the frozen dataset fixture digest
`6d90599e949bf2698b940e0c159e1fa24f3dc0c162005546bd270fc761aac7f2`,
and a 5,400-second emergency watchdog ceiling. The watchdog remained a safety
ceiling; the reviewed 300-update target controlled normal completion.

Each run's method-integrity census found 10,092,544 trainable LoRA parameters,
596,049,920 frozen parameters, zero unexpected trainable tensors, zero
incomplete adapter targets, and finite values. Runtime peak CUDA allocation was
3,302,711,808 bytes and reserved CUDA memory was 3,707,764,736 bytes in every
slot. Required telemetry coverage was 1.0, with maximum gaps between 1.2006 and
1.2179 seconds.

## Job-control exercises

Eight source-bound, non-training exercises separately passed:

- owned process-group cancellation and persisted cancellation milestones;
- same-user host-global lease exclusion across different state roots;
- restart recovery of an unattachable stale owner;
- pinned parent-verification event-sink boundaries;
- parent re-verification after sink lock;
- parent re-verification after sink fsync;
- crash-before-receipt preservation of a pending promotion; and
- rejection of an unreceipted terminal report during recovery.

These exercises used harmless bounded subprocesses after the endurance cohort
had sealed. They did not provoke an OOM or alter a training artifact.

## Evidence and custody

[`phase9-outcome.json`](phase9-outcome.json) publishes the sanitized slot,
counter, resource, job-control, and custody result.
[`sanitization-map.json`](sanitization-map.json) records the protected-to-public
projection. [`independent-review.json`](independent-review.json) binds the fresh
recomputation from the three restored sealed artifacts.
[`SHA256SUMS`](SHA256SUMS) binds every packet file.

All three artifacts passed deep seal verification, verified off-host copy, and
verified fresh retrieval. The protected evidence retains 129 manifested files
and 17,569,834 bytes per custody layer. Raw logs, unredacted telemetry, machine
and network identifiers, paths, process metadata, model and checkpoint bytes,
archives, manifests, receipts, journals, and restored trees remain outside Git.

## Phase boundary

Phase 9 is complete and independently reviewed. It establishes that the exact
frozen system completed its three-slot endurance and managed-job-control
contract on the intended host. Phase 10 has not been authorized or performed.
Cloud resource acquisition, external GPU orchestration, and multi-GPU work were
not used.

CUDA export verification in this campaign remains structural-file-tree only.
This packet does not claim a semantic fresh-process adapter reload on CUDA and
does not convert training loss into a model-quality conclusion.

## Claim boundary

This packet does not establish theology-domain quality, LoRA superiority,
production or release readiness, semantic CUDA adapter reload, a cross-run drift
conclusion, or behavior on another source, host, model revision, dataset, or
configuration. It does establish the reviewed bounded Phase 9 contract stated
above, including exact counters, aggregate rates, method integrity, telemetry,
custody, and managed-job controls.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [Phase 8 guarded configuration frontier](../2026-08-11-cuda-phase8-guarded-frontier/README.md)
