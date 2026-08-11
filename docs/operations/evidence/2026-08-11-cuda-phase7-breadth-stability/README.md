# CUDA Phase 7 Architecture-Breadth Stability

> **Status:** Complete and independently reviewed; the corrected Qwen3-0.6B LoRA breadth cell passed all three frozen exploratory slots and the common stability contract; Phase 7 is complete and Phase 8 is not authorized | **Authority:** Sanitized exact-host Phase 7 planning, runtime, stability, custody, and review evidence; not model-quality, broad-performance, production-safety, Phase 8, or release-readiness evidence | **Applies to:** The fresh Phase 7 breadth cohort on the intended Ubuntu RTX 3050 host at exact merged source `a41ae49` | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-11 | **Review by:** Before Phase 8 activation, protocol amendment, or host/cooling change

## Result

The fresh Qwen3-0.6B LoRA breadth cohort completed its one conditioning slot
and all three predeclared exploratory slots. Seeds `6101`, `6203`, and `6301`
each produced a sealed, protocol-valid native pass with healthy telemetry and
exactly 128 non-skipped optimizer steps. No slot was retried, replaced,
excluded, or imputed.

The three-run stability contract passed every frozen check:

| Measurement | Observed | Limit | Result |
| --- | ---: | ---: | --- |
| Duration MAD / median | 0.003030675 | 0.10 | Pass |
| Duration maximum / minimum | 1.010673734 | 1.20 | Pass |
| Peak-device-memory range | 0 bytes | 400,346,316.8 bytes | Pass |
| Telemetry coverage | 1.0 for every run | at least 0.99 | Pass |
| Maximum telemetry gap | 1.081517922 seconds | at most 2.5 seconds | Pass |

Median externally observed training duration was 186,851,072,999 ns. Median
peak device memory was 4,003,463,168 bytes, and runtime peak CUDA allocation
was 3,302,564,352 bytes in every run.

## Corrected execution boundary

The first breadth cohort stopped at model-data validation because serialized
state-dictionary tensor elements had been declared as unique loaded model
parameters. The reviewed parameter correction fixed the declaration to
596,049,920 unique loaded parameters; that cohort was never resumed.

A second fresh cohort passed conditioning, then its first exploratory slot
stopped before optimizer work because the generated Linux authorization probe
used only completely unused pages and excluded reclaimable page cache. The
protected diagnosis classified the cause as
`LINUX_RECLAIMABLE_CACHE_EXCLUDED`. That failed slot was not retried, its
remaining slots never started, and no replacement was created.

The Linux profiler, parent runtime probe, and generated CUDA training program
were corrected to use kernel `MemAvailable` with the prior raw-free probe only
as fallback. The correction merged at
`a41ae4941661867789034eaa63bb968f2e137aba`, tree
`431c2be0c41e7b0274d68da408538f89f2ac79bd`. Only then was the new cohort
`cohort_da14e0463f7568d511b8` prepared and independently reviewed. Its fresh
current-boot baseline contained 600 samples with zero misses.

## Model, method, and ledger

The reviewed cell binds `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0 licensing,
596,049,920 unique loaded parameters, BF16 single-device LoRA, all seven
attention and MLP projection targets, sequence length 256, effective batch 8,
and exactly 128 optimizer steps. Gemma remains license-excluded and Mistral
remains planner-ineligible under the earlier amendment; neither acquired a
runtime ledger or result.

The three exploratory durations were 188,273,142,250 ns, 186,851,072,999 ns,
and 186,284,788,036 ns in frozen seed order. Every artifact passed deep seal
verification, verified off-host copy, and verified fresh retrieval. Raw logs,
telemetry, machine and network identifiers, working paths, model bytes,
archives, and unredacted receipts remain outside Git.

## Evidence and custody

[`phase7-outcome.json`](phase7-outcome.json) publishes the sanitized bindings,
measurements, admission disposition, and custody counts.
[`sanitization-map.json`](sanitization-map.json) records the protected-to-public
projection. [`independent-review.json`](independent-review.json) binds the
procedurally separate protected final review and recomputed public packet.
[`SHA256SUMS`](SHA256SUMS) binds every packet file.

The protected Ubuntu record retains the cohort inputs, pre-execution review,
baseline, plans, contexts, freezes, action records, telemetry, sealed artifacts,
stability result, and final review. A separate Mac failure domain retains all
four verified sealed copies, fresh restorations, receipts, custody records, and
the final cohort records.

## Phase boundary

The reviewed same-family staircase and reviewed architecture-breadth cell are
both complete, so Phase 7 is complete. Phase 8 is not started or authorized by
this packet. Its next permissible action is a separate activation and headroom-
selection review that freezes the eligible Phase 7 base cell before any
frontier cohort or result exists.

## Claim boundary

This packet establishes one stable exact-host Qwen3-0.6B LoRA Phase 7 breadth
cell and the complete no-replacement disposition of its ledger. It does not
compare model quality, establish LoRA superiority, qualify Gemma or Mistral,
authorize Phase 8, generalize to another host or configuration, or support a
production-safety or release-readiness claim.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [Phase 7 same-family stability](../2026-08-11-cuda-phase7-same-family-stability/README.md)
- [Architecture-breadth amendment](../2026-08-11-cuda-phase7-breadth-amendment/README.md)
- [Breadth parameter-semantics correction](../2026-08-11-cuda-phase7-breadth-parameter-correction/README.md)
