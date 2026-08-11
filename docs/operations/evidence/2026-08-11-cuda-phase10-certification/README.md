# CUDA Phase 10 campaign certification

> **Status:** Complete and independently reviewed | **Evidence class:** Exact-host bounded campaign aggregation | **Source:** `3c56b37799be16d4698b6e75e76c690b4ba8e818` / tree `76d38631026fe55c0015a0608eae6d45500e4f8d` | **Protocol:** `005de18e60d8b707986dee6eef4b7199e7e1fecba385ab591ff708671b858335` | **Last reviewed:** 2026-08-11 | **Review by:** Before any claim broadening, new campaign, or release decision

Phase 10 closes the authorized RTX 3050 CUDA evidence campaign. It performed no
new training, retry, replacement run, cloud acquisition, or external-GPU work.
It aggregated the frozen Phase 5 through Phase 9 records, reverified their
public checksum packets and selected protected masters, recomputed the bounded
statistics, applied an independent review, and published only the sanitized
projection.

## Campaign disposition

The immutable measured, exploratory, frontier, and endurance cohorts contain
149 planned slots: 58 started and 91 remained planned-not-started. The started
set records 47 native passes, 7 cancellations, 3 bounded failures, and 1
conservatively normalized unknown outcome. Evidence status is protocol-valid
for 57 started slots and capture-invalid for 1. Exactly 47 slots are both a
native pass and protocol-valid. No replacement run was created.

Conditioning and setup actions remain separately audited and are not counted as
measured slots. The historical Phase 6 `activated-not-launched` record is
conservatively retained as started, native-outcome unknown, and capture-invalid;
it is neither hidden nor counted as a pass.

The reviewer independently recomputed all cohort equations and scalar
summaries, verified 13 prior checksum-bound public packets, and deeply verified
68 selected protected artifacts against their seals, manifests, file sizes,
digests, and no-extra-file rules. It also independently confirmed the Phase 8
probe-only frontier and the Phase 9 aggregate counters and rates.

## Evidence established

The certification reports bounded exact-host measurements for these stable
cells:

- SmolLM2-135M-Instruct LoRA: five-slot Phase 5 repeatability cell.
- SmolLM2-135M-Instruct Full: five-slot Phase 6 confirmatory cell.
- SmolLM2-135M-Instruct LoRA and Full, and SmolLM2-360M-Instruct LoRA: three-slot
  Phase 7 exploratory cells.
- Qwen3-0.6B LoRA: three-slot Phase 7 breadth cell and the separate three-slot,
  300-update Phase 9 endurance cell.

Phase 8 records 17 guarded frontier points: 16 bounded pilots started, 14
passed, 2 ended in telemetry-valid `CUDA_OOM`, and 1 remained
planned-not-started under the prior stop rule. Every started frontier point had
`full_training_executed=false`; full training was never used to provoke an OOM.
Phase 9 records 900 completed optimizer updates across three passing endurance
slots and eight passing bounded job-control exercises. Its throughput values
are aggregate rates only and do not support a cross-run drift claim.

## Boundaries that remain open

This packet does not establish Aptus 0.2 release readiness, task/model quality,
semantic CUDA adapter reload, production safety, DDP, LoRA FSDP, Developer ID
notarized distribution, or a confirmatory paired ranking of training methods.
Full and quantized FSDP remain unsupported; DDP and LoRA FSDP remain untested
and require a separately authorized multi-GPU campaign. All conclusions remain
bound to the exact host, source, model revisions, fixtures, and configurations
listed in the machine-readable certification.

Phase 10 completes the campaign certification. It does not authorize Phase 11
or broaden the campaign into remote-resource acquisition.

## Files

- [`phase10-certification.json`](phase10-certification.json) is the sanitized
  aggregate projection with cohort dispositions, stable-cell observations,
  Phase 8 and Phase 9 results, and release-gate mapping.
- [`independent-review.json`](independent-review.json) binds the aggregate input
  and records the separately recomputed review checks.
- [`phase10-decision.json`](phase10-decision.json) binds the certification and
  review and records the final `complete-reviewed` decision.
- [`sanitization-map.json`](sanitization-map.json) records included and excluded
  evidence classes.
- [`SHA256SUMS`](SHA256SUMS) binds the public packet files.

The protected aggregate, independent-review record, and verified artifact
inventory remain outside the repository under the private evidence retention
policy.
