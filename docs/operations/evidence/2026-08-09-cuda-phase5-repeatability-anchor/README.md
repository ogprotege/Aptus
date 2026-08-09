# CUDA Phase 5 Repeatability Anchor Outcome

> **Status:** Complete; repeatability anchor not established | **Authority:** Sanitized Phase 5 execution and stopping-rule evidence; not performance, promotion, or release-readiness evidence | **Applies to:** The frozen SmolLM2 135M LoRA repeatability cohort on the intended Ubuntu RTX 3050 host | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-09 | **Review by:** Before authorizing a replacement cohort or changing the capture protocol

## Result

Phase 5 completed with the repeatability anchor **not established**. The one
predeclared, nonqualifying conditioning attempt passed its 120-sample admission
window and dependency action. During model/data validation, the one-hertz
telemetry collector missed a scheduled sample after host-probe latency exceeded
the sampling interval. The safety controller cancelled the managed action and
the capture closed as `capture-invalid` with stable reason codes
`TELEMETRY_COLLECTOR_FAILURE` and `STREAM_CAPTURE_FAILURE`.

The frozen protocol permits exactly one conditioning attempt for this cell. A
capture-invalid conditioning attempt blocks the cell pending diagnosis, and it
cannot be replaced. Consequently, measured preflight and pilot were not
started, all five measured Phase 5 slots remained `planned-not-started`, and no
training or final export occurred.

## Frozen decision

- Required measured slots: 5.
- Started measured slots: 0.
- Completed optimizer steps: 0.
- Aggregate stability calculations: not applicable.
- Promotion: denied.
- Phase 6 authorization: denied.
- Required next action: diagnose the collector latency and review a new
  protocol/cohort before any new conditioning or measured attempt.

The failed conditioning attempt is not silently discarded and is not counted
as a measured repetition. The historical August 6 acceptance remains separate
and is not substituted into this cohort.

## Evidence and custody

The protected admission artifact and sealed capture-failure receipt both passed
seal verification, off-host copy verification, and retrieval testing. The ten
available files from the incomplete run were copied off-host and independently
matched against every digest and byte size bound by the sealed failure receipt.
The raw exception text, host identifiers, protected paths, and byte-exact logs
remain only in the protected vault.

[`phase5-outcome.json`](phase5-outcome.json) records the frozen slot
dispositions, bounded failure summary, source and protocol bindings, and public
custody proofs. [`SHA256SUMS`](SHA256SUMS) binds this packet's public files.

## Claim boundary

This packet proves that the frozen stopping rule was applied and that the
resulting failure evidence was preserved. It does not establish repeatability,
performance, memory stability, model quality, Phase 6 eligibility, production
safety, or release readiness.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [August 6 historical acceptance](../2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
