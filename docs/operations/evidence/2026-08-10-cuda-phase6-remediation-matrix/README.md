# CUDA Phase 6 Remediation Method Matrix

> **Status:** Complete; Full promoted from exploratory testing but did not establish confirmatory stability | **Authority:** Sanitized exact-host Phase 6 remediation execution and stopping-rule evidence; not model-quality, broad-performance, production-safety, or release-readiness evidence | **Applies to:** The corrected frozen SmolLM2 135M four-method matrix on the intended Ubuntu RTX 3050 host | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-10 | **Review by:** Before changing the capture contract, beginning a replacement Phase 6 cohort, or beginning Phase 7

## Result

Phase 6 remediation is complete with no stable method and no Phase 7
authorization. All 32 predeclared slots retained their frozen positions and no
slot was replaced.

- `full` was admitted, passed its conditioning run with protocol-valid and
  sealed evidence, and passed all three exploratory slots. It therefore met the
  frozen three-of-three promotion rule. Exactly five Full confirmatory slots
  then started: one passed and four were safely cancelled when the campaign
  observed unrelated GPU activity. Full did not establish confirmatory
  stability.
- `lora` was admitted. Two of its three exploratory slots passed; the other was
  safely cancelled for unrelated GPU activity. It was not promoted, so its five
  confirmatory positions remained `planned-not-started`.
- `int8-lora` was not admitted because explicit eight-bit support was not
  available on the exact host.
- `qlora` was not admitted because runtime-native four-bit support was not
  available on the exact host.

The six admitted exploratory slots and five promoted Full confirmatory slots
were all attempted. The remaining 21 frozen positions were retained as
`planned-not-started`. Every one of the 11 started matrix slots has
protocol-valid evidence, a verified seal, healthy telemetry, a verified
off-host copy, and a verified fresh restoration. Together with the passing
conditioning run, the private custody chain contains 12 sealed artifacts, 12
verified copies, 12 verified restorations, and 24 custody receipts.

## Frozen scope and comparison limits

The remediation matrix binds source commit
`af91225e1a2ba601a0b6dacd2366619e550babda`, source tree
`3a253447dff884da94cc347be299e988c7d5f87f`, protocol digest
`5da86458bc665410a0dfb95d867dfbab99b8eeff7a24274818ea360fc106b8e0`,
and immutable SmolLM2 135M Instruct revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. Full and LoRA used separately
frozen exact-source inputs. The common synthetic dataset was bound by SHA-256
`89f1d122deb7301db7f9665d146b37e44eb5dae24d9cf617bf06e925767e15eb`.

Sequence length, effective batch, optimizer-step target, seed policy,
checkpoint rule, capture protocol, host, and idle/cooldown protocol remained
frozen. Method-specific parameter scope, learning rate, adapter settings, and
quantization requirements remain comparison limitations. The matrix is a
resource-path and operational-stability characterization, not an isolated
causal comparison or model-quality ranking.

One Full E3 pre-action admission check refused activation for sustained thermal
warning. It produced no run or protected artifact and did not consume the
slot. After passive cooldown, the same unconsumed slot started and passed. This
was not a replacement. No external GPU query was issued during active slots.

## Evidence and custody

[`phase6-outcome.json`](phase6-outcome.json) publishes the aggregate result,
method decisions, 11 started-slot dispositions, 21 planned-not-started
dispositions, custody totals, and claim boundary without exposing protected
identifiers. [`sanitization-map.json`](sanitization-map.json) records the
raw-to-public boundary. [`independent-review.json`](independent-review.json)
records a procedurally separate deterministic review pass. [`SHA256SUMS`](SHA256SUMS)
binds every public packet file.

The four unrelated-GPU-activity cancellations are reported as observed safety
outcomes. This packet does not infer their external cause. Protected paths,
account and host names, network addresses, raw GPU and machine identifiers,
process, run, job, slot, artifact, and receipt identifiers, raw logs, archives,
checkpoints, model bytes, and adapter bytes remain outside Git.

## Relationship to the earlier packet

The [earlier Phase 6 packet](../2026-08-10-cuda-phase6-method-matrix/README.md)
remains an immutable historical record of its earlier cohort. This remediation
packet supersedes only the current operational Phase 6 status. It does not
alter or invalidate the bytes, observations, or claim boundary of the earlier
packet.

## Claim boundary

This packet establishes the corrected cohort's frozen slot dispositions, Full
exploratory promotion, the five-attempt Full confirmatory result, the absence
of a stable Phase 6 method, and the resulting lack of Phase 7 authorization.
It does not establish model quality, method superiority, broad performance,
population calibration, statistical significance, production throughput,
another artifact, host, or environment, production safety, or release
readiness.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [Machine-readable protocol](../../../reference/cuda-campaign-protocol.v1.json)
- [Phase 5 repeatability anchor](../2026-08-10-cuda-phase5-repeatability-anchor/README.md)
