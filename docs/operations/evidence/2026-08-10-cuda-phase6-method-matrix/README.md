# CUDA Phase 6 Same-Model Method Matrix

> **Status:** Complete; no method promoted | **Authority:** Sanitized Phase 6 exact-host execution and stopping-rule evidence; not model-quality, broad-performance, production-safety, or release-readiness evidence | **Applies to:** The frozen SmolLM2 135M four-method matrix on the intended Ubuntu RTX 3050 host | **Audience:** Maintainers, reviewers, and release-evidence consumers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-10 | **Review by:** Before authorizing a replacement cohort, changing the capture contract, or beginning Phase 7

## Result

Phase 6 is complete with no promoted method and no confirmatory comparison.
All 32 attempt slots remained in the frozen ledger and no slot was replaced.

- `full` passed static and resource admission, but its single conditioning run
  was native `passed` with `capture-invalid` evidence. Its three exploratory
  slots therefore remained `planned-not-started`.
- `lora` was admitted. E1 activated but did not launch because its execution
  configuration did not match the initially shared Phase 4 source freeze. E2
  was safely cancelled after telemetry observed an unmanaged GPU process. E3
  completed 128 non-skipped optimizer steps with native `passed`,
  `protocol-valid` evidence, healthy telemetry, a verified seal, a verified
  off-host copy, and a verified fresh retrieval.
- `int8-lora` was not admitted because the host did not provide explicit
  eight-bit support on every participating GPU.
- `qlora` was not admitted because the host did not provide explicit
  runtime-native four-bit support on every participating device.

The frozen promotion rule requires exactly three of three qualifying
exploratory slots for a method. No method met it, so all 20 confirmatory slots
remained `planned-not-started`. Phase 7 is not authorized because its entry
condition requires at least one stable Phase 6 cell.

## Frozen scope and comparison limits

The matrix binds source commit
`3bfec547d4cffedbaf049426d9713f1ccc25b5a2`, source tree
`6acaa096ad50b0e814e84e706d3dd12a3cc8cc33`, protocol digest
`5da86458bc665410a0dfb95d867dfbab99b8eeff7a24274818ea360fc106b8e0`,
and immutable SmolLM2 135M Instruct revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. Every method held sequence
length 256, micro-batch four, gradient accumulation two, effective batch eight,
single placement, BF16 compute, and a 128-step target.

Method-specific differences were frozen before results. Full fine-tuning used
all trainable parameters, rank and alpha zero, and learning rate `0.00002`.
The three adapter methods used rank 16, alpha 32, learning rate `0.0002`, and
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`
as targets. Every method used `adamw_torch`, a linear scheduler, zero weight
decay, maximum gradient norm 1.0, and gradient checkpointing. The quantized
methods also required their method-specific base quantization. These differences
make the matrix a resource-path characterization, not an isolated causal
comparison or a model-quality ranking.

## Evidence and custody

[`phase6-outcome.json`](phase6-outcome.json) publishes the complete 32-slot
ledger, method admissions, conditioning disposition, E2 safety stop, E3
measurements, promotion decisions, and public custody receipt identifiers.
[`sanitization-map.json`](sanitization-map.json) records the raw-to-public
boundary. [`independent-review.json`](independent-review.json) records a
procedurally separate deterministic review pass. [`SHA256SUMS`](SHA256SUMS)
binds the public packet.

The conditioning artifact and both sealed LoRA artifacts passed seal
verification, off-experiment-host copy verification, and fresh retrieval
verification. E1 produced activation records but no launched or sealed
artifact; it is retained as a private diagnostic and is not represented as a
protected run artifact.

Protected paths, account and host names, network addresses, raw GPU and machine
identifiers, process and job identifiers, raw logs, archives, checkpoints,
model bytes, and adapter bytes remain outside Git.

## Claim boundary

This packet establishes only the frozen Phase 6 slot dispositions and one
qualifying LoRA exploratory run on the exact host and scope. It establishes
that no method met the frozen three-of-three promotion rule and that no
confirmatory or Phase 7 conclusion is authorized. It does not establish model
quality, method superiority, stable repeatability for a Phase 6 method,
population calibration, statistical significance, production throughput,
another artifact or environment, production safety, or release readiness.

## Related documentation

- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
- [Machine-readable protocol](../../../reference/cuda-campaign-protocol.v1.json)
- [Phase 5 repeatability anchor](../2026-08-10-cuda-phase5-repeatability-anchor/README.md)
