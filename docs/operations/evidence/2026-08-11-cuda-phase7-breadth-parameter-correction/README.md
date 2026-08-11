# CUDA Phase 7 Breadth Parameter-Semantics Correction

> **Status:** Complete and independently reviewed; the first Qwen breadth cohort is stopped and not resumable; a new reviewed cohort may be created only after this correction merges | **Authority:** Sealed conditioning failure, exact offline runtime measurement, corrected planner evaluation, and independent review; not stability, model-quality, production-safety, or Phase 8 evidence | **Applies to:** The Qwen3-0.6B Phase 7 breadth cell on the intended Ubuntu RTX 3050 host | **Audience:** Maintainers, operators, and evidence reviewers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-11 | **Review by:** Before creating a corrected breadth ledger or activating Phase 8

## Outcome

The first reviewed breadth cohort stopped during its one conditioning slot.
Dependency validation passed, but model-data validation failed closed with
`PROCESS_EXIT_NONZERO` before measured preflight, pilot, or training. All three
exploratory slots remained planned-not-started. The stopped cohort is not
resumed and receives no replacement slot.

The failure was a parameter-semantics mismatch in the pre-execution evidence,
not a tokenizer, RoPE, CUDA-memory, or thermal failure. The prior amendment
counted 751,632,384 serialized state-dictionary tensor elements as model
parameters. The exact loaded model has 596,049,920 unique parameters because
its 155,582,464-element input embedding and output head share storage.

## Corrected contract

The plan-bound `parameters` value is corrected to 596,049,920 unique loaded
model parameters. The 751,632,384 value remains valid only as a serialized
state-dictionary tensor-element count; it must not be used as the runtime model
parameter declaration.

With the corrected value, exact planning still admits only single-device BF16
LoRA. Full fine-tuning remains infeasible, INT8-LoRA remains blocked without
explicit eight-bit support, and QLoRA remains blocked without runtime-native
four-bit support. The corrected seed-6101 LoRA candidate is
`cand_949a5ae4c30cd1c810f7` with an estimated upper device-memory envelope of
4,083,631,784 bytes.

## Evidence boundary

[`correction.json`](correction.json) records the sanitized stopped outcome,
exact counts, shared-storage observation, and corrected planner dispositions.
[`independent-review.json`](independent-review.json) confirms that the mismatch
is exactly explained, no exploratory slot started, and no retry or replacement
was created. [`sanitization-map.json`](sanitization-map.json) records the
protected-to-public projection, and [`SHA256SUMS`](SHA256SUMS) binds the public
packet.

The protected correction, review, and sealed failed conditioning artifact have
a verified off-host copy. Host paths, raw logs, telemetry, process identifiers,
and machine identifiers remain outside Git.

## Next boundary

After this correction merges, the next action is a fresh, independently
reviewed Phase 7 cohort using the corrected unique-parameter declaration. It
may contain the same one conditioning slot and three frozen exploratory seeds,
but it is a new cohort—not a resumed ledger, replacement slot, or informal
retry. Phase 8 remains unauthorized.

## Related documentation

- [Prior architecture-breadth amendment](../2026-08-11-cuda-phase7-breadth-amendment/README.md)
- [Reviewed same-family stability](../2026-08-11-cuda-phase7-same-family-stability/README.md)
- [Canonical CUDA empirical campaign](../../cuda-empirical-campaign.md)
- [Frozen CUDA campaign protocol](../../../reference/cuda-campaign-protocol.md)
