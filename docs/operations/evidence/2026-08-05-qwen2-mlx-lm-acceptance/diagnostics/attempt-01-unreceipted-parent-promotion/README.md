# Attempt 0 — pre-fix defect reproduction

> **Status:** Archived negative evidence; managed workflow failed and is excluded from acceptance
>
> **Last reviewed:** 2026-08-05
>
> **Review by:** On any MLX managed-completion or recovery-contract change

This nonqualifying attempt ran at Phase 6 static commit
`81bb1a286a45a5d5b424288699f8acdd8c051ecf`. Dependency, model/data,
measured preflight, pilot, full training, immutable adapter export, and fresh
reload all completed, but the managed job correctly failed its terminal
contract: the generated child had written `measured-run-pass` without the
required `aptus.parent-promotion.v1` receipt.

The host also contained a latent recovery predicate for a report
`schema_version` that real validation reports do not carry. Commit
`14ed44b52a76bb84d8d9db4f2303951aa641339b` moved current generated runners to
an explicit `execution-approved` handoff, kept final promotion parent-owned,
and corrected the recovery predicate. Focused regressions cover both the new
handoff and legacy frozen bundles.

This diagnostic is not one of the two qualifying repetitions. Raw state,
process identifiers, machine paths, model artifacts, adapter binaries, logs,
and generated text remain outside Git. `raw-artifact-digests.json` retains only
the content digests needed to identify the preserved temporary records.
