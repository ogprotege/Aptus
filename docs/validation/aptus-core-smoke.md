# Aptus Core Smoke Evidence

> **Documentation status:** Deprecated
>
> **Applies to:** Superseded Aptus v0.1 tiny-model smoke evidence
>
> **Last reviewed:** 2026-07-22
>
> **Next scheduled review:** 2027-07-22, or when this signpost or a successor moves
>
> This record does not authorize v0.2 execution. Use the
> [validation states](../reference/validation-states.md),
> [release gates](../operations/release-gates.md), and
> [historical index](../archive/index.md).

This v0.1 tiny-model smoke record is superseded. A generic tiny LoRA optimizer
step does not validate a selected v0.2 candidate.

Current validation labels a tiny synthetic check only as method and kernel
preflight evidence. A real-model pilot on the target hardware must pass before
`execution-approved`.

Release evidence belongs under a versioned evidence record and must bind the
plan ID, candidate ID, bundle fingerprint, model and tokenizer revision,
dataset digest, hardware profile, installed-environment binding, and job ID.

See [preflight and calibration](../methodology/preflight-calibration.md) and
[release gates](../operations/release-gates.md).
