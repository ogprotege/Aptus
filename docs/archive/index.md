# Historical documentation index

> **Documentation status:** Active navigation for deprecated and archived material
>
> **Applies to:** Preserved Aptus and pre-Aptus documentation
>
> **Last reviewed:** 2026-07-22
>
> **Next scheduled review:** 2027-07-22, or when a listed path or successor changes

This index separates historical evidence from current product guidance. A file
listed here can explain how Aptus developed, but it does not establish current
behavior, compatibility, release readiness, or support.

## Current guidance

Start with:

- [Documentation index](../index.md)
- [Current capabilities](../product/current-capabilities.md)
- [Methodology overview](../methodology/overview.md)
- [System architecture](../architecture/system.md)
- [Release gates](../operations/release-gates.md)

## Deprecated v0.1 signposts

These files remain at their original paths so old links resolve and readers
reach the current successor.

| Historical path | Why deprecated | Successor |
|---|---|---|
| [Aptus core vertical slice](../design/aptus-core-vertical-slice.md) | The v0.1 design predates the v0.2 product and methodology contracts | [Current capabilities](../product/current-capabilities.md), [system architecture](../architecture/system.md) |
| [Aptus core smoke evidence](../validation/aptus-core-smoke.md) | A tiny generic LoRA step does not validate a selected v0.2 candidate | [Preflight and calibration](../methodology/preflight-calibration.md), [release gates](../operations/release-gates.md) |

## Archived research intake

The retained [Reference packet](../../Reference/README.md) contains original
research and product-concept inputs. Three files are archived because they are
not safe current authorities:

- [FineTuneX product history](../../Reference/FineTuneX.README.md)
- [Unverified fine-tuning method names](../../Reference/Fine-Tuning_Methods.md)
- [Unverified hyperparameter notes](../../Reference/hparam_methods_reference.md)

The [Reference and TO-REVIEW reconciliation](../research/reference-and-to-review-reconciliation.md)
records what was integrated, routed to the roadmap, archived, or rejected.

## Archived legacy recovery audit

The [Aptus legacy recovery audit](../audits/aptus-legacy/README.md) preserves the
forensic review of the removed `HyperTune/` source tree. Its Markdown, JSON, and
JSONL files form one evidence bundle.

Do not rewrite those reports to match current Aptus. Add new findings as new
dated records. The audit generator and reproduction instructions use the current
`docs/audits/aptus-legacy/` path, so a move requires an explicit migration plan.

## Preservation rules

- Historical content remains unchanged except for status, provenance, legal, or
  security corrections.
- A warning and current successor must appear before misleading historical
  claims.
- Historical files do not enter the current support matrix.
- Machine-readable snapshots stay with their matching reports.
- Ignored local intake and disposable audit sandboxes are not archives.

## Related documentation

- [Documentation maintenance policy](../maintenance/documentation-policy.md)
- [Documentation inventory](../maintenance/documentation-inventory.md)
- [Research index](../research/index.md)
