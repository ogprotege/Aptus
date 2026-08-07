# Historical engineering-review index

> **Documentation status:** Active navigation for archived review evidence
>
> **Applies to:** Superseded Aptus implementation plans and point-in-time code reviews
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when a listed path or successor changes

This index keeps completed engineering reviews discoverable without presenting
them as active plans. The linked records preserve their original bodies. A
finding described there as current, open, blocked, or complete applies only to
the source snapshot named by that record.

For current behavior, start with the [documentation index](../../docs/index.md),
[current capabilities](../../docs/product/current-capabilities.md), and
[release gates](../../docs/operations/release-gates.md).

## Archived reviews and successors

| Historical review | Preserved subject | Current successor |
|---|---|---|
| [Aptus for Mac code and architecture review](aptus-macos/aptus-macos-code-review.md) | Point-in-time native host, sidecar, packaging, and macOS assessment | [macOS desktop architecture](../../docs/architecture/macos-desktop.md), [desktop build guide](../../desktop/macos/README.md) |
| [Pre-implementation codebase map](aptus-product-review/aptus-codebase-map.md) | Product-review source map | [Current code map](../../docs/architecture/code-map.md) |
| [Product and architecture review](aptus-product-review/aptus-product-review-code-review.md) | Pre-implementation product and release assessment | [Current capabilities](../../docs/product/current-capabilities.md), [release gates](../../docs/operations/release-gates.md) |
| [Maintained-client parity closeout](client-contract-parity-closeout/client-contract-parity-closeout-code-review.md) | Point-in-time React and Swift response-contract review | [API reference](../../docs/reference/api.md), [changing contracts](../../docs/contributing/changing-contracts.md) |
| [Initial model-compatibility concept review](model-compatibility-concept/model-compatibility-concept-code-review.md) | Pre-registry compatibility-authority assessment | [Model-policy snapshot](../../docs/reference/model-policy-snapshot.md) |
| [Phase 2 policy-registry review](model-compatibility-concept/phase2-policy-registry-code-review.md) | Host policy registry design boundary | [Model-policy snapshot](../../docs/reference/model-policy-snapshot.md) |
| [Phase 3 provenance-binding review](model-compatibility-concept/phase3-provenance-binding-code-review.md) | Receipt, decision, and plan-binding review | [Plan schema](../../docs/reference/plan-schema.md), [model-policy snapshot](../../docs/reference/model-policy-snapshot.md) |
| [Phase 4 portable-policy review](model-compatibility-concept/phase4-portable-policy-snapshot-code-review.md) | Portable snapshot and bundle review | [Model-policy snapshot](../../docs/reference/model-policy-snapshot.md), [bundle manifest](../../docs/reference/bundle-manifest.md) |
| [Phase 5 server-owned-policy UI review](model-compatibility-concept/phase5-server-owned-policy-ui-code-review.md) | Server-owned policy presentation review | [UI and UX contract](../../docs/product/ui-ux.md), [current capabilities](../../docs/product/current-capabilities.md) |
| [Phase 6 second-policy review](model-compatibility-concept/phase6-second-policy-code-review.md) | Qwen2 policy implementation before runtime closeout | [Exact-source MLX-LM acceptance](../../docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md), [model-policy snapshot](../../docs/reference/model-policy-snapshot.md) |
| [MoE compatibility implementation review](moe-compatibility/moe-compatibility-code-review.md) | Initial Qwen3 MoE implementation assessment | [Model-policy snapshot](../../docs/reference/model-policy-snapshot.md), [Qwen3 MoE admission evidence](../../docs/operations/evidence/2026-07-28-qwen3-moe-admission/README.md) |
| [MoE product-surface design plan](moe-compatibility/moe-compatibility-design-plan.md) | Historical UI design plan for MoE evidence | [UI and UX contract](../../docs/product/ui-ux.md), [capability matrix](../../docs/reference/capability-matrix.md) |

## Preservation rules

- Do not revise a historical body to match later implementation.
- Correct only status, navigation, provenance, legal, or security framing in
  place. Record new technical conclusions in a new dated review.
- Keep exact runtime evidence under `docs/operations/evidence/`; a review does
  not replace or broaden that evidence.
- Move a review back to `dev/active/` only when a maintainer explicitly reopens
  it with a new scope and review date.

## Related documentation

- [Historical documentation index](../../docs/archive/index.md)
- [Documentation maintenance policy](../../docs/maintenance/documentation-policy.md)
- [Documentation inventory](../../docs/maintenance/documentation-inventory.md)
- [Documentation debt log](../../docs/maintenance/documentation-debt.md)
