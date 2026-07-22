# Research and intake index

> **Documentation status:** Active research navigation
>
> **Authority:** Non-normative until admitted through Aptus runtime contracts
>
> **Applies to:** Retained source packets, reconciliation ledgers, and method research
>
> **Last reviewed:** 2026-07-22
>
> **Next scheduled review:** 2026-10-22, or when the research cutoff or method registry changes

Research helps Aptus decide what to investigate. It does not make a method
selectable. A paper can establish a mechanism or reported experiment, but it
cannot prove compatibility with Aptus's pinned stack, a user's exact model and
data, or a target host.

## Current research records

### Source packet

- [Reference packet index](../../Reference/README.md)
- [Top 50 algorithmic methods](../../Reference/top-50-llm-training-methods.pplx.md)

The Top 50 report is an active, non-normative research source with a dated
cutoff. Its rank order is editorial. Library mappings must be checked against
the versions Aptus actually pins before implementation.

### Reconciliation ledgers

- [Reference and former TO-REVIEW reconciliation](reference-and-to-review-reconciliation.md)
- [EXAMPLE forensic review and salvage ledger](example-intake-reconciliation.md)

These ledgers preserve source disposition and explain which concepts entered
current contracts, which moved to the roadmap, and which remain archived or
rejected.

### Current normalized method documentation

- [Fine-tuning method taxonomy](../methodology/method-taxonomy.md)
- [Machine-readable method research catalog](../methodology/method-catalog.json)
- [Capability matrix](../reference/capability-matrix.md)
- [Current capabilities](../product/current-capabilities.md)
- [Roadmap](../../ROADMAP.md)

`src/aptus/methods/registry.py` is authoritative for runtime lifecycle and
selectability. The research catalog explicitly remains documentation-only.

## Research admission checklist

A new method, objective, recipe, modifier, optimizer strategy, backend, or
distribution path does not become executable until Aptus has:

1. a stable identity and primary source;
2. a precise category in the method taxonomy;
3. a pinned maintained implementation or an owned compiler implementation;
4. explicit model, dataset, precision, quantization, backend, and distribution
   compatibility rules;
5. a trainable-parameter and optimizer-membership contract;
6. memory and disk estimates whose uncertainty is labeled;
7. checkpoint, continuation, cancellation, and failure semantics;
8. an export, reload, and verification contract;
9. planner and compiler integration with fail-closed unsupported cases;
10. static, dependency, model-data, preflight, pilot, and full-run evidence on
    the target hardware;
11. documentation and tests that match the shipped behavior.

## Archived intake

The following retained sources are historical only:

- [FineTuneX product history](../../Reference/FineTuneX.README.md)
- [Unverified fine-tuning method names](../../Reference/Fine-Tuning_Methods.md)
- [Unverified hyperparameter notes](../../Reference/hparam_methods_reference.md)

Read their status warnings and the reconciliation ledger before using any
concept. Known factual errors and unverified numeric heuristics must not enter
the planner.

## Related documentation

- [Historical documentation index](../archive/index.md)
- [Documentation maintenance policy](../maintenance/documentation-policy.md)
- [Documentation inventory](../maintenance/documentation-inventory.md)
- [Methodology overview](../methodology/overview.md)
