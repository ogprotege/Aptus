# Architecture

> **Status:** Active | **Authority:** Documentation navigation | **Applies to:** Aptus 0.2 | **Audience:** Contributors, operators, and integrators | **Last reviewed:** 2026-08-04 | **Review by:** 2027-01-22 or when system boundaries change

Read these pages in order when learning the implementation:

1. [System architecture](system.md) for the complete lifecycle.
2. [Code map](code-map.md) for module responsibilities and test ownership.
3. [Data and identity flow](data-and-identity-flow.md) for digests, bindings,
   copies, and trust boundaries.
4. [Artifact compiler](artifact-compiler.md) for no-clobber bundle publication.
5. [Execution orchestrator](execution-orchestrator.md) for jobs, admission,
   policy currency, cancellation, recovery, and completion.
6. [macOS desktop host](macos-desktop.md) for the native lifecycle, bridge,
   private session, and packaging boundary.
7. [Security boundaries](security-boundaries.md) for the local trust model.

The architecture explains why Aptus behaves as it does. Exact public values
belong in the [reference index](../reference/index.md).

## Related documentation

- [Documentation home](../index.md)
- [Contributor index](../contributing/index.md)
- [Methodology overview](../methodology/overview.md)
