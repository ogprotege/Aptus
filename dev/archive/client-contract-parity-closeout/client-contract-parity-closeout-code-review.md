# Maintained client contract-parity closeout code review

> **Documentation status:** Archived and superseded review evidence
>
> **Applies to:** Point-in-time maintained-client parity closeout recorded below
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or a named successor changes
>
> **Historical warning:** This closeout review is preserved without rewriting
> its body. Statements below that say a condition is current, open, or complete
> describe the reviewed snapshot, not the present repository. Use the
> [historical-review index](../README.md) to find current successors.

> **Last Updated:** 2026-08-05

## Executive Summary

The closeout follows Aptus's fail-closed architecture. React now validates the
remaining response normalizers before hydration, while the native host validates
all four HTTP responses it consumes. The Swift/OpenAPI checker protects endpoint,
required-field, constant, and closed-status parity; semantic behavior remains
covered by focused Vitest and XCTest matrices.

The combined review found three integration defects before the full gates:

1. selectable method rows did not validate the registry's required runtime
   bindings;
2. strict job decoding rejected the valid empty `bundle_dir` used by migrated
   legacy records; and
3. profile provenance accepted only two of the five domain-defined values.

All three were corrected in the reviewed diff. No critical or important issue
remains. Full repository, packaged-workbench, and macOS release-build
verification passed.

## Critical Issues (must fix)

None found after the combined review corrections.

## Important Improvements (should fix)

None required for this closeout.

## Minor Suggestions (nice to have)

1. `BootstrapResponse.bundle` and `CapabilitiesResponse.method_catalog` remain
   intentionally open dictionaries in the server model. A later HTTP-contract
   change could publish explicit nested response models and generate more of the
   browser boundary instead of maintaining it by hand.
2. The native checker protects required top-level OpenAPI fields and closed
   values; XCTest protects nested and cross-field semantics. If a generated
   Swift transport layer is adopted later, retain the semantic tests rather than
   treating generated decoding as sufficient evidence.
3. Job-service extension fields remain an open response surface. The browser
   validates the core v1 record plus the optional values it normalizes today;
   future rendered job fields should gain focused ingress checks when added.

## Architecture Considerations

- Python and OpenAPI remain the response authority. Generated OpenAPI JSON and
  TypeScript were not hand-edited.
- React preserves unknown extra properties for forward compatibility but rejects
  missing required fields, unknown schema identities, invalid closed values, and
  contradictory execution metadata before rendering.
- Restored bundles may omit a validation report; a live compile response may not.
- Native health readiness now depends on status, service version, and API
  contract identity rather than HTTP 200 alone.
- Runtime inventory decoding requires all six declared fields and independently
  reconciles advertised availability and compatibility with interpreter probes.
- Platform `unsupported` is a typed server outcome. Unknown statuses and malformed
  optional values remain invalid responses.
- The changes do not alter planner, compiler, runtime, evidence, or release
  semantics and therefore do not require a new runtime pilot.

## Verification Results

- Python: 550 tests passed.
- React: 130 tests passed; TypeScript and the production Vite build passed.
- Ruff formatting and lint, Python compileall, OpenAPI freshness, native client
  contract parity, version parity, and `git diff --check` passed.
- `desktop/macos/build.sh` passed its repeated Python/web gates, native tests,
  Release build, app-signature verification, app ZIP creation, and DMG creation.
- The generated production workbench is included; `desktop/macos/dist/` remains
  ignored and outside the change.

## Next Steps

1. Commit the reviewed source, tests, documentation, checker, and generated
   workbench together.
2. Publish a draft pull request with exact verification results and the remaining
   CUDA/public-distribution evidence boundary.
