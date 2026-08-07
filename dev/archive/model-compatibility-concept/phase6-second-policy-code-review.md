# Phase 6 code review: second model policy

> **Documentation status:** Archived and superseded review evidence
>
> **Applies to:** Point-in-time Phase 6 second-policy code review recorded below
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or a named successor changes
>
> **Historical warning:** This review is preserved without rewriting its body.
> Its statement that runtime acceptance remains open predates the later exact
> MLX-LM acceptance packet. Use the [historical-review index](../README.md) for
> the current successor and evidence boundary.

> **Last Updated:** 2026-08-04

## Executive Summary

The Phase 6 implementation is architecturally consistent with the portable
model-policy design. It adds a second reviewed MLX-LM QLoRA configuration
footprint as registry data, evaluates both host and bundle decisions through
the same portable rule engine, and keeps runtime evidence scoped to the exact
pinned artifact that produced it. The implementation does not add a Qwen2
branch to the planner, compiler, API, or browser.

The review found no critical or important code defects. Focused policy,
planning, portable-contract, and bundle-generation coverage passes. Current-head
runtime acceptance remains intentionally open and is the remaining Phase 6
release-evidence gate.

## Critical Issues (must fix)

None found.

## Important Improvements (should fix)

None required before the runtime evidence ladder.

## Minor Suggestions (nice to have)

1. A later registry-cleanup change could colocate each policy's explanatory
   reason text with its claims and constraints. The evaluator is already
   data-driven, but `current_model_policy_snapshot()` still assembles the
   shared reason-text table from module constants.
2. If compatibility evaluation becomes a measured hot path, cache a detached
   validated current snapshot rather than reconstructing it for every host
   decision. The current approach favors simplicity and makes mutation tests
   explicit; no present performance problem is established.
3. If third-party policy registration is introduced, replace the internal
   mapping-valued frozen policy fields with an explicitly immutable rule value
   type. Today the registry is private module data and the emitted snapshot is
   JSON-detached before use.

These are optional follow-ups. No changes from this section should be made
without a separately reviewed scope.

## Architecture Considerations

- `model.qwen2-24l.mlx-qlora` is a reviewed runtime-configuration footprint,
  not an artifact allowlist. Its compatibility subject deliberately excludes
  model ID and revision. The two historical measured runs remain bound by
  evidence text and revision to
  `mlx-community/Qwen2.5-0.5B-Instruct-4bit@53a32aee5e9447773fd2b85988395066aef3700a`.
- Claims use Qwen2 model type and architecture without claiming the entire
  `qwen` family. Exact identity, 24 layers, dense topology, explicit four-bit
  metadata, and the uniform group-size-64 layout are constraints. A sparse
  identity marker bypasses a dense claim and retains fail-closed sparse
  handling.
- The host evaluator consumes the canonical snapshot evaluator and rehydrates
  the primitive decision into the domain type. This removes the former
  callback/manual-snapshot dual authority.
- Receipt validation resolves `required_provenance_fields` from the matching
  policy row. Dense Qwen2 receipts do not invent provenance for a null `moe`
  fact; Qwen3 MoE continues to require its topology field.
- The dense MLX affine storage path now handles an explicit empty override list
  without requiring MoE topology. Non-empty overrides still take the existing
  topology-aware path and fail closed without MoE facts.
- The public API remains `aptus.api.v1`; plans remain
  `aptus.training-plan.v5`; bundles remain `aptus.bundle.v3`; policy snapshots
  remain `aptus.model-policy-snapshot.v1`. The additions extend closed enum and
  registry data without changing their structural shapes.
- The browser adds only the new adapter-profile and reason-code vocabulary. It
  does not reconstruct Qwen2 identity, layer, topology, or quantization rules.

## Next Steps

1. Finish synchronized active documentation without relabeling the historical
   July evidence as current-head acceptance.
2. Run Python, Ruff, web test, typecheck, build, OpenAPI, and packaged-resource
   gates.
3. After explicit authorization for dependency and model downloads plus
   training, run two clean current-contract dependency, model-data, measured
   preflight, pilot, confirmed train, final-export, and fresh-reload ladders.
4. Preserve a sanitized evidence packet bound to the exact implementation
   commit, snapshot digest, model revision, runtime, host, dataset, and outputs.
5. Re-review the final evidence and only then mark Phase 6 complete.
