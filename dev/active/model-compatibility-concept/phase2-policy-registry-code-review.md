# Phase 2 Compatibility Policy Registry Code Review

**Last Updated:** 2026-07-29
**Review basis:** Merged `main` at `e7ce942`, after PR #22.
**Scope:** Host-side domain types, compatibility policy evaluation, provider
inspection, candidate planning, method runtime bindings, API validation, tests,
and maintained documentation.

## Executive Summary

Phase 1 closed the public compatibility vocabulary, but the exact Qwen3 MoE
policy still exists separately in provider inspection and candidate planning.
The smallest safe Phase 2 correction is one host-side policy registry that both
surfaces call.

The registry must sit above `domain.py`, `catalog.py`, and the method registry.
It must sit below inspection, planning, and HTTP. Domain types remain free of
registry imports. Compiler, estimator, export, and evidence identities continue
to come from the method registry.

This phase must preserve `aptus.api.v1`, `aptus.training-plan.v3`, every current
candidate and plan identity, and the self-contained portable contract. Receipts,
policy versioning, plan bindings, policy snapshots, and browser simplification
belong to later phases.

## Critical Issues

### CRIT-1: Inspection and planning still decide the same policy separately

`inspection.py` matches exact identity, quantization layout, topology, runtime,
method, backend, distribution, and adapter profile. `planning.py` repeats those
predicates before it admits one Qwen3 MoE candidate. The two branches can drift
when a policy row changes.

**Required correction:** Add immutable domain decision and path types. Add one
`model_compatibility.py` registry and evaluator. Make inspection and planning
consume the same decision.

### CRIT-2: Core inspection depends on the HTTP response model

`inspection.py` imports `ModelCompatibilityResponse` only to seal its producer.
That reverses the documented dependency direction.

**Required correction:** Put structural and execution-path invariants below
HTTP. Inspection should adapt a validated domain decision to the unchanged v1
JSON shape. Pydantic remains a defensive response boundary.

### CRIT-3: A registry could become a second method-runtime authority

The method registry already owns compiler, estimator, export, backend, runtime,
distribution, and evidence-requirement bindings. Copying those values into a
model registry would replace one duplication with another.

**Required correction:** Store only the selected method, runtime, backend,
distribution, adapter profile, and target-module policy. Resolve the complete
`RuntimeContract` from the method registry.

## Important Improvements

### IMP-1: Claim sparse near-matches before dense-family fallback

Any exact or partial Qwen3 MoE marker, and any unreviewed MoE topology, must be
claimed by sparse policy evaluation and blocked when it does not match. It must
not fall through as a recognized dense family.

### IMP-2: Preserve planning identity byte for byte

The policy decision remains transient in Phase 2. It must not enter
`CandidatePlan`, `TrainingPlan`, their serializers, or plan identity. Exact
Qwen3 fixture candidate IDs and the plan ID need snapshot regressions.

### IMP-3: Preserve direct planning entry points

`plan_training()` should evaluate the model once and pass the immutable decision
through candidate enumeration. Direct `estimate_candidate()` calls must still
work by evaluating when no decision was supplied.

## Minor Suggestions

- Keep the existing public `conditional`, `recognized`, and `unsupported`
  variants until the versioned decision contract is introduced.
- Keep provider input-shape warnings separate from policy rejection.
- Add isolated import-order tests so a future refactor cannot create a domain,
  method-registry, and compatibility cycle.

## Architecture Considerations

The approved dependency direction is:

```text
domain types -> catalog and method registry -> model compatibility evaluator
             -> inspection and planning -> API and CLI
```

The portable `plan_contract.py` remains independent during Phase 2. It will
continue to repeat the Qwen policy until Phase 4 emits a canonical snapshot and
digest into each bundle.

The internal path record binds:

```text
method            qlora
runtime           mlx-lm
backend           mps
distribution      single
adapter profile   attention-qkvo.v1
target modules    q_proj, k_proj, v_proj, o_proj
```

The live method registry supplies the compiler, estimator, export, and evidence
requirement for that tuple.

## Next Steps

1. Add structural subject, path, and decision types to `domain.py`.
2. Add the host-side policy registry and shared evaluator.
3. Add a method-registry helper that builds a `RuntimeContract`.
4. Replace the inspection and planning policy branches with evaluator calls.
5. Delegate API execution-tuple validation to the shared path validator.
6. Add table-driven mutation, identity-preservation, import-order, and parity
   tests.
7. Update current architecture, API, capability, maintenance, and change notes.
8. Run full Python, web, generated-contract, native, packaging, and independent
   adversarial checks.

**Approval status:** The user approved continued phased implementation after
PR #22 merged. Phase 2 may proceed within the boundaries above.

## Final Adversarial Closeout

The first independent pass found three admission defects that the initial green
suite missed: policy-impossible API claims, sparse identity markers falling
through as dense families when topology was absent, and caller-injected policy
decisions in direct candidate estimation. The deeper pass also found incomplete
fact-error handling, an unsealed CLI projection, possible target-catalog drift,
and incomplete runtime-binding startup validation.

The last semantic scan found one presentation-boundary error: a Qwen near-match
with malformed facts could inherit exact-identity rejection copy. The evaluator
now distinguishes claimed identity from established exact identity, and the v1
projector allowlists the exact-identity failure reasons.

The implementation now rejects each case. API claims must match a complete
family-specific registry path and evidence requirement. Public candidate
estimation always evaluates its own model. Exact policy matches reject recorded
fact conflicts. The v1 projector seals every path against the model, adapter,
and method registries. Registry construction fails if adapter and family target
tuples diverge. Every runtime binding must produce a canonical runtime contract
at import time.

The final local gate passed 386 Python tests, 84 React tests, and 81 native
tests, plus formatting, lint, compilation, generated-contract, maintained-client,
version, typecheck, production-build, packaged-launch, signing, ZIP, and DMG
checks. OpenAPI artifacts and the v1 schema remain unchanged. Dense CUDA and
Qwen3 MoE v3 plan and candidate identities match the merged baseline. A final
read-only adversarial pass found no remaining code or schema blocker.
