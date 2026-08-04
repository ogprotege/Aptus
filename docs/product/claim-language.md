# Claim Language

> **Status:** Active | **Authority:** Normative claim policy | **Applies to:** Aptus 0.2 | **Audience:** Contributors and product writers | **Last reviewed:** 2026-08-04 | **Review by:** Every release

Product language must match the strongest available evidence.

## Planning claims

Use:

- “recommended within the enumerated v0.2 candidate set”;
- “analytic point estimate”;
- “heuristic upper envelope”;
- “eligible for the reviewed pilot path” when inspection binds the complete
  known runtime, compute backend, method, distribution, and adapter profile;
- “conditional on a target-host pilot”;
- “unsupported by the current compiler contract.”

Do not use:

- “universally optimal”;
- “guaranteed to fit”;
- “perfect configuration”;
- “automatic best method”;
- “the runtime supports this method” based only on model inspection;
- “zero-risk training.”

Inspection eligibility identifies a reviewed artifact and execution tuple. It
does not establish candidate feasibility, dependency readiness, model-data
validation, available capacity, or pilot success.

## Runtime claims

Use:

- “dependency validation passed for the recorded environment”;
- “every canonical row transformed successfully”;
- “measured preflight passed on the recorded hardware”;
- “CUDA checkpoint continuation was observed in the bounded pilot”;
- “the MLX adapter completed an uninterrupted pilot and reloaded for bounded
  generation in a fresh process”;
- “the recorded MLX-LM workflow completed confirmed full training, final export,
  fresh-process reload, and parent verification to `measured-run-pass`”;
- “current train admission passed”;
- “the parent verified the structural export file tree.”

Do not turn a historical pilot pass into a claim of current capacity. Admission
rechecks current resources. The two clean 2026-07-27 MLX-LM acceptance runs
prove only their exact model, immutable revision, synthetic dataset, host,
runtime, plan, bundle, and actions. They do not prove every Apple Silicon
configuration or model.

## Quality claims

Training loss, evaluation loss from a split, export structure, or job completion
does not establish task quality by itself. Quality language requires a named
dataset, metric, threshold, baseline, run binding, and result. V0.2 does not yet
provide that first-class evaluation contract.

## Dependency claims

Call generated `requirements.txt` “exact direct pins” or “direct constraints.”
Do not call it a complete lock file. The installed environment binding supplies
the actual resolved-runtime record.

## Artifact claims

“Structural export verification” means paths, sizes, hashes, safetensors keys,
index mappings, and model or adapter provenance were checked. It does not mean
the artifact was benchmarked, judged safe, or proven deployable.

## Policy snapshot claims

Use:

- “package-free validation confirmed the frozen policy snapshot's canonical
  integrity and reproduced its compatibility decision”;
- “installed Aptus confirmed that the bundle's snapshot matched the current host
  registry at validation or admission time”; and
- “the coherent saved plan requires deterministic replanning because its policy
  semantics or snapshot digest is no longer current.”

Do not:

- claim that package-free validation proved current host-policy currency;
- describe a coherent stale-policy plan as malformed or tampered;
- change a historical plan's schema, decision, snapshot, or digest and call that
  migration; or
- describe one host's current-policy check as proof that another host still uses
  the same registry.

Portable integrity and installed-host currency are separate claims. Current
`aptus.training-plan.v5` plans and `aptus.bundle.v3` bundles bind one canonical
`aptus.model-policy-snapshot.v1`; the installed registry remains the authority
for managed admission and execution.

## Release claims

Use:

- “two clean MLX-LM workflows reached `measured-run-pass` for the recorded
  acceptance configuration”;
- “the local desktop engineering gate passed 10 of 10 clean builds at
  implementation commit `1038ecdd13103418ef1135e1ced634c10370a961`”;
- “pull-request CI rebuilt and packaged GitHub's exact tested merge commit,” but
  only after that workflow has completed successfully;
- “the default desktop artifact is ad-hoc signed for review and testing”;
- “Phase 4's source, wheel, and desktop package gates passed, but no current-head
  MLX or CUDA target-runtime pilot was collected.”

Do not:

- apply the historical ten-build result or its artifact hashes to a later
  commit, branch head, or pull-request merge commit;
- describe ad-hoc signing as Developer ID distribution, notarization, or public
  release approval;
- treat desktop packaging as CUDA target-host acceptance;
- apply the July MLX-LM acceptance result to the later Phase 4 source head;
- call v0.2 release-ready while claimed CUDA target-host evidence and public
  Developer ID signing and notarization remain open.

Repository tests, real MLX target-host evidence, desktop stability evidence,
workflow-commit CI packaging, CUDA acceptance, and public distribution approval
are separate claims. State which one the evidence supports.

## Related documentation

- [Current capabilities](current-capabilities.md)
- [Model-policy snapshot](../reference/model-policy-snapshot.md)
- [Validation states](../reference/validation-states.md)
- [Release gates](../operations/release-gates.md)
