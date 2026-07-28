# Claim Language

> **Status:** Active | **Authority:** Normative claim policy | **Applies to:** Aptus 0.2 | **Audience:** Contributors and product writers | **Last reviewed:** 2026-07-27 | **Review by:** Every release

Product language must match the strongest available evidence.

## Planning claims

Use:

- “recommended within the enumerated v0.2 candidate set”;
- “analytic point estimate”;
- “heuristic upper envelope”;
- “conditional on a target-host pilot”;
- “unsupported by the current compiler contract.”

Do not use:

- “universally optimal”;
- “guaranteed to fit”;
- “perfect configuration”;
- “automatic best method”;
- “zero-risk training.”

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

## Release claims

Use:

- “two clean MLX-LM workflows reached `measured-run-pass` for the recorded
  acceptance configuration”;
- “the local desktop engineering gate passed 10 of 10 clean builds at
  implementation commit `1038ecdd13103418ef1135e1ced634c10370a961`”;
- “pull-request CI rebuilt and packaged GitHub's exact tested merge commit,” but
  only after that workflow has completed successfully; and
- “the default desktop artifact is ad-hoc signed for review and testing.”

Do not:

- apply the historical ten-build result or its artifact hashes to a later
  commit, branch head, or pull-request merge commit;
- describe ad-hoc signing as Developer ID distribution, notarization, or public
  release approval;
- treat desktop packaging as CUDA target-host acceptance; or
- call v0.2 release-ready while claimed CUDA target-host evidence and public
  Developer ID signing and notarization remain open.

Repository tests, real MLX target-host evidence, desktop stability evidence,
workflow-commit CI packaging, CUDA acceptance, and public distribution approval
are separate claims. State which one the evidence supports.

## Related documentation

- [Current capabilities](current-capabilities.md)
- [Validation states](../reference/validation-states.md)
- [Release gates](../operations/release-gates.md)
