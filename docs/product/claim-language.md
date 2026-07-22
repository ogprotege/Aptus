# Claim Language

> **Status:** Active | **Authority:** Normative claim policy | **Applies to:** Aptus 0.2 | **Audience:** Contributors and product writers | **Last reviewed:** 2026-07-22 | **Review by:** Every release

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
- “current train admission passed”;
- “the parent verified the structural export file tree.”

Do not turn a historical pilot pass into a claim of current capacity. Admission
rechecks current resources.

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

Do not call v0.2 release-ready while the release gates lack claimed target-host
CUDA and MLX pilot and full-run evidence. Repository tests are not a substitute.

## Related documentation

- [Current capabilities](current-capabilities.md)
- [Validation states](../reference/validation-states.md)
- [Release gates](../operations/release-gates.md)
