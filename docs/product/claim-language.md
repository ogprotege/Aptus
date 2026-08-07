# Claim Language

> **Status:** Active | **Authority:** Normative claim policy | **Applies to:** Aptus 0.2 | **Audience:** Contributors and product writers | **Last reviewed:** 2026-08-06 | **Review by:** Every release

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

Inspection eligibility identifies that the inspected artifact matches a
reviewed compatibility subject and execution tuple. A configuration-footprint
policy is not an artifact allowlist: runtime evidence for one exact artifact and
immutable revision does not transfer to another artifact merely because both
match the policy subject. Eligibility also does not establish candidate
feasibility, dependency readiness, model-data validation, available capacity,
or pilot success. Likewise, a selectable candidate and compiler binding prove
that Aptus can emit the reviewed execution path; they do not prove that path
ran successfully on a target host.

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
rechecks current resources. The two fresh, clean
[2026-08-05 MLX-LM exact-source acceptance
runs](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
prove only their exact Qwen2.5 artifact, immutable revision, synthetic dataset,
Apple M5 Pro host, Python and MLX-LM runtime, policy snapshot, v5 plan, v3
bundle, source commit `719255153e3fc7e38e83b5ff826d587e5e58bf80`,
source tree, bundle fingerprint
`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`,
and actions. They close the Phase 6
runtime gate for that exact fixture; they do not prove every Apple Silicon
configuration or every artifact that matches the Qwen2 configuration
footprint. Relative to the unchanged [original Phase 6
baseline](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md),
only manifested operator `README.md` and `runbook.md` changed; runtime programs
and requirements remained byte-identical. They establish neither safety, model
quality, performance, production throughput, production readiness, nor release
readiness, and they do not qualify CUDA. The 2026-07-27
v2-plan and v2-bundle runs remain historical evidence for their older exact
scope.

The separate [2026-08-06 CUDA LoRA single-device
record](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
supports this exact wording: “one exact SmolLM2 CUDA LoRA single-device
workflow reached `measured-run-pass` under the recorded source, host, runtime,
model revision, synthetic dataset, plan, policy, bundle, and five-job
sequence.” Do not shorten that to “CUDA passed.” The record is one execution,
does not establish repeatability, and does not qualify another CUDA method,
placement, artifact, device, host, or environment.

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

- “two fresh, clean current-contract MLX-LM workflows reached
  `measured-run-pass` for the exact recorded acceptance configuration at source
  commit `719255153e3fc7e38e83b5ff826d587e5e58bf80` and bundle fingerprint
  `ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`”;
- “the local desktop engineering gate passed 10 of 10 clean builds at
  implementation commit `1038ecdd13103418ef1135e1ced634c10370a961`”;
- “pull-request CI rebuilt and packaged GitHub's exact tested merge commit,” but
  only after that workflow has completed successfully;
- “the default desktop artifact is ad-hoc signed for review and testing”;
- “the exact Phase 6 Qwen2 fixture passed two current-contract v5/v3 MLX-LM
  ladders at the recorded acceptance source”;
- “one exact SmolLM2 CUDA LoRA single-device workflow completed the five-action
  ladder through `measured-run-pass`; repeatability and other CUDA paths remain
  open.”

Do not:

- apply the historical ten-build result or its artifact hashes to a later
  commit, branch head, or pull-request merge commit;
- describe ad-hoc signing as Developer ID distribution, notarization, or public
  release approval;
- treat desktop packaging as CUDA target-host acceptance;
- apply either the historical July MLX-LM result or the exact 2026-08-05
  acceptance to another artifact, revision, host, runtime, dataset, source
  commit, or CUDA path;
- apply the 2026-08-06 CUDA result to another method, placement, artifact,
  revision, device, host, runtime, dataset, source, plan, policy, or bundle;
- call v0.2 release-ready while remaining claimed CUDA target-host coverage and
  public Developer ID signing and notarization remain open.

Repository tests, real MLX target-host evidence, desktop stability evidence,
workflow-commit CI packaging, CUDA acceptance, and public distribution approval
are separate claims. State which one the evidence supports.

## Related documentation

- [Current capabilities](current-capabilities.md)
- [Model-policy snapshot](../reference/model-policy-snapshot.md)
- [Validation states](../reference/validation-states.md)
- [Release gates](../operations/release-gates.md)
- [2026-08-05 Qwen2 MLX-LM current-contract evidence at exact source](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
- [2026-08-06 SmolLM2 CUDA LoRA single-device acceptance](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
- [Original Phase 6 Qwen2 MLX-LM acceptance baseline](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
