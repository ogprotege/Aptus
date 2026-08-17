# Claim Language

> **Status:** Active | **Authority:** Normative claim policy | **Applies to:** Aptus 0.2 | **Audience:** Contributors and product writers | **Last reviewed:** 2026-08-16 | **Review by:** Every release

Product language must match the strongest available evidence.

## Planning claims

Use:

- “recommended within the enumerated v0.2 candidate set”;
- “analytic point estimate”;
- “heuristic upper envelope”;
- “method-class prior”;
- “below the instruction-SFT supervision prior of 100 rows”;
- “exceeds the instruction-SFT epoch-cap prior of 3”;
- “Aptus will not rewrite the requested epoch count”;
- “parrot/sycophancy over-training prior”;
- “eligible for the reviewed pilot path” when inspection binds the complete
  known runtime, compute backend, method, distribution, and adapter profile;
- “conditional on a target-host pilot”;
- “unsupported by the current compiler contract.”
- “operator-attested unreviewed runtime”;
- “not the reviewed Path Alpha 24-layer footprint.”

Do not use:

- “universally optimal”;
- “optimal LoRA rank”;
- “3 epochs is optimal”;
- “this dataset will produce a sycophant”;
- “loss proves the model is bad”;
- “guaranteed to fit”;
- “perfect configuration”;
- “automatic best method”;
- “the runtime supports this method” based only on model inspection;
- “zero-risk training.”
- “Aptus supports 7B Qwen like the reviewed 0.5B Path Alpha path”;
- “reviewed 7B MLX identity” after an unreviewed-runtime confirm.

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

The separate [2026-08-12 Path Alpha MLX
packet](../operations/evidence/2026-08-12-path-alpha-mlx-m3/README.md) supports
this exact wording: “two fresh v6-plan / v3-bundle MLX QLoRA ladders reached
`measured-run-pass` for the frozen Path Alpha identity at recorded source
`f4775c01e6b8f932e11c2d665e90859d6aedbe04`.” Do not call that current HEAD or
every Apple Silicon host.

The [2026-08-12 Path Beta CUDA
packet](../operations/evidence/2026-08-12-path-beta-cuda-lora-m4/README.md)
supports this exact wording: “one managed Path Beta LoRA ladder reached
`measured-run-pass` with structural PEFT export on the recorded Ubuntu / RTX
3050 host at recorded source
`93d69f63c7d3c1147ce186e810c355cdcf1a1b9c` plus the CUDA public-version pin
fix.” Do not call that current HEAD or a second host class.

The separate [2026-08-06 CUDA LoRA single-device
record](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
supports this exact wording: “one exact SmolLM2 CUDA LoRA single-device
workflow reached `measured-run-pass` under the recorded source, host, runtime,
model revision, synthetic dataset, plan, policy, bundle, and five-job
sequence.” Do not shorten that to “CUDA passed.” The record is one execution,
does not establish repeatability, and does not qualify another CUDA method,
placement, artifact, device, host, or environment.

The later [2026-08-10 Phase 5 repeatability
packet](../operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md)
supports this additional wording: “five of five predeclared SmolLM2 CUDA LoRA
single-device slots passed the frozen common stability and integrity contract
for the exact recorded source, host, environment, model revision, dataset, and
configuration.” It may be called an exact-host repeatability anchor. Do not
shorten it to “CUDA is repeatable,” transfer it to another scope, or describe
it as quality, broad performance, safety, production, or release evidence.

The [2026-08-11 Phase 10 campaign
certification](../operations/evidence/2026-08-11-cuda-phase10-certification/README.md)
supports this aggregate wording: “the bounded RTX 3050 campaign completed 149
planned slots with 58 starts, 91 predeclared-not-started dispositions, 47
qualifying outcomes, and no replacement runs.” It also supports naming the six
exact stable cells listed in that packet, the 17-point guarded frontier, and
the three-run 900-update endurance plus eight-exercise job-control outcome.
Always retain their exact-source, exact-host, exact-runtime, model, method, and
configuration boundaries.

Do not shorten that aggregate to “CUDA is certified,” “CUDA is stable,” or
“Aptus found the RTX 3050 limit.” The two Phase 8 `CUDA_OOM` outcomes were
bounded pilot results; they are not permission to run full training merely to
provoke an OOM. Phase 10 is the final approved campaign phase. Do not describe
release-gate work or a future campaign as Phase 11.

## Quality claims

Training loss, evaluation loss from a split, export structure, or job completion
does not establish task quality by itself. Instruction-SFT supervision and
epoch-cap priors label dataset size and requested epochs; they do not predict
model quality, sycophancy, or that a given epoch count is optimal. Quality
language requires a named dataset, metric, threshold, baseline, run binding,
and result. V0.2 provides that first-class binding as optional
`aptus.evaluation-contract.v1` / `aptus.evaluation-result.v1` with deterministic
`exact_match` only. A contract pass means the bound gold digest, supplied
predictions, metric implementation, and threshold were met. It is not general
quality, safety, human preference, or release evidence. `evaluation_fraction`
remains a train/validation split control. It is not this contract.

## Training-signal correction claims

After a measured run with recorded loss observations, Aptus may attach
`aptus.run-correction.v1` as a presentation-only next-plan hint.

Use:

- “training-signal correction”;
- “regularization heuristic”;
- “next plan”;
- “Train loss fell while validation loss rose” only as a curve description, not
  as an evaluation pass or fail.

Do not use:

- “the model is bad” / “the model is good” from loss alone;
- “overfit confirmed as quality”;
- “eval pass” from split validation loss;
- “AutoML” or “start a hyperparameter search” as the product response;
- “add weight decay as a sycophancy cure.”

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
`aptus.training-plan.v6` plans and `aptus.bundle.v3` bundles bind one canonical
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
- “source `edc6cfdec48daeb17af8cae7dbb9fde0d8112a81` has one Developer ID
  signed notarized arm64 app and DMG”, only with the 2026-08-13 packet
  hashes and notary IDs;
- “Path Beta LoRA single on the recorded RTX 3050 host reloaded its PEFT
  adapter in a fresh process and generated 1–4 tokens”, only with the
  2026-08-13 M7-C packet;
- “SmolLM2-360M-Instruct LoRA single reached `measured-run-pass` on the
  recorded RTX 3050 host”, only with the 2026-08-13 M7-A packet;
- “the exact Phase 6 Qwen2 fixture passed two current-contract v5/v3 MLX-LM
  ladders at the recorded acceptance source”;
- “five of five predeclared SmolLM2 CUDA LoRA single-device slots passed the
  frozen stability and integrity contract, establishing the exact-host Phase 5
  repeatability anchor within its exact recorded boundary”;
- “the Phase 10 review certified the bounded 149-slot campaign aggregate,
  including six exact stable cells, the guarded frontier, and the recorded
  endurance/job-control outcome, without adding training or replacement runs.”

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
- generalize the Phase 10 exact-host result to broad CUDA, cloud, multi-GPU,
  safety, quality, performance, or production readiness;
- invent or imply a Phase 11; or
- call v0.2 release-ready while semantic export, model-quality,
  production-safety, broader target-runtime coverage, and public Developer ID
  signing and notarization remain open.

Repository tests, real MLX target-host evidence, desktop stability evidence,
workflow-commit CI packaging, CUDA acceptance, and public distribution approval
are separate claims. State which one the evidence supports.

## Related documentation

- [Current capabilities](current-capabilities.md)
- [Model-policy snapshot](../reference/model-policy-snapshot.md)
- [Validation states](../reference/validation-states.md)
- [Release gates](../operations/release-gates.md)
- [2026-08-11 CUDA Phase 10 campaign certification](../operations/evidence/2026-08-11-cuda-phase10-certification/README.md)
- [2026-08-05 Qwen2 MLX-LM current-contract evidence at exact source](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
- [2026-08-06 SmolLM2 CUDA LoRA single-device acceptance](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
- [Original Phase 6 Qwen2 MLX-LM acceptance baseline](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
