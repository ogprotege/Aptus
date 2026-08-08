# Aptus Documentation

> **Status:** Active | **Authority:** Documentation navigation | **Applies to:** Aptus 0.2 | **Audience:** All readers | **Last reviewed:** 2026-08-08 | **Review by:** 2026-10-27 or when pages move

Aptus plans, compiles, validates, and locally runs a bounded set of supervised
fine-tuning strategies. These documents distinguish current product behavior,
operational evidence, future work, research inputs, and historical records.
The current portable contract uses `aptus.training-plan.v5`,
`aptus.bundle.v3`, and a canonical `aptus.model-policy-snapshot.v1`.

## Choose by goal

| I want to... | Read this first | Then continue with |
| --- | --- | --- |
| Understand what Aptus does | [Product vision](product/vision.md) | [Current capabilities](product/current-capabilities.md) |
| Run something safely on this Mac | [Install Aptus for Mac](getting-started/install.md#build-aptus-for-mac) | [Choose your path](getting-started/choose-your-path.md) |
| Plan for this Mac or a CUDA host | [Model, dataset, and hardware facts](guides/model-dataset-hardware.md) | [Compare plans](guides/compare-plans.md) |
| Choose a fine-tuning method | [Method selection guide](guides/choose-a-method.md) | [Method taxonomy](methodology/method-taxonomy.md) |
| Prepare a dataset | [Prepare a dataset](guides/prepare-a-dataset.md) | [Dataset schemas](reference/dataset-schemas.md) |
| Compile and run a bundle | [Quickstart](getting-started/quickstart.md) | [Operator checklist](operations/operator-checklist.md) |
| Understand a failure | [Troubleshooting](guides/troubleshooting.md) | [Error and finding codes](reference/error-codes.md) |
| Interpret a completed run | [Inspect results](guides/inspect-results.md) | [Design an evaluation](guides/design-an-evaluation.md) |
| Integrate Aptus | [API reference](reference/api.md) | [Plan schema](reference/plan-schema.md) |
| Understand portable policy identity and replanning | [Model-policy snapshot](reference/model-policy-snapshot.md) | [Plan schema](reference/plan-schema.md) |
| Change the code | [Contributor index](contributing/index.md) | [Code map](architecture/code-map.md) |
| Add a method | [Adding a method](contributing/adding-a-method.md) | [Method registry](reference/method-registry.md) |
| Prepare a release | [Release gates](operations/release-gates.md) | [Evidence template](operations/release-evidence-template.md) |
| Run the bounded RTX 3050 CUDA evidence campaign | [Canonical campaign plan](operations/cuda-empirical-campaign.md) | [State, storage, and retention](operations/state-storage-retention.md) |
| Review the Phase 2A source-tooling contract and Phase 2B preconditions | [Phase 2A tooling contract](operations/cuda-campaign-phase2-tooling.md) | [Canonical campaign plan](operations/cuda-empirical-campaign.md) |
| Inspect the exact CUDA acceptance | [2026-08-06 SmolLM2 CUDA LoRA single-device evidence](operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md) | [Complete packet and detailed results](operations/index.md#complete-ubuntu-cuda-acceptance-packet) |
| Inspect the current-contract MLX acceptance | [2026-08-05 Phase 6 MLX-LM evidence at exact source](operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md) | [Release gates](operations/release-gates.md) |
| Inspect the original Phase 6 MLX baseline | [2026-08-05 original acceptance](operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md) | [Current-contract evidence at exact source](operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md) |
| Inspect the historical MLX acceptance | [2026-07-27 MLX-LM evidence](operations/evidence/2026-07-27-mlx-lm-acceptance/README.md) | [Apple Silicon pilot matrix](operations/apple-silicon-pilot.md) |
| Inspect desktop build stability | [2026-07-27 desktop evidence](operations/evidence/2026-07-27-desktop-release/README.md) | [Release gates](operations/release-gates.md) |
| Inspect the Qwen3 MoE admission attempt | [2026-07-28 Qwen3 MoE evidence](operations/evidence/2026-07-28-qwen3-moe-admission/README.md) | [Capability matrix](reference/capability-matrix.md) |
| Review source research | [Research index](research/index.md) | [Retained Reference packet](../Reference/README.md) |

## Documentation status legend

- **Active:** current guidance or contract for Aptus 0.2.
- **Experimental:** a proposed procedure or capability without current product
  support.
- **Deprecated:** replaced guidance retained only to point to its successor.
- **Archived:** point-in-time evidence or intake material that must not define
  current behavior.

Authority labels have a separate meaning:

- **Normative:** owns a current product or data contract.
- **Operational:** owns a procedure or evidence gate.
- **Explanatory:** teaches or interprets a normative contract.
- **Historical:** preserves provenance and cannot authorize current behavior.

## Start and use Aptus

- [Getting-started index](getting-started/index.md)
- [Choose your path](getting-started/choose-your-path.md)
- [Install Aptus](getting-started/install.md)
- [First planning-only run](getting-started/first-plan.md)
- [Quickstart](getting-started/quickstart.md)
- [User workflows](product/user-workflows.md)
- [Prepare a dataset](guides/prepare-a-dataset.md)
- [Model, dataset, and hardware facts](guides/model-dataset-hardware.md)
- [Choose a method](guides/choose-a-method.md)
- [Compare plans](guides/compare-plans.md)
- [Compile, validate, and run](guides/compile-validate-run.md)
- [Inspect results](guides/inspect-results.md)
- [Design an evaluation](guides/design-an-evaluation.md)
- [Recovery and resume boundary](guides/resume-recover.md)
- [Troubleshooting](guides/troubleshooting.md)
- [Task-guide index](guides/index.md)

## Understand the methodology

- [Methodology overview](methodology/overview.md)
- [Facts and provenance](methodology/facts-and-provenance.md)
- [Fine-tuning method taxonomy](methodology/method-taxonomy.md)
- [Machine-readable research catalog](methodology/method-catalog.json)
- [Candidate enumeration](methodology/candidate-enumeration.md)
- [Precision and quantization](methodology/precision-quantization.md)
- [Memory estimation](methodology/memory-estimation.md)
- [Ranking and uncertainty](methodology/ranking-uncertainty.md)
- [Preflight and calibration](methodology/preflight-calibration.md)

## Understand the system

- [Product index](product/index.md)
- [Architecture index](architecture/index.md)
- [System architecture](architecture/system.md)
- [Code map](architecture/code-map.md)
- [Data and identity flow](architecture/data-and-identity-flow.md)
- [Artifact compiler](architecture/artifact-compiler.md)
- [Execution orchestrator](architecture/execution-orchestrator.md)
- [Security boundaries](architecture/security-boundaries.md)
- [macOS desktop host](architecture/macos-desktop.md)
- [UI and UX contract](product/ui-ux.md)
- [Product vision](product/vision.md)
- [Claim language](product/claim-language.md)

## Look up a contract

- [Reference index](reference/index.md)
- [CLI reference](reference/cli.md)
- [API reference](reference/api.md)
- [Generated OpenAPI contract](reference/openapi.v1.json)
- [Configuration and defaults](reference/configuration-defaults.md)
- [Dataset schemas](reference/dataset-schemas.md)
- [Method registry](reference/method-registry.md)
- [Plan schema](reference/plan-schema.md)
- [Model-policy snapshot](reference/model-policy-snapshot.md)
- [Bundle manifest](reference/bundle-manifest.md)
- [Evidence records](reference/evidence-records.md)
- [Capability matrix](reference/capability-matrix.md)
- [Validation states](reference/validation-states.md)
- [Run states](reference/run-states.md)
- [Error and finding codes](reference/error-codes.md)
- [Glossary](reference/glossary.md)
- [Reviewed corpus contract](reference/reviewed-corpus-contract.md)

## Operate and release

- [Operations index](operations/index.md)
- [Operator checklist](operations/operator-checklist.md)
- [State, storage, and retention](operations/state-storage-retention.md)
- [Release gates](operations/release-gates.md)
- [Release evidence template](operations/release-evidence-template.md)
- [RTX 3050 CUDA empirical evidence campaign](operations/cuda-empirical-campaign.md)
- [Phase 2A CUDA campaign tooling contract](operations/cuda-campaign-phase2-tooling.md)
- [2026-08-06 SmolLM2 CUDA LoRA single-device target-host acceptance](operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
- [Complete Ubuntu CUDA packet, detailed-result map, and raw-log retention boundary](operations/index.md#complete-ubuntu-cuda-acceptance-packet)
- [2026-08-05 Phase 6 Qwen2 MLX-LM current-contract evidence at exact source](operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
- [2026-08-05 original Phase 6 Qwen2 MLX-LM acceptance baseline](operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
- [2026-07-27 MLX-LM target-host acceptance](operations/evidence/2026-07-27-mlx-lm-acceptance/README.md)
- [2026-07-27 desktop engineering acceptance](operations/evidence/2026-07-27-desktop-release/README.md)
- [2026-07-28 Qwen3 MoE admission and performance evidence](operations/evidence/2026-07-28-qwen3-moe-admission/README.md)
- [2026-07-28 documentation drift audit](operations/evidence/2026-07-29-documentation-drift-audit/README.md)
- [Apple Silicon runtime and pilot matrix](operations/apple-silicon-pilot.md)
- [Security policy](../SECURITY.md)

## Contribute

- [Contributor index](contributing/index.md)
- [Code map](architecture/code-map.md)
- [Adding a method](contributing/adding-a-method.md)
- [Changing versioned contracts](contributing/changing-contracts.md)
- [Generated-code workflow](contributing/generated-code.md)
- [Workbench development](contributing/workbench.md)
- [Repository contribution policy](../CONTRIBUTING.md)
- [Support policy](../SUPPORT.md)

## Research and history

- [Research index](research/index.md)
- [Reference and former TO-REVIEW reconciliation](research/reference-and-to-review-reconciliation.md)
- [EXAMPLE forensic review and salvage ledger](research/example-intake-reconciliation.md)
- [Retained Reference packet](../Reference/README.md)
- [Historical archive index](archive/index.md)
- [Legacy recovery audit](audits/aptus-legacy/README.md)
- [Superseded v0.1 design](design/aptus-core-vertical-slice.md)
- [Superseded v0.1 smoke evidence](validation/aptus-core-smoke.md)

## Maintain the documentation

- [Documentation policy](maintenance/documentation-policy.md)
- [Documentation inventory](maintenance/documentation-inventory.md)
- [Documentation debt](maintenance/documentation-debt.md)
- [Documentation health](maintenance/documentation-health.md)
- [Changelog](../CHANGELOG.md)
- [Roadmap](../ROADMAP.md)

## Source-of-truth map

| Question | Normative owner |
| --- | --- |
| What is supported now? | [Capability matrix](reference/capability-matrix.md) and `src/aptus/methods/registry.py` |
| What does a validation state prove? | [Validation states](reference/validation-states.md) |
| What files belong to a bundle? | [Bundle manifest](reference/bundle-manifest.md) |
| How are policy decisions carried, checked, and kept current? | [Model-policy snapshot](reference/model-policy-snapshot.md), [plan schema](reference/plan-schema.md), [bundle manifest](reference/bundle-manifest.md), and [validation states](reference/validation-states.md) |
| What does the API accept? | [API reference](reference/api.md), `src/aptus/api_contracts.py`, and the [generated OpenAPI contract](reference/openapi.v1.json) |
| What does the CLI accept? | [CLI reference](reference/cli.md) and parser in `src/aptus/cli.py` |
| How are candidates ranked? | [Ranking and uncertainty](methodology/ranking-uncertainty.md) |
| How is memory estimated? | [Memory estimation](methodology/memory-estimation.md) |
| How does execution complete? | [Execution orchestrator](architecture/execution-orchestrator.md) |
| What blocks release? | [Release gates](operations/release-gates.md) |

## Evidence notice

Repository tests are necessary but do not replace target-runtime evidence.
Apple Silicon MLX-LM LoRA and QLoRA implement dependency, model-data, and
measured-preflight checks, an uninterrupted exact-model pilot, and confirmed
full-duration adapter training from the pinned base model. Two fresh, clean,
independent workflows
reached `measured-run-pass` in the
[2026-08-05 current-contract record at exact source](operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
under a v5 plan and v3 bundle at
`719255153e3fc7e38e83b5ff826d587e5e58bf80` with bundle fingerprint
`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`.
Relative to the unchanged [original Phase 6
baseline](operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md), only
manifested operator `README.md` and `runbook.md` changed; runtime programs and
requirements remained byte-identical. Fresh-process adapter reload and bounded
generation prove that the emitted adapter can be loaded. They do not prove
training resume, safety, model quality, performance, production throughput,
production readiness, or release readiness.

One separate exact SmolLM2 CUDA LoRA single-device workflow reached
`measured-run-pass` in the [2026-08-06 CUDA acceptance
record](operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
at source `c12c4d8db0037a2c278a2ad95a0a2cbda4387eed`. It completed the
five-action ladder, including checkpoint-continuation pilot, full training,
structural PEFT export, and parent promotion. That one execution does not
establish repeatability, general CUDA compatibility, quality, performance,
production readiness, or release readiness.

The August 5 MLX-LM acceptance supplies current-contract Phase 6 runtime
evidence at its exact acceptance source only for the pinned Qwen2.5 artifact
and revision, source and tree, host,
runtime,
dataset, policy snapshot, plan, bundle, and fingerprint. The 24-layer dense
Qwen2 policy remains a
configuration-footprint rule: every different matching artifact remains
conditional and must pass its own model-data, measured-preflight, and pilot
gates. The exact `qwen3_moe` MLX-LM QLoRA row remains conditional and has only
safe-refusal evidence; the recorded 30B attempt stopped before model loading
when live unified-memory admission failed. CUDA training remains an
external-host path on this Mac; only the exact separately recorded LoRA
single-device target-host workflow is qualified.
