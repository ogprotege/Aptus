# Aptus Documentation

> **Status:** Active | **Authority:** Documentation navigation | **Applies to:** Aptus 0.2 | **Audience:** All readers | **Last reviewed:** 2026-08-05 | **Review by:** 2026-10-27 or when pages move

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
| Inspect the current MLX acceptance | [2026-08-05 Phase 6 MLX-LM evidence](operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md) | [Release gates](operations/release-gates.md) |
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
- [2026-08-05 Phase 6 Qwen2 MLX-LM target-host acceptance](operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
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
full-duration adapter training from the pinned base model. Two clean workflows
reached `measured-run-pass` in the
[2026-08-05 record](operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
under a v5 plan and v3 bundle at
`14ed44b52a76bb84d8d9db4f2303951aa641339b`. Fresh-process adapter reload and
bounded generation prove that the emitted adapter can be loaded. They do not
prove training resume, model quality, or production throughput.

That acceptance closes the current-source Phase 6 MLX-LM runtime gate only for
the exact pinned Qwen2.5 artifact, host, runtime, dataset, policy snapshot,
plan, and bundle. The 24-layer dense Qwen2 policy remains a
configuration-footprint rule: every different matching artifact remains
conditional and must pass its own model-data, measured-preflight, and pilot
gates. The exact `qwen3_moe` MLX-LM QLoRA row remains conditional and has only
safe-refusal evidence; the recorded 30B attempt stopped before model loading
when live unified-memory admission failed. CUDA training remains an
external-host path on this Mac, with no qualifying target-host run recorded.
