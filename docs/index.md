# Aptus Documentation

> **Status:** Active | **Authority:** Documentation navigation | **Applies to:** Aptus 0.2 | **Audience:** All readers | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when pages move

Aptus plans, compiles, validates, and locally runs a bounded set of supervised
fine-tuning strategies. These documents distinguish current product behavior,
operational evidence, future work, research inputs, and historical records.

## Choose by goal

| I want to... | Read this first | Then continue with |
| --- | --- | --- |
| Understand what Aptus does | [Product vision](product/vision.md) | [Current capabilities](product/current-capabilities.md) |
| Run something safely on this Mac | [Choose your path](getting-started/choose-your-path.md) | [First-plan tutorial](getting-started/first-plan.md) |
| Plan for a CUDA host | [Model, dataset, and hardware facts](guides/model-dataset-hardware.md) | [Compare plans](guides/compare-plans.md) |
| Choose a fine-tuning method | [Method selection guide](guides/choose-a-method.md) | [Method taxonomy](methodology/method-taxonomy.md) |
| Prepare a dataset | [Prepare a dataset](guides/prepare-a-dataset.md) | [Dataset schemas](reference/dataset-schemas.md) |
| Compile and run a bundle | [Quickstart](getting-started/quickstart.md) | [Operator checklist](operations/operator-checklist.md) |
| Understand a failure | [Troubleshooting](guides/troubleshooting.md) | [Error and finding codes](reference/error-codes.md) |
| Interpret a completed run | [Inspect results](guides/inspect-results.md) | [Design an evaluation](guides/design-an-evaluation.md) |
| Integrate Aptus | [API reference](reference/api.md) | [Plan schema](reference/plan-schema.md) |
| Change the code | [Contributor index](contributing/index.md) | [Code map](architecture/code-map.md) |
| Add a method | [Adding a method](contributing/adding-a-method.md) | [Method registry](reference/method-registry.md) |
| Prepare a release | [Release gates](operations/release-gates.md) | [Evidence template](operations/release-evidence-template.md) |
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
- [UI and UX contract](product/ui-ux.md)
- [Product vision](product/vision.md)
- [Claim language](product/claim-language.md)

## Look up a contract

- [Reference index](reference/index.md)
- [CLI reference](reference/cli.md)
- [API reference](reference/api.md)
- [Configuration and defaults](reference/configuration-defaults.md)
- [Dataset schemas](reference/dataset-schemas.md)
- [Method registry](reference/method-registry.md)
- [Plan schema](reference/plan-schema.md)
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
- [Apple Silicon experiment matrix](operations/apple-silicon-pilot.md)
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
| What does the API accept? | [API reference](reference/api.md) and strict models in `src/aptus/api.py` |
| What does the CLI accept? | [CLI reference](reference/cli.md) and parser in `src/aptus/cli.py` |
| How are candidates ranked? | [Ranking and uncertainty](methodology/ranking-uncertainty.md) |
| How is memory estimated? | [Memory estimation](methodology/memory-estimation.md) |
| How does execution complete? | [Execution orchestrator](architecture/execution-orchestrator.md) |
| What blocks release? | [Release gates](operations/release-gates.md) |

## Evidence notice

Repository tests are necessary but do not replace a target-host pilot. No real
CUDA pilot has been completed on this development Mac. Current execution claims
remain conditional until the release record passes every applicable gate.
