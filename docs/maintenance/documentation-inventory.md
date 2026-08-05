# Documentation inventory

> **Documentation status:** Active governance
>
> **Applies to:** Current Aptus 0.2 documentation after the Phase 6 runtime-evidence closeout
>
> **Last reviewed:** 2026-08-05
>
> **Next scheduled review:** 2026-10-27, or after any documentation move

This inventory identifies Aptus documentation surfaces, their lifecycle, and
their authority. It includes prose, machine-readable documentation, generated
bundle guidance, package metadata, inline help, and workbench copy.

## Inventory summary

The repository contains 103 governed tracked Markdown documents. This count
includes the dedicated model-policy-snapshot reference added by the Phase 4
documentation synchronization and the two Markdown records in the Phase 6
runtime-acceptance packet: one active acceptance record and one archived
nonqualifying diagnostic. It excludes 13 tracked working or implementation
documents outside the governed product-documentation scope.

| Lifecycle | Markdown files | Meaning |
|---|---:|---|
| Active | 86 | Current guidance, governance, navigation, evidence, or current research with explicit limits |
| Deprecated | 2 | Superseded v0.1 signposts |
| Archived | 15 | Historical research intake, legacy-audit evidence, the dated documentation-drift audit, and the nonqualifying Phase 6 diagnostic |
| Total | 103 | Excludes ignored local, generated, and tracked implementation-working artifacts |

The repository also contains one active machine-readable research catalog and
12 archived machine-readable legacy-audit records.

The automated `maintained_documentation()` set contains 94 Markdown files. It
deliberately retains the legacy-audit README while excluding its nine
subordinate historical pages. The 103-file governed inventory includes all ten
legacy-audit Markdown records, so these two counts measure different scopes.

## Active root documents

| Path | Type | Scope |
|---|---|---|
| [README.md](../../README.md) | Product landing page | Product boundary, installation, workbench, CLI, and portable execution |
| [CHANGELOG.md](../../CHANGELOG.md) | Changelog | Unreleased v0.2 changes and evidence status |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | Contributor guide | Setup, checks, design rules, claims, and pull requests |
| [ROADMAP.md](../../ROADMAP.md) | Roadmap | Future work that is not current support |
| [SECURITY.md](../../SECURITY.md) | Security policy | Trust, data copies, execution, and dependency boundaries |
| [SUPPORT.md](../../SUPPORT.md) | Support policy | Evidence to collect, sensitive-data boundary, and reporting routes |
| [LICENSE](../../LICENSE) | Legal | MIT license |

## Active documentation tree

### Navigation and maintenance

- [Documentation index](../index.md)
- [Architecture index](../architecture/index.md)
- [Getting-started index](../getting-started/index.md)
- [Guide index](../guides/index.md)
- [Operations index](../operations/index.md)
- [Product index](../product/index.md)
- [Reference index](../reference/index.md)
- [Documentation maintenance policy](documentation-policy.md)
- [Documentation inventory](documentation-inventory.md)
- [Documentation debt log](documentation-debt.md)
- [Documentation health report](documentation-health.md)
- [Archive index](../archive/index.md)
- [Research index](../research/index.md)

### Getting started and guides

- [Getting started](../getting-started/index.md)
- [Choose your Aptus path](../getting-started/choose-your-path.md)
- [First planning-only run](../getting-started/first-plan.md)
- [Install Aptus](../getting-started/install.md)
- [Quickstart](../getting-started/quickstart.md)
- [Task guides](../guides/index.md)
- [Choose a fine-tuning method](../guides/choose-a-method.md)
- [Compare plans](../guides/compare-plans.md)
- [Compile, validate, and run](../guides/compile-validate-run.md)
- [Design an evaluation](../guides/design-an-evaluation.md)
- [Inspect results](../guides/inspect-results.md)
- [Model, dataset, and hardware facts](../guides/model-dataset-hardware.md)
- [Prepare a dataset](../guides/prepare-a-dataset.md)
- [Recovery and the resume boundary](../guides/resume-recover.md)
- [Troubleshooting](../guides/troubleshooting.md)

### Product and architecture

- [Product documentation](../product/index.md)
- [Claim language](../product/claim-language.md)
- [Current capabilities](../product/current-capabilities.md)
- [UI and UX contract](../product/ui-ux.md)
- [User workflows](../product/user-workflows.md)
- [Product vision](../product/vision.md)
- [Architecture documentation](../architecture/index.md)
- [Artifact compiler](../architecture/artifact-compiler.md)
- [Code map](../architecture/code-map.md)
- [Data and identity flow](../architecture/data-and-identity-flow.md)
- [Execution orchestrator](../architecture/execution-orchestrator.md)
- [Security boundaries](../architecture/security-boundaries.md)
- [System architecture](../architecture/system.md)

### Methodology

- [Methodology overview](../methodology/overview.md)
- [Facts and provenance](../methodology/facts-and-provenance.md)
- [Fine-tuning method taxonomy](../methodology/method-taxonomy.md)
- [Candidate enumeration](../methodology/candidate-enumeration.md)
- [Precision and quantization](../methodology/precision-quantization.md)
- [Memory estimation](../methodology/memory-estimation.md)
- [Ranking and uncertainty](../methodology/ranking-uncertainty.md)
- [Preflight and calibration](../methodology/preflight-calibration.md)
- [Machine-readable method research catalog](../methodology/method-catalog.json)

The JSON catalog is documentation-only. Runtime method lifecycle and
selectability come from `src/aptus/methods/registry.py`.

### Operations and reference

- [Operations documentation](../operations/index.md)
- [Apple Silicon pilot matrix](../operations/apple-silicon-pilot.md)
- [Operator checklist](../operations/operator-checklist.md)
- [Release evidence template](../operations/release-evidence-template.md)
- [Release gates](../operations/release-gates.md)
- [2026-08-05 Phase 6 Qwen2 MLX-LM target-host acceptance](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md), two current v5/v3 repetitions at `14ed44b52a76bb84d8d9db4f2303951aa641339b` for the exact recorded artifact and environment
- [2026-07-27 MLX-LM target-host acceptance](../operations/evidence/2026-07-27-mlx-lm-acceptance/README.md)
- [2026-07-27 desktop engineering acceptance](../operations/evidence/2026-07-27-desktop-release/README.md)
- [2026-07-28 Qwen3 MoE admission and performance evidence](../operations/evidence/2026-07-28-qwen3-moe-admission/README.md)
- [2026-07-28 documentation drift audit](../operations/evidence/2026-07-29-documentation-drift-audit/README.md), archived as an immutable point-in-time record
- [State, storage, and retention](../operations/state-storage-retention.md)
- [Reference documentation](../reference/index.md)
- [API reference](../reference/api.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Capability matrix](../reference/capability-matrix.md)
- [CLI reference](../reference/cli.md)
- [Configuration defaults](../reference/configuration-defaults.md)
- [Dataset schemas](../reference/dataset-schemas.md)
- [Evidence records](../reference/evidence-records.md)
- [Error and finding codes](../reference/error-codes.md)
- [Glossary](../reference/glossary.md)
- [Method registry](../reference/method-registry.md)
- [Model-policy snapshot](../reference/model-policy-snapshot.md)
- [Plan schema](../reference/plan-schema.md)
- [Reviewed corpus contract](../reference/reviewed-corpus-contract.md)
- [Run states](../reference/run-states.md)
- [Validation states](../reference/validation-states.md)

### Active research governance and examples

- [EXAMPLE forensic review and salvage ledger](../research/example-intake-reconciliation.md)
- [Reference and TO-REVIEW reconciliation](../research/reference-and-to-review-reconciliation.md)
- [Top 50 method research source](../../Reference/top-50-llm-training-methods.pplx.md), active only as a non-normative research source
- [Reference packet index](../../Reference/README.md)
- [Examples guide](../../examples/README.md)

### Contributor guides

- [Contributor guide](../contributing/index.md)
- [Changing contracts](../contributing/changing-contracts.md)
- [Adding a fine-tuning method](../contributing/adding-a-method.md)
- [Generated code and bundle changes](../contributing/generated-code.md)
- [Workbench development](../contributing/workbench.md)
- [Pull-request template](../../.github/PULL_REQUEST_TEMPLATE.md)

## Deprecated documents

| Path | Reason | Current successor |
|---|---|---|
| [docs/design/aptus-core-vertical-slice.md](../design/aptus-core-vertical-slice.md) | v0.1 design is superseded | [Current capabilities](../product/current-capabilities.md), [system architecture](../architecture/system.md) |
| [docs/validation/aptus-core-smoke.md](../validation/aptus-core-smoke.md) | v0.1 smoke does not prove a v0.2 candidate | [Preflight and calibration](../methodology/preflight-calibration.md), [release gates](../operations/release-gates.md) |

These files remain at their known paths as explicit signposts. They are indexed
under [historical documentation](../archive/index.md).

## Archived research intake

| Path | Reason |
|---|---|
| [Reference/FineTuneX.README.md](../../Reference/FineTuneX.README.md) | Historical product and MCP concept, including unimplemented service and pricing claims |
| [Reference/Fine-Tuning_Methods.md](../../Reference/Fine-Tuning_Methods.md) | Unverified intake list with known factual errors |
| [Reference/hparam_methods_reference.md](../../Reference/hparam_methods_reference.md) | Uncited heuristic notes whose numeric values are not planner defaults |

The [Reference packet index](../../Reference/README.md) and
[reconciliation ledger](../research/reference-and-to-review-reconciliation.md)
record their exact authority boundaries.

## Archived legacy audit

The complete [Aptus legacy recovery audit](../audits/aptus-legacy/README.md) is
historical evidence for the removed `HyperTune/` source tree.

Human-readable records:

- `docs/audits/aptus-legacy/README.md`
- `docs/audits/aptus-legacy/architecture-options.md`
- `docs/audits/aptus-legacy/executive-summary.md`
- `docs/audits/aptus-legacy/extraction-ledger.md`
- `docs/audits/aptus-legacy/failure-and-risk-register.md`
- `docs/audits/aptus-legacy/hidden-gems.md`
- `docs/audits/aptus-legacy/provenance-report.md`
- `docs/audits/aptus-legacy/sandbox-summary.md`
- `docs/audits/aptus-legacy/static-python.md`
- `docs/audits/aptus-legacy/static-typescript.md`

Machine-readable records:

- `baseline-manifest.json`
- `claims-and-provenance.jsonl`
- `classification-summary.json`
- `classification.jsonl`
- `duplicate-clusters.json`
- `generated-bundle-manifest.json`
- `inventory.jsonl`
- `reference-map.json`
- `sandbox-results.jsonl`
- `secret-scan.json`
- `tsconfig.audit.json`
- `version-families.json`

These records must stay together. Their generator and reproduction commands use
the current `docs/audits/aptus-legacy/` path.

## Archived documentation audits and diagnostics

The
[2026-07-28 documentation drift audit](../operations/evidence/2026-07-29-documentation-drift-audit/README.md)
is an immutable point-in-time record of the tree at `e98ff55`. PR #14 applied
its corrective work, and this follow-up closes six partially applied prose
locations without altering the historical record. The audit cannot authorize
current behavior.

The Phase 6 acceptance packet retains an
[archived pre-fix diagnostic](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/diagnostics/attempt-01-unreceipted-parent-promotion/README.md)
that reproduced a managed parent-promotion defect at the static policy commit.
It is excluded from the two qualifying repetitions and remains only as negative
evidence and defect provenance.

## Generated bundle documentation

`src/aptus/generation.py` creates three active operator documents in every
compiled bundle:

| Generated file | Purpose |
|---|---|
| `README.md` | Candidate and plan identity, evidence warning, and command sequence |
| `decision-report.md` | Candidate comparison, selected execution contract, assumptions, and warnings |
| `runbook.md` | Ordered dependency, model-data, preflight, pilot, and full-run procedure |

It also generates command help in `train.py`, `run.py`, `preflight.py`, and
`validate.py`. Template changes require generated-bundle tests.

## Package, API, CLI, and workbench surfaces

| Surface | Authoritative paths |
|---|---|
| Python package metadata | `pyproject.toml`, `src/aptus/__init__.py` |
| Installed CLI help | `src/aptus/cli.py` |
| HTTP request and response contracts | `src/aptus/api_contracts.py`, `src/aptus/api.py` |
| Generated OpenAPI contract | `tools/generate_openapi.py`, `docs/reference/openapi.v1.json` |
| Generated TypeScript contract | `web/scripts/generate-openapi-client.mjs`, `web/src/generated/openapi.ts` |
| Maintained client boundary | `web/src/api.ts`, `web/src/types.ts`, `web/src/lib/modelPolicy.ts`, `desktop/macos/Sources/`, `tools/check_client_contracts.py` |
| Web metadata | `web/package.json`, `web/index.html` |
| Workbench copy | `web/src/App.tsx`, `web/src/components/`, `web/src/stages/`, `web/src/demo.ts`, and user-facing errors under `web/src/lib/` and `web/src/api.ts` |
| Packaged web build | `src/aptus/_web/index.html` and `src/aptus/_web/assets/` |
| Documentation checks | `tests/aptus/test_documentation.py`, `.github/workflows/ci.yml` |
| Pull-request guidance | `.github/PULL_REQUEST_TEMPLATE.md` |
| Issue intake | `.github/ISSUE_TEMPLATE/bug-report.yml`, `.github/ISSUE_TEMPLATE/documentation.yml`, `.github/ISSUE_TEMPLATE/config.yml` |

The packaged web build is generated from `web/`. It is a distribution artifact,
not a second hand-edited copy source.

## Tracked Markdown outside the lifecycle count

Thirteen tracked implementation or workflow files sit outside the 103-file
product-documentation lifecycle count:

- `.github/PULL_REQUEST_TEMPLATE.md`;
- `desktop/macos/README.md`; and
- the eleven active review and design notes under `dev/active/`.

They remain tracked and may govern their narrow implementation workflows. They
are not classified as active, deprecated, or archived product documentation.
This distinction explains the repository-wide total of 116 tracked Markdown
files.

## Excluded local and generated material

The following paths are not maintained repository documentation:

- `.audit-sandboxes/`, ignored disposable generated bundles
- `src/aptus.egg-info/`, generated package metadata
- `build/`, generated build output
- `.pytest_cache/README.md`, tool-generated cache guidance
- `.venv/` and `web/node_modules/`, installed dependencies
- `WIP.md`, an ignored local resume note rather than current authority or
  release evidence
- `TempDoc-ForUserReview/`, review-only replacement candidates awaiting an
  explicit apply decision

Do not index or publish these as current Aptus guidance. Review material under
`TempDoc-ForUserReview/` only as a proposed replacement set.

## Related documentation

- [Documentation maintenance policy](documentation-policy.md)
- [Documentation debt log](documentation-debt.md)
- [Documentation health report](documentation-health.md)
- [Documentation index](../index.md)
