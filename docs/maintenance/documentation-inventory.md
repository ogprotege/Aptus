# Documentation inventory

> **Documentation status:** Active governance
>
> **Applies to:** Current Aptus 0.2 documentation after PR #41 and the canonical CUDA campaign integration, including the Phase 1 protocol freeze, Phase 2A source-tooling contract, Phase 2B sanitized recovery supplement, both immutable Phase 5 cohort outcomes, the historical Phase 6 packets, the Full confirmatory-stability outcome, the complete reviewed Phase 7 outcomes, the reviewed Phase 8 guarded-frontier outcome, the reviewed Phase 9 endurance outcome, and the reviewed Phase 10 campaign certification
>
> **Last reviewed:** 2026-08-16
>
> **Next scheduled review:** 2026-10-27, or after any documentation move

This inventory identifies Aptus documentation surfaces, their lifecycle, and
their authority. It includes prose, machine-readable documentation, generated
bundle guidance, package metadata, inline help, and workbench copy.

## Inventory summary

The repository tree contains 146 tracked Markdown documents. Of those, 145 are
governed: every tracked Markdown file except the pull-request template,
whose submitted-body contract remains exempt from reader-page metadata. PR #41
added the historical engineering-review index, classified the twelve completed
reviews under `dev/archive/`, brought the native desktop build guide into the
active set, and applied archived metadata to every legacy-audit report. This
canonical campaign integration added one active plan, and the Phase 1 protocol
freeze added one active human-readable protocol. Phase 2A added one active
source-tooling contract, and Phase 2B adds one active, independently reviewed
sanitized recovery-evidence packet. Phase 5 adds one active stopping-rule
outcome packet, and the successful replacement cohort adds a separate active
repeatability-anchor packet without overwriting that failure history. Phase 6
adds an active method-matrix outcome packet, and its corrected remediation adds
a separate active packet without overwriting the earlier cohort history. The
fixed-source Full cohort adds its active stability packet without altering
either prior Phase 6 packet. Phase 7 adds the historical stopped scale-staircase
packet and a separate current same-family stability packet without overwriting
that first cohort. Architecture breadth adds one reviewed amendment, one
append-only parameter-semantics correction, and one active final stability
packet without overwriting the amendment, correction, or stopped cohorts;
Phase 8 adds one active reviewed guarded-frontier packet, and Phase 9 adds one
active reviewed endurance and job-control packet. Phase 10 adds one active
reviewed certification packet and closes the campaign without authorizing a
later phase. The mission integrity program adds one active product program plan
at `docs/product/mission-integrity-plan.md` (working notes live outside
`docs/` under `.superpowers/mission-integrity-plan/`, not under `dev/active/`).
Path Alpha M3 adds the operator runbook
`docs/guides/path-alpha-mlx-operator.md` and the evidence packet README
`docs/operations/evidence/2026-08-12-path-alpha-mlx-m3/README.md`. Path Beta M4
adds the operator runbook `docs/guides/path-beta-cuda-lora-operator.md` and the
evidence packet README
`docs/operations/evidence/2026-08-12-path-beta-cuda-lora-m4/README.md`.
M6 adds the public Mac packaging packet
`docs/operations/evidence/2026-08-13-desktop-public-release/README.md`.
M7-C adds
`docs/operations/evidence/2026-08-13-path-beta-cuda-reload-m7c/README.md`.
M7-A adds
`docs/operations/evidence/2026-08-13-path-beta-360m-lora-m7a/README.md`.
The training-policy increment adds one active implementation plan at
`docs/superpowers/plans/2026-08-16-training-policy-and-run-correction.md`.

| Lifecycle | Markdown files | Meaning |
|---|---:|---|
| Active | 116 | Current guidance, governance, navigation, evidence, protocol, or current research with explicit limits |
| Deprecated | 2 | Superseded v0.1 signposts |
| Archived | 27 | Historical research intake, legacy-audit evidence, twelve engineering reviews, the dated documentation-drift audit, and the nonqualifying Phase 6 diagnostic |
| Total | 145 | Excludes only the pull-request workflow template from tracked Markdown governance |

The repository also contains one active machine-readable research catalog, one
active machine-readable CUDA campaign protocol companion, and 12 archived
machine-readable legacy-audit records.

The automated `maintained_documentation()` set contains the 145 governed
Markdown files. Together with the root `LICENSE`, that makes 146 maintained
reader documents.
Metadata, link, anchor, and primary-index reachability checks
therefore cover the native build guide, all ten legacy-audit reports, the
engineering-review index, and all twelve archived engineering reviews.

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
- [Historical engineering-review index](../../dev/archive/README.md)

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
- [Mission integrity program plan](../product/mission-integrity-plan.md)
- [Training policy and run-correction implementation plan](../superpowers/plans/2026-08-16-training-policy-and-run-correction.md)
- [Architecture documentation](../architecture/index.md)
- [Artifact compiler](../architecture/artifact-compiler.md)
- [Code map](../architecture/code-map.md)
- [Data and identity flow](../architecture/data-and-identity-flow.md)
- [Execution orchestrator](../architecture/execution-orchestrator.md)
- [Security boundaries](../architecture/security-boundaries.md)
- [System architecture](../architecture/system.md)
- [Desktop implementation and build guide](../../desktop/macos/README.md)

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
- [RTX 3050 CUDA empirical evidence campaign](../operations/cuda-empirical-campaign.md)
- [CUDA campaign Phase 2 tooling contract](../operations/cuda-campaign-phase2-tooling.md), implemented source and review authority that supplies no new Ubuntu or empirical result
- [CUDA campaign protocol](../reference/cuda-campaign-protocol.md), the frozen Phase 1 human contract; it implements no runtime behavior
- [CUDA campaign protocol machine companion](../reference/cuda-campaign-protocol.v1.json), the canonical machine-readable projection of the same frozen decisions
- [Apple Silicon pilot matrix](../operations/apple-silicon-pilot.md)
- [Operator checklist](../operations/operator-checklist.md)
- [Release evidence template](../operations/release-evidence-template.md)
- [Release gates](../operations/release-gates.md)
- [2026-08-11 CUDA Phase 10 campaign certification](../operations/evidence/2026-08-11-cuda-phase10-certification/README.md), the reviewed 149-slot aggregate and final campaign decision with no new training, no replacement runs, exact-host claim boundaries, and an explicit end to the Phase 0–10 campaign
- [2026-08-11 CUDA Phase 9 endurance and job control](../operations/evidence/2026-08-11-cuda-phase9-endurance/README.md), the reviewed three-slot 300-update endurance outcome with exact counters, aggregate rates, complete custody, and eight passing controlled job-service exercises; its then-current Phase 10 non-authorization remains immutable history
- [2026-08-11 CUDA Phase 8 guarded configuration frontier](../operations/evidence/2026-08-11-cuda-phase8-guarded-frontier/README.md), the reviewed bounded-pilot frontier outcome with three closed one-axis ladders, complete custody, a deterministic Phase 9 candidate, and no Phase 9 authorization
- [2026-08-09 Phase 2B sanitized Phase 0 recovery supplement](../operations/evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md), independently reviewed recovery-integrity evidence for the protected August 6 records; not target-runtime, performance, repeatability, or release-readiness evidence
- [2026-08-09 CUDA Phase 5 repeatability-anchor outcome](../operations/evidence/2026-08-09-cuda-phase5-repeatability-anchor/README.md), a target-host conditioning capture failure that applies the frozen no-replacement rule and does not establish repeatability or Phase 6 eligibility
- [2026-08-10 CUDA Phase 5 repeatability anchor](../operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md), the separate successful five-slot replacement cohort that establishes the exact-host anchor and Phase 6 eligibility within its frozen boundary
- [2026-08-10 CUDA Phase 6 Full confirmatory stability](../operations/evidence/2026-08-10-cuda-phase6-confirmatory-stability/README.md), the separate fixed-source five-slot cohort that establishes one stable exact-host Full cell and authorizes the bounded Phase 7 procedure
- [2026-08-10 CUDA Phase 7 scale staircase](../operations/evidence/2026-08-10-cuda-phase7-scale-staircase/README.md), the completed fail-closed staircase outcome with one passing 135M LoRA slot, a thermal admission stop before slot two activated, no stable Phase 7 cell, and no Phase 8 authorization
- [2026-08-11 CUDA Phase 7 same-family stability](../operations/evidence/2026-08-11-cuda-phase7-same-family-stability/README.md), the new no-replacement cohort with stable 135M LoRA, 135M Full, and 360M LoRA cells; architecture breadth still requires separate review and Phase 8 remains unauthorized
- [2026-08-11 CUDA Phase 7 architecture-breadth stability](../operations/evidence/2026-08-11-cuda-phase7-breadth-stability/README.md), the final no-replacement breadth cohort with a stable Qwen3-0.6B LoRA cell; Phase 7 is complete and Phase 8 remains unauthorized pending separate activation review
- [2026-08-10 CUDA Phase 6 remediation method matrix](../operations/evidence/2026-08-10-cuda-phase6-remediation-matrix/README.md), the corrected 32-slot outcome in which Full was promoted, then failed to establish confirmatory stability; no stable method or Phase 7 authorization resulted
- [2026-08-10 historical CUDA Phase 6 method matrix](../operations/evidence/2026-08-10-cuda-phase6-method-matrix/README.md), the immutable earlier-cohort outcome with no promoted method or confirmatory execution
- [2026-08-06 SmolLM2 CUDA LoRA single-device target-host acceptance](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md), one exact five-job `measured-run-pass` workflow at `c12c4d8db0037a2c278a2ad95a0a2cbda4387eed` for the recorded host, runtime, model revision, synthetic dataset, plan, policy, and bundle
- [2026-08-05 Phase 6 Qwen2 MLX-LM current-contract evidence at exact source](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md), two fresh v5/v3 repetitions at `719255153e3fc7e38e83b5ff826d587e5e58bf80` for the exact recorded artifact, source tree, M5 Pro host, Python/MLX runtime, dataset, plan, policy snapshot, and bundle fingerprint `ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`
- [2026-08-05 original Phase 6 Qwen2 MLX-LM acceptance baseline](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md), the unchanged historical baseline at `14ed44b52a76bb84d8d9db4f2303951aa641339b`
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

## Archived engineering reviews

The [historical engineering-review index](../../dev/archive/README.md) governs
twelve preserved records at their subject-relative paths under `dev/archive/`:

- one macOS host review;
- two product and codebase reviews;
- one maintained-client parity closeout;
- six model-compatibility policy reviews spanning the initial concept and
  Phases 2 through 6; and
- two MoE compatibility review and design records.

Each record carries an archived and superseded warning before its unchanged
historical body. The index maps every record to current architecture, product,
contract, or evidence guidance. No file remains under `dev/active/` in the
merged tree.

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
`validate.py`. MLX bundles additionally generate `reload.py`, whose fresh-child
adapter reload is inference evidence rather than task-quality evidence. A
registry-derived test matrix must equal every executable runtime, backend,
method, and placement row, then compile each row and verify the three operator
documents before a template change can pass.

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

## Workflow-template metadata exemption

The single tracked Markdown file outside the 139-file lifecycle count is
`.github/PULL_REQUEST_TEMPLATE.md`. It is still a governed workflow interface,
but reader-page metadata would leak into every submitted pull-request body.
Its fields and review path are instead named by the maintenance policy and
contributor documentation. This exemption explains the repository-wide total
of 140 tracked Markdown files.

## Excluded local and generated material

The following paths are not maintained repository documentation:

- `.audit-sandboxes/`, ignored disposable generated bundles
- `src/aptus.egg-info/`, generated package metadata
- `build/`, generated build output
- `.pytest_cache/README.md`, tool-generated cache guidance
- `.venv/` and `web/node_modules/`, installed dependencies
- `WIP.md`, an ignored local resume note rather than current authority or
  release evidence
- `TempDoc-ForUserReview/`, local review provenance that is not current
  repository authority

Do not index or publish these as current Aptus guidance. Material under
`TempDoc-ForUserReview/` is local review provenance after PR #41 and has no
authority over the merged documentation.

## Related documentation

- [Documentation maintenance policy](documentation-policy.md)
- [Documentation debt log](documentation-debt.md)
- [Documentation health report](documentation-health.md)
- [Documentation index](../index.md)
