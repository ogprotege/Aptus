# Documentation health report

> **Documentation status:** Active governance
>
> **Applies to:** Repository documentation through merged PR #41, the canonical
> RTX 3050 CUDA campaign integration, the Phase 1 protocol freeze, and the
> Phase 2A source-tooling contract, Phase 2B sanitized recovery supplement, and
> the two immutable Phase 5 cohort outcomes, and both the historical and
> corrected Phase 6 method-matrix outcomes
>
> **Last reviewed:** 2026-08-10
>
> **Next scheduled review:** 2026-11-01, or after the next contract-changing pull request

## Overall assessment

Aptus has a substantive v0.2 documentation set. It explains evidence boundaries,
supported methods, planning assumptions, bundle structure, validation states,
execution controls, and known release blockers with unusual care. The main risk
is maintenance scale. The same contracts appear in prose, code, generated
artifacts, API shapes, CLI help, and workbench copy; generated surfaces and
maintained decoders therefore need explicit parity checks.

Overall documentation health is **strong after a bounded reconciliation, with
release-evidence gaps still explicit**. The 2026-08-06 repository-wide audit
found one high-impact documentation omission, several medium-impact consistency
and lifecycle gaps, and low-impact freshness defects. PR #41 corrected
those documentation surfaces and added regression coverage without changing
runtime behavior or broadening measured evidence. The earlier
2026-07-28 drift audit identified 32 confirmed locations across 12 root causes.
PR #14 corrected the implementation defect and the core documentation, while a
strict closeout found six partially completed locations across root causes B, D,
and E. PR #19 closed those locations and added semantic regression checks
without rewriting the immutable audit record. PR #20 added compatibility-copy
coverage, but post-merge review found its placement assertion could pass against
`single-device` in the reason even if the visible placement clause disappeared.
PR #21 corrected that assertion, introduced a status-discriminated API contract,
sealed the producer, and added browser fail-closed normalization. PR #22 closed
the remaining open vocabulary. Conditional compatibility now
uses known runtime, compute-backend, method, distribution, and adapter-profile
IDs, validates the tuple against the method registry, and describes only
eligibility for a reviewed pilot path. PR #23 implemented the Phase 2
host-side policy registry for provider inspection, sparse candidate admission,
and API execution-path validation. That historical phase intentionally
preserved the v1 API, v3 plan identity, portable contract, and evidence
boundary. PR #24 implemented the historical Phase 3 v4 plan, versioned
decisions, inspection receipts, and exact-path bindings. PR #25 advanced the
current Phase 4 contract to `aptus.training-plan.v6` and `aptus.bundle.v3` with
a deterministic portable policy snapshot. PRs #26 through #28 hardened
malformed-input handling, rejected non-object manifests, and completed the
remaining contract and package-free regression coverage. PR #29 synchronized
the central contract guidance. PR #30 completed the independent re-review,
closed remaining plan and manifest parser boundaries, and recorded the final
source/contract verdict. PR #15 preserved and indexed the historical audit
without making it current authority. The current Phase 5 implementation removes
browser-side policy reconstruction, adds strict v2 decision, path, receipt, and
candidate/report ingress, and presents artifact match, selected candidate path,
and evidence readiness as separate records. The typed HTTP 422
`no_feasible_plan` path preserves the same server policy chain, correlates it
with the required model subject, request, and receipt, and requires rejected-only
complete candidate tuples.

The current Phase 6 implementation expands the same data-driven registry with
`model.qwen2-24l.mlx-qlora` and
`mlx-lm.qlora.single.dense-causal-lm.v1`. Exact 24-layer dense Qwen2 identity,
uniform four-bit group-size-64 layout, seven-module adapter scope, portable
parity, per-policy receipt provenance, planner binding, and canonical evidence
are implemented without a new top-level contract version. The policy describes
a reviewed configuration footprint rather than an artifact allowlist. The
[2026-08-05 Qwen2 MLX-LM exact-source
refresh](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
records two fresh, clean, independent current-contract
`aptus.training-plan.v6` and `aptus.bundle.v3` `measured-run-pass` repetitions
for the exact pinned artifact and revision, source commit
`719255153e3fc7e38e83b5ff826d587e5e58bf80`, source tree
`be99f5664ccb580f2600471f1ae3241a294b1a7e`, Apple M5 Pro host, Python/MLX
runtime, dataset, policy snapshot, plan, and bundle fingerprint
`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`.
The [original Phase 6 acceptance](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
remains the unchanged historical baseline.

The current Phase 4 contract uses a frozen snapshot for package-free portable
integrity and policy-decision parity. Portable validation cannot determine host
policy currency. Validation under an installed Aptus host additionally compares
the bundle bindings with the current host registry and requires replanning when
that registry has changed.

Phase 5 leaves that portable contract intact. The workbench rejects malformed or
misbound server policy records before UI hydration, requires a non-null binding
for exact path equality, and applies validation evidence only when plan,
candidate, and model-revision bindings match. Unbound and rejected rows receive
no synthesized policy ladder or action. The same exact three-field binding gates
stage completion and validation or run actions. Recommendations structurally
equal their complete listed candidate records, and provider path-matched receipts
cannot satisfy provenance with inferred-only observations. Evidence completeness
stays separate from the optional typed `authorization_status` values `current`,
`deferred`, and `blocked`; their boolean and diagnostic fields must agree, and a
tuple with no non-null member means not checked. The client does not infer
status from prose or mutate the report after a generic training-request failure. Non-current
authorization does not itself mean stale policy or replanning. The receipt's v2
decision is the one inspection-time browser policy source; the unused flattened
compatibility normalizer was removed. The MoE rail owns topology and
resident-versus-active memory only. The second Phase 6 policy reaches that same
generic browser boundary without family-specific reconstruction.

That refresh supplies current-contract Phase 6 MLX-LM runtime evidence at its
exact acceptance source. A different matching artifact remains conditional and must pass
its own model-data, measured-preflight, and pilot gates. The record does not
qualify CUDA, establish safety or model quality, establish general Qwen2
compatibility, support performance or production-throughput claims, or
establish production or release readiness. Relative to the historical
baseline, only manifested operator `README.md` and `runbook.md` changed;
runtime programs and requirements remained byte-identical. The two fresh runs
independently qualify the new fingerprint.

The [2026-08-06 CUDA
record](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
adds one exact SmolLM2 LoRA single-device workflow through
`measured-run-pass`, including a two-phase checkpoint-continuation pilot, full
training, structural PEFT export, and parent promotion. It is bound to its
exact source, host, runtime, model, dataset, plan, policy, and bundle. It does
not establish repeatability or general CUDA compatibility.

On 2026-08-08, campaign Phase 0 completed in the protected private evidence
layer with a complete disposition, two verified copies in separate failure
domains, and verified off-host retrieval. No private locator, machine or job
identity, or raw record is published by this status. Phase 2B used the exact
merged Phase 2A source to publish the independently reviewed [sanitized
recovery supplement](../operations/evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md).
It accounts for all 40 logical rows: 39 recovered with matching digests and one
raw model-file manifest not found; the separately searched Python transcript
was also not found or reconstructed. The
[CUDA campaign protocol](../reference/cuda-campaign-protocol.md) and its
[machine companion](../reference/cuda-campaign-protocol.v1.json) freeze Phase 1
as design authority; those two protocol documents implement no runtime
behavior. The separate [Phase 2A source-tooling
contract](../operations/cuda-campaign-phase2-tooling.md) documents implemented
and independently reviewed Phase 4 authority, admission/activation, outcome,
capture, telemetry, watchdog, custody, sanitization, eligibility, and
publication interfaces. Their integrated development-Mac source gates passed.
Neither phase connected to or mutated the Ubuntu host, ran a model or GPU
workload, or added a new empirical result. Phases 3 and 4 later completed their
selection, measurement, rehearsal, and freeze controls. The separately merged
[Phase 5 repeatability packet](../operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md)
records five of five frozen SmolLM2 LoRA single-device slots passing the common
stability and integrity contract and authorizes Phase 6 from that exact anchor.
The corrected
[Phase 6 remediation packet](../operations/evidence/2026-08-10-cuda-phase6-remediation-matrix/README.md)
records Full promotion from three passing exploratory attempts, followed by one
pass and four unrelated-GPU-activity safety cancellations in five confirmatory
attempts. No stable Phase 6 method was established and Phase 7 remains
unauthorized. The earlier Phase 6 packet remains an immutable historical cohort
record.

The exact Qwen3 30B MLX-LM attempt remains safe-refusal evidence: it stopped
before model loading. It is not a passing pilot or training result. The release
itself remains blocked until the remaining CUDA target-host evidence and a
Developer ID signed and notarized public desktop distribution exist for the
capabilities being claimed. The July MLX-LM acceptance and local 10-build
desktop engineering gate and original Phase 6 acceptance remain historical at
their recorded commits; the August 5 refresh is bounded current-contract
MLX-LM evidence at its exact acceptance source. The exact CUDA LoRA
single-device repeatability anchor is now established; the corrected Phase 6
matrix establishes no stable method. Other CUDA methods, placements, artifacts,
hosts, and environments remain unqualified. The repository checks its
principal navigation and executable-reference surfaces, but it does not yet
derive every default, status, and response field from one source.

## Scorecard

| Area | Result | Evidence |
|---|---|---|
| Product boundaries | Good | Current capabilities, claim language, roadmap, and release gates distinguish implemented, conditional, unsupported, and future work |
| Evidence language | Good | Planning estimates, measured checks, structural export verification, and task quality are kept separate |
| User workflow coverage | Good | Installation, quickstart, facts, comparison, compilation, validation, execution, recovery, and troubleshooting are present |
| API and CLI reference | Good | Automated checks cover commands, options, choices, defaults, routes, static API error codes, explicit response models, generated OpenAPI JSON and TypeScript types, the request-correlated typed no-feasible policy chain, strict v2 decision/path/receipt/candidate/binding/report ingress, and maintained client boundaries |
| Architecture and methodology | Good | Major boundaries and estimator assumptions are documented with versioned contracts; the v5 plan binds decision provenance and the two-entry portable snapshot digest, installed-host registry currency remains a separate admission check, workbench presentation no longer reconstructs policy from topology, the CUDA Phase 1 protocol remains frozen design authority, and the Phase 2 contract separates implemented source tooling and completed recovery publication from pending target-host evidence |
| Historical separation | Good | Reference intake, superseded v0.1 pages, all ten legacy-audit reports, twelve archived engineering reviews, the immutable drift audit, and the failed Phase 6 diagnostic display explicit status boundaries |
| Discoverability | Good | The central and archive indexes expose current journeys, desktop implementation guidance, legacy evidence, and historical engineering reviews |
| Freshness metadata | Good | Every governed Markdown page identifies status, review date, and a review trigger; the README review date is reconciled to 2026-08-06 |
| Automation | Good | Tests cover links, anchors, fences, navigation reachability, metadata, structured CLI parser parity, API routes and static errors, all 11 executable generated-operator-document rows, method overlap, stale contracts, model-policy provenance, all six snapshot finding codes, plan-ID snapshot binding, portable-integrity versus host-currency semantics, Phase 4 surfaces, Phase 5 request/candidate/report ingress, Phase 6 two-policy claims, mutation parity, exact-source evidence pointers, evidence-packet checksums and sanitization including the Phase 2B inventory, dispositions, independent review, and publication decision, exact bundle comparison, distinct validation/admission presentation, bundle-environment safety, the 2026-07-28 audit closeout, strict maintained React response normalization, and all four native HTTP response boundaries checked against OpenAPI required and closed fields |
| Release evidence | Partial | The August 5 refresh records two fresh current-contract MLX-LM repetitions at their exact acceptance source, the August 6 packet records one exact CUDA LoRA single-device `measured-run-pass`, and the August 10 packets establish the exact frozen five-slot CUDA LoRA repeatability anchor and a corrected Phase 6 matrix with Full exploratory promotion but no confirmatory stability; Phase 0 recovery is privately protected and the independently reviewed Phase 2B supplement establishes only recovery integrity, not target-runtime evidence; the earlier Phase 6 packet remains historical, remaining CUDA coverage and public notarized distribution remain open, and none is a safety, quality, broad-performance, production-readiness, or release-readiness claim |

## Freshness and classification

The [documentation inventory](documentation-inventory.md) classifies 127
governed tracked Markdown documents:

- 98 active;
- 2 deprecated;
- 27 archived.

The repository contains 128 tracked Markdown files in total. The pull-request
template is the only metadata-exempt workflow interface. The automated
`maintained_documentation()` set covers all other 127 files, including the
desktop build guide, all legacy reports, and the archived engineering reviews.

The deprecated pages point to current successors. Archived research, legacy,
diagnostic, and engineering-review records carry visible warnings before their
historical bodies. The active research source is labeled non-normative and
cannot be read as a support matrix.

Review metadata now covers current pages and historical entry points. Future
pages must follow [the maintenance policy](documentation-policy.md) so this does
not become a one-time cleanup.

## Drift findings

### High impact

The audit found that the API and configuration references named a 2 GiB user
reserve without explaining the effective 8 GiB Apple MPS/unified-memory floor
enforced at the CLI and HTTP planning boundaries. The merged reference copy
now distinguishes the supplied value from the effective floor and pins that
claim in documentation tests. No implementation default changes.

### Medium impact

The audit also found overbroad README support wording; setup-directory and
quality-gate inconsistencies; incomplete CUDA dependency, API inference-default,
first-plan artifact, and workbench command guidance; twelve completed reviews
still under `dev/active/`; incomplete governance for the desktop build guide and
nine legacy reports; aging `current-source` language for exact-source MLX
evidence; and stale review and merge metadata. PR #41 corrected those
surfaces, moved only the reviewed records, preserved every historical
body and immutable evidence packet, and expanded lifecycle regression checks.
DOC-028 records the resolution and its non-runtime boundary.

The [documentation debt log](documentation-debt.md) records owners, acceptance
criteria, and status for each finding.

## Organization findings

The current organization is broadly sound:

- `docs/getting-started/` and `docs/guides/` serve operators;
- `docs/product/` defines scope and claims;
- `docs/architecture/` explains component boundaries;
- `docs/methodology/` explains planning and evidence logic;
- `docs/reference/` records precise interfaces and states;
- `docs/operations/` holds release and machine procedures;
- `docs/contributing/` gives contract-specific implementation guidance;
- `docs/maintenance/` owns documentation governance and health;
- `docs/research/` records intake and reconciliation;
- `docs/archive/` separates historical navigation from current guidance;
- `docs/audits/aptus-legacy/` preserves one historical evidence bundle.
- `dev/archive/` preserves and indexes superseded engineering reviews; and
- `desktop/macos/README.md` is active build guidance linked from architecture.

The capitalized `Reference/` packet predates that structure and could be
confused with `docs/reference/`. This batch retains its paths to preserve source
provenance, but [Reference/README.md](../../Reference/README.md) now makes the
boundary explicit.

The two superseded v0.1 pages remain at their known paths as deprecation
signposts. [The archive index](../archive/index.md) groups them without changing
links or audit reproduction paths.

## Validation result

For the 2026-08-08 Phase 2A source-tooling documentation closeout:

- all 33 documentation tests passed in 0.363 seconds on the development host;
- the checks covered local links and anchors, primary-index reachability,
  metadata, the 123/122 repository and governed counts, the 93/2/27 lifecycle
  split, and the explicit source-only, no-Ubuntu-result, exact-host, and host-freeze
  boundaries; and
- this result validates documentation consistency only. It is not an Ubuntu,
  CUDA-runtime, GPU-resource, timing, thermal, model, or empirical result. The
  integrated source-gate result is recorded in the Phase 2A tooling contract
  after its stable run.

For the 2026-08-09 Phase 2B recovery-publication closeout:

- all 34 documentation tests passed, including exact packet inventory, both
  checksum layers, all 40 dispositions, distinct reviewer and finalizer
  bindings, publication eligibility, sanitization, navigation, and lifecycle
  counts;
- the same code and packet bytes passed all 889 Python tests, Ruff lint and
  formatting, all 130 React tests, TypeScript checking, and the tracked
  production web build; and
- these are development-host repository gates. They add no Ubuntu, CUDA
  runtime, GPU, performance, model, or empirical evidence.

The final Phase 2A repository closeout then passed all 888 Python tests in
45.777 seconds, all 130 React tests across 20 files, Ruff lint and formatting,
Python compilation, server and generated OpenAPI checks, maintained Swift
client contracts, version agreement, TypeScript checking, and the tracked Vite
build. The required macOS build also passed native tests, produced and verified
the ad-hoc-signed app, and created the app archive and DMG. These are
development-host source, packaging, and compatibility gates; they add no
Ubuntu, CUDA-runtime, model, resource, timing, thermal, or empirical result.

For this 2026-08-07 canonical campaign integration:

- all 32 documentation tests and all 554 Python tests passed; the full Python
  suite completed in 22.560 seconds on the development host;
- Ruff lint passed, all 54 checked Python files were already formatted, and
  Python compilation passed;
- all 130 React tests across 20 files passed, TypeScript checking passed, and
  the tracked Vite production build completed successfully;
- server OpenAPI, generated TypeScript OpenAPI, maintained Swift client,
  version-parity, and whitespace checks passed; and
- these are development-worktree gates for a documentation change, not Ubuntu
  target-host, CUDA runtime, performance, resource, or empirical campaign
  evidence. No native desktop build or CUDA run was required.

For the merged 2026-08-06 reconciliation in PR #41:

- the baseline was merged main `170fd0e`;
- immutable runtime evidence packet bodies remain unchanged;
- twelve completed review records moved from `dev/active/` to the same relative
  subpaths under `dev/archive/` and gain only leading lifecycle warnings;
- all ten legacy-audit reports carry archived metadata and remain one evidence
  bundle;
- the native desktop build guide became active governed documentation; and
- focused tests cover the 120/119 repository and governed counts, the
  90/2/27 lifecycle split, metadata, navigation, archive placement, and the
  corrected reserve and generated-MLX inventory claims;
- all 31 documentation tests and all 553 Python tests passed in an isolated
  overlay of merged `main`;
- all 130 React tests, TypeScript checking, the Vite production build, OpenAPI
  server and generated-client freshness, maintained Swift client parity,
  version parity, Ruff lint and formatting, Python compilation, and whitespace
  validation passed; and
- no native desktop build was required because the change affected Markdown
  and documentation regression coverage, not the host, bridge, packaged
  workbench, or desktop runtime.

For the merged compatibility-contract correction in PR #21:

- changed-file local links were checked against the repository;
- the repository documentation tests passed for links and anchors, balanced
  fences, navigation reachability, review metadata, CLI commands and options,
  API routes and static errors, method-registry overlap, stale v1 contracts, and
  sealed-bundle environment safety;
- historical content and evidence were preserved;
- the real MLX and desktop engineering records were indexed without committing
  review binaries;
- ignored local intake and disposable generated bundles were not indexed as
  current documentation;
- newly written governance text and status banners contain no em dash
  characters. Pre-existing archive prose remains unchanged;
- the immutable documentation-drift audit remains preserved at its original
  path, while the live docs now satisfy every recorded remediation requirement;
- semantic tests distinguish CUDA and MLX validation ownership, require the
  complete Qwen3 MoE evidence boundary, and pin the MLX model-data artifact
  contract;
- PR #19 merged as `60be63f`, and PR #20 merged as `fc923ac`;
- all 363 Python tests passed, including 15 documentation tests and the new
  response-boundary rejection cases;
- all 83 React tests passed, including exact match and mismatch copy, malformed
  compatibility ingestion, and direct presentation defense;
- Ruff formatting and lint, Python bytecode compilation, OpenAPI JSON and
  TypeScript generation parity, maintained-client contract checks, version
  verification, TypeScript checking, the production web build, and whitespace
  validation passed;
- the full native package gate passed 81 Swift tests, the packaged-app launch
  probe, code-signature verification, ZIP creation, and DMG creation and
  verification; and
- the generated TypeScript compatibility type is a three-variant union keyed by
  `status`, rather than one permissive object with contradictory combinations.

Merged PR #22 added closed execution vocabularies, explicit
compute-backend and adapter-profile identity, method-registry tuple validation,
adapter-method validation, and pilot-eligibility wording. The final local gate
passed all 363 Python tests, including 15 documentation tests, all 84 React
tests, and all 81 native tests. Ruff formatting and lint, Python bytecode
compilation, OpenAPI and maintained-client parity, version parity, TypeScript
checking, the production web build, packaged-app launch, strict ad-hoc signing,
ZIP integrity, DMG verification, and artifact checksums also passed. An
independent adversarial re-review found no remaining Phase 1 defect. These
checks do not claim target-host pilot evidence or public notarization.

The Phase 2 implementation candidate added immutable model-policy subject,
path, and decision types plus one host-side registry. Provider inspection and
candidate planning call the same evaluator. API model-family path validation
and the CLI response projector seal claims against that registry. Runtime
contracts are constructed from the method registry. Sparse identity markers,
contradictory facts, unregistered family paths, forged decisions,
catalog-target drift, and invalid runtime bindings now fail closed.

The final local gate passed all 386 Python tests, including 16 documentation
tests, all 84 React tests, and all 81 native tests. Ruff formatting and lint,
Python bytecode compilation, OpenAPI and maintained-client parity, version
parity, TypeScript checking, the production web build, packaged-app launch,
strict ad-hoc signing, ZIP creation, and DMG creation also passed. The checked
OpenAPI JSON and generated TypeScript hashes remain unchanged. Dense CUDA and
Qwen3 MoE plan and candidate identities match the merged Phase 1 baseline. The
final independent adversarial pass found no remaining code or schema blocker.
These checks do not claim target-host pilot evidence or public notarization.

The historical Phase 3 implementation added `aptus.training-plan.v4`,
`aptus.model-compatibility.v2`, `aptus.model-inspection-receipt.v1`, and
`aptus.model-policy-binding.v1`. That phase's guidance distinguished the
compatibility-subject digest from the broader observed-planning-facts digest,
provider-inspection from user-attested plans, and tamper evidence from
authentication. It also recorded strict replanning for pre-v4 and then-obsolete
v4 policy state while preserving historical Phase 2 and dated audit statements.
The full documentation suite passed 17 tests, including semantic Phase 3
coverage and the local link-and-anchor check. This was documentation and
contract evidence, not a passing real-model pilot.

The current Phase 4 contract uses `aptus.training-plan.v6` and
`aptus.bundle.v3`. The canonical `aptus.model-policy-snapshot.v1` is generated
deterministically from the host registry, cross-bound by digest in the plan and
manifest, and evaluated inside the bundle without importing Aptus. V4 and older
plans require replanning. The package-free portable validator checks its frozen
snapshot for contract validity, canonical encoding, digest and path integrity,
and decision parity; it cannot determine host policy currency. Validation under
an installed Aptus host additionally compares the bindings with the current
host registry and requires replanning when the host digest has changed. PRs #26
through #28 add typed malformed-snapshot findings, controlled scalar-manifest
rejection, package-free negative coverage, the complete legacy/API matrices,
and removal of the retired handwritten policy branch. PR #29 completed the
semantic reference set, and PR #30 made non-object roots, excessive nesting,
oversized integers, and nested malformed plan values controlled invalid-input
results across their covered boundaries. Installed-host validation covers plan,
manifest, trainer, and snapshot documents; package-free validation covers plan,
manifest, and snapshot documents. At that closeout, Phase 5 browser cleanup and
Phase 6 policy expansion remained separate future work.

The Phase 4 closeout source tree passed 505 Python tests, 91 web tests, and 81
native tests, plus Ruff, bytecode compilation, generated-contract parity, a
fresh installed-wheel smoke, and the complete signed-app/ZIP/DMG engineering
gate. All five GitHub checks passed for PR #30. These results establish the
source, contract, packaging, and documentation baseline only. At that closeout,
neither the July MLX-LM evidence nor the source tree supplied a current-head MLX
or CUDA target pilot, and neither established public notarized release
readiness. The later original August 5 packet closed only its exact Phase 6
MLX-LM scope and now remains the historical baseline for the exact-source
refresh described above.

The Phase 4 documentation synchronization passed 506 Python tests, including
23 documentation and 61 bundle-generation tests, plus 91 web tests and 81
native tests. Ruff, bytecode compilation, OpenAPI/client/version parity, the
production web build, a clean installed-wheel smoke, and the complete
ad-hoc-signed app/ZIP/DMG engineering gate also passed. No target-runtime pilot
was run: the generated-bundle change is operator README prose and does not alter
runtime programs, dependencies, planning, estimation, validation, or execution
semantics.

The Phase 5 maintained-guidance closeout passed all 24 focused documentation
tests. That set includes the new semantic check for strict decision, path,
receipt, candidate, binding, and report ingress; request correlation; the three
model-policy records; exact-path binding; plan/candidate/revision evidence
binding and action gating; structural recommendation equality; provider-declared
path-match provenance; typed authorization coherence without prose inference or
generic-failure mutation; topology and resident-memory separation; the typed
no-feasible policy chain; and the then-pending Phase 6 boundary. Ruff lint and
format checks passed for the touched Python test file. This focused result does
not claim the full Python, web, native, packaging, target-runtime, or
notarization gates.

The Phase 6 implementation candidate adds the second registry-driven Qwen2
configuration footprint while preserving the Phase 5 browser authority and
v1 snapshot shape. Focused contract coverage verifies two-policy snapshot
generation, claims-subset safety, host/portable exact and mutation parity,
sparse near-match refusal, exactly one seven-target planner binding, per-policy
receipt provenance, dense empty-override storage, and artifact-scoped canonical
evidence. These are source and contract results only. They do not replace the
runtime ladder. The subsequent original
[2026-08-05 acceptance packet](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
records two qualifying v5/v3 repetitions, resolves DOC-024 for its exact
artifact, source, host, runtime, dataset, and policy-snapshot scope. Other
matching artifacts remain gated, and CUDA, quality, and production-throughput
claims remain outside that result. It is now the unchanged historical baseline.
The [exact-source refresh](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
records two fresh v5/v3 repetitions at source
`719255153e3fc7e38e83b5ff826d587e5e58bf80` and resolves DOC-025 only for its
exact artifact and revision, source and tree, M5 Pro host, Python/MLX runtime,
dataset, plan, policy snapshot, and new bundle fingerprint. The refresh does not
broaden the original claim boundary.

The maintained-client parity closeout resolves DOC-026. React rejects malformed
required response records without weakening the existing Phase 5 policy ingress.
Swift rejects incomplete health and runtime configuration responses, requires
all six runtime-inventory fields with advertised-versus-measured path parity,
and distinguishes valid platform `unsupported` responses from malformed status
or payload shapes. Mutation tests keep the native checker aligned with OpenAPI
when a required field, closed value, decoder marker, or endpoint changes.

The exact CUDA packet resolves DOC-027. It records one qualifying LoRA
single-device execution without broadening the claim to repeatability or
another CUDA matrix cell. DOC-011 remains in progress for the remaining
target-host and public-distribution evidence.

The PR #21 implementation candidate separately passed the full Python, web, and
native test gates, generated-contract checks, packaged launch, app-signature
checks, and DMG verification. Two clean real MLX workflows and ten consecutive
desktop engineering builds and one exact CUDA LoRA workflow supply bounded
runtime and packaging evidence. These checks do not replace remaining CUDA
target-host coverage or public notarization.

## Recommended actions by impact

1. Preserve the frozen [Phase 1 CUDA campaign
   protocol](../reference/cuda-campaign-protocol.md) and its machine companion
   as the design authority; they do not authorize runtime execution.
2. Preserve the completed [Phase 2 tooling and recovery-publication
   record](../operations/cuda-campaign-phase2-tooling.md) and its dated
   [sanitized supplement](../operations/evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md);
   they are not Ubuntu or empirical evidence.
3. Preserve the completed Phase 3 selection controls, Phase 4 freeze, and
   independently reviewed Phase 5 repeatability packet.
4. Continue the [canonical RTX 3050 CUDA empirical evidence
   campaign](../operations/cuda-empirical-campaign.md) from Phase 6 through its
   remaining ordered measured phases. Treat every result as evidence for the exact
   RTX 3050 host and configuration, not Aptus's global, cloud, or multi-GPU
   ceiling.
5. Complete the remaining distributed CUDA target-host coverage and public
   desktop distribution evidence that the single-GPU campaign cannot supply.
6. Revisit the repository-Markdown delivery decision only when versioning,
   search, or a named site owner changes the cost-benefit analysis.

## Next health review

Refresh this report when any of these occurs:

- the v0.2 release evidence record is added;
- a plan, bundle, validation, execution, CLI, API, or method contract changes;
- documentation files move;
- a new backend or executable method becomes selectable;
- the scheduled review date arrives.

## Related documentation

- [Documentation maintenance policy](documentation-policy.md)
- [Documentation inventory](documentation-inventory.md)
- [Documentation debt log](documentation-debt.md)
- [Documentation index](../index.md)
- [CUDA campaign Phase 2A tooling contract](../operations/cuda-campaign-phase2-tooling.md)
- [RTX 3050 CUDA empirical evidence campaign](../operations/cuda-empirical-campaign.md)
- [CUDA campaign protocol](../reference/cuda-campaign-protocol.md)
- [Release gates](../operations/release-gates.md)
