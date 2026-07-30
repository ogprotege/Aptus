# Documentation health report

> **Documentation status:** Active governance
>
> **Applies to:** Repository documentation through PR #22, merge `e7ce942`, plus
> the current Phase 2 and Phase 3 compatibility-policy follow-ups
>
> **Last reviewed:** 2026-07-29
>
> **Next scheduled review:** 2026-10-27, or after the next contract-changing pull request

## Overall assessment

Aptus has a substantive v0.2 documentation set. It explains evidence boundaries,
supported methods, planning assumptions, bundle structure, validation states,
execution controls, and known release blockers with unusual care. The main risk
is maintenance scale. The same contracts appear in prose, code, generated
artifacts, API shapes, CLI help, and workbench copy without enough automated
parity checks.

Overall documentation health is **strong with named maintenance gaps**. The
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
eligibility for a reviewed pilot path. The Phase 2 follow-up introduced
one host-side policy registry for provider inspection, sparse candidate
admission, and API execution-path validation. That historical phase
intentionally preserved the v1 API, v3 plan identity, portable contract, and
evidence boundary. The current Phase 3 follow-up migrates plans to v4, persists
versioned decisions, receipts, and exact-path bindings, and keeps the API,
facts, runtime-contract, and bundle versions unchanged. PR #15 preserved and
indexed the historical audit without making it current authority.
The exact Qwen3 30B MLX-LM attempt remains safe-refusal evidence: it
stopped before model loading. It is not a passing pilot or training result. The
release itself remains blocked until qualifying CUDA target-host evidence and a
Developer ID signed and notarized public desktop distribution exist for the
capabilities being claimed. Real MLX-LM acceptance and a local 10-build desktop
engineering gate have passed. The repository checks its principal navigation
and executable-reference surfaces, but it does not yet derive every default,
status, and response field from one source.

## Scorecard

| Area | Result | Evidence |
|---|---|---|
| Product boundaries | Good | Current capabilities, claim language, roadmap, and release gates distinguish implemented, conditional, unsupported, and future work |
| Evidence language | Good | Planning estimates, measured checks, structural export verification, and task quality are kept separate |
| User workflow coverage | Good | Installation, quickstart, facts, comparison, compilation, validation, execution, recovery, and troubleshooting are present |
| API and CLI reference | Good | Automated checks cover commands, options, routes, static API error codes, explicit response models, generated OpenAPI JSON and TypeScript types, a status-discriminated compatibility contract with closed execution IDs, and maintained client boundaries; structured CLI default and choice parity remains incomplete |
| Architecture and methodology | Good | Major boundaries and estimator assumptions are documented with versioned contracts; host model-policy decisions have one named authority, and v4 plans bind decision, receipt source, and exact matching path |
| Historical separation | Good | Reference intake, superseded v0.1 pages, the legacy audit, and the indexed immutable drift audit display explicit status boundaries |
| Discoverability | Good | The central index exposes reader journeys, and every current non-legacy page has contextual outgoing navigation |
| Freshness metadata | Good | Current pages and historical entry points identify status, authority, review date, and a review trigger |
| Automation | Good with gaps | Tests cover links, anchors, fences, navigation reachability, metadata, CLI surface, API routes and static errors, method overlap, stale contracts, Phase 3 decision, receipt, and binding documentation, compatibility-schema semantics, fail-closed browser normalization, bundle-environment safety, and the 2026-07-28 audit-closeout semantics; generated-doc and structured default parity remain partial |
| Release evidence | Partial | Two clean MLX-LM workflows reached `measured-run-pass`, and 10 of 10 local desktop engineering builds passed at their tested commit; CUDA target-host and public notarized distribution evidence remain open |

## Freshness and classification

The [documentation inventory](documentation-inventory.md) classifies 100
governed tracked Markdown documents:

- 84 active;
- 2 deprecated;
- 14 archived.

The automated `maintained_documentation()` set contains 91 files because it
retains the legacy-audit README but excludes nine subordinate historical audit
pages. That automation scope is intentionally narrower than the governed
inventory.

The deprecated pages point to current successors. Archived research and legacy
records have visible warnings at their entry points. The active research source
is labeled non-normative and cannot be read as a support matrix.

Review metadata now covers current pages and historical entry points. Future
pages must follow [the maintenance policy](documentation-policy.md) so this does
not become a one-time cleanup.

## Drift findings

### High impact

1. CLI choices and defaults still appear in both `src/aptus/cli.py` and prose.
   Tests cover the parser's commands, subcommands, and long options, but not yet
   every structured value.
2. Generated OpenAPI JSON and TypeScript schema and path types have stale-file
   checks. Model compatibility has semantic schema tests for its discriminated
   variants, known execution IDs, compute backend, adapter profile, and React
   runtime normalization. Other React normalization code and Swift response
   decoders remain maintained client boundaries that require contract tests.
3. Generated bundle guidance is operationally important but embedded in large
   source templates. Representative output needs stronger contract testing.

### Medium impact

1. GitHub private vulnerability reporting is verified disabled. The security
   policy therefore lacks a guaranteed private reporting route.
2. Production npm dependencies have no known advisory, but four high-severity
   transitive advisories remain in the OpenAPI generator development chain.

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

The capitalized `Reference/` packet predates that structure and could be
confused with `docs/reference/`. This batch retains its paths to preserve source
provenance, but [Reference/README.md](../../Reference/README.md) now makes the
boundary explicit.

The two superseded v0.1 pages remain at their known paths as deprecation
signposts. [The archive index](../archive/index.md) groups them without changing
links or audit reproduction paths.

## Validation result

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

The current Phase 3 candidate adds `aptus.training-plan.v4`,
`aptus.model-compatibility.v2`, `aptus.model-inspection-receipt.v1`, and
`aptus.model-policy-binding.v1`. Current guidance now distinguishes the
compatibility-subject digest from the broader observed-planning-facts digest,
provider-inspection from user-attested plans, and tamper evidence from
authentication. It also records strict replanning for pre-v4 and obsolete-policy
plans while preserving historical Phase 2 and dated audit statements. The full
documentation suite passes 17 tests, including semantic Phase 3 coverage and
the local link-and-anchor check. This is documentation and contract evidence,
not a passing real-model pilot.

The PR #21 implementation candidate separately passed the full Python, web, and
native test gates, generated-contract checks, packaged launch, app-signature
checks, and DMG verification. Two clean real MLX workflows and ten consecutive
desktop engineering builds supply bounded runtime and packaging evidence. These
checks do not replace CUDA target-host acceptance or public notarization.

## Recommended actions by impact

1. Close the remaining maintained client, CLI-default, and generated-bundle
   parity gaps.
2. Publish a concrete private security-reporting route.
3. Resolve the OpenAPI generator development advisories.
4. Complete qualifying CUDA target-host and public desktop distribution evidence.
5. Revisit the repository-Markdown delivery decision only when versioning,
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
- [Release gates](../operations/release-gates.md)
