# Documentation health report

> **Documentation status:** Active governance
>
> **Applies to:** Repository documentation through PR #18, merge `eec90f8`
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
and E. The 2026-07-29 corrective follow-up closes those locations and adds
semantic regression checks without rewriting the immutable audit record. The
root-cause K regression now covers the runtime match and mismatch branches. PR
#15 preserved and indexed the historical audit without making it current
authority. The exact Qwen3 30B MLX-LM attempt remains safe-refusal evidence: it
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
| API and CLI reference | Good | Automated checks cover commands, options, routes, static API error codes, explicit response models, generated OpenAPI JSON and TypeScript types, and maintained client boundaries; structured CLI default and choice parity remains incomplete |
| Architecture and methodology | Good | Major boundaries and estimator assumptions are documented with versioned contracts |
| Historical separation | Good | Reference intake, superseded v0.1 pages, the legacy audit, and the indexed immutable drift audit display explicit status boundaries |
| Discoverability | Good | The central index exposes reader journeys, and every current non-legacy page has contextual outgoing navigation |
| Freshness metadata | Good | Current pages and historical entry points identify status, authority, review date, and a review trigger |
| Automation | Good with gaps | Tests cover links, anchors, fences, navigation reachability, metadata, CLI surface, API routes and static errors, method overlap, stale contracts, bundle-environment safety, and the 2026-07-28 audit-closeout semantics; generated-doc and structured default parity remain partial |
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
2. Generated OpenAPI JSON and TypeScript schema and path types now have stale-file
   checks. React request and runtime-normalization code plus Swift response
   decoders remain maintained client boundaries that require contract tests.
3. Generated bundle guidance is operationally important but embedded in large
   source templates. Representative output needs stronger contract testing.

### Medium impact

1. GitHub private vulnerability reporting is verified disabled. The security
   policy therefore lacks a guaranteed private reporting route.
2. Production npm dependencies have no known advisory, but four high-severity
   transitive advisories remain in the OpenAPI generator development chain.
3. `compatibility` evidence still has one presentation owner,
   `ExpertTopologyRail`. Root cause K's mismatch regression asserts runtime,
   method, distribution, and reason. The matching branch is rendered and
   accessibility-checked, but does not yet assert the same four-field parity.

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

For the 2026-07-29 product-documentation revision:

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
  contract; and
- PR #18 merged as `eec90f8` after all five checks passed. The targeted React
  regression and documentation link, metadata, and reachability tests also
  pass in the reviewed correction set.

The implementation candidate separately passed the full Python, web, and native
test gates, generated-contract checks, packaged launch, app-signature checks,
and DMG verification. Two clean real MLX workflows and ten consecutive desktop
engineering builds supply bounded runtime and packaging evidence. These checks
do not replace CUDA target-host acceptance or public notarization.

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
