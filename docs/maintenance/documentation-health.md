# Documentation health report

> **Documentation status:** Active governance
>
> **Applies to:** Repository snapshot reviewed on 2026-07-27
>
> **Last reviewed:** 2026-07-27
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
release itself remains blocked until qualifying CUDA target-host and desktop
distribution evidence exists for the capabilities being claimed. The repository now checks
its principal navigation and executable-reference surfaces, but it does not yet
derive every default, status, and response field from one source.

## Scorecard

| Area | Result | Evidence |
|---|---|---|
| Product boundaries | Good | Current capabilities, claim language, roadmap, and release gates distinguish implemented, conditional, unsupported, and future work |
| Evidence language | Good | Planning estimates, measured checks, structural export verification, and task quality are kept separate |
| User workflow coverage | Good | Installation, quickstart, facts, comparison, compilation, validation, execution, recovery, and troubleshooting are present |
| API and CLI reference | Good | Automated checks cover commands, options, routes, static API error codes, explicit response models, and the generated `aptus.api.v1` OpenAPI artifact; structured CLI default and choice parity remains incomplete |
| Architecture and methodology | Good | Major boundaries and estimator assumptions are documented with versioned contracts |
| Historical separation | Good after this batch | Reference intake, superseded v0.1 pages, and the legacy audit now display explicit status boundaries |
| Discoverability | Good | The central index exposes reader journeys, and every current non-legacy page has contextual outgoing navigation |
| Freshness metadata | Good | Current pages and historical entry points identify status, authority, review date, and a review trigger |
| Automation | Good with gaps | Tests cover links, anchors, fences, navigation reachability, metadata, CLI surface, API routes and static errors, method overlap, stale contracts, and bundle-environment safety; generated-doc and structured default parity remain partial |
| Release evidence | Partial | Two clean MLX-LM workflows reached `measured-run-pass`; CUDA target-host and public desktop distribution evidence remain open |

## Freshness and classification

The [documentation inventory](documentation-inventory.md) classifies 96 tracked
Markdown documents:

- 81 active;
- 2 deprecated;
- 13 archived.

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
2. React TypeScript and Swift native response types remain manually maintained
   client boundaries. Tests cover their current contracts, but generated SDKs
   are not implemented.
3. Generated bundle guidance is operationally important but embedded in large
   source templates. Representative output needs stronger contract testing.

### Medium impact

1. The security policy lacks a guaranteed private reporting route and response
   window.

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

For the 2026-07-22 governance batch:

- changed-file local links were checked against the repository;
- the repository documentation tests passed for links and anchors, balanced
  fences, navigation reachability, review metadata, CLI commands and options,
  API routes and static errors, method-registry overlap, stale v1 contracts, and
  sealed-bundle environment safety;
- historical content was preserved except for the requested status banners;
- no historical files were moved;
- ignored local intake and disposable generated bundles were not indexed as
  current documentation;
- newly written governance text and status banners contain no em dash
  characters. Pre-existing archive prose remains unchanged.

The same change passed the full Python suite, Python static and compile checks,
the web test, typecheck, and production-build gate, an executable planning-only
tutorial, and an isolated wheel smoke test. These checks establish repository
consistency. They do not replace the required target-host CUDA or MLX evidence.

## Recommended actions by impact

1. Close the remaining client-type, CLI-default, and generated-bundle parity gaps.
2. Publish a concrete private security-reporting route.
3. Complete qualifying CUDA target-host and public desktop distribution evidence.
4. Revisit the repository-Markdown delivery decision only when versioning,
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
