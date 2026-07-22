# Documentation debt log

> **Documentation status:** Active governance
>
> **Applies to:** Open and recently resolved documentation work
>
> **Last reviewed:** 2026-07-22
>
> **Next scheduled review:** At every documentation pull request and before 2026-10-22

This log records documentation work that cannot be trusted to memory or scattered
TODO comments. Status reflects the repository at the review date. Update an item
when its evidence, owner, or resolution changes.

## Status vocabulary

| Status | Meaning |
|---|---|
| Open | Confirmed work remains |
| In progress | A bounded change is underway |
| Blocked | Completion requires external evidence or a product decision |
| Resolved | The named acceptance criteria passed |
| Accepted | The limitation is intentional and documented |

## Priority definitions

| Priority | Meaning |
|---|---|
| P0 | A current instruction can cause unsafe execution or data loss |
| P1 | A current contract is wrong, incomplete, or likely to mislead operators |
| P2 | Discoverability, maintenance, or contributor reliability is materially weak |
| P3 | Useful polish with low immediate operational risk |

## Tracked debt

### DOC-001: Add contextual navigation to current pages

- **Priority:** P2
- **Status:** Resolved
- **Resolution:** Every current non-legacy page under `docs/` now has at least
  one contextual outgoing link. The central index groups reader journeys,
  reference, architecture, operations, research, maintenance, and contributor
  material.
- **Owner:** Documentation maintainers

### DOC-002: Complete review metadata coverage

- **Priority:** P2
- **Status:** Resolved
- **Resolution:** Current pages now identify status, authority, scope or
  audience, last review, and a scheduled or event-driven review trigger.
  Historical entry points carry explicit deprecated or archived warnings.
- **Owner:** Documentation maintainers

### DOC-003: Enforce CLI reference parity

- **Priority:** P1
- **Status:** In progress
- **Evidence:** Installed help explains user-facing fact flags, commands,
  actions, choices, defaults, and hardware scope. Documentation tests now walk
  the live parser tree and require every command, subcommand, and long option to
  appear in the CLI reference.
- **Required result:** Extend parity checks to compare choices and default
  values as structured data instead of relying on reviewed prose.
- **Owner:** CLI and documentation maintainers

### DOC-004: Enforce API and error-reference parity

- **Priority:** P1
- **Status:** In progress
- **Evidence:** The manual API and error references now inventory emitted
  routes, filesystem errors, lifecycle errors, fallback errors, and validation
  channels. Documentation tests require every decorated route and statically
  emitted API error code to appear in those references.
- **Required result:** Add stable response models and descriptions to OpenAPI,
  then compare documented status codes and response fields with that contract.
- **Owner:** API and documentation maintainers

### DOC-005: Validate method-catalog overlap

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** The documentation suite parses the research catalog and
  checks its current method IDs, parameter scope, parameterization, base
  storage, supported distributions, export kinds, and non-selectable research
  entries against the typed runtime registry. Research-only fields remain
  documentation.
- **Owner:** Planner and documentation maintainers

### DOC-006: Provide a concrete private security-reporting route

- **Priority:** P1
- **Status:** Open
- **Evidence:** `SECURITY.md` directs reporters to GitHub private vulnerability
  reporting when the repository exposes it, with an existing private maintainer
  channel as fallback. It now publishes supported-version, response-target, and
  coordinated-disclosure rules. The GitHub feature still needs repository-level
  availability verification before the route can be called guaranteed.
- **Required result:** Add a maintained private reporting method, expected
  response window, supported-version table, and disclosure process.
- **Owner:** Repository owner

### DOC-007: Improve package and repository discovery metadata

- **Priority:** P2
- **Status:** Resolved
- **Resolution:** `pyproject.toml` now provides classifiers, search keywords,
  and project URLs for documentation, source, issues, and the changelog. The
  package build and installed-wheel smoke checks remain release gates.
- **Owner:** Packaging maintainers

### DOC-008: Add contributor workflow templates

- **Priority:** P2
- **Status:** Resolved
- **Resolution:** The repository now has a contract-aware pull-request template,
  structured bug and documentation issue forms, a private-security contact link,
  a contributor section, and a support policy. The templates request evidence,
  target context, documentation impact, and sensitive-data review without adding
  governance files that have no named purpose.
- **Owner:** Repository maintainers

### DOC-009: Decide whether to publish a documentation site

- **Priority:** P3
- **Status:** Accepted
- **Decision:** Repository Markdown is the v0.2 delivery surface. Its indexes,
  relative links, review metadata, and automated navigation checks provide the
  maintained path without adding a publishing dependency. Revisit this decision
  when versioned releases require parallel documentation, search becomes a
  measured reader need, or a maintainer owns a site deployment.
- **Owner:** Product and repository maintainers

### DOC-010: Document local state retention and cleanup

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** [State, storage, and retention](../operations/state-storage-retention.md)
  inventories persistent locations, mutability and sensitivity, active-job
  checks, attestation effects, retention classes, and disk-pressure response. It
  states that Aptus has no automated cleanup command.
- **Owner:** Execution and operations maintainers

### DOC-011: Publish versioned target-host release evidence

- **Priority:** P1
- **Status:** Blocked
- **Evidence:** [Release gates](../operations/release-gates.md) correctly state
  that no qualifying CUDA or MLX pilot and full-run release evidence has been
  recorded on the applicable target hardware.
- **Required result:** Add a dated, immutable release-evidence record that binds
  code revision, package versions, model and dataset identities, hardware,
  plans, validation reports, pilots, full runs, job-control checks, and known
  failures.
- **Blocker:** Access to approved CUDA or Apple Silicon target hosts and
  authorized model and dataset inputs
- **Owner:** Release maintainers

### DOC-012: Test generated operator documentation as a contract

- **Priority:** P1
- **Status:** In progress
- **Evidence:** Bundle `README.md`, `decision-report.md`, `runbook.md`, and
  generated script help come from large templates in `src/aptus/generation.py`.
  Generated guidance now explains the external-environment rule, evidence
  boundary, model-data behavior, ordered actions, and recovery boundary.
  Assertions cover these critical instructions, but not every executable method
  and placement.
- **Required result:** Generate representative bundles for all executable
  methods and placements, then test command order, evidence boundaries,
  platform notes, file names, and successor links.
- **Owner:** Compiler and documentation maintainers

## Resolved in the 2026-07-22 governance batch

### DOC-013: Separate raw Reference material from current authority

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** Added [Reference/README.md](../../Reference/README.md) and
  visible status warnings to all four retained source files. The packet now
  distinguishes one active non-normative research source from three archived
  intake files.
- **Verification:** Local links in changed files resolve.

### DOC-014: Add historical navigation without moving evidence

- **Priority:** P2
- **Status:** Resolved
- **Resolution:** Added [the archive index](../archive/index.md) and status
  warnings to the two superseded v0.1 signposts and the legacy-audit entry page.
  Historical content and reproduction paths remain unchanged.
- **Verification:** Local links in changed files resolve.

### DOC-015: Establish documentation governance artifacts

- **Priority:** P2
- **Status:** Resolved
- **Resolution:** Added policy, inventory, debt, and health documents with a
  common classification and review model.
- **Verification:** The documents cross-link and use relative local links.

## Related documentation

- [Documentation maintenance policy](documentation-policy.md)
- [Documentation inventory](documentation-inventory.md)
- [Documentation health report](documentation-health.md)
- [Release gates](../operations/release-gates.md)
