# Documentation maintenance policy

> **Documentation status:** Active governance
>
> **Applies to:** Aptus v0.2 and later documentation changes
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2026-10-22, or before the next release candidate

This policy defines how Aptus documentation is classified, changed, reviewed,
and tested. Its purpose is practical: a reader should be able to tell what Aptus
does now, what remains conditional, what is future work, and what exists only as
historical evidence.

## Documentation principles

1. Code, tests, and measured evidence establish behavior. Prose explains that
   behavior but cannot create a capability.
2. Current guidance must identify its product version and evidence boundary.
3. Research presence does not imply implementation, compatibility, or support.
4. Historical records remain available but must not resemble current guidance.
5. Estimates, pilots, structural checks, and task-quality evaluation are
   different claims. Documentation must keep them separate.
6. Every behavior change updates the affected user, API, CLI, schema, bundle,
   security, and operations documentation in the same pull request.

## Lifecycle classification

Every maintained document has one lifecycle classification.

| Lifecycle | Meaning | Reader action |
|---|---|---|
| Active | Current and safe to use within its stated scope | Follow it, while respecting named release and evidence gates |
| Deprecated | Replaced or discouraged, but retained temporarily as a signpost | Use the named successor |
| Archived | Historical evidence or intake material only | Do not use it as current product guidance |

`Research-only` is an authority qualifier, not a fourth lifecycle. An active
research index can describe current research while remaining non-normative.

## Required metadata

New maintained Markdown documents should include a visible metadata block near
the title. A compact one-line block may use `Status` for `Documentation status`
and `Review by` for `Next scheduled review`. A visible metadata table may use
`Status` and `Next review`. These forms carry the same meaning and are accepted
by the documentation checks.

| Field | Required content |
|---|---|
| Documentation status | Active, Deprecated, or Archived, plus a useful scope label |
| Applies to | Product version, subsystem, evidence snapshot, or source packet |
| Last reviewed | ISO date in `YYYY-MM-DD` form |
| Next scheduled review | ISO date or a precise event trigger |
| Authority note | Required when the document is research-only, generated, or non-normative |

Historical evidence should retain its original date and content. Add the status
block to every human-readable archived record, not only its directory entry
page, without rewriting the historical body.

Repository-host workflow templates and generated bundle documents are
operational interfaces, not reader pages. They are exempt from the banner so
that metadata does not appear inside a submitted pull request or portable
runbook. The inventory must still name their authority and validation path.

## Review cadence

| Document type | Maximum review window | Event triggers |
|---|---:|---|
| Current capabilities, schemas, API, CLI, and validation contracts | 90 days | Any contract, option, endpoint, state, or default change |
| Security and operations | 90 days | Any trust-boundary, dependency, process, storage, or release-gate change |
| Getting-started and task guides | 180 days | Any command, workflow, prerequisite, or supported-platform change |
| Architecture and methodology | 180 days | Any component boundary, estimator, planner, compiler, or evidence change |
| Product vision and roadmap | 180 days | Release planning or scope change |
| Active research indexes | 90 days | Source cutoff, library support, or runtime-admission change |
| Archived material | 12 months | Broken link, misleading placement, provenance correction, or legal need |

The event trigger takes precedence over the calendar date.

## Authority map

Use the narrowest executable authority when checking documentation.

| Topic | Primary implementation authority | Primary documentation |
|---|---|---|
| Fact, plan, and candidate contracts | `src/aptus/domain.py`, `src/aptus/plan_contract.py` | [Plan schema](../reference/plan-schema.md), [facts and provenance](../methodology/facts-and-provenance.md) |
| Method lifecycle and selectability | `src/aptus/methods/registry.py` | [Method taxonomy](../methodology/method-taxonomy.md), [capability matrix](../reference/capability-matrix.md) |
| Candidate enumeration and ranking | `src/aptus/planning.py` | [Candidate enumeration](../methodology/candidate-enumeration.md), [ranking and uncertainty](../methodology/ranking-uncertainty.md) |
| Bundle contents and generated prose | `src/aptus/generation.py` | [Artifact compiler](../architecture/artifact-compiler.md), [bundle manifest](../reference/bundle-manifest.md) |
| Validation and evidence states | `src/aptus/validation.py` | [Validation states](../reference/validation-states.md), [preflight and calibration](../methodology/preflight-calibration.md) |
| Jobs, leases, cancellation, and completion | `src/aptus/execution.py`, `src/aptus/runtime_lease.py` | [Execution orchestrator](../architecture/execution-orchestrator.md), [run states](../reference/run-states.md) |
| CLI | `src/aptus/cli.py` | [CLI reference](../reference/cli.md) |
| API | `src/aptus/api.py` | [API reference](../reference/api.md), [error codes](../reference/error-codes.md) |
| Workbench copy and behavior | `web/src/` | [UI and UX contract](../product/ui-ux.md) |

If code and prose disagree, treat the documentation as drift. Do not silently
change code to preserve an unsupported prose claim.

## Change workflow

Documentation changes follow this sequence:

1. Inventory the affected files and their incoming and outgoing links.
2. Classify each file as active, deprecated, or archived.
3. Compare claims against implementation, tests, generated artifacts, and
   measured evidence.
4. Record drift and missing coverage in the
   [documentation debt log](documentation-debt.md).
5. Propose moves before executing them. Preserve redirects or successor links
   when a known path changes.
6. Update related documents together and add contextual cross-links.
7. Update review metadata.
8. Run documentation and product checks appropriate to the changed contract.
9. Refresh the [documentation health report](documentation-health.md).

## Link and navigation rules

- Use relative repository links for local documentation.
- Link to primary sources for research claims.
- Link each task guide to the relevant reference and troubleshooting pages.
- Link each reference page back to at least one explanatory guide or
  architecture page.
- Keep [the documentation index](../index.md) as the main current-documents hub.
- Keep historical navigation under [the archive index](../archive/index.md).
- Keep completed implementation and code reviews under the indexed
  `dev/archive/` tree; `dev/active/` is only for explicitly open work.
- Keep research intake under [the research index](../research/index.md).
- Do not link ignored local intake, private data, caches, model weights, or
  disposable audit sandboxes.

## Generated documentation

Generated bundle files such as `README.md`, `decision-report.md`, and
`runbook.md` are part of the product contract. Their source templates live in
`src/aptus/generation.py`.

- Change the template, not one generated output.
- Test critical commands and evidence language.
- Bind generated claims to plan and compiler identities.
- Keep generated instructions usable without access to the source repository.
- Never describe direct dependency pins as a complete transitive lock.

The packaged web application under `src/aptus/_web/` is also generated. Edit
the source under `web/`, rebuild, and verify the packaged assets.

## Historical preservation

Archived material can contain incorrect, incomplete, or unsafe claims. Its
value is provenance, not operational authority.

- Add a warning before the historical content.
- Preserve the historical body unless correcting provenance or a security
  problem.
- Point readers to the current successor.
- Keep machine-readable audit snapshots with their matching human reports.
- Record a move before changing a path used by audit reproduction tooling.

The native desktop build guide at `desktop/macos/README.md` is maintained
documentation, not an implementation-working exemption. It must carry active
metadata, remain reachable from the architecture tree, and be reviewed with
every desktop packaging or deployment-target change.

## Documentation validation

At minimum, documentation changes must pass:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest tests.aptus.test_documentation -v
git diff --check
```

Contract changes also require the full repository checks in
[CONTRIBUTING.md](../../CONTRIBUTING.md). Link checks prove that destinations
exist. They do not prove that the destination supports the claim, so reviewers
must still compare prose with code and tests.

## Ownership and escalation

Repository maintainers own the documentation system. A pull request that
changes a product contract must name the documentation surfaces reviewed. If a
claim cannot be verified, mark it unknown, conditional, future, deprecated, or
archived. Record the unresolved work in the debt log instead of presenting an
assumption as fact.

## Related documentation

- [Documentation inventory](documentation-inventory.md)
- [Documentation debt log](documentation-debt.md)
- [Documentation health report](documentation-health.md)
- [Release gates](../operations/release-gates.md)
- [Contributing](../../CONTRIBUTING.md)
