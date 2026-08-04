# Documentation debt log

> **Documentation status:** Active governance
>
> **Applies to:** Open and recently resolved documentation work
>
> **Last reviewed:** 2026-08-04
>
> **Next scheduled review:** At every documentation pull request and before 2026-11-01

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
- **Status:** Resolved
- **Resolution:** Every success route has an explicit Pydantic response model,
  the API reports contract identity `aptus.api.v1`, and
  `docs/reference/openapi.v1.json` is generated and checked byte-for-byte against
  the running application schema. `web/src/generated/openapi.ts` is also
  generated and checked from that artifact. React consumes its schema and path
  types through maintained request, normalization, domain, and presentation
  layers. Swift decoders remain maintained source and are checked against the
  covered OpenAPI boundary.
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
  reporting when available, with an existing private maintainer channel as
  fallback. It publishes supported-version, response-target, and
  coordinated-disclosure rules. Repository inspection on 2026-07-27 confirmed
  that GitHub private vulnerability reporting is disabled.
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
- **Status:** In progress
- **Evidence:** The
  [2026-07-27 MLX-LM acceptance](../operations/evidence/2026-07-27-mlx-lm-acceptance/README.md)
  binds a clean commit, exact runtime, public model revision, synthetic dataset,
  plans, bundles, two measured preflights, pilots, full runs, reloads, exports,
  timings, memory, and hashes. Both clean repetitions reached
  `measured-run-pass`. The
  [desktop engineering acceptance](../operations/evidence/2026-07-27-desktop-release/README.md)
  binds a separate 10-of-10 local build result, package hashes, timing, test
  counts, signing state, and limitations to its exact tested commit. The
  [2026-07-28 Qwen3 MoE admission record](../operations/evidence/2026-07-28-qwen3-moe-admission/README.md)
  adds exact fail-closed evidence for the 30B checkpoint. It stopped before
  model loading and is not passing pilot, training, reload, export,
  performance, or quality evidence. The July MLX-LM and desktop records predate
  the current Phase 5 source head; no current-head CUDA or MLX target-runtime
  pilot was collected for the Phase 5 closeout.
- **Required result:** Renew qualifying current-head MLX-LM target-runtime
  evidence, add equivalent evidence for every claimed CUDA method and placement,
  and add Developer ID signed and notarized desktop evidence for the exact
  public release commit.
- **Blocker:** Access to approved CUDA and MLX target hosts and public
  notarization credentials for a public Mac artifact
- **Owner:** Release maintainers

### DOC-012: Test generated operator documentation as a contract

- **Priority:** P1
- **Status:** In progress
- **Evidence:** Portable CUDA and MLX program sources now live as packaged
  resources with emitted-byte and manifest parity tests across source, wheel,
  and frozen layouts. Bundle `README.md`, `decision-report.md`, and `runbook.md`
  guidance still comes from compiler templates. Representative CUDA and MLX
  README assertions cover the v5 plan, v3 bundle, frozen-snapshot integrity,
  and installed-host currency boundary, but not every executable method and
  placement.
- **Required result:** Generate representative bundles for all executable
  methods and placements, then test command order, evidence boundaries,
  platform notes, file names, and successor links.
- **Owner:** Compiler and documentation maintainers

### DOC-016: Resolve OpenAPI generator development advisories

- **Priority:** P1
- **Status:** Open
- **Evidence:** Rechecked 2026-07-29. `npm audit --omit=dev` reports zero
  production advisories. The full audit reports four high-severity transitive
  advisories through
  `openapi-typescript` and `@redocly/openapi-core`: `@redocly/openapi-core`,
  `js-yaml`, `minimatch`, and `brace-expansion`. The generator consumes the
  trusted checked-in OpenAPI document during development and release builds.
- **Required result:** Upgrade or replace the generator dependency chain without
  changing the generated contract unexpectedly, then record a clean full audit.
- **Owner:** Web and release maintainers

### DOC-017: Complete the 2026-07-28 documentation-drift remediation

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** The six partially remediated locations left by PR #14 now
  state the MLX model-data evidence boundary, exact Qwen3 MoE navigation and
  planner filters, and runtime-specific CUDA versus MLX validation ownership.
  Semantic documentation tests pin those claims to the current generated
  runtime sources.
- **Evidence:** The immutable
  [2026-07-28 documentation drift audit](../operations/evidence/2026-07-29-documentation-drift-audit/README.md)
  remains unchanged. `test_documentation_drift_audit_closeout_invariants`,
  `test_qwen3_documentation_slice_is_complete`, and
  `test_bundle_manifest_distinguishes_runtime_validation_ownership` guard the
  corrective follow-up.
- **Owner:** Runtime and documentation maintainers

### DOC-018: Preserve compatibility-evidence presentation parity

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** At the Phase 1 boundary, `ExpertTopologyRail` was the single
  presentation owner for compatibility evidence. Post-merge review found that
  PR #20's substring check
  for `single` could pass against `single-device` inside the reason, even if the
  placement clause disappeared. The corrected tests assert each complete support
  or mismatch sentence and assert the reason separately. The API now uses three
  closed, status-discriminated variants. The producer, API response boundary,
  browser ingestion path, and presentation component all fail closed when a
  conditional result is incomplete or contradictory. Phase 5 later separated
  topology from current policy presentation; DOC-023 records that boundary.
- **Historical verification:**
  `test_model_compatibility_contract_rejects_contradictory_evidence`,
  `test_model_inspection_response_rejects_malformed_compatibility`,
  `test_model_compatibility_reference_matches_discriminated_contract`, and
  the Phase 1 `ExpertTopologyRail` tests pinned the corrected contract. Current
  policy-presentation verification is listed under DOC-023.
- **Owner:** Workbench maintainers

### DOC-019: Close compatibility vocabulary and inspection claim boundaries

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** At the Phase 1 boundary, conditional model compatibility used
  known runtime, compute-backend, method, distribution, and adapter-profile IDs.
  It carries an
  explicit `compute_backend`, identifies the reviewed attention target policy as
  `attention-qkvo.v1`, and validates each execution tuple against the typed
  method registry. A profile cannot be paired with full fine-tuning. Unknown IDs
  fail closed at the Python and browser boundaries. Known but unregistered
  runtime, backend, method, and distribution tuples fail at the producer and API
  response boundary. Inspection presentation said that an exact match was
  eligible for the reviewed pilot path and did not claim that the runtime had
  passed validation. Phase 5 now presents the decoded v2 decision and selected
  candidate path separately; DOC-023 records the current surface.
- **Historical verification:**
  `test_model_compatibility_contract_rejects_contradictory_evidence`,
  `test_model_compatibility_reference_matches_discriminated_contract`,
  the Phase 1 `modelCompatibility` tests, and the Phase 1 `ExpertTopologyRail`
  tests pinned the vocabulary, fail-closed normalization, backend binding, and
  complete eligibility copy. The Phase 5 importer audit found no production
  consumer of that flattened browser projection, so its normalizer and test were
  removed. Current presentation consumes the receipt's v2 decision and is
  verified under DOC-023.
- **Owner:** API, planner-registry, workbench, and documentation maintainers

### DOC-020: Establish one host model-compatibility policy authority

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** Immutable subject, path, and decision types now feed one
  host-side model compatibility registry. Provider inspection and sparse
  candidate admission call the same evaluator. The API response model delegates
  model-family path coherence to the same registry. The model policy selects
  method, runtime, backend, distribution, adapter profile, and target modules,
  while the method registry remains authoritative for compiler, estimator,
  export, and evidence-requirement identities. Sparse near-matches and sparse
  identity markers with missing topology cannot fall through as recognized
  dense families.
- **Boundary:** Phase 2 intentionally preserves `aptus.api.v1`,
  `aptus.training-plan.v3`, candidate and plan identities, the handwritten
  portable contract, and browser reconstruction. Versioned receipts and plan
  bindings belong to Phase 3. Portable snapshots belong to Phase 4. Browser
  simplification belongs to Phase 5.
- **Verification:**
  `test_exact_qwen_policy_emits_one_registry_bound_path`,
  `test_qwen_policy_mutations_fail_at_the_first_predicate`,
  `test_sparse_identity_markers_block_when_topology_is_missing`,
  `test_conditional_response_rejects_unregistered_model_policy_claims`,
  `test_qwen_v3_plan_and_candidate_identities_do_not_change`,
  `test_host_policy_import_order_has_no_cycle`, and
  `test_model_compatibility_policy_has_one_host_authority` protect the shared
  authority, method-registry binding, migration boundary, and dependency
  direction.
- **Owner:** Domain, inspection, planning, API, and documentation maintainers

### DOC-021: Bind model-policy provenance into persisted plans

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** `aptus.training-plan.v4` persists one
  `aptus.model-compatibility.v2` decision and an explicit
  `provider-inspection` or `user-attested` source. Successful provider
  inspection emits `aptus.model-inspection-receipt.v1` with separate
  compatibility-subject and observed-planning-facts digests. Every candidate
  links to the decision. Only a candidate that exactly matches a registered
  path carries `aptus.model-policy-binding.v1`. The exact Qwen3 MoE row has a
  stable policy ID, semantic policy version, and path ID. Parameter count and
  training permission remain outside the receipt.
- **Failure boundary:** A present malformed, stale, mismatched, or modified
  receipt is rejected instead of downgraded to user-attested. V3, v2,
  schema-less, and stale-policy v4 plans return `replan_required` without
  rewriting saved state. Receipt and identity hashes are tamper-evident, not
  authenticated signatures.
- **Boundary:** Phase 3 intentionally preserves `aptus.api.v1`,
  `aptus.facts.v3`, `aptus.runtime-contract.v1`, and `aptus.bundle.v2`. A
  portable policy snapshot and generic evaluator belong to Phase 4. Removal of
  browser-side policy reconstruction belongs to Phase 5.
- **Verification:**
  `test_provider_receipt_is_recomputed_from_every_observed_plan_fact`,
  `test_registered_path_requires_binding_and_other_paths_forbid_it`,
  `test_stale_registered_policy_has_a_dedicated_replan_error`,
  `test_tampered_provider_receipt_is_rejected_instead_of_downgraded`, and
  `test_phase3_policy_provenance_docs_match_persisted_contracts` protect the
  receipt, decision, binding, migration, and documentation boundaries.
- **Owner:** Domain, inspection, planning, API, client, and documentation maintainers

### DOC-022: Make the model-policy contract portable

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** `aptus.training-plan.v5` binds the SHA-256 of deterministic
  canonical `aptus.model-policy-snapshot.v1` bytes. `aptus.bundle.v3` contains
  those bytes, repeats the digest, manifests the snapshot file, and includes a
  generic evaluator that runs without an installed Aptus package. A package-free
  bundle evaluates its frozen snapshot for integrity and decision parity. An
  installed Aptus host uses the current host registry for policy currency.
- **Failure boundary:** V4 and older plans return `replan_required`. Portable
  validation rejects a missing, malformed, noncanonical, path-invalid, or
  digest-inconsistent snapshot, but it cannot determine host policy currency.
  Installed-host validation additionally compares the snapshot, plan, and
  manifest bindings with the current host digest; a different host binding is
  stale and requires replanning.
- **Boundary:** At the Phase 4 closeout, browser policy reconstruction remained
  assigned to Phase 5. DOC-023 records its completion. A second reviewed policy
  remains pending in Phase 6.
- **Verification:** Deterministic double generation, host-versus-portable
  decision parity, package-free validation, host-registry currency, exact digest
  binding, scalar plan and manifest rejection, controlled excessive-nesting and
  oversized-integer failures, stale admission and completion-promotion denial,
  and typed snapshot mutation tests protect the contract.
- **Owner:** Policy, planning, compiler, validation, and documentation maintainers

### DOC-023: Remove browser-side model-policy reconstruction

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** Phase 5 makes the server v2 policy chain authoritative at the
  maintained client boundary. Exact runtime decoders validate the decision,
  nested paths, optional inspection receipt, every candidate's complete tuple
  and explicit nullable binding, and validation-report identity before UI
  hydration. The response's required model subject, expected policy source, and
  receipt identity must correlate with the submitted request. Facts and Compare
  show model-policy match, selected candidate path, and evidence readiness as
  three separate records. The receipt decision is the one inspection-time
  browser policy source; the legacy flattened compatibility normalizer has no
  production consumer and is removed. Provider path-matched receipts require
  provider-declared provenance, not inferred-only observations. A successful
  recommendation must structurally equal its complete listed candidate record.
- **Presentation boundary:** Exact path equality requires a non-null candidate
  binding. Truly unbound and rejected rows receive no browser-invented policy
  ladder or impossible validation action. A report applies only when its plan
  ID, candidate ID, and immutable model revision match the current selection;
  the same tuple gates stage completion and validation or run actions. Required
  validation is incomplete or complete independently from the optional typed
  `authorization_status` values `current`, `deferred`, and `blocked`. Current
  pairs with true and no error; deferred or blocked pairs with false and a
  non-empty diagnostic; a tuple with no non-null member means not checked. The
  browser never infers status from diagnostic prose or mutates a report after a generic
  training-request failure. A non-current status does not itself mean stale
  policy or replanning. The separate `replan_required` lifecycle result owns
  that instruction. `ExpertTopologyRail` owns topology and
  resident-versus-active memory only.
- **Failure boundary:** The closed HTTP 422 `no_feasible_plan` response carries
  rejected candidates plus the same decision, source, and nullable receipt.
  Provider-backed failures require the matching receipt, user-attested failures
  forbid one, the required model subject must match the submitted ID and
  immutable revision, request and receipt identities must correlate, and every
  candidate must be rejected with its complete method, distribution, status,
  feasibility, rejection, target, runtime, decision, and binding tuple. Any
  broken chain is rejected before the non-compilable rows render.
- **Boundary:** `aptus.api.v1`, `aptus.facts.v3`,
  `aptus.training-plan.v5`, `aptus.bundle.v3`, and
  `aptus.runtime-contract.v1` remain unchanged. Phase 6 remains pending and owns
  a second reviewed policy with its own runtime evidence.
- **Verification:** Strict ingress and presentation cases live in
  `web/src/api.test.ts`, `web/src/lib/modelPolicy.test.ts`,
  `web/src/components/ModelPolicyPanel.test.tsx`,
  `web/src/components/ExpertTopologyRail.test.tsx`, and `web/src/App.test.tsx`.
  `test_plan_openapi_declares_typed_no_feasible_policy_chain`,
  `test_no_fit_response_preserves_provider_inspection_receipt`,
  `test_no_feasible_provider_plan_preserves_policy_receipt_chain`, and
  `test_phase5_workbench_policy_authority_is_documented` protect the server,
  planner, client, presentation, and maintained-guidance boundaries.
- **Owner:** API, planning, workbench, and documentation maintainers

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
