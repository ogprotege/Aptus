# Documentation debt log

> **Documentation status:** Active governance
>
> **Applies to:** Open and recently resolved documentation work
>
> **Last reviewed:** 2026-08-09
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
- **Status:** Resolved
- **Resolution:** `docs/reference/cli.md` now embeds
  `aptus.cli-parser-contract.v1`. The documentation test recursively projects
  the live parser tree, expands shared planning-fact groups, and compares every
  command, subcommand, exposed argument, choice, and non-suppressed default as
  structured data. Booleans, nulls, lists, numbers, strings, and path defaults
  are no longer validated by prose substring matching.
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
- **Status:** Resolved
- **Resolution:** The repository owner confirmed the dedicated
  `aptus-security@proton.me` mailbox is active. `SECURITY.md` and the GitHub
  issue-routing configuration now publish the mailbox, with the chooser linking
  to the policy page that exposes its private `mailto:` route. The policy retains
  the supported-version table, three-business-day acknowledgment target,
  seven-day initial-assessment target, and coordinated-disclosure process.
  Reporters no longer depend on GitHub private vulnerability reporting being
  available for the repository's visibility state.
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
  performance, or quality evidence. The
  [2026-08-05 Qwen2 MLX-LM exact-source
  refresh](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
  now records two fresh, clean current-contract v5/v3 `measured-run-pass`
  repetitions for its exact pinned artifact and revision, source commit and
  tree, Apple M5 Pro host, Python/MLX runtime, dataset, policy snapshot, plan,
  bundle, and new fingerprint. It closes the Phase 6 MLX-LM runtime gate only
  for that scope. The [original Phase 6
  acceptance](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md),
  July MLX-LM record, and desktop record remain historical at their tested
  commits.
  The [2026-08-06 SmolLM2 CUDA LoRA single-device
  acceptance](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
  adds one fresh five-job `measured-run-pass` workflow at exact source
  `c12c4d8db0037a2c278a2ad95a0a2cbda4387eed`, with checkpoint-continuation
  pilot, full training, structural PEFT export, and parent promotion bound to
  its recorded Ubuntu/RTX 3050 environment. It is one execution, not
  repeatability or general CUDA acceptance.
  Phase 0 recovery completed privately on 2026-08-08 with a complete private
  disposition, two verified copies in separate failure domains, and verified
  off-host retrieval. This status publishes no protected paths, machine or job
  identities, or raw content. Phase 2A implements and independently reviews
  opt-in Phase 4 authority, admission/activation, all seven native outcomes,
  capture, telemetry, watchdog, semantic sealing, custody, sanitization,
  eligibility, and two-pass rollback-safe publication. Its [source-tooling
  contract](../operations/cuda-campaign-phase2-tooling.md) is not operator
  authorization and reports no new Ubuntu or empirical run. Phase 2B used the
  exact merged source to publish the independently reviewed [sanitized recovery
  supplement](../operations/evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md):
  39 of 40 logical rows were recovered with matching digests, while the raw
  model-file manifest and separately searched Python transcript remain absent.
  It also reports no new Ubuntu or empirical run. The [CUDA campaign
  protocol](../reference/cuda-campaign-protocol.md) and
  [machine-readable companion](../reference/cuda-campaign-protocol.v1.json)
  freeze the Phase 1 decisions; they do not implement runtime behavior.
  The later [2026-08-10 Phase 5 repeatability
  packet](../operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md)
  records five of five predeclared SmolLM2 LoRA single-device slots passing the
  frozen stability and integrity contract, with verified off-host copies and
  fresh retrieval. It establishes the exact frozen anchor and Phase 6
  eligibility, while the remaining campaign and release evidence stay open.
- **Required result:** Execute the [canonical RTX 3050 CUDA empirical evidence
  campaign](../operations/cuda-empirical-campaign.md) to publish the reviewed
  sanitized projection of the privately protected prior records, establish
  complete capture and retrieval for new attempts, and characterize the
  remaining admitted single-device methods, model scale, guarded
  configuration frontiers, endurance, and Ubuntu job control. Add a later
  multi-GPU campaign for DDP and conditional LoRA FSDP, plus Developer ID signed
  and notarized desktop evidence for the exact public release commit.
- **Milestones:** Phase 0 raw recovery is privately complete, Phase 1 is frozen,
  Phase 2A source tooling is merged and source-gated, and Phase 2B publication
  is complete and independently reviewed. Phase 3 explicit candidate selection
  and measurement controls are implemented under the v6 plan contract. Phase 4
  rehearsal and freeze are complete, and the successful Phase 5 five-attempt
  LoRA anchor packet is merged. The campaign may next run the Phase 6
  four-method matrix, then the size and configuration staircases, endurance,
  and independently reviewed dated packets. Capability claims change only
  after the applicable packets merge.
- **Blocker:** The Phase 6 same-model method matrix and later campaign phases
  remain pending. The intended RTX 3050 host has one GPU and cannot close DDP
  or FSDP; its local boundaries are
  not Aptus's cloud or multi-GPU ceiling. Those rows require approved multi-GPU
  access. Public notarization also requires the corresponding Apple
  credentials.
- **Owner:** Release maintainers

### DOC-012: Test generated operator documentation as a contract

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** A registry-derived test matrix must equal every executable
  runtime, backend, method, and placement row, then compile a genuine
  deterministic recommendation for each row to `static-pass`. The current 11
  rows cover CUDA Full, LoRA, int8-LoRA, and QLoRA across their single, DDP, and
  LoRA-FSDP placements, plus single-device MLX-LM LoRA and QLoRA. Every emitted
  `README.md`, `decision-report.md`, and `runbook.md` is checked for command
  order, evidence and quality boundaries, platform notes, filenames, successor
  links, selected compiler/export identities, and placement-specific guidance.
  The resulting template corrections change manifested operator prose only;
  runtime programs and dependencies are unchanged, and prior exact-bundle
  runtime evidence is not transferred to a newly compiled fingerprint.
- **Owner:** Compiler and documentation maintainers

### DOC-016: Resolve OpenAPI generator development advisories

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** A lockfile-only refresh moved `@redocly/openapi-core` to
  1.34.18, its exact `js-yaml` dependency to 4.3.0, `brace-expansion` to 2.1.4,
  and `postcss` to 8.5.25. A clean install and the full 2026-08-05 `npm audit`
  report zero vulnerabilities. The declared dependency ranges, checked OpenAPI
  schema, generated TypeScript client, and bundled web output remain unchanged;
  OpenAPI parity, type checking, 127 web tests, and the production build pass.
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
- **Resolution:** `aptus.training-plan.v6` binds the SHA-256 of deterministic
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
  assigned to Phase 5. DOC-023 records its completion. The second reviewed
  policy was still pending at that historical closeout; DOC-024 records its
  current implementation, exact runtime closeout, and preserved evidence scope.
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
  `aptus.training-plan.v6`, `aptus.bundle.v3`, and
  `aptus.runtime-contract.v1` remain unchanged. Phase 6 has since implemented a
  second reviewed configuration-footprint policy; DOC-024 records the exact
  Phase 6 runtime closeout and the limits that remain in force.
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

### DOC-024: Close Phase 6 runtime evidence for the second model policy

- **Priority:** P1
- **Status:** Resolved
- **Implemented boundary:** The data-driven host registry and canonical
  snapshot now contain `model.qwen2-24l.mlx-qlora`. It binds exact `qwen`,
  `qwen2`, and `Qwen2ForCausalLM` identity, 24 layers, dense topology, a uniform
  four-bit group-size-64 layout with no overrides, and the single
  `mlx-lm.qlora.single.dense-causal-lm.v1` path. The
  `dense-causal-lm.v1` profile covers all seven attention and MLP projection
  targets. Host and portable evaluation, planner binding, per-policy receipt
  provenance, canonical evidence, compiler storage, and mutation tests cover
  the implementation.
- **Resolution:** The
  [2026-08-05 Qwen2 MLX-LM acceptance](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
  records two clean `measured-run-pass` repetitions from acceptance source
  `14ed44b52a76bb84d8d9db4f2303951aa641339b`. Both repetitions completed the
  dependency, exact model-data, measured-preflight, uninterrupted real-model
  pilot, full-train, immutable export, fresh-process reload, and parent-owned
  reconciliation ladder under `aptus.training-plan.v6` and `aptus.bundle.v3`.
  The packet binds the exact pinned Qwen2.5 0.5B artifact and revision, source,
  Apple M5 Pro host, Python/MLX runtime, dataset, policy snapshot, plan, bundle,
  metrics, and artifacts.
- **Evidence boundary:** This supplies Phase 6 MLX-LM runtime evidence at its
  exact acceptance source only for that scope. The policy remains a reviewed configuration
  footprint, not an artifact allowlist; a different matching artifact must
  complete its own model-data, measured-preflight, and pilot gates. The result
  does not qualify CUDA, establish model quality, or promise production-scale
  throughput. The retained July runtime evidence remains historical under its
  own v2/v2 scope.
- **Verification:** The acceptance packet's machine-readable summary and
  `SHA256SUMS` bind both five-job repetitions, terminal validation reports,
  immutable exports, reload evidence, and parent-promotion receipts.
- **Owner:** Policy, MLX runtime, release-evidence, and documentation maintainers

### DOC-025: Refresh Phase 6 evidence at the exact acceptance source

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** The
  [2026-08-05 exact-source refresh](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
  records two fresh, clean, independent v5-plan/v3-bundle workflows through
  `measured-run-pass` at exact source commit
  `719255153e3fc7e38e83b5ff826d587e5e58bf80`, source tree
  `be99f5664ccb580f2600471f1ae3241a294b1a7e`, bundle fingerprint
  `ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`, and
  ZIP SHA-256
  `fcad829b4c845c6b5d1e548b293ec1107ccd7a78ea08b63bc7a1b8ca487be9b1`.
  The [original Phase 6 acceptance
  packet](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
  remains byte-for-byte unchanged as the historical baseline.
- **Comparison boundary:** Relative to that baseline, exactly the manifested
  operator `README.md` and `runbook.md` changed. The runtime programs,
  `requirements.txt`, retained plan, policy snapshot, and split contract stayed
  byte-identical. The original runs do not transfer to the new bundle identity;
  the two fresh workflows independently qualify the new fingerprint.
- **Evidence boundary:** The refresh applies only to the exact pinned Qwen2.5
  artifact and immutable revision, source commit and tree, M5 Pro host,
  Python/MLX runtime and resolved environment, four-row synthetic dataset, v5
  plan, v3 bundle, policy snapshot, and new fingerprint. It does not qualify
  CUDA, generalize Qwen2 compatibility, establish safety or model quality,
  support performance or throughput claims, or establish production or release
  readiness.
- **Verification:** The packet's `SHA256SUMS` covers each committed projection
  exactly once; its machine-readable comparison binds the two changed operator
  documents and unchanged runtime/dependency hashes, and its sanitized run
  projections bind both terminal `measured-run-pass` workflows without raw job
  state or local absolute paths.
- **Owner:** MLX runtime, release-evidence, and documentation maintainers

### DOC-026: Close maintained client response-contract parity

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** React now fails closed on malformed job and profile records,
  live compile responses without their required report, and method catalogs that
  violate schema identity, lifecycle, or runtime-binding coverage invariants.
  Restored bundles retain their documented allowance for an absent report.
  Swift now validates the health contract and service version before readiness, requires
  the complete persisted runtime-configuration response, consumes all six
  runtime-inventory fields with advertised-versus-measured availability and
  compatibility parity, and handles only the documented `ok` and `unsupported`
  platform statuses.
- **Verification:** Focused React and XCTest regression tables cover canonical
  fixtures and malformed required, closed-vocabulary, and cross-field variants.
  `tools/check_client_contracts.py` now binds all four native HTTP routes to
  their OpenAPI response models, required top-level fields, constants, and
  closed status values. Python mutation tests prove that new required fields,
  changed closed values, missing decoder markers, and endpoint drift fail the
  checker.
- **Boundary:** OpenAPI JSON and generated TypeScript remain generated. React
  normalization and Swift decoding remain maintained layers: unknown extra
  response properties stay forward-compatible, while missing required fields,
  malformed values, and unknown closed statuses fail closed.
- **Owner:** Web, native host, API-contract, and documentation maintainers

### DOC-027: Record first exact CUDA LoRA single-device acceptance

- **Priority:** P1
- **Status:** Resolved
- **Resolution:** The [2026-08-06 SmolLM2 CUDA LoRA single-device
  packet](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
  records one fresh qualifying five-job workflow through
  `measured-run-pass` at source
  `c12c4d8db0037a2c278a2ad95a0a2cbda4387eed`. It binds the exact Ubuntu/RTX
  3050 host, Python/CUDA runtime closure, immutable model revision, four-row
  synthetic dataset, v5 plan, v3 bundle, policy snapshot, pilot metrics, full
  metrics, structural PEFT export, and parent-promotion receipt.
- **Verification:** Two independent compilations were byte-identical. The
  qualifying state contains exactly dependency, model-data, preflight, pilot,
  and train records, all completed with return code zero. Independent checks
  rehashed the final tree and parent evidence, and `SHA256SUMS` covers every
  committed packet projection.
- **Boundary:** This is one exact LoRA single-device execution. It does not
  establish repeatability, other CUDA methods or placements, semantic adapter
  reload, quality, safety, benchmark performance, production readiness, or
  release readiness. DOC-011 therefore remains in progress.
- **Owner:** CUDA runtime, release-evidence, and documentation maintainers

### DOC-028: Reconcile post-Phase 6 documentation and lifecycle drift

- **Priority:** P1
- **Status:** Resolved by PR #41
- **Finding:** The 2026-08-06 repository-wide audit found one operationally
  important reserve-default omission and several maintenance gaps: the README
  had conflicting review dates and overbroad support wording; setup blocks could
  leave contributors in `web/`; quality-gate lists, CUDA dependencies, inference
  defaults, and artifact descriptions had drifted; twelve completed reviews
  remained under `dev/active/`; the native build guide and nine subordinate
  legacy reports were outside metadata and navigation checks; and active MLX
  guidance used aging `current-source` language for exact-source evidence.
- **Resolution:** The merged tree aligns reserve, dependency, API-default,
  setup, gate, artifact, capability, and exact-source wording with code and
  evidence. It moves the twelve completed reviews to an indexed `dev/archive/`
  tree without rewriting their bodies, adds archived warnings to every legacy
  report, governs the active desktop build guide, updates README review metadata,
  and inventories the generated MLX `reload.py` surface.
- **Verification:** Documentation tests enumerate all 120 tracked Markdown
  files, classify 119 governed pages as 90 active, 2 deprecated, and 27
  archived, require metadata and primary-index reachability for all 119, require
  the review archive to be empty under `dev/active/`, and pin the corrected
  reserve and lifecycle claims. Local links, anchors, and code fences remain
  part of the same gate.
- **Boundary:** PR #41 changed documentation and documentation regression tests
  only; it did not broaden MLX, CUDA, quality, performance, notarization, or
  release evidence.
- **Owner:** Documentation, runtime, desktop, API, and contributor-workflow maintainers

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
- [CUDA campaign protocol](../reference/cuda-campaign-protocol.md)
- [Release gates](../operations/release-gates.md)
