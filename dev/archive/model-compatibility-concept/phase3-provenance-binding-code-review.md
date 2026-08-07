# Phase 3 Provenance Binding Code Review

> **Documentation status:** Archived and superseded review evidence
>
> **Applies to:** Point-in-time Phase 3 provenance-binding review recorded below
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or a named successor changes
>
> **Historical warning:** This review is preserved without rewriting its body.
> Statements below that say a condition is current, open, or complete describe
> the reviewed snapshot, not the present repository. Use the
> [historical-review index](../README.md) to find current successors.

**Last Updated:** 2026-07-29
**Review basis:** Current `feat/compatibility-provenance-binding` working tree.
**Scope:** Provider inspection, receipt construction and validation, compatibility
decision identity, candidate policy binding, plan v4 identity, saved-plan loading,
project bootstrap and recovery, CLI and API ingestion, browser receipt lifecycle,
portable validation, tests, and maintained documentation.

## Executive Summary

Phase 3 has the right overall shape. The implementation keeps separate
compatibility-subject and observed-planning-facts digests. Decision identity
excludes explanatory prose. Every candidate links to one decision. Only an exact
registered path receives a policy binding. V3, v2, and schema-less saved plans
fail closed. The receipt survives API, CLI, browser, project recovery, and
planner-parity paths. Phase 4 and Phase 5 work remains deliberately out of scope.

The branch is not ready to merge yet. One critical provenance defect lets a
receipt relabel a decision as `provider-inspection` even when the decision facts
are wholly user-attested or absent from the receipt. Three important issues also
remain. The stale-policy classifier accepts internally inconsistent plans as
legitimate historical state. Mutable config retrieval can be treated as
revision-bound from a repository-controlled body field. The generated API and
browser types do not express the required v4 provenance fields.

The focused Python review suite passed 183 tests. Those passing tests do not
cover the adversarial cases below.

## Critical Issues

### CRIT-1: Receipt presence can relabel a user-attested decision as provider-inspection

`ModelInspectionProvenance` accepts every generic `ProvenanceKind`, including
`user-attested`, `unknown`, and `measured`; it does not restrict receipt entries
to facts actually observed or inferred by inspection
(`src/aptus/domain.py:743-771`). Receipt construction copies any such kind
without a source-class invariant (`src/aptus/model_compatibility.py:631-658`).
For an unregistered dense or unknown decision, the validator sets no provenance
requirement at all (`src/aptus/model_compatibility.py:659-671`,
`src/aptus/model_compatibility.py:728-748`). The portable validator has the same
open-kind behavior and no decision-subject coverage rule
(`src/aptus/plan_contract.py:856-921`). Finally, the planner chooses
`provider-inspection` from receipt presence alone
(`src/aptus/planning.py:928-946`).

Two read-only reproductions succeeded:

```text
family-recognized provider-inspection ['user-attested'] None False
provider-inspection ['license_name'] llama family-recognized
```

The first receipt marked every covered fact `user-attested`. The second covered
only provider-declared `license_name`; family, model type, architecture, and
layers used by the persisted compatibility decision were not receipt-covered.
Both plans were accepted as `provider-inspection`.

This defeats the principal Phase 3 distinction. A content hash need not
authenticate the caller, but a structurally valid receipt must not contradict
its own source label. The documentation also says the observed digest covers
provider-declared or inferred planning facts and that omitted fields remain
user-attested (`docs/methodology/facts-and-provenance.md:66-85`).

**Required correction:** Define receipt-specific provenance kinds. For v1 they
should be `provider-declared` and `inferred`. Require every non-null
compatibility-subject field behind a `provider-inspection` decision to appear in
the receipt with an allowed inspection kind. Keep omitted non-policy planning
fields outside the receipt and explicitly user-attested in the model ledger.
Apply the same rule in the host and portable validators. Add dense, unknown, and
exact-Qwen negative tests for all-user, unknown-kind, and partial-subject
receipts.

The exact v1 coverage rule should be:

1. Build the canonical subject-field set from `family`, `model_type`,
   `architecture`, `layers`, `quantization_bits`, `quantization_layout`, and
   `moe`.
2. Every non-null value in that set must have one sorted receipt provenance
   entry at the receipt revision.
3. Receipt entries may be `provider-declared` or `inferred` only.
4. A provider-inspection subject must contain at least one
   `provider-declared` subject entry. Inferred family or architecture facts do
   not establish a provider observation by themselves.
5. A path-matched registered policy must also pass its existing stricter
   required-field rule. For Qwen3 MoE, raw architecture, layers, model type, MoE
   topology, quantization bits, and quantization layout remain
   `provider-declared`; family remains an Aptus inference.
6. Non-subject planning facts such as context length, hidden size,
   intermediate size, and license may be receipt-covered only with the same two
   inspection kinds. If omitted, their plan provenance must remain
   user-attested.

## Important Improvements

### IMP-1: The stale-policy classifier does not establish that the old plan is internally valid

`require_current_model_policy()` promises that malformed or tampered v4 state
remains a plain `ValueError`. Its stale branch verifies only the nested decision
shape, decision content ID, subject digest, policy ID, and semantic version
(`src/aptus/plan_contract.py:620-676`). It does not verify candidate decision
links, policy bindings, candidate IDs, receipt identity, or the plan ID before
raising `StaleModelPolicyError`.

The current regression test encodes this defect. It changes `policy_version`
and recomputes only `decision_id`, leaving every dependent identifier stale, and
still expects `replan_required` (`tests/aptus/test_plan_contract.py:121-143`). A
read-only reproduction confirmed:

```text
candidate links agree with changed decision? False
plan ID still valid? False
StaleModelPolicyError ... replan_required
```

Execution remains blocked, but the system misclassifies corrupted or edited
state as a legitimate older policy artifact. That weakens the error boundary
and contradicts the documented requirement that a syntactically valid v4 plan
enters `replan_required` (`docs/reference/plan-schema.md:11-20`).

**Required correction:** Before classifying a version mismatch as stale, verify
the historical plan's internal semantic chain independently of the current
policy: decision ID, receipt ID and digest, candidate decision links, candidate
bindings, candidate IDs, recommendation, and plan ID. Update the positive stale
fixture so all dependent historical IDs are coherent. Add a separate test that
the present partial rewrite is classified as tampering.

The stale classifier should require these dependent-integrity checks before it
raises `StaleModelPolicyError`:

1. Recompute `subject_facts_sha256` from the normalized model and compare it to
   the persisted decision.
2. Recompute the persisted decision ID from its semantic payload, excluding
   only explanatory prose.
3. Require every candidate's `model_policy_decision_id` to equal that persisted
   decision ID.
4. For each non-null binding, require decision ID, subject digest, policy ID,
   policy version, reason codes, evidence IDs, source, and receipt ID to agree
   with the persisted decision and plan. Require its `path_id` to exist in the
   persisted decision and require the candidate method, distribution, target
   modules, and runtime contract to match that path. Forbid a binding on every
   nonmatching candidate.
5. For a provider source, validate the receipt against the persisted decision,
   model ID, revision, observed-facts digest, provenance coverage, and receipt
   ID without substituting the current decision. For a user source, require a
   null receipt and the closed all-fields user-attested ledger.
6. Recompute every candidate ID, require the recommended candidate to equal its
   listed record, and recompute the plan ID.
7. Only after steps 1 through 6 pass should a known registered policy ID with a
   different semantic version be classified as stale. Any broken dependency is
   tampering or corruption, not `replan_required` historical state.

### IMP-2: A mutable config response can self-assert an immutable revision

Inspection correctly fetches metadata through the resolved commit URL. The
config side is weaker. It fetches `config.json` through the caller's requested
branch or tag and accepts either the transport header or the config body's
`_commit_hash` as the resolved revision (`src/aptus/inspection.py:403-449`). A
repository owner controls `config.json`, so `_commit_hash` is not independent
transport evidence that the returned bytes came from that commit.

A read-only reproduction supplied `revision=main`, no commit header, and a body
with `_commit_hash = "a" * 40`. Inspection returned `status=ok`, issued a
receipt for that hash, and recorded config-derived sources as:

```text
https://huggingface.co/org/model/resolve/main/config.json
```

Those mutable source URLs are also persisted for config facts and Aptus
inferences (`src/aptus/inspection.py:492-499`,
`src/aptus/inspection.py:532-559`). The existing negative test covers a missing
hash, but not a body-asserted hash (`tests/aptus/test_inspection.py:537-540`).

The model-data gate still protects execution by loading the pinned revision, so
this is not an execution bypass. It is an inspection-provenance defect.

**Required correction:** For a mutable requested ref, require a provider
transport commit header or refetch `config.json` through the resolved immutable
URL and derive facts from that response. Record the resolved config URL as the
reproducible source while retaining the requested URL separately if useful.
Reject a body-only `_commit_hash` for mutable refs. Add positive immutable-ref,
header-bound mutable-ref, and negative body-only tests.

### IMP-3: The public plan response and browser types leave v4 provenance optional and unknown

The runtime response contains the new fields, but `TrainingPlanResponse` still
declares only a free-form schema string, free-form candidate dictionaries, and
no decision, source, or receipt (`src/aptus/api_contracts.py:455-462`). Generated
OpenAPI therefore exposes candidates as unknown objects and
`schema_version: string` (`web/src/generated/openapi.ts:1540-1562`). The
handwritten browser types then make `model_policy_decision_id`,
`policy_binding`, `model_policy_decision`, `model_policy_decision_source`, and
`inspection_receipt` optional (`web/src/types.ts:300-325`,
`web/src/types.ts:359-380`). `normalizePlan()` casts the unvalidated payload to
that interface (`web/src/api.ts:371-420`).

This does not bypass backend validation. It does mean the generated contract
cannot tell a client how to preserve the new provenance chain, and the browser
can silently accept a purported v4 plan missing fields that the backend treats
as mandatory. It also recreates drift between generated and handwritten client
types during the contract migration.

**Required correction:** Add open-but-typed v4 response models for the plan,
candidate decision link, nullable policy binding, decision source, and nullable
receipt. Keep `aptus.api.v1`, but make the plan schema literal v4 and the new v4
fields required. Generate the TypeScript surface from that model and validate
the required provenance keys at browser ingestion. A no-feasible comparison
view can remain a separate explicitly partial type.

## Minor Suggestions

No minor item should be prioritized ahead of the four findings above.

## Architecture Considerations

The implementation otherwise respects the intended phase boundaries:

- Phase 3 persists a versioned receipt, decision, source, candidate link, and
  exact-path binding.
- Decision IDs and plan IDs correctly exclude explanatory policy prose while
  retaining semantic fields.
- The receipt carries both compatibility-subject and broader observed-facts
  digests. Parameter count and training permission remain outside it.
- Every candidate links to the plan decision. Nonmatching candidates serialize
  `policy_binding: null`.
- V3, v2, and schema-less plan readers preserve source bytes and require
  replanning.
- Project bootstrap, recovery, CLI compilation, API loading, and planner parity
  contain the expected v4 checks.
- Content hashes are accurately documented as tamper-evident rather than
  authenticated signatures.
- The handwritten portable policy evaluator remains for Phase 3. A portable
  snapshot and generic evaluator remain Phase 4 work. Browser policy
  reconstruction remains Phase 5 work.

Until Phase 4 removes the handwritten portable policy copy, add an explicit
host-versus-portable decision parity table for exact Qwen, each blocking reason,
dense recognized, unreviewed sparse, and unknown inputs. Current generated-plan
tests cover the main executable rows, but not every decision state.

## Next Steps

1. Seal provider-inspection source semantics and subject-fact provenance
   coverage in both validators.
2. Require complete historical identity coherence before returning
   `replan_required` for a stale v4 policy.
3. Remove body-only mutable-ref resolution and persist reproducible resolved
   config sources.
4. Type the required v4 provenance fields in OpenAPI and browser ingestion.
5. Add adversarial tests for these cases plus bootstrap and recovery of a fully
   coherent stale v4 plan.
6. Run the full Python, generated-contract, browser, native, packaged-launch,
   signing, ZIP, and DMG gates after fixes.

## Re-review 1: 2026-07-29

### Outcome

No Critical finding remains. The original receipt-source defect is fixed in the
host and portable validators. The original stale-version dependency checks,
mutable-config binding, and typed API/browser contract findings are also fixed.

Three Important findings remain, so Phase 3 is not yet merge-ready.

### Verified original corrections

- Receipt entries are limited to `provider-declared` and `inferred`; every
  non-null compatibility-subject fact must be covered; at least one subject fact
  must be provider-declared; and exact Qwen policy facts retain their stricter
  provider-declared rule (`src/aptus/domain.py:820-831`,
  `src/aptus/model_compatibility.py:680-709`,
  `src/aptus/plan_contract.py:908-1003`).
- A coherent historical Qwen policy version, including a fully relinked provider
  receipt, raises `StaleModelPolicyError`. Partial identity rewrites and a stale
  plan with rehashed forged evidence remain plain `ValueError` corruption
  (`src/aptus/plan_contract.py:667-729`).
- Mutable refs require a transport commit binding; a body-only `_commit_hash` is
  rejected, immutable requested revisions remain accepted, and persisted config
  sources use the resolved URL (`src/aptus/inspection.py:429-460`,
  `src/aptus/inspection.py:539-570`).
- `TrainingPlanResponse`, generated OpenAPI, handwritten browser types, and
  browser ingestion now require v4 decision, source, nullable receipt, candidate
  link, and explicit nullable binding fields (`src/aptus/api_contracts.py:454-523`,
  `web/src/api.ts:365-415`).
- Full evidence-record content is part of plan identity and is checked against
  the portable canonical registry. All seven embedded record hashes matched the
  host registry. Rehashed mutations remained invalid
  (`src/aptus/plan_contract.py:2000-2006`,
  `src/aptus/plan_contract.py:2372-2417`).
- Known adapter targets replayed under an unknown family are rejected for every
  adapter candidate (`src/aptus/plan_contract.py:2765-2778`).

### Remaining Important findings

#### RE-IMP-1: Historical stale validation still depends on today's target catalog

`require_current_model_policy()` sends the persisted decision into historical
validation (`src/aptus/plan_contract.py:715-720`), but candidate validation still
reads the current `MODEL_TARGET_MODULES` table
(`src/aptus/plan_contract.py:2765-2782`). A plan that validated under its original
catalog was tested after a simulated policy addition and target-catalog change:

```text
old_catalog_validation ()
future_exception_type ValueError
future_is_stale False
Saved model policy dependencies are malformed or tampered: Candidate 3 target modules do not match...
```

An evaluator-only future policy addition correctly raised
`StaleModelPolicyError`; the failure appears when the addition changes the target
catalog. Historical classification must use persisted policy semantics or stable
historical invariants, not today's family target mapping. The current test at
`tests/aptus/test_plan_contract.py:284-314` patches only the evaluator and misses
this case.

#### RE-IMP-2: A generated unknown-family plan fails its own portable validator

The planner correctly leaves adapter targets empty and marks those candidates
unsupported for an unknown family (`src/aptus/planning.py:423-429`). Empty targets
produce zero trainable parameters and zero checkpoint retention
(`src/aptus/planning.py:747-756`). The portable validator nevertheless requires
positive checkpoint retention for every candidate, including unsupported ones
(`src/aptus/plan_contract.py:2592-2599`).

A canonical generated unknown-family plan produced nine errors, one for each
adapter candidate from Candidate 3 through Candidate 11:

```text
Candidate 3 checkpoint_retention_bytes must be positive.
...
Candidate 11 checkpoint_retention_bytes must be positive.
```

The API saves a newly generated plan before portable replay validation
(`src/aptus/api.py:1422-1446`), so it can persist and return a plan that later
fails reload or compilation. Define an honest zero-storage contract for an
unsupported zero-parameter candidate, or another non-fabricated representation,
and test actual unknown-family generation, save, and reload.

#### RE-IMP-3: A malformed family type can crash portable validation

The validator records a non-string model family as invalid
(`src/aptus/plan_contract.py:2055-2061`) but later passes it unguarded to
`MODEL_TARGET_MODULES.get()` (`src/aptus/plan_contract.py:2765`). A coherently
reidentified JSON plan with `model.family = ["unknown-model"]` raised:

```text
TypeError: unhashable type: 'list'
```

This fails closed, so it is not an execution bypass. It can still cause a CLI
traceback or API 500 instead of a typed validation error. Guard the lookup and
add malformed JSON scalar-type coverage.

### Focused verification results

- 212 focused Python tests passed.
- 19 web test files and 90 tests passed; TypeScript typecheck passed.
- OpenAPI generation check, maintained-client parity, and version parity passed.
- Ruff and `git diff --check` passed.
- Custom probes confirmed canonical evidence digest parity, coherent stale
  provider-receipt classification, forged-evidence corruption classification,
  and unknown-family target replay rejection.

## Re-review 2: 2026-07-29

### Outcome

No Critical or Important finding remains. RE-IMP-1, RE-IMP-2, and RE-IMP-3 are
resolved. An additional host-versus-portable uppercase-family discrepancy found
during this re-review was also corrected before the final run. Phase 3 is
merge-ready from the architecture and integration scope reviewed here.

### Verified Re-review 1 corrections

- Historical stale-plan validation no longer reads today's adapter-target
  catalog as the historical truth. It derives the relevant targets from the
  persisted decision paths or from the internally consistent persisted
  candidates (`src/aptus/plan_contract.py:2391-2443`) and uses that result for
  candidate replay (`src/aptus/plan_contract.py:2871-2882`). The required probe
  patched both `_current_model_policy_decision` and `MODEL_TARGET_MODULES`; an
  internally coherent old plan raised `StaleModelPolicyError`, not corruption.
- A genuinely unknown-family plan now has an explicit honest adapter contract:
  unsupported adapter candidates carry empty target modules and zero checkpoint
  retention. The portable validator accepts that exact combination while
  retaining the positive-retention rule for feasible candidates. Generated-plan
  validation passed, and API save plus reload returned the same plan ID
  (`tests/aptus/test_plan_contract.py:318-345`,
  `tests/aptus/test_api.py:792-814`).
- Portable validation is total over the exercised malformed JSON scalar space.
  Enum-like membership checks require strings, target-table lookup is guarded,
  finite-number checks handle integers outside the float range, and the public
  validator converts structural `OverflowError`, `TypeError`, and related
  failures into typed validation errors (`src/aptus/plan_contract.py:342-354`,
  `src/aptus/plan_contract.py:3066-3073`). A coherently relinked plan with
  `model.family` as a JSON array returned a family validation error without an
  exception (`tests/aptus/test_plan_contract.py:420-445`). Positive and negative
  `10**400` values for evaluation fraction and learning rate also returned
  errors without escaping (`tests/aptus/test_plan_contract.py:347-405`).
- The host-versus-portable table covers exact Qwen, each plan-representable Qwen
  blocker, dense recognized, unreviewed sparse, and unknown decisions
  (`tests/aptus/test_plan_contract.py:62-144`). An independent probe additionally
  found that uppercase `QWEN3_MOE` was previously normalized by the host but not
  by portable exact matching. The final tree establishes lowercase family as a
  boundary invariant in `ModelSpec`, `ModelCompatibilitySubject`, portable
  subject construction, and plan validation (`src/aptus/domain.py:457-460`,
  `src/aptus/domain.py:548-564`, `src/aptus/plan_contract.py:444-447`,
  `src/aptus/plan_contract.py:2086`). Both host and portable boundaries now
  reject that value consistently (`tests/aptus/test_plan_contract.py:407-418`).

### Final verification results

- All seven directly affected current-tree regressions passed, including the
  evaluator-plus-target-catalog stale probe, unknown generation and API
  save/reload, malformed scalar and oversized-number totality, uppercase-family
  boundary parity, the full decision-state parity table, and coherent array
  family rejection.
- The broader focused Python module run passed 217 tests immediately before the
  final numeric and lowercase-boundary corrections. The seven affected tests
  were then rerun successfully on the final current tree.
- Custom current-tree probes confirmed `StaleModelPolicyError` after a simulated
  future target-catalog change, portable acceptance of the unknown-family zero
  contract, typed rejection of a coherently relinked array family, and matching
  uppercase-family rejection at both policy boundaries.
- No implementation, test, or product documentation file was edited during
  this review. Only this review record was updated.

### Final disposition

There are **zero Critical findings and zero Important findings remaining** in
the requested Phase 3 review scope. No new lower-severity issue warrants holding
the phase.
