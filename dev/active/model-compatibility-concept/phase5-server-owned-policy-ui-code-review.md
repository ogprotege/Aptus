# Phase 5 Server-Owned Policy UI Code Review

Last Updated: 2026-08-04

## Executive Summary

Phase 5 has the right architectural center: `moeCompatibilityFromPlan()` is
gone, the topology component no longer decides compatibility, policy records
use generated OpenAPI types, the client deeply decodes the nested v2 decision
and binding chain, the selected Compare row supplies its own nullable binding,
and `422 no_feasible_plan` now preserves the server decision, source, and
receipt. The focused Python and web suites, TypeScript, Ruff, OpenAPI generation,
client-contract checks, and `git diff --check` all pass.

The packet is not ready to merge. Two release-blocking issues remain:

1. The packaged workbench was not regenerated, so the Python package still
   serves the old compatibility UI.
2. The evidence-readiness state machine does not match the server's
   authorization semantics and can both mislabel an ordinary deferred admission
   as stale and show `Ready` when authorization was never checked.

There are also important boundary gaps around unbound candidates, no-feasible
responses, validation-report correlation, and the mismatch between what OpenAPI
guarantees and what the runtime decoder assumes.

## Critical Issues (must fix)

### CRIT-1: The production package still serves the pre-Phase-5 workbench

The source UI changed substantially, but `src/aptus/_web/` has no diff. The
tracked bundle still contains `Exact MoE path recognized` and does not contain
`Server-owned compatibility` or `Model-policy match`
(`src/aptus/_web/assets/index-B9Q0Q2zI.js`). This means `aptus serve` from the
wheel/package does not expose the implementation under review even though the
Vite source tests pass.

This violates the repository's explicit generated-asset rule: a source-only web
change is incomplete until the packaged build is regenerated and its installed
asset smoke test passes.

**Required change:** after the source and contract findings below are resolved,
run the normal production build, review the tracked `_web` diff, and run the
packaged/installed-workbench smoke test. Do not hand-edit the generated bundle.

### CRIT-2: Evidence readiness conflates validation evidence, deferred admission, and stale authorization

`buildModelPolicyPresentation()` considers only an explicit
`authorization_current === false` stale. A candidate-bound `pilot-pass` report
with the field missing or null reaches `nextAction === null` and is labeled
`Ready` (`web/src/lib/modelPolicy.ts:781-800`). The panel then simultaneously
says that every required model-policy gate passed and shows host authorization
as `Not checked` (`web/src/components/ModelPolicyPanel.tsx:247-258,286-295`). A
missing security-relevant field must not produce a positive readiness claim.

The opposite path is also wrong. The API deliberately emits
`authorization_current: false` after ordinary pilot validation because deep
pilot/capacity admission is deferred until full-training submission, not because
the plan is stale (`src/aptus/api.py:1687-1697`). Bootstrap and job polling use
the same meaning (`src/aptus/api.py:1213-1225`,
`src/aptus/execution.py:3634-3646`). The new UI treats every false value as stale,
labels it `Blocked`/`Stale`, and directs the user to create a new plan
(`web/src/lib/modelPolicy.ts:787-800`,
`web/src/components/ModelPolicyPanel.tsx:237-269`). A normal passing pilot can
therefore send the operator into a pointless replan/compile/validate loop.

The report itself is not runtime-decoded. Job reports are cast from arbitrary
objects (`web/src/api.ts:309-312`), bootstrap bundle reports are cast wholesale
(`web/src/api.ts:717-720`), and synchronous validation is returned as a manual
`ValidationReport` (`web/src/api.ts:847-867`). The Phase 5 view then trusts a
known state string and one copied `bindings.candidate_id`. This is too weak a
basis for a positive evidence statement.

**Required change:** model these as separate states, for example:

- validation evidence incomplete;
- required validation evidence complete, admission not checked;
- admission current for the active launch;
- coherent evidence but stale policy/artifact binding;
- implementation blocked or invalid report.

Prefer a server-owned discriminated authorization status/reason over inferring
meaning from a boolean plus prose. Runtime-decode the report fields consumed by
the presentation. A positive readiness state must require a coherent known
state and all required exact bindings; absence must remain unknown or blocked,
never ready.

## Important Improvements (should fix)

### IMP-1: The browser still invents a validation ladder for an unbound candidate

For a null policy binding, `requiredLevelsForCandidate()` expands the candidate
runtime's `pilot-required` value into the hard-coded sequence `model-data ->
measured-preflight -> pilot` (`web/src/lib/modelPolicy.ts:705-712`). This is a
browser-owned policy rule, not a decoded server path. The same panel correctly
says that an unbound candidate has no registered model-policy path and that no
policy-path claim applies (`web/src/components/ModelPolicyPanel.tsx:151-158,
220-224`), but its evidence record can still claim those three policy gates.

This is especially misleading for a rejected candidate in a no-feasible
comparison. The client selects the first rejected row, then tells the user to
validate it even though a no-feasible partial plan cannot be compiled.

**Required change:** derive model-policy validation levels only from the exact
decoded path referenced by that candidate's binding. Keep generic runtime
requirements and planner feasibility as separate records. An unbound,
unsupported, or infeasible candidate should not receive a browser-synthesized
policy ladder or an impossible “validate this candidate” next step.

### IMP-2: The published candidate response is weaker than the client contract

`PlanCandidateResponse` and its generated type require only
`candidate_id`, `model_policy_decision_id`, and `policy_binding`
(`src/aptus/api_contracts.py:477-480`,
`web/src/generated/openapi.ts:1111-1119`). Yet the Phase 5 decoder and view require
or consume method, distribution, target modules, runtime contract, status, and
feasibility. A payload with only the three documented fields is valid according
to `NoFeasiblePlanResponse`; this was reproduced with
`NoFeasiblePlanResponse.model_validate()`. The web no-feasible decoder rejects
that valid published shape because `method` is absent (`web/src/api.ts:487-498`).
For a bound normal candidate, binding validation also assumes distribution,
targets, and runtime fields that OpenAPI does not require
(`web/src/lib/modelPolicy.ts:534-545`).

The success and failure validators are not symmetric either. The new
`NoFeasiblePlanResponse` checks each binding's receipt ID, while
`TrainingPlanResponse` checks only decision ID and source
(`src/aptus/api_contracts.py:497-524,535-565`). The domain producer is coherent,
but the public HTTP model does not express the same guarantee that the client
enforces.

**Required change:** publish a typed candidate execution tuple sufficient for
the UI and binding equality checks, with the same cross-record validators on
success and no-feasible responses. Then derive maintained browser types from
that generated schema. If the intentionally open full candidate remains, add a
closed nested presentation/binding record rather than relying on undeclared
extras.

### IMP-3: Null bindings bypass decoding of the execution values rendered by React

The plan ingress loop deeply validates candidate execution fields only when
`policy_binding` is non-null (`web/src/api.ts:417-442`). An unbound candidate is
accepted after checking IDs and the explicit null. The no-feasible path adds
only a `method` string check (`web/src/api.ts:487-511`). Raw distribution,
target-module, status, and runtime-contract values are then cast to
`CandidatePlan` and copied into the presentation (`web/src/api.ts:570-602`,
`web/src/lib/modelPolicy.ts:802-817`).

Consequences include malformed values becoming display claims and non-renderable
objects reaching React rather than failing with a contract/version error. This
is most exposed on exactly the unbound and rejected rows that Phase 5 is meant
to keep explicit.

**Required change:** decode the complete presentation-facing candidate tuple for
every row before hydration, independent of binding presence. Binding validation
should add cross-record equality checks, not be the only thing that makes the
candidate fields typed.

### IMP-4: Evidence reports are correlated by candidate ID alone

The selected-row isolation itself is correct: changing the inspected row does
not reuse another row's report. However, the only correlation is
`report.bindings.candidate_id === candidate.candidate_id`
(`web/src/lib/modelPolicy.ts:778-785`). The report already carries `plan_id` and
`model_revision`, and the server relies on a larger binding set including bundle,
dataset, environment, hardware, and metrics (`src/aptus/validation.py:1815-1843`).
The browser neither validates nor checks those identities. It also does not
recompute candidate content IDs, so a matching-looking 20-hex ID is not itself a
proof that the report belongs to the active plan.

**Required change:** pass the active plan identity and immutable model revision
into the view-model builder and require at least plan ID, candidate ID, and model
revision equality before consuming report state. Validate the report's state and
binding value types at ingress. Keep deeper launch authorization on the server;
this check is for honest presentation, not a new browser authorization engine.

### IMP-5: The no-feasible decoder is not closed at its top-level boundary

The server model is closed and requires a non-empty `message`
(`src/aptus/api_contracts.py:527-533`). The client does not check exact top-level
keys and accepts a missing or non-string message by substituting the generic
`ApiError` message (`web/src/api.ts:461-540,672-688,805-818`). It also does not
verify that every candidate is actually rejected. Thus the runtime behavior is
weaker and different from the generated contract even though the nested policy
objects are strict.

**Required change:** decode the complete `NoFeasiblePlanResponse` shape, require
its non-empty message, reject extras if the server contract remains closed, and
require a coherent rejected status/feasibility state for each row.

### IMP-6: Receipt and request-subject correlation is incomplete on partial responses

Success-plan decoding correlates a provider receipt with `payload.model` only if
that undeclared extra happens to contain string `model_id` and `revision`
(`web/src/api.ts:397-406`). A no-feasible response calls
`decodeModelInspectionReceipt()` with only the policy decision, so a receipt for
a different model or revision can pass client ingress
(`web/src/api.ts:461-476`). The `api.plan()` normalizer also does not require the
returned source to agree with whether the submitted request carried a receipt.
The server planner validates these relationships, but the advertised strict
browser boundary does not.

**Required change:** pass the request's exact model ID, immutable revision, and
expected source/receipt ID into both normalizers, or include a closed subject
record in the typed 422 response. Reject provider-to-attested downgrades and
cross-artifact receipts instead of accepting an internally coherent chain for
the wrong request.

### IMP-7: The browser still consumes the retired compatibility projection as a second decision

`api.inspectModel()` continues to normalize the legacy
`ModelCompatibilityResponse`, requires it on every successful inspection, and
returns it even though no production component reads it
(`web/src/api.ts:748-790`; the only remaining production caller of
`normalizeModelCompatibility()` is there). `web/src/lib/modelCompatibility.ts`
therefore retains a second runtime/backend/method/adapter compatibility
normalizer beside the v2 decision decoder.

This no longer reconstructs Qwen predicates, so it is safer than the removed
function, but it contradicts the Phase 5 goal of one browser-consumed
server-produced decision and keeps unnecessary version coupling. A valid v2
receipt with a missing legacy projection is rejected.

**Required change:** make the decoded receipt decision the workbench policy
source and retire the unused legacy client projection once any non-workbench API
compatibility commitment is accounted for. If it must remain temporarily, name
and test it as a legacy transport field and do not make it a prerequisite for
the v2 UI.

## Minor Suggestions (nice to have)

### MIN-1: Finish generated-type ownership for containing records

The policy aliases in `web/src/types.ts:93-99` are a good improvement. The
containing `ModelInspectionResponse` still widens a generated closed status with
`| string`, duplicates most fields manually, and allows arbitrary fact keys
(`web/src/types.ts:101-130`). `TrainingPlan`, `CandidatePlan`, and
`ValidationReport` are likewise broad maintained casts. Define explicit decoded
view types from generated inputs instead of making a generated type strict and
then widening its container.

### MIN-2: Differentiate contract/version failure in App-level tests

The decoder correctly gives an unsupported v2 schema a version-skew error rather
than turning it into a genuine blocked model decision. Add an App integration
assertion that the operator sees that recovery guidance and that no policy panel
hydrates. The current component test proves only the pure decoder distinction.

## Architecture Considerations

The intended dependency direction is now visible and should be kept:

```text
provider observation
    -> server model-policy decision
        -> candidate assessment and optional exact binding
            -> candidate/plan-bound validation evidence
                -> ephemeral launch admission
```

Phase 5 currently compresses the last two nodes into one `authorization_current`
boolean and reconstructs a validation ladder when the binding is absent. Keep
all five records separate. Structural browser checks are appropriate—schema,
known vocabulary, exact keys, and equality between records—but the browser
should neither invent a path/gate nor decide why the host declined to mark an
authorization current.

The content-derived candidate ID is useful correlation, but the web client does
not verify its content derivation. It should therefore compare the additional
server-provided plan/report identities before making a positive display claim.
Actual training admission remains server-owned and fail-closed.

## Test Gaps

Add focused regressions for:

1. A normal `pilot-pass` report with `authorization_current: false` and the
   server's “checked atomically at submission” reason; it must not say stale or
   instruct replan.
2. `pilot-pass` with missing/null authorization; it must not become a positive
   ready state.
3. Same candidate ID but wrong plan ID or model revision; the report must not be
   consumed.
4. A malformed unbound candidate execution tuple; reject it before React
   hydration.
5. A no-feasible comparison in the App, including an infeasible bound row; do
   not suggest compiling or validating that row.
6. Provider-backed no-feasible decoding with wrong receipt model, revision,
   source, receipt ID, and semantic decision.
7. Missing/blank message, extra top-level field, duplicate candidate ID, and a
   purportedly feasible candidate in `no_feasible_plan`.
8. Success and no-feasible OpenAPI models requiring every execution field the
   Phase 5 decoder consumes, with symmetric receipt-binding validators.
9. A packaged-workbench assertion that the generated asset contains the new
   panel and no old `Exact MoE path recognized` policy copy.

## Verification Performed

Passed on the uncommitted tree:

- `npm --prefix web test -- --run` — 21 files, 107 tests.
- `npm --prefix web run typecheck`.
- `PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.aptus.test_planning tests.aptus.test_api -v` — 87 tests.
- `.venv/bin/ruff check` on the changed Python/test files.
- `.venv/bin/python tools/generate_openapi.py --check`.
- `npm --prefix web run openapi:check`.
- `.venv/bin/python tools/check_client_contracts.py`.
- `git diff --check`.

No full repository test run, production Vite build, or macOS desktop build was
performed as part of this read-only review.

## Next Steps

1. Correct and type the evidence/admission state machine first.
2. Close the server/client candidate and no-feasible contracts, then harden all
   presentation-facing decoders.
3. Remove the remaining browser-invented unbound gate ladder and legacy
   compatibility dependency.
4. Add the missing integration and negative tests.
5. Regenerate and verify the packaged workbench last, then run the complete
   repository gates.

Please review the findings and approve which changes to implement before I
proceed with any fixes.

---

## Final Re-review Disposition — 2026-08-04

### Outcome

No Critical issues remain. The accepted fixes resolve the two former release
blockers and all seven former Important findings at their structural boundaries:

- the tracked packaged workbench is rebuilt and byte-identical to a fresh Vite
  production build;
- missing authorization now remains `validation-complete` / `Not checked`, an
  ordinary false authorization no longer means stale policy or replanning, and
  true authorization is the only `authorized` state;
- unbound and rejected candidates receive no browser-invented policy ladder or
  impossible validation action;
- OpenAPI now requires the complete candidate execution tuple, and every
  candidate is runtime-decoded whether its binding is exact or null;
- report-backed evidence requires exact plan ID, candidate ID, and immutable
  model-revision correlation;
- `422 no_feasible_plan` is closed, requires a nonblank message and model
  subject, rejects viable rows, and is correlated with the submitted model,
  source, and receipt;
- success and failure contracts enforce exact/null bindings plus receipt,
  subject, and semantic-decision coherence;
- the unused legacy compatibility normalizer and its tests are removed; and
- bootstrap contract/version skew is surfaced without hydrating a policy panel.

One Important issue remains, so the final tree is not yet architecture-clean.

### IMP-FINAL-1: Admission meaning is still inferred and sometimes authored by the browser

The new readiness states are structurally separate, but the wire contract still
provides only `authorization_current` plus free-form `authorization_error` text.
The browser decides that a false value means deferred admission by searching for
the sentence fragment `performed atomically when full training is submitted`
(`web/src/lib/modelPolicy.ts:145-146,1011-1012`). A server copy edit can therefore
change `admission-deferred` into `authorization-blocked` without any schema or
reason-code change. Conversely, any denied message that happens to contain that
fragment receives the retry-oriented deferred action. The current server emits
three prose variants for this behavior (`src/aptus/api.py:1215-1219,1693-1698`;
`src/aptus/execution.py:3634-3639`), but no typed discriminator makes their
meaning stable.

The browser also creates an authorization result on its own. Every failed train
request overwrites the selected report with `authorization_current: false` and
claims that the server's atomic admission rejected the launch
(`web/src/App.tsx:817-823`). That catch also handles a local missing-project
precondition, transport failures, and distinct server responses such as
`replan_required`, `job_prerequisite`, `runtime_unavailable`, and
`active_job_conflict` (`src/aptus/api.py:1782-1829`). In those cases no atomic
admission denial was necessarily returned, yet the server-owned policy panel
shows one.

This remains Important rather than Critical because both paths fail closed and
cannot fabricate current authorization. It still breaks the claimed authority
boundary and can give the operator the wrong next action. Carry a decoded,
server-owned authorization status and reason code (for example
`not-checked | deferred | current | blocked`) and update the report only from
that typed result. Preserve the existing report when a precondition, transport,
or unrelated typed API error occurs.

### Final Verification

Passed on the final uncommitted tree:

- `npm --prefix web test` — 20 files, 118 tests.
- `npm --prefix web run typecheck`.
- `PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.aptus.test_api tests.aptus.test_documentation tests.aptus.test_planning -v`
  — 115 tests.
- Ruff on the changed Python and Python-test files.
- `.venv/bin/python tools/generate_openapi.py --check`.
- `npm --prefix web run openapi:check`.
- `.venv/bin/python tools/check_client_contracts.py`.
- `git diff --check`.
- A fresh Vite production build into a temporary directory; its `index.html`,
  `index-Waq_5c0F.js`, and `index-CNzTMeOx.css` are byte-identical to
  `src/aptus/_web/`.

No implementation files were changed by this re-review.

---

## Post-hardening Architecture Sign-off — 2026-08-04

### Outcome

Phase 5 is architecture-clean for release: **no Critical or Important findings
remain**. This disposition supersedes `IMP-FINAL-1` above. The final corrections
replace prose interpretation with the typed `current | deferred | blocked`
authorization status, preserve the report after unrelated training-request
failures, and consistently require the active plan ID, recommended candidate ID,
and immutable model revision before validation evidence can unlock a stage or
action.

The other adversarial boundaries are also closed:

- success and typed `422 no_feasible_plan` responses retain and validate the
  complete model/decision/receipt/candidate chain;
- the recommended record must be structurally equal to its listed candidate,
  including planning extras, with object-key order ignored and array order
  preserved;
- path-matched receipts require coherent, revision-bound provenance containing
  provider-declared evidence, and inferred-only summaries fail closed;
- bootstrap surfaces plan, receipt, and binding contract errors instead of
  hiding them or hydrating a policy panel;
- authorization decorations are server-authored for synchronous validation,
  bootstrap, and job reads, while project persistence strips their ephemeral
  status, boolean, diagnostic, and capacity fields; and
- the current packaged asset contains the corrected server-owned presentation
  and no retired browser compatibility reconstruction.

### Remaining Findings

- **Critical:** none.
- **Important:** none.
- **Minor — generated containing-view ownership:** `web/src/types.ts` still has
  broad manually maintained containers such as `ModelInspectionResponse`,
  `TrainingPlan`, and `ValidationReport` around generated policy aliases. The
  Phase 5 ingress decoders validate every policy, candidate, receipt,
  authorization, and report-binding field used for decisions before hydration,
  so this is non-blocking. A later cleanup can derive narrower decoded view
  types from generated inputs and remove widenings such as inspection
  `status | string`.

### Verification Performed

Passed on the final uncommitted tree:

- `npm --prefix web test -- --run src/lib/modelPolicy.test.ts src/api.test.ts src/App.test.tsx src/stages/ValidateStage.test.tsx src/stages/RunStage.test.tsx`
  — 5 files, 76 tests.
- `PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.aptus.test_api tests.aptus.test_execution tests.aptus.test_planning tests.aptus.test_projects -q`
  — 173 tests; the only diagnostic was the known Starlette/httpx deprecation
  warning.
- `npm --prefix web run typecheck`.
- `.venv/bin/python tools/generate_openapi.py --check`.
- `npm --prefix web run openapi:check`.
- `.venv/bin/python tools/check_client_contracts.py`.
- `git diff --check`.
- Packaged-asset inspection of `index-eXcVscNl.js`: the server-owned policy and
  typed-authorization decoder strings are present; `Exact MoE path recognized`
  and the retired deferred-admission prose predicate are absent.

A subsequent adversarial parity recheck also found zero Critical, Important, or
Minor findings after the client began rejecting non-null prelaunch-capacity
evidence without a coherent typed authorization tuple. The focused 15-test
model-policy decoder suite and the matching Python authorization-contract
regression passed against the final packaged asset.

No implementation files were changed by this sign-off.
