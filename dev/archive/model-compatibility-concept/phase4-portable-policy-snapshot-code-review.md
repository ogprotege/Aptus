# Phase 4 pre-commit review: portable policy snapshot

> **Documentation status:** Archived and superseded review evidence
>
> **Applies to:** Point-in-time Phase 4 portable-policy review recorded below
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or a named successor changes
>
> **Historical warning:** This review is preserved without rewriting its body.
> Statements below that say a condition is current, open, or complete describe
> the reviewed snapshot, not the present repository. Use the
> [historical-review index](../README.md) to find current successors.

- **Date:** 2026-08-02
- **Branch:** `feat/portable-policy-snapshot` (uncommitted working tree on `246e15c`, the merged PR #24 / Phase 3 commit)
- **Scope:** the full uncommitted Phase 4 change packet — 49 modified tracked files (+922 / −161), two new source files (`src/aptus/policy_snapshot.py`, `tests/aptus/test_policy_snapshot.py`), one regenerated packaged web asset
- **Method:** ten independent review dimensions (snapshot core, host parity, plan contract, bundle generation, validation, API/web, test quality, docs, hygiene, phase boundary) fanned out over the diff, followed by adversarial verification of every Critical/Important claim (three verifiers per Critical, one per Important, refute-by-default). 51 agents total; divergence claims were reproduced empirically by executing both evaluators, not argued from reading. Findings the verifiers refuted are recorded below rather than dropped.
- **Verification gates run during this review (all green):** 446/446 Python tests, 90/90 web tests across 19 files, ruff format + lint, compileall, OpenAPI parity (`generate_openapi.py --check`), maintained-client parity (`check_client_contracts.py`), version parity (0.2.0), `git diff --check`, TypeScript checks, and a web rebuild that reproduced the tracked asset hash `index-B9Q0Q2zI.js` byte-identically (no tree mutation). Native/desktop gates were not run (no `desktop/` files in the diff); they remain required for the final pre-commit gate per the closeout plan.

## Post-review implementation disposition — 2026-08-02

This document preserves the findings as they were observed. The subsequent fix
round has now resolved A1, A2, C3, and A3:

- The portable evaluator normalizes only the fixed eight compatibility-subject
  fields, sorts `fact_errors` before hashing, and applies host-equivalent
  fact-error precedence. Full-decision parity regressions cover exact Qwen,
  identity near-match, dense, sparse, unknown, and unsorted multi-error inputs.
- Policy lookup now prefers an explicit snapshot, then the current registry when
  `plan_contract` is running inside installed Aptus, and uses the bundle-root
  snapshot only in top-level/package-free execution.
- Host submission, pilot authorization, artifact verification, and parent
  completion promotion enforce current policy. Stale job submission is typed as
  HTTP 409 `replan_required`; recovered pending evidence cannot cross a host
  policy change into measured-run promotion.
- Host admission captures one manifest, plan, and snapshot state, proves the
  embedded decision coherent, binds the observed artifact fingerprint into the
  job, and builds the command from that admitted plan rather than rereading
  mutable files. Workers and train-capacity authorization recheck the same
  artifact and current host snapshot before launch.
- Managed children receive the admitted artifact fingerprint and authorized
  policy-snapshot digest. Portable entrypoints enforce those bindings before
  using plan state, including the direct CUDA `train.py` entrypoint, while
  manual package-free execution remains valid when managed bindings are absent.
- Parent completion holds one report critical section, rechecks bundle and host
  policy immediately before publication, and atomically commits terminal state
  with an `aptus.parent-promotion.v1` receipt bound to the job, run, artifact,
  and measured evidence. Restart idempotency accepts only an exact receipt-bearing
  terminal report; pending, fabricated, drifted, or interrupted promotion fails
  closed without publishing stale terminal evidence.
- Regressions exercise a generated standalone preflight under a deliberately
  different discoverable host registry, installed-host validation, synchronous
  submission, pilot authorization, restart recovery, direct managed CUDA
  training, crash-before-commit behavior, artifact/plan mutation, and the real
  API compile-to-submit flow.

The final independent blocker-only review found no remaining Critical or
Important A3 issue. The post-fix authoritative gate passed 467 Python tests, 90
web tests, Ruff, `compileall`, TypeScript, regenerated OpenAPI and client parity,
version parity, and 81 native tests. A clean installed-wheel smoke verified the
packaged snapshot, CUDA programs, API, workbench, and hashed asset. The macOS
Release app passed ad-hoc signature verification and produced the app ZIP and
DMG. The unchanged npm installation still reports three high-severity
development-dependency advisories. No CUDA or exact-model MLX runtime acceptance
was claimed or collected in this policy-contract slice.

A second follow-up now resolves B1 and B2:

- Host validation accepts only exact lowercase 64-character hexadecimal text
  for the snapshot, plan, manifest, and current-host digest bindings. It never
  coerces hostile values, and one `POLICY_SNAPSHOT_DIGEST` finding identifies
  every invalid binding and every valid binding that differs from the snapshot.
- JSON `null`, oversized integer, and excessive-nesting snapshot documents now
  produce typed INVALID findings instead of escaping the host or portable
  manifest boundaries.
- The portable validator checks every constraint kind's exact operand set,
  requires one plain `{layer}` template, positive non-boolean quantization
  integers, non-empty unpadded field and identity values, exactly one
  `exact_identity`, coherent claims, exact path/runtime-contract primitives, and
  well-formed reason, evidence, and provenance collections before evaluation.
- Mutation regressions cover hostile digest values in all four bindings,
  simultaneous invalid/different bindings, null/resource-hostile JSON, and the
  full constraint/claim/path/reason/evidence shape matrix. Independent recursive,
  structural, and template fuzzing found no escaped evaluator exception in this
  scope.

The second follow-up's authoritative gate passed 481 Python tests, 90 web tests,
Ruff, `compileall`, TypeScript, OpenAPI/client/version parity, and 81 native
tests. The Release app passed signature verification and produced its ZIP and
DMG. A clean installed-wheel smoke verified the packaged snapshot and portable
manifest behavior. No CUDA or exact-model MLX runtime evidence was required or
collected because this packet changes contract rejection only, not runtime
execution semantics.

The other review findings remain open, notably C1/C2, the remaining smaller
C4 gaps, D1/D2, and the listed minor cleanup. The follow-up pull request should
therefore remain draft and disclose those follow-ups rather than treating this
disposition as a full review closeout.

## Verdict

**The architecture is right and the phase charter is substantively met, but the packet is not commit-ready.** There are zero Critical findings. There are confirmed Important findings in three clusters that should be resolved (or explicitly decided) before the commit/PR steps: two genuine host-versus-portable parity divergences plus one behavioral regression at execution admission, a set of robustness gaps where hostile-but-schema-valid inputs crash instead of failing closed with a report, and a large but mechanical documentation sweep (the v4→v5 migration reached roughly half of each affected doc). None of these invalidate the design; all of them are exactly the categories Phase 4 exists to get right.

## Repository state (corrects the stale handoff note)

The "we are at the Phase 3 closeout boundary" note is stale. Verified state:

| Item | State |
| --- | --- |
| Phases 1–3 (PRs #20–#24) | Merged. `main` and `origin/main` both at `246e15c` — the fast-forward closeout already happened. |
| Phase 4 | **Implemented, uncommitted**, on `feat/portable-policy-snapshot` (branched from merged main). |
| Phase 5 / Phase 6 | Not started; verified absent from this diff (see boundary results below). |
| Protected untracked files | `TempDoc-ForUserReview/`, `.agents/`, `AGENTS.md`, `skills-lock.json` untouched by the diff. `WIP.md` at the repo root is untracked/ignored (local-only). |
| New files that MUST be staged deliberately | `src/aptus/policy_snapshot.py`, `tests/aptus/test_policy_snapshot.py`, `src/aptus/_web/assets/index-B9Q0Q2zI.js`. A naive `git add -u` commit would produce a tree where `aptus` cannot import (model_compatibility.py:49, validation.py:44 import the untracked module) and the workbench's only script 404s (index.html references the untracked asset). |

## What is right (verified, not assumed)

Ninety-seven correctness confirmations were collected with file:line evidence; the load-bearing ones:

1. **Canonical JSON and digest chain are correct and self-consistent.** `sort_keys` + compact separators + `ensure_ascii=False` + `allow_nan=False`, digest computed over the exact emitted bytes including the single trailing newline; file bytes, canonical bytes, and sha256 cannot drift apart (policy_snapshot.py:32–65; consumers agree at plan_contract.py:326–333 and generation.py:646–647, 772–774).
2. **Byte-for-byte identity (charter 11) is genuinely proven twice:** synthetic registry with reversed insertion order (test_policy_snapshot.py:127–135) and real registry — bundle-emitted bytes equal an independent regeneration (test_generation.py:540–544). Every collection feeding the snapshot has an enforced deterministic order.
3. **Decision-identity parity is exact where tested.** Same nine-key identity payload, same `compat_` + sha256[:20] derivation, same canonicalization; two independent parity suites compare full decision dicts including `decision_id` and `subject_facts_sha256` across nine subject shapes (test_model_compatibility.py:87–134 at layers=48, test_plan_contract.py:63–147).
4. **The subtle ordering conventions all match.** `sorted(range(layers), key=str)` (policy_snapshot.py:263) is byte-identical to pre-existing host semantics (catalog.py:41) and to the repository invariant that `module_overrides` sort by `module_path` (domain.py:410–413, inspection.py:205) — including double-digit layers. Sparse-marker substring matching is identical to the host's (`("moe", "mixtral")`, model_compatibility.py:280–287) and its over-blocking direction is fail-closed.
5. **Package independence is real, not claimed.** The portable module imports only hashlib/json/typing; plan_contract.py loads it standalone via `importlib.util.spec_from_file_location` when the package is absent (plan_contract.py:24–46); the compiler copies it into every bundle; and two subprocess tests execute a generated bundle's own preflight against a broken copy (test_generation.py:2091, 2121).
6. **The four-way digest binding closes the swap-tamper vector at the host.** Snapshot file bytes = plan digest = manifest digest = current host digest, checked unskippably at every validation level, each failure class with a distinct fail-closed finding code, each tested to INVALID (validation.py:1226–1289, test_validation.py:565–628).
7. **Fail-closed migration behavior is preserved and strengthened at plan load.** v4 joins v3/v2/schema-less in `replan_required`; `require_current_model_policy` now requires *both* current snapshot digest *and* semantic decision equality, preserving the stale-versus-tampered classification (plan_contract.py:817–871).
8. **The compiler refuses wrong bindings up front** — a plan bound to a different snapshot digest cannot compile (generation.py:673–677, tested at test_generation.py:556), non-empty-output refusal and atomicity are intact, and emitted paths are fixed relative constants (no traversal).
9. **The schema-advancement decision (charter 12) is explicit and consistent in code:** `aptus.training-plan.v5`, `aptus.bundle.v3`, `aptus.model-policy-snapshot.v1` agree across domain.py, plan_contract.py, generation.py, api_contracts.py, the generated OpenAPI artifact and TS client (both verified byte-fresh against regeneration), hand-maintained web types, and the updated reference docs that were reached by the sweep.
10. **The web surface is clean:** `moeCompatibilityFromPlan()` survives untouched except version literals (no accidental Phase 5); the client adds a fail-closed digest gate mirroring the server regex (web/src/api.ts:354–365); demo fixtures use an obvious placeholder digest (no fabricated measured facts).
11. **No Phase 6 leakage, and extensibility fails closed:** the snapshot generator hard-fails on any policy it cannot express portably (model_compatibility.py:401–404).
12. **Hygiene is clean:** no secrets, no debug leftovers, no whitespace errors, no unrelated files in the tracked diff; claim-rule language holds (the Qwen3 MoE 47.759 GiB admission refusal remains framed as refusal; no estimate became a guarantee).

## Confirmed findings

Zero Critical. Important findings below survived adversarial verification (most with empirical reproduction).

### A. Parity and fail-closed behavior (code)

**A1. Portable evaluator diverges from the host on `fact_errors` for non-claiming subjects — fail-open direction.**
`src/aptus/policy_snapshot.py:326` vs `src/aptus/model_compatibility.py:1082`. The host blocks *every* subject with fact_errors before the policy loop; the portable evaluator checks fact_errors only inside a claiming policy. Reproduced: `family="llama"` + `fact_errors=("quantization: contradictory",)` → host `blocked/invalid-compatibility-facts`, portable `family-recognized`; mixtral-with-errors and unknown-with-errors also diverge in kind/reason/decision_id. Reachability nuance (why this is Important, not Critical): today's production bundle path builds subjects via `_compatibility_subject_payload`, which emits `fact_errors: []` (plan_contract.py:525), so the divergent branch is not reachable through a compiled bundle — but the module *is* the portable contract Phase 4 exists to establish, the divergence violates the exact-parity charter (and docs/architecture/data-and-identity-flow.md:176–177), and the direction is fail-open. **Fix:** hoist the fact_errors handling above the policy loop mirroring the host exactly (claiming + exact identity → invalid; claiming + identity mismatch → identity reason; otherwise → blocked/invalid), then add fact_errors subjects to both parity suites (see C3).

**A2. Portable subject digest diverges for unsorted `fact_errors`.**
`src/aptus/policy_snapshot.py:194` vs `model_compatibility.py:599`. The host hashes the subject with `fact_errors` sorted; the portable evaluator hashes the mapping as given. Reproduced: same subject, errors `("z: later", "a: earlier")` → different `subject_facts_sha256` and `decision_id` even when semantics agree. inspection.py emits `quantization:` before `moe:` (reverse-lexicographic), so real dual-error subjects would hit this. Same reachability nuance as A1. **Fix:** sort `fact_errors` inside the portable digest/decision path to match the host byte-for-byte.

**A3. Execution admission no longer enforces policy currency — a silent behavioral regression from Phase 3.**
`src/aptus/plan_contract.py:771–777, 2252–2256` with callers `execution.py:425, 1557, 2669` (reached from `JobService.submit` and pilot authorization). Because `_policy_snapshot_for_validation(root=...)` now returns the bundle's *own frozen snapshot*, every admission/verification call that passes `root=bundle` validates the plan against the bundle itself — tautologically satisfied, since generation bound them together. Pre-Phase-4, these call sites recomputed the decision from the *current host* policy, so a host policy change blocked submission of stale bundles. `execution.py` now contains no reference to `current_model_policy_snapshot*` or `require_current_model_policy`; host currency is enforced only when `validate_bundle` (validation.py's four-way check) runs. **Decision required:** either restore host-currency enforcement at host-side admission (recommended — the host has the registry; frozen-snapshot self-validation is the right semantics only *inside* bundles), or explicitly document frozen-snapshot admission as intended and add a test pinning it. Charter item 8 ("preserve all current fail-closed migration behavior") currently reads as violated.

### B. Robustness: schema-valid inputs that crash instead of failing closed (code)

**B1. `validate_bundle` crashes with an unhandled `TypeError` on unhashable recorded digests.**
`src/aptus/validation.py:1282`. `set(bindings.values())` is built from unvalidated plan/manifest fields; `"model_policy_snapshot_sha256": []` in a tampered plan.json aborts `validate_bundle` before any report exists (reproduced), and the API validate endpoint would 500 instead of returning INVALID. **Fix:** coerce/type-check the four binding values as strings before the set comparison; non-string → emit `POLICY_SNAPSHOT_DIGEST` (or a dedicated finding), never raise.

**B2. Snapshot constraint validation checks key presence but not value types; validated snapshots can crash the evaluator.**
`src/aptus/policy_snapshot.py:171`. Reproduced three ways with snapshots that pass `validate_model_policy_snapshot`: `override_module_template=42` → `AttributeError` at line 257; template `"model.{other}.gate"` → `KeyError`; `field_equals` with a list-valued `field` → `TypeError` at line 214. The validator's stated purpose (line 75) is to screen exactly this. **Fix:** validate constraint payload value types (template must be a str containing `{layer}` and no other fields; `field` must be a non-empty str; bits/group sizes ints), and while there: require ≥1 `exact_identity` constraint per policy (closes the related `StopIteration` at line 327 flagged in minors), and reject a snapshot file whose JSON is `null` in validation.py's snapshot block (validation.py:1250 currently skips every POLICY_SNAPSHOT_* check when `json.loads` returns None; downstream layers still catch it, but this layer silently passes).

### C. Test-coverage gaps against the charter (tests)

**C1. The plan-level stale-snapshot-digest replan path has zero coverage** (charter 4/8/10). No test exercises the digest gate in `require_current_model_policy` (plan_contract.py:820–826) or the `"stale or tampered; replan_required"` error (plan_contract.py:2275) for a well-formed v5 plan whose digest differs while its decision is semantically unchanged — the exact scenario Phase 6 will create the day a second policy changes the host digest. Verified by execution that the behavior is correct; it is simply unpinned.

**C2. The in-bundle snapshot checks (`plan_contract.validate_bundle_manifest`, plan_contract.py:312–345) have zero negative-path coverage.** Every missing/malformed/noncanonical/tampered test asserts validation.py's *host-side duplicate* instead; no test runs a generated bundle program subprocess against corrupted snapshot *data* (the two subprocess tests only cover a syntactically broken module). Charter 10 requires the bundle layer explicitly.

**C3. Neither parity suite includes a subject with non-empty `fact_errors`** — which is precisely where A1/A2 live. Adding the divergent subjects makes the suites fail today; they should, until A1/A2 are fixed.

**C4. Smaller gaps:** the API-layer legacy sweep still iterates only v3/v2/None (test_api.py:846 — v4 rejection is covered at the plan-contract layer but not the API layer); `test_pre_v4_plan_requires_replanning` still substitutes v3 (test_plan_contract.py:196); `POLICY_SNAPSHOT_CONTRACT` and `POLICY_SNAPSHOT_PATH` codes are never triggered by any test; the stale-*host* direction of `POLICY_SNAPSHOT_DIGEST` (patching the host digest) is untested; the web client's new digest rejection branch has no negative test (web/src/api.test.ts).

### D. Documentation drift (docs — largest cluster by volume, mechanical to fix)

**D1. Six new finding codes are undocumented.** `POLICY_SNAPSHOT_MISSING / _JSON_ERROR / _CONTRACT / _NONCANONICAL / _DIGEST / _PATH` (validation.py:1232–1362) appear nowhere in docs/ — `docs/reference/error-codes.md`'s host-validator tables were edited in this same diff yet gained no snapshot rows, violating the same-change documentation policy.

**D2. The v4→v5 sweep reached only part of each affected file.** Confirmed stale-as-current-contract references, several of them sentences *adjacent to* lines this diff updated:

- `docs/reference/api.md` — four normative sections: :167–172 ("required v4 schema… Create a deterministic v4 plan"), :476–480 (response described as v4, `model_policy_snapshot_sha256` omitted), :488 (understates v5 rejection scope), :680–682 (remedy still "Create a new v4 plan" — an operator following it loops on 409 forever).
- `docs/reference/cli.md` — :23 ("standalone v4 plan"), :183–187 ("exact v4 domain contract… Do not relabel it as v4") vs :152 which correctly says v5.
- `docs/reference/error-codes.md:189` — operator response still "create a new v4 plan" vs :61 correctly v5.
- `docs/reference/plan-schema.md:438–445` — the normative plan-identity enumeration **omits `model_policy_snapshot_sha256`**, which `plan_id_for_payload` now includes (plan_contract.py:2144); also :452's "only after a coherent historical chain" reads wrong now that v4 itself triggers replan.
- `docs/reference/evidence-records.md:189` — "The v4 plan ID binds…" and same identity-input omission.
- `docs/architecture/data-and-identity-flow.md` — :55, :108–111 (plan-ID list missing the digest), :119–123.
- `docs/product/ui-ux.md` — :31–37, :125.
- `README.md` — :207, :261, :267–268 (vs :168–170 correctly v5).
- `docs/methodology/facts-and-provenance.md:170`, `docs/maintenance/documentation-health.md:38/59`, `docs/architecture/artifact-compiler.md:79` ("v4 host compiler", "Phase 3 handwritten check" retained).
- `docs/methodology/overview.md:66` lists the just-bumped `aptus.bundle.v3` in the *unchanged*-contracts sentence.
- `docs/reference/validation-states.md:91–105` omits stale-snapshot v5 replanning and omits `policy_snapshot.py` from the static AST-parse list.
- `docs/architecture/system.md:115` overstates: in-bundle (package-free) validation cannot detect *stale* — staleness is a host-registry comparison; only the host four-way check detects it.

`tests/aptus/test_documentation.py` checks links/fences/identifiers, not these prose claims, which is why the suite is green anyway.

## Minor findings and notes (selected)

- **Dead code shipped into every bundle:** `_retired_handwritten_model_policy_decision` (plan_contract.py:617, ~128 lines) has zero callers in src/ or tests/, carries a now-false docstring, and rides the copied plan_contract.py into every compiled bundle. The charter's precondition for removal (exact parity tests first) is met — remove it in this packet.
- **Live v5 error text still says v4:** `require_current_model_policy` docstring and messages (plan_contract.py:796, 809) — including inside bundle copies.
- `POLICY_SNAPSHOT_DIGEST` conflates stale-host and tampered-binding in one message; reporting which binding diverged would aid operators (validation.py:1286).
- Snapshot claims matching is case-sensitive on family where the host lowercases first; unreachable for current data (host emits lowercase) but worth normalizing while touching A1.
- The snapshot `reasons` map carries 10 of the host's 12 reason strings (path-restriction and blocked-inspection prose omitted) — fine today, worth a comment.
- Validator does not enforce lowercase `dense_families`/`sparse_identity_markers`, though the evaluator compares against lowercased subject values.
- Stale test names: `web/src/api.test.ts:257` ("rejects v4 plan responses…" over a v5 fixture), `tests/aptus/test_api.py:1416`.
- `tests/aptus/test_projects.py:821` changed the legacy-import assertion from a bundle-fingerprint equality to `imported["bundle"] == {}` — verified as the *deliberate* fail-closed consequence of the v2→v3 manifest bump (legacy bundles no longer validate), but the leftover unused `bundle_fingerprint` variable and the unstated behavior change deserve a comment or WIP note.
- `cuda/preflight.py:49`'s new `require_static` entry for `policy_snapshot.py` is unreachable as the failing check (plan_contract import execs the module first) — harmless, documented here for accuracy.

## Findings raised and refuted (for the record)

- **"Untracked new files make this Critical"** — refuted: the working tree is complete and imports cleanly; the hazard exists only for a careless `git add -u`-only commit, so it is a commit-step checklist item (recorded in the state table above), not a code defect.
- **`StopIteration` on snapshots lacking `exact_identity`** — refuted on reachability (host generation always emits it; production subjects come from `_compatibility_subject_payload`); retained as a minor hardening item folded into B2.
- **Subject-digest divergence as an independently reachable defect** — the divergence is real (A2) but duplicates claiming it reachable through production bundle callers were refuted (`fact_errors` pinned to `[]` there).
- **test_projects fingerprint assertion weakened** — refuted as a defect; deliberate fail-closed consequence (see minors).

## Assessment of the proposed closeout plan

The eight-step plan (final review → reconcile → gate → commit → PR → verify GitHub → merge → post-merge review) is sound and its stop conditions are right. Amendments based on this review:

1. **Insert a fix round before step 2.** Resolve A1–A3, B1–B2, C1–C4, D1–D2 (and the dead-code/minor wording items) first; anything short of that ships a portable contract whose first external property — exact parity — is provably violated by two subjects, plus a silent admission-semantics change nobody decided.
2. **A3 needs an explicit decision, not just a patch** — record whether admission enforces host currency (recommended) or frozen-snapshot semantics, and test whichever is chosen.
3. Step 4's staging instruction is load-bearing: stage the three new files explicitly; `git add -u` alone produces a broken package and a broken workbench.
4. The plan's local-state premise ("fast-forward main to 246e15c") is already satisfied; the Phase 3 post-merge closeout is effectively done and this review covers the re-confirmation items it listed.
5. Claimed gate numbers check out (446 Python / 90 web); the 81-native-test, release-build, signing, ZIP/DMG, and dependency-audit steps remain to be run at final gate time as planned.
6. After fixes, rerun the focused tests and then the complete authoritative gate, per the plan's own rule.

## Recommended fix order

1. A1 + A2 (portable evaluator parity) + C3 (parity cases that pin them) — one change, red→green.
2. A3 decision + implementation + test (admission currency).
3. B1 + B2 (+ StopIteration guard, `null`-snapshot handling) with negative tests (folds in C4's `POLICY_SNAPSHOT_CONTRACT`/`_PATH`/stale-host cases).
4. C1 + C2 (stale-digest plan test; bundle-layer `validate_bundle_manifest` negatives, ideally via a generated-program subprocess).
5. Remove `_retired_handwritten_model_policy_decision`; fix v4 wording in live messages and stale test names.
6. D1 + D2 documentation sweep (grep-driven: `v4 plan`, `the v4`, `POLICY_SNAPSHOT`), including the plan-identity enumerations in plan-schema.md and evidence-records.md.
7. Full authoritative gate; then proceed with the existing steps 2–8.

## Phase boundary confirmation

- Phase 5: `moeCompatibilityFromPlan()` exists, is exported, and its decision predicate is untouched (web/src/lib/modelInspection.ts — version-literal changes only). Not absorbed.
- Phase 6: exactly one registered policy; the snapshot generator fail-closes on unexpressible policies. Not absorbed.
- Runtime acceptance: unchanged and correctly worded — the Qwen3 30B MoE admission-failure evidence remains evidence of refusal.

## Final resolution and independent re-review — 2026-08-04

This section supersedes the provisional verdict and open-follow-up language
above. The original review remains intact as the historical record of what was
found on 2026-08-02; it is not the current disposition. Phase 4 was repaired in
small, separately reviewed changes and then re-reviewed as one contract before
this closeout.

### Original finding dispositions

| Finding | Final disposition | Change and verification |
| --- | --- | --- |
| A1 | Resolved | PR #25 (`65200be`) applies non-empty `fact_errors` before policy matching, with host-equivalent precedence. Full-decision parity tests cover claiming, near-match, dense, sparse, unknown, and invalid subjects. |
| A2 | Resolved | PR #25 normalizes the fixed compatibility-subject fields and sorts `fact_errors` before hashing. Unsorted multi-error parity is pinned. |
| A3 | Resolved | PR #25 made the intended semantic decision explicit: package-free bundle programs validate their frozen snapshot, while installed-host admission, pilot authorization, worker launch, artifact verification, and completion promotion enforce the current host policy. API submission maps stale policy to HTTP 409 `replan_required`. |
| B1 | Resolved | PR #26 (`7c2e22a`) validates all four digest bindings as lowercase 64-character hexadecimal strings before comparison and names invalid and differing bindings in a typed `POLICY_SNAPSHOT_DIGEST` finding. |
| B2 | Resolved | PR #26 validates exact constraint operands, templates, claims, paths, runtime contracts, reasons, evidence, provenance, and exactly one `exact_identity`. JSON `null`, excessive nesting, oversized integers, and hostile primitive shapes produce typed INVALID reports. |
| C1 | Resolved | PR #28 (`13afb49`) pins a well-formed v5 plan whose stale snapshot digest has a semantically unchanged decision and requires replanning. |
| C2 | Resolved | PR #28 exercises generated package-free manifest validation against missing, malformed, noncanonical, and digest-tampered snapshot inputs. Direct host and contract tests cover the remaining typed path and contract findings. |
| C3 | Resolved | PR #25 adds full host/portable decision-dictionary equality for all reviewed subject classes, including unsorted non-empty `fact_errors`. |
| C4 | Resolved | PR #26 covers `POLICY_SNAPSHOT_CONTRACT`; PR #28 covers `POLICY_SNAPSHOT_PATH`, stale-host `POLICY_SNAPSHOT_DIGEST`, the v4 API legacy case, malformed or missing web digests, and corrected stale test names. |
| D1 | Resolved | PR #29 (`9e8433f`) documents all six snapshot finding codes and their operator responses. |
| D2 | Resolved | PR #29 updates current-contract guidance to v5 while preserving genuinely historical Phase 3/v4 references, adds the snapshot digest to every plan-identity enumeration, and distinguishes portable integrity from host-only currency. Focused documentation guards pin those statements. |

The additional scalar-manifest defect identified after the original review was
resolved in PR #27 (`73f0910`): `validate_bundle_manifest()` now rejects every
non-object manifest root with a controlled error, including generated
package-free execution.

The final re-review then found a related malformed-input cluster that the broad
pre-closeout mutation sweeps had not reached. Host validation could call
mapping methods on scalar or nested scalar plan fields; semantic traversal and
current-policy identity checks could leak `RecursionError`; CUDA entrypoints
could bind devices before rejecting the plan; and raw JSON reads in generated
programs, saved-plan API/CLI paths, execution admission, and project recovery
did not uniformly normalize parser resource errors.

The closeout change resolves that cluster at shared boundaries:

- host plans, manifests, and trainer configurations must be JSON objects before
  semantic consumers run, and only a contract-valid plan supplies bindings or
  runtime evidence;
- semantic plan validation and current-policy recomputation normalize malformed
  shapes and recursive values to controlled contract errors;
- every direct CUDA and MLX entrypoint plan load uses the portable object
  parser, while CUDA validation and execution validate the plan before device
  binding;
- execution admission, local-state/project recovery, saved-plan loading and
  restoration, and CLI compilation normalize oversized integers and excessive
  nesting instead of leaking parser exceptions.

Direct regressions cover all five non-object JSON roots, four nested top-level
plan fields across five primitive shapes, 500-level semantic nesting,
10,000-level decision and JSON nesting, 5,000-digit integers, both execution
object loaders, API and CLI surfaces, package-free CUDA and MLX entrypoints, and
CUDA validation/run ordering across five malformed recommendation shapes.

### Minor findings and historical staging hazards

- The live `require_current_model_policy` v4 wording was corrected in PR #25.
  PR #28 removed the retired handwritten evaluator and corrected the CLI wording
  and stale test names.
- Digest diagnostics now identify invalid and differing bindings. The original
  diagnostic-conflation note is resolved.
- Case-sensitive raw snapshot claims and lowercase collection enforcement remain
  bounded inputs: the host registry emits normalized values, the plan/domain
  contract enforces the subject shape, and current-host currency rejects an
  altered embedded policy. They are not reachable fail-open paths in the
  admitted contract.
- The ten snapshot reasons are the evaluator's intentional portable subset; the
  two omitted host strings belong to inspection/path handling outside that
  evaluator.
- CUDA's explicit static parse of `policy_snapshot.py` remains harmless
  defense-in-depth even though import validation normally fails first.
- The legacy-project empty-bundle assertion is the deliberate v2-to-v3
  fail-closed migration behavior. The leftover unused local identified by the
  review was removed in this closeout.
- The former new-file staging warning was valid for the original uncommitted
  packet. The snapshot module, its tests, the hashed web asset, and this review
  document have all been tracked since PR #25. The closeout packet deliberately
  comprises 23 tracked files: this review and the error-code reference, 14 host
  and generated-program source files, and seven regression modules. Protected
  local files, ignored state, and build artifacts remain excluded.

No Critical or Important finding remains open, and every actionable minor has a
recorded resolution or bounded disposition above. Independent architecture,
adversarial, and documentation passes reviewed the merged result and successive
closeout diffs; final raw-boundary and evidence passes reviewed the frozen
23-file tree. The earlier 9,420 nested-plan and 636 snapshot mutations applied
to the pre-closeout tree and did not cover the parser-depth and nested-consumer
seams found later; the targeted matrices above were added specifically for those
seams. Focused development runs included 275 affected-module tests and 19
project/state tests; the authoritative final-tree run below subsumed that
coverage.

### Final source and packaging gates

| Gate | Final result |
| --- | --- |
| Python unit suite | 505/505 passed |
| Web/Vitest suite | 91/91 passed across 19 files |
| Native XCTest suite | 81/81 passed; zero failures or skips |
| Python style and syntax | Ruff format, Ruff lint, and `compileall` passed |
| Maintained contracts | OpenAPI regeneration check, maintained client-contract check, and version parity (`0.2.0`) passed |
| Web build | OpenAPI client check, TypeScript, and production build passed; tracked `index-B9Q0Q2zI.js` reproduced without drift |
| Repository hygiene | `git diff --check` passed; no generated tracked drift |
| Installed wheel | A fresh wheel built from the final closeout tree; Aptus imported from the isolated wheel target; snapshot validation and digest generation passed; all changed CUDA/MLX program resources were present; packaged CUDA validation/run source placed plan validation before device binding; packaged API health, workbench index, and hashed asset each returned HTTP 200 |
| macOS package | `desktop/macos/build.sh` passed its integrated 505-Python/91-web/81-native gates, Release build, deep strict ad-hoc signature verification, packaged backend/window/workbench/React launch probe, ZIP creation, DMG creation, mounted-DMG layout verification, and checksums |

The clean web dependency install reported four advisory findings (one moderate
and three high) in the existing dependency tree. This closeout does not change
the dependency graph or claim to resolve that separate maintenance work. The
macOS artifacts are ad-hoc signed development evidence; notarization and
clean-machine distribution acceptance were not run or claimed.

### Target-runtime evidence and phase boundary

No current-head CUDA or MLX target-runtime pilot was collected. This Apple
Silicon host cannot supply CUDA evidence, and no external CUDA target host was
available for the required two-phase checkpoint-continuation pilot. The host can
support MLX hardware, but no compatible MLX interpreter is currently
configured, and the active pilot record does not authorize another download or
training run without explicit selection of the immutable model revision, corpus
revision, method, interpreter, disk budget, and output directory. The historical
July 27 MLX acceptance predates Phase 4 and does not bind this source head.

Repository, wheel, and desktop gates therefore close the Phase 4 source and
contract review only. The absence of current-head target pilots remains a release
evidence limitation for this Phase 4 result; these gates do not renew target
acceptance, establish v0.2 release readiness, or substitute for the required
pilots. Notarization and clean-machine distribution acceptance were also not
run. Phase 5 and Phase 6 remain absent and out of scope.

**Final verdict: Phase 4's portable policy-snapshot source and contract are
commit-ready and ready for normal pull-request review, with all Critical and
Important review findings resolved.**
