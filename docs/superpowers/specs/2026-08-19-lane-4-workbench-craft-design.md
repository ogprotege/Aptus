# Lane 4 — Workbench craft specification

> **Status:** APPROVED 2026-08-19 — owner replied "I approve"  
> **Authority:** Subordinate to claim language, current capabilities, the UI/UX contract, and mission invariants I1–I12  
> **Increment:** Lane 4. **Not M10.** **Not 0.3.** **Not a 0.2 ship.**  
> **Implementation plan:** `docs/superpowers/plans/2026-08-19-lane-4-workbench-craft.md`  
> **Last reviewed:** 2026-08-19  
> **Next scheduled review:** After Tasks 1–7 merge, or when a later increment opens

Owner sign-off (chat 2026-08-19):

- Make the referee look as serious as it is (not Unsloth Train chrome)
- Five-stage React workbench only (not Mac Home / Machine / Models this round)
- Keep teal / amber / red and current type; tighten craft
- One pass across all five stages
- Increment name **Lane 4**, not a version bump

---

## 1. Goal

Aptus 0.2 is cut-frozen as the five-stage referee. Lane 4 does not grow that
contract. It makes the existing React workbench *look* as honest as the
product already *is*.

The operator should read status, caution, and blocked before they read
decoration. Empty, omitted, and “no last call” must look empty — never like
a silent yes.

## 2. Non-goals

- Growing Aptus 0.2 (new stages, new verbs, new planner statuses)
- Bumping the product version to 0.3
- Naming this increment M10
- Redesigning the five-stage flow or collapsing Run’s five actions
- Unsloth-style Train chrome: live-win charts, celebration, “your model is ready”
- Inferring Use from `measured-run-pass` or Stop from gold fail
- Changing Python planner, compiler, JobService, or HTTP contracts
- Restyling the native Mac Home / Machine / Models destinations
- AutoML, more epochs, a third journey on the Done recipes
- Committing `aptus-work/`
- New claim-language quality yeses

## 3. Surfaces

In scope: the packaged React workbench (`web/src/`), including the copy
embedded in the Mac Workbench tab.

Out of scope: SwiftUI shell, CLI stderr layout, generated OpenAPI, bundle
`train.py`.

## 4. Visual identity (keep)

Do not replace the identity. Tighten it.

| Token | Meaning |
| --- | --- |
| Familjen Grotesk | Display |
| Atkinson Hyperlegible | Body |
| IBM Plex Mono | IDs, paths, numbers |
| Teal (`--circuit-teal`) | Open path / current action |
| Amber (`--calibration-amber`) | Caution / conditional / unreviewed |
| Red (`--fault-red`) | Blocked / refused / unsupported |
| Graphite on porcelain/cloud | Default text and ground |

Light and dark already exist. Keep both. Color is never the only status
signal (existing a11y rule).

## 5. Craft rubric

One rubric on Facts, Compare, Compile, Validate, and Run:

1. **Hierarchy.** Status and evidence kind are the first read. Decoration is last.
2. **Scale and space.** One type scale and one spacing rhythm in `web/src/styles.css`. No one-off magic margins that fight the rail.
3. **First-class empty.** Missing inspection, no last call, omitted eval, and unbound policy look omitted — not like a default Use or a fake complete ladder.
4. **Caution is visible.** Conditional, unreviewed-runtime, and `replan_required` stay amber-meaning, not a green path.
5. **Blocked is visible.** Unsupported and infeasible stay in the catalog. They do not hide and do not look like a failed load.
6. **Last call is a door.** Use / Done / Stop is a responsibility attest, not a trophy.
7. **Motion.** Existing `--motion-fast` (150ms) and `prefers-reduced-motion` only. No celebration, confetti, or score-up animations.
8. **No new verbs.** Copy stays inside claim language and the UI/UX contract.

## 6. Stage jobs (behavior unchanged)

Lane 4 may restyle, regroup visually, and fix hierarchy. It may not change
what each stage *does*.

| Stage | Must still |
| --- | --- |
| Facts | Distinguish attested vs inferred vs inspected. Inspection cannot look like a training permission. |
| Compare | Show feasible, conditional, infeasible, and unsupported. Point estimate ≠ upper envelope ≠ measured. Recommendation is “within the enumerated set,” not best. |
| Compile | New path only. No overwrite. Success is “bundle written,” not ship. |
| Validate | Ladder rungs are evidence levels. Analytic ≠ measured. Not model quality. |
| Run | Five ordered actions, not one Train. `verifying` is not success. Last call missing ≠ Use. |

Project history, example-mode labeling, and the three Model-policy records
(match, selected path, evidence readiness) stay. They get the same rubric.

## 7. Files (implementation, after spec approval)

Primary:

- `web/src/styles.css` — scale, space, shared states
- `web/src/stages/*.tsx` — Facts, Compare, Compile, Validate, Run
- `web/src/components/` — `StatusBadge`, `StageHeader`, `EmptyStage`, `WorkflowRail`, `CandidateComparison`, `FitLedger`, `RunConsole`, `ValidationGates`, `ModelPolicyPanel`, `ProvenanceBadge`

Also, in the same implementation change as visible copy or layout:

- `docs/product/ui-ux.md` (last reviewed + any craft sentences that become contract)
- Colocated `*.test.tsx` for states this increment restyles (empty, blocked, last-call door, example mode)
- Packaged `src/aptus/_web/` after `npm --prefix web run build`

Do not hand-edit `docs/reference/openapi.v1.json` or `web/src/generated/openapi.ts`.

## 8. Testing and verification

- Existing Vitest + Testing Library stage tests stay green.
- Accessibility acceptance (`web/src/accessibility.test.tsx`) stays green: labels, live regions, keyboard, color-not-only.
- New or extended tests cover: empty/omitted last call; unsupported row still visible; example mode still labeled; last-call buttons still the three verbs.
- Typecheck and production build pass. Packaged `_web` assets commit with the visual change.
- Desktop and mobile viewports of the workbench (user rule): layout craft is verified at both, not screenshot-only.
- No Python unittest suite expansion required unless a claim-language sentence is added.
- No CUDA/MLX pilot (no trainer, memory, or export change).
- `desktop/macos/build.sh` is not required unless a native file is touched (it must not be).

## 9. Claim language

Allowed (already locked, still true):

- “operator-attested run disposition”
- “no last call recorded”
- “Use is not a quality yes”
- “Done closes this recipe; it does not ship Aptus 0.2.”
- “cut-freeze 0.2 as the five-stage referee”

Forbidden (still true):

- “Aptus decided this adapter is good”
- “gold fail means Stop”
- “measured-run-pass means Use”
- “reviewed 7B”
- “universally optimal” / “guaranteed to fit”

Lane 4 may add one allowed sentence if the implementation needs a lock:

- “craft is not a quality yes”

It may not add a sentence that treats a prettier screen as a stronger evidence
level.

## 10. Frozen decisions

| Decision | Choice |
| --- | --- |
| Increment name | Lane 4 |
| Version bump | No (not 0.3) |
| Product line | New increment; do not grow 0.2 |
| Surfaces | React five-stage workbench only |
| Identity | Keep type + teal/amber/red meanings |
| Scope of pass | All five stages in one increment |
| Native shell | Out of scope |
| Unsloth Train chrome | Forbidden |
| Contracts | Presentation/CSS/components only |

## 11. Success

Lane 4 is done when a novice and an expert, looking at the same five screens,
can tell path from caution from blocked without reading a legend, and when no
screen looks more certain than its evidence.

It is not done when it looks “more like a shipped consumer app” if that look
hides a no.
