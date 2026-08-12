# Mission program — phase execution protocol

> **Status:** Active operating procedure  
> **Authority:** Binds how every mission phase is run  
> **Established:** 2026-08-11 with Phase M0  
> **Method:** Subagent-driven development + anti-rush completion gates  
> **Program plan:** [`docs/product/mission-integrity-plan.md`](../../../docs/product/mission-integrity-plan.md)

This protocol exists so phases cannot “feel done.” Every completion is written
down, reviewed, and checked against the mission before the next phase opens.

---

## 1. Roles

| Role | Who | Allowed to |
| --- | --- | --- |
| **Controller** | Primary agent in this chat | Dispatch tasks, maintain ledger, adjudicate reviews, never skip gates |
| **Implementer** | Fresh subagent per task | Do only the task brief; write report file; no next-phase work |
| **Task reviewer** | Fresh subagent after each task | Spec compliance + quality; no implementation |
| **Owner** | Human (you) | Freeze decisions, Path identities, phase sign-off, cost authorization |

Never let the controller “just fix” review findings without a fix-round
subagent when the work is material. For pure planning freezes (M0), the
controller may assemble freeze docs **only after** research subagents return
and a reviewer has checked them.

---

## 2. Artifacts that must stay current

| Artifact | Path | Updated when |
| --- | --- | --- |
| Program plan | `docs/product/mission-integrity-plan.md` | Phase exits change checkboxes / freeze tables |
| SDD ledger | `.superpowers/sdd/2026-08-11-mission-trust-when-it-says-no/progress.md` | Every task start/complete/fix/block |
| Decision log | `.superpowers/mission-integrity-plan/DECISIONS.md` | Any product choice |
| Phase completion note | `.superpowers/mission-integrity-plan/M{N}-COMPLETION.md` | Phase exit only |
| Program status | `.superpowers/mission-integrity-plan/STATUS.md` | After every phase completion (and on block) |
| Task briefs/reports | `.superpowers/sdd/.../task-*-brief.md` / `task-*-report.md` | Per task |

**Recovery rule:** After context loss, trust **ledger + STATUS + git**, not chat memory.

---

## 3. Anti-rush gates (hard stops)

A phase is **not** complete unless **all** of the following are true:

1. **Exit criteria** in the program plan are checked with evidence links (not vibes).  
2. **Completion note** exists (`M{N}-COMPLETION.md`) with:  
   - phase ID and title  
   - start/end UTC timestamps  
   - baseline git commit at phase start and end  
   - tasks completed (list)  
   - decisions recorded (DECISION IDs)  
   - evidence / artifact paths  
   - explicit **non-claims**  
   - **what was deliberately not done**  
   - open risks carried forward  
   - “next phase allowed?” yes/no with justification  
3. **STATUS.md** points at the completion note and sets `current_phase` to the **next** open phase only if exit passed.  
4. **Independent review** of the phase packet (subagent or human) found no Critical/Important mission violations, or each residual is parked with a written ruling in the ledger.  
5. **Owner sign-off** line is filled for freezes that require human authority (M0 Path tables, cost-bearing measured runs, public claim changes).  
6. **No next-phase files** were created “early” except design notes explicitly labeled `DRAFT-NOT-AUTHORIZED`.

If any gate fails → phase stays **OPEN**. Do not start the next phase.

---

## 4. Per-task loop (subagent-driven)

```text
1. Write / extract task brief (single source of requirements)
2. Record BASE = git rev-parse HEAD (if commits expected)
3. Dispatch implementer → report file only
4. Dispatch task reviewer with brief + report (+ diff package if code)
5. Fix loop max 5 rounds if Critical/Important
6. Append ledger: Task N complete | blocked | parked
7. Update STATUS only if phase-level (not every micro-task)
```

**Parallelism:** Do **not** run two implementers that write the same files.
Independent **read-only research** tasks may run in parallel when outputs are
disjoint files.

**Commits:** Planning phases may produce untracked docs under `dev/active/`
without a commit until the owner wants them on a branch. Runtime/code phases
commit per task when the plan requires it. Never commit secrets or raw job logs.

---

## 5. STATUS.md schema

```markdown
# Mission program status

- **Updated:** ISO-8601 UTC
- **HEAD:** full commit
- **Current phase:** M{N} title — OPEN | COMPLETE | BLOCKED
- **Last completed phase:** M{N} — link to completion note
- **Next allowed action:** one sentence
- **Blocked on:** none | description
- **Anti-rush:** phase N+1 NOT started (yes/no)
```

---

## 6. Completion note schema

See Section 3.2. Template path: `.superpowers/mission-integrity-plan/COMPLETION-TEMPLATE.md`.

---

## 7. Claim-language check (every phase)

Before marking complete, answer:

1. Did we invent support that evidence does not back?  
2. Did we hide a refusal?  
3. Did we use “optimal / guaranteed / works for all X”?  
4. Are Path Alpha/Beta identities still exact-bound?

Any “yes” to 1–3 → fail the phase.

---

## 8. Established with M0

This protocol itself is an M0 deliverable. Later phases refine it only via a
`DECISION-*` entry — never silently.
