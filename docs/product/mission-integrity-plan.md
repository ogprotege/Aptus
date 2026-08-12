# Mission Integrity Plan — Trust When It Says No

> **Status:** Active | **Authority:** Product program plan (subordinate to claim language, current capabilities, and release gates) | **Applies to:** Aptus path from engineering preview 0.2 toward a release-honest individual-operator product | **Audience:** Repository owner and agents executing the program | **Last reviewed:** 2026-08-11 | **Review by:** At the start of every phase and before any public claim change

> **Baseline commit at plan authorship:** `7fcdf161224ee0d4b75285b6c7d664b17af53df5`  
> **Method:** KISS · fail-closed · evidence-bound · depth before breadth  
> **Working workspace (local, not under `dev/active/`):** `.superpowers/mission-integrity-plan/`

> **For agentic workers:** Execute **one phase at a time**. Do not start Phase \(N+1\) until Phase \(N\) exit criteria are checked off with evidence links. Prefer `superpowers:subagent-driven-development` or `superpowers:executing-plans` only **inside** a single phase work package. Never “optimize” by skipping a refusal, softening claim language, or inventing unmeasured support.

---

## 0. How to use this document

### 0.1 Purpose

This is the **forensic step-by-step program** to reach the Aptus mission without drifting into feature glitter, overclaim, or premature breadth.

It is intentionally long. Length is for **integrity and followability**, not complexity of the product.

### 0.2 Reading order for a cold start

1. Section 1 (mission charter) — memorize  
2. Section 2 (invariants) — never violate  
3. Section 3 (KISS rules) — cut scope using these  
4. Section 4 (persona + success) — define “done”  
5. Section 5 (baseline truth) — do not re-prove completed work blindly  
6. Section 6 (phase map) — orientation  
7. The **current open phase** only — execute checkboxes  

### 0.3 Checkbox convention

- `- [ ]` work not done  
- `- [x]` done **only** when the exit evidence exists (commit, test output, or dated evidence record)  
- Never check a box because “it probably works”

### 0.4 Related normative sources (do not contradict)

| Source | Owns |
| --- | --- |
| [`claim-language.md`](claim-language.md) | What product text may say |
| [`current-capabilities.md`](current-capabilities.md) | Present support boundary |
| [`../operations/release-gates.md`](../operations/release-gates.md) | Release checklist |
| [`../operations/release-evidence-template.md`](../operations/release-evidence-template.md) | Evidence packet shape |
| [`../reference/capability-matrix.md`](../reference/capability-matrix.md) | Method × placement matrix |
| [`../../ROADMAP.md`](../../ROADMAP.md) | Chronology and open work |
| This document | **Program order and mission gates** |

If this plan and a normative page disagree on a **fact of current support**, the normative page wins and this plan must be corrected. If they disagree on **what to build next**, this plan wins only after an explicit human decision recorded in Section 12.

---

## 1. Locked mission charter

### 1.1 Origin (why this exists)

In 2024–2025, individual fine-tuning meant:

- spending real money on GPU instances before knowing whether method, params, model, and hardware agreed;
- no local authority that could check feasibility honestly;
- scripts and logs after the fact instead of a **pre-spend truth**;
- no tool that would refuse, explain, and point at a justified correction.

That wall is the product problem.

### 1.2 Mission statement (locked)

> **Aptus exists so an individual does not have to spend thousands learning that a fine-tune was wrong for their model, data, or hardware. It checks first, explains refusals, and produces the best supported path it can justify — or says no when it cannot. People must be able to trust Aptus when it says no.**

### 1.3 Product one-liner (locked)

> Decide whether a supervised fine-tune will actually run — before you spend the compute — then compile a reviewable bundle and climb an evidence ladder that refuses unsupported claims.

### 1.4 North-star user outcome

A single human operator, on their own machine (and optionally one CUDA host they control), can:

1. pin model + data + hardware facts;  
2. receive a ranked, **bounded** decision with visible rejections;  
3. understand **why** something failed;  
4. accept a **justified** corrected candidate when one exists;  
5. compile a no-clobber bundle;  
6. run ordered gates until `measured-run-pass` **or** a hard refuse with evidence;  
7. keep an audit trail they can re-read months later without self-gaslighting.

### 1.5 Explicit anti-mission (what success is *not*)

Success is **not**:

- the largest method catalog;
- AutoML / hyperparameter lottery;
- cloud marketplace UX;
- “one click fine-tune anything”;
- guaranteeing model quality;
- hiding unsupported paths to look more capable.

---

## 2. Global invariants (never violate)

These are non-negotiable for the entire program. Any PR that breaks one is rejected, no matter how useful the feature feels.

| ID | Invariant |
| --- | --- |
| I1 | **Provenance stays distinct.** User-attested, provider-declared, inferred, estimated, and measured facts never collapse into one blob. |
| I2 | **Estimates ≠ measurements.** Point estimates and upper envelopes never become “it fits.” |
| I3 | **Unsupported stays visible.** Never hide a rejected candidate to make the product look smarter. |
| I4 | **Recommendations name their set.** “Recommended within the enumerated Aptus candidate set,” never “optimal.” |
| I5 | **No clobber.** Never overwrite prior plan, bundle, or run output. |
| I6 | **Parent owns completion.** A child process cannot certify its own successful completion. |
| I7 | **Pilot is not quality.** Training loss and structural export are not model quality. |
| I8 | **Exact binding.** Runtime evidence binds exact plan, candidate, model revision, data, host, runtime, and policy snapshot. Evidence does not transfer by family name alone. |
| I9 | **Fail closed on ambiguity.** Unknown, partial, or contradictory claims refuse rather than default. |
| I10 | **Claim language follows evidence.** README, UI, PR text, and marketing obey [`claim-language.md`](claim-language.md). |
| I11 | **Depth before breadth.** No new method/family/host claim until the current path’s gates are closed with evidence. |
| I12 | **KISS for user-facing work.** Prefer one clear path that works over five clever paths that almost work. |

---

## 3. KISS rules for this program

### 3.1 Product KISS

1. **One persona** until Path Alpha and Path Beta both exit.  
2. **Two happy paths only** for first release-honest product promise (Mac MLX adapter; CUDA single-device LoRA handoff).  
3. **Four executable methods max** until both paths have release evidence (already true: Full, LoRA, int8-LoRA, QLoRA). Do not implement DoRA/BitFit/etc. in this program.  
4. **One correction action** from a refusal: replan with the recommended viable candidate (or show no viable path). No multi-objective searcher.  
5. **No new surface** (cloud, MCP training auth, multi-tenant service) until Section 10 says so.

### 3.2 Engineering KISS

1. Prefer configuration and registry rows over new frameworks.  
2. Prefer tests that pin claim language and fail-closed behavior over abstract refactors.  
3. Do not split `execution.py` “for cleanliness” mid-path unless a phase explicitly budgets a safety refactor with zero behavior change.  
4. One PR = one phase work package or smaller. No “while we’re here” method expansions.  
5. Documentation updates ship in the same change as behavior (repo rule).

### 3.3 Decision KISS when stuck

Ask, in order:

1. Does this make the **“no” more trustworthy**?  
2. Does this make the **“yes, with this path” more proven** for Path Alpha or Beta?  
3. Does this make the **why + fix** clearer without new paradigms?  

If the answer to all three is no, **do not build it** in this program.

---

## 4. Persona, journey, and success definition

### 4.1 Primary persona — “Solo operator”

| Field | Definition |
| --- | --- |
| Who | One technical individual (researcher, indie, small-team engineer) |
| Resources | One Mac (Apple Silicon) and/or one CUDA box they rent or own |
| Budget pain | Real money and wall-clock; cannot afford wrong 8–24h runs |
| Skill | Can pin a model revision and prepare JSONL; is not a distributed-training expert |
| Need | Pre-spend feasibility, honest refusal, justified alternative, reviewable artifacts |
| Not | Enterprise multi-tenant admin; AutoML novice expecting magic |

### 4.2 Canonical journeys (product promise targets)

#### Journey A — Local Apple Silicon (Path Alpha)

1. Install Aptus (wheel or Mac app).  
2. Select exact MLX-ready Python interpreter via doctor (no silent install).  
3. Pin eligible model revision + small supervised dataset + local hardware.  
4. Plan → see feasible/conditional/rejected with reasons.  
5. Compile bundle.  
6. Run dependency → model-data → preflight → pilot → confirmed train.  
7. Reach `measured-run-pass` **or** clear refuse with evidence.  
8. Export/artifacts remain reviewable with hashes.

#### Journey B — Mac control plane + CUDA host (Path Beta)

1. On Mac: pin facts for a CUDA host (manual or inspected profile).  
2. Plan and compile CUDA bundle.  
3. Transfer bundle to CUDA host.  
4. On host: install exact requirements, run ordered gates.  
5. Reach `measured-run-pass` **or** clear refuse.  
6. Bring evidence/summary back; no claim broader than that host/cell.

### 4.3 Definition of “worthy” for this program

The program is **worthy** when **both** are true:

1. **Trustworthy no:** On Journeys A and B, unsupported or infeasible setups are refused **before** expensive full training, with reasons a solo operator can act on, without false hope.  
2. **Release-honest yes:** For **at least one** exact Alpha configuration and **at least one** exact Beta configuration, a clean-checkout operator can complete `measured-run-pass` using documented steps, with a dated evidence record bound to commit, host, model, data, and bundle.

Optional later worth (not required to exit this program): quality evaluation contract; second host; second model family; notarized public DMG.

### 4.4 Metrics that count (and vanity metrics that do not)

| Counts | Does not count |
| --- | --- |
| Refusal correctness (false “yes” rate on known-bad fixtures) | Number of methods in registry |
| Operator time from facts → decision | GitHub stars |
| Money-not-spent stories (refused OOM before train) | Lines of code |
| Exact `measured-run-pass` evidence packets | “AI wrote the plan” |
| Claim-language audit findings = 0 on release surfaces | Demo video without digests |

---

## 5. Baseline truth (do not pretend this is zero)

As of baseline commit `7fcdf16` / plan authorship 2026-08-11:

### 5.1 Already built (engineering)

- Facts → Compare → Compile → Validate → Run workbench + CLI + API  
- Method registry (4 gated-executable + experimental/research visible)  
- CUDA + MLX portable bundle programs  
- Model policy snapshot v1 + plan v6 + bundle v3  
- Server-owned policy UI; fail-closed no-feasible payloads  
- Job service, leases, parent promotion  
- macOS app packaging (ad-hoc); CI desktop artifacts  
- Strong static test base (946 Python / 130 web verified 2026-08-11 on authoring host)

### 5.2 Already measured (runtime — narrow)

- MLX: exact Qwen2.5 0.5B 4-bit path to `measured-run-pass` (exact-source refresh packet)  
- CUDA: Phase 0–10 campaign on **one** RTX 3050; 149 planned / 58 started / 47 qualifying; six stable cells; endurance/job-control  
- MoE 30B: admission refuse on memory shortfall (not training acceptance)

### 5.3 Open relative to mission

| Gap | Mission impact |
| --- | --- |
| 0.2 unreleased; no single release-honest product promise packet | Operators cannot “just install and trust” |
| Evidence is exact-host / exact-artifact | “Works for me” still requires re-proof |
| CUDA semantic adapter reload / quality open | “Yes” is weaker than mission ideal for CUDA export |
| Correction UX incomplete as a first-class “ideal fix” loop | 2024 pain “what do I change?” not fully productized |
| Public notarization open | Cannot honestly ship Mac to strangers |
| Multi-GPU / DDP / FSDP unproven | Must stay unsupported or conditional without campaign |
| Experimental methods not executable | Correct — keep them nonselectable |

### 5.4 What this plan will **not** redo without cause

- Re-run entire CUDA Phase 0–10  
- Re-implement policy snapshot architecture  
- Expand method catalog for prestige  
- “General MoE support”

---

## 6. Phase map (program architecture)

```text
M0 Mission freeze          → charter + Path Alpha/Beta identities locked
M1 Promise audit           → gap register (evidence-linked)
M2 Trust the "no"          → refusal integrity + why surfaces
M3 Path Alpha (MLX)        → release-honest local Apple path
M4 Path Beta (CUDA)        → release-honest single-device LoRA handoff
M5 Correction loop         → ideal fix + why (KISS)
M6 Public Mac integrity    → Developer ID + notarization (if releasing publicly)
M7 One controlled expansion→ second host OR second artifact class (not both)
M8 Evaluation contract     → optional quality lane (still not "guaranteed quality")
M9 Sustain & stop list     → ops, retention, anti-drift forever
```

**Dependency rule:**  
`M0 → M1 → M2 → (M3 ∥ M4 after M2) → M5 → M6 → M7 → M8 → M9`

- M3 and M4 may be sequenced either order after M2; **default order is M3 then M4** (local path first, cheaper iteration).  
- M5 may start design notes during M3/M4 but **must not ship** until at least one of M3/M4 has a green release-evidence draft for its path.  
- M6 is required only for **public** distribution claims. Local personal use can stop at M3+M4+M5.  
- M7+ are post-promise expansion; skip if mission is already “worthy” and you choose to stop.

### 6.1 Time attitude

Duration is **not** a KPI. Integrity is.  
A phase that takes months but produces one honest path is success.  
A phase that takes a week and invents three unsupported methods is failure.

---

## 7. Phase M0 — Mission freeze

### 7.1 Goal

Lock the mission, persona, Path Alpha/Beta identities, and non-goals so later work cannot quietly redefine success.

### 7.2 In scope

- Written freeze decisions  
- Identity selection for Path Alpha and Path Beta  
- Human sign-off  

### 7.3 Out of scope

- Code changes  
- New features  
- New methods  

### 7.4 Preconditions

- [ ] Clean understanding of Section 1–4 of this document  
- [ ] Access to current capabilities + capability matrix  

### 7.5 Work packages

#### M0.1 Confirm mission text

- [ ] Re-read Section 1.2 aloud (or to a future-you note).  
- [ ] Confirm no word changes without recording a Section 12 decision.  
- [ ] Copy mission statement into a durable place you will re-read before every phase kickoff (this file is that place).

#### M0.2 Freeze Path Alpha identity (fill once; do not leave blank)

Complete this table before exiting M0. Prefer the **already measured** Qwen2.5 path unless hardware or rights block it.

| Field | Value (frozen M0 — see `.superpowers/mission-integrity-plan/M0-PATH-ALPHA-FREEZE.md`) |
| --- | --- |
| Path ID | `path-alpha-mlx-qlora-v1` |
| Training runtime | `mlx-lm` |
| Method | `qlora` |
| Placement | `single` |
| Model repo + immutable revision | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `53a32aee5e9447773fd2b85988395066aef3700a` |
| Policy identity (if any) | `model.qwen2-24l.mlx-qlora` v1.0.0 → path `mlx-lm.qlora.single.dense-causal-lm.v1` (footprint, not allowlist) |
| Dataset | `examples/support-sft.jsonl` SHA-256 `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Host class | Apple M5 Pro, arm64, 64 GiB unified (historical evidence host) |
| Python / MLX pins | Python 3.12.13; `mlx==0.31.2`; `mlx-lm==0.31.3` (as measured) |
| Success state | `measured-run-pass` with parent promotion |
| Historical anchor | source `719255153e3fc7e38e83b5ff826d587e5e58bf80`; bundle fingerprint `ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919` |
| Explicit non-claim | Not all Qwen2; not CUDA; not multi-GPU; not quality; not current-HEAD re-proof (M3 required) |

**Selection rule (KISS):** Choose the **smallest** configuration that still proves the mission story (“I checked, then it really ran”). Prefer reusing the August 2025 MLX acceptance artifact class if still available.

#### M0.3 Freeze Path Beta identity

| Field | Value (frozen M0 — see `.superpowers/mission-integrity-plan/M0-PATH-BETA-FREEZE.md`) |
| --- | --- |
| Path ID | `path-beta-cuda-lora-single-v1` |
| Training runtime | `transformers-peft-cuda` |
| Method | `lora` (BF16; not Full) |
| Placement | `single` |
| Model repo + immutable revision | `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Dataset | `examples/support-sft.jsonl` SHA-256 `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Host class | Ubuntu 24.04.4 + NVIDIA RTX 3050 (~8 GiB class) |
| Success state | `measured-run-pass` including structural PEFT export verification (semantic reload not claimed) |
| Historical anchor | source `c12c4d8db0037a2c278a2ad95a0a2cbda4387eed`; bundle fingerprint `296fb7b710f60345a590748f053eb15f9b5b4f4b3fec539ae3a705e31d6a640b` |
| Explicit non-claim | Not DDP/FSDP; not all CUDA cards; not quality; not semantic-load; not current-HEAD re-proof (M4 required) |

#### M0.4 Freeze non-goals for program duration

- [x] No DoRA / BitFit / AdaLoRA / ShareLoRA / LoReFT / AFLoRA / BiLoRA compilers  
- [x] No ROCm / CPU training  
- [x] No cloud runner product  
- [x] No MCP training authorization  
- [x] No full-training resume  
- [x] No “general MoE”  
- [x] No multi-GPU campaign unless a later Section 12 decision opens M7 multi-GPU (default: closed)

#### M0.5 Human sign-off

- [x] Owner initials/date: owner chat authorization 2026-08-11T21:20:48Z  
- [x] Path Alpha table filled and accepted  
- [x] Path Beta table filled and accepted  
- [x] Non-goals accepted  

### 7.6 Exit criteria

- [x] Sections 7.5 tables complete  
- [x] No code required  
- [x] Owner sign-off on Alpha, Beta, non-goals  
- [x] `M0-COMPLETION.md` written and STATUS advanced  
- [x] Proceed to M1 authorized by owner  

### 7.7 Stop conditions

Stop if you cannot name Path Alpha or Beta without saying “whatever models users want.” That means the mission is still too vague — re-read Section 1.

**M0 working methodology established 2026-08-11:** `.superpowers/mission-integrity-plan/PHASE-PROTOCOL.md`, SDD ledger, per-phase completion notes, anti-rush gates.

---

## 8. Phase M1 — Promise audit (forensic gap register)

### 8.1 Goal

Produce a single gap register that maps **mission promises** to **code / tests / measured evidence / docs**, so work is never driven by vibe.

### 8.2 In scope

- Read-only audit  
- Gap register document  
- Priority tagging  

### 8.3 Out of scope

- Fixes (except critical claim-language lies discovered mid-audit — then open a tiny fix PR and return)

### 8.4 Deliverable path

Create:

```text
.superpowers/mission-integrity-plan/M1-promise-audit.md
.superpowers/mission-integrity-plan/M1-gap-register.csv
```

(or `.tsv` if you prefer plain text)

### 8.5 Promise inventory (audit every row)

For each promise below, record columns:

`promise_id | promise_text | code_status | test_status | measured_status | docs_status | gap | priority | notes | evidence_links`

**Code status:** `absent | partial | present`  
**Test status:** `none | static-only | integration | e2e-sim`  
**Measured status:** `none | historical | current-head | n/a`  
**Docs status:** `missing | drift | current`  
**Priority:** `P0` false-yes risk · `P1` mission blocker · `P2` path polish · `P3` later  

#### Core promises to audit

| ID | Promise |
| --- | --- |
| P-01 | Enumerates supported methods and keeps rejects visible |
| P-02 | Separates estimate from measurement in UI and CLI |
| P-03 | Refuses unsupported method/placement with explicit reason |
| P-04 | Model policy fail-closed (no silent family allow) |
| P-05 | Plan identity changes when bound facts change |
| P-06 | Compile is atomic and no-clobber |
| P-07 | Bundle runs package-free for validation subset |
| P-08 | Ordered gates cannot be skipped for managed train |
| P-09 | Parent promotion required for measured-run-pass |
| P-10 | Stale policy requires replan, not silent accept |
| P-11 | MLX resume arguments fail closed |
| P-12 | CUDA multi-GPU not falsely claimed ready |
| P-13 | Job cancellation does not mark success |
| P-14 | Doctor does not silently install packages |
| P-15 | Claim language on README matches evidence |
| P-16 | Path Alpha can be completed by solo operator runbook |
| P-17 | Path Beta can be completed by solo operator runbook |
| P-18 | Refusal explains what fact would need to change |
| P-19 | Recommended alternative is within enumerated set only |
| P-20 | Export verification matches release-gates for path |

### 8.6 Work packages

#### M1.1 Static suite snapshot

- [ ] Run and record:

```bash
cd /path/to/Aptus
source .venv/bin/activate
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -t . -v
npm --prefix web test -- --run
.venv/bin/ruff check src tests tools
.venv/bin/python tools/verify_versions.py
```

- [ ] Save pass counts, duration, commit, date into `M1-promise-audit.md`.

#### M1.2 Walk Journey A/B on paper

- [ ] Using only docs + UI labels (no wishful thinking), write the exact click/command sequence for Path Alpha and Path Beta as **currently** possible.  
- [ ] Mark every step `works / partial / missing / unknown`.  

#### M1.3 Evidence packet index

- [ ] List every directory under `docs/operations/evidence/` with one-line claim boundary.  
- [ ] Tag which packets support Path Alpha, Path Beta, neither, or tooling-only.  

#### M1.4 False-yes hunt (highest mission risk)

For each known-bad scenario, confirm Aptus says no (or conditional with pilot required), never silent yes:

- [ ] Full FP16 request  
- [ ] Full FSDP  
- [ ] Quantized FSDP  
- [ ] MLX Full  
- [ ] CUDA profile on Mac as if local CUDA train  
- [ ] Stale plan schema load for compile/job  
- [ ] MoE near-match without topology  
- [ ] Experimental method select attempt  

Record results in gap register. Missing coverage → P0/P1 rows.

#### M1.5 Claim-language audit of public surfaces

- [ ] README  
- [ ] `docs/product/current-capabilities.md`  
- [ ] Workbench empty/example labels  
- [ ] CLI help strings for plan/run  

Flag any “guaranteed,” “optimal,” “supports X” without evidence.

### 8.7 Exit criteria

- [ ] `M1-gap-register` complete for P-01…P-20  
- [ ] Every P0/P1 has an owner phase (M2–M6) assignment  
- [ ] No P0 “false yes” left unassigned  
- [ ] Proceed to M2  

### 8.8 Stop conditions

If audit finds a live **false yes** on a public path, open a fix PR **before** any new feature work. That fix is still “M1 integrity,” not scope creep.

---

## 9. Phase M2 — Trust the “no” (refusal integrity)

### 9.1 Goal

Make refusals the most trustworthy part of the product: correct, visible, explainable, and actionable — without building AutoML.

### 9.2 Mission link

This phase is the direct answer to 2024–2025 pain: **being corrected before spend**.

### 9.3 In scope

- Reason-code inventory and operator-facing explanations  
- Gaps from M1 marked P0/P1 for refusal integrity  
- UI/CLI “why” completeness for Path Alpha/Beta planning  
- Tests that pin refusal behavior  
- Docs: troubleshooting entries for each high-frequency refuse  

### 9.4 Out of scope

- New training methods  
- New model families  
- Public notarization  
- Quality metrics  

### 9.5 Design rule (KISS)

Every operator-facing refusal must answer **three** questions only:

1. **What was refused?** (method / placement / policy / capacity / contract)  
2. **Why?** (one primary reason + optional secondary facts)  
3. **What can change the answer?** (which fact(s) to alter — or “nothing in current catalog”)

If you cannot answer (3), say so explicitly:  
`No supported correction exists in the current Aptus catalog for these facts.`

That sentence is a **feature**, not a failure.

### 9.6 Work packages

#### M2.1 Build the refusal catalog

Create `.superpowers/mission-integrity-plan/M2-refusal-catalog.md` with a table:

| reason_code | surface (plan/compile/validate/run) | user-visible title | explanation | changeable_facts | tests | docs |

- [ ] Mine codes from `docs/reference/error-codes.md`, validation findings, planning rejection reasons, model compatibility reasons.  
- [ ] Deduplicate synonyms.  
- [ ] Mark `operator_actionable: yes/no`.  

#### M2.2 Close P0 false-yes gaps from M1

For each P0:

- [ ] Write failing regression first (Python and/or web).  
- [ ] Minimal fix.  
- [ ] Update claim docs if wording was wrong.  
- [ ] Commit with subject `fix:` or `test:` only.  

#### M2.3 Compare stage “why” completeness (Path Alpha/Beta facts)

- [ ] For rejected and infeasible candidates, ensure the workbench shows reason text that maps to the catalog.  
- [ ] For conditional candidates, ensure “pilot required” is not styled as success.  
- [ ] Add/extend component tests under `web/src/components/` and stage tests.  

#### M2.4 CLI parity for refusals

- [ ] `aptus spec-plan` / plan flows print the same reason essence as API.  
- [ ] Add CLI tests if missing.  

#### M2.5 Operator troubleshooting map

- [ ] Update `docs/guides/troubleshooting.md` with the top 10 refusals from the catalog.  
- [ ] Each entry: symptom → meaning → what to change → what not to try.  

#### M2.6 Full static gate

- [ ] Full Python + web + ruff + contract checks green.  
- [ ] No claim-language regressions.  

### 9.7 Exit criteria

- [ ] Refusal catalog published in `dev/active/...`  
- [ ] All M1 P0 closed or explicitly accepted with Section 12 decision  
- [ ] P1 refusal gaps either closed or scheduled into M3/M4 with IDs  
- [ ] Solo operator can explain any Path Alpha/Beta plan refusal using UI/CLI alone  

### 9.8 Stop conditions

Do not exit M2 if the UI still shows a green success path for a candidate that is only `conditional` without pilot, or if experimental methods are selectable.

---

## 10. Phase M3 — Path Alpha release-honest (Apple Silicon MLX)

### 10.1 Goal

One **release-honest** local path: Journey A from clean install to `measured-run-pass` for the frozen Path Alpha identity, with operator runbook and dated evidence.

### 10.2 Mission link

Proves: “I can check, then run for real, on my Mac, without burning cloud money blindly.”

### 10.3 In scope

- Operator runbook for Path Alpha  
- Current-head re-proof (or explicit delta from last acceptance source)  
- Packaging/install path used by solo operators  
- Evidence packet using release-evidence template (path-scoped; overall product may remain unreleased)  
- Bugfixes required to complete the path  

### 10.4 Out of scope

- CUDA  
- Other model families  
- Notarization (unless you choose public Alpha — then also M6)  
- Quality evaluation  

### 10.5 Preconditions

- [x] M0 Path Alpha table filled  
- [x] M2 exit complete  
- [x] Apple Silicon host available  
- [x] Disk budget for model + scratch recorded  
- [x] Explicit human authorization for model download / training compute  

### 10.6 Work packages

#### M3.1 Write the operator runbook (before re-running)

Create:

```text
docs/guides/path-alpha-mlx-operator.md
```

Must include:

- [x] Hardware prerequisites  
- [x] Install Aptus (exact commands)  
- [x] Interpreter selection / doctor  
- [x] Exact model id + revision  
- [x] Exact dataset path + digest method  
- [x] Exact plan command **or** UI steps  
- [x] Expected candidate outcome (feasible/conditional)  
- [x] Compile output location rules  
- [x] Each runtime action and what success looks like  
- [x] Failure appendix with catalog reason codes  
- [x] What the runbook does **not** claim  

#### M3.2 Static preflight for Alpha

- [x] Clean venv install from current tree or wheel  
- [x] `aptus doctor` against intended interpreter  
- [x] Plan+compile dry run without train if possible  
- [x] Record bundle fingerprint  

#### M3.3 Full evidence ladder (measured)

Follow release gates Section 3 MLX path:

- [x] dependency  
- [x] model-data  
- [x] measured preflight  
- [x] uninterrupted pilot (≥2 optimizer updates)  
- [x] fresh-process adapter reload (1–4 tokens)  
- [x] confirmed full-duration train  
- [x] parent promotion → `measured-run-pass`  

Do this **twice** if gates require repeatability for the claim you want (prefer two clean repetitions for Alpha acceptance).

#### M3.4 Publish evidence packet

- [x] Copy `docs/operations/release-evidence-template.md` structure into:

```text
docs/operations/evidence/YYYY-MM-DD-path-alpha-mlx-.../README.md
```

- [x] Bind: commit, tree, host, interpreter versions, model revision, dataset digest, plan id, bundle fingerprint, job ids, digests of protected transcripts  
- [x] Independent review checklist (even if self-review + later second pass)  
- [x] Claim boundary paragraph in claim-language vocabulary  

#### M3.5 Fix only path blockers

If a step fails:

- [x] Classify: operator error / aptus bug / env bug / capacity refuse  
- [x] If aptus bug: regression test → fix → restart ladder from clean state  
- [x] If capacity refuse: document as trustworthy no; either shrink config within Path Alpha freeze or revise M0 Alpha with Section 12  

#### M3.6 Workbench path walk

- [x] Complete Journey A once via UI (not only CLI)  
- [x] Note any dead ends; fix P1 UX blockers only  

#### M3.7 Docs synchronization

- [x] Update `current-capabilities.md` **only** if the new evidence changes the normative boundary  
- [x] Update getting-started links to Path Alpha runbook  
- [x] Documentation tests pass  

### 10.7 Exit criteria

- [x] Path Alpha runbook merged  
- [x] Dated evidence packet merged with checksums  
- [x] At least one (prefer two) `measured-run-pass` at **this program’s** acceptance source  
- [x] Claim boundary states exact non-transfers  
- [x] Solo operator can follow runbook without private tribal knowledge  

### 10.8 Stop conditions

- Do not claim “Apple Silicon support” broadly.  
- Do not expand Alpha to a second model in this phase.  
- If MLX pins change, restart measured ladder; do not inherit old evidence.

---

## 11. Phase M4 — Path Beta release-honest (CUDA single-device LoRA handoff)

### 11.1 Goal

One **release-honest** CUDA path for solo operators: plan/compile on control machine, execute ordered gates on one CUDA host, complete `measured-run-pass` for frozen Path Beta identity.

### 11.2 Mission link

Proves: “Before I rent the GPU for a long job, Aptus already refused the impossible — and the possible path actually runs.”

### 11.3 In scope

- Operator runbook (Mac or any control plane → CUDA host)  
- Bundle handoff procedure  
- Clean-env dependency install proof for Beta  
- Measured ladder including release-gates CUDA requirements applicable to Beta  
- Semantic export / adapter checks **required by gates for the claim you make** (if you cannot close semantic reload, **narrow the claim** rather than fake it)  
- Evidence packet  

### 11.4 Out of scope

- DDP / FSDP  
- Multi-GPU  
- Full method matrix expansion  
- Cloud automation  

### 11.5 Preconditions

- [x] M0 Path Beta filled  
- [x] M2 complete  
- [x] CUDA host available (prefer class already campaign-proven)  
- [x] Human authorization for instance cost  
- [x] Decision: reuse campaign cell identity vs new revision (document either way)

### 11.6 Work packages

#### M4.1 Operator runbook

Create:

```text
docs/guides/path-beta-cuda-lora-operator.md
```

Must include:

- [x] How to declare CUDA hardware facts from Mac without implying Mac runs CUDA train  
- [x] Exact model + revision + dataset  
- [x] Plan/compile commands  
- [x] What to copy to the host (bundle zip, checksums)  
- [x] Host Python/CUDA/driver prerequisites  
- [x] `pip install -r requirements.txt` from bundle in clean venv  
- [x] Ordered actions and managed vs portable invocation  
- [x] How to read refuse vs pass  
- [x] Cost-control tips (stop at first failed gate)  

#### M4.2 Handoff integrity checklist

- [x] Bundle fingerprint verified on host  
- [x] Policy snapshot portable validation package-free  
- [x] Host Aptus or portable programs identity recorded  

#### M4.3 Measured ladder on host

Per release gates CUDA section for single-device LoRA:

- [x] dependency (clean env)  
- [x] model-data + trainable census  
- [x] measured preflight  
- [x] two-phase pilot with continuation observation if required  
- [x] deep admission  
- [x] full train unique run id  
- [x] structural export verification  
- [x] parent promotion  

If semantic adapter reload is still open:

- [x] Either implement+prove it for Beta **or**  
- [x] Document claim as “structural export verified; semantic reload not claimed” and ensure UI/docs match  

#### M4.4 Job-control smoke on host

- [x] Cancellation does not report success  
- [x] Lease behavior documented  
- [x] Record results in evidence packet  

#### M4.5 Evidence packet

```text
docs/operations/evidence/YYYY-MM-DD-path-beta-cuda-lora-.../
```

- [x] Bind host inventory, driver, torch, commit, digests  
- [x] Map to release-gates rows explicitly (pass/not run/fail)  
- [x] Independent recompute of key counters if campaign-style  

#### M4.6 Cross-surface honesty

- [x] Mac UI never shows CUDA train as local-complete without host evidence  
- [x] README “what runs where” still accurate  

### 11.7 Exit criteria

- [x] Beta runbook merged  
- [x] Evidence packet merged  
- [x] `measured-run-pass` for frozen Beta identity at acceptance source  
- [x] Clean-env dependency install recorded  
- [x] Claim boundary honest about semantic reload / quality / host class  

### 11.8 Stop conditions

- Do not widen to Full/int8/QLoRA matrix in this phase.  
- Do not claim multi-GPU.  
- If host OOM at pilot: treat as success of **no** if planner should have refused — file P0 if planner said feasible without pilot caveat; otherwise document measured refuse.

---

## 12. Phase M5 — Correction loop (“ideal fix and why”)

### 12.1 Goal

When Aptus says no (or conditional), the solo operator gets a **simple, justified next action** — not a research paper and not a hyperparameter search.

### 12.2 Mission link

This is the “tell me straight and produce the ideal fix and why” clause of the origin story.

### 12.3 KISS product definition

**One primary correction** per plan result:

| Situation | Product behavior |
| --- | --- |
| ≥1 feasible candidate | Recommend top-ranked feasible; explain ranking objective |
| 0 feasible, ≥1 conditional | Recommend top conditional; explain pilot requirement |
| 0 viable | State no supported path; list top blocking reasons; suggest which fact classes could change (VRAM, sequence, batch, method) **without inventing unsupported methods** |

Never invent a 5th training method as a “fix.”

### 12.4 In scope

- Server-side structured “correction” object (or equivalent existing fields normalized)  
- UI panel + CLI section  
- Tests  
- Docs  

### 12.5 Out of scope

- Optuna / Bayesian search  
- Automatic continuous replan loops  
- Cloud cost estimators beyond local arithmetic already present  

### 12.6 Suggested contract shape (implement only after reading current API)

Prefer extending existing plan response rather than new microservices.

Illustrative fields (adjust names to match repo style during implementation):

```json
{
  "correction": {
    "kind": "select-candidate | replan-with-fact-hints | no-path",
    "summary": "string",
    "primary_reason_codes": ["..."],
    "recommended_candidate_id": "string|null",
    "fact_hints": [
      {"fact": "sequence_length", "direction": "decrease", "why": "..."}
    ],
    "disallowed_suggestions": [
      "Do not enable full FSDP; unsupported in v0.2"
    ]
  }
}
```

Rules:

- Hints must not suggest unsupported contracts.  
- Hints must be derived from actual rejection reasons, not LLM free text in v1.  
- If you later add LLM wording, it may **only paraphrase** structured hints, never invent gates.

### 12.7 Work packages

#### M5.1 Spec freeze

- [ ] Write `.superpowers/mission-integrity-plan/M5-correction-spec.md` with exact fields and examples for Alpha/Beta.  
- [ ] Human approve.  

#### M5.2 TDD API

- [ ] Failing API tests for feasible / conditional / no-path cases  
- [ ] Implement minimal  
- [ ] OpenAPI regenerate + client checks  
- [ ] Web types/normalizers  

#### M5.3 UI

- [ ] Compare stage shows correction summary  
- [ ] One button: “Apply recommended candidate” **or** “Show what to change”  
- [ ] No second navigation maze  

#### M5.4 CLI

- [ ] Print correction block in `spec-plan` / plan output  

#### M5.5 Docs

- [ ] Guide section “When Aptus refuses”  
- [ ] Claim language: correction is not optimality  

#### M5.6 Evidence

- [ ] Static suite green  
- [ ] Manual Alpha/Beta walk showing correction on intentional bad batch/sequence  

### 12.8 Exit criteria

- [ ] Structured correction available on plan and no-feasible paths  
- [ ] UI + CLI expose it  
- [ ] Tests pin “no unsupported suggestion”  
- [ ] Mission story complete for “why + fix” at KISS level  

---

## 13. Phase M6 — Public Mac distribution integrity (optional for private use)

### 13.1 Goal

If (and only if) you want strangers to download Aptus for Mac, close packaging trust gates.

### 13.2 In scope

- Developer ID Application signing  
- Notarization + stapling  
- Gatekeeper assessment  
- Release evidence rows for desktop  
- SHA256SUMS + COMMIT binding  

### 13.3 Out of scope

- App Store submission (unless separate decision)  
- Windows  

### 13.4 Work packages

- [ ] Obtain/confirm signing identity and notary profile  
- [ ] `APTUS_REQUIRE_CLEAN_CHECKOUT=1` release build  
- [ ] `tools/repeat_desktop_release_gate.zsh` if required by gates  
- [ ] Notarize app + DMG  
- [ ] Record in dated release evidence  
- [ ] Update README distribution claims  

### 13.5 Exit criteria

- [ ] Gatekeeper-clean public artifact **or** explicit decision to remain private/ad-hoc only  

### 13.6 Stop conditions

Never call ad-hoc DMG a public release.

---

## 14. Phase M7 — One controlled expansion

### 14.1 Goal

After Alpha+Beta+correction, expand **one axis only**.

### 14.2 Choose exactly one (Section 12 decision)

| Option | Expands | Forbidden simultaneous with |
| --- | --- | --- |
| M7-A | Second **model artifact** on same runtime as Alpha or Beta | Second host |
| M7-B | Second **host class** for same Beta cell | Second model |
| M7-C | Semantic CUDA adapter reload proof for Beta (if deferred) | New method |

### 14.3 Process (same every time)

1. Freeze identity table (like M0)  
2. Write runbook delta  
3. Measured ladder  
4. Evidence packet  
5. Update capability docs with non-transfer language  
6. Stop  

### 14.4 Exit criteria

- [ ] One new evidence-bound claim, no silent generalization  

---

## 15. Phase M8 — Evaluation contract (optional, still honest)

### 15.1 Goal

Allow “training finished” and “meets evaluation target” to be distinct, first-class states — without claiming general quality.

### 15.2 In scope

- Eval dataset binding  
- Metric + threshold fields  
- Post-train eval job optional  
- Evidence binding to export artifact  

### 15.3 Out of scope

- Leaderboards  
- Automatic safety red-team as product default  
- Claiming human preference quality without human labels  

### 15.4 Exit criteria

- [ ] Operator can attach an eval contract  
- [ ] UI never equates loss curves with eval pass  
- [ ] Docs use claim language  

---

## 16. Phase M9 — Sustain and anti-drift forever

### 16.1 Goal

Keep the mission intact after success.

### 16.2 Standing checklist (every PR)

- [ ] Does this make false-yes more or less likely?  
- [ ] Does claim language still match evidence?  
- [ ] Did we hide a rejection?  
- [ ] Did we add a method without compiler + gates?  
- [ ] Did we expand a claim without a packet?  

### 16.3 Standing stop list (do not build without new program)

- Universal method recommendation across undocumented models  
- Silent dependency installation  
- Resume for MLX without full state contract  
- Remote multi-user job service without auth boundary  
- “AI agent trains for you” that bypasses evidence ladder  

### 16.4 Retention

- Follow `docs/operations/state-storage-retention.md`  
- Keep private raw logs out of git; store digests in public packets  

---

## 17. Cross-cutting engineering protocol (every implementation phase)

### 17.1 Branching

```text
main (clean)
  └── mission/mN-<short-name>
```

One phase work package per PR when possible.

### 17.2 Required local gate before claiming a work package done

```bash
source .venv/bin/activate
.venv/bin/ruff format --check src/aptus tests/aptus
.venv/bin/ruff check src tests tools
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -t . -v
npm --prefix web test -- --run
npm --prefix web run typecheck
.venv/bin/python tools/generate_openapi.py --check
npm --prefix web run openapi:check
.venv/bin/python tools/check_client_contracts.py
.venv/bin/python tools/verify_versions.py
git diff --check
```

Desktop gate when host/bridge/workbench packaging touched:

```bash
desktop/macos/build.sh
```

### 17.3 Runtime-affecting changes

If generation, memory math, precision, quantization, checkpoint/snapshot, reload, or export changes:

- [ ] Run real pilot on **every affected** runtime (CUDA and/or MLX) per project rules  
- [ ] New or updated evidence packet  

### 17.4 Commit style

Short imperative subjects: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`  
PR body must list verification commands and claim boundary impact.

### 17.5 Documentation policy

- Update user/API/bundle/capability docs in the same change  
- Never rewrite historical evidence packets; add new dated ones  
- `tests/aptus/test_documentation.py` must stay green  

---

## 18. Risk register (mission-specific)

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Feature envy (more methods) | Dilutes trust | I11 + M0 non-goals |
| Softened refusals for demos | Recreates 2024 pain | M2 catalog + claim audit |
| Evidence rot (old commits) | False confidence | Current-head re-proof for Alpha/Beta |
| Scope coupling (Alpha+Beta+M7 together) | Never finishes | Hard phase exits |
| LLM-written “fixes” inventing gates | False hope | M5 structured hints only |
| Giant refactors mid-path | Stalls mission | Behavior-change-only rule |
| Public overclaim on CUDA campaign | Trust collapse | Exact-host language forever |

---

## 19. Decision log template (Section 12 decisions)

Append-only in:

```text
.superpowers/mission-integrity-plan/DECISIONS.md
```

```markdown
## DECISION-YYYYMMDD-NN
- Date:
- Phase:
- Question:
- Options:
- Choice:
- Mission justification (which invariant/outcome):
- What we explicitly will not do because of this:
- Evidence / links:
- Owner:
```

---

## 20. Minimal “start tomorrow” checklist

If you want the shortest honest start:

1. [ ] Read Sections 1–4 of this document  
2. [ ] Execute Phase M0 tables (fill Alpha/Beta)  
3. [ ] Execute Phase M1 audit into `.superpowers/mission-integrity-plan/`  
4. [ ] Close any P0 false-yes before anything fun  
5. [ ] Enter M2  

Do **not** start a new method, MoE training, or notarization sprint before M0–M2.

---

## 21. Definition of program complete

This program is complete when:

1. Mission statement still matches product behavior.  
2. Path Alpha exit criteria met.  
3. Path Beta exit criteria met.  
4. M5 correction loop shipped at KISS level.  
5. Public surfaces pass claim-language audit.  
6. Owner can say without flinching:

> “If Aptus refuses your fine-tune, believe it. If it accepts a path, that path has been proven for the exact identity we document — not for the entire universe of models.”

M6–M8 may remain open without voiding program completion **if** the product is positioned as private/engineering-supported rather than public consumer download.

---

## 22. Self-review of this plan (spec coverage)

| Mission element | Covered in |
| --- | --- |
| Pre-spend check | M1–M4, Journey A/B |
| Trust when it says no | M2, invariants I2–I3, I9 |
| Why + ideal fix | M5 |
| Real run after check | M3, M4 |
| Evidence / audit trail | M3.4, M4.5, release template |
| Save money/time/energy | Persona metrics; stop at first failed gate |
| KISS | Section 3; one correction; two paths; no method sprawl |
| Integrity over speed | Time attitude; phase exits; stop lists |
| Fail-closed | Invariants; claim language; non-goals |

No intentional placeholders for “figure out later” on mission structure. Path Alpha/Beta **identity values** are intentionally blank tables for the owner to freeze in M0 — that is a required human decision, not plan incompleteness.

---

## 23. Related files to create during execution

| Path | Phase |
| --- | --- |
| `.superpowers/mission-integrity-plan/M1-promise-audit.md` | M1 |
| `.superpowers/mission-integrity-plan/M1-gap-register.csv` | M1 |
| `.superpowers/mission-integrity-plan/M2-refusal-catalog.md` | M2 |
| `.superpowers/mission-integrity-plan/M5-correction-spec.md` | M5 |
| `.superpowers/mission-integrity-plan/DECISIONS.md` | ongoing |
| `docs/guides/path-alpha-mlx-operator.md` | M3 |
| `docs/guides/path-beta-cuda-lora-operator.md` | M4 |
| `docs/operations/evidence/YYYY-MM-DD-path-alpha-.../` | M3 |
| `docs/operations/evidence/YYYY-MM-DD-path-beta-.../` | M4 |

---

**End of plan.**  
Next human action: execute **Phase M0** only.
