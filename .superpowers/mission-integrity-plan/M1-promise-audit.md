# M1 — Promise audit (executive summary)

> **Status:** Complete forensic audit deliverable (M1 open until protocol exit + owner gates)  
> **Phase:** M1 Promise audit  
> **Date:** 2026-08-11  
> **Authority packet:** `docs/product/mission-integrity-plan.md` §8  
> **Register:** [`M1-gap-register.csv`](./M1-gap-register.csv)  
> **Baseline HEAD at audit:** program STATUS points at `7fcdf161224ee0d4b75285b6c7d664b17af53df5` (M0 completion); this audit is documentation-only  

## 1. Methodology

Read-only forensic mapping of mission promises **P-01…P-20** to four evidence classes:

| Class | Sources used |
| --- | --- |
| **Code** | `src/aptus/` planner, plan contract, generation, execution, diagnostics, model compatibility, bundle programs under `src/aptus/_bundle_programs/` |
| **Tests** | `tests/aptus/` (unittest), `web/src/**/*.test.ts(x)` where ingress/UI contracts bind promises |
| **Measured** | `docs/operations/evidence/**` plus M0 freezes (identity anchors only — never treated as current-HEAD re-proof) |
| **Docs** | `docs/product/current-capabilities.md`, `docs/reference/capability-matrix.md`, `docs/product/claim-language.md`, `docs/operations/release-gates.md`, getting-started/guides/operator docs, `README.md`, M0 freezes |

**Conservative scoring rules (applied):**

- Between `present` and `partial` for code → prefer **partial**.
- Measured only at a past commit/packet → **historical**, never `current-head`.
- Fail-closed behavior without runtime training → measured **n/a** or **none** as appropriate.
- Priority: **P0** = live false-yes risk; **P1** = mission blocker for trust-when-no or release-honest yes; **P2** = polish / already-strong residual; **P3** = later.
- Every **P0/P1** row has `owner_phase` in **M2–M6** (or **M9** only for sustain claim-language anti-drift).

Also walked M1.4 **false-yes hunt** scenarios against tests/docs (Full FP16, Full FSDP, quantized FSDP, MLX Full, multi-GPU on single GPU, stale plan, MoE near-match, experimental method select). All appear **fail-closed with explicit reasons** in planner/API tests; **no live public false-yes** was found that requires an emergency fix PR before M2.

### M1.3 Evidence packet index (claim boundaries)

| Packet | Claim boundary (one line) | Path support |
| --- | --- | --- |
| `2026-07-27-mlx-lm-acceptance/` | Historical v2 plan/bundle MLX ladder for pinned Qwen2.5 | Alpha historical only |
| `2026-07-27-desktop-release/` | Desktop engineering build gate (not training proof) | Tooling |
| `2026-07-28-qwen3-moe-admission/` | MoE admission refuse (memory envelope); not training acceptance | Neither path |
| `2026-07-29-documentation-drift-audit/` | Docs drift audit | Tooling |
| `2026-08-05-qwen2-mlx-lm-acceptance/` | Historical Phase 6 baseline MLX `measured-run-pass` | Alpha baseline |
| `2026-08-05-qwen2-mlx-lm-exact-source-refresh/` | **M0 Path Alpha identity freeze** — two clean v5/v3 ladders at exact source `71925515…` / fingerprint `ca2548cf…` | **Alpha primary historical** |
| `2026-08-06-smollm2-cuda-lora-single-acceptance/` | **M0 Path Beta identity freeze** — one SmolLM2 LoRA single five-job `measured-run-pass` | **Beta primary historical** |
| `2026-08-09-cuda-phase0-recovery-supplement/` | Campaign recovery publication hygiene | CUDA campaign / neither path product |
| `2026-08-09/10-cuda-phase5-repeatability-*` | Exact-host SmolLM2 LoRA repeatability (supporting; different source/dataset) | Beta supporting |
| `2026-08-10-cuda-phase6-*` | Method matrix / remediation / confirmatory Full cell (immutable cohorts) | Neither (campaign) |
| `2026-08-10-cuda-phase7-scale-staircase/` | Stopped scale staircase history | Neither |
| `2026-08-11-cuda-phase7-*` | Same-family + breadth stability; Phase 7 complete | Neither (campaign) |
| `2026-08-11-cuda-phase8-guarded-frontier/` | Probe-only frontier | Neither |
| `2026-08-11-cuda-phase9-endurance/` | Endurance + job-control exercises | Supporting job integrity (P-13) |
| `2026-08-11-cuda-phase10-certification/` | Campaign aggregation; **not** release readiness / multi-GPU / quality | Supporting claim boundary only |

## 2. Suite results (M1.1)

Recorded on controller host after documentation governance fix for the mission plan.

| Gate | Result |
| --- | --- |
| Python `unittest discover` | **946 OK** in 49.814s (2026-08-11T21:30:04Z) |
| Web Vitest | **130 OK** (20 files) |
| Ruff | All checks passed |
| `tools/verify_versions.py` | Aptus version surfaces agree on **0.2.0** |
| Commit | `7fcdf161224ee0d4b75285b6c7d664b17af53df5` |
| Interpreter | Python 3.12.13 |

**First suite run during M1** failed 4 documentation tests because mission working notes lived under `dev/active/` (forbidden) and new product docs lacked inventory/metadata. **Fix applied in M1:** workspace moved to `.superpowers/mission-integrity-plan/`, product plan metadata + inventory counts updated, pointer under `docs/superpowers/` removed. Suite re-run: **green**. This was documentation governance integrity, not a training false-yes.

**P-15 owner_phase note:** assigned **M9** (claim-language sustain) rather than M2–M6; recorded as intentional exception at M1 completion (anti-drift forever, not a path-proof gap).

## 3. Register headline (P-01…P-20)

| ID | Priority | code | test | measured | docs | owner |
| --- | --- | --- | --- | --- | --- | --- |
| P-01 Enumerate + visible rejects | P2 | present | integration | n/a | current | M2 |
| P-02 Estimate ≠ measurement | P2 | present | integration | n/a | current | M2 |
| P-03 Explicit refuse method/placement | P1 | present | integration | n/a | current | M2 |
| P-04 Model policy fail-closed | P2 | present | integration | historical | current | none |
| P-05 Plan identity on fact change | P2 | present | integration | n/a | current | none |
| P-06 Atomic no-clobber compile | P2 | present | integration | n/a | current | none |
| P-07 Package-free validation subset | P2 | present | e2e-sim | historical | current | none |
| P-08 Ordered managed gates | P1 | present | integration | historical | current | M3 |
| P-09 Parent promotion for pass | P1 | present | integration | historical | current | M3 |
| P-10 Stale policy → replan | P2 | present | integration | n/a | current | none |
| P-11 MLX resume fail-closed | P2 | present | integration | n/a | current | none |
| P-12 Multi-GPU not claimed ready | P1 | present | integration | none | current | M2 |
| P-13 Cancel ≠ success | P2 | present | integration | historical | current | none |
| P-14 Doctor no silent install | P2 | present | integration | n/a | current | none |
| P-15 README claim language | P1 | present | static-only | historical | current | M9 |
| P-16 Path Alpha solo runbook | P1 | partial | static-only | historical | partial | M3 |
| P-17 Path Beta solo runbook | P1 | partial | static-only | historical | partial | M4 |
| P-18 Refusal changeable facts | P1 | partial | static-only | n/a | drift | M2 |
| P-19 Recommended ⊆ enumerated | P2 | present | integration | n/a | current | M5 |
| P-20 Export vs release-gates | P1 | present | integration | historical | current | M3 |

Full columns and evidence links: [`M1-gap-register.csv`](./M1-gap-register.csv).

## 4. Top P0 / P1 list

### P0 — false-yes risk

**None assigned.**  

False-yes hunt (M1.4) found planner/API fail-closed coverage for known-bad combinations; README/capability-matrix claim language matches evidence boundaries. Residual **styling** risk (conditional or planner-supported looking “ready”) is tracked as **P1 under M2** (P-12, P-03/P-18), not as a confirmed live false yes.

If a later static suite run or UI walk discovers a green path for unsupported work, open a fix PR as M1 integrity before M2 feature work (per plan §8.8).

### P1 — mission blockers (ordered for program)

| Rank | ID | Why it blocks the mission | Owner |
| ---: | --- | --- | --- |
| 1 | **P-18** | Operators still lack structured “what fact must change” on every refuse — core 2024 pain | **M2** |
| 2 | **P-03** | Explicit reasons exist but need catalog + UI/CLI completeness for Path Alpha/Beta facts | **M2** |
| 3 | **P-12** | Dual vocabulary (planner-supported vs runtime-qualified) can still mislead on multi-GPU | **M2** |
| 4 | **P-16** | No identity-bound Alpha solo runbook + no current-HEAD `measured-run-pass` | **M3** |
| 5 | **P-08 / P-09 / P-20** | Mechanisms present; path-scoped current-HEAD ordered ladder, parent promotion, export closeout required for release-honest yes | **M3** (Beta export/ladder re-proof in **M4**) |
| 6 | **P-17** | No identity-bound Beta handoff runbook + no current-HEAD re-proof on host class | **M4** |
| 7 | **P-15** | No immediate lie found; anti-drift is permanent mission risk after path packets land | **M9** |

## 5. Phase assignment summary

| Phase | Owns (from this register) | Mission intent |
| --- | --- | --- |
| **M2** | P-01 (why polish), P-02 (CLI parity), **P-03**, **P-12**, **P-18** | Trust the “no”: refusal catalog, actionable change facts, no false-ready styling |
| **M3** | **P-08**, **P-09**, **P-16**, **P-20** (Alpha export/reload) | Path Alpha release-honest re-proof + solo runbook |
| **M4** | **P-17** (+ Beta re-proof of P-08/P-09/P-20 structural export) | Path Beta release-honest handoff |
| **M5** | P-19 (correction UX within enumerated set) | One-correction loop; no AutoML |
| **M6** | *(none from P-01…P-20)* | Public notarization only if shipping Mac publicly |
| **M7** | *(none; multi-GPU remains NG-07 unless DECISION)* | Controlled expansion only after DECISION |
| **M8** | *(none; quality still non-goal as guarantee)* | Optional evaluation contract |
| **M9** | **P-15** | Claim-language sustain / anti-drift |
| **none** | P-04, P-05, P-06, P-07, P-10, P-11, P-13, P-14 | Implemented fail-closed foundations; keep green, no dedicated phase work |

**Count:** 0×P0 · 9×P1 · 11×P2 · 0×P3 · 0×`accepted` residual false-yes.

## 6. Journey paper-walk (M1.2 summary)

### Path Alpha (frozen `path-alpha-mlx-qlora-v1`)

| Step | Status |
| --- | --- |
| Install Aptus / open Mac app or workbench | **works** (docs + packaging) |
| Doctor / choose MLX Python (no silent install) | **works** (P-14) |
| Pin frozen model revision + `examples/support-sft.jsonl` | **partial** (generic docs; identity pins only in M0 freeze/evidence not a solo Alpha runbook) |
| Plan → see rejects + QLoRA single recommendation | **works** (P-01/P-03/P-19) |
| Compile no-clobber | **works** (P-06) |
| Ordered gates → pilot → confirmed train → parent promotion | **partial** (code/tests present; **measured historical only**) |
| Identity-bound current-HEAD evidence packet | **missing** (M3) |

### Path Beta (frozen `path-beta-cuda-lora-single-v1`)

| Step | Status |
| --- | --- |
| Plan/compile CUDA LoRA single on control plane | **works** |
| Transfer bundle + clean-env `requirements.txt` install | **partial** (generic guide; no identity-bound handoff runbook) |
| Five ordered managed jobs on RTX 3050-class host | **partial** (historical acceptance; not HEAD) |
| Structural PEFT export + parent promotion | **partial** (historical; semantic reload **explicit non-claim**) |
| Solo operator complete from cold start using only docs | **missing** as Alpha/Beta-identity promise (M4) |

## 7. Claim-language audit (M1.5) — brief

| Surface | Finding |
| --- | --- |
| `README.md` | Matches evidence: enumerated recommendation, exact packets, multi-GPU open, unreleased 0.2. **No fix required.** |
| `docs/product/current-capabilities.md` | Normative, evidence-linked, last reviewed 2026-08-11. **Current.** |
| Workbench labels | Labeled-example demo data; policy UI fail-closed at ingress (server-authoritative). Residual: conditional styling (M2). |
| CLI plan/run help | Doctor probe-only language correct; refusal actionability weaker than UI (M2). |

Forbidden words from `claim-language.md` (“guaranteed to fit”, “optimal”, “supports all CUDA”) were **not** found as live product claims on README primary surfaces during this audit.

## 8. Exit criteria checklist (for controller)

- [x] `M1-gap-register.csv` complete for P-01…P-20 with required columns  
- [x] Every P0/P1 has owner_phase (none P0; all P1 assigned M2–M4 or M9)  
- [x] No confirmed P0 false-yes left unassigned  
- [ ] M1.1 live suite snapshot recorded (placeholder above — still open)  
- [ ] Independent review + `M1-COMPLETION.md` + STATUS advance (controller/protocol; not this implementer task)

## 9. Explicit non-claims of this audit

- Does **not** re-prove Path Alpha or Path Beta at current HEAD.  
- Does **not** authorize M2 implementation or measured runs.  
- Does **not** amend M0 freezes or non-goals.  
- Does **not** mark Aptus 0.2 released or release-ready.  
- Suite pass counts in §5.1 of the program plan are **prior baseline**, not this session’s measurement.

## 10. Sources consulted

1. `docs/product/mission-integrity-plan.md` §8  
2. `docs/product/current-capabilities.md`  
3. `docs/reference/capability-matrix.md`  
4. `docs/product/claim-language.md`  
5. `docs/operations/release-gates.md`  
6. `.superpowers/mission-integrity-plan/M0-PATH-ALPHA-FREEZE.md`  
7. `.superpowers/mission-integrity-plan/M0-PATH-BETA-FREEZE.md`  
8. `.superpowers/mission-integrity-plan/M0-NONGOALS-FREEZE.md`  
9. `README.md`  
10. `docs/getting-started/choose-your-path.md`, `docs/guides/compile-validate-run.md`, `docs/operations/operator-checklist.md`, `docs/guides/troubleshooting.md`  
11. Targeted code/tests under `src/aptus/` and `tests/aptus/` as linked in the CSV  
12. Evidence directories under `docs/operations/evidence/` (index in §1)
