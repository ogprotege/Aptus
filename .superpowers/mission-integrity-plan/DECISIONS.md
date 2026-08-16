# Mission integrity — decision log

Append-only. Every product choice that freezes scope or identity gets an entry.

---

## DECISION-20260816-01

- **Date:** 2026-08-16
- **Phase:** TP0
- **Question:** How does Aptus treat &lt;100-row datasets and `max_epochs` &gt; 3 without breaking Path Alpha or becoming AutoML?
- **Options:** (a) infeasible below 100 rows always (b) conditional below 100 when epochs≤3; infeasible only for the long-train interactions (c) warn-only, never change status
- **Choice:** (b) — owner approved 2026-08-16.
- **Mission justification:** Fail-closed on “train longer on too little data.” Keep the 4-row Path Alpha proof runnable as conditional, not a silent yes and not a destroyed identity.
- **Explicitly will not do:** Rewrite `max_epochs`; invent rows; Optuna; quality badge from loss; `weight_decay` 0.01 as a sycophancy fix; call this M10.
- **Evidence / links:** `docs/superpowers/plans/2026-08-16-training-policy-and-run-correction.md`; `TP0-training-policy-spec.md`; `examples/support-sft.jsonl` is 4 rows.
- **Owner:** Wilson (approved option b in session, 2026-08-16)

Template:

```markdown
## DECISION-YYYYMMDD-NN
- Date:
- Phase:
- Question:
- Options:
- Choice:
- Mission justification:
- Explicitly will not do:
- Evidence / links:
- Owner:
```

---

## DECISION-20260813-04

- **Date:** 2026-08-13
- **Phase:** M9
- **Question:** What does executing M9 mean after M0–M8?
- **Options:** (a) re-prove Path Alpha/Beta at live HEAD (b) audit what shipped and make the standing checklist durable (c) skip
- **Choice:** (b) — owner asked for a check on work already done. No new method, host, or measured ladder.
- **Mission justification:** Evidence rot is named in the risk register. Rewriting “current HEAD” as recorded source is honest. Re-running training to relabel HEAD is a new measured program, not M9.
- **Explicitly will not do:** Invent a current-HEAD transfer; reopen M7-B; add methods; treat eval pass as quality.
- **Evidence / links:** `M9-AUDIT.md`
- **Owner:** Wilson (authorized M9 as an audit)

---

## DECISION-20260813-03

- **Date:** 2026-08-13
- **Phase:** M8
- **Question:** What is the first evaluation contract, and does scoring require GPU generation?
- **Options:** (a) GPU eval job that generates then scores (b) operator-supplied predictions + deterministic exact-match (c) LLM-as-judge
- **Choice:** (b) — `aptus.evaluation-contract.v1` / `aptus.evaluation-result.v1` with `exact_match` only. Optional post-train scoring, not a JobService ladder step, not `plan_id` material.
- **Mission justification:** Training finished must stay distinct from eval pass. Generating tokens would be runtime-affecting and would invite quality claims Aptus cannot support. Exact match is checkable and fail-closed.
- **Explicitly will not do:** Leaderboards; default red-team; human-preference claims without labels; treat split `evaluation_fraction` or train loss as this decision.
- **Evidence / links:** `docs/product/mission-integrity-plan.md` §15; `M8-eval-spec.md`
- **Owner:** Wilson (authorized M8 after M7 merge green)

---

## DECISION-20260813-02

- **Date:** 2026-08-13
- **Phase:** M7-B
- **Question:** How to finish M7 with no second CUDA instance?
- **Options:** (a) relabel another Sherminator 135M run as M7-B (b) skip M7-B (c) wait indefinitely for another GPU box
- **Choice:** (b) — owner has no other CUDA instance. M7 closes as **A + C only**. B stays an explicit non-claim, not a hidden pass.
- **Mission justification:** A same-host repeat is not a second host class. Skipping is honest; inventing a transfer claim is not.
- **Explicitly will not do:** Call Sherminator a second host; transfer Path Beta to other GPUs; leave B “open” as if it were still in progress.
- **Evidence / links:** owner message 2026-08-13; only CUDA SSH target `wts@192.168.1.12`
- **Owner:** Wilson

---

## DECISION-20260813-01

- **Date:** 2026-08-13
- **Phase:** M7-C
- **Question:** Which one M7 axis first, and what does “semantic CUDA adapter reload” mean?
- **Options:** (a) M7-A second model (b) M7-B second host (c) M7-C reload (d) all three as one mixed claim
- **Choice:** (c) first. Owner wants A/B/C sequenced later. Reload = fresh-process PEFT load of the Path Beta adapter + 1–4 generated tokens, schema `aptus.cuda-reload-evidence.v1`. CUDA `measured-run-pass` stays structural-export until a later explicit contract bump.
- **Mission justification:** Closes the named CUDA honesty gap without widening model or host claims.
- **Explicitly will not do:** M7-A or M7-B in this packet; treat reload as quality; require reload for all historical CUDA campaign cells.
- **Evidence / links:** `M7-C-IDENTITY-FREEZE.md`
- **Owner:** Wilson (boot Sherminator + “you can begin”)

---

## DECISION-20260811-01

- **Date:** 2026-08-11
- **Phase:** M0
- **Question:** How will mission phases be executed so work stays current and cannot rush?
- **Options:** (a) ad-hoc chat only (b) written phase protocol + SDD ledger + completion notes + STATUS
- **Choice:** (b) — `PHASE-PROTOCOL.md`, SDD ledger under `.superpowers/sdd/2026-08-11-mission-trust-when-it-says-no/`, per-phase `M{N}-COMPLETION.md`, live `STATUS.md`
- **Mission justification:** Trust-when-it-says-no requires process integrity equal to product integrity; anti-rush gates prevent false phase completion
- **Explicitly will not do:** Mark phases complete from chat memory alone; start phase N+1 without completion note + owner gates where required
- **Evidence / links:** `.superpowers/mission-integrity-plan/PHASE-PROTOCOL.md`
- **Owner:** program (established with human request for M0 subagent-driven methodology)

---

## DECISION-20260811-02

- **Date:** 2026-08-11
- **Phase:** M0
- **Question:** What is Path Alpha (first release-honest local Apple path identity)?
- **Options:** (a) already-measured MLX Qwen2.5-0.5B 4-bit QLoRA single (b) invent a new Alpha model (c) leave blank
- **Choice:** (a) — identity in `M0-PATH-ALPHA-FREEZE.md`  
  Model `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `53a32aee5e9447773fd2b85988395066aef3700a`; method `qlora`; placement `single`; dataset `examples/support-sft.jsonl` digest `bf2dca3d…`; historical source `71925515…`; bundle fingerprint `ca2548cf…`
- **Mission justification:** Depth-first; reuses two clean historical `measured-run-pass` ladders; smallest honest local path
- **Explicitly will not do:** All Qwen2 / all Apple Silicon; CUDA; multi-GPU; quality; claiming M3 done; treating config footprint as allowlist
- **Evidence / links:** `M0-PATH-ALPHA-FREEZE.md`; `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/`; independent review `task-M0-review.md` Spec PASS
- **Owner:** accepted by owner 2026-08-11 (chat authorization)

---

## DECISION-20260811-03

- **Date:** 2026-08-11
- **Phase:** M0
- **Question:** What is Path Beta (first release-honest CUDA single LoRA handoff identity)?
- **Options:** (a) SmolLM2-135M LoRA single BF16 from 2026-08-06 acceptance (b) Phase 5 cohort as primary (c) Full method primary (d) invent new model
- **Choice:** (a) — identity in `M0-PATH-BETA-FREEZE.md`  
  Model `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac`; method `lora` BF16; placement `single`; same support-sft digest; host class Ubuntu RTX 3050; historical source `c12c4d8d…`; bundle fingerprint `296fb7b7…`  
  Phase 5 / Phase 10 are **supporting only**, not merged
- **Mission justification:** Prefer exact `measured-run-pass` five-job ladder over campaign-only cells; LoRA single matches solo-operator first CUDA path
- **Explicitly will not do:** multi-GPU; Full as primary; semantic adapter reload claim; quality; all CUDA cards; current-HEAD pass at M0
- **Evidence / links:** `M0-PATH-BETA-FREEZE.md`; `docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/`; review Spec PASS
- **Owner:** accepted by owner 2026-08-11 (chat authorization)

---

## DECISION-20260811-04

- **Date:** 2026-08-11
- **Phase:** M0
- **Question:** What non-goals bind the program until a later decision?
- **Options:** soft list in chat vs written freeze
- **Choice:** Written freeze `M0-NONGOALS-FREEZE.md` (NG-01…NG-10) including no experimental compilers, no multi-GPU by default, no cloud/MCP training auth, no resume, no general MoE, no quality guarantees, no silent installs
- **Mission justification:** KISS + prevent 2024-style hope features diluting trustworthy no
- **Explicitly will not do:** Open NG-* items without a new DECISION entry
- **Evidence / links:** `M0-NONGOALS-FREEZE.md`
- **Owner:** accepted by owner 2026-08-11 (chat authorization)

---

## DECISION-20260812-01

- **Date:** 2026-08-12
- **Phase:** M2
- **Question:** How should structured refusal guidance attach without breaking plan identity?
- **Options:** (a) embed in plan/candidate identity payload (b) presentation-only module + CLI stderr + web mapping
- **Choice:** (b) — `aptus.refusal` + web `lib/refusal.ts`; plan JSON digests unchanged
- **Mission justification:** Trust-when-no without inventing new plan schema churn mid-path
- **Explicitly will not do:** OpenAPI refusal_guidance field in M2; AutoML “fix” generators
- **Evidence / links:** `src/aptus/refusal.py`, `M2-COMPLETION.md`
- **Owner:** program execution 2026-08-12
