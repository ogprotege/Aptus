# Mission integrity — decision log

Append-only. Every product choice that freezes scope or identity gets an entry.

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
