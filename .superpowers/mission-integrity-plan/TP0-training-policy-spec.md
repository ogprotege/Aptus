# TP0 — Training policy specification (freeze)

> **Status:** APPROVED 2026-08-16 — owner chose DECISION-20260816-01 option (b)
> **Authority:** Subordinate to claim language, current capabilities, and mission invariants I1–I12  
> **Increment:** Training Policy (TP). **Not M10.**  
> **Implementation plan:** `docs/superpowers/plans/2026-08-16-training-policy-and-run-correction.md`  
> **Last reviewed:** 2026-08-16

Owner sign-off: Wilson (chat 2026-08-16: "1. approve b") date: 2026-08-16

---

## 1. Goal

Pre-run instruction-SFT priors for dataset size and epoch count join the existing candidate status machine. Existing rank, alpha, learning rate, and completions-mask priors become visible. After a run, loss-curve signals produce one presentation-only next-plan correction. None of this is model quality.

## 2. Non-goals

- AutoML / Optuna / grid search  
- Rewriting `target.max_epochs`  
- Inventing dataset rows  
- Changing `weight_decay` (stays 0.0) or adding warmup  
- Changing completions-only mask behavior  
- Promoting loss to an M8 eval decision  
- Implementing BiLoRA, Flexora, DoRA, AdaLoRA, ShareLoRA  
- Naming this increment M10  

## 3. Path Alpha

`examples/support-sft.jsonl` has **4 rows**. Path Alpha must remain runnable.

`example_count < 100` and `max_epochs <= 3` → **conditional**, not infeasible.

## 4. Frozen constants

| Name | Value |
| --- | ---: |
| `INSTRUCTION_SFT_MIN_ROWS` | 100 |
| `INSTRUCTION_SFT_EPOCH_CAP` | 3 |
| `INSTRUCTION_SFT_SMALL_CORPUS_MAX` | 299 |
| `INSTRUCTION_SFT_PARROT_EPOCHS` | 10 |
| `TRAINING_POLICY_VERSION` | `aptus-training-policy-v1` |

Applies when `target.task == "sft"`.

## 5. Status rules

| Condition | Status | Planner reason (exact) |
| --- | --- | --- |
| `example_count < 100` and `max_epochs <= 3` | conditional | `Dataset example_count is below the instruction-SFT supervision prior of 100 rows; this is not a justified domain adaptation.` |
| `example_count < 100` and `max_epochs > 3` | infeasible | `Dataset example_count is below 100 rows; Aptus will not endorse training longer than 3 epochs on that set.` |
| `example_count >= 100` and `max_epochs > 3` | conditional | `Requested max_epochs exceeds the instruction-SFT epoch-cap prior of 3; Aptus will not rewrite the requested epoch count.` |
| `100 <= example_count < 300` and `max_epochs >= 10` | infeasible | `Small instruction corpus (under 300 rows) with max_epochs >= 10 matches the parrot/sycophancy over-training prior.` |

Strictest status wins. Operator facts are never rewritten.

## 6. Identity

- Add `training_policy_version` to plan identity.  
- Do not bump memory formula versions.  
- Do not hash presentation objects (`training_policy`, `correction`, later `run_correction`).

## 7. Schemas

- Plan presentation: `aptus.training-policy.v1` (not identity).  
- Plan correction kinds unchanged: `aptus.plan-correction.v1` = `select-candidate | no-path`.  
- Run presentation (TP4/TP5): **new** `aptus.run-correction.v1`.  

## 8. Claim sentences

Use: method-class prior; supervision prior; epoch-cap prior; will not rewrite the requested epoch count; training loss is not model quality.

Do not use: optimal; this model is bad/good because loss; decay will stop sycophancy.

## 9. Weight decay and masking

Leave compiler `weight_decay = 0.0` and `warmup_steps = 0`. Name completions-only masking; do not change it.
