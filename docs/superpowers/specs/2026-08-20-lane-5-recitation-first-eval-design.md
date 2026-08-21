# Lane 5 — Recitation-first specialist eval

> **Status:** APPROVED 2026-08-20 — owner: "I really dont care what it takes, just find, fix, and execute."  
> **Authority:** Subordinate to claim language, current capabilities, TP0, and mission invariants I1–I12  
> **Increment:** Lane 5. **Not M10.** **Not 0.3.** **Not a 0.2 ship.**  
> **Implementation plan:** not written until this spec is approved  
> **Last reviewed:** 2026-08-20  
> **Next scheduled review:** After owner approval, or when Journey B2 recitation evidence exists  
> **Shipped later:** PR #103 (2026-08-20) added host `aptus eval-generate` as a subprocess of bundle `eval.py`. It does not import MLX into the referee. The non-goal below still forbids in-process MLX in Aptus.

Owner prompt that opened this increment (chat 2026-08-20): 0/62 gold
exact-match on Journey B is not “gold is hard”; it needs to be fixed.
This spec is the freeze for rank, split, and eval so the next train has
to recit before anyone calls it a specialist.

---

## 1. Goal

Stop treating a paraphrasing adapter as a magisterium specialist.

Journey B (`job_87419924`, adapter `ae99c34e`) scored gold exact-match
**0/62**. Re-probe on this Mac showed the same adapter also scores
**0/5 exact on train rows** and **0/8 exact on in-distribution valid
rows**, and some completions invert the source (CCEO 818 predicted as
“a baptized male is incapable of celebrating marriage”; Unitatis
Redintegratio attributed to Paul VI).

`measured-run-pass` is files, reload, and bindings. Lane 3 **Done**
means leave the files. Neither is recitation. Lane 5 makes recitation
the first eval door.

## 2. Non-goals

- Growing Aptus 0.2 (new planner statuses, new rank formula, new epoch cap)
- Bumping the product version to 0.3
- Naming this increment M10
- Retraining the Done job `job_87419924` or overwriting
  `aptus-work/magisterium/`
- Committing `aptus-work/`
- Host-side `aptus eval-generate` that loads MLX inside the referee
- Changing `aptus.exact-match.v1` to a fuzzy or doctrinal judge
- Inferring Lane 3 **Stop** from gold fail (Lane 3 still holds)
- Inferring **Use** from recitation pass
- Relabeling 7B as Path Alpha or a reviewed identity
- GGUF / LM Studio export
- AutoML, Optuna, or an unconstrained rank/LR search
- Rewriting gold completions into shorter quotes (dataset text stays)

## 3. What was actually wrong

Stacked, all measured:

| Fact | Value |
| --- | --- |
| CLI default objective | `memory` → `_rank_prior` returns **8** |
| Token volume | 34 474 estimated tokens (not ≥ 1 000 000, so quality objective is rank **16**, not 32) |
| Epochs | 2 (owner cap; TP0 allows 3 without conditional) |
| Gold split | 13 `split_group`s with **zero** overlap vs train’s 86 groups |
| Eval contract | exact-match threshold **1.0** on ~300-character unique paragraphs |
| Generation format | CompletionsDataset chat template (this part was not a bug) |
| Train loss last-50 mean | 0.905 |
| Val loss last-33 mean | 1.999 |
| Train recitation probe | 0/5 exact |
| Valid recitation probe | 0/8 exact |
| Gold | 0/62 exact |

Rank 8 was not a missing enumerator value. It was `--objective memory`.
Gold 0/62 was not a strict metric on a reciting model. The model was
not reciting.

## 4. Surfaces

In scope:

- This spec (Lane 5 protocol)
- Journey B2 operator recipe on this Mac (new local identity)
- Claim-language sentences listed in §10 (applied in the same change
  as any code that uses them)
- Optional presentation-only eval-contract copy that names recitation
  vs gold

Out of scope: planner `_rank_prior`, TP0 constants, bundle `train.py`
mask, OpenAPI, workbench craft, Mac Home / Machine / Models.

## 5. Rank (existing enumerator, no formula change)

`src/aptus/planning.py` `_rank_prior`:

- `objective == memory` → rank **8**
- else if `total_estimated_tokens >= 1_000_000` → rank **32**
- else → rank **16**

Journey B2 uses **`--objective quality`**. Combined corpus is ~40 730
estimated tokens. Rank is therefore **16**. Alpha stays `2 * rank` =
32. Learning rate stays the adapter prior `2e-4`.

Do not pass `--objective memory`. Do not add rank 32/64 to the
enumerator in this increment.

If rank 16 and 3 epochs still produce **0 train-recitation exact**,
the next authorized recipe is **6 epochs at rank 16** (TP0:
`example_count >= 100` and `max_epochs > 3` is **conditional**, not
infeasible; 306 rows is not the <300 parrot infeasible). That train
requires an explicit owner confirm of the conditional reason. It is
not a silent rewrite of `max_epochs`.

A rank-prior change is a later increment, and only after 3-epoch and
6-epoch recitation evidence exists.

## 6. Split (row holdout, same documents)

Merge the existing local files. Do not regenerate from Corpus Doctorum
unless a row is corrupt.

| Source | Rows |
| --- | ---: |
| `aptus-work/magisterium/train.jsonl` | 333 |
| `aptus-work/magisterium/gold.jsonl` | 62 |
| Unique prompts | 395 |
| `split_group`s | 99 |

**Algorithm** (deterministic, seed `20260820`):

- Groups with `n == 1`: all rows stay in the Aptus training file
  (15 singletons; cannot hold out without emptying the group).
- Groups with `2 <= n < 8`: hold out **1** gold row.
- Groups with `n >= 8`: hold out **2** gold rows.

Expected sizes from that seed on the current files:

| File | Rows | Groups |
| --- | ---: | ---: |
| Journey B2 training JSONL | 306 | 99 |
| Journey B2 gold JSONL | 89 | 84 |
| Group overlap | 84 | gold-only groups: **0** |

Aptus then applies its usual 0.1 validation split to the 306-row
training file (~275 train / ~31 valid). Those valid rows are
in-distribution holdout. They are **not** the gold file.

Leave Journey A and the Done Journey B tree untouched.

New paths (uncommitted):

```text
aptus-work/magisterium-b2/
  corpus.jsonl          # 395-row merge
  train.jsonl           # 306-row Aptus training file
  gold.jsonl            # 89-row stratified gold
  split-manifest.json   # seed, per-group counts, sha256 of both files
  bundle/               # new compile
  recitation-train-predictions.jsonl
  recitation-valid-predictions.jsonl
  gold-predictions.jsonl
  recitation-train-contract.json
  recitation-valid-contract.json
  recitation-train-result.json
  recitation-valid-result.json
  gold-contract.json
  gold-result.json
```

State dir: `--state-dir ./aptus-work/magisterium-b2-state`.

## 7. Eval ladder (this is the door)

Generation for every prediction file:

- Load the **new** adapter only (not `ae99c34e`)
- Same CompletionsDataset shape mlx-lm trains: user = `prompt`,
  assistant = `completion`, `apply_chat_template(...,
  add_generation_prompt=True)`
- Greedy (`temp=0.0`), seed `17`, `max_tokens=256`
- Do not wrap the prompt a second time

**Step A — Train recitation.** Score exact-match of a **fixed** sample:
the first 32 rows of compiled `bundle/data/mlx/train.jsonl` after
compile (chat `messages` form: gold text is the assistant content).
Contract threshold is a reporting floor, not a quality yes.

**Step B — Valid recitation.** Score exact-match of **all** compiled
valid rows.

**Step C — Gold.** Score exact-match of the 89-row gold file. Same
metric. **Do not run C as specialist evidence if A is 0/32.**

| A train exact | What you may say | What you may not say |
| --- | --- | --- |
| 0/32 | Recipe failed. Adapter did not recit seen rows. Inspect 8 misses for inverted doctrine. Stop the specialist claim. | “Gold is hard.” “Not a trainer failure.” “measured-run-pass means it learned.” |
| 1–7/32 | Recipe is copying sometimes. Report A, B, and C honestly. Not a specialist. | “Specialist.” “Gold fail is unrelated.” |
| ≥ 8/32 (0.25) | Gold is a meaningful in-distribution holdout. Report all three scores. | “Quality yes.” “Reviewed 7B.” |
| ≥ 16/32 (0.50) | Recitation door is open enough to *discuss* specialist use. Still requires a human read of gold misses for inverted doctrine. Lane 3 last call is still the operator’s. | “Aptus decided this adapter is good.” |

Gold threshold **1.0 is forbidden** for this corpus (median completion
287 characters, unique paragraphs). The gold contract still uses
`aptus.exact-match.v1` with threshold `0.000001` so 0/N is official
Aptus `fail`. A later non-zero gold score may make Aptus print
`pass` at that floor. That `pass` is **not** specialist language.
Specialist language follows the table above plus a human read of
misses. Do not set recitation or gold threshold to 1.0 unless a
later increment proves train recitation can hit it.

## 8. Journey B2 train recipe

Same model pin as Journey B. Still unreviewed 28-layer MLX QLoRA.
Still requires `--confirm-unreviewed-runtime`. Still not Path Alpha.

| Knob | Value | Why |
| --- | --- | --- |
| Model | `mlx-community/Qwen2.5-7B-Instruct-4bit` @ `c26a38f6…` | Same pin; new plan identity |
| Method | QLoRA single | Same as B; still the fitting path |
| Objective | `quality` | Rank 16 from existing prior |
| Rank / alpha | 16 / 32 | Enumerator, not a new prior |
| Epochs | 3 | CLI default; TP0 within-prior |
| Sequence | 1024 | Unchanged |
| Batch | 1 | Unchanged |
| LR | 2e-4 | Adapter prior |
| Confirm | `--confirm-unreviewed-runtime` and `--confirm-full-train` | Same doors |
| Work dir | `aptus-work/magisterium-b2/` | Do not overwrite B |

Owner approved execution 2026-08-20. Journey B2 may run.

## 9. Product vs operator

This increment is **operator-normative first**. The referee already
scores exact-match. It does not generate predictions. Lane 5 does not
put MLX generation into `aptus eval`.

Required local artifacts (uncommitted): the prediction JSONLs and the
three result JSONs. The generate script lives next to them, private,
and must match §7. It is not an Aptus product path.

Claim-language (§10) is the product lock so a later agent cannot
dismiss 0 recitation again.

A later increment may add `aptus eval-generate`. Not this one.

## 10. Claim language

Allowed:

- “train recitation”
- “valid recitation”
- “recitation door”
- “Zero exact-match on train recitation is a recipe failure.”
- “Gold exact-match is not specialist evidence while train recitation is 0.”
- “in-distribution row holdout” (the new gold)
- “Rank 16 is the quality-objective prior, not a tuned optimum.”

Forbidden:

- “Gold is hard” as the explanation of 0 recitation
- “Not a trainer failure” for 0/N train recitation
- “exact-match 0.0 is not a problem”
- “reviewed 7B”
- “specialist” unless train recitation ≥ 0.50 and a human read of gold
  misses reports no inverted doctrine
- “Aptus decided this adapter is good”
- Inferring Lane 3 **Stop** from gold fail
- Inferring **Use** from recitation pass

## 11. Frozen decisions

| Decision | Choice |
| --- | --- |
| Increment name | Lane 5 |
| Version bump | No |
| 0.2 line | Do not grow |
| Rank | 16 via `--objective quality` |
| Rank formula | Unchanged |
| First epoch count | 3 |
| If recitation 0 | Next recipe is 6 epochs at rank 16, owner-confirmed conditional; not a rank-prior change |
| Split | Stratified row holdout, seed 20260820, 89/306, 0 gold-only groups |
| Old gold | Retired for B2; keep the B tree on disk |
| Eval order | Train recitation → valid recitation → gold |
| Gold threshold 1.0 | Forbidden on this corpus |
| Host eval-generate | Out of scope for this spec increment. Shipped later in PR #103 as a bundle subprocess, not MLX-in-the-referee |
| Done job | Do not retrain |

## 12. Success

Lane 5 the spec is done when this file is approved.

Lane 5 the first train is done when Journey B2 has:

1. A new plan identity at rank 16, 3 epochs, unreviewed-runtime confirmed
2. `measured-run-pass` on a new job (not `job_87419924`)
3. Train recitation, valid recitation, and (if A > 0) gold results on disk
4. Honest language: 0/32 train recitation is a recipe failure; a
   non-zero recitation is still not a quality yes
5. A human read of misses for inverted doctrine before anyone says
   specialist

It is not done when gold is 0/89 and the write-up says the metric is
strict.

## 13. Testing

- Spec lands with documentation inventory, reachability, and
  review-metadata gates green.
- No Python planner tests change (rank formula unchanged).
- No CUDA/MLX compiler tests change.
- Journey B2 runtime evidence is local and uncommitted.
- Recitation generate script is private; do not add it to the wheel.
