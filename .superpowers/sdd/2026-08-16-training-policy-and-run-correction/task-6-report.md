# Task 6 report — Refusal catalog + identity version

**Status:** COMPLETE  
**Commit message:** `feat: bind training-policy version and refusal codes`

## Changes

### Refusal catalog (`src/aptus/refusal.py`)

| Substring (lowered) | Code | Changeable facts | Actionable |
| --- | --- | --- | --- |
| `below the instruction-sft supervision prior` | `dataset_below_sft_prior` | `dataset.example_count` | True |
| `will not endorse training longer than 3 epochs` | `dataset_too_small_for_requested_epochs` | `dataset.example_count`, `target.max_epochs` | True |
| `exceeds the instruction-sft epoch-cap prior` | `epoch_cap_prior` | `target.max_epochs` | True |
| `parrot/sycophancy over-training prior` | `small_corpus_high_epoch` | `dataset.example_count`, `target.max_epochs` | True |

### Fact-hint directions (`src/aptus/correction.py`)

- `example_count` → `increase`
- `max_epochs` → `decrease` for `dataset_too_small_for_requested_epochs`, `epoch_cap_prior`, `small_corpus_high_epoch`
- `max_epochs` → `review` for `dataset_below_sft_prior` (supervision-only conditional)

### Identity (`TRAINING_POLICY_VERSION = "aptus-training-policy-v1"`)

- Single definition in `plan_contract.py`; re-exported/used via import in `training_policy.py`
- Stored on `TrainingPlan` next to `formula_version`
- Included in `plan_id_for_payload` identity material
- Validated like `formula_version` (missing/wrong → contract error)
- Set by `plan_training` on every plan
- Presentation `training_policy` / `correction` not hashed

### Docs / completion

- `docs/reference/plan-schema.md` — new plan field + presentation note
- `.superpowers/mission-integrity-plan/TP2-COMPLETION.md` — Path Alpha remains conditional; four rows is not a justified SFT

## Tests

```text
PYTHONPATH=src:. …/python -m unittest \
  tests.aptus.test_refusal tests.aptus.test_plan_contract \
  tests.aptus.test_planning tests.aptus.test_correction \
  tests.aptus.test_training_policy tests.aptus.test_documentation -v
```

**Result:** Ran 182 tests — OK

## Concerns

- None for Task 6 scope. TP3 surfaces (Compare/CLI methodology wording) intentionally not done.
