# TP4 — Run-correction specification (freeze)

> **Status:** APPROVED 2026-08-16 — detection table signed with the TP implementation plan; owner authorized subagent-driven execution  
> **Authority:** Subordinate to I1–I12, claim language, and M8 eval contract  
> **Increment:** TP (not M10)  
> **Implementation:** TP5  
> **Last reviewed:** 2026-08-16

Owner sign-off: Wilson (approved the parent plan and DECISION-20260816-01; numeric heuristics frozen here from that plan)

---

## 1. Goal

After a measured run that already wrote finite loss observations, Aptus may attach **one** presentation-only next-plan hint derived from the curve. This is a regularization alarm. It is not model quality and not an M8 evaluation decision.

## 2. Non-goals

- Auto-replan, auto-stop, or trainer changes  
- Changing `measured-run-pass`  
- Emitting `aptus.evaluation-result.v1`  
- Weight decay as a suggested fix  
- A third kind on `aptus.plan-correction.v1`  
- Fabricating a curve when observations are missing  

## 3. Schema `aptus.run-correction.v1`

```json
{
  "schema_version": "aptus.run-correction.v1",
  "kind": "loss-collapsed | loss-flat | eval-rose | none",
  "summary": "string",
  "source": "train_loss_observations+validation_loss_observations",
  "next_plan_hints": [],
  "disallowed_suggestions": [],
  "operator_next_step": {"action": "replan-with-fact-hints | none", "label": "string"},
  "non_claims": [
    "Training loss is not model quality.",
    "Validation split loss is not an aptus.evaluation-result.v1 decision."
  ]
}
```

Not in `plan_id`. Not a validation level. Not required for `measured-run-pass`.

Attachment: optional `run_correction` on job GET when `metrics.json` (or the action’s metrics file) is readable; CLI inspect-results; Run UI panel titled “Training-signal correction (not quality).”

## 4. Inputs

- `train_loss_observations`: list of finite numbers, or missing  
- `validation_loss_observations`: list of finite numbers, or missing  

Missing / empty / non-finite / length < 2 as required → `kind=none` (abstain). Never invent points.

## 5. Detection (one primary; first match in this order)

Let `t0, tN` be first and last of train; `v0, vN` first and last of eval.

| Order | Condition | kind |
| --- | --- | --- |
| 1 | both series `len>=2` and `tN < t0` and `vN > v0` | `eval-rose` |
| 2 | train `len>=2` and `tN < t0 * 0.2` and `tN < 0.2` | `loss-collapsed` |
| 3 | train `len>=2` and `tN > t0 * 0.85` | `loss-flat` |
| 4 | otherwise | `none` |

`eval-rose` wins when it also looks collapsed (BiLoRA rule).

## 6. Fixture series (normative tests)

| name | train | eval | kind |
| --- | --- | --- | --- |
| rose | `[1.0, 0.4]` | `[0.9, 1.1]` | `eval-rose` |
| collapsed | `[1.0, 0.05]` | missing or `[0.8, 0.4]` | `loss-collapsed` |
| flat | `[1.0, 0.95]` | missing | `loss-flat` |
| single | `[1.0]` | `[]` | `none` |
| empty | missing | missing | `none` |
| both-down | `[1.0, 0.4]` | `[0.9, 0.5]` | `none` (not rose; not collapsed enough) |

## 7. Next-plan hints (enumerated only)

| kind | hints |
| --- | --- |
| `eval-rose` | decrease `target.max_epochs`; why: train fell while validation rose; not an eval pass/fail |
| `loss-collapsed` | decrease `target.max_epochs` toward the epoch cap, **or** set alpha = rank on the next plan (one primary hint: max_epochs first) |
| `loss-flat` | increase `target.max_epochs` only up to 3, or review rank; never suggest decay |
| `none` | empty hints; action `none` |

Always include disallowed: `no_automl`, `no_quality_from_loss`, `no_weight_decay_as_sycophancy_fix`.

## 8. Claim language

Use: training-signal correction; regularization heuristic; next plan.

Do not use: the model is bad/good; overfit confirmed as quality; eval pass.
