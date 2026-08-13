# M8 — Evaluation contract specification (freeze)

> **Status:** APPROVED 2026-08-13 (owner asked to execute M8 after M7 merge green)  
> **Authority:** Mission integrity plan §15  
> **Implementation:** `feat/mission-m8-eval-contract`

## 1. Goal (one sentence)

Make **training finished** and **meets evaluation target** distinct first-class
states, without claiming general quality, safety, or human preference.

## 2. Non-goals

- Leaderboards or cross-run ranking  
- Automatic safety red-team as a product default  
- Human-preference or LLM-as-judge scores without human labels  
- GPU/runtime generation of predictions (operator supplies them)  
- Making eval a validation level or a `JobService` train-ladder action  
- Putting the contract into `plan_id` material  

## 3. Schemas

- `aptus.evaluation-contract.v1` — operator-authored binding  
- `aptus.evaluation-result.v1` — scored decision  
- Metric v1: `exact_match` via `aptus.exact-match.v1` only  

## 4. Decision rule

| Situation | Decision |
| --- | --- |
| Gold empty, digest mismatch, row-count mismatch, missing/extra IDs, export-digest mismatch, unsupported metric | `abstain` |
| Complete alignment and `score >= threshold` | `pass` |
| Complete alignment and `score < threshold` | `fail` |

`measured-run-pass`, train loss, and split evaluation loss never produce this
decision.

## 5. Surfaces

- CLI: `aptus eval-contract`, `aptus eval`  
- API: `POST /api/v1/evaluations/contracts`, `POST /api/v1/evaluations`  
- Optional `evaluation_contract` on a plan payload is presentation-only  
- UI: never equate loss curves or `evaluation_fraction` with eval pass  

## 6. Claim language

A pass means: on this bound gold digest, these predictions, this metric
implementation, and this threshold, the named artifact met the contract. It
does not mean the model is good, safe, preferred, or release-ready.
