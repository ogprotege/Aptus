# M8 — COMPLETION NOTE

> **Phase status:** **COMPLETE** (pending PR merge)

| Field | Value |
| --- | --- |
| Phase | M8 |
| Title | Evaluation contract (optional, still honest) |
| Spec | `M8-eval-spec.md` |
| Schemas | `aptus.evaluation-contract.v1`, `aptus.evaluation-result.v1` |
| Branch | `feat/mission-m8-eval-contract` |
| Started (UTC) | 2026-08-13 after M7 merge `027f1a3` |
| Start commit | `027f1a313c2f6da2c0e8f55a143474a126367fe2` |
| Owner sign-off | authorized by owner request after M7 CI green |

## Mission check

- Mission statement still accurate? yes  
- Invariants I1–I12 intact? yes  
- Claim-language violations introduced? none  

## Deliverables

| Package | Artifact |
| --- | --- |
| M8.1 | Spec freeze (`M8-eval-spec.md`, DECISION-20260813-03) |
| M8.2 | `src/aptus/evaluation.py` + `tests/aptus/test_evaluation.py` |
| M8.3 | CLI `eval-contract` / `eval`; optional presentation attach |
| M8.4 | API `POST /api/v1/evaluations/contracts` and `POST /api/v1/evaluations` |
| M8.5 | Workbench copy: Facts split ≠ eval; Validate/Run loss ≠ eval pass |
| M8.6 | Claim language, capabilities, CLI/API reference, design-an-evaluation |

## Verification

- `tests.aptus.test_evaluation` (12) OK  
- CLI parser contract and API route docs updated  
- OpenAPI regenerated; web Facts/Validate/Run tests OK  

## Explicit non-claims

- Exact-match pass is not general model quality, safety, or human preference  
- Training finished / `measured-run-pass` / train loss is not this decision  
- Aptus does not generate predictions  
- Contract is not `plan_id` material  

## Deliberately not done

- Leaderboards and LLM-as-judge  
- Automatic safety red-team  
- GPU eval job / JobService ladder step  
- Workbench-hosted prediction generation  

## Next phase allowed?

**Yes — M9** (sustain / anti-drift) only if the owner wants a standing checklist phase; otherwise the program can stop here.
