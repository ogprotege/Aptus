# M5 — COMPLETION NOTE

> **Phase status:** **COMPLETE** (pending PR merge)

| Field | Value |
| --- | --- |
| Phase | M5 |
| Title | Correction loop (“ideal fix and why”) |
| Spec | `M5-correction-spec.md` (approved 2026-08-12) |
| Schema | `aptus.plan-correction.v1` |
| Branch | `feat/mission-m5-correction` |

## Deliverables

| Package | Artifact |
| --- | --- |
| M5.1 | Spec freeze approved |
| M5.2 | `src/aptus/correction.py`, API attach, OpenAPI, web types/normalizers, unit tests |
| M5.3 | Compare stage correction panel + CTA (`CompareStage.tsx`, styles) |
| M5.4 | CLI stderr correction block; no-path exit 2 with correction |
| M5.5 | Troubleshooting + compare-plans docs; claim language |

## Verification

- `tests.aptus.test_correction` (5) OK  
- CLI: plan JSON lacks `correction`; stderr prints `select-candidate` / `no-path`  
- API suite + web `api.test.ts` / `App.test.tsx` / typecheck OK  
- Documentation suite OK  

## Non-claims

- Correction is not optimality or model quality  
- Does not invent methods, FSDP, multi-GPU, or packing as “fixes”  
- Does not enter `plan_id` material  

## Deliberately not done

- LLM paraphrase layer  
- Auto-replan loops / hyperparameter search  
- Full UI-managed measured train demo  

## Next phase allowed?

**Yes — M6** (public Mac distribution) only if pursuing public ship; otherwise program can stop at M3+M4+M5 for private local use.
