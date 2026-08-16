# TP0 completion — Training policy freeze

- **Phase:** TP0
- **Status:** COMPLETE
- **Start:** 2026-08-16T13:18:00Z (spec draft)
- **End:** 2026-08-16T13:46:55Z
- **Baseline at phase start:** `5789ff57b64427f7ab9f3e4af77d9208d4da5a46` (controller checkout; increment implements on `feat/training-policy` from `origin/main`)
- **Owner sign-off:** Wilson — DECISION-20260816-01 option (b)

## Tasks completed

- Freeze spec written: `TP0-training-policy-spec.md`
- Decision recorded and signed: DECISION-20260816-01
- Implementation plan: `docs/superpowers/plans/2026-08-16-training-policy-and-run-correction.md`

## Decisions

- DECISION-20260816-01 (b): `<100` rows + `max_epochs<=3` is **conditional**; long-train interactions are **infeasible**; never rewrite `max_epochs`.

## Non-claims

- No planner, compiler, or UI behavior changed in TP0.
- Path Alpha remains a 4-row **proof**, not a justified SFT.

## Deliberately not done

- TP1+ implementation (authorized next)
- Weight decay / warmup changes
- Run-correction schema (TP4)

## Next phase allowed?

Yes — TP1 only. TP2 waits until TP1 completion note exists.
