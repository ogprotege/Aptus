# TP1 completion — Surface existing training knobs

- **Phase:** TP1
- **Status:** COMPLETE
- **Start:** 2026-08-16T13:51:57Z (first TP1 commit `c04cfe5`)
- **End:** 2026-08-16T14:05:00Z (docs: name training knobs as priors)
- **Baseline at phase start:** `957e10a` (TP0 freeze on `feat/training-policy`)
- **Owner sign-off:** authorized after TP0 (DECISION-20260816-01); TP1 is presentation-only

## Tasks completed

- Presentation module and tests: rank, alpha, learning rate, completions-mask labeled as priors (`c04cfe5`, `6c0ff7f`)
- CLI stderr block and API `training_policy` attachment (`6c0ff7f`)
- Compare “Why these training knobs” panel (`2c83e23`)
- Docs name knobs as Aptus v0.2 method-class / compiler priors, not optima (this docs commit)

## Commits on this branch for TP1

- `c04cfe5` test: add training-policy presentation tests
- `6c0ff7f` feat: surface rank alpha lr mask priors
- `2c83e23` feat: show training-knob rationale on Compare
- docs commit: `docs: name training knobs as priors`

## Decisions

- No candidate `status` change in TP1.
- `training_policy` is presentation-only and is not hashed into `plan_id`.
- Weight decay stays `0.0`; warmup stays `0`.

## Non-claims

- We did not add epoch/dataset gates yet (TP2).
- We did not change rank/alpha/LR formulas or completions-mask behavior.
- We did not rewrite operator `max_epochs` or invent dataset rows.
- Naming knobs as priors is not a claim that those values are optimal or predict model quality.

## Deliberately not done

- TP2 dataset-size and epoch-cap capability checks
- `training_policy_version` in plan identity (TP2)
- Run-correction schema (TP4+)
- Weight decay / warmup changes

## Next phase allowed?

Yes — TP2 only. Dataset-size and epoch-cap status rules wait until this note exists.
