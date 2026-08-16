# TP2 completion — Dataset/epoch priors + policy version identity

- **Phase:** TP2
- **Status:** COMPLETE
- **Completed (UTC):** 2026-08-16
- **Owner sign-off:** pending

## Tasks completed

- Instruction-SFT row and epoch priors applied as candidate status rules (earlier TP2 commits)
- Refusal catalog codes for the four supervision/epoch reasons
- `training_policy_version` (`aptus-training-policy-v1`) stored on `TrainingPlan`, bound into `plan_id`, validated like `formula_version`
- Fact-hint directions: `example_count` → increase; `max_epochs` → decrease on high-epoch codes, review on supervision-only conditional

## Path Alpha

Path Alpha (4 rows, `max_epochs<=3`) remains **compile-and-run eligible as conditional**.

## Explicit non-claims

- Four rows is **not** a justified SFT / domain adaptation.
- The four refusal rows are priors and operator guidance, not proof of model quality or sycophancy.
- Presentation objects (`training_policy`, `correction`) are not hashed into `plan_id`.

## Deliberately not done

- TP3 Compare/CLI/methodology surfaces for the new reasons
- Run-correction schema (TP4+)

## Next phase allowed?

Yes — TP3 only (surfaces for dataset/epoch priors on Compare and CLI).
