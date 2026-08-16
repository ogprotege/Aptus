# TP5 completion — Run-correction from recorded loss series

- **Phase:** TP5
- **Status:** COMPLETE
- **Completed (UTC):** 2026-08-16
- **Owner sign-off:** pending

## Tasks completed

- `classify_run_loss_signal` implements TP4 detection order
  (eval-rose → loss-collapsed → loss-flat → none) on finite observation lists
- Job GET attaches presentation-only `run_correction` when a completed train
  job has readable `metrics.json`; omit when the file is missing
- CLI `jobs --id` prints a stderr block titled like training knobs
- Run UI panel title exactly: “Training-signal correction (not quality)”
- OpenAPI + web client regenerated for optional `JobResponse.run_correction`
- Docs: `inspect-results.md`, `claim-language.md`

## Explicit non-claims

- We did not auto-replan
- We did not auto-stop training
- We did not change weight decay
- We did not block `measured-run-pass` on this object
- We did not emit `aptus.evaluation-result.v1`
- We did not add a third kind to `aptus.plan-correction.v1`

## Surfaces

| Surface | Behavior |
| --- | --- |
| Schema | `aptus.run-correction.v1` with required disallowed codes and non_claims |
| Job GET | Optional `run_correction` from metrics observations |
| CLI | stderr training-signal correction block |
| Run UI | Panel “Training-signal correction (not quality)” |

## Deliberately not done

- Trainer / weight_decay / AutoML changes
- Writing `run_correction` into persisted job JSON or `plan_id`
- TP6 bibliography / Desktop cleanup

## Next phase allowed?

Yes — TP6 only (bibliography and Desktop cleanup; no runtime change).
