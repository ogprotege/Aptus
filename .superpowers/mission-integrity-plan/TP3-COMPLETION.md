# TP3 completion — Surfaces for dataset/epoch priors

- **Phase:** TP3
- **Status:** COMPLETE
- **Completed (UTC):** 2026-08-16
- **Owner sign-off:** pending

## Tasks completed

- `build_training_policy_presentation` emits `epochs` and `dataset_size` knobs
  that quote the instruction-SFT verdict when status ≠ none, otherwise state
  the request is within the instruction-SFT prior
- CLI training-policy stderr block iterates knobs and lists the new ones
- Compare “Why these training knobs” panel already maps those names; Path Alpha
  4-row / 1-epoch fixture test asserts the supervision-prior sentence
- Methodology and claim-language docs list the allowed prior sentences

## Surfaces

| Surface | Behavior |
| --- | --- |
| Presentation knobs | `epochs`, `dataset_size` with `prior_kind=method-class-prior` |
| CLI stderr | Lists every knob from the presentation object |
| Compare UI | Renders Dataset size / Epochs with supervision-prior rationale |

## Explicit non-claims

- Does not say a dataset will produce a sycophant
- Does not say 3 epochs is optimal
- Does not say loss proves the model is bad
- Does not rewrite operator `max_epochs` or invent rows

## Deliberately not done

- Run-correction schema or trainer changes (TP4+)
- No hand-edits of generated OpenAPI (schema already allowed `epochs` / `dataset_size`)

## Suggested PR

Open or extend `feat/training-policy` covering TP1–TP3.

## Next phase allowed?

Yes — TP4 only (run-correction **spec** freeze; no trainer changes).
