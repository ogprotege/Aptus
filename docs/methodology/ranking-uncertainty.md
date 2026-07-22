# Ranking and Uncertainty

Methodology version: `aptus-ranking-v2`.

Ranking never makes an infeasible or unsupported candidate viable. V0.2 ranks
the feasible and conditional candidates with a deterministic lexicographic
policy.

## Pareto annotation

For each viable candidate, Aptus compares:

- upper per-device VRAM;
- method fidelity order: full, LoRA, 8-bit LoRA, then QLoRA;
- gradient accumulation steps.

A candidate is marked `pareto_frontier=true` when no other viable candidate is
no worse on all three values and strictly better on at least one. This flag is
informational in v0.2. Dominated viable candidates remain eligible for the
objective ordering.

## Objective ordering

Every ordering prefers `feasible` over `conditional` first.

The remaining keys are:

| Objective | Lexicographic keys after status |
| --- | --- |
| `memory` | upper VRAM, preferred method, accumulation |
| `speed` | accumulation, preferred method, fidelity order |
| `quality` | fidelity order, preferred method, upper VRAM |

The speed and quality orders are policy priors. They are not measured
throughput or model-quality predictions. Stable enumeration order resolves any
remaining exact tie.

The serialized `preference_score` is the negative rank position. It is not a
normalized utility score. `ranking_basis` states the objective and repeats that
no quality or throughput value was fabricated.

## Recommendation statement

The plan records:

- the selected candidate ID, method, and distribution;
- the number of viable candidates;
- the objective policy;
- every candidate, status, reason, and Pareto annotation;
- the uncalibrated memory warning;
- the need for runtime validation.

## Uncertainty

Every v0.2 candidate uses `uncalibrated-pilot-required` confidence. This is
a warning label, not a probability. Validation states record later runtime
evidence without rewriting that planning-time field.

No v0.2 VRAM envelope may be labeled 90%, 95%, or 99% confidence.

Normalized weighted scoring, benchmark-scoped quality metrics, predicted
throughput, evidence-derived confidence tiers, and user-defined scoring weights
remain future work.
