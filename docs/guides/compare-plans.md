# Compare Plans

> **Status:** Active | **Authority:** Explanatory planning guide | **Applies to:** Aptus 0.2 | **Audience:** Practitioners | **Last reviewed:** 2026-08-16 | **Review by:** 2026-10-22 or when ranking changes

Aptus enumerates a bounded candidate matrix. It applies explicit support rules,
estimates resources, and ranks viable candidates. Viable means `feasible` or
`conditional`, with fully feasible candidates ranked ahead of conditional ones.

## Candidate dimensions

The v0.2 matrix combines:

- parameter strategy: full, LoRA, int8-LoRA, or QLoRA;
- placement: single device, DDP, or FSDP where supported;
- precision and quantization required by that strategy;
- method-specific target modules and optimizer configuration.

The presence of a row does not mean it is executable. Read each candidate's
support status and reason.

## Current support rules

- Full fine-tuning and int8-LoRA require CUDA. Full fine-tuning requires BF16.
  Adapter methods select BF16 when declared and can select FP16 otherwise,
  subject to the exact pilot.
- LoRA and QLoRA also compile on one Apple unified-memory device through the
  `mlx-lm` runtime, at single placement only. There, QLoRA eligibility comes
  from explicit four-bit MLX quantization metadata in the pinned model revision,
  not from a device four-bit capability fact.
- Full, LoRA, int8-LoRA, and QLoRA can be considered on one device or DDP when
  their capability and memory checks pass.
- LoRA FSDP is conditional.
- Full FSDP is unsupported.
- int8-LoRA FSDP and QLoRA FSDP are unsupported.

## Memory evidence

Each candidate has a point estimate and a heuristic upper envelope. The ledger
separates model weights, trainable parameters, gradients, optimizer state,
activations, quantization overhead, communication or sharding costs, and reserve
where applicable.

Neither number is a measured peak. The preflight and pilot produce host-specific
observations for the selected candidate. Train admission uses the measured pilot
peak, not the analytic point estimate, for its current free-VRAM check.

## Plan-level correction

Alongside the candidate table, Aptus publishes one **correction** summary
(`aptus.plan-correction.v1`): what to do next, whether a pilot is required, and
(when no path is viable) which facts to change. CLI prints it on stderr after
`spec-plan`; the Compare stage shows a “Next action” panel with a single CTA.
Correction is presentation-only and is **not** optimality, quality, or a license
to invent unsupported methods. See [Troubleshooting — When Aptus refuses or
corrects](troubleshooting.md#when-aptus-refuses-or-corrects).

## Training-knob priors

Compare also shows a “Why these training knobs” panel (`aptus.training-policy.v1`)
for rank, alpha, learning rate, completions-mask, epochs, and dataset size.
Instruction-SFT rows and epoch rules can mark a candidate conditional or
infeasible: below the instruction-SFT supervision prior of 100 rows; exceeds
the instruction-SFT epoch-cap prior of 3 (Aptus will not rewrite the requested
epoch count); or the parrot/sycophancy over-training prior on a small corpus.
Those reasons appear on candidates and in the knobs panel. They are labeled
priors, not claims that a dataset will produce a sycophant, that 3 epochs is
optimal, or that loss proves the model is bad.

## Ranking

The target objective can favor quality, memory, or speed. Ranking is deterministic
within the enumerated catalog and recorded facts. It does not compare methods
outside that catalog and does not predict final model quality.

Before selecting a candidate, inspect:

- all unsupported reasons;
- point and upper memory margins;
- required host RAM and disk;
- distribution and world size;
- precision and quantization;
- target-module assumptions;
- source evidence and warnings;
- the gap between user facts and measured facts.

If no candidate is viable, change an explicit input or use different hardware.
A conditional recommendation still carries unresolved assumptions and requires
its exact pilot. Do not silently shorten the sequence, reduce the batch, or
change the method.

## Related documentation

- [Choose a method](choose-a-method.md)
- [Candidate enumeration](../methodology/candidate-enumeration.md)
- [Memory estimation](../methodology/memory-estimation.md)
- [Ranking and uncertainty](../methodology/ranking-uncertainty.md)
