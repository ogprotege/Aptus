# Compare Plans

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

- CUDA is required. Full fine-tuning requires BF16. Adapter methods select BF16
  when declared and can select FP16 otherwise, subject to the exact pilot.
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
