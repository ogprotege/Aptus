# Glossary

## Analytic point estimate

The sum of named planner memory components before the separate uncertainty
envelope. It is not a measurement.

## Upper envelope

A conservative heuristic combination of component upper bounds. It remains
uncalibrated until compared with target-host measurements.

## Canonical row

A validated supervised row serialized as deterministic JSONL while retaining
its supported source schema, such as text, prompt-completion,
instruction-output, or messages. Compilation writes every such row to
`data/training.jsonl`. Model-data validation and training later perform the
tokenizer-specific input and loss-mask transformation.

## Candidate

One method, precision, quantization, distribution, batch, and configuration
combination considered by the versioned planner catalog.

## DDP

Distributed Data Parallel. Each rank holds its own model replica and processes a
partition of a global batch.

## Direct pins

The exact top-level package versions emitted in `requirements.txt`. They do not
form a complete transitive dependency lock.

## FSDP

Fully Sharded Data Parallel. V0.2 treats LoRA FSDP as conditional and rejects
full and quantized FSDP paths.

## Host-global Aptus lease

A per-user local coordination record that permits one Aptus GPU action across
managed state roots and POSIX portable bundle launches. It does not coordinate
unrelated programs that bypass Aptus.

## Immutable revision

A 40 to 64 character hexadecimal provider commit identifier. Mutable branches
and tags are outside the model contract.

## Model-data validation

The runtime level that resolves the pinned model and tokenizer, verifies model
facts and adapter targets, and transforms every canonical row.

## Parent promotion

The completion transaction in which the managed-job parent or portable `run.py`
verifies pending full-run evidence before changing the report to
`measured-run-pass`.

## Pilot

A bounded real-model and real-data run in two fresh processes. It measures CUDA
peaks, checks artifacts, and observes checkpoint continuation for the selected
bundle.

## Preflight

A selected-method synthetic CUDA execution used to expose stack and memory-path
failures before the real-model pilot.

## Structural export file tree

A recursive path, size, and digest manifest plus safetensors structure and
provenance checks. It does not evaluate model behavior.

## Training authorization

The atomic train-admission result for the current bundle, pilot, environment,
hardware, host RAM, and disk. A cached UI status is not the admission decision.
