# Preflight and Calibration

> **Status:** Active | **Authority:** Normative methodology | **Applies to:** Aptus 0.2 | **Audience:** Practitioners and contributors | **Last reviewed:** 2026-07-28 | **Review by:** 2027-01-22 or when a runtime gate changes

Methodology version: `aptus-preflight-v2`.

V0.2 separates a synthetic method check from the first exact model and data
step. The exact work differs by runtime. Neither establishes final task quality.

## Model-data validation

The model-data gate uses the pinned revision to resolve the configuration,
tokenizer, and exact model weights. CUDA validation checks parameter count,
plan-driving structure, adapter targets, every canonical row, visible device
count, and the selected bitsandbytes load contract. MLX-LM validation loads the
pinned revision through MLX-LM and tokenizes every compiled train and validation
row. MLX QLoRA additionally requires explicit four-bit MLX quantization metadata
in that pinned revision. It does not consult a CUDA device flag or substitute
bitsandbytes.

The CUDA gate prepares the selected method exactly far enough to prove its
trainable scope. It disables model cache use, enables gradient checkpointing,
performs k-bit preparation when required, injects PEFT LoRA for adapter methods,
and computes the plan-bound trainable census. It then releases the model. The
MLX-LM gate verifies load and tokenization compatibility without an optimizer
step.

Before loading anything, the MLX-LM gate also enforces a live unified-memory
admission check and the exact model-architecture contract. It measures the pinned
snapshot's safetensors bytes, adds any excess over the planned resident bytes to
the point and upper estimates, and refuses unless available unified memory is at
least that adjusted estimate plus `max(user reserve, 8 GiB)`. A refusal here
reports exact required, available, and shortfall byte counts and writes no
evidence; a pass binds `aptus.mlx-unified-memory-admission.v2`. This is the first
live-memory gate in the MLX ladder, and it can stop the run before any weights
are read.

This gate does not construct the optimizer, run a batch, compute a loss, mutate
weights, or take an optimizer step. Its temporary allocation is not recorded as
calibrated fit evidence. Those measured execution claims belong to measured
preflight and the real-model pilot.

## Measured preflight

For `transformers-peft-cuda`, the measured-preflight gate runs a small synthetic
Llama causal model. It exercises the selected broad method:

- full or PEFT LoRA construction;
- an 8-bit or 4-bit bitsandbytes kernel when selected;
- one forward pass and finite-loss check;
- backward propagation and one AdamW step;
- peak allocated CUDA memory capture.

The synthetic model uses fixed small dimensions. Adapter rank and alpha are
capped for this probe. The recorded metric is method and kernel evidence. It is
not an observed peak for the planned model, sequence length, or batch.

For `mlx-lm`, measured preflight runs the bounded compiler slice against the
plan-pinned model and compiled MLX train and validation files. It permits one to
eight iterations, completes at least one optimizer update, writes an MLX
adapter, and records `measured_peak_bytes`, active memory, cache memory, exact
target binding, positive adapter delta, and an adapter manifest under
`aptus.runtime-metrics.v1`. This stronger real-input smoke still has scope
`bounded-compiler-smoke-not-pilot-evidence`. It does not prove checkpoint
continuation, pilot-pass, or full-run fit.

## Real-model pilot

For the CUDA runtime, the pilot is the first exact model and dataset training
gate. It must:

- load the pinned model and tokenizer revision;
- apply the selected precision, quantization, adapters, distribution, batch,
  sequence, and loss-mask policy;
- run a bounded real training step;
- record allocated and reserved CUDA memory;
- save a checkpoint;
- restart a fresh worker;
- reload the checkpoint;
- continue for at least one step;
- preserve finite loss;
- emit bound metrics and checkpoint evidence.

Only the runtime-specific pilot gate can support `pilot-pass`. Full training
requires explicit confirmation against the same pilot-bound bundle.

For MLX-LM, the pilot is one uninterrupted exact-model and exact-data run from
the pinned base. It completes the fixed two-update schedule, requires finite
train and validation losses, proves exact target coverage and positive adapter
delta, records positive MLX peak and live headroom admission, and seals its
action-owned artifacts. A fresh child process then loads the pinned base plus
adapter and generates one to four tokens. This adapter reload is not training
continuation. `pilot-pass` can authorize an uninterrupted full-duration adapter
run, while every resume argument remains fail-closed.

## Calibration status

V0.2 stores preflight and pilot outputs, but it does not use a calibration
database to claim statistical coverage.

A future calibrated release must:

1. define cohorts by hardware, runtime, architecture, method, precision,
   quantization, sequence, batch, checkpointing, attention, and distribution;
2. separate training and held-out calibration runs;
3. publish residual distributions and underprediction rates;
4. version every fitted correction;
5. widen intervals or abstain outside cohort support;
6. retain every source run and artifact fingerprint.

Until those gates pass, `M_upper` remains an analytical planning envelope, not
a calibrated quantile.

## Related documentation

- [Validation states](../reference/validation-states.md)
- [Method registry](../reference/method-registry.md)
- [Compile, validate, and run](../guides/compile-validate-run.md)
- [Memory estimation](memory-estimation.md)
