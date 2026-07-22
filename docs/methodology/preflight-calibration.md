# Preflight and Calibration

Methodology version: `aptus-preflight-v2`.

V0.2 separates a synthetic method check from the first exact model and data
step. Neither establishes final task quality.

## Model-data validation

The model-data gate uses the pinned revision to resolve the configuration,
tokenizer, and exact model weights. It validates the loaded parameter count and
plan-driving structural config fields against the plan, confirms every adapter
target-module name exists, transforms
every canonical row under the selected sequence length, and confirms visible
CUDA device count. Quantized candidates use the same 4-bit or 8-bit base-load
contract as training. The gate then releases the loaded model.

This gate does not inject adapters, enable checkpointing, run a batch, compute a
loss, mutate weights, or take an optimizer step. Its temporary allocation is not
recorded as calibrated fit evidence. Those measured execution claims belong to
the pilot.

## Measured preflight

The measured-preflight gate runs a small synthetic Llama causal model on CUDA.
It exercises the selected broad method:

- full or PEFT LoRA construction;
- an 8-bit or 4-bit bitsandbytes kernel when selected;
- one forward pass and finite-loss check;
- backward propagation and one AdamW step;
- peak allocated CUDA memory capture.

The synthetic model uses fixed small dimensions. Adapter rank and alpha are
capped for this probe. The recorded metric is method and kernel evidence. It is
not an observed peak for the planned model, sequence length, or batch.

## Real-model pilot

The pilot is the first exact model and dataset training gate. It must:

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

Only this gate can support `pilot-pass`. Full training requires explicit
confirmation against the same pilot-bound bundle.

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
