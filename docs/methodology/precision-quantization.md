# Precision and Quantization

Methodology version: `aptus-precision-v2`.

V0.2 serializes one compute `precision` string and one optional `quantization`
string per candidate. The memory model separately publishes its state
coefficients. It does not yet serialize a full dtype record for every state.

## Compute precision

The planner selects:

- `bf16` when every selected device declares BF16 support;
- otherwise `fp16`.

The generated `TrainingArguments` activates the corresponding mixed-precision
flag. The planner does not independently verify FP16 capability or AMP scaling
behavior. Runtime validation and the real-model pilot remain required. See the
[PyTorch AMP contract](https://docs.pytorch.org/docs/stable/amp.html).

FP32 compute and FP8 are not enumerated in v0.2.

## Method representations

| Method | Plan quantization | Generated loading path |
| --- | --- | --- |
| Full | `null` | BF16 or FP16 base model |
| LoRA | `null` | BF16 or FP16 base plus PEFT adapter |
| 8-bit LoRA | `int8-bitsandbytes` | `BitsAndBytesConfig(load_in_8bit=True)` |
| QLoRA | `nf4-double-quant` | NF4, double quantization, selected compute dtype |

Quantization applies to the base-loading path. The v0.2 memory formula uses
32-bit coefficients for adapter weights and gradients and an 8-byte coefficient
for optimizer state per trainable parameter.

QLoRA follows the method described in the
[QLoRA paper](https://arxiv.org/abs/2305.14314), within the exact generated
Transformers, PEFT, and bitsandbytes path. Paper results are not universal fit
evidence.

## Distribution boundary

V0.2 rejects 8-bit LoRA and QLoRA with FSDP. DDP replicates state on every
device. Full-parameter FSDP is unsupported. LoRA FSDP uses the generated
Accelerate FSDP policy with `use_orig_params=true` and requires the real-model
pilot.

## Runtime evidence

Dependency validation checks pinned packages. Model-data validation loads the
exact pinned weights through the selected base-quantization path, checks the
parameter count and plan-driving structural config fields, and verifies adapter
target names. Measured preflight exercises the selected quantization kernel on
a small synthetic tensor. The pilot is the
first gate to apply the complete selected training representation and exercise
bound optimizer steps, artifacts, and checkpoint continuation on the real model
and compiled data.

Future schemas may add separate base, compute, adapter, gradient, optimizer,
master-weight, and reduction dtypes. Those fields are not implicit in v0.2.
