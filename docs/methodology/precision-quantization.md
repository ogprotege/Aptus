# Precision and Quantization

> **Status:** Active | **Authority:** Normative methodology | **Applies to:** Aptus 0.2 | **Audience:** Practitioners and contributors | **Last reviewed:** 2026-07-22 | **Review by:** 2027-01-22 or when runtime dtype policy changes

Methodology version: `aptus-precision-v2`.

V0.2 serializes one compute `precision` string and one optional `quantization`
string per candidate. The memory model separately publishes its state
coefficients. It does not yet serialize a full dtype record for every state.

## Compute precision

The planner serializes:

- `bf16` when every selected device declares BF16 support;
- otherwise `fp16`.

The CUDA compiler activates the corresponding PyTorch mixed-precision flag. The
planner does not independently verify FP16 capability or AMP scaling behavior.
Runtime validation and the real-model pilot remain required. See the
[PyTorch AMP contract](https://docs.pytorch.org/docs/stable/amp.html). The MLX
compiler records the candidate precision for identity but uses the pinned
MLX-LM load and compute behavior. It does not add a CUDA gradient scaler.

FP32 compute and FP8 are not enumerated in v0.2.

## Method representations

| Runtime and method | Plan quantization | Generated loading path |
| --- | --- | --- |
| CUDA full | `null` | BF16 base model; FP16 full training is fail-closed |
| CUDA LoRA | `null` | BF16 or FP16 base plus PEFT adapter |
| CUDA 8-bit LoRA | `int8-bitsandbytes` | `BitsAndBytesConfig(load_in_8bit=True)` |
| CUDA QLoRA | `nf4-double-quant` | NF4, double quantization, selected compute dtype |
| MLX-LM LoRA | `null` | Unquantized pinned MLX model plus MLX adapter |
| MLX-LM QLoRA | `mlx-4bit-groupwise` | Already four-bit pinned MLX model plus MLX adapter |

Quantization applies to the base-loading path. The v0.2 memory formula uses
32-bit coefficients for adapter weights and gradients and an 8-byte coefficient
for optimizer state per trainable parameter.

CUDA QLoRA follows the method described in the
[QLoRA paper](https://arxiv.org/abs/2305.14314), within the exact generated
Transformers, PEFT, and bitsandbytes path. MLX-LM QLoRA uses MLX-native
groupwise four-bit storage. Its eligibility comes from explicit quantization
metadata in the pinned model revision, not a CUDA device flag. Aptus never
silently quantizes an unbound model or substitutes bitsandbytes. Paper results
are not universal fit evidence.

## Distribution boundary

V0.2 rejects 8-bit LoRA and QLoRA with FSDP. CUDA DDP replicates state on every
device. Full-parameter FSDP is unsupported. CUDA LoRA FSDP uses the generated
Accelerate FSDP policy with `use_orig_params=true` and requires the real-model
pilot. MLX-LM supports single-device LoRA and QLoRA only.

## Runtime evidence

Dependency validation checks runtime-specific pins. CUDA model-data validation
loads exact weights through the selected quantization path, checks plan-driving
structure, and verifies adapter targets. CUDA measured preflight exercises the
selected kernel on a small synthetic model. Its pilot is the first complete
real-model continuation gate.

MLX-LM model-data validation loads the pinned model and tokenizes every compiled
row. Its measured preflight runs a bounded real-input adapter smoke and records
MLX memory, exact target binding, optimizer updates, adapter delta, and adapter
artifacts. Its pilot then runs the exact model and data without interruption for
at least two optimizer updates. A fresh process reloads the emitted adapter on
the pinned base and generates one to four tokens. That proves bounded adapter
inference, not crash resume. `pilot-pass` can authorize uninterrupted
full-duration LoRA or QLoRA training. Full-parameter MLX and DoRA remain
unimplemented.

Future schemas may add separate base, compute, adapter, gradient, optimizer,
master-weight, and reduction dtypes. Those fields are not implicit in v0.2.

## Related documentation

- [Capability matrix](../reference/capability-matrix.md)
- [Method selection guide](../guides/choose-a-method.md)
- [Memory estimation](memory-estimation.md)
- [Preflight and calibration](preflight-calibration.md)
