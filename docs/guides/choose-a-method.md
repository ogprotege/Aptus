# Choose a Fine-Tuning Method

> **Status:** Active | **Audience:** Fine-tuning practitioners | **Authority:** Explanatory | **Applies to:** Aptus 0.2 | **Owner:** Planner | **Last reviewed:** 2026-08-04 | **Review by:** 2026-10-22

Aptus compares a bounded method catalog against explicit model, dataset,
hardware, and target facts. Method choice is one part of a candidate. Precision,
quantization, placement, batch arithmetic, dependencies, validation, and export
must agree with it.

## The four selectable methods

| Method | What trains | Base storage | Current placements | Output | Main tradeoff |
|---|---|---|---|---|---|
| Full | Every model parameter | Unquantized | Single CUDA device or DDP | Complete model safetensors | Highest trainable state, host RAM, checkpoint, and export cost |
| LoRA | Injected low-rank matrices on inspected targets | Unquantized frozen base | Single CUDA device, DDP, conditional FSDP, or one Apple unified-memory device | PEFT adapter safetensors on CUDA, MLX-LM adapter on Apple | Lower trainable state, but the full unquantized base must still load |
| int8-LoRA | LoRA adapters | Frozen bitsandbytes INT8 base | Single CUDA device or DDP | PEFT adapter safetensors | Lower base storage with an exact INT8 kernel and compute-capability gate |
| QLoRA | LoRA adapters | Frozen runtime-native four-bit base: NF4 double-quantized on CUDA, MLX four-bit on Apple | Single CUDA device, DDP, or one Apple unified-memory device | PEFT adapter safetensors on CUDA, MLX-LM adapter on Apple | Lowest current base-storage path, with four-bit kernel and quantization constraints |

Every row still needs a supported backend, a supported model family, exact batch
arithmetic, resource checks, model-data validation, measured preflight, and the
selected real-model pilot. Full and int8-LoRA require CUDA. LoRA and QLoRA also
compile on one Apple unified-memory device through the `mlx-lm` runtime, at
single placement only. “Supported” in the catalog means a guarded compiler
path exists. It does not mean a particular model and device combination has
already passed.

## Start from the binding constraint

### Choose full fine-tuning only when all-parameter updates are required

Full training requires BF16 on every participating device. Aptus rejects the
current full FP16 path because it cannot prove verified FP32 trainable master
weights. It also rejects full FSDP because the pinned runtime's shard and
full-state export transients are not calibrated.

Use full training only when the host RAM, per-device upper memory envelope,
checkpoint retention, disk, and complete-model export are acceptable. The fact
that full training has the highest planner fidelity prior is not evidence that
it will produce the best task result.

### Choose LoRA when adapter portability matters

LoRA freezes the base and trains one A/B pair for every inspected target-module
instance. The generated runtime rejects a missing pair, an extra trainable
tensor, a non-finite parameter, and optimizer membership that differs from the
validated trainable set.

LoRA avoids quantized base loading, which can simplify compatibility. It still
loads the unquantized base and therefore may not fit when QLoRA does. LoRA FSDP
is conditional and must pass the exact multi-rank pilot. LoRA also compiles on
one Apple unified-memory device through `mlx-lm`, at single placement only.

### Choose int8-LoRA only for a verified eight-bit path

Every participating CUDA device must declare eight-bit support. Runtime
validation requires compute capability 7.5 or newer for the LLM.int8 path. DDP
replicates the model on each device, so device memory is not pooled.

### Choose QLoRA when base-weight memory is the primary constraint

On CUDA, every participating device must declare four-bit support. Runtime
validation requires compute capability 6.0 or newer. The compiler selects an
NF4 base with double quantization and LoRA adapters. Quantized FSDP is outside
the current contract.

The device four-bit requirement does not apply to `mlx-lm`. On Apple Silicon,
QLoRA eligibility belongs to the pinned model revision rather than a CUDA-style
device capability bit: the generated model-data gate requires explicit four-bit
MLX quantization metadata in that revision before work can advance. The reviewed
Qwen3 MoE slice is executable only as single-device MLX-LM QLoRA with
attention-only adapters.

The second reviewed policy matches a 24-layer dense `qwen` / `qwen2` /
`Qwen2ForCausalLM` configuration with no MoE topology and a uniform four-bit
group-size-64 layout with no overrides. It permits only single-device MLX-LM
QLoRA with `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`. A match remains conditional and does not transfer runtime evidence
between artifacts.

QLoRA reduces base storage. It does not guarantee faster training, equal final
quality, or a fit. Quantization metadata, adapters, optimizer state,
activations, workspaces, load transients, and reserve still consume memory.

## Understand ranking before setting a preference

Aptus ranks feasible candidates before conditional candidates. It then applies
the selected objective:

- `memory` compares heuristic upper memory envelopes first;
- `speed` compares gradient accumulation first;
- `quality` uses the explicit fidelity prior `full`, `lora`, `int8-lora`, then
  `qlora`.

Method preference is a secondary ordering input. It cannot make an unsupported
or infeasible candidate viable, and it does not replace the objective's primary
ordering. None of these policies predicts measured throughput or model quality.

Review the full candidate record before accepting the recommendation:

1. status and every rejection or conditional reason;
2. bound device indices, placement, and world size;
3. compute precision and base quantization;
4. point estimate, upper envelope, usable memory, and reserve;
5. host RAM, disk, checkpoint, and export estimates;
6. micro-batch, accumulation, and exact effective batch;
7. rank, alpha, learning-rate prior, and target modules;
8. assumptions, confidence, and evidence IDs.

## Methods visible but not selectable

The runtime registry also explains seven methods that cannot enter a plan:

- Experimental: DoRA, BitFit, AdaLoRA, and ShareLoRA.
- Research-only: LoReFT, AFLoRA, and BiLoRA.

Each has an evidence identity, blocker, and required pilot. None has an Aptus
compiler ID, export contract, supported backend, or supported placement. Do not
translate a research paper, a library flag, or a local script into an Aptus
support claim.

BitFit is especially architecture-dependent. It updates existing bias tensors,
but many decoder architectures expose few or no eligible biases. A valid BitFit
path must prove a non-empty selected set and a bias-delta save and reload
contract before it becomes selectable.

## Related documentation

- [Fine-tuning method taxonomy](../methodology/method-taxonomy.md)
- [Capability matrix](../reference/capability-matrix.md)
- [Compare plans](compare-plans.md)
- [Memory estimation](../methodology/memory-estimation.md)
- [Preflight and calibration](../methodology/preflight-calibration.md)
