# Candidate Enumeration

Methodology version: `aptus-candidates-v2`.

Aptus v0.2 enumerates a fixed, finite strategy set. It does not ask an
unconstrained optimizer to invent configurations.

## Raw candidate set

The enumerator takes the Cartesian product of four methods and three placement
policies:

- full fine-tuning, LoRA, 8-bit LoRA, and QLoRA;
- single device, DDP, and FSDP.

This produces 12 raw candidates in deterministic order. Capability checks may
mark a candidate `feasible`, `conditional`, `infeasible`, or `unsupported`.
Every candidate remains in `plan.json`.

## Derived strategy values

V0.2 derives one strategy for each method and placement pair:

- BF16 when every selected device declares BF16 support, otherwise FP16;
- no base quantization for full and LoRA;
- the bitsandbytes 8-bit path for 8-bit LoRA;
- NF4 with double quantization for QLoRA;
- rank 8 for the memory objective;
- otherwise rank 32 for profiles with at least one million estimated tokens;
- otherwise rank 16;
- LoRA alpha equal to twice the rank;
- learning rate $2\times10^{-5}$ for full training and $2\times10^{-4}$ for
  adapter methods;
- gradient checkpointing enabled in the compiled trainer.

Adapter target modules come from the versioned model-family catalog. They are
priors, not inspected module facts. Model-data validation must verify that the
catalog targets resolve on the pinned revision, and the real-model pilot must
exercise the selected adapter path.

## Batch search

For world size $w$, micro-batch $b$, and accumulation $g$:

$$
B_{\mathrm{effective}} = b \times g \times w
$$

V0.2 considers integer micro-batches from
$\min(32,B_{\mathrm{effective}})$ down to 1 when they divide the requested
global batch exactly. It selects the largest value whose upper envelope fits.
If only a point estimate fits, the candidate is conditional and uses the
largest point-fitting value found. If no point estimate fits, the candidate is
infeasible.

The plan records micro-batch, accumulation, world size, and exact effective
batch. V0.2 does not calculate the final partial accumulation window.

## Capability and feasibility checks

The current checks cover:

- CUDA-only execution;
- model context versus requested sequence length;
- supported row schemas and SFT task selection;
- disabled sequence packing;
- at least two devices for DDP or FSDP;
- no 8-bit LoRA or QLoRA with FSDP;
- unsupported full-parameter FSDP and conditional LoRA FSDP under the
  simplified v0.2 sharding prior;
- declared 8-bit or 4-bit device support for quantized methods;
- exact global-batch arithmetic;
- point and upper per-device VRAM fit;
- a distribution-aware host-RAM loading heuristic and a disk heuristic covering
  model staging, source and canonical data, the bounded pilot set, pilot
  workspace, three retained checkpoints, and final export.

Rejection and conditional reasons are plain, structured candidate fields. V0.2
does not yet emit a separate pass, fail, or unknown record for every rule.

## Deterministic identity

The candidate ID hashes normalized model, dataset, hardware, and target facts
plus the executable strategy fields. Those fields include method, distribution,
precision, quantization, batch arithmetic, devices, adapter configuration,
learning rate, resource requirements, status, target modules, and the complete
normalized memory record. The public ID uses the first 20 hexadecimal characters
of that digest.

The plan ID separately binds the schema and memory-formula versions, normalized
facts, the ordered candidate IDs, and the recommended candidate ID.

## Current boundary

User-bounded rank grids, optimizer choices, attention backends, selected-layer
search, inspected module shapes, search-truncation records, and partial-window
reporting are future work. They are not implicit v0.2 dimensions.
