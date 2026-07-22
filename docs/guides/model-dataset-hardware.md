# Model, Dataset, and Hardware Facts

> **Status:** Active | **Authority:** Operational fact guide | **Applies to:** Aptus 0.2 | **Audience:** Practitioners and operators | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when fact contracts change

A plan is only as credible as its facts. Aptus records provenance and refuses to
infer permission or unsupported hardware capability.

## Model facts

Supply:

- provider repository ID;
- immutable revision, normally a commit hash;
- family;
- parameter count;
- hidden size and optional intermediate size;
- layer count and context length;
- license label;
- explicit confirmation that training is allowed.

`aptus inspect model` can retrieve bounded provider-declared model configuration
and repository metadata. It does not fetch tokenizer artifacts. Review the
returned facts before copying values. Provider metadata can be wrong or
incomplete. Inspection never decides license rights or training permission.

Model-data validation later resolves the pinned revision with the installed
Transformers and PEFT stack. It requires the loaded hidden size, layer count,
context length, and supplied intermediate size to match the plan. It also
compares the loaded parameter count under an explicit tolerance and records the
result.

## Dataset facts

Supported input suffixes are `.jsonl`, `.json`, `.csv`, and `.txt`. Profiling
records the source hash, byte size, sampled statistics, schema, and canonical
row shape. Compilation then:

1. Copies the exact source to `data/dataset.*`.
2. Refuses the bundle if the copy does not match the profiled hash.
3. Validates every supported source-schema row and writes it deterministically
   to `data/training.jsonl`.
4. Writes a bounded, repeated pressure set to `data/pilot-sample.jsonl`.

The sample limit bounds profiling statistics. It does not truncate the canonical
training data placed in the bundle.

An MLX-LM bundle requires at least two usable rows. Compilation writes disjoint
`data/mlx/train.jsonl` and `data/mlx/valid.jsonl` files for the bounded compiler
and full-duration adapter path. It pads each side within that side to a complete
micro-batch and binds source and compiled counts in `aptus.mlx-split.v1`. The
current MLX split is not the CUDA group-aware full-run split. Review it before
training and keep a final test set outside the source.

Rows may declare a non-empty `split_group` either at the top level or under
`metadata.split_group`. When both locations are present, they must agree. Full
training assigns every row with the same declared value to one deterministic
train or evaluation side. The metrics record row, declared-group, split-unit,
target, realized, row-error, dataset-digest, and assignment-digest evidence
without publishing group names. Ungrouped data uses
`deterministic-exact-row-count-sha256`; grouped data uses
`deterministic-size-aware-group-sha256`. The grouped solver finds an exact
declared-group row-count subset when one can reach the target with the available
ungrouped rows. Otherwise it selects the globally closest feasible row count. A
large indivisible group can therefore still prevent an exact requested fraction.
The trainer rejects canonical data that changes across split passes or lazy
consumption, and distributed ranks must agree on the binding. Rows without a
group remain independent split units. Keep a final test set outside the training
JSONL.

Inspect source and compiled rows for secrets, private data, consent, rights,
malformed turns, label leakage, and task mismatch before compilation.

## Hardware facts

For a manual homogeneous CUDA profile, provide:

- GPU count;
- total and currently free VRAM for the repeated device profile;
- BF16 support;
- 4-bit and 8-bit capability flags;
- total and currently free host RAM;
- per-device reserve;
- currently free disk.

Local CUDA scanning preserves each visible device. A single-device row binds
the method-compatible CUDA GPU with the greatest usable free VRAM after
reserve, with the lowest visible index breaking ties. Distributed rows bind
every visible GPU and use the limiting VRAM plus capabilities shared by all
ranks. The plan records the exact device indices. The manual API describes one
device profile repeated by GPU count, so use local scan for heterogeneous CUDA
hosts.

On Darwin arm64 without CUDA, local discovery records one `mps` compatibility
device backed by shared unified memory. It does not call that pool dedicated
VRAM or copy host availability into `free_vram_bytes`. The hardware profile
records current available host memory separately. MLX planning uses the lesser
of that live value and the Metal compatibility capacity, then subtracts the
reserve. A local API scan raises an explicitly selected Apple runtime reserve to
at least 8 GiB.

The `mlx-lm` runtime compiles conditional single-device LoRA and QLoRA bundles.
The generated bundle can pass dependency, model-data, measured-preflight,
uninterrupted pilot, and explicitly confirmed full-duration adapter actions with
a compatible configured Python. Pilot reloads the emitted adapter in a fresh
process and generates one to four tokens, but training resume remains
unsupported. `pytorch-mps` is a known runtime identity without a compiler. See
the
[Apple Silicon runtime and pilot matrix](../operations/apple-silicon-pilot.md).

Manual facts are useful for planning a different host, but Aptus does not
accept a manually entered compute-capability value as runtime evidence.
Target-host validation measures the visible devices and requires compute
capability 6.0 or newer for NF4/FP4 and 7.5 or newer for LLM.int8. A target-host
pilot remains required.

CUDA supports the guarded full, LoRA, eight-bit LoRA, and QLoRA compiler paths.
Full CUDA fine-tuning requires BF16. MLX-LM supports conditional LoRA and QLoRA
on one Apple unified-memory device. MLX QLoRA eligibility comes from explicit
four-bit quantization metadata in the pinned model revision, not the CUDA-style
device capability flag. Declaring total memory is not enough. CUDA train
admission checks current free VRAM against measured pilot pressure plus reserve.
MLX planning, measured preflight, pilot, adapter reload, and full training use
live unified-memory admission where the runtime contract requires it. Train
admission also rechecks measured pilot pressure plus reserve and free disk.

LM Studio and oMLX can list models and generate through their exact loopback
origins. They are inference-only integrations and never satisfy a training
runtime requirement.

## Distributed facts

DDP requires one process per declared CUDA device. LoRA FSDP is conditional on
the catalog and generated runtime checks. Full-parameter FSDP is unsupported in
v0.2 because the pinned runtime path cannot yet support the planner's memory
claim safely. Quantized FSDP combinations are also unsupported.

## Provenance labels

- `user-attested`: supplied and affirmed by the operator.
- `provider-declared`: retrieved from a repository endpoint.
- `inferred`: produced by a versioned planner rule.
- `measured`: observed from local files, devices, or runtime checks. Dataset
  digest and size facts use this label.
- `unknown`: no defensible value is available.

Do not relabel an inferred value as measured.

## Related documentation

- [Prepare a dataset](prepare-a-dataset.md)
- [Dataset schemas](../reference/dataset-schemas.md)
- [Facts and provenance](../methodology/facts-and-provenance.md)
- [Configuration and defaults](../reference/configuration-defaults.md)
