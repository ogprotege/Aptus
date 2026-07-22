# Model, Dataset, and Hardware Facts

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

On Darwin arm64 without CUDA, local discovery records one `mps` device backed
by the measured shared unified-memory pool. It does not call that pool dedicated
VRAM, infer unsupported precision or quantization flags, or substitute total
memory when current availability is unknown. All v0.2 execution candidates then
remain explicitly unsupported. See the separate
[Apple Silicon pilot matrix](../operations/apple-silicon-pilot.md) for the
proposed MLX validation sequence.

Manual facts are useful for planning a different host, but Aptus does not
accept a manually entered compute-capability value as runtime evidence.
Target-host validation measures the visible devices and requires compute
capability 6.0 or newer for NF4/FP4 and 7.5 or newer for LLM.int8. A target-host
pilot remains required.

The current executable catalog requires CUDA. Full fine-tuning also requires
BF16. Adapter methods can select FP16, but the exact path must pass the pilot.
Declaring total VRAM is not enough. CUDA train admission checks current free
VRAM against the measured pilot peak plus reserve. It also checks current free
host RAM and disk.

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
