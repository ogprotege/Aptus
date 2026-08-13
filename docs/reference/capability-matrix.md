# Capability Matrix

| Metadata | Value |
| --- | --- |
| Status | Active, unreleased engineering preview |
| Audience | Operators, product owners, method authors, and release reviewers |
| Authority | Normative v0.2 support boundary |
| Last reviewed | 2026-08-11 |
| Next review | 2026-11-01, or sooner when the method registry, planner, compiler, or model policy changes |

This matrix distinguishes a planner path from target-host proof. A planner row
marked supported can become viable when all facts and analytic gates pass. It
still requires runtime evidence for the exact bundle and host. CUDA training
requires static, dependency, model-data, measured-preflight, and pilot evidence.
MLX-LM uses the same state ladder with a runtime-specific uninterrupted pilot.
A current `pilot-pass` can authorize explicit full-duration adapter training.

Support also depends on the installed host's current model-policy registry. A
package-free bundle can prove the integrity and decision parity of its embedded
frozen snapshot, but that result does not establish policy currency. Installed
Aptus rejects a coherent stale plan whose decision or snapshot digest is no longer
current and requires deterministic replanning. An earlier `pilot-pass` cannot
authorize training or completion promotion after that policy boundary changes.

Two fresh, clean Apple Silicon MLX-LM workflows reached `measured-run-pass`
under their recorded v5 plan and v3 bundle contracts in the
[2026-08-05 exact-source acceptance record](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md).
The acceptance source is `719255153e3fc7e38e83b5ff826d587e5e58bf80`, its tree is
`be99f5664ccb580f2600471f1ae3241a294b1a7e`, and its bundle fingerprint is
`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`.
That evidence closes the Phase 6 MLX-LM runtime gate only for its exact pinned
Qwen2.5 artifact and revision, source and tree, host, runtime, dataset, policy
snapshot, plan, bundle, and fingerprint. It does not transfer to another
artifact that matches the reviewed Qwen2 configuration footprint. The
[original Phase 6 record](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
remains the unchanged historical baseline, and the
[2026-07-27 record](../operations/evidence/2026-07-27-mlx-lm-acceptance/README.md)
remains historical v2/v2 evidence for the same pinned artifact.

One exact SmolLM2 CUDA LoRA single-device workflow separately reached
`measured-run-pass` in the [2026-08-06 CUDA acceptance
record](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
at source `c12c4d8db0037a2c278a2ad95a0a2cbda4387eed`. It binds dependency,
model-data, measured preflight, two-phase checkpoint-continuation pilot, full
training, structural PEFT export, and parent promotion to the exact recorded
host/runtime/model/dataset/plan/policy/bundle scope. It does not establish
repeatability or qualify any other CUDA matrix cell or environment.

The separate [2026-08-10 Phase 5 repeatability
packet](../operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md)
records five of five new, predeclared SmolLM2 LoRA single-device slots passing
at source `3bfec547d4cffedbaf049426d9713f1ccc25b5a2`. All five completed 128
optimizer steps with protocol-valid evidence, passed the frozen duration and
peak-memory stability thresholds, and passed off-host copy and fresh retrieval
verification. This establishes the exact frozen anchor and Phase 6 eligibility;
it does not qualify another method, artifact, host, environment, or release.

The historical [Phase 6 method matrix](../operations/evidence/2026-08-10-cuda-phase6-method-matrix/README.md)
and later [remediation matrix](../operations/evidence/2026-08-10-cuda-phase6-remediation-matrix/README.md)
remain immutable nonqualifying cohorts. After two Aptus-owned pre-launch
registration races were corrected, a separate [five-slot Full
cohort](../operations/evidence/2026-08-10-cuda-phase6-confirmatory-stability/README.md)
at exact merged source `2bc4d9a38f88cb0be1087b6e35a329587d1942bf`
passed the frozen stability and integrity contract. It establishes one stable
exact-host Full cell and authorizes the bounded Phase 7 procedure; it does not
qualify another method, artifact, host, environment, or release.

The current reviewed [Phase 7 same-family stability
packet](../operations/evidence/2026-08-11-cuda-phase7-same-family-stability/README.md)
binds a new cohort at exact merged source
`412095bd66618fee9d3e1936e79b90da12a4c61b`. The 135M LoRA, 135M Full, and
360M LoRA cells each passed three of three 128-step exploratory slots and the
frozen stability and integrity contract without replacements. The exact planner
left 360M Full and both 1.7B cells unadmitted and planned-not-started. A
separate reviewed [architecture-breadth
amendment](../operations/evidence/2026-08-11-cuda-phase7-breadth-amendment/README.md)
admitted one Qwen3-0.6B LoRA cell. Its first cohort stopped during model-data
validation because serialized tensor elements had been declared as unique
runtime parameters. The reviewed [parameter-semantics
correction](../operations/evidence/2026-08-11-cuda-phase7-breadth-parameter-correction/README.md)
bound that stopped outcome and corrected the declaration to 596,049,920 unique
parameters. A second fresh cohort stopped before optimizer work because Linux
admission excluded reclaimable page cache. After the probe was corrected at
exact merged source `a41ae4941661867789034eaa63bb968f2e137aba`, a third,
independently reviewed [breadth stability
cohort](../operations/evidence/2026-08-11-cuda-phase7-breadth-stability/README.md)
passed conditioning and all three 128-step exploratory slots without
replacement. The Qwen3-0.6B LoRA cell passed the frozen stability and integrity
contract, completing Phase 7. The separately reviewed [Phase 8 guarded-frontier
packet](../operations/evidence/2026-08-11-cuda-phase8-guarded-frontier/README.md)
closed the three frozen one-axis ladders at exact source `59993d7` with fourteen
bounded-pilot passes, two controlled bounded-pilot OOM nonpasses, one
planned-not-started point, complete custody, and no full training. The later
[Phase 9 endurance packet](../operations/evidence/2026-08-11-cuda-phase9-endurance/README.md)
records the selected sequence-256, effective-batch-32, micro-4, accumulation-8
configuration passing three of three 300-update slots and eight controlled
job-service exercises without replacement. The reviewed [Phase 10 campaign
certification](../operations/evidence/2026-08-11-cuda-phase10-certification/README.md)
then reconciles every frozen campaign disposition, recomputes the published
statistics, and closes the campaign with 149 planned, 58 started, 91
planned-not-started, and 47 qualifying passes. This remains exact-host bounded
evidence, not broad capability or model-quality evidence. Gemma remains
license-excluded and Mistral remains planner-ineligible. The [earlier stopped Phase 7
cohort](../operations/evidence/2026-08-10-cuda-phase7-scale-staircase/README.md)
remains an immutable historical record.

A separate local desktop gate passed 10 of 10 clean engineering builds at
implementation commit `1038ecdd13103418ef1135e1ced634c10370a961`. That result
does not transfer to later commits. Pull-request CI rebuilds and packages the
exact GitHub-tested merge commit and records it in `COMMIT`. One Developer ID signed notarized arm64 Mac identity passed its packaging
gate at `edc6cfdec48daeb17af8cae7dbb9fde0d8112a81`
([2026-08-13 packet](../operations/evidence/2026-08-13-desktop-public-release/README.md)).
That identity does not transfer to other commits. One Path Beta CUDA
fresh-process adapter reload is recorded in the
[2026-08-13 M7-C packet](../operations/evidence/2026-08-13-path-beta-cuda-reload-m7c/README.md)
and does not transfer to other cells. Neither the one
exact CUDA acceptance nor the MLX-LM acceptance establishes model quality,
production throughput, broad runtime support, or release readiness. Aptus
v0.2 remains unreleased.

## CUDA method and placement matrix

| Method | Single | DDP | FSDP | Export contract |
| --- | --- | --- | --- | --- |
| Full | Planner-supported with BF16 | Planner-supported with BF16 and at least 2 GPUs | Unsupported | Full-model safetensors |
| LoRA | Planner-supported | Planner-supported with at least 2 GPUs | Conditional with at least 2 GPUs | PEFT adapter safetensors |
| int8-LoRA | Planner-supported with eight-bit capability | Planner-supported with shared eight-bit capability and at least 2 GPUs | Unsupported | PEFT adapter safetensors |
| QLoRA | Planner-supported with four-bit capability | Planner-supported with shared four-bit capability and at least 2 GPUs | Unsupported | PEFT adapter safetensors |

For each selected training runtime, all 12 method and placement pairs remain in
the candidate matrix. Unsupported and infeasible rows are evidence, not hidden
branches.

## MLX-LM method and placement matrix

| Method | Single | DDP | FSDP | Export contract |
| --- | --- | --- | --- | --- |
| Full | No compiler | Unsupported | Unsupported | None |
| LoRA | Conditional through uninterrupted pilot and full-duration adapter training | Unsupported | Unsupported | MLX-LM adapter |
| int8-LoRA | No compiler | Unsupported | Unsupported | None |
| QLoRA | Conditional through uninterrupted pilot and full-duration adapter training, with explicit four-bit quantization metadata in the pinned MLX model revision; no device four-bit capability fact is required | Unsupported | Unsupported | MLX-LM adapter |

MLX-LM uses the `mps` compute backend and `aptus-memory-mlx-v2` estimator. Its
LoRA and QLoRA candidates always remain conditional and pilot-required. Its
pilot is one uninterrupted run from the pinned base with at least two optimizer
updates, finite losses, exact target binding, positive memory and adapter-delta
evidence, live headroom, immutable artifacts, and fresh-process adapter reload
with one to four generated tokens. The reload is inference proof, not training
resume. A pass can admit an uninterrupted full-duration adapter run.

### CUDA method-specific hard boundaries

- Full training in FP16 is unsupported because the generated path does not
  retain a verified FP32 trainable master-weight contract.
- Full FSDP is unsupported because the pinned transient and export path is not
  calibrated safely.
- int8-LoRA FSDP and QLoRA FSDP are outside the verified compiler matrix.
- LoRA FSDP uses `use_orig_params=true` and remains conditional even when the
  analytic envelope fits.
- CUDA adapter methods require every target module in the family catalog to
  exist on the loaded revision.

## Precision and quantization

| Path | Planner rule | Runtime proof still required |
| --- | --- | --- |
| BF16 | Selected only when every participating device declares BF16 | Actual device, stack, method, and pilot behavior |
| FP16 full | Always unsupported | No launch |
| FP16 adapters | Selected when participating devices do not all declare BF16 | AMP behavior and exact pilot |
| Unquantized base | Full and LoRA | Model load and measured peak |
| INT8 bitsandbytes base | int8-LoRA only | Exact bitsandbytes load and kernel path |
| NF4 double-quantized base | QLoRA only | Exact bitsandbytes load and kernel path |
| MLX unquantized base | MLX-LM LoRA | Exact MLX model load, measured preflight, uninterrupted pilot, and adapter reload |
| MLX four-bit groupwise base | MLX-LM QLoRA only | Explicit quantization metadata in the pinned MLX model, measured preflight, uninterrupted pilot, and adapter reload |
| FP32 compute | Not enumerated | Future contract |
| FP8 | Not enumerated | Future contract |

CUDA probe fallback derives four-bit eligibility at CUDA compute capability
6.0 or newer and eight-bit eligibility at 7.5 or newer. Manual planning still
requires explicit capability flags. Runtime model-data, synthetic, and pilot
checks remain authoritative for the pinned software stack.

MLX-LM QLoRA does not use bitsandbytes or NF4 assumptions. Aptus does not
quantize an unbound model during training. The pinned model revision must
already declare its MLX four-bit quantization metadata. This general dense
QLoRA rule does not admit an MoE checkpoint. The Qwen3 MoE row also requires
the exact mixed layout in the model-support table below.

## Backend matrix

| Backend | Accepted fact value | Local discovery | Planner execution rows | Compiler and runtime |
| --- | ---: | ---: | ---: | ---: |
| CUDA | Yes | Yes | Yes | Yes, subject to gates |
| ROCm | Yes | No supported probe result | Explicitly unsupported | No |
| MPS | Yes | Apple shared-memory and Metal inventory | MLX-LM LoRA and QLoRA single-device rows | MLX-LM uninterrupted adapter training; PyTorch MPS has no compiler |
| CPU | Yes | No supported accelerator result | Explicitly unsupported | No |
| MLX | Not a backend enum | Separate runtime probe | Runtime selected as `mlx-lm` over `mps` | Separate MLX-LM compiler |

On Apple Silicon, discovery reports one `mps` compatibility device backed by the
Metal working-set advisory when available, otherwise the measured unified-memory
capacity. It does not infer BF16, four-bit, eight-bit, or free VRAM. Apple
platform probing reports current host memory headroom and an optional Metal GPU
core count separately. MLX planning caps usable unified memory by the measured
free host RAM when available. The CUDA compiler never silently routes through
MPS or MLX.

### Training-runtime matrix

| Runtime | Discovery and configuration | Current compiler | Highest reachable or recorded evidence |
| --- | --- | --- | --- |
| `transformers-peft-cuda` | Exact active CUDA Python environment | Full, LoRA, int8-LoRA, QLoRA | The Phase 10 packet certifies the 149-slot exact-host campaign aggregate, six listed stable single-device cells, a guarded 17-point frontier, and the recorded 900-update endurance plus eight-exercise job-control result; every unlisted method, placement, artifact, host, and environment remains unqualified |
| `mlx-lm` | Exact external Python executable, including persisted Mac selection | Single-device LoRA and QLoRA, including the conditional Qwen3 MoE row and reviewed 24-layer dense Qwen2 footprint | Two current v5/v3 dense QLoRA workflows reached `measured-run-pass` for the exact accepted Qwen2.5 artifact; every different Qwen2 artifact and the Qwen3 MoE row remain conditional and pilot-required |
| `pytorch-mps` | Discoverable and configurable exact external Python | None | No compiled runtime evidence |

LM Studio and oMLX are not training runtimes. They are loopback inference-only
services for model listing and text generation.

### Desktop delivery boundary

| Evidence layer | Current result | Claim boundary |
| --- | --- | --- |
| Local desktop stability | 10 of 10 clean builds passed at `1038ecdd13103418ef1135e1ced634c10370a961` | Historical engineering evidence for that exact source commit |
| Pull-request packaging | Workflow builds GitHub's synthetic merge commit and uploads app ZIP, DMG, checksums, and source marker | Passed only after that exact workflow commit's GitHub Actions job succeeds |
| Default signature | Ad-hoc signing is built and verified | Local review integrity, not public distribution approval |
| Public Mac distribution | Developer ID signing, notarization, stapling, and Gatekeeper assessment are implemented as required gates | Open until real credentials produce accepted artifacts bound to the exact release commit |
| CUDA target-host execution | Phase 10 reconciled 149 planned slots to 58 starts, 91 predeclared-not-started, 47 qualifying outcomes, and no replacement runs | Complete for the bounded campaign; partial for the product because only the six listed stable cells, guarded frontier, and endurance/job-control scope are qualified |

## Distribution behavior

MLX-LM supports only `single`. The table below defines the CUDA placement
behavior.

| Distribution | World size | Device binding | Memory rule |
| --- | ---: | --- | --- |
| `single` | 1 | Compatible device with greatest usable memory, stable index as tie-break | Selected device free-or-total minus reserve |
| `ddp` | All planned devices | Every planned index in order | Least per-device usable memory; state replicated |
| `fsdp` | All planned devices | Every planned index in order | Simplified sharding prior; LoRA only and conditional |

The requested global batch must divide exactly by world size. Aptus tests
per-device micro-batches from the largest exact divisor at or below 32 and
selects the largest whose upper envelope fits. DDP never sums VRAM across
devices.

## Model support

### Adapter target-module families

| Family | Target modules |
| --- | --- |
| `llama` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `mistral` | Same seven dense projection names |
| `gemma` | Same seven dense projection names |
| `qwen` | Same seven dense projection names |
| `qwen3_moe` | `q_proj`, `k_proj`, `v_proj`, `o_proj` only |

Full training does not need adapter target modules. Provider inspection performs
only exact alias normalization:

- `qwen2` and `qwen3` to `qwen`;
- `gemma2`, `gemma3`, and `gemma3_text` to `gemma`; and
- `gemma3` only for explicitly accepted text architectures.

Multimodal, prefix-matched, and unknown architectures are not silently mapped.
The reviewed dense Qwen2 configuration footprint requires all fields below:

| Qwen2 field | Required value |
| --- | --- |
| Aptus family | `qwen` |
| Provider model type | `qwen2` |
| Architecture | `Qwen2ForCausalLM` |
| Layer count | 24 |
| Topology | Dense, with no MoE configuration |
| Checkpoint layout | Uniform four-bit, group size 64, with no module overrides |
| Runtime and backend | `mlx-lm` on `mps` |
| Method and placement | QLoRA, `single` |
| Adapter scope | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` |
| Evidence | Two v5/v3 `measured-run-pass` repetitions for the exact 2026-08-05 accepted artifact; `pilot-required` for every different artifact |

This is a reviewed configuration footprint, not an artifact allowlist. An
inspection receipt binds the exact model ID and immutable revision. The August
5 record closes the current Phase 6 ladder for its exact artifact and acceptance
source only. Its runtime evidence does not transfer to another artifact, which
must complete its own model-data, measured-preflight, and pilot gates.

The only sparse exception is exact and requires all fields below:

| Qwen3 MoE field | Required value |
| --- | --- |
| Aptus family | `qwen3_moe` |
| Provider model type | `qwen3_moe` |
| Architecture | `Qwen3MoeForCausalLM` |
| Default checkpoint layout | Four-bit, group size 64 |
| Module overrides | Exactly one eight-bit, group-size-64 `model.layers.N.mlp.gate` override for every layer, sorted by module path |
| Shared expert | Absent |
| Runtime and backend | `mlx-lm` on `mps` |
| Method and placement | QLoRA, `single` |
| Adapter scope | Attention `q_proj`, `k_proj`, `v_proj`, and `o_proj` |
| Evidence | `pilot-required` |

The plan records total resident parameters separately from backend-derived
active parameters and sparse-layer count. Active parameters describe routed
per-token computation. They never reduce the base-weight residency budget.
The quantization layout is canonical and bound into plan identity. A different
override count, module path, bit width, group size, or ordering remains
unsupported. Aptus can inspect an arbitrary immutable revision, but it admits
that revision only when every structural and runtime gate matches this row.
Target-host acceptance applies only to the exact revision that produced the
evidence. Every other MoE family, precision, runtime, method, and placement
remains unsupported.

CUDA model-data validation checks the loaded parameter count, hidden size,
optional intermediate size, layers, context length, and adapter targets.
MLX-LM model-data validation loads the exact revision, validates QLoRA
quantization metadata, the uniform Qwen2 or mixed Qwen3 layout bound by the
selected policy when applicable, and exact MoE topology when applicable. It
then tokenizes every bound row.

## Dataset support

| Capability | V0.2 behavior |
| --- | --- |
| File formats | `.jsonl`, `.json`, `.csv`, `.txt` |
| Row schemas | Text, content alias, prompt-completion, instruction-output, messages |
| Mixed schema file | Supported when every row independently validates |
| Empty supported row | Ignored by profiling and canonical compilation |
| Malformed structured row | Rejected with row context |
| Canonical compilation | Every valid row, deterministic sorted-key JSONL |
| Tokenization | Exact pinned tokenizer at model-data and training time |
| Loss mask | Full text for text/content; completion only for structured rows |
| Sequence packing | Unsupported |
| Split grouping | Optional `split_group` or `metadata.split_group` |
| Evaluation fraction | Deterministic, exact for ungrouped rows when possible; closest atomic grouped result otherwise |

The source, canonical dataset, and split assignments are digest-bound. Full
training checks the canonical file across three split passes and during lazy
consumption. Distributed ranks must agree on digest, assignments, and counts.

## Target support

| Target dimension | Supported values |
| --- | --- |
| Task | `sft` only |
| Objective | `quality`, `memory`, `speed` ranking policies |
| Effective batch | Positive exact global batch |
| Sequence length | Positive and within model context |
| Evaluation fraction | `[0, 1)` |
| Checkpoint interval | Positive steps |
| Packing | False only |
| Maximum wall time | No enforced value |
| Quality metric or threshold | Optional `aptus.evaluation-contract.v1` exact-match only; not plan identity |

The `quality` objective is a deterministic method-fidelity ordering. It does not
predict downstream model quality.

## Method registry visibility

The API and workbench expose 11 typed descriptors:

- four selectable `gated-executable` methods: Full, LoRA, int8-LoRA, QLoRA;
- four nonselectable `experimental` methods: DoRA, BitFit, AdaLoRA, ShareLoRA;
  and
- three nonselectable `research-only` methods: LoReFT, AFLoRA, BiLoRA.

Only selectable descriptors enter the 12 planner rows. The other descriptors
have no compiler ID, export kind, backend, or distribution contract.

## Compiler and bundle support

Supported now:

- atomic no-clobber directory publication;
- deterministic no-clobber ZIP publication;
- exact direct package pins by selected method and runtime;
- a versioned candidate runtime contract with compute, runtime, compiler,
  estimator, evidence, and export identities;
- single, DDP, and conditional LoRA FSDP Accelerate configuration;
- portable CUDA contract, validation, preflight, training-child, and run-parent
  code;
- separate MLX-LM validation, bounded preflight, adapter, data, and run-wrapper
  artifacts;
- structural CUDA full-model or adapter safetensors export;
- MLX-LM adapter output at measured preflight; and
- cleartext source, canonical, and pilot data copies.

Not implemented:

- transitive dependency lock generation;
- encrypted bundle data;
- provider provisioning or cloud infrastructure;
- merged or deployment-specific exporter plugins;
- general or CUDA semantic export inference validation beyond the bounded MLX
  adapter reload check; and
- arbitrary user overrides of generated training source.

## Execution support

Supported now:

- five ordered managed actions;
- exact external Python runtime probing, selection, and private persisted
  configuration;
- persisted local jobs and logs;
- POSIX process-group cancellation;
- one per-user host-global Aptus lease across state roots;
- runtime-specific current train capacity admission;
- unique full-run output paths;
- parent-owned completion verification and recovery; and
- structural output integrity attestation.

Runtime-specific limit:

- CUDA can proceed through all five actions when every prerequisite passes.
- MLX-LM can proceed through all five actions for conditional single-device LoRA
  and QLoRA. Pilot and full training run uninterrupted from the pinned base, and
  crash resume remains unsupported.
- PyTorch MPS has no executable compiler path.
- LM Studio and oMLX remain inference-only and never enter managed training.

Explicitly unsupported:

- a secure multi-user or public jobs service;
- remote scheduler semantics;
- coordination with non-Aptus CUDA programs;
- full-training resume;
- direct portable child execution on Windows; and
- quality, safety, or deployment approval from run completion.

## Future seams

MLX-LM crash resume, full-parameter MLX, DoRA, a PyTorch MPS compiler, ROCm, CPU
training, cloud runners, provider provisioning, automated cost selection,
leaderboards and LLM-judge evaluation policies, exporter plugins,
experiment-tracker integration, and MCP adapters are outside the current
support contract. Optional exact-match scoring is documented separately and
does not promote run completion into quality.

## Related documentation

- [Method registry](method-registry.md)
- [Dataset schemas](dataset-schemas.md)
- [Configuration defaults](configuration-defaults.md)
- [Plan schema](plan-schema.md)
- [Model-policy snapshot](model-policy-snapshot.md)
- [Validation states](validation-states.md)
- [Current capabilities](../product/current-capabilities.md)
- [Release gates](../operations/release-gates.md)
- [2026-08-05 Phase 6 MLX-LM exact-source acceptance](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
- [Original Phase 6 MLX-LM acceptance baseline](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
