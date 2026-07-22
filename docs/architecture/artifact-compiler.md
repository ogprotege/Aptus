# Artifact Compiler

> **Status:** Active | **Authority:** Normative architecture | **Applies to:** Aptus 0.2 | **Audience:** Contributors and operators | **Last reviewed:** 2026-07-22 | **Review by:** 2027-01-22 or when bundle generation changes

The compiler turns one identity-bound plan and selected candidate into a portable
directory and deterministic ZIP. It does not train a model.

## Inputs

- Valid `aptus.training-plan.v2` payload.
- Recommended candidate embedded in that plan.
- Source dataset whose content still matches the profiled digest.
- Empty or absent output directory.
- Archive path that does not already exist.

## Publication algorithm

1. Resolve and validate the output path.
2. Create a temporary sibling directory.
3. Copy the exact source dataset.
4. Verify its digest against the plan.
5. Validate every supported source-schema row and serialize it deterministically
   into `data/training.jsonl`.
6. Build the runtime-specific data artifacts. CUDA receives a bounded pilot
   pressure set. MLX also receives disjoint, microbatch-padded train and
   validation files with an `aptus.mlx-split.v1` contract.
7. Emit plan, evidence, configuration, direct pins, generated programs, and
   reports.
8. Hash compiler-managed files into `bundle-manifest.json`.
9. Run static bundle validation.
10. Atomically publish the directory.
11. Create a deterministic no-clobber archive.

Any failure removes the temporary directory. The compiler does not merge into or
replace a populated output.

## Generated execution material

- `requirements.txt`: exact direct method pins, not a transitive lock.
- `config/trainer.json`: runtime-neutral training values plus the selected
  descriptor's compiler and export identifiers.
- `config/accelerate.yaml`: CUDA distributed launch configuration when
  applicable.
- `config/mlx-lm.yaml`: MLX-LM LoRA or QLoRA configuration for MLX bundles.
- `validate.py`: portable validation parent.
- `preflight.py`: cumulative runtime-validation orchestrator for dependency,
  model-data, measured-preflight, and pilot levels.
- `train.py`: runtime-specific measured preflight, pilot, and full-training
  child implementation. CUDA measured preflight is synthetic. MLX measured
  preflight performs one bounded adapter update on the pinned model and data.
- `run.py`: portable full-run parent and completion promoter.
- `reload.py`: fresh-process adapter reload and bounded generation for MLX
  bundles.
- `plan_contract.py`: self-contained plan and manifest contract checks.
- `runtime_lease.py`: self-contained per-user host lease and process-group
  coordination used by portable entrypoints.

Generated code reads semantic values from `plan.json`. It validates plan and
candidate identities before runtime use. Distributed launch uses the active
Python interpreter's Accelerate module rather than an unbound shell executable.

## Dataset outputs

The bundle contains:

- the copied source as `data/dataset.*`;
- every validated source-schema row as deterministic `data/training.jsonl`;
- a bounded, repeated CUDA pilot pressure set as `data/pilot-sample.jsonl`; and
- for MLX bundles, disjoint `data/mlx/train.jsonl` and `valid.jsonl` files,
  padded only within each split to complete the final microbatch, plus
  `data/mlx/split-contract.json`.

These are cleartext copies and are also present in the ZIP.

The generated CUDA full trainer computes the train and evaluation split at
runtime. It keeps declared `split_group` units intact, binds canonical and
assignment digests, detects mutation, and records requested and realized split
sizes. MLX uses its compiler-created disjoint split. It does not claim the CUDA
group-aware split contract. Each generated runtime enforces its selected
method's trainable scope before optimizer construction.

## Integrity boundary

The compiler manifest covers compiler-created inputs and programs. Runtime
directories such as `pilot-output/` and `runs/` are intentionally outside that
immutable file list. Runtime reports bind those outputs separately by recursive
path, size, and digest evidence.

Changing a compiler-managed file invalidates the bundle. Recompile rather than
editing generated source or configuration in place.

## Current method boundary

The typed registry exposes four selectable `gated-executable` methods. The CUDA
compiler can emit their guarded single-device and DDP configurations, plus
conditional LoRA FSDP. The MLX compiler emits supported LoRA and QLoRA adapter
execution only. MLX full-parameter training and DoRA are unimplemented.
Experimental and research-only descriptors have no compiler or export
identifiers and cannot enter this boundary. The compiler refuses full-parameter
FSDP and quantized FSDP. It does not emit cloud infrastructure, provider
provisioning, evaluation pipelines, MCP tools, or deployment exporters.

## Related documentation

- [Bundle manifest](../reference/bundle-manifest.md)
- [Generated-code workflow](../contributing/generated-code.md)
- [Data and identity flow](data-and-identity-flow.md)
- [Security boundaries](security-boundaries.md)
