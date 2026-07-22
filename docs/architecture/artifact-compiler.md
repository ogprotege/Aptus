# Artifact Compiler

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
6. Build a bounded pilot pressure set.
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
- `config/trainer.json`: Trainer arguments plus the selected descriptor's
  compiler and export identifiers.
- `config/accelerate.yaml`: distributed launch configuration when applicable.
- `validate.py`: portable validation parent.
- `preflight.py`: selected-method synthetic CUDA check.
- `train.py`: pilot and full training child implementation.
- `run.py`: portable full-run parent and completion promoter.
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
- a bounded, repeated pilot pressure set as `data/pilot-sample.jsonl`.

These are cleartext copies and are also present in the ZIP.

The generated full trainer computes the train and evaluation split at runtime.
It keeps declared `split_group` units intact, binds canonical and assignment
digests, detects mutation, and records requested and realized split sizes. The
same generated runtime enforces the selected method's trainable scope before it
constructs an optimizer.

## Integrity boundary

The compiler manifest covers compiler-created inputs and programs. Runtime
directories such as `pilot-output/` and `runs/` are intentionally outside that
immutable file list. Runtime reports bind those outputs separately by recursive
path, size, and digest evidence.

Changing a compiler-managed file invalidates the bundle. Recompile rather than
editing generated source or configuration in place.

## Current method boundary

The typed registry exposes four selectable `gated-executable` methods. The
compiler can emit their guarded single-device and DDP configurations, plus
conditional LoRA FSDP. Experimental and research-only descriptors have no
compiler or export identifiers and cannot enter this boundary. The compiler
refuses full-parameter FSDP and quantized FSDP. It does not emit cloud
infrastructure, provider provisioning, evaluation pipelines, MCP tools, or
deployment exporters.
