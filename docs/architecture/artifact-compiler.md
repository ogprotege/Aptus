# Artifact Compiler

> **Status:** Active | **Authority:** Normative architecture | **Applies to:** Aptus 0.2 | **Audience:** Contributors and operators | **Last reviewed:** 2026-08-05 | **Review by:** 2027-01-27 or when bundle generation changes

The compiler turns one identity-bound plan and selected candidate into a portable
directory and deterministic ZIP. It does not train a model.

## Inputs

- Valid `aptus.training-plan.v5` payload whose current policy decision,
  canonical policy-snapshot digest,
  decision source, optional inspection receipt, candidate decision links, and
  exact-path binding all revalidate.
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

## Project publication and conflict cleanup

The API publishes the compiled bundle to project history with a
compare-and-swap against the revision that authorized compilation. The new
revision records the bundle-manifest SHA-256 as `artifact_fingerprint`. It also
records the ZIP SHA-256 and exact byte size.

Another writer can advance the project while compilation is running. In that
case, Aptus returns `project_revision_conflict` and removes only outputs it can
still prove are the files and directory created by that compile attempt. The
cleanup compares directory or file identity plus the recorded manifest or ZIP
digest and size. If another process replaces either path, Aptus preserves that
replacement. An initially empty caller-created directory is recreated with its
original mode only when the Aptus-created directory still owns the path.

## Generated execution material

- `requirements.txt`: exact direct method pins, not a transitive lock.
- `config/trainer.json`: runtime-neutral training values plus the selected
  descriptor's compiler and export identifiers.
- `config/accelerate.yaml`: CUDA distributed launch configuration when
  applicable.
- `config/mlx-lm.yaml`: MLX-LM LoRA or QLoRA configuration for MLX bundles.
- `validate.py`: portable validation parent.
- `preflight.py`: for CUDA, the cumulative `--level` orchestrator for dependency,
  model-data, measured-preflight, and pilot levels. For MLX-LM, an
  argument-free Apple-silicon and pinned-dependency gate that `validate.py`
  spawns as a subprocess; the MLX level sequencing lives in `validate.py`.
- `train.py`: runtime-specific measured preflight, pilot, and full-training
  child implementation. CUDA measured preflight is synthetic. MLX measured
  preflight performs one bounded adapter update on the pinned model and data.
- `run.py`: portable full-run parent and completion promoter.
- `reload.py`: fresh-process adapter reload and bounded generation for MLX
  bundles.
- `plan_contract.py`: self-contained plan and manifest contract checks.
- `policy_snapshot.py`: self-contained snapshot validation and generic policy
  evaluation.
- `runtime_lease.py`: self-contained per-user host lease and process-group
  coordination used by portable entrypoints.

Generated code reads semantic values from `plan.json`. It validates plan and
candidate identities before runtime use. The installed Aptus host compiler also
compares a v5 plan's snapshot digest and decision with the current registry and
returns `replan_required` when either is non-current. V4, v3, v2, and
schema-less plans are preserved but never compiled or relabeled. Distributed
launch uses the active Python interpreter's Accelerate module rather than an
unbound shell executable.

The historical Phase 3 `aptus.bundle.v2` contract used handwritten
self-contained policy checks. Phase 4 changed the bundle contract to
`aptus.bundle.v3` and added the portable policy snapshot plus generic evaluator;
Phase 5 subsequently removed browser policy reconstruction without changing
this compiler contract. Phase 6 now adds the second registry-driven
`model.qwen2-24l.mlx-qlora` configuration-footprint policy without changing the
bundle schema. Its path is
`mlx-lm.qlora.single.dense-causal-lm.v1`, and generated bundles carry the same
two-policy snapshot bytes and digest as their v5 plan and manifest. Because the
registry addition changes those canonical bytes, pre-expansion v5 plans require
replanning.

The compiler path and its exact current-contract runtime evidence are complete.
The [2026-08-05 Qwen2 MLX-LM acceptance
record](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
binds two clean `aptus.training-plan.v5` and `aptus.bundle.v3` repetitions at
commit `14ed44b52a76bb84d8d9db4f2303951aa641339b` through dependency, model-data,
measured preflight, pilot and reload, confirmed full training, final reload and
export, parent-owned promotion, and `measured-run-pass`. This closes the Phase 6
runtime gate only for the exact recorded artifact, revision, host, runtime,
dataset, snapshot, plan, and bundle. Compiler eligibility remains a
configuration-footprint decision, so another matching artifact must establish
its own runtime evidence. CUDA target-runtime acceptance remains open, and the
record establishes neither model quality nor production throughput.

The canonical program bytes live under
`src/aptus/_bundle_programs/{cuda,mlx}/`. `generation.py` reads them through
`importlib.resources`; it does not keep parallel Python string copies. The wheel
declares both resource trees as package data. The PyInstaller specification
collects them into the frozen sidecar. Parity tests compare each emitted file
and manifest digest with its resource bytes across source, wheel, and frozen
layouts.

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
execution only. Within that generic dense support, the Qwen2 24-layer policy
binds MLX-LM QLoRA to `dense-causal-lm.v1` and all seven attention and MLP
projection targets. MLX full-parameter training and DoRA are unimplemented.
Experimental and research-only descriptors have no compiler or export
identifiers and cannot enter this boundary. The compiler refuses full-parameter
FSDP and quantized FSDP. It does not emit cloud infrastructure, provider
provisioning, evaluation pipelines, MCP tools, or deployment exporters.

## Related documentation

- [Bundle manifest](../reference/bundle-manifest.md)
- [Generated-code workflow](../contributing/generated-code.md)
- [Data and identity flow](data-and-identity-flow.md)
- [Security boundaries](security-boundaries.md)
