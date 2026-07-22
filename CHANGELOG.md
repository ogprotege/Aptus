# Changelog

All notable changes are recorded here.

## 0.2.0 - Unreleased

### Added

- V2 fact, candidate, plan, bundle, validation, and job contracts.
- Full, LoRA, int8-LoRA, and QLoRA candidate enumeration with explicit
  distribution feasibility.
- Point and upper memory estimates with evidence records.
- Atomic no-clobber bundle compilation and deterministic archives.
- Portable dependency, model-data, measured preflight, pilot, and full-run
  entrypoints.
- Local FastAPI service, React workbench, CLI, persisted jobs, cancellation, and
  a per-user host-global Aptus execution lease.
- Immutable full-run output IDs and parent-owned completion verification.
- Pilot checkpoint continuation evidence and measured capacity admission.
- Structural safetensors export file-tree verification and environment bindings.

### Changed

- Generated dependencies are now in `requirements.txt` as exact direct pins.
  This file is not described as a transitive lock.
- Model-data validation transforms and checks every canonical training row.
- Runtime execution follows five ordered actions: dependency, model-data,
  preflight, pilot, and train.
- Full training uses `python run.py --confirm-full-train` for portable bundles.
- Full-parameter FSDP is fail-closed. LoRA FSDP is conditional.
- A successful process exit is no longer enough to mark a training job complete.
  The parent verifies and promotes pending evidence.

### Removed

- Pass-through full-training resume.
- Claims of automatic FP16 fallback.
- Claims that analysis alone proves fit, speed, cost, or quality.

### Evidence status

Repository tests and static checks are necessary but not sufficient for release.
No real CUDA pilot has been completed on the current development Mac. Version
0.2.0 remains unreleased.
