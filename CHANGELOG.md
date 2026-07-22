# Changelog

All notable changes are recorded here.

## 0.2.0 - Unreleased

### Added

- V2 fact, candidate, plan, bundle, validation, and job contracts.
- Full, LoRA, int8-LoRA, and QLoRA candidate enumeration with explicit
  distribution feasibility.
- A typed 11-descriptor method registry. Four gated executable methods are
  selectable. DoRA, BitFit, AdaLoRA, and ShareLoRA are experimental. LoReFT,
  AFLoRA, and BiLoRA are research-only.
- Bootstrap API and workbench readiness metadata for method lifecycle, evidence,
  blockers, required pilots, compiler contracts, and export contracts.
- Point and upper memory estimates with evidence records.
- Atomic no-clobber bundle compilation and deterministic archives.
- Portable dependency, model-data, measured preflight, pilot, and full-run
  entrypoints.
- Local FastAPI service, React workbench, CLI, persisted jobs, cancellation, and
  a per-user host-global Aptus execution lease.
- Immutable full-run output IDs and parent-owned completion verification.
- Pilot checkpoint continuation evidence and measured capacity admission.
- Method-specific trainable-parameter census checks before optimizer creation,
  with strict typed counts, finite values, one LoRA A/B pair per inspected target
  instance, exact optimizer membership, and a stable name-shape-dtype descriptor
  digest in measured evidence.
- Deterministic full-dataset splitting with optional `split_group` isolation,
  exact subset selection when a grouped target is attainable, closest feasible
  grouped selection otherwise, canonical and assignment digests, target and
  realized evaluation sizes, cross-rank agreement, and mutation detection during
  split and consumption.
- Fail-closed Apple Silicon discovery that records measured shared unified memory
  without claiming MPS or MLX execution.
- Structural safetensors export file-tree verification and environment bindings.
- A governed reviewed-corpus contract, Apple Silicon pilot matrix, and complete
  reconciliation ledgers for the retained Reference packet, removed TO-REVIEW
  staging files, and ignored local EXAMPLE intake.

### Changed

- Generated dependencies are now in `requirements.txt` as exact direct pins.
  This file is not described as a transitive lock.
- Model-data validation transforms and checks every canonical training row.
- Method readiness is now separate from selectability. Documentation-only
  research entries do not enter the planner.
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
- Reviewed TO-REVIEW implementation sketches after their exact snapshot and
  dispositions were preserved in the reconciliation ledger.

### Evidence status

Repository tests and static checks are necessary but not sufficient for release.
No real CUDA pilot has been completed on the current development Mac. Version
0.2.0 remains unreleased.
