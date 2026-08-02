# Bundle and Manifest Reference

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Bundle operators, compiler maintainers, and security reviewers |
| Authority | Normative reference for `aptus.bundle.v3` and its mutable runtime boundary |
| Last reviewed | 2026-07-30 |
| Next review | 2026-10-22, or sooner when generation or manifest validation changes |

`bundle-manifest.json` is the immutable integrity root for a compiled Aptus
bundle. It covers compiler-created inputs, code, configuration, and evidence.
It does not cover the manifest file itself or the explicitly allowed mutable
runtime paths.

The bundle fingerprint is the SHA-256 of the manifest bytes. A compiled project
revision stores that fingerprint as `artifact_fingerprint`. It stores the ZIP
separately by SHA-256 and exact byte size because the archive is not a bundle
manifest entry. Validation, jobs, bootstrap, and project recovery compare the
revision's recorded identity with the exact bundle they use.

## Manifest object

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Exact value `aptus.bundle.v3` |
| `compiler` | object | Compiler `name` and `version` |
| `stack_versions` | object | Exact direct runtime package versions |
| `plan_id` | string | Bound plan identity |
| `plan_sha256` | string | SHA-256 of the bundled `plan.json` bytes |
| `policy_snapshot_path` | string | Exact path `policy/model-policy-snapshot.v1.json` |
| `policy_snapshot_sha256` | string | SHA-256 of the canonical snapshot bytes; must equal the plan binding and manifested file digest |
| `candidate_id` | string | Bound recommended candidate |
| `formula_version` | string | Bound memory formula version |
| `entrypoints` | object | Paths for `run`, `train`, `preflight`, `validate`, and MLX `reload` when present |
| `validation` | object | Supported levels and the state required before full training |
| `files` | array | Sorted path, size, and SHA-256 entries |

The entrypoint map is:

```json
{
  "run": "run.py",
  "train": "train.py",
  "preflight": "preflight.py",
  "validate": "validate.py"
}
```

An MLX-LM bundle also declares `"reload": "reload.py"`.

The manifest declares all six public validation levels and
`required_before_full_training: pilot-pass`.

## Compiler-managed bundle tree

```text
bundle/
  README.md
  accelerate_config.yaml
  bundle-manifest.json
  candidates.json
  decision-report.md
  evidence.jsonl
  plan.json
  plan_contract.py
  policy_snapshot.py
  preflight.py
  requirements.txt
  run.py
  runbook.md
  runtime_lease.py
  reload.py                 MLX-LM only
  train.py
  validate.py
  config/
    accelerate.yaml
    mlx-lm.yaml              MLX-LM only
    trainer.json
  data/
    dataset.<source suffix>
    pilot-sample.jsonl
    training.jsonl
    mlx/                     MLX-LM only
      split-contract.json
      train.jsonl
      valid.jsonl
  profiles/
    dataset.json
    hardware.json
    model.json
  policy/
    model-policy-snapshot.v1.json
```

After compilation, the directory also contains mutable
`validation-report.json` and `.validation-report.lock`. They are created by the
static validation performed before atomic publication and are intentionally not
manifested.

### File purposes

| Path | Purpose |
| --- | --- |
| `plan.json` | Portable plan with a bundle-relative dataset path |
| `candidates.json` | Complete 12-row candidate matrix |
| `evidence.jsonl` | One resolved plan evidence record per line |
| `decision-report.md` | Human-readable recommendation, policy, and candidate ledger |
| `profiles/*.json` | Model, dataset, and hardware fact snapshots |
| `data/dataset.*` | Byte-for-byte copied source dataset |
| `data/training.jsonl` | Every valid non-empty source row, serialized deterministically |
| `data/pilot-sample.jsonl` | Repeated bounded pressure set, at least 32 rows or two effective batches |
| `requirements.txt` | Exact direct method pins, not a transitive lock |
| `config/trainer.json` | Bound trainer and method-compiler configuration |
| `config/accelerate.yaml` | Canonical Accelerate launch configuration |
| `accelerate_config.yaml` | Compatibility copy of the canonical Accelerate configuration |
| `plan_contract.py` | Self-contained plan and manifest validation |
| `policy_snapshot.py` | Self-contained schema validator and generic policy evaluator; it does not import Aptus |
| `policy/model-policy-snapshot.v1.json` | Canonical deterministic snapshot generated from the host registry |
| `runtime_lease.py` | Self-contained per-user host execution lease |
| `reload.py` | MLX-only fresh-process adapter reload and one-to-four-token generation verifier |
| `validate.py` | Public portable validator and report writer |
| `preflight.py` | CUDA cumulative `--level` executor and lease-bound action owner; MLX argument-free Apple silicon and pinned-dependency gate with no lease |
| `train.py` | Runtime-specific model-data, preflight, pilot, and training implementation |
| `run.py` | Runtime-specific action owner, verifier, and full-run parent |
| `data/mlx/*` | MLX-only disjoint, in-split-padded train and validation data plus `aptus.mlx-split.v1` counts |

## Entrypoint semantics

### `validate.py`

`validate.py` is the public portable validation command. In a CUDA bundle it
binds selected device visibility, acquires the portable execution lease, and
invokes `preflight.py --level <requested>` under the inherited lease token. When
that child succeeds, `validate.py` writes a bound `validation-report.json`.
CUDA pilot orchestration also performs marker- and attestation-guarded cleanup
of stale Aptus-owned pilot roots.

In an MLX-LM bundle, `validate.py` owns `--level` and the cumulative validation
ladder but does not acquire the execution lease itself. For dependency and
higher levels it invokes the argument-free `preflight.py` platform-and-pin
gate. It performs model-data validation directly and invokes
`run.py --bounded-smoke` or `run.py --pilot` for measured work. Those `run.py`
processes acquire the portable execution lease before launching `train.py`.
The MLX validator verifies the resulting evidence and writes the bound
`validation-report.json`.

### `preflight.py`

In a CUDA bundle, `preflight.py` parses `--level`, enters the portable execution
lease, and executes the requested level cumulatively:

1. contract validation;
2. Python static parsing;
3. exact direct dependency checks;
4. `train.py --preflight-model-data`;
5. runtime-specific measured preflight; and
6. runtime-specific pilot.

When CUDA `validate.py` launches this program, `preflight.py` reuses the
inherited lease token.

The MLX-LM bundle ships a different `preflight.py`. It takes no arguments,
acquires no execution lease, and only asserts Apple silicon macOS plus the
pinned `mlx` and `mlx-lm` versions. MLX `validate.py` owns the level ladder.
Its measured actions run through the lease-owning MLX `run.py`.

The selected-method synthetic CUDA check is specifically
`train.py --synthetic-preflight`. Calling `preflight.py --level pilot` executes
every lower level first. The MLX pilot is one uninterrupted training process
plus a fresh adapter-reload process.

### `train.py`

`train.py` is a lease-bound child. Operators should not launch it directly. It
implements:

- exact model and tokenizer loading;
- method preparation and trainable-scope census;
- canonical-row transformation;
- CUDA synthetic preflight, checkpoint-continuation pilot, splitting, lazy
  reads, full training, and checkpoints; or
- MLX measured adapter updates, uninterrupted pilot and full-duration adapter
  training, and non-resumable weight snapshots; and
- structural export creation.

### `run.py`

`run.py` owns runtime outputs and verification. CUDA full training requires
explicit confirmation, creates or validates one fresh `runs/run_*` output,
launches the selected single or interpreter-bound Accelerate command, waits for
aggregate completion, persists pending evidence, verifies output, and promotes
the report. MLX `run.py` owns bounded smoke, pilot, and confirmed full actions in
fresh output roots, then seals their artifacts. Neither runtime supports full
training resume.

## Mutable runtime paths

The manifest permits only these unmanifested files and prefixes:

```text
.validation-report.lock
model-data-evidence.json
validation-report.json
preflight-metrics.json
pilot-output/
runs/
```

`model-data-evidence.json` is written into the bundle root by the MLX model-data
gate. It binds `aptus.mlx-model-data-evidence.v1`, and the validation report
binds its SHA-256 under `bindings.model_data_evidence`. It is absent until that
gate passes, and absent entirely for CUDA bundles. The compiled `bundle.zip` is
written before model-data runs, so the archive never contains it.

Any other unmanifested file invalidates the bundle. This includes a virtual
environment, dependency cache, editor backup, `__pycache__`, or manually added
configuration. Create the Python environment outside the bundle and run the
entrypoints with that interpreter.

### Measured preflight output

CUDA `preflight-metrics.json` uses `aptus.preflight-metrics.v1`. It binds the
candidate, method, precision, quantization, distribution, world size, measured
CUDA peak, and trainable-parameter census. MLX uses
`aptus.runtime-metrics.v1`, including exact target binding, optimizer updates,
adapter delta, MLX memory, unified-memory admission, and adapter manifest.

### Pilot output

Each CUDA pilot first claims a fresh `runs/pilot_*` root with an
`.aptus-pilot-run.json` marker. It creates:

```text
runs/pilot_<id>/
  .aptus-pilot-run.json
  phase-1/
    checkpoint-1/
    final/
    final-export.json
    metrics.json
  phase-2/
    checkpoint-2/
    final/
    final-export.json
    metrics.json
pilot-output/
  metrics.json
```

The aggregate `pilot-output/metrics.json` binds both phase records, both
checkpoint manifests, the shared trainable census, checkpoint continuation,
and measured checkpoint and export sizes. Phase directories remain under their
run ID so the aggregate can bind their paths.

An MLX pilot instead creates:

```text
pilot-output/
  metrics.json
  pilot_<id>/
    .aptus-run.json
    adapters/
      adapter_config.json
      adapters.safetensors
    training-metrics.json
    reload-evidence.json
    artifact-manifest.json
    metrics.json
```

The owned MLX metrics require at least two optimizer updates, finite train and
validation losses, exact target binding, positive MLX peak and adapter delta,
live headroom, and immutable artifacts. `reload-evidence.json` proves that a
fresh child loaded the pinned base plus adapter and generated one to four tokens.
It does not prove training resume.

### Full-run output

Each admitted CUDA full run receives a new output:

```text
runs/run_<id>/
  .aptus-run.json
  checkpoint-*/
  final/
  final-export.json
  metrics.json
```

`final-export.json` is an `aptus.final-export.v1` structural file-tree manifest.
`metrics.json` binds the run, plan, candidate, rank peaks, finite guards,
trainable census, dataset split, optimizer membership, and final export. Parent
verification promotes only matching pending evidence.

An admitted MLX full run receives:

```text
runs/run_<id>/
  .aptus-run.json
  final/
    adapter_config.json
    adapters.safetensors
  training-metrics.json
  reload-evidence.json
  artifact-manifest.json
  final-export.json
  metrics.json
```

MLX `final-export.json` uses `aptus.mlx-final-export.v1`. The run starts from the
pinned base and completes without interruption. Its periodic MLX files are
weight snapshots, not resumable checkpoints.

Managed job records and logs live under the service state directory, outside
the bundle.

## Integrity rules

Manifest validation rejects:

- a symlink bundle root;
- any symlink anywhere in the bundle tree;
- an absent or malformed manifest;
- a plan digest mismatch;
- an empty, duplicate, absolute, or parent-traversing file entry;
- a missing, changed, or size-mismatched manifested file; and
- any unmanifested file outside the allowed mutable paths.

The manifest does not make the directory immutable at the filesystem level.
Changing a compiler-managed file invalidates the bundle. Recompile instead of
editing a generated artifact.

## Deterministic ZIP boundary

The archive sorts paths, uses a fixed 1980 timestamp, stores mode `0644`, and
uses deflate level 9. Publication is no-clobber and uses a temporary file plus a
hard-link claim.

The ZIP excludes:

- `.validation-report.lock`;
- `validation-report.json`;
- `preflight-metrics.json`;
- every file below `pilot-output/`; and
- every file below `runs/`.

Therefore the archive is a deterministic compiler artifact, not a snapshot of
later validation or training evidence.

## Cleartext data notice

The copied source, canonical training set, pilot pressure set, and MLX train and
validation files are cleartext inside the bundle and archive. Runtime CUDA
checkpoints, MLX weight snapshots, metrics, adapters, and exports can also
contain sensitive derived artifacts. The manifest provides integrity, not
encryption, access control, consent review, or retention policy.

## Related documentation

- [Plan schema](plan-schema.md)
- [Dataset schemas](dataset-schemas.md)
- [Configuration defaults](configuration-defaults.md)
- [Validation states](validation-states.md)
- [Run states](run-states.md)
- [Security boundaries](../architecture/security-boundaries.md)
