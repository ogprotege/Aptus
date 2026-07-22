# Bundle Manifest

`bundle-manifest.json` is the compiler integrity root for an Aptus bundle.

## Top-level fields

| Field | Meaning |
|---|---|
| `schema_version` | `aptus.bundle.v2` |
| `compiler` | Compiler name and version |
| `stack_versions` | Direct runtime package versions selected by Aptus |
| `plan_id` | Bound plan identity |
| `plan_sha256` | Digest of `plan.json` |
| `candidate_id` | Bound selected candidate identity |
| `formula_version` | Memory formula version |
| `entrypoints` | Portable validation, preflight, child training, and parent run programs |
| `validation` | Supported levels and required state before full training |
| `files` | Sorted compiler-managed path, size, and SHA-256 entries |

## Expected bundle tree

```text
bundle/
  README.md
  bundle-manifest.json
  candidates.json
  decision-report.md
  evidence.jsonl
  plan.json
  plan_contract.py
  preflight.py
  requirements.txt
  run.py
  runbook.md
  runtime_lease.py
  train.py
  validate.py
  .validation-report.lock  # mutable report lock; not in manifest files
  validation-report.json  # mutable runtime report; not in manifest files
  preflight-metrics.json  # created by measured preflight; not in manifest files
  accelerate_config.yaml
  config/
    accelerate.yaml
    trainer.json
  data/
    dataset.<source suffix>
    pilot-sample.jsonl
    training.jsonl
  profiles/
    dataset.json
    hardware.json
    model.json
```

`requirements.txt` contains exact direct pins. It is not a transitive lock.

## Runtime exclusions

The compiler file list excludes `.validation-report.lock`,
`validation-report.json`, `preflight-metrics.json`, and runtime directories such
as `pilot-output/` and `runs/`. These paths can change after compilation. The
report binds `preflight-metrics.json` by digest and structured content. Pilot and
run outputs use separate metrics and recursive file-tree manifests. The lock
file carries no evidence.

## Mutation rule

Any compiler-managed file change invalidates the manifest. Generate a new plan
and bundle instead of editing in place. Runtime output must stay under its
assigned directory and cannot replace a prior run.
