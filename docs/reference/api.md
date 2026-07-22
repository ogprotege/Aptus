# API Reference

The FastAPI service is a trusted-user local interface. Default base URL:
`http://127.0.0.1:8787`. JSON request models reject unknown fields. Default
Host-header validation accepts only loopback names and addresses.

## Service endpoints

### `GET /api/v1/health`

Returns service status and version.

### `GET /api/v1/bootstrap`

Returns capabilities, stack versions, evidence registry, defaults, and
restorable bundle, report, and job references. When a bundle is restorable, its
embedded plan is returned too. A standalone plan that was never compiled is not
selected automatically after restart. Authorization shown here can be cached.
Train admission performs the authoritative deep recheck.

`capabilities.methods` contains only methods accepted by the current planner.
`capabilities.method_catalog` is the wider typed registry. Each descriptor
separates research identity, lifecycle, selectable state, compiler and export
contracts, supported backends and distributions, evidence IDs, required pilot,
and any blocker. The registry contains four selectable `gated-executable`
descriptors, four nonselectable `experimental` descriptors, and three
nonselectable `research-only` descriptors. A method appearing in that catalog
does not make it executable.

## Fact inspection

### `GET /api/v1/hardware`

Measures CUDA devices, free VRAM, host RAM, free host RAM, reserve, and disk on
the service host. On Darwin arm64 without CUDA, it returns one `mps` record for
the measured Apple shared unified-memory pool. That record is discovery
evidence, not MPS or MLX execution support. Availability remains `null` when the
host cannot measure it. Aptus never substitutes total unified memory for free
unified memory. Returns `status: unavailable` when measurement cannot run.
Hardware inspection is blocked while a managed Aptus accelerator job is active.

### `POST /api/v1/models/inspect`

```json
{
  "model_id": "provider/model",
  "revision": "0123456789abcdef0123456789abcdef01234567",
  "timeout_seconds": 10
}
```

Returns bounded provider-declared fields, provenance, and warnings. License and
training permission remain user facts.

### `POST /api/v1/profile`

```json
{
  "dataset_path": "/absolute/data.jsonl",
  "sample_limit": 512,
  "sequence_length": 1024
}
```

Returns a dataset profile. Profiling does not compile or truncate the later
canonical training set.

## Planning and compilation

### `POST /api/v1/plan`

```json
{
  "model": {
    "model_id": "provider/model",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "family": "llama",
    "parameters_b": 7,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "layers": 32,
    "context_length": 4096,
    "license_name": "license-label",
    "training_allowed": true
  },
  "hardware": {
    "discovery": "manual",
    "backend": "cuda",
    "gpu_count": 1,
    "vram_gib": 24,
    "free_vram_gib": 22,
    "supports_bf16": true,
    "supports_4bit": true,
    "supports_8bit": true,
    "host_ram_gib": 64,
    "host_ram_free_gib": 48,
    "reserve_gib": 2,
    "disk_free_gib": 200
  },
  "target": {
    "task": "sft",
    "objective": "memory",
    "sequence_length": 1024,
    "effective_batch_size": 16,
    "max_epochs": 1,
    "method_preference": "qlora",
    "evaluation_fraction": 0.1,
    "packing": false,
    "checkpoint_steps": 100
  },
  "dataset_path": "/absolute/data.jsonl",
  "sample_limit": 512
}
```

With `hardware.discovery=local-scan`, local scanning is blocked during an active
managed job. Manual facts do not probe the declared backend. A local Apple
Silicon scan produces unsupported candidate rows because v0.2 execution remains
CUDA-only.

### `GET /api/v1/plans/{plan_id}`

Returns a persisted plan or `404 plan_not_found`.

### `POST /api/v1/compile`

```json
{
  "plan_id": "plan_0123456789abcdef0123",
  "output_dir": "/absolute/new-bundle"
}
```

Returns bundle path, archive path, file list, and static report. Output and
archive are no-clobber.

## Validation

### `POST /api/v1/validate`

```json
{
  "bundle_dir": "/absolute/bundle",
  "level": "static",
  "run": false
}
```

Runtime work requested with `run=true` returns `409
runtime_validation_requires_job` and names the suggested job action. Submit it
through the jobs endpoint for cancellation and serialization.

## Jobs

### `POST /api/v1/jobs`

```json
{
  "bundle_dir": "/absolute/bundle",
  "action": "pilot",
  "confirm_full_train": false
}
```

Actions are `dependency`, `model-data`, `preflight`, `pilot`, and `train`. Submit
them in that order. The service returns `409 job_prerequisite_not_met` when the
required preceding report state has not passed. Each higher validation action
cumulatively rechecks the lower validation levels inside its own reviewable job.
Train requires `confirm_full_train: true`. Extra fields, including a full-resume
path, are rejected. Train admission deeply rechecks current pilot bindings and
capacity before the job record is created.

### `GET /api/v1/jobs`

Lists persisted job records.

### `GET /api/v1/jobs/{job_id}`

Returns a reconciled job record. Historical artifact integrity is a
completion-time attestation plus a cheap current presence status, not a repeated
deep hash of large trees.

### `POST /api/v1/jobs/{job_id}/cancel`

Requests cancellation and returns the reconciled current record. A job in its
non-cancellable completion-verification phase is not interrupted; the endpoint
returns that running or terminal record unchanged.

## Error format

Errors use a stable category with details, for example:

```json
{
  "error": "active_job_conflict",
  "message": "Aptus already has an active job."
}
```

Common status codes are `400` invalid request, `404` missing resource, `409`
state conflict, and `422` schema or feasibility failure.
