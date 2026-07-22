# API Reference

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Workbench developers, local integrators, and API clients |
| Authority | Normative reference for the Aptus v0.2 HTTP contract |
| Last reviewed | 2026-07-22 |
| Next review | 2026-10-22, or sooner when `src/aptus/api.py` changes |

The FastAPI service is a trusted-user local interface. The default origin is
`http://127.0.0.1:8787`. Install the `server` optional dependency group before
creating the app or running `aptus serve`.

Request models are strict. Unknown fields produce `422 request_validation`.
Default Host-header validation accepts `127.0.0.1`, `localhost`, `[::1]`, and
`testserver`. The CLI can explicitly allow all hosts, but that acknowledgment
does not add authentication or authorization.

## Endpoint summary

| Method and path | Purpose | Main side effect |
| --- | --- | --- |
| `GET /api/v1/health` | Service status | None |
| `GET /health` | Hidden health alias | None |
| `GET /api/v1/bootstrap` | Capabilities and restorable state | Reconciles persisted jobs |
| `GET /api/v1/hardware` | Probe service-host hardware | None; blocked by an active job |
| `POST /api/v1/models/inspect` | Inspect provider metadata | Bounded provider requests |
| `POST /api/v1/profile` | Profile a local dataset | Reads and hashes the source |
| `POST /api/v1/plan` | Create and persist a plan | Writes `STATE_DIR/plans/{plan_id}.json` |
| `GET /api/v1/plans/{plan_id}` | Load a persisted plan | None |
| `POST /api/v1/compile` | Compile and archive a plan | Writes bundle, ZIP, and current-bundle reference |
| `POST /api/v1/validate` | Run direct contract or static validation | Updates bundle report |
| `POST /api/v1/jobs` | Submit one managed runtime action | Writes job, log, lease, report, and runtime output |
| `GET /api/v1/jobs` | List persisted jobs | Reconciles each job |
| `GET /api/v1/jobs/{job_id}` | Read one reconciled job | Can reconcile stale state |
| `POST /api/v1/jobs/{job_id}/cancel` | Request cancellation | Updates job and terminates owned work |
| `GET /{full_path:path}` | Hidden workbench asset and SPA fallback | Reads the packaged or selected static build |

## Service and state

### `GET /api/v1/health`

Response:

```json
{
  "status": "ok",
  "version": "0.2.0"
}
```

`GET /health` returns the same object but is omitted from the OpenAPI schema.

### `GET /api/v1/bootstrap`

Bootstrap always returns:

- top-level `version`, `service`, `calibrated`, `stack_versions`, `defaults`,
  and `evidence`;
- top-level compatibility copies of the capability fields;
- `capabilities.backends`, which contains only `cuda`;
- `capabilities.known_backends`, which contains `cuda`, `rocm`, `mps`, and
  `cpu`;
- `capabilities.methods`, which contains the four selectable method IDs;
- `capabilities.method_catalog`, which contains all 11 descriptors;
- objectives, supported model families, and validation levels.

When restorable state exists, the response can also contain:

- `job`, chosen from the active job or latest matching bundle job;
- `plan`, loaded from the restorable bundle;
- `bundle.bundle_dir`, `archive_path`, current file list, and report.

A standalone plan that was never compiled is not restored automatically.
Bootstrap validates the plan, manifest, and copied dataset digest before
returning a bundle. It does not deep-hash large pilot or completed-run trees.
For a historical pilot or run, `authorization_current` is false unless an
active admitted train job carries the matching cached capacity check. Train
submission remains the authoritative deep authorization transaction.

## Fact inspection

### `GET /api/v1/hardware`

Success:

```json
{
  "status": "ok",
  "scope": "server-local",
  "hardware": {}
}
```

Probe failure is a normal `200` response with `status: unavailable`, an `error`,
and `manual_facts_supported: true`. CUDA hosts report each visible device. On
Darwin arm64 without CUDA, the probe reports one `mps` shared-memory inventory
record. That record does not make a candidate executable.

An active managed job causes `409 active_job_conflict`. This guard prevents a
probe from competing with Aptus-owned accelerator work.

### `POST /api/v1/models/inspect`

Request fields:

| Field | Type | Required | Default or constraint |
| --- | --- | ---: | --- |
| `model_id` | string | Yes | Non-empty provider identifier |
| `revision` | string | Yes | Requested provider revision |
| `timeout_seconds` | number | No | `10.0`, greater than 0 and at most 30 |

Example:

```json
{
  "model_id": "provider/model",
  "revision": "0123456789abcdef0123456789abcdef01234567",
  "timeout_seconds": 10
}
```

The inspector reads at most 4 MiB from each provider response. It returns
`status: ok`, `unavailable`, or `unsupported`. An `ok` response includes the
resolved immutable revision, provider-declared facts, provenance, warnings, and
`explicit_user_facts_required`. Parameter count and training permission remain
explicit user facts. Exact aliases can normalize supported dense Qwen and Gemma
model types, but prefix matching never admits MoE or multimodal variants.

### `POST /api/v1/profile`

| Field | Type | Required | Default or constraint |
| --- | --- | ---: | --- |
| `dataset_path` | string | Yes | Local path visible to the service process |
| `sample_limit` | integer or null | No | `512`; positive when not null |
| `sequence_length` | integer or null | No | `null`; positive when supplied |

Profiling reads every source row, calculates a SHA-256 digest, validates schema,
counts supported and empty rows, and computes totals. The sample limit applies
only to deterministic reservoir statistics. It does not truncate compilation.

## Planning

### `POST /api/v1/plan`

Top-level request:

| Field | Type | Required | Default |
| --- | --- | ---: | --- |
| `model` | object | Yes | None |
| `hardware` | object | Yes | None |
| `target` | object | Yes | None |
| `dataset_path` | string | Yes | None |
| `sample_limit` | integer or null | No | `512` |

`model` fields:

| Field | Type | Required | Constraint |
| --- | --- | ---: | --- |
| `model_id` | string | Yes | Provider repository identifier |
| `revision` | string | Yes | Domain layer requires immutable 40 to 64 hex |
| `family` | string | Yes | Must resolve in the current target-module catalog for adapters |
| `parameters_b` | number | Yes | Greater than 0 |
| `hidden_size` | integer | Yes | Greater than 0 |
| `intermediate_size` | integer or null | No | Greater than 0 when present |
| `layers` | integer | Yes | Greater than 0 |
| `context_length` | integer | Yes | Greater than 0 |
| `license_name` | string | Yes | Non-empty in the domain layer |
| `training_allowed` | boolean | Yes | Must be true in the domain layer |

`hardware` fields:

| Field | Type | Required | Default or constraint |
| --- | --- | ---: | --- |
| `discovery` | `manual` or `local-scan` | No | `manual` |
| `backend` | known backend enum | No | `cuda` |
| `gpu_count` | integer | Yes | Greater than 0 |
| `vram_gib` | number | Yes | Greater than 0 |
| `free_vram_gib` | number or null | No | `null`; greater than 0 when present |
| `supports_bf16` | boolean | No | False |
| `supports_4bit` | boolean | No | False |
| `supports_8bit` | boolean | No | False |
| `host_ram_gib` | number | Yes | Greater than 0 |
| `host_ram_free_gib` | number or null | No | `null`; greater than 0 when present |
| `reserve_gib` | number | No | `2.0`; non-negative |
| `disk_free_gib` | number or null | No | `null`; greater than 0 when present |

The strict request schema still requires `gpu_count`, `vram_gib`, and
`host_ram_gib` for `local-scan`. The planner ignores those submitted values and
re-probes the host, retaining only the submitted reserve. Local scan is blocked
during an active managed job. Manual mode performs no probe.

`target` fields:

| Field | Type | Required | Default or constraint |
| --- | --- | ---: | --- |
| `objective` | `quality`, `memory`, or `speed` | Yes | None |
| `sequence_length` | integer | Yes | Greater than 0 |
| `effective_batch_size` | integer | No | `16` |
| `max_epochs` | integer | No | `3` |
| `method_preference` | executable method or null | No | `null` |
| `task` | string | No | `sft`; other values are rejected by planning |
| `evaluation_fraction` | number | No | `0.1`, in `[0, 1)` |
| `packing` | boolean | No | False; true is rejected by planning |
| `checkpoint_steps` | integer | No | `100` |

Success persists and returns one full `aptus.training-plan.v2` object. When no
candidate is viable, the response is `422 no_feasible_plan` and still includes
the complete rejected candidate matrix.

### `GET /api/v1/plans/{plan_id}`

The ID must have the exact form `plan_` plus 20 lowercase hexadecimal
characters. Invalid or missing IDs return `404 plan_not_found`. A valid stored
plan is rehydrated through the strict domain contract before it is returned.

## Compilation and direct validation

### `POST /api/v1/compile`

```json
{
  "plan_id": "plan_0123456789abcdef0123",
  "output_dir": "/absolute/new-bundle"
}
```

The endpoint loads the persisted plan, compiles to an absent or empty path,
runs static validation, publishes atomically, and creates a sibling path whose
suffix is `.zip`. The bundle and archive are no-clobber. Success writes
`STATE_DIR/current-bundle.json` and returns:

```json
{
  "bundle_dir": "/absolute/new-bundle",
  "archive_path": "/absolute/new-bundle.zip",
  "files": [],
  "report": {}
}
```

The file list is the current directory tree, including mutable files that are
not part of the immutable manifest.

### `POST /api/v1/validate`

| Field | Type | Required | Default |
| --- | --- | ---: | --- |
| `bundle_dir` | string | Yes | None |
| `level` | validation level | No | `static` |
| `run` | boolean | No | False |

`contract` and `static` validation run synchronously while holding the service
validation guard. Runtime levels with `run: false` return the direct static
report plus `RUNTIME_NOT_EXECUTED`. Runtime levels with `run: true` return
`409 runtime_validation_requires_job` and a `suggested_action`. Runtime work
must use jobs so it is serialized and cancellable.

## Jobs

### `POST /api/v1/jobs`

| Field | Type | Required | Default |
| --- | --- | ---: | --- |
| `bundle_dir` | string | Yes | None |
| `action` | `dependency`, `model-data`, `preflight`, `pilot`, or `train` | No | `preflight` |
| `confirm_full_train` | boolean | No | False |

The endpoint rejects forward skips with `409 job_prerequisite_not_met`. It
rejects a competing job with `409 active_job_conflict`. Confirmation is valid
only for `train`, and training requires it to be true. No resume field is
accepted.

Train submission validates the immutable bundle and prior state, then deeply
checks the current pilot, environment, hardware, free VRAM, host RAM, disk,
checkpoint contracts, and export contracts while holding the global lease and
record locks. The queued record is written only after that admission succeeds.

### `GET /api/v1/jobs`

Returns reconciled records in reverse creation order. List responses omit the
attached validation report but include the current 16,000-byte log tail.

### `GET /api/v1/jobs/{job_id}`

Returns one reconciled record plus its current validation report. Completed
train jobs receive a cheap artifact presence status. Polling does not repeat
the completion-time recursive hash verification.

### `POST /api/v1/jobs/{job_id}/cancel`

Returns terminal records unchanged. For active work, cancellation succeeds only
through the owning `JobService`. The parent completion-verification phase is not
cancellable. A missing ID returns `404 job_not_found`.

## Error envelopes

The service emits several envelope shapes. Clients must branch on `error` and
then inspect the remaining fields.

```json
{
  "error": "path_forbidden",
  "details": "Permission denied"
}
```

```json
{
  "error": "job_prerequisite_not_met",
  "message": "The requested action is not yet allowed.",
  "action": "pilot",
  "required_state": "measured-preflight-pass",
  "current_state": "model-data-pass",
  "reason": "..."
}
```

| HTTP status | Emitted errors |
| ---: | --- |
| `400` | `invalid_request`, `filesystem_error` |
| `403` | `path_forbidden` |
| `404` | `path_not_found`, `plan_not_found`, `job_not_found`, route `not_found` |
| `409` | `path_conflict`, `active_job_conflict`, `job_prerequisite_not_met`, `runtime_validation_requires_job` |
| `422` | `request_validation`, `no_feasible_plan` |

An explicit framework HTTP error with non-object detail uses `http_error`.
Invalid Host headers are rejected by middleware before the endpoint contract.

## Static workbench routes

When a workbench build is available, the hidden `GET /{full_path:path}` route
serves matching non-API assets inside the selected static root. Other non-API
paths return `index.html` for
client-side routing. Unknown paths under `/api/` return JSON `404 not_found`.

## Related documentation

- [Configuration defaults](configuration-defaults.md)
- [CLI reference](cli.md)
- [Method registry](method-registry.md)
- [Evidence records](evidence-records.md)
- [Run states](run-states.md)
- [Error and finding codes](error-codes.md)
