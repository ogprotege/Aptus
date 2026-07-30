# API Reference

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Workbench developers, local integrators, and API clients |
| Authority | Normative reference for the Aptus v0.2 HTTP contract |
| Last reviewed | 2026-07-29 |
| Next review | 2026-10-27, or sooner when `src/aptus/api.py`, `src/aptus/api_contracts.py`, or a client contract changes |

The FastAPI service is an authenticated single-user local interface when
started by `aptus serve`. The default origin is `http://127.0.0.1:8787`.
Install the `server` optional dependency group before creating the app or
running the command.

Request models are strict. Unknown fields produce `422 request_validation`.
Default Host-header validation accepts `127.0.0.1`, `localhost`, `[::1]`, and
`testserver`. The CLI can explicitly allow all hosts for non-loopback serving.
Session authentication remains active in that mode, but TLS, tenant isolation,
filesystem scoping, and worker isolation do not appear automatically.

Every success route has an explicit Pydantic response model. The HTTP contract
identity is `aptus.api.v1`. Health and bootstrap return that identity as
`api_contract_version`; clients must reject an unknown nonempty value. The
generated OpenAPI artifact is checked in at
[`openapi.v1.json`](openapi.v1.json). Regenerate or verify it with:

```bash
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/generate_openapi.py
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/generate_openapi.py --check
npm --prefix web run openapi:generate
npm --prefix web run openapi:check
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/check_client_contracts.py
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/verify_versions.py
```

Run the generators after changing a request or response contract. Run the check
forms without changing files in validation and CI.

The OpenAPI artifact and
[`web/src/generated/openapi.ts`](../../web/src/generated/openapi.ts) are
generated. The TypeScript file supplies schema and path types to the React API
layer. It is not a complete generated SDK. Request construction, response
normalization, UI domain types, and presentation remain maintained in
`web/src/api.ts` and `web/src/types.ts`. Swift native decoders also remain
maintained source. Their covered endpoints and required runtime-inventory fields
are checked against OpenAPI by `tools/check_client_contracts.py`.

## Authentication

`aptus serve` generates a fresh token for every launch and prints both a
workbench handoff URL and the token for API clients. The handoff URL has this
shape:

```text
http://127.0.0.1:8787/?aptus_session_token=TOKEN
```

A valid token on a GET to a public workbench path produces an immediate `303`
response. The response sets `aptus_desktop_session=TOKEN` with `HttpOnly`,
`SameSite=Strict`, and `Path=/`, then redirects to the same path without the
token parameter. Other query parameters are preserved. An invalid handoff token
returns `403 desktop_session_required`.

API clients can avoid the cookie exchange:

```http
Authorization: Bearer TOKEN
```

Only `GET /api/v1/health`, `GET /health`, and non-API static workbench assets
are public. Every other `/api` route, plus `/docs`, `/redoc`, and
`/openapi.json`, requires the valid cookie or bearer token. The stable rejection
code remains `desktop_session_required` for both desktop and ordinary serve
sessions.

The CLI disables Uvicorn access logs to keep the handoff query out of normal
request logging. Operators must still protect terminal output, browser history,
and the token itself. Explicit non-loopback serving uses plain HTTP, so a
network observer can steal either credential. Put that mode behind approved TLS
and network controls.

Programmatic `create_app()` callers can omit `session_token`. That direct mode
has no application authentication. The caller owns its security boundary.

## Endpoint summary

| Method and path | Purpose | Main side effect |
| --- | --- | --- |
| `GET /api/v1/health` | Service status | None |
| `GET /health` | Hidden health alias | None |
| `GET /api/v1/bootstrap` | Capabilities and restorable state | Reconciles persisted jobs |
| `GET /api/v1/hardware` | Probe service-host hardware | None; blocked by an active job |
| `GET /api/v1/platform` | Probe Apple platform and unified-memory facts | None |
| `GET /api/v1/runtimes` | Probe configured Python training runtimes | Starts bounded interpreter probes |
| `POST /api/v1/runtimes/configure` | Validate and persist one training interpreter | Writes private runtime configuration |
| `GET /api/v1/inference/services` | Probe known LM Studio and oMLX loopback origins | Bounded local requests |
| `POST /api/v1/inference/models` | List models from one explicit local inference service | Bounded local request |
| `POST /api/v1/inference/generate` | Generate through one explicit local inference service | Local inference request |
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
| `POST /api/v1/projects` | Create a named project | Writes a private project manifest |
| `GET /api/v1/projects` | List healthy named projects | Can quarantine an unreadable project manifest |
| `GET /api/v1/projects/{project_id}` | Read a project and latest immutable revision | Can repair a manifest to its latest safe revision |
| `GET /api/v1/projects/{project_id}/revisions` | List immutable revision summaries | None |
| `GET /api/v1/projects/{project_id}/revisions/{revision_id}` | Read one content-hashed revision | None |
| `POST /api/v1/projects/{project_id}/recover` | Append a new revision from an older verified revision | Writes a new revision with authorization false |
| `GET /{full_path:path}` | Hidden workbench asset and SPA fallback | Reads the packaged or selected static build |

## Service and state

### `GET /api/v1/health`

Response:

```json
{
  "status": "ok",
  "version": "0.2.0",
  "api_contract_version": "aptus.api.v1"
}
```

`GET /health` returns the same object but is omitted from the OpenAPI schema.

### `GET /api/v1/bootstrap`

Bootstrap always returns:

- top-level `api_contract_version`, `version`, `service`, `calibrated`, `stack_versions`, `defaults`,
  and `evidence`;
- top-level compatibility copies of the capability fields;
- `capabilities.backends`, which contains `cuda` and `mps`;
- `capabilities.known_backends`, which contains `cuda`, `rocm`, `mps`, and
  `cpu`;
- `capabilities.methods`, which contains the four selectable method IDs;
- `capabilities.method_catalog`, which contains all 11 descriptors;
- training-runtime IDs, inference-service IDs, objectives, supported model
  families, and validation levels.
- `projects`, the current `project` when present, and its `project_history`.

On macOS, bootstrap selects `mps`, `mlx-lm`, and an 8 GiB reserve as the new
workspace defaults. Other hosts retain `cuda`, `transformers-peft-cuda`, and a
2 GiB reserve. These defaults do not replace measured hardware facts.

When restorable state exists, the response can also contain:

- `job`, chosen from the active job or latest matching bundle job;
- `plan`, loaded from the current project's latest valid revision or restorable bundle;
- `bundle.bundle_dir`, `archive_path`, current file list, and report.

If the current revision or restorable bundle contains a v3 plan, a v2 plan, or
a plan with no schema identifier, bootstrap does not return it as `plan` or
restore its bundle into the executable workspace. It returns `replan_required`
with `status`, optional `plan_id`, optional `found_schema`, required v4 schema,
source, project identities when known, and an operator message. The same result
applies to a v4 plan whose policy decision or registered path is stale. The
source is `project-revision` or `compiled-bundle`. The saved plan and source
revision stay unchanged. Create a deterministic v4 plan from the preserved
facts.

A standalone project plan can be restored from the current immutable revision.
Bootstrap returns a current-project bundle only when its resolved path, plan ID,
selected candidate, and manifest fingerprint match that exact revision. It also
validates the manifest and copied dataset digest. An active or completed job
must appear in the revision's job IDs and use the same resolved bundle path.
An identical plan or bundle from another project cannot enter the restored
surface. Bootstrap does not deep-hash large pilot or completed-run trees.
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
record. The inventory alone does not make a candidate executable. An MLX-LM
candidate also requires a registered compiler, a compatible exact Python
runtime, model-data validation, and measured runtime gates.

An active managed job causes `409 active_job_conflict`. This guard prevents a
probe from competing with Aptus-owned accelerator work.

### `GET /api/v1/platform`

On Apple Silicon this returns macOS version and build, chip identity, logical
CPU count, the built-in Metal GPU core count when reported, total unified
memory, measured current availability when obtainable, memory pressure, swap
facts, Metal's recommended working set when obtainable, and separate MLX,
MLX-LM, and PyTorch MPS runtime facts. Missing measurements remain `null`. A
non-Apple host returns `status: unsupported`.

### `GET /api/v1/runtimes`

Returns `aptus.runtime-inventory.v1`. Every record identifies the exact Python
executable, source, Python version, package versions, measured import
availability, and executable compatibility for `mlx-lm`, `pytorch-mps`, and
`transformers-peft-cuda`. The `available` mapping reports successful capability
probes. The separate `compatible` mapping reports interpreters Aptus can use.
For MLX-LM, compatibility requires the exact reviewed `mlx` and `mlx-lm` pins.
The endpoint can use
the explicit `APTUS_MLX_PYTHON`, `APTUS_PYTORCH_PYTHON`, and
`APTUS_CUDA_PYTHON` paths. It never treats LM Studio or oMLX as a training
interpreter.

`POST /api/v1/runtimes/configure` accepts `runtime_id` and an
`interpreter_path`. Aptus executes a bounded capability probe, requires the
selected runtime to be compatible, preserves its absolute command path without
resolving a virtual-environment symlink, and persists it in the private state
directory. Finder-launched Mac builds use this
route because they cannot depend on shell startup environment variables.

## Local inference services

LM Studio and oMLX are inference and evaluation services. They are not Aptus
training runtimes. Aptus accepts only explicit HTTP loopback origins with an
explicit port. It disables proxies and redirects, applies request and response
bounds, and never scans local ports.

`GET /api/v1/inference/services` checks only the documented defaults,
`127.0.0.1:1234` for LM Studio and `127.0.0.1:8000` for oMLX.

`POST /api/v1/inference/models` accepts `service`, optional `endpoint`, and
optional `timeout_seconds`, then returns the service's OpenAI-compatible model
list.

`POST /api/v1/inference/generate` adds `model`, `messages`, `max_tokens`, and
`temperature`. It performs one non-streaming chat-completions request. A local
service error uses a nested structured error with service, operation, code,
message, and upstream status.

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
`explicit_user_facts_required`. It also includes one
`aptus.model-inspection-receipt.v1`. Parameter count and training permission
remain explicit user facts and never enter the receipt. `facts` can include
exact `model_type`, `architecture`,
`quantization_bits`, `quantization_layout`, and a `moe` topology with expert
count, experts per token, expert width, sparse cadence, dense-only layer
indices, and optional shared expert width. The separate `compatibility` object
uses a closed, status-discriminated contract:

- `conditional` requires an unpadded, nonempty family and reason, a known
  `supported_runtime`, one or more known `supported_methods`, a known
  `compute_backend`, a known `distribution`, and a known
  `adapter_profile_id`. Its evidence requirement is exactly `pilot-required`.
  Every family, runtime, backend, method, distribution, and adapter-profile
  tuple must resolve server-side through the host compatibility registry. Its
  path validator derives the complete runtime contract from the typed method
  registry.
- `recognized` carries no executable runtime, backend, method, distribution, or
  adapter-profile claim. It identifies a known dense family and leaves
  execution selection to the planner.
- `unsupported` carries no executable runtime, backend, method, distribution,
  or adapter-profile claim. Its evidence requirement is exactly
  `implementation-required`.

Runtime IDs are `transformers-peft-cuda`, `mlx-lm`, and `pytorch-mps`. Backend
IDs are `cuda`, `rocm`, `mps`, and `cpu`. Method IDs are `full`, `lora`,
`int8-lora`, and `qlora`. Distribution IDs are `single`, `ddp`, and `fsdp`.
The closed adapter-profile vocabulary contains `attention-qkvo.v1`, which binds
the attention `q_proj`, `k_proj`, `v_proj`, and `o_proj` target policy.
Unknown IDs and malformed combinations fail closed at the producer, API, and
browser client. A known tuple that is not registered for the stated model family
fails at the producer and API response boundary. The browser validates the
closed IDs, runtime-backend pairing, adapter-method applicability, and response
shape without copying the method registry into a second policy authority.
Invalid evidence cannot be presented as eligible for the reviewed pilot path.

Provider inspection and candidate planning call the same host-side policy
evaluator. The API model delegates model-family path coherence to the same
registry instead of maintaining its own runtime-binding rules. The v1 response
remains a single flattened path shape. Its producer rejects a future
heterogeneous path set rather than selecting or dropping one path silently.
The flattened `compatibility` projection remains unchanged under
`aptus.api.v1`. The sibling inspection receipt carries the complete
`aptus.model-compatibility.v2` decision, stable reason and evidence IDs, and any
registered policy ID, semantic version, and path ID. Its
`subject_facts_sha256` covers compatibility inputs. Its separate
`observed_facts_sha256` covers every provider-declared or inferred planning fact
actually carried from inspection. Each covered field has a provenance kind,
source, observation time, and the same resolved revision.

Receipt entries use only `provider-declared` or `inferred`. They cover every
non-null compatibility subject fact and include at least one provider-declared
subject observation. Registered policies can require additional fields to be
provider-declared.

Receipt content IDs and digests are tamper-evident, not authenticated
signatures. A caller that passes the receipt to planning must trust its local
client boundary. Aptus still recomputes the receipt, observed facts, policy
decision, provenance requirements, and content identity before using it.

Exact aliases normalize reviewed dense Qwen and Gemma model types. The first
sparse compatibility row requires `qwen3_moe`, `Qwen3MoeForCausalLM`, a
four-bit group-64 default layout, one eight-bit group-64 router-gate override
per layer, a complete reviewed topology, and no shared expert. It reports
conditional eligibility for the reviewed single-device `mlx-lm` on `mps` QLoRA
pilot path with adapter profile `attention-qkvo.v1`. Prefix matching never
admits MoE or multimodal variants. Sparse model-type and architecture markers
remain unsupported when provider topology is absent, even if their normalized
family has a dense policy.

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
| `inspection_receipt` | object or null | No | Receipt returned by a successful model inspection |
| `project_id` | string or null | No | Existing `project_` plus 32 lowercase hex |
| `project_name` | string or null | No | Name for a new project; 1 to 120 characters |

`model` fields:

| Field | Type | Required | Constraint |
| --- | --- | ---: | --- |
| `model_id` | string | Yes | Provider repository identifier |
| `revision` | string | Yes | Domain layer requires immutable 40 to 64 hex |
| `family` | string | Yes | Normalized to its canonical lowercase identity; must resolve in the current target-module catalog for adapter eligibility |
| `parameters_b` | number | Yes | Greater than 0 |
| `hidden_size` | integer | Yes | Greater than 0 |
| `intermediate_size` | integer or null | No | Greater than 0 when present |
| `layers` | integer | Yes | Greater than 0 |
| `context_length` | integer | Yes | Greater than 0 |
| `license_name` | string | Yes | Non-empty in the domain layer |
| `training_allowed` | boolean | Yes | Must be true in the domain layer |
| `model_type` | string or null | No | Exact provider identity; required by the sparse allowlist |
| `architecture` | string or null | No | Exact provider class; omitted dense requests use the domain default |
| `quantization_bits` | integer or null | No | From 1 through 16; the first sparse row requires 4 |
| `quantization_layout` | object or null | No | Canonical MLX groupwise defaults and overrides; required by the first sparse row |
| `moe` | object or null | No | Exact routed-expert topology |

`quantization_layout` contains positive `default_bits` and
`default_group_size`, plus a `module_overrides` array. Each override contains a
dotted `module_path`, `bits`, and `group_size`. Override paths must be sorted and
unique. The exact Qwen3 MoE row requires four-bit group-64 defaults and exactly
one eight-bit group-64 `model.layers.N.mlp.gate` override per layer. The layout
is bound into plan identity. A merely four-bit Qwen3 MoE request is not enough.

`moe` fields:

| Field | Type | Required | Constraint |
| --- | --- | ---: | --- |
| `expert_count` | integer | Yes | Greater than 0 |
| `experts_per_token` | integer | Yes | Greater than 0 and no greater than `expert_count` |
| `expert_intermediate_size` | integer | Yes | Greater than 0 |
| `decoder_sparse_step` | integer | Yes | Greater than 0 |
| `mlp_only_layers` | integer array | No | Empty by default; domain validation requires sorted, unique, in-range indices |
| `shared_expert_intermediate_size` | integer or null | No | Greater than 0 when present; unsupported by the first sparse row |

The request never accepts `active_parameters` or `sparse_layer_count`. The
backend derives both and serializes them in the plan. `parameters_b` remains the
user-attested total resident parameter count.

Without `inspection_receipt`, all submitted model facts use the explicit
`user-attested` policy-decision source. With a receipt, the model ID, revision,
covered planning facts, compatibility decision, provenance requirements, and
receipt identity must all match. A present but malformed, stale, mismatched, or
modified receipt fails. It never falls back to the user-attested path.

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

For CUDA, `supports_4bit` is a device and kernel eligibility fact. MLX-LM
QLoRA does not reuse that CUDA-shaped flag. It remains conditional until the
model-data gate verifies explicit four-bit MLX quantization metadata on the
pinned model revision.

`target` fields:

| Field | Type | Required | Default or constraint |
| --- | --- | ---: | --- |
| `objective` | `quality`, `memory`, or `speed` | Yes | None |
| `sequence_length` | integer | Yes | Greater than 0 |
| `effective_batch_size` | integer | No | `16` |
| `max_epochs` | integer | No | `3` |
| `method_preference` | executable method or null | No | `null` |
| `training_runtime` | runtime ID or null | No | Inferred from the compute backend |
| `task` | string | No | `sft`; other values are rejected by planning |
| `evaluation_fraction` | number | No | `0.1`, in `[0, 1)` |
| `packing` | boolean | No | False; true is rejected by planning |
| `checkpoint_steps` | integer | No | `100`; CUDA checkpoint/evaluation interval, while MLX uses non-resumable weight snapshots |

Success persists and returns one full `aptus.training-plan.v4` object plus
`project_id` and `project_revision_id`. Supplying `project_id` appends to that
project. Otherwise Aptus creates a named project, using `project_name` or a
model-derived default. When no candidate is viable, the response is
`422 no_feasible_plan` and still includes the complete rejected candidate
matrix.

The OpenAPI response requires the v4 schema and plan ID, recommendation,
candidates, warnings, rationale, model-policy decision, decision source, and
nullable inspection receipt. Every candidate requires its candidate ID,
decision ID, and nullable policy binding. The maintained browser client rejects
purported v4 plans that omit this provenance chain. A no-feasible comparison is
an explicitly partial view and cannot be submitted for compilation.

### `GET /api/v1/plans/{plan_id}`

The ID must have the exact form `plan_` plus 20 lowercase hexadecimal
characters. Invalid or missing IDs return `404 plan_not_found`. A valid stored
plan is rehydrated through the strict domain contract before it is returned. A
saved v3 plan, v2 plan, schema-less plan, or v4 plan with a stale policy returns
`409 replan_required` without changing the file.

## Compilation and direct validation

### `POST /api/v1/compile`

```json
{
  "plan_id": "plan_0123456789abcdef0123",
  "output_dir": "/absolute/new-bundle",
  "project_id": "project_...",
  "expected_project_revision_id": "revision_..."
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
  "runtime_contract": {},
  "report": {},
  "project_id": "project_...",
  "project_revision_id": "revision_..."
}
```

The file list is the current directory tree, including mutable files that are
not part of the immutable manifest. Compile requires the exact project and
current revision that own the plan. The persisted plan must equal the
revision's immutable plan snapshot. A mismatch returns
`409 project_plan_snapshot_mismatch`. Success appends a revision that records
the bundle-manifest fingerprint plus the ZIP SHA-256 and byte size, then returns
the new project identities.

Project attachment uses a compare-and-swap. If the project advances during
compilation, Aptus returns `409 project_revision_conflict`. It removes only the
unchanged bundle and ZIP created by that request. A replacement at either path
is preserved.

A persisted v2 plan or plan with no schema identifier returns
`409 replan_required` before bundle creation. The stored plan and project
revision remain unchanged.

### `POST /api/v1/validate`

| Field | Type | Required | Default |
| --- | --- | ---: | --- |
| `bundle_dir` | string | Yes | None |
| `project_id` | project ID | Yes | None |
| `expected_project_revision_id` | revision ID | Yes | None |
| `level` | validation level | No | `static` |
| `run` | boolean | No | False |

`contract` and `static` validation run synchronously while holding the service
validation guard. Runtime levels with `run: false` return the direct static
report plus `RUNTIME_NOT_EXECUTED`. Runtime levels with `run: true` return
`409 runtime_validation_requires_job` and a `suggested_action`. Runtime work
must use jobs so it is serialized and cancellable.
Validation requires the exact project and current revision that own the bundle.
It rejects a changed saved plan snapshot, plan ID, candidate, resolved bundle
path, manifest content, or recorded manifest fingerprint. A path that belongs
to the revision but fails this deeper identity check returns
`409 project_bundle_binding_mismatch`. Success appends an immutable revision
and returns `project_id` and `project_revision_id`.

## Jobs

A service can be started with local execution disabled. In that mode the
planning, compilation, and static-validation surfaces stay available, but every
endpoint that would run work on this host fails closed with
`403 desktop_execution_disabled`: `POST /api/v1/jobs`, and `POST /api/v1/validate`
whenever `run` is true and the requested level is `dependency`, `model-data`,
`measured-preflight`, or `pilot`. `GET /api/v1/bootstrap` reports the current
mode as `local_execution_enabled`. Read that field before offering an execution
action rather than discovering the mode from a rejected request.

### `POST /api/v1/jobs`

| Field | Type | Required | Default |
| --- | --- | ---: | --- |
| `bundle_dir` | string | Yes | None |
| `project_id` | project ID | Yes | None |
| `expected_project_revision_id` | revision ID | Yes | None |
| `action` | `dependency`, `model-data`, `preflight`, `pilot`, or `train` | No | `preflight` |
| `confirm_full_train` | boolean | No | False |

The endpoint rejects forward skips with `409 job_prerequisite_not_met`. It
rejects a competing job with `409 active_job_conflict`. Confirmation is valid
only for `train`, and training requires it to be true. No resume field is
accepted. Aptus for Mac can execute an Apple bundle only when the exact external
MLX-LM or PyTorch MPS interpreter is available. A missing interpreter returns
`409 runtime_unavailable`. CUDA bundles still require a CUDA runtime, whether
local or on a transferred target host.

Train submission validates the immutable bundle and prior state, then deeply
checks the current runtime-specific pilot and capacity while holding the global
lease and record locks. CUDA admission verifies environment, hardware, free
VRAM, host RAM, disk, checkpoint contracts, and export contracts. MLX admission
verifies the owned uninterrupted pilot, current available unified memory above
measured peak plus reserve, and current disk against planned and measured
adapter artifacts. The queued record is written only after admission succeeds.
The request must name the exact project and current revision that own the
bundle. Aptus verifies the saved plan snapshot, plan ID, selected candidate,
resolved path, manifest, and project-bound fingerprint. It persists the job
association before starting the worker. If that project write fails, the worker
never starts. The permit launcher repeats the manifest fingerprint and file
checks from the bundle working directory immediately before execution.
An exact-path bundle whose deeper identity changed returns
`409 project_bundle_binding_mismatch` before launch.
Successful submission appends a revision and returns its project identities
with the job.

MLX `pilot` means an uninterrupted exact-model run with at least two optimizer
updates plus fresh-process adapter reload and one-to-four-token generation. MLX
`train` starts again from the pinned base and runs for the plan-derived duration.
Neither action accepts resume state.

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

Job responses use `aptus.job-record.v1`. A schema-less legacy record migrates on
read with durable authorization cleared. Corrupt, symlinked, or unsupported job
records move to a private quarantine with a reason receipt. Healthy records
continue to load.

## Projects and immutable revisions

### `POST /api/v1/projects`

Accepts `{ "name": "Project name" }`. The name must contain 1 to 120
characters. Success returns `201` with an `aptus.project.v1` project object.

### `GET /api/v1/projects`

Returns healthy project summaries in most-recently-updated order. Each summary
contains its ID, name, timestamps, latest revision identity, revision count, and
latest revision summary when one exists.

### Project detail and history

`GET /api/v1/projects/{project_id}` returns the manifest and latest full
revision. `GET /api/v1/projects/{project_id}/revisions` returns ordered summary
records. `GET /api/v1/projects/{project_id}/revisions/{revision_id}` returns one
`aptus.project-revision.v1` record.

A revision binds its parent, ordinal, reason, available facts and plan snapshot,
selected candidate, bundle, durable validation summary, job IDs, and
`content_sha256`. Revisions are append-only. Validation persistence removes
current authorization and capacity fields. `training_authorization.current` is
always false on disk.

### `POST /api/v1/projects/{project_id}/recover`

Accepts `{ "revision_id": "revision_..." }`. Aptus verifies any referenced
local plan and bundle, then appends a new revision derived from the requested
one. It does not rewrite the source revision. Success returns:

```json
{
  "status": "recovered",
  "project_id": "project_...",
  "revision": {},
  "training_authorization_current": false
}
```

Recovery is not training resume. Revalidate current evidence and submit a new
explicitly confirmed train action. Missing projects or revisions return the
corresponding `project_not_found` or `project_revision_not_found` error.
A revision whose plan snapshot uses v3, v2, or no schema identifier returns
`409 replan_required`. Aptus preserves the source revision and appends no
replacement revision. A stale v4 policy decision has the same result. Create a
new v4 plan from the source facts instead.

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
| `400` | `invalid_request`, `filesystem_error`, `runtime_configuration_invalid`, local-inference configuration errors |
| `403` | `path_forbidden`, `desktop_session_required`, `desktop_execution_disabled` |
| `404` | `path_not_found`, `plan_not_found`, `job_not_found`, `project_not_found`, `project_revision_not_found`, route `not_found` |
| `409` | `path_conflict`, `active_job_conflict`, `job_prerequisite_not_met`, `runtime_validation_requires_job`, `runtime_unavailable`, `replan_required`, `project_revision_conflict`, `project_plan_mismatch`, `project_plan_snapshot_mismatch`, `project_bundle_mismatch`, `project_bundle_binding_mismatch` |
| `422` | `request_validation`, `no_feasible_plan` |
| `502`, `504` | Bounded local-inference service or timeout errors |

An explicit framework HTTP error with non-object detail uses `http_error`.
Invalid Host headers are rejected by middleware before the endpoint contract.
The internal macOS desktop entrypoint preinstalls the same protected-API cookie
through an exact-origin native path before WebKit makes its first request. It
never carries the token in a URL. Ordinary `aptus serve` instead uses the
printed query-to-cookie handoff described above. Both paths accept the bearer
header for protected API requests.

## Static workbench routes

When a workbench build is available, the hidden `GET /{full_path:path}` route
serves matching non-API assets inside the selected static root. Other non-API
paths return `index.html` for
client-side routing. Unknown paths under `/api/` return JSON `404 not_found`.
Static files are public so the browser can load the application shell. The
application cannot read protected state until its API requests carry the valid
cookie.

## Related documentation

- [Configuration defaults](configuration-defaults.md)
- [CLI reference](cli.md)
- [Method registry](method-registry.md)
- [Evidence records](evidence-records.md)
- [Run states](run-states.md)
- [Error and finding codes](error-codes.md)
