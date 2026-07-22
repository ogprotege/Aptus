# Error and Finding Codes

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | API clients, CLI operators, UI developers, and support engineers |
| Authority | Normative inventory of host API errors and host validator findings in v0.2 |
| Last reviewed | 2026-07-22 |
| Next review | 2026-10-22, or sooner when API handlers or validation findings change |

API errors, managed-job errors, and validation findings are separate channels.
An API error describes why a request failed. A failed job can carry a runtime
error string and log. A validation finding describes evidence discovered while
checking a bundle.

## API error envelopes

Most filesystem and value errors use `details`:

```json
{
  "error": "path_forbidden",
  "details": "Permission denied"
}
```

Lifecycle conflicts use structured fields:

```json
{
  "error": "job_prerequisite_not_met",
  "message": "The requested action is not allowed yet.",
  "action": "pilot",
  "required_state": "measured-preflight-pass",
  "current_state": "model-data-pass",
  "reason": "..."
}
```

### Emitted API codes

| HTTP | Code | Meaning |
| ---: | --- | --- |
| `400` | `invalid_request` | A value or operation violated an Aptus contract |
| `400` | `filesystem_error` | An uncategorized operating-system filesystem error occurred |
| `403` | `path_forbidden` | The service process lacks permission for a path |
| `403` | `desktop_session_required` | The private macOS service did not receive its per-launch session cookie |
| `403` | `desktop_execution_disabled` | The macOS sidecar received a runtime validation or job submission; transfer the bundle to a CUDA host |
| `404` | `path_not_found` | A required filesystem path does not exist |
| `404` | `plan_not_found` | The requested content-addressed plan is not persisted |
| `404` | `job_not_found` | The requested job record is not persisted |
| `404` | `not_found` | A static-enabled request under `/api/` matched no route |
| `409` | `path_conflict` | A no-clobber destination already exists |
| `409` | `active_job_conflict` | A guarded operation conflicts with active Aptus work |
| `409` | `job_prerequisite_not_met` | A managed action was submitted before its required state |
| `409` | `runtime_validation_requires_job` | Runtime validation was requested through the synchronous endpoint |
| `422` | `request_validation` | The strict Pydantic request model rejected shape, type, range, or extra fields |
| `422` | `no_feasible_plan` | All 12 candidate rows were rejected |
| varies | `http_error` | FastAPI emitted a non-object HTTP error detail |

`no_feasible_plan` includes the complete candidate matrix. Request-validation
errors include Pydantic `details`. Prerequisite errors include the action,
required state, current state, and reason. Clients should preserve those fields
instead of replacing them with a generic message.

Invalid Host headers are rejected by TrustedHost middleware and are not wrapped
in an Aptus error object.

## Managed-job errors

Job submission errors use the API codes above. After a job has been accepted,
runtime failure is recorded through:

- job `state: failed` or `state: cancelled`;
- `return_code` when a child launched;
- job `error` for launch, ownership, cancellation, or verification failure;
- `validation_report_error` when polling cannot read the current report; and
- `log` plus the current `log_tail`.

Generated runtime programs raise descriptive exceptions rather than a stable
enumerated code for every CUDA, model, checkpoint, split, or export failure.
Automation must use terminal job state and retain the log.

## Validation finding shape

```json
{
  "code": "MANIFEST_MISMATCH",
  "message": "Checksum or size mismatch: train.py",
  "severity": "error",
  "path": "train.py"
}
```

Host validation sets report state to `invalid` when any finding has severity
`error`. A warning alone does not invalidate the bundle.

## Host validator finding codes

### Required inputs and JSON

| Code | Severity | Meaning |
| --- | --- | --- |
| `MISSING_FILE` | error | A required compiler file is absent |
| `PLAN_JSON_ERROR` | error | `plan.json` cannot be read or parsed |
| `TRAINER_CONFIG_JSON_ERROR` | error | `config/trainer.json` cannot be read or parsed |
| `MANIFEST_JSON_ERROR` | error | `bundle-manifest.json` cannot be read or parsed |

### Plan and deterministic parity

| Code | Severity | Meaning |
| --- | --- | --- |
| `PLAN_CONTRACT_ERROR` | error | A plan schema, identity, range, dataset, or semantic rule failed |
| `PLANNER_PARITY_ERROR` | error | The validator could not reconstruct planning from bound facts |
| `PLANNER_PARITY_MISMATCH` | error | Replanning produced different candidates, recommendation, or plan ID |

### Manifest integrity

| Code | Severity | Meaning |
| --- | --- | --- |
| `MANIFEST_SCHEMA` | error | Schema is not `aptus.bundle.v2` |
| `MANIFEST_PLAN_DIGEST` | error | `plan_sha256` does not match `plan.json` |
| `MANIFEST_EMPTY` | error | `files` is absent, invalid, or empty |
| `MANIFEST_ENTRY_INVALID` | error | A file entry lacks a string path |
| `MANIFEST_PATH_INVALID` | error | A file path is duplicate, absolute, or traverses a parent |
| `MANIFEST_FILE_MISSING` | error | A manifested file is absent |
| `MANIFEST_MISMATCH` | error | A manifested size or SHA-256 differs |
| `MANIFEST_INTEGRITY` | error | The self-contained manifest validator found symlinks, unexpected files, or another integrity failure |

### Generated source and configuration

| Code | Severity | Meaning |
| --- | --- | --- |
| `PYTHON_PARSE_ERROR` | error | A generated Python entrypoint fails AST parsing |
| `UNRESOLVED_TEMPLATE` | error | A generated file contains `{{`, `}}`, or `TODO` |
| `DEPENDENCY_SET_MISMATCH` | error | Direct pins do not equal the selected method set |
| `USER_VALUE_EMBEDDED_IN_SOURCE` | error | Generated `train.py` embeds a user model ID or dataset path |
| `TRAINER_CONFIG_MISMATCH` | error | A bound trainer field differs from the recommendation or target |

### Runtime invocation and attestation

| Code | Severity | Meaning |
| --- | --- | --- |
| `RUNTIME_NOT_EXECUTED` | warning | A runtime level was requested with `run=false`; state remains `static-pass` |
| `RUNTIME_VALIDATION_FAILED` | error | Portable `validate.py` exited nonzero |
| `RUNTIME_ATTESTATION_INVALID` | error | Runtime validation did not publish a readable bound report |
| `PREFLIGHT_METRICS_INVALID` | error | Synthetic metrics are missing, malformed, non-positive, or misbound |
| `PREFLIGHT_METRICS_UNBOUND` | error | The report digest or embedded metrics do not match the measured file |

The self-contained runtime validator can stop before the host wrapper converts a
specific generated-script exception into one of these finding codes. In that
case, inspect `RUNTIME_VALIDATION_FAILED`, the runtime evidence tail, managed job
error, and full log together.

## Operator response by class

| Class | Correct response |
| --- | --- |
| Request validation | Correct the submitted shape or range; do not retry unchanged |
| Missing path or permission | Correct service-visible paths or process permissions |
| No feasible plan | Inspect every candidate reason and correct facts or requirements |
| Active-job conflict | Wait, poll the owning job, or cancel it through its owner |
| Prerequisite conflict | Complete or recheck the named prior action |
| Manifest or plan finding | Recompile from the trusted plan and source |
| Runtime dependency or model-data failure | Correct the environment, facts, or source and rerun the ordered gate |
| Capacity failure | Re-probe on the target host or select a different viable plan |
| Pilot, checkpoint, split, or export failure | Preserve artifacts and logs; do not authorize training |

Never suppress an error by editing `validation-report.json`. Report content is
evidence, not an override mechanism.

## Related documentation

- [API reference](api.md)
- [CLI reference](cli.md)
- [Validation states](validation-states.md)
- [Run states](run-states.md)
- [Bundle manifest](bundle-manifest.md)
- [Troubleshooting](../guides/troubleshooting.md)
