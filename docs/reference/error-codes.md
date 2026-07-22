# Error and Finding Codes

API errors and validation findings serve different purposes. API errors describe
request or lifecycle failure. Validation findings describe bundle evidence.

## Common API errors

| Error | Meaning |
|---|---|
| `request_validation` | JSON did not match the strict request model |
| `invalid_request` | Values or state violated an operation contract |
| `no_feasible_plan` | No enumerated candidate can proceed |
| `plan_not_found` | Requested plan ID is not persisted |
| `job_not_found` | Requested job ID is not persisted |
| `job_prerequisite_not_met` | A managed action was submitted before its required prior validation state passed |
| `active_job_conflict` | Managed job or guarded validation conflicts with the global lease |
| `runtime_validation_requires_job` | Runtime work was requested through the direct validation endpoint |
| `http_error` | Other explicit HTTP failure |

## Validation finding families

Finding codes are emitted with severity, message, and optional path. Important
families include:

- missing or unexpected required files;
- invalid JSON, plan, candidate, or manifest identity;
- file digest or size mismatch;
- source dataset digest mismatch;
- unsafe path or generated-source structure;
- direct requirement-set mismatch;
- dependency or environment mismatch;
- pinned model, tokenizer, parameter-count, target-module, method capability, or
  trainable-scope mismatch;
- canonical-row transformation failure;
- invalid or conflicting `split_group` declaration;
- canonical dataset mutation, split-assignment disagreement, or inconsistent
  split strategy, digest, count, target, realized-fraction, or row-error evidence;
- CUDA, precision, quantization, distribution, or world-size mismatch;
- zero, non-finite, boolean-typed, unplanned, incomplete-LoRA-pair, optimizer-set,
  malformed, or cross-phase-inconsistent trainable-parameter census;
- pilot checkpoint continuation or artifact-manifest failure;
- stale report bindings or insufficient current capacity;
- full-run metrics, rank evidence, or structural export failure.

Consumers must not decide success from the absence of one named code. Use the
report state and inspect every finding.

Some runtime failures are attached to a failed job as an error message rather
than a portable static finding code. The job state and complete log remain part
of the operator decision.

## Operator response

Correct source facts or generator code, then create a new plan and bundle when a
compiler-managed file is wrong. Do not suppress findings by editing the report.
Runtime capacity failures require a new successful check on the target host.
