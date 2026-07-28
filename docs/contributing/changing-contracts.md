# Changing Contracts

> **Status:** Active | **Audience:** Core contributors | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Architecture | **Last reviewed:** 2026-07-27 | **Review by:** 2026-10-27

Aptus contracts bind facts, decisions, generated files, runtime evidence, and
completion. A field change can alter identity even when its JSON shape looks
compatible. Decide the semantic effect before editing code.

## Contract inventory

| Contract | Current identifier | Primary authority |
|---|---|---|
| Facts | `aptus.facts.v2` | `domain.py` and interface request models |
| Training plan | `aptus.training-plan.v2` | `domain.py` and `plan_contract.py` |
| Memory formula | `aptus-memory-v2` | `planning.py` |
| Method descriptor | `aptus.method-descriptor.v1` | `methods/contracts.py` and registry |
| Bundle manifest | `aptus.bundle.v2` | `generation.py` and `plan_contract.py` |
| Trainer configuration | `aptus.trainer-config.v2` | `generation.py` |
| CUDA trainable census | `aptus.trainable-parameter-census.v1` | `attestation.py` and generated CUDA runtime |
| MLX target binding | `aptus.mlx-trainable-target-binding.v1` | generated MLX runtime and parent verifier |
| CUDA dataset split | `aptus.dataset-split.v1` | generated CUDA trainer and completion verifier |
| MLX dataset split | `aptus.mlx-split.v1` | compiler, generated MLX runtime, and parent verifier |
| CUDA preflight metrics | `aptus.preflight-metrics.v1` | generated CUDA preflight and validation |
| MLX preflight, pilot, and full metrics | `aptus.runtime-metrics.v1` | generated MLX runtime and parent verifier |
| CUDA pilot metrics | `aptus.pilot-metrics.v2` | generated validation and job verification |
| CUDA pilot run | `aptus.pilot-run.v1` | generated runtime and job verification |
| MLX pilot and full-run output | `aptus.mlx-run-output.v1` | generated MLX runtime and parent verifier |
| CUDA full-run output | `aptus.run-output.v1` | generated CUDA trainer and parent verifier |
| CUDA final export | `aptus.final-export.v1` | generated exporter and parent verifier |
| MLX final export | `aptus.mlx-final-export.v1` | generated MLX runtime and parent verifier |
| MLX reload evidence | `aptus.mlx-reload-evidence.v1` | fresh reload process and parent verifier |
| MLX artifact manifest | `aptus.mlx-artifact-manifest.v1` | generated MLX action owner and parent verifier |
| GPU lease | `aptus.gpu-lease.v1` | host and portable lease implementations |
| Validator behavior | `aptus-validator-v2` | `validation.py` |
| HTTP API | `aptus.api.v1` | `api_contracts.py`, `api.py`, and generated OpenAPI |
| Job record | `aptus.job-record.v1` | `execution.py` |
| Project | `aptus.project.v1` | `projects.py` |
| Project revision | `aptus.project-revision.v1` | `projects.py` |
| Diagnostic report | `aptus.diagnostics.v1` | `diagnostics.py` |

The documentation methodology names additional rule-set versions for candidate
enumeration, precision, ranking, and preflight. Keep those labels aligned with
their executable rules.

## Decide whether the change is semantic

A semantic change alters what the same input means, what can execute, how an ID
is derived, how resources are calculated, what evidence proves, or what files
an artifact must contain. Examples include:

- adding or removing a bound fact;
- changing a default that affects generated training;
- changing candidate feasibility or ranking order;
- changing a memory coefficient or upper-bound rule;
- changing loss masking, truncation, splitting, parameter scope, or optimizer
  membership;
- changing checkpoint, export, completion, or lease semantics;
- changing the required bundle file set.

Increment the relevant version or introduce a new contract when old and new
readers cannot safely assign the same meaning. A prose clarification, error
message improvement, or additional non-semantic warning may remain compatible,
but it still needs tests.

## Trace identity consumers

Candidate identity normalizes model, dataset, hardware, target, strategy,
memory, status, and resource facts. Plan identity binds normalized facts, the
ordered candidate IDs, and the recommendation. Bundle identity binds the plan
digest and compiler-managed file manifest.

When a new field changes execution or selection:

1. add it to the typed model;
2. add it to canonical identity normalization;
3. reject missing, malformed, non-finite, or unknown values as appropriate;
4. include it in generated configuration and evidence;
5. verify it at every runtime boundary that consumes it;
6. add mutation tests showing that a changed value changes identity or fails;
7. update schema and methodology documentation.

Do not add an execution-affecting field only to a report or UI model. That
creates an unbound setting.

## Keep host and portable implementations aligned

The bundle copies `plan_contract.py` and `runtime_lease.py`. It emits trainer,
runner, preflight, validator, and MLX reload programs from package resources
under `src/aptus/_bundle_programs/`.
Host-side validation and `JobService` independently verify related evidence.

A contract change may therefore touch:

- the application dataclass or Pydantic request;
- canonical identity and manifest validation;
- compiler output;
- one or more generated programs;
- host validation;
- job admission and parent completion verification;
- CLI output;
- API response and web types;
- current reference documentation.

Add cross-boundary tests that feed bad generated evidence to the host verifier.
Do not assume shared field names prove shared semantics.

## Persisted state and compatibility

Current plan and bundle validators require exact schema identifiers. There is no
general artifact migration command. Never reinterpret an old artifact under new
semantics without an explicit reader and migration policy.

For an incompatible change, choose one of these:

- keep a versioned reader and prove its old semantics;
- reject the older version with a clear finding;
- provide an explicit no-clobber migration that records source and destination
  identities.

Do not mutate a persisted plan, bundle, validation report, job, or run in place
to make it appear current.

## API and workbench changes

FastAPI request models reject unknown fields. Additions and removals must be
reflected in `api.py`, `api_contracts.py`, API tests, maintained request and
normalization code in `web/src/api.ts`, UI domain types in `web/src/types.ts`
when needed, Swift decoders when applicable, stage state, restoration logic, and
the API reference. Regenerate and check both derived contract artifacts:

```bash
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/generate_openapi.py
npm --prefix web run openapi:generate
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/generate_openapi.py --check
npm --prefix web run openapi:check
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/check_client_contracts.py
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/verify_versions.py
```

`docs/reference/openapi.v1.json` and `web/src/generated/openapi.ts` are
generated. The TypeScript artifact supplies schema and path types. It does not
replace React's maintained request construction, runtime normalization, domain
types, or presentation logic. Swift decoders are maintained and contract
checked. Describe each boundary precisely instead of calling either client
wholly generated or wholly hand-maintained.

If a field can be unavailable, preserve `null` or an explicit unknown state.
Do not substitute total memory for free memory or provider declarations for user
permission.

## State and transition changes

Validation states and run states are public lifecycle contracts. When changing
one:

- define legal predecessors and successors;
- define whether the state is persisted or a derived phase;
- define admission, cancellation, and crash-recovery behavior;
- update API and UI polling semantics;
- add interruption and idempotency tests;
- update validation-state and run-state references.

Keep `verifying` a phase unless a deliberate versioned change makes it a
persisted state. Do not mark success from child exit alone.

## Contract-change checklist

- [ ] Current and proposed semantics are written down.
- [ ] Compatibility and version decision is explicit.
- [ ] Canonical identity includes every execution-affecting field.
- [ ] Unknown, malformed, stale, and tampered forms fail closed.
- [ ] Host and portable implementations agree.
- [ ] API, CLI, and web consumers agree.
- [ ] Generated OpenAPI JSON and TypeScript schema and path types are current.
- [ ] Maintained React adapters and covered Swift decoders pass their contract
      checks.
- [ ] Positive, negative, mutation, and interruption tests pass.
- [ ] A fresh bundle was compiled and reviewed.
- [ ] Reference and methodology pages were updated.
- [ ] Target-host evidence was refreshed when runtime meaning changed.
- [ ] `python tools/verify_versions.py` confirms package, web, desktop, and
      OpenAPI version parity.

## Related documentation

- [Plan schema](../reference/plan-schema.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Data and identity flow](../architecture/data-and-identity-flow.md)
- [Generated code](generated-code.md)
- [Validation states](../reference/validation-states.md)
